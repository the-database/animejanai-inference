"""Compare native temporal workspace changes with a reference backend on the public CUDA video ABI.
Run once per --case. Both shared libraries and the model must match the installed runtime pack.
"""
import argparse
from contextlib import ExitStack
import ctypes as C
import os
from pathlib import Path
from tempfile import TemporaryDirectory

P, I, S = C.c_void_p, C.c_int, C.c_size_t


class Params(C.Structure):
    _fields_ = [('version', C.c_uint32), ('cu_ctx', P), ('conf', C.c_char_p), ('models', C.c_char_p),
                ('trtexec', C.c_char_p), ('env', C.c_char_p), ('slot', I), ('rife', C.c_char_p),
                ('async_build', I), ('d3d', P), ('engine', C.c_char_p), ('max_w', I), ('max_h', I),
                ('log', P), ('opaque', P)]


class Frame(C.Structure):
    _fields_ = [(s, I) for s in ('width', 'height', 'fmt', 'matrix', 'range', 'siting')] + [
        ('planes', P * 3), ('strides', C.c_ssize_t * 3)]


def bind(lib, name, result, *args):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f


def check(rc):
    if rc:
        raise RuntimeError(f'CUDA error {rc}')


class Device:
    def __init__(self, runtime, stack):
        self.lib = C.CDLL(str(runtime.resolve()))
        self.malloc = bind(self.lib, 'cudaMalloc', I, C.POINTER(P), S)
        self.free = bind(self.lib, 'cudaFree', I, P)
        self.copy = bind(self.lib, 'cudaMemcpyAsync', I, P, P, S, I, P)
        self.sync = bind(self.lib, 'cudaStreamSynchronize', I, P)
        self.stream = P()
        check(bind(self.lib, 'cudaStreamCreateWithFlags', I, C.POINTER(P), I)(C.byref(self.stream), 1))
        stack.callback(bind(self.lib, 'cudaStreamDestroy', I, P), self.stream)

    def frame(self, w, h, fmt, stack):
        pitch = w if fmt == 1 else 2 * w
        ptr = P()
        check(self.malloc(C.byref(ptr), pitch * h * 3 // 2))
        stack.callback(self.free, ptr)
        return Frame(w, h, fmt, 2, 1, 1, (P * 3)(ptr.value, ptr.value + pitch * h, None),
                     (C.c_ssize_t * 3)(pitch, pitch, 0))

    def upload(self, frame, data):
        check(self.copy(frame.planes[0], C.c_char_p(data), len(data), 1, self.stream))
        check(self.sync(self.stream))

    def read(self, frame):
        data = C.create_string_buffer(frame.strides[0] * frame.height * 3 // 2)
        check(self.copy(data, frame.planes[0], len(data), 2, self.stream))
        check(self.sync(self.stream))
        return data.raw


class Backend:
    def __init__(self, path, conf, model_dir, stack):
        lib = C.CDLL(str(path.resolve()))
        self.error = bind(lib, 'aji_last_error', C.c_char_p, P)
        self.log = bind(lib, 'aji_current_log', C.c_char_p, P)
        self.configure = bind(lib, 'aji_configure', I, P, I, I, C.c_double, C.POINTER(I), C.POINTER(I))
        self.ingest = bind(lib, 'aji_ingest', I, P, C.POINTER(Frame), C.c_uint64, P)
        self.seq = bind(lib, 'aji_infer_seq', I, P, C.c_uint64, C.POINTER(Frame), P)
        self.reset = bind(lib, 'aji_temporal_reset', None, P)
        params = Params(version=9, conf=os.fsencode(conf), models=os.fsencode(model_dir), slot=1,
                        rife=os.fsencode(model_dir))
        self.ctx = P(bind(lib, 'aji_create', P, C.POINTER(Params))(C.byref(params)))
        if not self.ctx:
            raise RuntimeError(f'aji_create failed for {path.name}')
        stack.callback(bind(lib, 'aji_destroy', None, C.POINTER(P)), C.byref(self.ctx))

    def shape(self, w, h, family):
        ow, oh = I(), I()
        rc = self.configure(self.ctx, w, h, 24, C.byref(ow), C.byref(oh))
        log = self.log(self.ctx).decode(errors='replace')
        if rc != 1 or (ow.value, oh.value) != (w, h) or family not in log:
            raise RuntimeError(f'configure {w}x{h}: {self.error(self.ctx)!r}\n{log}')

    def ok(self, rc):
        if rc:
            raise RuntimeError(f'aji status {rc}: {self.error(self.ctx)!r}')


def pixels(w, h, number, bright):
    # A moving texture in two non-overlapping luma bands gives the built-in detector a definite cut.
    def tiled(rows, height):
        chunk = b''.join((row * ((w + len(row) - 1) // len(row)))[:w] for row in rows)
        return (chunk * ((height + len(rows) - 1) // len(rows)))[:w * height]

    base = 154 if bright else 22
    luma = [bytes(base + ((x + 3*y + number*7) % 54) + 14*((x//9 ^ y//5) & 1)
                  for x in range(128)) for y in range(32)]
    chroma = [bytes(64 + ((3*x + y + number*11) % 126) for x in range(128)) for y in range(16)]
    return tiled(luma, h) + tiled(chroma, h//2)


def same(label, expected, actual):
    if actual != expected:
        offset = next(i for i, (a, b) in enumerate(zip(expected, actual)) if a != b)
        raise RuntimeError(f'{label}, byte {offset}: main={expected[offset]} candidate={actual[offset]}')


def temporal(device, backends):
    count = 0
    # The second shape crosses the batched-stage cutoff. Reconfigure the same contexts.
    for w, h in ((1920, 1080), (2048, 1152)):
        for api in backends:
            api.shape(w, h, 'TemporalFix, 7 frames')
        with ExitStack() as surfaces:
            inp = device.frame(w, h, 1, surfaces)
            outs = [device.frame(w, h, 2, surfaces) for _ in backends]

            def compare(seq, phase):
                for api, out in zip(backends, outs):
                    api.ok(api.seq(api.ctx, seq, C.byref(out), device.stream))
                same(f'{w}x{h} {phase} center={seq}', device.read(outs[0]), device.read(outs[1]))

            for epoch, length in enumerate((16, 9) if w == 1920 else (9,)):
                if epoch:
                    for api in backends:
                        api.reset(api.ctx)
                for i in range(length):
                    # Seek reuses sequence numbers for new pixels. Neither cache may return the old generation.
                    device.upload(inp, pixels(w, h, i + 31*epoch, (i >= 7) != bool(epoch)))
                    for api in backends:
                        api.ok(api.ingest(api.ctx, C.byref(inp), i, device.stream))
                    center = i if i < 3 else i - 3
                    compare(center, f'epoch={epoch} ingested={i}')
                    count += 1
                    if i == 10:
                        compare(center, 'repeat held window')
                        count += 1
                for i in range(length - 3, length):
                    compare(i, f'epoch={epoch} drain')
                    count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('temporal', 'temporal-no-cache'), required=True)
    p.add_argument('--reference', type=Path, required=True, help='backend built from main')
    p.add_argument('--backend', type=Path, required=True, help='backend under test')
    p.add_argument('--cuda-runtime', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='TemporalFix S1 ONNX')
    a = p.parse_args()
    os.environ['AJI_NATIVE_FRAME_CACHE'] = 'off' if a.case == 'temporal-no-cache' else 'on'
    model = a.model.resolve(strict=True)
    chain = f'chain_1_temporal_fix_model={model.stem}\n'
    with ExitStack() as stack:
        if os.name == 'nt':
            for path in {a.reference, a.backend, a.cuda_runtime}:
                stack.enter_context(os.add_dll_directory(str(path.resolve().parent)))
        conf = Path(stack.enter_context(TemporaryDirectory())) / 'test.conf'
        conf.write_text('[global]\nconfig_version=2\nbackend=TensorRT\nnative=on\nlogging=yes\n'
                        '[slot_1]\nchain_1_max_fps=120\n' + chain)
        device = Device(a.cuda_runtime, stack)
        backends = [Backend(path, conf, model.parent, stack) for path in (a.reference, a.backend)]
        count = temporal(device, backends)
        print(f'{a.case}: {count} full outputs matched main byte for byte')


if __name__ == '__main__':
    main()
