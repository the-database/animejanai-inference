"""Compare caller-plane CUDA graph replay with eager inference through aji_infer.
Supply a spatial chain config and its models, or a TensorRT engine of the input size.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend', type=Path, required=True)
p.add_argument('--reference-backend', type=Path, help='use this backend for eager reference outputs')
p.add_argument('--cuda-runtime', type=Path, required=True)
model = p.add_mutually_exclusive_group(required=True)
model.add_argument('--engine', type=Path)
model.add_argument('--conf', type=Path)
p.add_argument('--models', type=Path)
p.add_argument('--trtexec', type=Path, help='builder for TensorRT steps in a configuration')
p.add_argument('--slot', type=int, default=1)
p.add_argument('--width', type=int, default=1920)
p.add_argument('--height', type=int, default=1080)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--fault-backend', type=Path, help='test DLL built with direct_graph_faults.h')
faults = ('begin', 'body', 'end', 'nodes', 'instantiate', 'update', 'warmup', 'launch')
p.add_argument('--worker', choices=('eager', 'updated', *faults), help=argparse.SUPPRESS)
a = p.parse_args()
if a.conf and not a.models:
    p.error('--conf requires --models')
if not a.worker:
    a.output.mkdir(parents=True, exist_ok=False)
    for mode in ('eager', 'updated', *(faults if a.fault_backend else ())):
        env = os.environ.copy()
        for key in ('AJI_NO_GRAPH', 'AJI_TEST_DIRECT_GRAPH_FAULT'):
            env.pop(key, None)
        if mode == 'eager':
            env['AJI_NO_GRAPH'] = '1'
        if mode in faults:
            env['AJI_TEST_DIRECT_GRAPH_FAULT'] = mode
        subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', mode], env=env, check=True)
    sys.exit(0)

P, I, S = C.c_void_p, C.c_int, C.c_size_t
log_fn = C.CFUNCTYPE(None, P, I, C.c_char_p)

class Params(C.Structure):
    _fields_ = [('version', C.c_uint32), ('context', P), ('conf', C.c_char_p), ('models', C.c_char_p),
                ('trtexec', C.c_char_p), ('environment', C.c_char_p), ('slot', I), ('rife', C.c_char_p),
                ('async_build', I), ('d3d', P), ('engine', C.c_char_p), ('width', I), ('height', I),
                ('log', log_fn), ('opaque', P)]

class Frame(C.Structure):
    _fields_ = [(name, I) for name in ('width', 'height', 'format', 'matrix', 'range', 'siting')] + [
        ('planes', P * 3), ('strides', C.c_ssize_t * 3)]

def bind(lib, name, result, *args):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f

def ck(rc):
    if rc:
        raise RuntimeError(f'CUDA error {rc}')

dll = a.fault_backend if a.worker in faults else a.backend
if a.worker == 'eager' and a.reference_backend:
    dll = a.reference_backend
if os.name == 'nt':
    dirs = [os.add_dll_directory(str(d.resolve())) for d in {dll.parent, a.cuda_runtime.parent}]
cuda, lib = C.CDLL(str(a.cuda_runtime.resolve())), C.CDLL(str(dll.resolve()))
alloc = bind(cuda, 'cudaMalloc', I, C.POINTER(P), S)
free = bind(cuda, 'cudaFree', I, P)
copy = bind(cuda, 'cudaMemcpy2D', I, P, S, P, S, S, S, I)
fill = bind(cuda, 'cudaMemsetAsync', I, P, I, S, P)
stream = P()
ck(bind(cuda, 'cudaStreamCreateWithFlags', I, C.POINTER(P), I)(C.byref(stream), 1))
sync = bind(cuda, 'cudaStreamSynchronize', I, P)
create = bind(lib, 'aji_create', P, C.POINTER(Params))
configure = bind(lib, 'aji_configure', I, P, I, I, C.c_double, C.POINTER(I), C.POINTER(I))
set_slot = bind(lib, 'aji_set_slot', I, P, I)
infer = bind(lib, 'aji_infer', I, P, C.POINTER(Frame), C.POINTER(Frame), P)
last_error = bind(lib, 'aji_last_error', C.c_char_p, P)
messages = []
callback = log_fn(lambda _, level, line: messages.append(line.decode(errors='replace')))
params = Params(version=9, width=a.width, height=a.height, slot=a.slot, log=callback,
                engine=os.fsencode(a.engine) if a.engine else None, conf=os.fsencode(a.conf) if a.conf else None,
                models=os.fsencode(a.models) if a.models else None,
                trtexec=os.fsencode(a.trtexec) if a.trtexec else None)
ctx = P(create(C.byref(params)))
ow, oh = I(), I()

def configure_chain():
    if not ctx or configure(ctx, a.width, a.height, 24, C.byref(ow), C.byref(oh)) != 1:
        raise RuntimeError(last_error(ctx) if ctx else 'aji_create failed')

configure_chain()
if a.conf and a.worker == 'updated':
    print(bind(lib, 'aji_current_log', C.c_char_p, P)(ctx).decode(errors='replace'), flush=True)

def geometry(w, h, fmt):
    row = w * (1 if fmt == 1 else 2)
    return [(row, h), (row, h // 2)] if fmt in (1, 2) else [(row, h)] * 3

def surface(w, h, fmt, index, input_frame):
    frame = Frame(w, h, fmt, 2, 1, 1)
    for j, (row, rows) in enumerate(geometry(w, h, fmt)):
        stride = row + 64 * (1 + (index + j) % 3)
        ptr = P()
        ck(alloc(C.byref(ptr), (row + 256) * rows))
        frame.planes[j], frame.strides[j] = ptr.value, stride
        if input_frame:
            # Pad with a different pattern so stale pitch is observable even on a flat chroma plane.
            ck(fill(ptr, 0x3d, (row + 256) * rows, None))
            data = random.Random(1234 + 79 * index + j).randbytes(row * rows)
            if fmt == 2:  # P010 stores the ten significant bits at the top of each 16-bit word.
                data = bytearray(data)
                data[::2] = data[::2].translate(bytes(v & 0xc0 for v in range(256)))
                data = bytes(data)
            ck(copy(ptr, stride, C.c_char_p(data), row, row, rows, 1))
    return frame

def digest(frame):
    h = hashlib.sha256()
    for j, (row, rows) in enumerate(geometry(frame.width, frame.height, frame.format)):
        buf = C.create_string_buffer(row * rows)
        ck(copy(buf, row, frame.planes[j], frame.strides[j], row, rows, 2))
        h.update(buf)
    return h.hexdigest()

reference_file = a.output / 'eager.json'
reference = {} if a.worker == 'eager' else json.loads(reference_file.read_text())
observed = {}
def submit(fi, fo, chosen_stream):
    rc = infer(ctx, C.byref(fi), C.byref(fo), chosen_stream)
    if rc:
        if a.worker in ('warmup', 'launch') and lib.aji_test_graph_failures() == 1 and last_error(ctx):
            print(f'{a.worker}: genuine execution error propagated')
            sys.exit(0)
        raise RuntimeError(last_error(ctx))

def captures():
    return sum('captured updatable frame graph' in m for m in messages)

# Three distinct pitched surfaces on each side. Queue three launches before inspecting any output.
for infmt, outfmt in ((1, 2), (1, 1), (1, 3), (2, 2), (3, 3)):
    ins = [surface(a.width, a.height, infmt, k, True) for k in range(3)]
    outs = [surface(ow.value, oh.value, outfmt, k, False) for k in range(3)]
    # Vary output color tags (including unset tags) across the pool. The chain uses input colorimetry.
    for fo, tags in zip(outs, ((2, 1, 1), (1, 2, 2), (0, 0, 0))):
        fo.matrix, fo.range, fo.siting = tags
    ck(sync(None))
    for batch, (shift, color_range) in enumerate(((0, 1), (1, 1), (2, 1), (0, 2), (1, 1))):
        before = captures()
        if a.conf and (infmt, outfmt) == (1, 2) and batch in (1, 2):
            # Slot selection takes effect at configure. These frames still use the active chain.
            if set_slot(ctx, 0 if batch == 1 else a.slot) != 0:
                raise RuntimeError(last_error(ctx))
        for k, fi in enumerate(ins):
            fi.range = color_range
            fi.siting = 3 if batch == 3 else 1  # top-left on the color change, left on restore
            fo = outs[(k + shift) % 3]
            for j, (row, rows) in enumerate(geometry(fo.width, fo.height, fo.format)):
                ck(fill(fo.planes[j], 0xab, (row + 256) * rows, stream))
            submit(fi, fo, stream)
        ck(sync(stream))
        for k, fi in enumerate(ins):
            dest = (k + shift) % 3
            key = f'{infmt}->{outfmt} batch={batch} in={k} out={dest} range={color_range}'
            actual = digest(outs[dest])
            if a.worker != 'eager' and actual != reference[key]:
                raise RuntimeError(f'{key}: expected {reference[key][:16]}, got {actual[:16]}')
            observed[key] = actual
        if a.worker == 'updated':
            expected_captures = int(batch in (0, 3, 4))
            capture_count = captures() - before
            if capture_count != expected_captures:
                raise RuntimeError(f'{infmt}->{outfmt} batch={batch}: expected {expected_captures} capture(s), '
                                   f'observed {capture_count}')
        if (infmt, outfmt) == (1, 2) and batch < 3 and a.worker in faults:
            staged = sum('captured as staged CUDA graph' in m for m in messages)
            if staged != 1 or captures() != int(a.worker == 'update') or lib.aji_test_graph_last_error():
                raise RuntimeError(f'{a.worker}: expected a clean staged fallback through surface rotation')
    # A default-stream call may rebuild conversion plans. Returning to the old layout must recapture.
    ins[0].siting = 2
    before = captures()
    submit(ins[0], outs[0], None)
    ck(sync(None))
    key = f'{infmt}->{outfmt} default stream'
    observed[key] = digest(outs[0])
    if a.worker != 'eager' and observed[key] != reference[key]:
        raise RuntimeError(f'{key}: output mismatch')
    if a.worker == 'updated' and captures() != before:
        raise RuntimeError('default-stream call captured')
    ins[0].siting = 1
    for phase in ('restore layout', 'reconfigure'):
        if phase == 'reconfigure':
            configure_chain()
        before = captures()
        submit(ins[0], outs[0], stream)
        ck(sync(stream))
        key = f'{infmt}->{outfmt} {phase}'
        observed[key] = digest(outs[0])
        if a.worker != 'eager' and observed[key] != reference[key]:
            raise RuntimeError(f'{key}: output mismatch')
        if a.worker == 'updated' and captures() != before + 1:
            raise RuntimeError(f'{phase}: expected a fresh capture')
    for frame in ins + outs:
        for ptr in frame.planes:
            if ptr:
                ck(free(ptr))
if a.worker in faults:
    if lib.aji_test_graph_failures() != 1 or lib.aji_test_graph_last_error() != 0:
        raise RuntimeError(f'{a.worker}: fault was not consumed cleanly')
    if a.worker in ('warmup', 'launch') or not any('captured as staged CUDA graph' in m for m in messages):
        raise RuntimeError(f'{a.worker}: expected fallback or execution error')
if a.worker == 'eager':
    reference_file.write_text(json.dumps(observed, indent=2))
else:
    print(f'{a.worker}: {len(observed)} outputs matched eager inference; {captures()} direct captures', flush=True)
    if a.worker == 'updated':
        if any('using staged fallback' in m for m in messages):
            raise RuntimeError('unexpected staged fallback on the rotating caller surfaces')
        for message in sorted({m for m in messages if 'captured updatable frame graph' in m}):
            if 'captured updatable frame graph (2 conversion nodes)' not in message:
                raise RuntimeError(f'expected one input and one output conversion: {message}')
            print(message, flush=True)
bind(lib, 'aji_destroy', None, C.POINTER(P))(C.byref(ctx))
ck(bind(cuda, 'cudaStreamDestroy', I, P)(stream))
