"""Compare uncaptured, engine-graph and chain-graph RIFE outputs.

Requires numpy, a built backend/runtime, a cached static 1080p RIFE 4.22 engine,
and three files of 24 decoded 1920x1080 NV12 frames each. No timing claims are
made. Run without another player/GPU test; the output directory must be new.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend', type=Path, required=True)
p.add_argument('--cuda-runtime', type=Path, required=True)
p.add_argument('--rife-dir', type=Path, required=True)
p.add_argument('--frames', type=Path, nargs=3, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--worker', choices=['off', 'engine', 'chain'])
a = p.parse_args()

if not a.worker:
    a.output.mkdir(parents=True, exist_ok=False)
    for mode in ['off', 'engine', 'chain']:
        env = os.environ.copy()
        env.pop('AJI_RIFE_GRAPH', None)
        if mode != 'off':
            env['AJI_RIFE_GRAPH'] = mode
        subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', mode],
                       env=env, check=True)
    records = {mode: json.loads((a.output / (mode + '.json')).read_text())
               for mode in ['off', 'engine', 'chain']}
    for mode in ['engine', 'chain']:
        if records[mode]['frames'] != records['off']['frames']:
            raise RuntimeError(f'{mode}: output or scene decision differs')
    print(f'PASS: {len(records["off"]["frames"])} calls match in all three modes')
    sys.exit(0)

cookies = []
if os.name == 'nt':
    cookies = [os.add_dll_directory(str(d.resolve()))
               for d in {a.backend.parent, a.cuda_runtime.parent}]
cuda = C.CDLL(str(a.cuda_runtime.resolve()))
backend = C.CDLL(str(a.backend.resolve()))

class Params(C.Structure):
    _fields_ = [('api', C.c_uint32), ('context', C.c_void_p),
                ('conf', C.c_char_p), ('models', C.c_char_p),
                ('trtexec', C.c_char_p), ('env', C.c_char_p),
                ('slot', C.c_int), ('rife', C.c_char_p), ('async_build', C.c_int),
                ('d3d', C.c_void_p), ('engine', C.c_char_p),
                ('maxw', C.c_int), ('maxh', C.c_int),
                ('log', C.c_void_p), ('opaque', C.c_void_p)]

class Frame(C.Structure):
    _fields_ = [('width', C.c_int), ('height', C.c_int), ('format', C.c_int),
                ('matrix', C.c_int), ('range', C.c_int), ('siting', C.c_int),
                ('planes', C.c_void_p * 3), ('strides', C.c_ssize_t * 3)]

def bind(lib, name, args, result=C.c_int):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f

malloc = bind(cuda, 'cudaMalloc', [C.POINTER(C.c_void_p), C.c_size_t])
free = bind(cuda, 'cudaFree', [C.c_void_p])
copy = bind(cuda, 'cudaMemcpy', [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int])
stream_create = bind(cuda, 'cudaStreamCreateWithFlags', [C.POINTER(C.c_void_p), C.c_uint])
sync = bind(cuda, 'cudaStreamSynchronize', [C.c_void_p])
stream_destroy = bind(cuda, 'cudaStreamDestroy', [C.c_void_p])
create = bind(backend, 'aji_create', [C.POINTER(Params)], C.c_void_p)
configure = bind(backend, 'aji_configure', [C.c_void_p, C.c_int, C.c_int,
                  C.c_double, C.POINTER(C.c_int), C.POINTER(C.c_int)])
factor = bind(backend, 'aji_rife_factor', [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)])
infer = bind(backend, 'aji_infer_rife', [C.c_void_p, C.POINTER(Frame), C.POINTER(Frame),
                                       C.c_double, C.POINTER(Frame), C.c_void_p])
last_error = bind(backend, 'aji_last_error', [C.c_void_p], C.c_char_p)
destroy = bind(backend, 'aji_destroy', [C.POINTER(C.c_void_p)], None)

def ck(code):
    if code:
        raise RuntimeError(f'CUDA error {code}')

logs = []
@C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_char_p)
def log_callback(opaque, severity, message):
    logs.append(message.decode('utf-8', errors='replace'))

conf = a.output / (a.worker + '.conf')
conf.write_text('[global]\nbackend=TensorRT\n[slot_1]\nchain_1_rife=yes\n'
                'chain_1_rife_model=422\nchain_1_rife_factor_numerator=2\n'
                'chain_1_rife_factor_denominator=1\nchain_1_rife_before_upscale=yes\n'
                'chain_1_rife_scene_detect_threshold=0.150\n')
params = Params(api=9, conf=str(conf.resolve()).encode(),
                models=str(a.output.resolve()).encode(), slot=1,
                rife=str(a.rife_dir.resolve()).encode(), async_build=2,
                log=C.cast(log_callback, C.c_void_p))
ctx = C.c_void_p(create(C.byref(params)))
if not ctx:
    raise RuntimeError('aji_create failed')

def setup():
    w, h, num, den = C.c_int(), C.c_int(), C.c_int(), C.c_int()
    rc = configure(ctx, 1920, 1080, 24000 / 1001, C.byref(w), C.byref(h))
    if rc != 0 or (w.value, h.value) != (1920, 1080):
        raise RuntimeError(last_error(ctx))
    if factor(ctx, C.byref(num), C.byref(den)) != 1 or (num.value, den.value) != (2, 1):
        raise RuntimeError('RIFE 2x was not enabled')

setup()
stream = C.c_void_p()
ck(stream_create(C.byref(stream), 1))
size = 1920 * 1080 * 3 // 2
buffers = []
for _ in range(3):
    ptr = C.c_void_p()
    ck(malloc(C.byref(ptr), size * 2))
    buffers.append(ptr)
rows = []

def run(label, left, right, t=.5, fmt=1, color_range=1):
    bpp = 1 if fmt == 1 else 2
    images = [left, right] if fmt == 1 else [
        (left.astype(np.uint16) << 8), (right.astype(np.uint16) << 8)]
    frames = [Frame(1920, 1080, fmt, 2, color_range, 1,
                    (C.c_void_p * 3)(ptr.value, ptr.value + 1920 * 1080 * bpp, None),
                    (C.c_ssize_t * 3)(1920 * bpp, 1920 * bpp, 0))
              for ptr in buffers]
    for i, image in enumerate(images):
        ck(copy(buffers[i], image.ctypes.data, size * bpp, 1))
    rc = infer(ctx, C.byref(frames[0]), C.byref(frames[1]), t,
               C.byref(frames[2]), stream)
    if rc not in (0, 1):
        raise RuntimeError(last_error(ctx))
    ck(sync(stream))
    row = dict(label=label, time=t, format=fmt, range=color_range, status=rc)
    if rc == 0:
        output = np.empty(size * bpp, dtype=np.uint8)
        ck(copy(output.ctypes.data, buffers[2], output.nbytes, 2))
        row['sha256'] = hashlib.sha256(output).hexdigest()
    rows.append(row)

for file in a.frames:
    clip = np.fromfile(file, dtype=np.uint8).reshape(24, size)
    for i in range(23):
        run(f'{file.stem}-{i}', clip[i], clip[i + 1])
for t in [.25, .75, .5]:
    run(f'timestep-{t}', clip[8], clip[9], t)
black = np.full(size, 128, dtype=np.uint8)
black[:1920 * 1080] = 16
white = black.copy()
white[:1920 * 1080] = 235
run('black-white-cut', black, white)
run('before-reset', clip[8], clip[9])
setup()
run('after-reset', clip[8], clip[9])
if rows[-1].get('sha256') != rows[-2].get('sha256'):
    raise RuntimeError('Reset changed the output')

# Reuse caller addresses while changing format/range and then frame contents.
for index, (fmt, color_range) in enumerate([(1, 2), (2, 1), (2, 2), (1, 1), (1, 2)]):
    for replay in range(2):
        run(f'transition-{index}-{replay}', clip[8 + replay], clip[9 + replay],
            fmt=fmt, color_range=color_range)

captures = sum('RIFE CUDA graph captured:' in message for message in logs)
expected = {'off': 0, 'engine': 2, 'chain': 7}[a.worker]
if captures != expected:
    raise RuntimeError(f'{a.worker}: expected {expected} captures, got {captures}: {logs}')
(a.output / (a.worker + '.json')).write_text(json.dumps(
    dict(captures=captures, frames=rows), indent=2))
(a.output / (a.worker + '.log')).write_text('\n'.join(logs))
ck(stream_destroy(stream))
destroy(C.byref(ctx))
for ptr in buffers:
    ck(free(ptr))
print(f'{a.worker}: {len(rows)} calls, {captures} captures')
