"""Compare staged/direct CUDA graphs across NV12/P010 and chroma-siting changes.

Requires numpy, a built TensorRT backend/runtime, and a static 1080p 2x engine.
Run with no concurrent GPU/player benchmark. Output directories must be new.
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
p.add_argument('--engine', type=Path, required=True)
p.add_argument('--cuda-runtime', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--worker', choices=['staged', 'direct'])
a = p.parse_args()

if not a.worker:
    a.output.mkdir(parents=True, exist_ok=False)
    for mode in ['staged', 'direct']:
        env = os.environ.copy()
        env.pop('AJI_NO_GRAPH', None)
        env.pop('AJI_DIRECT_GRAPH', None)
        if mode == 'direct':
            env['AJI_DIRECT_GRAPH'] = '1'
        subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', mode],
                       env=env, check=True)
    results = {mode: json.loads((a.output / (mode + '.json')).read_text())
               for mode in ['staged', 'direct']}
    if results['staged'] != results['direct']:
        raise RuntimeError('Output differs across capture/replay or a format transition')
    print(f'PASS: {len(results["direct"])} exact capture/replay outputs')
    sys.exit(0)

# Keep handles alive for Windows dependency resolution; Linux uses its library path.
directories = []
if os.name == 'nt':
    directories = [os.add_dll_directory(str(path.resolve()))
                   for path in {a.backend.parent, a.cuda_runtime.parent}]
cuda = C.CDLL(str(a.cuda_runtime.resolve()))
backend = C.CDLL(str(a.backend.resolve()))

class Params(C.Structure):
    _fields_ = [('version', C.c_uint32), ('context', C.c_void_p),
                ('conf', C.c_char_p), ('models', C.c_char_p),
                ('trtexec', C.c_char_p), ('environment', C.c_char_p),
                ('slot', C.c_int), ('rife', C.c_char_p), ('async_build', C.c_int),
                ('d3d', C.c_void_p), ('engine', C.c_char_p),
                ('width', C.c_int), ('height', C.c_int),
                ('log', C.c_void_p), ('opaque', C.c_void_p)]

class Frame(C.Structure):
    _fields_ = [('width', C.c_int), ('height', C.c_int), ('format', C.c_int),
                ('matrix', C.c_int), ('range', C.c_int), ('siting', C.c_int),
                ('planes', C.c_void_p * 3), ('strides', C.c_ssize_t * 3)]

cuda.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
cuda.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
cuda.cudaStreamCreate.argtypes = [C.POINTER(C.c_void_p)]
cuda.cudaStreamSynchronize.argtypes = [C.c_void_p]
cuda.cudaStreamDestroy.argtypes = [C.c_void_p]
cuda.cudaFree.argtypes = [C.c_void_p]
backend.aji_create.argtypes = [C.POINTER(Params)]
backend.aji_create.restype = C.c_void_p
backend.aji_configure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_double,
                                C.POINTER(C.c_int), C.POINTER(C.c_int)]
backend.aji_infer.argtypes = [C.c_void_p, C.POINTER(Frame), C.POINTER(Frame), C.c_void_p]
backend.aji_last_error.argtypes = [C.c_void_p]
backend.aji_last_error.restype = C.c_char_p
backend.aji_destroy.argtypes = [C.POINTER(C.c_void_p)]

def ck(code):
    if code:
        raise RuntimeError(f'CUDA error {code}')

messages = []
@C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_char_p)
def logger(opaque, level, message):
    messages.append(message.decode('utf-8', errors='replace'))

ck(cuda.cudaSetDevice(0))
stream = C.c_void_p()
ck(cuda.cudaStreamCreate(C.byref(stream)))
params = Params(version=9, engine=os.fsencode(a.engine.resolve()), width=1920,
                height=1080, log=C.cast(logger, C.c_void_p))
ctx = C.c_void_p(backend.aji_create(C.byref(params)))
if not ctx:
    raise RuntimeError('Could not create backend context')
ow, oh = C.c_int(), C.c_int()
if backend.aji_configure(ctx, 1920, 1080, 24, C.byref(ow), C.byref(oh)) != 1:
    raise RuntimeError(backend.aji_last_error(ctx))
if (ow.value, oh.value) != (3840, 2160):
    raise RuntimeError('Use a static 1080p 2x engine')
inp, out = C.c_void_p(), C.c_void_p()
ck(cuda.cudaMalloc(C.byref(inp), 1920 * 1080 * 3))
plane = 3840 * 2160 * 2
ck(cuda.cudaMalloc(C.byref(out), plane * 3))
output = Frame(3840, 2160, 3, 2, 1, 1,
               (C.c_void_p * 3)(out.value, out.value + plane, out.value + 2 * plane),
               (C.c_ssize_t * 3)(7680, 7680, 7680))
host_out = np.empty(plane * 3, dtype=np.uint8)
records = []
for case, (fmt, siting) in enumerate([(1, 1), (2, 3), (1, 1), (1, 2)]):
    bpp = 1 if fmt == 1 else 2
    frame = Frame(1920, 1080, fmt, 2, 1, siting,
                  (C.c_void_p * 3)(inp.value, inp.value + 1920 * 1080 * bpp, None),
                  (C.c_ssize_t * 3)(1920 * bpp, 1920 * bpp, 0))
    for replay in range(2):
        rng = np.random.default_rng(100 + case * 2 + replay)
        data = rng.integers(16, 236, 1920 * 1080 * 3 // 2, dtype=np.uint16)
        data = (data << 8).astype(np.uint16) if bpp == 2 else data.astype(np.uint8)
        ck(cuda.cudaMemcpy(inp, C.c_void_p(data.ctypes.data), data.nbytes, 1))
        if backend.aji_infer(ctx, C.byref(frame), C.byref(output), stream):
            raise RuntimeError(backend.aji_last_error(ctx))
        ck(cuda.cudaStreamSynchronize(stream))
        ck(cuda.cudaMemcpy(C.c_void_p(host_out.ctypes.data), out, host_out.nbytes, 2))
        records.append(dict(input_format=fmt, siting=siting, replay=replay,
                            sha256=hashlib.sha256(host_out).hexdigest()))
if a.worker == 'direct' and sum('direct CUDA graph cached' in m for m in messages) != 4:
    raise RuntimeError('Expected four direct-graph captures, each followed by replay')
backend.aji_destroy(C.byref(ctx))
ck(cuda.cudaFree(inp))
ck(cuda.cudaFree(out))
ck(cuda.cudaStreamDestroy(stream))
(a.output / (a.worker + '.json')).write_text(json.dumps(records, indent=2))
print(f'{a.worker}: {len(records)} frames completed')
