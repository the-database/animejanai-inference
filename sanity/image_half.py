"""Compare complete image-ABI output and latency using one shared engine cache.

Requires numpy, compatible TensorRT backend DLLs/runtime, the original model,
and one decoded RGB24 frame. Run alone on the GPU; warmup/builds are not timed.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--trtexec', type=Path, required=True)
p.add_argument('--rgb-frame', type=Path, required=True)
p.add_argument('--width', type=int, default=1920)
p.add_argument('--height', type=int, default=1080)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
cookies = []
if os.name == 'nt':
    cookies = [os.add_dll_directory(str(d.resolve()))
               for d in {a.reference.parent, a.candidate.parent, a.runtime}]

class Params(C.Structure):
    _fields_ = [('api', C.c_uint32), ('backend', C.c_int), ('device', C.c_uint64),
                ('cache', C.c_char_p), ('trtexec', C.c_char_p), ('env', C.c_char_p),
                ('log', C.c_void_p), ('opaque', C.c_void_p)]

class ModelParams(C.Structure):
    _fields_ = [('onnx', C.c_char_p), ('tile_w', C.c_int), ('tile_h', C.c_int),
                ('overlap', C.c_int), ('flags', C.c_int)]

class Info(C.Structure):
    _fields_ = [(name, C.c_int) for name in
                ('scale', 'in_channels', 'out_channels', 'tile_w', 'tile_h', 'overlap')]

class Buffer(C.Structure):
    _fields_ = [('width', C.c_int), ('height', C.c_int), ('channels', C.c_int),
                ('stride', C.c_ssize_t), ('data', C.c_void_p)]

def bind(lib, name, args, result=C.c_int):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, result
    return fn

contexts = []
for label, path in [('reference', a.reference), ('candidate', a.candidate)]:
    lib = C.CDLL(str(path.resolve()))
    create = bind(lib, 'aji_img_create', [C.POINTER(Params), C.c_void_p, C.c_size_t], C.c_void_p)
    load = bind(lib, 'aji_img_load', [C.c_void_p, C.POINTER(ModelParams)], C.c_void_p)
    last_error = bind(lib, 'aji_img_last_error', [C.c_void_p], C.c_char_p)
    get_info = bind(lib, 'aji_img_model_info_get', [C.c_void_p, C.POINTER(Info)])
    upscale = bind(lib, 'aji_img_upscale', [C.c_void_p, C.POINTER(Buffer),
                   C.POINTER(Buffer), C.c_void_p, C.c_void_p])
    release = bind(lib, 'aji_img_release', [C.POINTER(C.c_void_p)], None)
    destroy = bind(lib, 'aji_img_destroy', [C.POINTER(C.c_void_p)], None)
    params = Params(api=1, backend=1, cache=str(a.cache.resolve()).encode(),
                    trtexec=str(a.trtexec.resolve()).encode())
    error = C.create_string_buffer(2048)
    ctx = C.c_void_p(create(C.byref(params), error, len(error)))
    if not ctx:
        raise RuntimeError(error.value)
    model_params = ModelParams(str(a.model.resolve()).encode(), a.width, a.height, 32, 0)
    model = C.c_void_p(load(ctx, C.byref(model_params)))
    if not model:
        raise RuntimeError(last_error(ctx))
    info = Info()
    if get_info(model, C.byref(info)) or (info.scale, info.in_channels, info.out_channels) != (2, 3, 3):
        raise RuntimeError('Expected a 2x RGB image model')
    contexts.append(dict(label=label, lib=lib, ctx=ctx, model=model, info=info,
                         upscale=upscale, error=last_error, release=release, destroy=destroy))
assert contexts[0]['lib']._handle != contexts[1]['lib']._handle

rgb = np.fromfile(a.rgb_frame, dtype=np.uint8).reshape(a.height, a.width, 3)
cases = [('full-frame', rgb),
         ('gray-reflect-pad', np.ascontiguousarray(rgb[:320, :500, :1])),
         ('overlapping-tiles', np.pad(rgb, ((0, 72), (0, 64), (0, 0)), mode='reflect'))]
records = []
for name, image in cases:
    height, width, channels = image.shape
    output = np.empty((height * 2, width * 2, channels), dtype=np.uint8)
    source = Buffer(width, height, channels, image.strides[0], image.ctypes.data)
    target = Buffer(width * 2, height * 2, channels, output.strides[0], output.ctypes.data)
    reference = None
    timings = {c['label']: [] for c in contexts}
    for round_index in range(-1, 4):
        order = contexts if round_index % 2 == 0 else list(reversed(contexts))
        for c in order:
            before = time.perf_counter_ns()
            rc = c['upscale'](c['model'], C.byref(source), C.byref(target), None, None)
            elapsed = (time.perf_counter_ns() - before) / 1e6
            if rc:
                raise RuntimeError(c['error'](c['ctx']))
            digest = hashlib.sha256(output).hexdigest()
            if reference is None:
                reference = digest
            if digest != reference:
                raise RuntimeError(f'{name}: {c["label"]} changed output pixels')
            if round_index >= 0:
                timings[c['label']].append(elapsed)
    medians = {k: statistics.median(v) for k, v in timings.items()}
    records.append(dict(case=name, width=width, height=height, channels=channels,
                        sha256=reference, exact=True, milliseconds=timings,
                        medians=medians,
                        timeReductionPercent=100 * (1 - medians['candidate'] / medians['reference'])))
    print(json.dumps(records[-1]), flush=True)
for c in contexts:
    c['release'](C.byref(c['model']))
    c['destroy'](C.byref(c['ctx']))
(a.output / 'results.json').write_text(json.dumps(records, indent=2))
print('PASS: complete image-ABI output is byte-identical for all cases and repeats')
