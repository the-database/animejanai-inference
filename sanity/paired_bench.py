"""Compare two CUDA backends on the same GPU. Uses only the Python standard library.

Video defaults to 1080p NV12 -> 4K P010, built-in HD Balanced, four rotating
surfaces, 32 warmup frames and 40 balanced-order pairs of eight frames.
Temporal and RIFE cases take a configuration and default to same-size NV12 output.
Use --case temporal --scale 2 for a 2x temporal upscaler. Image cases time synchronous public RGB8 upscales.
Image timing loads one model per process. Successive process pairs alternate which build runs first.
Warm VRAM is measured per build after caller surfaces exist.
"""
import argparse
from contextlib import ExitStack
import ctypes as C
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from temporal_workspace import Backend, Device, I, P, S, bind, check, pixels

INPUT_FORMATS = {'nv12': 1, 'p010': 2, 'yuv444p16': 3, 'yuv444p10msb': 5, 'yuv444p12msb': 6}


class ImageParams(C.Structure):
    _fields_ = [('version', C.c_uint32), ('backend', I), ('device', C.c_uint64),
                ('cache', C.c_char_p), ('builder', C.c_char_p), ('env', C.c_char_p), ('log', P), ('user', P)]


class Tile(C.Structure):
    _fields_ = [('onnx', C.c_char_p), ('w', I), ('h', I), ('overlap', I), ('flags', I)]


class RGB(C.Structure):
    _fields_ = [('width', I), ('height', I), ('channels', I), ('stride', C.c_ssize_t), ('data', P)]


def image_arm(args, path, stack):
    lib = C.CDLL(str(path))
    by = C.byref
    create = bind(lib, 'aji_img_create', P, C.POINTER(ImageParams), P, S)
    msg = C.create_string_buffer(1024)
    messages = []
    callback = C.CFUNCTYPE(None, P, I, C.c_char_p)(lambda _, level, line: messages.append(line.decode(errors='replace')))
    # Destroy the model and context before releasing their log callback.
    stack.callback(lambda keep=callback: None)
    params = ImageParams(version=1, backend=1, log=C.cast(callback, P),
                         builder=os.fsencode(args.trtexec) if args.trtexec else None)
    ctx = P(create(by(params), msg, len(msg)))
    if not ctx:
        raise RuntimeError(msg.value)
    stack.callback(bind(lib, 'aji_img_destroy', None, C.POINTER(P)), by(ctx))
    error = bind(lib, 'aji_img_last_error', C.c_char_p, P)
    params = Tile(os.fsencode(args.models / args.image_model), args.tile or args.width,
                  args.tile or args.height, args.overlap, 2)
    model = P(bind(lib, 'aji_img_load', P, P, C.POINTER(Tile))(ctx, by(params)))
    if not model:
        raise RuntimeError(error(ctx))
    stack.callback(bind(lib, 'aji_img_release', None, C.POINTER(P)), by(model))
    info = (I * 6)()
    if bind(lib, 'aji_img_model_info_get', I, P, C.POINTER(I))(model, info):
        raise RuntimeError(error(ctx))
    w, h, scale = args.width, args.height, info[0]
    src = C.create_string_buffer(random.Random(793).randbytes(w*h*3))
    dst = C.create_string_buffer(w*h*3*scale*scale)
    inp = RGB(w, h, 3, w*3, C.cast(src, P))
    out = RGB(w*scale, h*scale, 3, w*scale*3, C.cast(dst, P))
    run = bind(lib, 'aji_img_upscale', I, P, C.POINTER(RGB), C.POINTER(RGB), P, P)

    def issue(_):
        if run(model, by(inp), by(out), None, None):
            raise RuntimeError(error(ctx))
    return issue, lambda: '\n'.join(messages)


def surfaces(args, dev, stack):
    w, h = args.width, args.height
    scale = args.scale if args.case in ('video', 'temporal') else 1
    count = 4 if args.case == 'video' else 8
    if args.case == 'temporal' and scale > 1:
        count = 9  # Give TFDAT nine distinct input frames per window.
    fmt = INPUT_FORMATS[args.input_format]
    ins = [dev.frame(w, h, fmt, stack) for _ in range(count)]
    outs = [dev.frame(w*scale, h*scale, 2 if args.case == 'video' else 1, stack) for _ in ins]
    clip = args.raw_input or args.nv12
    stream = stack.enter_context(clip.open('rb')) if clip else None
    for i, frame in enumerate(ins):
        data = stream.read(frame.nbytes) if stream else pixels(w, h, i, False, fmt)
        if len(data) != frame.nbytes:
            raise ValueError(f'{clip} needs {count} full {args.input_format} frames of {frame.nbytes} bytes')
        dev.upload(frame, data)
    return ins, outs


def video_arm(args, path, dev, frames, stack):
    api = Backend(path, args.conf, args.models, stack, args.slot, args.trtexec)
    ow, oh = I(), I()
    rc = api.configure(api.ctx, args.width, args.height, 24000/1001, C.byref(ow), C.byref(oh))
    ins, outs = frames
    want_rife = args.case.startswith('rife')
    if args.case == 'rife-ids' and api.rife_ids is None:
        raise ValueError(f'{path} does not export aji_infer_rife_ids')
    if rc != (0 if want_rife else 1) or (ow.value, oh.value) != (outs[0].width, outs[0].height):
        raise RuntimeError(f'configure returned {rc}, {ow.value}x{oh.value}: {api.error(api.ctx)!r}')
    radius = api.radius(api.ctx)
    if (radius > 0) != (args.case == 'temporal'):
        raise ValueError(f'Temporal radius {radius} is incompatible with {args.case}')

    def issue(i):
        a = C.byref(ins[i % len(ins)])
        out = C.byref(outs[i % len(outs)])
        if want_rife:
            b = C.byref(ins[(i+1) % len(ins)])
            if args.case == 'rife-ids':
                api.ok(api.rife_ids(api.ctx, a, i, b, i+1, 0.5, out, dev.stream, -1))
            else:
                api.ok(api.rife(api.ctx, a, b, 0.5, out, dev.stream))
        elif args.case == 'temporal':
            api.ok(api.ingest(api.ctx, a, i, dev.stream))
            api.ok(api.seq(api.ctx, max(0, i-radius), out, dev.stream))
        else:
            api.ok(api.infer(api.ctx, a, out, dev.stream))
    return issue, lambda: api.log(api.ctx).decode(errors='replace')


def summarize(pairs):
    first, second = zip(*pairs)
    pct = sorted(100*(b/a-1) for a, b in pairs)
    # Exact two-sided sign interval for the median, at least 95% coverage.
    n = len(pct)
    tail = 0
    for k in range(n):
        tail += math.comb(n, k)
        if tail * 40 > 2**n:
            break
    return dict(reference_ms=median(first), candidate_ms=median(second),
                paired_change_pct=median(pct), interval95_pct=[pct[k-1], pct[n-k]])


def worker(arm, image_batches=0):
    cmd = [sys.executable, __file__, *sys.argv[1:], '--memory-worker', arm]
    if image_batches:
        cmd.extend(['--image-batches', str(image_batches)])
    output = subprocess.check_output(cmd, text=True)
    return json.loads(output.splitlines()[-1])


def isolated_images(args):
    names = ('reference', 'candidate')
    first = names[::-1] if args.swap_load_order else names
    groups = args.image_process_pairs
    batches, extra = divmod(args.pairs, groups)
    wall, process_pairs = [], []
    warm = {name: [] for name in names}
    memory = {}
    for i in range(groups):
        count = batches + (i < extra)
        arms = {}
        for name in (first if i % 2 == 0 else first[::-1]):
            result = worker(name, count)
            arms[name] = result['batches_ms']
            warm[name].append(result['warm_bytes'])
            memory.setdefault(name, {'dispatch': result['dispatch']})
        a, b = (arms[name] for name in names)
        wall.extend(zip(a, b))
        process_pairs.append([median(a), median(b)])
    for name in names:
        memory[name]['warm_bytes'] = int(median(warm[name]))
    # Batches in one process share placement and warm state. Each process contributes one median.
    return {'case': 'image', 'input_size': [args.width, args.height], 'model': args.image_model,
            'tile': args.tile or [args.width, args.height], 'overlap': args.overlap,
            'pairs': args.pairs, 'frames_per_batch': args.batch, 'load_order': first,
            'timing': 'isolated_processes', 'warmup_pages_per_process': 8, 'memory': memory,
            'wall': summarize(process_pairs), 'process_pairs_ms': process_pairs, 'wall_pairs_ms': wall}


def run(args):
    with ExitStack() as stack:
        if os.name == 'nt':
            paths = [getattr(args, args.memory_worker)] if args.memory_worker else [args.reference, args.candidate]
            for folder in {p.parent for p in [args.cuda_runtime, *paths]}:
                stack.enter_context(os.add_dll_directory(str(folder)))
        dev = Device(args.cuda_runtime, stack)
        is_image = args.case == 'image'
        if not is_image and args.conf is None:
            args.conf = Path(stack.enter_context(TemporaryDirectory())) / 'builtins.conf'
            args.conf.write_text('[global]\nconfig_version=2\nnative=on\nlogging=yes\n', encoding='utf-8')
        frames = None if is_image else surfaces(args, dev, stack)
        free, total = S(), S()
        mem = bind(dev.lib, 'cudaMemGetInfo', I, C.POINTER(S), C.POINTER(S))
        if is_image:
            check(mem(C.byref(free), C.byref(total)))
            initial = free.value
            issue, log = image_arm(args, getattr(args, args.memory_worker), stack)
            for i in range(8):
                issue(i)
            check(mem(C.byref(free), C.byref(total)))
            result = {'warm_bytes': initial - free.value, 'dispatch': log(), 'batches_ms': []}
            for _ in range(args.image_batches):
                start = time.perf_counter_ns()
                for i in range(args.batch):
                    issue(i)
                result['batches_ms'].append((time.perf_counter_ns() - start) / (args.batch * 1e6))
            return result
        record = bind(dev.lib, 'cudaEventRecord', I, P, P)
        done = bind(dev.lib, 'cudaEventSynchronize', I, P)
        elapsed = bind(dev.lib, 'cudaEventElapsedTime', I, C.POINTER(C.c_float), P, P)
        events = [P(), P()]
        for event in events:
            check(bind(dev.lib, 'cudaEventCreate', I, C.POINTER(P))(C.byref(event)))
            stack.callback(bind(dev.lib, 'cudaEventDestroy', I, P), event)
        check(mem(C.byref(free), C.byref(total)))
        initial = free.value
        arm_names = ('reference', 'candidate')
        load_order = arm_names[::-1] if args.swap_load_order else arm_names
        arms = {}
        for name in ((args.memory_worker,) if args.memory_worker else load_order):
            path = getattr(args, name)
            issue, log = video_arm(args, path, dev, frames, stack)
            arms[name] = issue
            for i in range(32):
                issue(i)
            check(dev.sync(dev.stream))
            if args.memory_worker:
                check(mem(C.byref(free), C.byref(total)))
                return {'warm_bytes': initial-free.value, 'dispatch': log()}
        order = [(0, 1), (1, 0)] * (args.pairs//2)
        random.Random(147).shuffle(order)
        wall, gpu = [], []
        for j, pair in enumerate(order):
            wall_ms, gpu_ms = [0, 0], [0, 0]
            for arm in pair:
                issue = arms[arm_names[arm]]
                check(record(events[0], dev.stream))
                start = time.perf_counter_ns()
                for n in range(args.batch):
                    issue(32+j*args.batch+n)
                check(record(events[1], dev.stream))
                check(done(events[1]))
                wall_ms[arm] = (time.perf_counter_ns()-start)/(args.batch*1e6)
                ms = C.c_float()
                check(elapsed(C.byref(ms), *events))
                gpu_ms[arm] = ms.value/args.batch
            wall.append(wall_ms)
            gpu.append(gpu_ms)
        result = {'case': args.case, 'input_size': [args.width, args.height], 'pairs': args.pairs, 'frames_per_batch': args.batch,
                  'load_order': load_order, 'wall': summarize(wall), 'wall_pairs_ms': wall}
        out = frames[1][0]
        result.update(slot=args.slot, input_format=args.input_format, output_size=[out.width, out.height],
                      output_format='p010' if args.case == 'video' else 'nv12', gpu=summarize(gpu), gpu_pairs_ms=gpu)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'models', 'cuda-runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', choices=('video', 'temporal', 'rife', 'rife-ids', 'image'), default='video')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--scale', type=int, help='output scale for video (default 2) or temporal (default 1)')
    p.add_argument('--slot', type=int, default=1002)
    p.add_argument('--conf', type=Path, help='optional configuration, required for temporal and RIFE-only chains')
    p.add_argument('--trtexec', type=Path, help='TensorRT builder if the selected configuration needs an engine')
    p.add_argument('--input-format', choices=INPUT_FORMATS, default='nv12', help='video input surface format (default nv12)')
    clips = p.add_mutually_exclusive_group()
    clips.add_argument('--raw-input', type=Path, help='tightly packed frames in --input-format, little-endian for 16-bit words')
    clips.add_argument('--nv12', type=Path, help='tightly packed NV12 input clip')
    p.add_argument('--image-model', help='image ONNX filename under --models')
    p.add_argument('--tile', type=int, help='maximum square image tile, defaults to the page dimensions')
    p.add_argument('--overlap', type=int, default=32)
    p.add_argument('--pairs', type=int, default=40)
    p.add_argument('--swap-load-order', action='store_true',
                   help='start with the candidate (first process pair for images). Columns stay reference, candidate')
    p.add_argument('--image-process-pairs', type=int, default=6,
                   help='independent reference/candidate process pairs for image timing (default 6). '
                        '--pairs timed batches are distributed across these processes')
    p.add_argument('--image-batches', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--batch', type=int, help='frames per arm per pair, default 8 for video or 1 for image')
    p.add_argument('--output', type=Path, help='also save raw pairs and dispatch logs as JSON')
    p.add_argument('--memory-worker', choices=('reference','candidate'), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.scale is None:
        args.scale = 2 if args.case == 'video' else 1
    if args.nv12 and args.input_format != 'nv12':
        p.error('use --raw-input with non-NV12 formats')
    if args.case == 'image' and (args.raw_input or args.input_format != 'nv12'):
        p.error('--input-format and --raw-input apply to video frames')
    for name in ('reference','candidate','cuda_runtime','models'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.batch is None:
        args.batch = 1 if args.case == 'image' else 8
    if args.pairs < 6 or args.pairs % 2 or min(args.width,args.height,args.batch,args.scale) <= 0:
        p.error('use positive dimensions and batch size, and an even pair count of at least six')
    if args.case == 'image' and (args.image_process_pairs < 6 or args.image_process_pairs % 2 or args.image_process_pairs > args.pairs):
        p.error('--image-process-pairs must be even and between 6 and --pairs')
    if args.case != 'image' and (args.width % 2 or args.height % 2):
        p.error('video frames need even width and height')
    if args.case == 'image' and not args.image_model:
        p.error('--case image needs --image-model')
    if args.case in ('temporal','rife','rife-ids') and not args.conf:
        p.error('supply --conf and --slot for a temporal or RIFE-only chain')
    if args.memory_worker:
        print(json.dumps(run(args)))
        return
    if args.case == 'image':
        result = isolated_images(args)
    else:
        memory = {arm: worker(arm) for arm in ('reference', 'candidate')}
        result = run(args)
        result['memory'] = memory
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_pairs_ms')}, indent=2))


if __name__ == '__main__':
    main()
