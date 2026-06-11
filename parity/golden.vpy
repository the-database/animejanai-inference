# Golden generator for aji kernel parity, run under the shipped full-package
# VSPipe so the reference is the literal VapourSynth/zimg users run.
#
# Replicates the exact conversions animejanai_core.py performs:
#   pre:       YUV420 -> RGBH   resize.Spline36(format=RGBH, matrix_in_s=MAT)
#   post:      RGBH -> YUV420   resize.Spline36(format=fmt_in, matrix_s=MAT)
#   resize:    RGBH -> RGBH     resize.Spline36(width=, height=)
#   roundtrip: pre + post composed (no model)
#
# Range and chroma location travel as frame props (as they do from mpv's
# vapoursynth filter); matrix is passed explicitly like the reference code.
#
# vspipe --arg parameters (all strings):
#   mode=pre|post|resize|roundtrip|e2e|rife
#   input=<raw file>  w= h=           input dims
#   fmt=yuv8|yuv10|rgbh                input layout (planar yuv / fp16 rgb)
#   matrix=709|170m   vrange=limited|full   siting=left|center|topleft
#   (the range key is named vrange so it can't shadow the builtin)
#   outfmt=yuv8|yuv10                  post/roundtrip output depth
#   dw= dh=                            resize target
#   engine= vstrt= dlldir=             e2e: TRT engine, vstrt.dll, vsmlrt-cuda
#                                      (e2e = pre + core.trt.Model + post,
#                                      the real upscale2x_trt path)
#   rifemodel= thr= pkgroot=           rife: model int code, SCDetect
#                                      threshold, package root (for vsmlrt
#                                      import). Replicates animejanai_core's
#                                      rife block + rife_cuda + vsmlrt
#                                      video_player 2x; vsmlrt builds/caches
#                                      its own engine next to the onnx.
#
# Raw layouts: yuv8/yuv10 are planar Y,U,V (u8 / u16 LSB-aligned 10-bit);
# rgbh is planar R,G,B fp16. Frames are read until EOF.

import ctypes
import os
import vapoursynth as vs

core = vs.core

mode = globals().get("mode", "pre")
path = globals()["input"]
w = int(globals()["w"])
h = int(globals()["h"])
fmt = globals().get("fmt", "yuv8")
matrix = globals().get("matrix", "709")
vrange = globals().get("vrange", "limited")
siting = globals().get("siting", "left")
outfmt = globals().get("outfmt", "yuv8")
dw = int(globals().get("dw", 0))
dh = int(globals().get("dh", 0))

CL = {"left": 0, "center": 1, "topleft": 2}[siting]
CR = {"limited": 1, "full": 0}[vrange]


def load_raw(path, w, h, fmt):
    if fmt == "rgbh":
        vsfmt, np_bytes, planes = vs.RGBH, 2, [(w, h), (w, h), (w, h)]
        cast = "e"
    elif fmt == "yuv10":
        vsfmt, np_bytes, planes = vs.YUV420P10, 2, [(w, h), (w // 2, h // 2), (w // 2, h // 2)]
        cast = "H"
    else:
        vsfmt, np_bytes, planes = vs.YUV420P8, 1, [(w, h), (w // 2, h // 2), (w // 2, h // 2)]
        cast = "B"

    frame_sz = sum(pw * ph for pw, ph in planes) * np_bytes
    n = os.path.getsize(path) // frame_sz
    assert n > 0, f"no full frames in {path}"
    with open(path, "rb") as fh:
        data = fh.read()

    blank = core.std.BlankClip(width=w, height=h, format=vsfmt, length=n,
                               fpsnum=24000, fpsden=1001, keep=1)

    # ctypes row copies: stdlib-only (the embedded python has no numpy) and
    # safe against VS plane stride padding.
    def fill(n, f):
        fout = f.copy()
        off = n * frame_sz
        for p, (pw, ph) in enumerate(planes):
            row_b = pw * np_bytes
            stride = fout.get_stride(p)
            ptr = fout.get_write_ptr(p).value
            for y in range(ph):
                ctypes.memmove(ptr + y * stride,
                               data[off + y * row_b: off + (y + 1) * row_b],
                               row_b)
            off += ph * row_b
        return fout

    return core.std.ModifyFrame(blank, blank, fill)


clip = load_raw(path, w, h, fmt)
clip = core.std.SetFrameProps(clip, _ColorRange=CR, _ChromaLocation=CL)

if mode == "pre":
    out = core.resize.Spline36(clip, format=vs.RGBH, matrix_in_s=matrix)
elif mode == "post":
    tgt = vs.YUV420P10 if outfmt == "yuv10" else vs.YUV420P8
    out = core.resize.Spline36(clip, format=tgt, matrix_s=matrix,
                               range=1 if CR == 0 else None)
elif mode == "resize":
    out = core.resize.Spline36(clip, width=dw, height=dh)
elif mode == "roundtrip":
    rgb = core.resize.Spline36(clip, format=vs.RGBH, matrix_in_s=matrix)
    tgt = vs.YUV420P10 if outfmt == "yuv10" else vs.YUV420P8
    out = core.resize.Spline36(rgb, format=tgt, matrix_s=matrix,
                               range=1 if CR == 0 else None)
elif mode == "rife":
    # animejanai_core: SCDetect happens inside rife() (sc_mode=1), padding
    # outside; order here matches the reference call sequence.
    import fractions
    import sys
    pkgroot = globals().get("pkgroot", "")
    if pkgroot and pkgroot not in sys.path:
        sys.path.insert(0, pkgroot)
    import vsmlrt

    thr = float(globals().get("thr", "0.150"))
    model = int(globals().get("rifemodel", "414"))

    target_w, target_h = clip.width, clip.height
    pad_w = ((target_w + 63) // 64 * 64) - target_w
    pad_h = ((target_h + 63) // 64 * 64) - target_h
    work = clip
    if pad_w or pad_h:
        work = core.std.AddBorders(work, left=pad_w // 2,
                                   right=pad_w - (pad_w // 2),
                                   top=pad_h // 2,
                                   bottom=pad_h - (pad_h // 2))
    work = core.misc.SCDetect(clip=work, threshold=thr)
    rgb = core.resize.Bilinear(clip=work, format=vs.RGBH, matrix_in_s="709")
    fin = vsmlrt.RIFE(clip=rgb, multi=fractions.Fraction(2, 1), scale=1,
                      model=model, ensemble=False, _implementation=1,
                      video_player=True,
                      backend=vsmlrt.BackendV2.TRT(
                          num_streams=1, force_fp16=True, output_format=1,
                          workspace=None, use_cuda_graph=True,
                          use_cublas=False, use_cudnn=False,
                          static_shape=False, min_shapes=[320, 256],
                          opt_shapes=[1920, 1088], max_shapes=[4096, 2176],
                          device_id=0, short_path=True))
    tgt = vs.YUV420P10 if fmt == "yuv10" else vs.YUV420P8
    out = core.resize.Bilinear(clip=fin, format=tgt, matrix_s="709",
                               range=1 if CR == 0 else None)
    if pad_w or pad_h:
        out = core.std.Crop(out, left=pad_w // 2,
                            right=pad_w - (pad_w // 2),
                            top=pad_h // 2, bottom=pad_h - (pad_h // 2))
elif mode == "e2e":
    # the portable package auto-loads vs-plugins; load explicitly only if not
    if not hasattr(core, "trt"):
        os.add_dll_directory(globals()["dlldir"])
        core.std.LoadPlugin(path=globals()["vstrt"])
    rgb = core.resize.Spline36(clip, format=vs.RGBH, matrix_in_s=matrix)
    up = core.trt.Model(rgb, engine_path=globals()["engine"], num_streams=1)
    tgt = vs.YUV420P10 if outfmt == "yuv10" else vs.YUV420P8
    out = core.resize.Spline36(up, format=tgt, matrix_s=matrix,
                               range=1 if CR == 0 else None)
else:
    raise ValueError(f"unknown mode {mode}")

out.set_output()
