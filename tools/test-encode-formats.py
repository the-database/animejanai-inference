"""GPU regression for aji_encode's frame formats: every source layout through every encoder.

Usage: python tools/test-encode-formats.py <runtime-dir> <ffmpeg-bin-dir> <work-dir> [<model.onnx>]
<runtime-dir> holds aji_encode.exe, aji.dll, aji_trt.dll and the DLLs they load (TensorRT,
cudart, the ffmpeg libraries); <ffmpeg-bin-dir> holds ffmpeg.exe and ffprobe.exe. Needs an
NVIDIA GPU. Sources: 8- and 10-bit 4:2:0, 4:2:2 and 4:4:4 (H.264 and HEVC), and an odd-width
one. Each case is decoded by NVDEC (where it can) and on the CPU, encoded at 8 and 10 bits,
with and without --final-resize-height and --deinterlace, by NVENC and the CPU encoders, and
must give the source's frames: the right pixel format and size, every frame, and a PSNR
against the source (ffmpeg's own decode and scale) of at least 30 dB on each plane. Without a
model only passthrough (no chain active) is tested; with a 2x model (e.g.
2x_AnimeJaNai_HD_V3_Performance_SPANF3_b5f48_unshuffle_fp16.onnx) the upscale path too, which
builds an engine the first time.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys

runtime, ffbin, work = (Path(a).resolve() for a in sys.argv[1:4])
model = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
work.mkdir(parents=True, exist_ok=True)
encode = runtime / "aji_encode.exe"
ffmpeg, ffprobe = ffbin / "ffmpeg.exe", ffbin / "ffprobe.exe"
failures = []

conf = work / "animejanai.conf"
conf.write_text("[global]\nconfig_version=2\nbackend=TensorRT\n"
                "[slot_1]\nprofile_name=Passthrough\nchain_1_max_resolution=320x240\n"
                "chain_1_max_fps=31\nchain_1_model_1_name=none\n"
                + (f"[slot_2]\nprofile_name=Upscale\nchain_1_max_resolution=1280x720\n"
                   f"chain_1_max_fps=31\nchain_1_model_1_name={model.stem}\n" if model else ""),
                encoding="utf-8")
if model:
    shutil.copyfile(model, work / model.name)

FRAMES = 48
SOURCES = {     # name: (encoder args, size)
    "420p8": (["-c:v", "libx264", "-pix_fmt", "yuv420p"], "1280x720"),
    "420p10": (["-c:v", "libx265", "-pix_fmt", "yuv420p10le"], "1280x720"),
    "422p8": (["-c:v", "libx264", "-pix_fmt", "yuv422p"], "1280x720"),
    "422p10": (["-c:v", "libx265", "-pix_fmt", "yuv422p10le"], "1280x720"),
    "444p8": (["-c:v", "libx264", "-pix_fmt", "yuv444p"], "1280x720"),
    "444p10": (["-c:v", "libx265", "-pix_fmt", "yuv444p10le"], "1280x720"),
    "odd420p8": (["-c:v", "libx264", "-pix_fmt", "yuv420p"], "853x480"),
}
for name, (args, size) in SOURCES.items():
    src = work / f"{name}.mkv"
    if not src.exists():
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=s={size}:r=24000/1001", "-frames:v", str(FRAMES), *args,
                        "-x265-params", "log-level=error", "-crf", "12", src], check=True)


def probe(path):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v", "-count_packets",
                          "-show_entries", "stream=pix_fmt,width,height,nb_read_packets",
                          "-of", "default=nw=1", path], capture_output=True, text=True).stdout
    return dict(l.split("=", 1) for l in out.split())


def psnr(out, src, w, h, fmt, upscaled):
    # compare at the output's size, or at the source's for an upscale. Both get the same
    # colour tags first: the output is tagged (BT.709 for an untagged HD source), and
    # libavfilter would otherwise convert one side's matrix to the other's.
    tags = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv,"
    if upscaled:
        sw, sh = probe(src)["width"], probe(src)["height"]
        a, b = f"scale={sw}:{sh}:flags=area,format={fmt}", f"format={fmt}"
    else:
        a, b = f"format={fmt}", f"crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0,scale={w}:{h}:flags=bicubic,format={fmt}"
    log = subprocess.run([ffmpeg, "-hide_banner", "-i", out, "-i", src, "-filter_complex",
                          f"[0:v]{tags}{a}[a];[1:v]{tags}{b}[b];[a][b]psnr", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.search(r"PSNR y:([\d.inf]+) u:([\d.inf]+) v:([\d.inf]+)", log)
    return [float(x) for x in m.groups()] if m else [0, 0, 0]


def case(src, decoder, vcodec, pix_fmt, extra=(), slot="1"):
    tag = f"{src} {decoder} {vcodec} {pix_fmt} {' '.join(extra)} slot{slot}".strip()
    out = work / ("out-" + re.sub(r"[^\w]+", "_", tag) + ".mkv")
    p = subprocess.run([encode, "--input", work / f"{src}.mkv", "--output", out, "--conf", conf,
                        "--slot", slot, "--model-dir", work, "--overwrite", "--progress", "none",
                        "--decoder", decoder, "--vcodec", vcodec, "--pix-fmt", pix_fmt, *extra],
                       capture_output=True, text=True, errors="replace", cwd=runtime)
    if p.returncode != 0:
        print(f"FAIL {tag}: exit {p.returncode}: {p.stderr.strip().splitlines()[-1:]}", flush=True)
        failures.append(tag)
        return
    info = probe(out)
    want = {"yuv420p": "yuv420p", "yuv420p10": "yuv420p10le",
            "yuv444p": "yuv444p", "yuv444p10": "yuv444p10le"}[pix_fmt]
    w, h = int(info["width"]), int(info["height"])
    frames = int(info["nb_read_packets"])
    fmt = "yuv444p10le" if "444" in pix_fmt else "yuv420p10le"
    y, u, v = psnr(out, work / f"{src}.mkv", w, h, fmt, slot == "2")
    # bwdif at field rate doubles the frames
    want_frames = FRAMES * (2 if "bwdif" in extra else 1)
    ok = info["pix_fmt"] == want and frames == want_frames and min(y, u, v) >= 30
    print(f"{'PASS' if ok else 'FAIL'} {tag}: {info['pix_fmt']} {w}x{h} {frames} frames, "
          f"PSNR {y:.1f}/{u:.1f}/{v:.1f}", flush=True)
    if not ok:
        failures.append(tag)


for src in SOURCES:
    for dec in ("auto", "cpu"):
        for pf in ("yuv420p", "yuv420p10"):
            case(src, dec, "hevc_nvenc", pf)
            case(src, dec, "hevc_nvenc", pf, ["--final-resize-height", "360"])
        for enc in ("libx264", "libx265", "libsvtav1"):
            case(src, dec, enc, "yuv420p10")
        case(src, dec, "libx264", "yuv420p", ["--final-resize-height", "360"])
        if "444" in src:
            case(src, dec, "libx265", "yuv444p10")
    case(src, "auto", "hevc_nvenc", "yuv420p10", ["--deinterlace", "bwdif"])
    case(src, "auto", "libx265", "yuv420p", ["--deinterlace", "bwdif"])
    if model:
        for dec in ("auto", "cpu"):
            for pf in ("yuv420p", "yuv420p10"):
                case(src, dec, "hevc_nvenc", pf, slot="2")
            case(src, dec, "libx265", "yuv420p10", slot="2")

print(f"{len(failures)} failure(s)" if failures else "all passed")
sys.exit(1 if failures else 0)
