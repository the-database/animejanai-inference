"""GPU regression for aji_encode's damaged-input handling (--on-damage) and read errors.

Usage: python tools/test-encode-damage.py <runtime-dir> <ffmpeg-bin-dir> [<work-dir>]
<runtime-dir> holds aji_encode.exe, aji.dll, aji_trt.dll and the DLLs they load (TensorRT,
cudart, the ffmpeg libraries); <ffmpeg-bin-dir> holds ffmpeg.exe and ffprobe.exe. Needs an
NVIDIA GPU. Builds no engine: the conf's only chain doesn't match, so every encode is a
passthrough transcode. Windows only: the read error is a byte-range lock (msvcrt.locking)
held on the middle of the input, which makes the reads there fail with EACCES.
"""
import msvcrt
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

runtime, ffbin = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path(tempfile.mkdtemp(prefix="aji-damage-"))
work.mkdir(parents=True, exist_ok=True)
encode = runtime / "aji_encode.exe"
ffmpeg, ffprobe = ffbin / "ffmpeg.exe", ffbin / "ffprobe.exe"
DAMAGE = re.compile(r"^DAMAGE t=(\d+\.\d{3}) frame=(-1|\d+)$")
failures = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)


conf = work / "animejanai.conf"
conf.write_text("[global]\nconfig_version=2\nbackend=TensorRT\n[slot_1]\nprofile_name=Passthrough\n"
                "chain_1_max_resolution=320x240\nchain_1_max_fps=31\n"
                "chain_1_model_1_name=none\n", encoding="utf-8")

# 10 s of 1280x720 H.264 at 23.976 with a 2 s GOP, plus audio, in three containers
clean = work / "clean.mkv"
subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=24000/1001:d=10",
                "-f", "lavfi", "-i", "sine=d=10", "-c:v", "libx264", "-preset", "fast", "-g", "48",
                "-pix_fmt", "yuv420p", "-c:a", "aac", clean], check=True)
for ext in ("ts", "mp4"):
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", clean, "-map", "0", "-c", "copy",
                    work / f"clean.{ext}"], check=True)


def damaged(ext):
    data = bytearray((work / f"clean.{ext}").read_bytes())
    rnd = random.Random(7)
    for frac in (0.3, 0.6):
        off = int(len(data) * frac)
        data[off:off + 4000] = bytes(rnd.randrange(256) for _ in range(4000))
    out = work / f"damaged.{ext}"
    out.write_bytes(bytes(data))
    return out


def run(src, *args):
    out = work / f"out-{src.stem}.mkv"
    p = subprocess.run([encode, "--input", src, "--output", out, "--conf", conf, "--slot", "1",
                        "--model-dir", work, "--overwrite", "--progress", "none", *args],
                       capture_output=True, text=True, errors="replace", cwd=runtime)
    lines = [l for l in p.stdout.splitlines() if l.startswith("DAMAGE")]
    return p.returncode, lines, p.stderr


for ext in ("mkv", "ts", "mp4"):
    for dec in ("auto", "cpu"):
        rc, lines, _ = run(work / f"clean.{ext}", "--decoder", dec)
        check(rc == 0 and not lines, f"clean {ext} {dec}: exit 0, no DAMAGE (got {rc}, {lines})")
    src = damaged(ext)
    for dec in ("auto", "cpu"):
        rc_f, lines_f, err_f = run(src, "--decoder", dec)
        rc_c, lines_c, err_c = run(src, "--decoder", dec, "--on-damage", "continue")
        check(all(DAMAGE.match(l) for l in lines_f + lines_c), f"damaged {ext} {dec}: DAMAGE line format")
        if dec == "cpu":    # NVDEC conceals macroblock damage unseen
            check(rc_f == 3, f"damaged {ext} {dec}: fail exits 3 (got {rc_f})")
        if rc_f == 3:
            check(len(lines_f) == 1 and re.search(r"damaged video at \d+:\d\d\.\d \(", err_f),
                  f"damaged {ext} {dec}: fail prints one DAMAGE line and says where on stderr")
            check(rc_c == 0 and lines_c and lines_c[0] == lines_f[0],
                  f"damaged {ext} {dec}: continue exits 0 and lists the same first spot")
            check("spot(s), encoded past" in err_c, f"damaged {ext} {dec}: continue prints a total")
        else:
            check(rc_f == 0 and not lines_f and rc_c == 0 and not lines_c,
                  f"damaged {ext} {dec}: nothing seen, both modes exit 0 (got {rc_f}/{rc_c})")

# a read error halfway: the encode fails, whatever --on-damage says
locked = work / "locked.mkv"
shutil.copyfile(clean, locked)
with open(locked, "rb") as f:
    f.seek(locked.stat().st_size // 2)
    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 65536)
    for mode in ("fail", "continue"):
        rc, lines, err = run(locked, "--on-damage", mode)
        check(rc == 1 and "reading the input failed" in err, f"read error, {mode}: exit 1 (got {rc})")
    f.seek(locked.stat().st_size // 2)
    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 65536)

print(f"{len(failures)} failure(s)" if failures else "all passed")
sys.exit(1 if failures else 0)
