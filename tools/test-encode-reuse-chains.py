"""GPU regression for held-frame reuse in aji_encode's RIFE chains.

Usage: python tools/test-encode-reuse-chains.py --runtime <dir> --ffmpeg-bin <dir>
           --model-dir <dir> --rife-model-dir <dir> --clip <file> [--clip <file> ...]
           [--backend tensorrt|directml] [--device <id>] [--chains R,UR,RU,RUpre]
           [--factors 2,3,5/2,3/2] [--frames N] [--lossless] [--pix-fmt yuv420p10]
           [--damage] [--work <dir>]

<runtime> holds aji_encode.exe and the DLLs it loads; <ffmpeg-bin> holds ffmpeg.exe and
ffprobe.exe. The models: --model (default the 2x Balanced V3.1 model) in --model-dir, RIFE 4.22
in --rife-model-dir. Needs a GPU (NVIDIA for NVENC; engines are built on first use).

For each clip it encodes the plain upscale chain once with AJI_REUSE_TRACE, then each RIFE chain
(R = RIFE only, UR = upscale then RIFE, RU = RIFE first, RUpre = RIFE first with a pre-resize to
720 lines) at each factor twice, with reuse (traced) and with --no-reuse-held-frames, and checks:
- packet count and the pts list are the same;
- the trace is byte-identical to the plain chain's (RIFE only interpolates the frames in the
  output format, so its comparisons see them converted: exactly from 8 to 10 bits, but rounded
  from 10 to 8, so for a 10-bit source at 8-bit output the trace is reported, not checked);
- with --lossless (NVENC lossless, so the decoded frames are what aji_encode put out): every
  output frame no held frame touches is bit-identical to the no-reuse encode, and the others
  are at least --min-psnr dB from it. Output frame k sits in the pair (i, i+1) with
  i = floor(k * den / num) at phase acc = k * den mod num; a held frame H touches the frames of
  the pair that ends on it (they repeat i) and, unless the chain is RIFE only (whose frames go
  out as they are), H itself and the in-betweens of the pair that starts on it. A held frame is
  near-identical to its anchor, so the PSNR floor can't tell which of them a touched frame
  repeats; which pairs are skipped is checked by the trace and the logged in-between count.
--damage also encodes a clip with damaged packets (CPU decoder, --on-damage continue) through
each chain, with and without reuse: packet count and pts match, and the first frame of each
damaged spot (DAMAGE line) and the frame after it are upscaled.
"""
import argparse
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--runtime", required=True)
ap.add_argument("--ffmpeg-bin", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--rife-model-dir", required=True)
ap.add_argument("--model", default="2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16")
ap.add_argument("--clip", action="append", default=[])
ap.add_argument("--backend", default="tensorrt", choices=("tensorrt", "directml"))
ap.add_argument("--device", default="")
ap.add_argument("--chains", default="R,UR,RU,RUpre")
ap.add_argument("--factors", default="2,3,5/2,3/2")
ap.add_argument("--frames", type=int, default=0, help="only the first N frames of each clip")
ap.add_argument("--lossless", action="store_true")
ap.add_argument("--vquality", default="-preset p1 -b:v 50M")
ap.add_argument("--pix-fmt", default="yuv420p10")
ap.add_argument("--min-psnr", type=float, default=44.0,
                help="floor for touched frames. Held frames are the plain chain's reused frames"
                     " (~48 dB and up); an in-between of a held pair repeats N-1 where RIFE would"
                     " put out its own frame between two near-identical frames, which differs from"
                     " N-1 by about as much (45-49 dB on testclip)")
ap.add_argument("--damage", action="store_true")
ap.add_argument("--work", default="")
ap.add_argument("--trtexec", default="")
a = ap.parse_args()

runtime = Path(a.runtime).resolve()
encode = runtime / "aji_encode.exe"
ffbin = Path(a.ffmpeg_bin).resolve()
ffmpeg, ffprobe = ffbin / "ffmpeg.exe", ffbin / "ffprobe.exe"
work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix="aji-reuse-chains-"))
work.mkdir(parents=True, exist_ok=True)
VQ = "-preset p4 -tune lossless" if a.lossless else a.vquality
failures = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)
    return ok


def conf(path, chain, num=2, den=1):
    """chain: U (plain), R, UR, RU, RUpre"""
    be = "DirectML" if a.backend == "directml" else "TensorRT"
    lines = ["[global]", "config_version=2", "logging=yes", f"backend={be}", "[slot_1]",
             "profile_name=encode", "chain_1_min_resolution=0x0", "chain_1_max_resolution=0x0",
             "chain_1_min_fps=0", "chain_1_max_fps=0"]
    if chain != "R":
        lines += [f"chain_1_model_1_resize_height_before_upscale={720 if chain == 'RUpre' else 0}",
                  "chain_1_model_1_resize_factor_before_upscale=100",
                  f"chain_1_model_1_name={a.model}"]
    lines += [f"chain_1_rife={'no' if chain == 'U' else 'yes'}", "chain_1_rife_model=422",
              f"chain_1_rife_factor_numerator={num}", f"chain_1_rife_factor_denominator={den}",
              "chain_1_rife_scene_detect_threshold=0.150", "chain_1_rife_ensemble=no",
              f"chain_1_rife_before_upscale={'no' if chain == 'UR' else 'yes'}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(src, cfg, out, trace=None, reuse=True, extra=(), vq=None):
    args = [encode, "--input", src, "--output", out, "--conf", cfg, "--slot", "1",
            "--model-dir", a.model_dir, "--rife-model-dir", a.rife_model_dir,
            "--backend", a.backend, "--vcodec", "hevc_nvenc", "--vquality", vq or VQ,
            "--pix-fmt", a.pix_fmt, "--overwrite", "--progress", "none", "--no-audio",
            "--no-subs", "--no-chapters", *extra]
    if a.trtexec:
        args += ["--trtexec", a.trtexec]
    if a.device:
        args += ["--device", a.device]
    if not reuse:
        args.append("--no-reuse-held-frames")
    env = dict(os.environ)
    env.pop("AJI_REUSE_TRACE", None)
    if trace:
        Path(trace).unlink(missing_ok=True)
        env["AJI_REUSE_TRACE"] = str(trace)
    p = subprocess.run([str(x) for x in args], capture_output=True, text=True, errors="replace",
                       cwd=runtime, env=env)
    Path(out).with_suffix(".log").write_text(p.stderr, encoding="utf-8")
    return p


def pts(path):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts", "-of", "csv=p=0", path], capture_output=True, text=True,
                         check=True).stdout
    # sorted: NVENC picks frame types from content, and repeated frames change its picks
    # (e.g. IIIBBBP with reuse, IBBBP without), so packet order can differ while pts can't
    return sorted(int(x) for x in out.split() if x.strip() and x.strip() != "N/A")


def held_set(trace):
    held = set()
    for line in Path(trace).read_text().splitlines():
        f = line.split(",")
        if len(f) == 2 and f[1] == "reuse":
            held.add(int(f[0]))
    return held


def decisions(trace):
    return {int(f[0]): f[1] for f in (l.split(",") for l in Path(trace).read_text().splitlines())
            if len(f) == 2}


def touched(k, num, den, held, chain):
    i, acc = divmod(k * den, num)
    if i + 1 in held and acc:
        return True                      # an in-between of a held pair
    if chain == "R":
        return False                     # frames go out as they are
    return i in held                     # H itself, or interpolated from it


def frame_stats(x, y):
    """per output frame (psnr_avg, psnr_y) of x against y; inf = identical"""
    stats = work / "psnr.txt"
    stats.unlink(missing_ok=True)
    subprocess.run([ffmpeg, "-v", "error", "-i", x, "-i", y, "-lavfi",
                    f"[0:v][1:v]psnr=stats_file='{stats.name}'", "-f", "null", "-"],
                   check=True, cwd=work)
    res = []
    for line in stats.read_text().splitlines():
        m = dict(kv.split(":", 1) for kv in line.split() if ":" in kv)
        res.append((float(m["psnr_avg"]), float(m["psnr_y"])))
    return res


def src_depth(path):
    pf = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=pix_fmt", "-of", "csv=p=0", path], capture_output=True,
                        text=True, check=True).stdout.strip()
    return 10 if "10" in pf else 8


factors = [tuple(int(x) for x in (f.split("/") + ["1"])[:2]) for f in a.factors.split(",")]
chains = a.chains.split(",")
out_depth = 10 if "10" in a.pix_fmt else 8

for clip in a.clip:
    clip = Path(clip).resolve()
    name = clip.stem
    if a.frames:
        cut = work / f"{name}_{a.frames}.mkv"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", clip, "-map", "0:v:0", "-c", "copy",
                        "-frames:v", str(a.frames), cut], check=True)
        clip, name = cut, cut.stem
    plain_trace = work / f"{name}_U.csv"
    p = run(clip, conf(work / "U.conf", "U"), work / f"{name}_U.mkv", plain_trace, vq=a.vquality)
    check(p.returncode == 0, f"{name} U: plain chain encodes (exit {p.returncode})")
    (work / f"{name}_U.mkv").unlink(missing_ok=True)
    plain = plain_trace.read_bytes()
    held = held_set(plain_trace)
    print(f"{name}: plain chain reused {len(held)} frames", flush=True)
    for chain in chains:
        for num, den in factors:
            tag = f"{name}_{chain}_{num}-{den}"
            cfg = conf(work / f"{chain}_{num}-{den}.conf", chain, num, den)
            on, off, tr = work / f"{tag}_reuse.mkv", work / f"{tag}_noreuse.mkv", work / f"{tag}.csv"
            p1 = run(clip, cfg, on, tr)
            p0 = run(clip, cfg, off, reuse=False)
            if not check(p1.returncode == 0 and p0.returncode == 0,
                         f"{tag}: both encodes succeed (exit {p1.returncode}/{p0.returncode})"):
                print(p1.stderr[-1500:], p0.stderr[-1500:])
                continue
            hp = re.search(r"rife: (\d+) held pairs not interpolated \((\d+) in-betweens", p1.stderr)
            print(f"  {tag}: " + "; ".join(l for l in p1.stderr.splitlines()
                                            if l.startswith(("rife:", "held frames: reused"))),
                  flush=True)
            check(hp is not None and "held pairs" not in p0.stderr,
                  f"{tag}: held pairs are logged with reuse only")
            t1, t0 = pts(on), pts(off)
            check(len(t1) == len(t0) and t1 == t0,
                  f"{tag}: packet count and pts match --no-reuse-held-frames ({len(t1)} / {len(t0)})")
            same = tr.read_bytes() == plain
            if chain == "R" and src_depth(clip) > out_depth:
                d1, d0 = decisions(tr), decisions(plain_trace)
                diff = sum(1 for k in d0 if d1.get(k) != d0[k])
                print(f"INFO {tag}: RIFE only compares {out_depth}-bit frames, the plain chain "
                      f"{src_depth(clip)}-bit: trace {'identical' if same else 'differs'}, "
                      f"{diff} of {len(d0)} decisions differ", flush=True)
                held_r = held_set(tr)
            else:
                check(same, f"{tag}: trace byte-identical to the plain chain's")
                held_r = held
            if hp:
                exp = sum(1 for k in range(len(t1)) if touched(k, num, den, held_r, "R")
                          and divmod(k * den, num)[1])
                check(int(hp.group(2)) == exp,
                      f"{tag}: {hp.group(2)} in-betweens repeated, {exp} expected from the trace")
            if a.lossless:
                st = frame_stats(on, off)
                check(len(st) == len(t1), f"{tag}: {len(st)} frames compared")
                bad_same = [k for k, (pa, _) in enumerate(st)
                            if not touched(k, num, den, held_r, chain) and pa != float("inf")]
                tk = [st[k] for k in range(len(st)) if touched(k, num, den, held_r, chain)]
                lo = min((pa for pa, _ in tk), default=float("inf"))
                lo_y = min((py for _, py in tk), default=float("inf"))
                n_same = sum(1 for pa, _ in tk if pa == float("inf"))
                check(not bad_same, f"{tag}: {len(st) - len(tk)} untouched frames bit-identical"
                      + (f" (not: {bad_same[:10]})" if bad_same else ""))
                check(lo >= a.min_psnr, f"{tag}: {len(tk)} touched frames ({n_same} identical), "
                      f"min PSNR {lo:.2f} dB (Y {lo_y:.2f})")
                kinds = {}
                for k, (pa, _) in enumerate(st):
                    i, acc = divmod(k * den, num)
                    kind = ("held frame" if not acc and i in held_r and chain != "R" else
                            "held pair" if acc and i + 1 in held_r else
                            "after held" if acc and i in held_r and chain != "R" else None)
                    if kind:
                        kinds.setdefault(kind, []).append(pa)
                print(f"  {tag}: min / median PSNR: " + ", ".join(
                    f"{kd} {min(v):.2f} / {sorted(v)[len(v) // 2]:.2f} dB (n={len(v)})"
                    for kd, v in kinds.items()), flush=True)
            for f in (on, off):
                f.unlink(missing_ok=True)

if a.damage:
    # 10 s of 1080p drawings held for three frames (testsrc2 at 8 fps), H.264 with a 2 s GOP,
    # damaged at 30% and 60% as in test-encode-damage.py
    clean = work / "dmg_clean.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=1920x1080:r=8:d=10,fps=24000/1001", "-c:v", "libx264", "-preset",
                    "fast", "-g", "48", "-pix_fmt", "yuv420p", clean], check=True)
    data = bytearray(clean.read_bytes())
    rnd = random.Random(7)
    for frac in (0.3, 0.6):
        off = int(len(data) * frac)
        data[off:off + 4000] = bytes(rnd.randrange(256) for _ in range(4000))
    dmg = work / "dmg.mkv"
    dmg.write_bytes(bytes(data))
    for chain in chains:
        tr = work / f"dmg_{chain}.csv"
        out, out0 = work / f"dmg_{chain}.mkv", work / f"dmg_{chain}_noreuse.mkv"
        cfg = conf(work / f"dmg_{chain}.conf", chain)
        p = run(dmg, cfg, out, tr, vq=a.vquality,
                extra=("--decoder", "cpu", "--on-damage", "continue"))
        p0 = run(dmg, cfg, out0, reuse=False, vq=a.vquality,
                 extra=("--decoder", "cpu", "--on-damage", "continue"))
        if p.returncode == 0 and p0.returncode == 0:
            t1, t0 = pts(out), pts(out0)
            check(t1 == t0, f"damage {chain}: packet count and pts match --no-reuse-held-frames "
                  f"({len(t1)} / {len(t0)})")
        spots = [int(m.group(1)) for m in re.finditer(r"^DAMAGE t=\S+ frame=(\d+)$", p.stdout, re.M)]
        d = decisions(tr) if tr.exists() else {}
        reused = sum(1 for v in d.values() if v == "reuse")
        bad = [f for f in spots for g in (f, f + 1) if d.get(g) == "reuse"]
        check(p.returncode == 0 and spots and reused and not bad,
              f"damage {chain}: exit {p.returncode}, damaged frames {spots} and the frames after "
              f"them upscaled ({reused} others reused)")
        out.unlink(missing_ok=True)
        out0.unlink(missing_ok=True)

print(f"{len(failures)} failure(s)" if failures else "all passed")
sys.exit(1 if failures else 0)
