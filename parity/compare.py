#!/usr/bin/env python3
"""Per-plane PSNR between aji kernel output and VS/zimg goldens.

usage:
  compare.py rgbf16 OURS GOLDEN W H [--crop N] [--label S]
  compare.py yuv8   OURS GOLDEN W H [--crop N] [--label S]   # ours NV12, golden planar
  compare.py yuv10  OURS GOLDEN W H [--crop N] [--label S]   # ours P010, golden planar 10-bit

Prints per-plane PSNR (dB, peak = format max), max abs error, and the
fraction of samples differing. --crop N excludes an N-pixel border
(diagnostic: separates boundary-handling deltas from interior math).
"""
import sys

import numpy as np


def psnr(a, b, peak):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = np.mean(d * d)
    if mse == 0:
        return float("inf"), 0.0, 0.0
    return 10 * np.log10(peak * peak / mse), np.max(np.abs(d)), np.mean(d != 0)


def crop(p, n):
    return p[..., n:p.shape[-2] - n, n:p.shape[-1] - n] if n else p


def load_rgbf16(path, w, h):
    a = np.fromfile(path, dtype=np.float16)
    n = a.size // (w * h * 3)
    return a[: n * w * h * 3].reshape(n, 3, h, w).astype(np.float32)


def load_nv12(path, w, h, bits):
    dt = np.uint16 if bits > 8 else np.uint8
    a = np.fromfile(path, dtype=dt)
    fsz = w * h * 3 // 2
    n = a.size // fsz
    a = a[: n * fsz].reshape(n, fsz)
    y = a[:, : w * h].reshape(n, h, w)
    uv = a[:, w * h:].reshape(n, h // 2, w // 2, 2)
    u, v = uv[..., 0], uv[..., 1]
    if bits == 10:  # P010 MSB-aligned -> 10-bit
        y, u, v = (np.round(p / 64.0).astype(np.uint16) for p in (y, u, v))
    return y, u, v


def load_planar(path, w, h, bits):
    dt = np.uint16 if bits > 8 else np.uint8
    a = np.fromfile(path, dtype=dt)
    fsz = w * h + 2 * (w // 2) * (h // 2)
    n = a.size // fsz
    a = a[: n * fsz].reshape(n, fsz)
    y = a[:, : w * h].reshape(n, h, w)
    u = a[:, w * h: w * h + (w // 2) * (h // 2)].reshape(n, h // 2, w // 2)
    v = a[:, w * h + (w // 2) * (h // 2):].reshape(n, h // 2, w // 2)
    return y, u, v


def main():
    kind, ours, golden = sys.argv[1], sys.argv[2], sys.argv[3]
    w, h = int(sys.argv[4]), int(sys.argv[5])
    border = 0
    label = f"{kind} {w}x{h}"
    args = sys.argv[6:]
    while args:
        if args[0] == "--crop":
            border = int(args[1]); args = args[2:]
        elif args[0] == "--label":
            label = args[1]; args = args[2:]
        else:
            raise SystemExit(f"unknown arg {args[0]}")

    rows = []
    if kind == "rgbf16":
        a, b = load_rgbf16(ours, w, h), load_rgbf16(golden, w, h)
        nf = min(a.shape[0], b.shape[0])
        a, b = crop(a[:nf], border), crop(b[:nf], border)
        for i, pl in enumerate("RGB"):
            rows.append((pl, *psnr(a[:, i], b[:, i], 1.0)))
    else:
        bits = 10 if kind == "yuv10" else 8
        ya, ua, va = load_nv12(ours, w, h, bits)
        yb, ub, vb = load_planar(golden, w, h, bits)
        nf = min(ya.shape[0], yb.shape[0])
        peak = (1 << bits) - 1
        cb = max(border // 2, 1) if border else 0
        rows.append(("Y", *psnr(crop(ya[:nf], border), crop(yb[:nf], border), peak)))
        rows.append(("U", *psnr(crop(ua[:nf], cb), crop(ub[:nf], cb), peak)))
        rows.append(("V", *psnr(crop(va[:nf], cb), crop(vb[:nf], cb), peak)))

    worst = min(r[1] for r in rows)
    detail = "  ".join(f"{pl}:{p:7.2f}dB max:{m:g} diff:{fr * 100:.2f}%"
                       for pl, p, m, fr in rows)
    print(f"{label:46s} worst {worst:7.2f} dB   {detail}")


if __name__ == "__main__":
    main()
