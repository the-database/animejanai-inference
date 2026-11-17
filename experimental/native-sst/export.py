"""Seeded SST exports for the native kernels, and PyTorch reference outputs.

  python export.py PRESET OUTDIR [--ckpt PATH]        -> OUTDIR/4x_<preset>_<seed|ckpt>.onnx
  python export.py PRESET OUTDIR --ref IN.ppm OUT.ppm  -> the fp32 reference upscale of IN

Without --ckpt the weights are seeded random, made to exercise every path: the RIB factors and
LayerNorm affines are larger than upstream's init (so the position bias and norms matter), and the
final conv is scaled so the residual on top of the nearest-upsampled input stays near 0.1 (a
plausible image, not a clipped one). The export is fp32, opset 17, dynamic H/W, traced at 72x72
(not a multiple of any window, so every layer has its reflect Pad).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sst_ref  # noqa: E402


def seeded(name, seed=0):
    torch.manual_seed(seed)
    m = sst_ref.build(name).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, sst_ref.LayerNorm):
                mod.weight.normal_(1.0, 0.1)
                mod.bias.normal_(0.0, 0.1)
            if isinstance(mod, sst_ref.WindowAttention2D):
                mod.to_q.normal_(0.0, 0.5)
                mod.to_k.normal_(0.0, 0.5)
                mod.hidden_b.normal_(0.0, 0.1)
        x = torch.rand(1, 3, 64, 64)
        res = m(x) - torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")
        k = 0.1 / float(res.std())
        m.upsampler.final_conv.weight.mul_(k)
        m.upsampler.final_conv.bias.mul_(k)
    return m


def load(name, ckpt):
    if not ckpt:
        return seeded(name)
    m = sst_ref.build(name).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("params_ema", sd.get("params", sd))
    m.load_state_dict(sd, strict=True)
    return m


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    parts, i = [], 0
    while len(parts) < 4:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        parts.append(data[i:j])
        i = j
    i += 1
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(data[i:i + w * h * 3], np.uint8).reshape(h, w, 3)


def write_ppm(path, a):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (a.shape[1], a.shape[0]))
        f.write(np.ascontiguousarray(a).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=sorted(sst_ref.PRESETS))
    ap.add_argument("outdir")
    ap.add_argument("--ckpt")
    ap.add_argument("--ref", nargs=2, metavar=("IN", "OUT"))
    a = ap.parse_args()
    m = load(a.preset, a.ckpt)
    if a.ref:
        x = torch.from_numpy(read_ppm(a.ref[0]).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            y = m.to(dev)(x.to(dev)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        write_ppm(a.ref[1], (y * 255.0 + 0.5).astype(np.uint8))
        np.save(a.ref[1] + ".npy", y.astype(np.float32))
        return
    os.makedirs(a.outdir, exist_ok=True)
    tag = "ckpt" if a.ckpt else "seed0"
    path = os.path.join(a.outdir, f"4x_{a.preset}_{tag}.onnx")
    torch.onnx.export(m, torch.rand(1, 3, 72, 72), path, opset_version=17, dynamo=False, input_names=["input"],
                      output_names=["output"], dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}})
    print(path)


if __name__ == "__main__":
    main()
