"""Dump a SPANF3 2x (unshuffle) ONNX model's weights for spanf3_native.cu, a real-frame NCHW fp16 input,
and the ONNX Runtime CUDA fp16 output as reference.

usage: python prep_full.py MODEL.onnx FRAME.rgba W H OUTDIR
weights.bin layout (all conv weights [ky][kx][cin][cout] fp16, then fp32 bias):
  conv_1 (cin 12), blocks 1..N: c1, c2, c3, att (att is [cin][cout]), conv_cat (cin 4C, 1x1), to_pixel
"""
import os, site, sys
import numpy as np
import onnx
from onnx import numpy_helper

model, frame, W, H, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(out, exist_ok=True)
Wt = {t.name: numpy_helper.to_array(t) for t in onnx.load(model).graph.initializer}
nb = 1
while f"block_{nb + 1}.c1_r.eval_conv.weight" in Wt:
    nb += 1
C = Wt["conv_1.1.eval_conv.weight"].shape[0]

def conv(f, wname, bname):
    w = Wt[wname].astype(np.float16)
    f.write(np.ascontiguousarray(w.transpose(2, 3, 1, 0)).tobytes())
    f.write(Wt[bname].astype(np.float32).tobytes())

with open(os.path.join(out, "weights.bin"), "wb") as f:
    conv(f, "conv_1.1.eval_conv.weight", "conv_1.1.eval_conv.bias")
    for b in range(1, nb + 1):
        for c in ("c1_r", "c2_r", "c3_r"):
            conv(f, f"block_{b}.{c}.eval_conv.weight", f"block_{b}.{c}.eval_conv.bias")
        conv(f, f"block_{b}.att_conv.weight", f"block_{b}.att_conv.bias")
    conv(f, "conv_cat.weight", "conv_cat.bias")
    conv(f, "to_pixel.eval_conv.weight", "to_pixel.eval_conv.bias")
open(os.path.join(out, "model.txt"), "w").write(f"{C} {nb}\n")

src = np.fromfile(frame, np.uint8).reshape(H, W, 4)[..., :3]
x = np.ascontiguousarray((src.astype(np.float32) / 255).transpose(2, 0, 1)[None]).astype(np.float16)
x.tofile(os.path.join(out, "input.bin"))

for sp in site.getsitepackages():
    p = os.path.join(sp, "tensorrt_libs")
    if os.path.isdir(p): os.add_dll_directory(p)
import onnxruntime as ort
ort.preload_dlls()
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
y = ort.InferenceSession(model, so, providers=["CUDAExecutionProvider"]).run(None, {"input": x})[0]
y.astype(np.float16).tofile(os.path.join(out, "ref_out.bin"))
print(f"C={C} blocks={nb} input {x.shape} output {y.shape}")
