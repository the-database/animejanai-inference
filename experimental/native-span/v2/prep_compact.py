"""Dump an SRVGG Compact 2x ONNX model (conv+PReLU stack, depth-to-space(2), + nearest-upsampled input) for
compact_native.cu, plus a real-frame NCHW fp16 input and the ONNX Runtime CUDA fp16 output.

usage: python prep_compact.py MODEL.onnx FRAME.rgba W H OUTDIR
weights.bin: per conv in graph order: W [ky][kx][cin][cout] fp16, bias fp32, then (if a PReLU follows) slope fp32[cout]
"""
import os, site, sys
import numpy as np
import onnx
from onnx import numpy_helper

model, frame, W, H, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(out, exist_ok=True)
g = onnx.load(model).graph
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
convs = [n for n in g.node if n.op_type == "Conv"]
consumers = {}
for n in g.node:
    for i in n.input: consumers.setdefault(i, []).append(n)
C = init[convs[0].input[1]].shape[0]
with open(os.path.join(out, "weights.bin"), "wb") as f:
    for cv in convs:
        w = init[cv.input[1]].astype(np.float16)
        f.write(np.ascontiguousarray(w.transpose(2, 3, 1, 0)).tobytes())
        f.write(init[cv.input[2]].astype(np.float32).tobytes())
        nxt = consumers.get(cv.output[0], [])
        if nxt and nxt[0].op_type == "PRelu":
            f.write(init[nxt[0].input[1]].astype(np.float32).reshape(-1).tobytes())
open(os.path.join(out, "model.txt"), "w").write(f"{C} {len(convs)}\n")

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
print(f"C={C} convs={len(convs)} input {x.shape} output {y.shape}")
