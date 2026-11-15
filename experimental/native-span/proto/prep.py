import os, site, sys, numpy as np, onnx
from onnx import numpy_helper, utils
for sp in site.getsitepackages():
    d = os.path.join(sp, "tensorrt_libs")
    if os.path.isdir(d): os.add_dll_directory(d)
import onnxruntime as ort
ort.preload_dlls()
model, frame, out = sys.argv[1], sys.argv[2], sys.argv[3]
FEAT = "/conv_1/conv_1.1/eval_conv/Conv_output_0"
OUTS = ["/block_1/Mul_output_0", "/block_5/Mul_output_0", "/block_5/c1_r/eval_conv/Conv_output_0"]
utils.extract_model(model, f"{out}/head.onnx", ["input"], [FEAT])
utils.extract_model(model, f"{out}/body.onnx", [FEAT], OUTS)
m = onnx.load(model); W = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
with open(f"{out}/weights.bin", "wb") as f:
    for b in range(1, 6):
        for c in ("c1_r", "c2_r", "c3_r"):
            w = W[f"block_{b}.{c}.eval_conv.weight"].astype(np.float16)        # [cout][cin][3][3]
            f.write(np.ascontiguousarray(w.transpose(2, 3, 1, 0)).tobytes())  # [ky][kx][cin][cout]
            f.write(W[f"block_{b}.{c}.eval_conv.bias"].astype(np.float32).tobytes())
        w = W[f"block_{b}.att_conv.weight"].astype(np.float16)[:, :, 0, 0]    # [cout][cin]
        f.write(np.ascontiguousarray(w.T).tobytes())                          # [cin][cout]
        f.write(W[f"block_{b}.att_conv.bias"].astype(np.float32).tobytes())
h, w = 1080, 1920
src = np.fromfile(frame, np.uint8).reshape(h, w, 4)[..., :3]
x = (src.astype(np.float32) / 255).transpose(2, 0, 1)[None].astype(np.float16)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
feat = ort.InferenceSession(f"{out}/head.onnx", so, providers=["CUDAExecutionProvider"]).run(None, {"input": x})[0]
print("feat", feat.shape, feat.dtype)
nhwc = lambda a: np.ascontiguousarray(a[0].transpose(1, 2, 0)).astype(np.float16)
nhwc(feat).tofile(f"{out}/feat.bin")
for prov, tag in ((["CUDAExecutionProvider"], "ref_cuda16"),):
    outs = ort.InferenceSession(f"{out}/body.onnx", so, providers=prov).run(None, {FEAT: feat})
    for name, a in zip(("outb1", "out5", "mid5"), outs): nhwc(a).tofile(f"{out}/{tag}_{name}.bin")
# fp32 reference: same weights upcast, run on CPU-free CUDA fp32
mb = onnx.load(f"{out}/body.onnx")
from onnx import TensorProto, helper
for t in mb.graph.initializer:
    if t.data_type == TensorProto.FLOAT16:
        t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
for v in list(mb.graph.input) + list(mb.graph.output) + list(mb.graph.value_info):
    if v.type.tensor_type.elem_type == TensorProto.FLOAT16: v.type.tensor_type.elem_type = TensorProto.FLOAT
for n in mb.graph.node:
    for a in n.attribute:
        if a.name == "to" and a.i == TensorProto.FLOAT16: a.i = TensorProto.FLOAT
onnx.save(mb, f"{out}/body_fp32.onnx")
outs = ort.InferenceSession(f"{out}/body_fp32.onnx", so, providers=["CUDAExecutionProvider"]).run(None, {FEAT: feat.astype(np.float32)})
for name, a in zip(("outb1", "out5", "mid5"), outs): np.ascontiguousarray(a[0].transpose(1, 2, 0)).astype(np.float32).tofile(f"{out}/ref_fp32_{name}.bin")
print("done", [o.shape for o in outs])
