"""Build SPANF3-family variants (random init), export ONNX fp16, extract head/body, write fused-kernel weights,
real-frame features and fp32 ONNX Runtime references.

usage: py -3.13 sweep_prep.py OUTDIR NAME C BLOCKS ACT ATT FRAME.rgba
"""
import os, sys, site
import numpy as np
import torch
from torch import nn

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "archsrc"))
from traiNNer.archs.spanf3_arch import SPABV2, spanf3  # noqa: E402

out_root, name, C, NB, ACT, ATT, frame = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
d = os.path.join(out_root, name); os.makedirs(d, exist_ok=True)
H, W = 1080, 1920


class Block(SPABV2):
    def __init__(self, c, act, att):
        super().__init__(c)
        self.att_kind = att
        if act == "silu":
            self.act1 = nn.SiLU()
        if att != "tanh1x1":
            del self.att_conv

    def forward(self, x):
        out1 = self.c1_r(x)
        out2 = self.c2_r(self.act1(out1))
        out3 = self.c3_r(self.act1(out2))
        if self.att_kind == "tanh1x1":
            m = torch.tanh(self.att_conv(out3)); out = (x + out3) * m
        elif self.att_kind == "sigmoid":
            m = torch.sigmoid(out3) - 0.5; out = (out3 + x) * m
        else:
            out = x + out3; m = out
        return out, out1, m


torch.manual_seed(0)
model = spanf3(scale=2, feature_channels=C, num_blocks=NB, unshuffle_mod=True)
for i in range(1, NB + 1):
    setattr(model, f"block_{i}", Block(C, ACT, ATT))
model.eval(); model.switch_to_deploy()
with torch.no_grad():  # keep random activations bounded through the stack
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            fan = mod.weight[0].numel()
            mod.weight.normal_(0, 0.7 / np.sqrt(fan)); mod.bias.normal_(0, 0.02)
model = model.half().cuda()

onnx_path = os.path.join(d, "model.onnx")
dummy = torch.rand(1, 3, 256, 256, device="cuda").half()
torch.onnx.export(model, dummy, onnx_path, opset_version=17, input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "oh", 3: "ow"}}, dynamo=False)

# fused-kernel weights: per block [ky][kx][cin][cout] fp16 + fp32 bias (+ att [cin][cout])
with open(os.path.join(d, "weights.bin"), "wb") as f:
    for i in range(1, NB + 1):
        blk = getattr(model, f"block_{i}")
        for cv in (blk.c1_r.eval_conv, blk.c2_r.eval_conv, blk.c3_r.eval_conv):
            f.write(cv.weight.detach().half().cpu().numpy().transpose(2, 3, 1, 0).copy().tobytes())
            f.write(cv.bias.detach().float().cpu().numpy().tobytes())
        if ATT == "tanh1x1":
            f.write(blk.att_conv.weight.detach().half().cpu().numpy()[:, :, 0, 0].T.copy().tobytes())
            f.write(blk.att_conv.bias.detach().float().cpu().numpy().tobytes())

import onnx
from onnx import utils, numpy_helper, TensorProto
g = onnx.load(onnx_path).graph
byname = {n.name: n for n in g.node}
FEAT = byname["/conv_1/conv_1.1/eval_conv/Conv"].output[0]
cat = next(n for n in g.node if n.op_type == "Concat" and any(c.input[0] == n.output[0] for c in g.node if c.name == "/conv_cat/Conv"))
OUTS = [cat.input[2], cat.input[1], cat.input[3]]  # out_b1, out (last block), last_mid
utils.extract_model(onnx_path, os.path.join(d, "head.onnx"), ["input"], [FEAT])
utils.extract_model(onnx_path, os.path.join(d, "body.onnx"), [FEAT], OUTS)
open(os.path.join(d, "body_io.txt"), "w").write(FEAT + "\n" + "\n".join(OUTS) + "\n")

for sp in site.getsitepackages():
    p = os.path.join(sp, "tensorrt_libs")
    if os.path.isdir(p): os.add_dll_directory(p)
import onnxruntime as ort
ort.preload_dlls()
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
src = np.fromfile(frame, np.uint8).reshape(H, W, 4)[..., :3]
x = (src.astype(np.float32) / 255).transpose(2, 0, 1)[None].astype(np.float16)
feat = ort.InferenceSession(os.path.join(d, "head.onnx"), so, providers=["CUDAExecutionProvider"]).run(None, {"input": x})[0]
np.ascontiguousarray(feat[0].transpose(1, 2, 0)).astype(np.float16).tofile(os.path.join(d, "feat.bin"))
mb = onnx.load(os.path.join(d, "body.onnx"))
for t in mb.graph.initializer:
    if t.data_type == TensorProto.FLOAT16:
        t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
for v in list(mb.graph.input) + list(mb.graph.output) + list(mb.graph.value_info):
    if v.type.tensor_type.elem_type == TensorProto.FLOAT16: v.type.tensor_type.elem_type = TensorProto.FLOAT
for n in mb.graph.node:
    for a in n.attribute:
        if a.name == "to" and a.i == TensorProto.FLOAT16: a.i = TensorProto.FLOAT
        if a.name == "value" and a.t.data_type == TensorProto.FLOAT16:
            a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32)))
outs = ort.InferenceSession(mb.SerializeToString(), so, providers=["CUDAExecutionProvider"]).run(None, {FEAT: feat.astype(np.float32)})
for nm, a in zip(("outb1", "outlast", "midlast"), outs):
    np.ascontiguousarray(a[0].transpose(1, 2, 0)).astype(np.float32).tofile(os.path.join(d, f"ref_{nm}.bin"))
print(f"{name}: C={C} blocks={NB} act={ACT} att={ATT} feat={feat.shape} outmax={[float(np.abs(o).max()) for o in outs]}")
