# Random-weight TFDAT v2 exports from the dis2-tfdat2 source, named as traiNNer's ONNX wrapper does
# (/model/...), 4D input 1 x (T*3) x H x W, fp32 and fp16.
# usage: export.py SCALE CLIP OUTDIR [embed_dim]
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub"))
from traiNNer.archs.tfdat_arch import tfdat  # noqa: E402

scale, clip, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
ed = int(sys.argv[4]) if len(sys.argv) > 4 else 120
torch.manual_seed(1000 * scale + clip)
net = tfdat(scale=scale, clip_size=clip, embed_dim=ed).eval()
# Trained-looking random weights; the zero-initialised heads (flow, align, conf, hr_refine) get
# non-zero values too, so every branch contributes.
with torch.no_grad():
    for n, p in net.named_parameters():
        if p.dim() > 1:
            p.normal_(0, 0.6 / p[0].numel() ** 0.5)
            if "flow_net.dec" in n and ".4." in n:
                p.mul_(4)  # flow heads: a few pixels of motion
        elif n.endswith("temp"):
            p.uniform_(0.5, 1.5)
        elif (".n1." in n or ".n2." in n) and n.endswith("weight"):
            p.normal_(1, 0.1)
        else:
            p.normal_(0, 0.02)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x):
        return self.model(x)


w = Wrap(net).eval()
H, W = 64, 96
x = torch.rand(1, clip * 3, H, W)
with torch.no_grad():
    y = w(x)
print("scale", scale, "clip", clip, "out", tuple(y.shape), "range %.3f %.3f" % (y.min(), y.max()))
name = os.path.join(out, "%dx_tfdat2_rand_c%d" % (scale, clip))
dyn = {"input": {2: "H", 3: "W"}, "output": {2: "H2", 3: "W2"}}
torch.onnx.export(w, x, name + "_fp32.onnx", input_names=["input"], output_names=["output"],
                  opset_version=20, dynamic_axes=dyn, dynamo=False)
w.half()
torch.onnx.export(w, x.half(), name + "_fp16.onnx", input_names=["input"], output_names=["output"],
                  opset_version=20, dynamic_axes=dyn, dynamo=False)
print("wrote", name)
