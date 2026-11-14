# TFDAT v2 parity at any scale and clip length: T frames (raw rgb24, FW x FH) cropped to W x H at
# (X, Y), through the native runner (fp16 model) and ONNX Runtime CPU on the fp32 export.
# usage: ref.py RUNNER MODEL_FP16 MODEL_FP32 FRAMES.rgb FW FH X Y W H T WORKDIR [ITERS]
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

runner, m16, m32, raw = sys.argv[1:5]
FW, FH, X, Y, W, H, T = map(int, sys.argv[5:12])
wd = sys.argv[12]
iters = sys.argv[13] if len(sys.argv) > 13 else "3"
fr = np.fromfile(raw, np.uint8).reshape(-1, FH, FW, 3)
fr = fr[(len(fr) - T) // 2:(len(fr) - T) // 2 + T, Y:Y + H, X:X + W].astype(np.float32) / 255
x = np.ascontiguousarray(fr.transpose(0, 3, 1, 2).reshape(1, 3 * T, H, W)).astype(np.float16)
inp = os.path.join(wd, "v2_in.f16")
x.tofile(inp)
out = os.path.join(wd, "v2_out.f16")
r = subprocess.run([runner, m16, inp, str(W), str(H), out, iters], capture_output=True, text=True)
print((r.stdout.strip() or r.stderr.strip())[-500:])
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(m32, so, providers=["CPUExecutionProvider"])
ref = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0][0]
if r.returncode == 0 and os.path.exists(out):
    nat = np.fromfile(out, np.float16).astype(np.float32)
    if nat.size != ref.size:
        print("SIZE native %d vs ref %s" % (nat.size, ref.shape))
        sys.exit(1)
    nat = nat.reshape(ref.shape)
    a = np.clip(nat, 0, 1) * 255
    b = np.clip(ref, 0, 1) * 255
    d = np.abs(a - b)
    mse = np.mean(d ** 2)
    print("native vs fp32 %s: PSNR %.1f dB, max %.1f, >2 levels %.4f%%" % (
        "x".join(map(str, ref.shape)), 10 * np.log10(255 ** 2 / mse) if mse else 99, d.max(), 100 * np.mean(d > 2)))
