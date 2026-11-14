import numpy as np, sys
# multi-frame NV12 / P010 comparison: worst-frame PSNR and largest difference, Y and UV, in levels
a, b, w, h, fmt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dt, shift, peak = (np.uint8, 0, 255) if fmt == "nv12" else (np.uint16, 6, 1023)
fs = w * h * 3 // 2
A = np.fromfile(a, dt).astype(np.int32) >> shift; B = np.fromfile(b, dt).astype(np.int32) >> shift
n = min(len(A), len(B)) // fs
if n == 0 or len(A) != len(B): print(f"SIZE MISMATCH {len(A)} vs {len(B)}"); sys.exit(1)
res = {}
for k in range(n):
    for name, sl in (("Y", slice(k * fs, k * fs + w * h)), ("UV", slice(k * fs + w * h, (k + 1) * fs))):
        d = A[sl] - B[sl]; mse = np.mean(d.astype(np.float64) ** 2)
        p = 10 * np.log10(peak ** 2 / mse) if mse else 99.0
        r = res.setdefault(name, [99.0, 0]); r[0] = min(r[0], p); r[1] = max(r[1], int(np.abs(d).max()))
print(f"{n} frames; " + "; ".join(f"{k} min PSNR {v[0]:.1f} dB max {v[1]}" for k, v in res.items()))
