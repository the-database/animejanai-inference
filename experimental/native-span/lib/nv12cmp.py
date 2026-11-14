import numpy as np, sys
a, b, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fs = w * h * 3 // 2
A = np.fromfile(a, np.uint8, count=fs).astype(np.int32); B = np.fromfile(b, np.uint8, count=fs).astype(np.int32)
for name, sl in (("Y", slice(0, w * h)), ("UV", slice(w * h, fs))):
    d = A[sl] - B[sl]; mse = np.mean(d.astype(np.float64) ** 2)
    print(f"{name}: PSNR {10*np.log10(255**2/mse) if mse else 99:.2f} dB, max |diff| {np.abs(d).max()}, >1 level {np.mean(np.abs(d)>1)*100:.4f}%")
