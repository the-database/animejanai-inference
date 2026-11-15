import numpy as np, sys
d = sys.argv[1]
a = np.fromfile(f"{d}/native_out.bin", np.float16).astype(np.float32)
r = np.fromfile(f"{d}/ref_out.bin", np.float16).astype(np.float32)
q = lambda x: np.clip(np.round(x * 255), 0, 255)
d8 = np.abs(q(a) - q(r)); mse = np.mean(d8 ** 2)
print(f"native vs ORT-CUDA fp16: PSNR {10*np.log10(255**2/mse) if mse else 99:.2f} dB, max |diff| {int(d8.max())} levels, "
      f"mean {d8.mean():.4f}, >1 level {np.mean(d8>1)*100:.4f}%  (float max abs {np.abs(a-r).max():.4f}, nan {np.isnan(a).sum()})")
