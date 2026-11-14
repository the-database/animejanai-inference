import numpy as np, sys
d = sys.argv[1]; N = 540 * 960 * 48
for tag in ("acc16", "acc32"):
    for name in ("outb1", "out5", "mid5"):
        p = np.fromfile(f"{d}/proto_{tag}_{name}.bin", np.float16).astype(np.float32)
        r32 = np.fromfile(f"{d}/ref_fp32_{name}.bin", np.float32)
        r16 = np.fromfile(f"{d}/ref_cuda16_{name}.bin", np.float16).astype(np.float32)
        rng = np.abs(r32).max()
        e = lambda a: (np.abs(a - r32).max(), np.sqrt(np.mean((a - r32) ** 2)))
        print(f"{tag} {name:6s} |ref| max {rng:7.2f}  proto vs fp32: max {e(p)[0]:.4f} rms {e(p)[1]:.5f}   ORT-fp16 vs fp32: max {e(r16)[0]:.4f} rms {e(r16)[1]:.5f}")
