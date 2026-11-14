import numpy as np, sys
d = sys.argv[1]; out = []
for nm in ("outb1", "outlast", "midlast"):
    p = np.fromfile(f"{d}/proto_{nm}.bin", np.float16).astype(np.float32)
    r = np.fromfile(f"{d}/ref_{nm}.bin", np.float32)
    rng = float(np.abs(r).max()) or 1.0
    out.append(f"{nm} max err {np.abs(p - r).max() / rng * 100:.3f}% of range")
print(" | ".join(out))
