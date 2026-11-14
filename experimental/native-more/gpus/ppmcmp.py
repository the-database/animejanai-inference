# PSNR and largest difference between two PPM/PGM images (TensorRT vs native output).
import sys

import numpy as np


def rd(p):
    b = open(p, "rb").read()
    parts = b.split(maxsplit=4)
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(parts[4], np.uint8).reshape(h, w, -1 if parts[0] == b"P6" else 1).astype(np.float64)


try:
    a, b = rd(sys.argv[1]), rd(sys.argv[2])
except Exception as e:  # noqa: BLE001
    print("parity: n/a (%s)" % e)
    sys.exit(0)
if a.shape != b.shape:
    print("parity: SHAPE", a.shape, b.shape)
    sys.exit(0)
d = np.abs(a - b)
mse = np.mean(d ** 2)
print("parity %dx%d: PSNR %.1f dB, max %d" % (a.shape[1], a.shape[0], 10 * np.log10(255 ** 2 / mse) if mse else 99, d.max()))
