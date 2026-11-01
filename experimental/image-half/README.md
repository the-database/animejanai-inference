# Image-ABI FP16 host-conversion experiment

`aji_img_trt.cpp` currently packs and unpacks host tensors with the scalar
functions in `src/core/half.h`. This standalone Windows/MSVC experiment measures
eight-lane F16C conversion against those exact helpers. It does **not** change
the library, model math, video playback, CPU requirements of the library, or
ordinary engine building.

## Why preserve this prototype

On an i9-13900K, ten alternating-order samples after warmup measured the
following combined input-pack and 2x-output-unpack times:

| Input -> output | Scalar | F16C | Conversion-only speedup |
| --- | ---: | ---: | ---: |
| 512x512 -> 1024x1024 | 4.651 ms | 0.411 ms | 11.3x |
| 1920x1080 -> 3840x2160 | 37.032 ms | 6.942 ms | 5.33x |

These historical numbers exclude TensorRT, CPU/GPU transfers, tile blending
and decoding/rendering. They are not an image-inference or video-FPS gain.
Timing from a fresh run depends on the machine and current CPU load.

The test checks every finite FP16 bit pattern, including subnormals and signed
zeros, plus sampled floats in the image range [-0.1, 1.1]. It reports median
pack/unpack times separately and fails if those tested values differ.

## Known blocker to integrating the intrinsic directly

F16C conversion and the scalar helpers do not have identical NaN behavior:
the scalar float-to-half helper canonicalizes NaNs, and the scalar half-to-float
helper preserves signaling bits; the instruction's behavior differs. The
prototype now reports bit differences for all 2,048 nonfinite half encodings
separately. These differences are expected findings, not a passing parity
claim. It does not establish floating-point exception/MXCSR equivalence.

A production implementation would need deliberate handling of those semantics,
runtime CPU/OS feature dispatch, an unchanged path for other CPUs, and a real
image-ABI end-to-end comparison. None of that is silently enabled here. FP32
and BF16 staging are outside this experiment. The prototype should remain
research until those integration requirements are addressed.

## Reproduce

Use an x64 Native Tools Command Prompt for Visual Studio on an AVX2/F16C CPU
with OS AVX state support. From this directory:

```bat
cl /nologo /O2 /EHsc /std:c++17 /MT /arch:AVX2 half-profile.cpp /Fe:half-profile.exe
half-profile.exe
```

The standalone binary requires AVX2; it is not a portable library dispatcher.
It checks the expected CPU/OS features and returns 2 if unavailable, but should
only be run on the stated target hardware. Codes 3-6 identify finite/sample
parity failures. JSON lines on stdout contain the CPU, nonfinite findings and
conversion-only timings. No GPU, model weights or engine is required. Linux
and non-MSVC builds of this Windows prototype have not been validated.

A fresh packaging run on the same CPU passed the finite/image-range checks.
It measured combined conversion totals of 4.6845 -> 0.45045 ms (512 square)
and 37.2718 -> 6.9774 ms (1080p). The nonfinite probe recorded 2,044 pack and
1,022 unpack bit differences, confirming why this remains a prototype.
