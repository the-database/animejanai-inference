# Exact FP16 host conversion for the image ABI

`aji_img_trt.cpp` uses array conversion functions for its FP16 host tensors.
Windows MSVC x64 builds select an eight-lane F16C implementation when CPUID
reports AVX, F16C and OSXSAVE and XGETBV confirms XMM/YMM state support.
Other CPUs and builds use the original scalar helpers in `src/core/half.h`.
The AVX implementation is a separate translation unit; CPU detection and the
scalar implementation are compiled without an AVX requirement.

This accelerates host staging in the still-image TensorRT path. FP32 and BF16
staging, model math, engine building, video inference and Vulkan settings are
unchanged. The shared source is also used by the optional TensorRT-RTX backend,
but that backend has not been built or exercised in this validation.

## Exact conversion semantics

Using the F16C intrinsics directly differed from the scalar helpers on NaNs.
The implementation now handles nonfinite lanes with the original helpers:
float-to-half keeps their signed canonical NaN, and half-to-float keeps their
NaN payload and signaling bit. Finite lanes use explicit round-to-nearest-even.
Unaligned arrays and short tails are supported.

F16C can raise floating-point conversion exceptions. Each vector conversion
call masks those exceptions and restores the complete saved MXCSR afterwards,
so the caller's masks, rounding controls and sticky flags remain unchanged.
The original scalar helpers use integer operations. The regression checks
these semantics, including unmasked exceptions and non-default controls.

## Validation and measured scope

On Windows with MSVC 19.51 and an i9-13900K, both the dispatched F16C build
and a scalar-only build passed:

- All 65,536 half encodings, including signed zero, subnormals and NaNs.
- 1,239,047 float patterns: half rounding boundaries and neighbours,
  special values and a deterministic random sample.
- Unaligned buffers, short arrays, scalar tails and unchanged MXCSR under
  four control settings.

The CMake Release test and benchmark targets built successfully. The complete
classic TensorRT backend also compiled against TensorRT 11.3 / CUDA 13.3.
Linux, other compilers, non-x86 machines and TensorRT-RTX were not tested.

The complete image ABI passed an original-versus-optimized comparison on the
i9-13900K / RTX 5090 using original Balanced 3.1, a shared static 1920x1080
tile engine and the same decoded RGB frame. Full-frame RGB, gray reflect-padding
and overlapping tiles all produced byte-identical output on every call: one
warmup and four timed calls per backend per case (30 image outputs total).

| Image case | Original median | Optimized median | Latency reduction |
| --- | ---: | ---: | ---: |
| 1920x1080 RGB -> 3840x2160 | 102.35 ms | 82.82 ms | 19.1% |
| 500x320 gray, reflect-padded to a 1920x1080 tile | 53.54 ms | 37.48 ms | 30.0% |
| 1984x1152 RGB, overlapping 1920x1080 tiles | 253.47 ms | 179.58 ms | 29.2% |

These are alternating-order `aji_img_upscale` call timings, including staging,
transfers, inference, tiling and blending, excluding loading, building, warmup
and output hashing. Both libraries reused the identical compatible engine;
this did not retest the unchanged engine builder. The gray test uses the same
large tile for parity coverage and does not represent default 512x512 tiling.
No other player or TensorRT builder ran during the comparison. These are
still-image results, not a video-playback FPS improvement.

Ten alternating-order samples after warmup, using the production conversion
functions on the i9-13900K, gave these sums of median input-pack/output-unpack
times:

| Input -> output | Scalar | Dispatched F16C | Conversion-only speedup |
| --- | ---: | ---: | ---: |
| 512x512 -> 1024x1024 | 4.831 ms | 0.823 ms | 5.87x |
| 1920x1080 -> 3840x2160 | 37.619 ms | 8.205 ms | 4.58x |

These measurements exclude TensorRT, transfers, tile blending, decoding and
rendering. Timings depend on CPU and current load. The nonfinite probe reports
zero pack and unpack differences with the integrated implementation.

## Reproduce

From a normally configured Windows MSVC x64 CMake build:

```bat
cmake --build build --config Release --target aji_half_test aji_half_benchmark
```

Run the resulting `aji_half_test.exe` and `aji_half_benchmark.exe` (under
`build/Release` for multi-configuration generators, or `build` for Ninja).
They do not execute GPU work. The benchmark is excluded from the default build.
Its JSON lines report CPU, nonfinite parity and separate pack/unpack medians.
On a machine without F16C, the dispatched measurements exercise scalar code.

A scalar-only comparison can also be built from the repository root in an
x64 Native Tools Command Prompt, without the F16C object or feature definition:

```bat
cl /nologo /O2 /EHsc /std:c++17 /I src sanity/half_array_test.cpp src/core/half_array.cpp /Fe:aji_half_scalar_test.exe
aji_half_scalar_test.exe
```

For the complete image ABI, `sanity/image_half.py` compares old and new backend
libraries using one shared engine cache and one RGB24 input frame. It checks
full-frame RGB, gray reflect-padding and overlapping tiles, including repeated
calls, and reports exact output hashes plus alternating-order API-call timings.
Warmup, loading and engine building are excluded from timing. Requires numpy,
compatible TensorRT libraries, a 2x RGB model and exclusive GPU access:

```text
python sanity/image_half.py --reference <old-aji_trt.dll> --candidate <new-aji_trt.dll> --runtime <runtime-dll-directory> --model <original-fp16.onnx> --cache <isolated-shared-cache> --trtexec <trtexec.exe> --rgb-frame <1920x1080.rgb> --output <new-results-directory>
```
