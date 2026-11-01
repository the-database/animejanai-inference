# Experimental RIFE 4.22 warp fusion

Four sampling stages in the tested FP16 RIFE export cast feature maps to FP32,
perform GridSample, then cast back. `AJRifeWarp` reads FP16 features directly,
keeps grids and interpolation arithmetic in FP32, and writes FP16. The learned
weights and the surrounding graph are unchanged. Explicit FMA/rounding choices
mean the result is **not bit-identical to stock TensorRT GridSample**.

This is an opt-in experiment, not a default model replacement. The backend only
selects it when `AJI_RIFE_WARP=1`; unset the variable to use ordinary RIFE again.
Selection requires non-ensemble RIFE 4.22 with 1920x1080 input, Windows and
classic TensorRT 11. For 4K upscaling, set RIFE **before** the spatial model.
Other shapes, models and backends are not enabled by this experiment.

## Reproduce

Use Python with ONNX, an x64 MSVC developer prompt, CUDA 13.x, its matching CCCL
headers and TensorRT 11 headers:

```bat
python experimental\rife-warp\prepare.py --model "C:\models\rife_v4.22.onnx" --output "C:\rife-warp-build\rife"
python experimental\rife-warp\build.py --cuda "C:\CUDA" --trt-include "C:\TensorRT\include" --cccl-include "C:\CCCL\include" --output "C:\rife-warp-build"
```

The transform verifies the exact tested source SHA-256, checks all four sampling
patterns, preserves initializer bytes and runs the ONNX checker. Build from a
CUDA-supported compiler, or explicitly pass `--allow-unsupported-compiler` for
a local compiler needing NVCC's override. Keep `--fmad=false`; the kernel's
explicit `fmaf` operations still fuse as written. Native SM89 and SM120 code is
generated, but the historical RIFE evaluation was on the 5090 only.

Build this branch's `aji_trt` normally. In an isolated test install, put
`rife-warp-plugin.dll` beside `aji_trt.dll` and the generated
`rife_v4.22_fp16_warp_v1.onnx` beside the original RIFE ONNX. Set
`AJI_RIFE_WARP=1` before launching the player with non-ensemble RIFE 4.22 enabled.
The named-model path loads the plugin before deserialization and supplies it to
trtexec for a new static 1x11x1088x1920 engine. The distinct ONNX/cache name
prevents stock/experimental cache collisions. Do not rename this export over
the original ONNX or reuse the old same-name engine cache from early prototypes.

The normal builder runs once when the new cache entry is absent. Its default
settings are preserved; optimization-level-5 RIFE was slower in the experiment
and is not enabled. Spatial-model settings are unaffected.

## Evidence and limits

Historical 5090 native-4K playback with RIFE 2x before optimized Balanced:

| Experiment | Baseline mean | Warp mean | Change |
| --- | ---: | ---: | ---: |
| Longer A-B-B-A, 10 s warmup / 20 s measurement | 114.250441 fps | 117.371735 fps | +2.732% |
| Earlier valid shorter-warmup comparison | 114.199713 fps | 111.318457 fps | -2.523% |

Do not discard the negative result. The longer run is promising, not proof of a
universal win. The experimental runtime in those measurements also contained
other already-held-constant optimizations; this port does not enable them.
There is no measured RIFE gain on a 4090 in this work.

The original 76-call sequence produced 73 outputs with aggregate direct NV12
PSNR 70.055862 dB and worst-frame PSNR 65.887006 dB versus the stock RIFE output.
Scene decisions matched and reset checks were stable. One generated frame after
Balanced at 4K measured 68.261066 dB RGB48 PSNR, SSIM 0.999932713. This is finite
content coverage, not a guarantee of visual equivalence.

Fresh current-upstream packaging validation compiled the complete backend and
plugin and matched all 75 applicable calls against the previously saved warp
results: 73 output hashes, two built-in scene-cut decisions, multiple timesteps,
and reconfigure/reset. The older explicit external scene-override test is not
available in this upstream API and is excluded. Cached engines were used for
these runtime checks; a new full engine search and native-playback measurement
of this port remain necessary before promotion.

With the experiment disabled, a separate fresh 75-call stock-RIFE run also
matched all saved stock output hashes, scene decisions and reset results.

Use identical source frames, renderer, interpolation phase, warmup and display
dimensions when comparing. Keep Vulkan queues at three and reject overlapping
player/GPU work. Scene cut frames and original/interpolated frame phases must
not be mixed in quality comparisons.
