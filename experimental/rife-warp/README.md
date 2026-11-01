# Experimental RIFE 4.22 warp fusion

Four sampling stages in the tested FP16 RIFE export cast feature maps to FP32,
perform GridSample, then cast back. `AJRifeWarp` reads FP16 features directly,
keeps grids and interpolation arithmetic in FP32, and writes FP16. The learned
weights and the surrounding graph are unchanged. Explicit FMA/rounding choices
mean the result is **not bit-identical to stock TensorRT GridSample**.

This is an opt-in experiment, not a default model replacement. Select it per
profile with `chain_1_rife_fp16_warp=yes` (default: `no`).
The accompanying `profiles.conf` provides Original and Warp profiles, both with
RIFE 2x before original Balanced. They can switch in one player process.
`AJI_RIFE_WARP=1` remains a whole-process override set before launch; leave it
unset when comparing the two profiles.
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
generated. The historical performance evaluation below was on the 5090; the
current integration also has clean-build and switching checks on a 4090.

Build this branch's `aji_trt` normally. In an isolated test install, put
`rife-warp-plugin.dll` beside `aji_trt.dll` and the generated
`rife_v4.22_fp16_warp_v1.onnx` beside the original RIFE ONNX. Use the supplied
configuration in that isolated install, or add `chain_1_rife_fp16_warp=yes` to
a non-ensemble RIFE 4.22 profile. Normal profiles omit the key or set it to `no`.
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
Those historical measurements do not establish a 4090 speedup.

The original 76-call sequence produced 73 outputs with aggregate direct NV12
PSNR 70.055862 dB and worst-frame PSNR 65.887006 dB versus the stock RIFE output.
Scene decisions matched and reset checks were stable. One generated frame after
Balanced at 4K measured 68.261066 dB RGB48 PSNR, SSIM 0.999932713. This is finite
content coverage, not a guarantee of visual equivalence.

Fresh current-upstream packaging validation compiled the complete backend and
plugin and matched all 75 applicable calls against the previously saved warp
results: 73 output hashes, two built-in scene-cut decisions, multiple timesteps,
and reconfigure/reset. The older explicit external scene-override test is not
available in this upstream API and is excluded. Those initial port checks used
cached engines; the named-profile integration was then tested separately below.

With the experiment disabled, a separate fresh 75-call stock-RIFE run also
matched all saved stock output hashes, scene decisions and reset results.

The integrated per-profile version built separate stock/warp engines from empty
caches on both the 4090 and 5090. The API sequence switches stock -> warp -> stock
within one context, checks matching scene decisions and exact stock reloads, and
compares real adjacent frames and quarter/half/three-quarter timesteps. Selection
does not rely on changing a process environment variable after loading the DLL.
The corrected stream-ordered test makes 149 calls per GPU, including 71 generated
warp frames. Aggregate/worst direct NV12 PSNR is 70.055767 / 65.895593 dB on the
5090 and 70.758267 / 67.055773 dB on the 4090. Scene decisions match and stock
reloads are exact. `validation.json` records these results and playback samples.

`sanity/rife_profiles.py` reproduces the build/load, separate-cache, sequence and
switch checks on Windows with NumPy and the backend's CUDA 13 runtime. Supply an
isolated directory containing both RIFE ONNX exports, plus one or more 24-frame
raw 1920x1080 NV12 clips:

```bat
python sanity\rife_profiles.py --backend "C:\test\aji_trt.dll" --runtime "C:\test" --models "C:\test\rife" --frames "C:\clips\1080p.nv12" --output "C:\checks\rife" --clean-cache
```

Fresh 5090 native 3840x2160 playback completed for both profiles with RIFE 2x
before original Balanced, three Vulkan queues and no recorded drops. The two
stock timings were 79.58 and 49.27 fps; warp was 79.51 and 79.11 fps. That unstable
control prevents a meaningful new speedup claim. Playback testing is currently
paused; new 4090 native-display timing and wider content/driver coverage remain
outstanding. DirectML, TensorRT-RTX and Linux execution were not tested.

Use identical source frames, renderer, interpolation phase, warmup and display
dimensions when comparing. Keep Vulkan queues at three and reject overlapping
player/GPU work. Scene cut frames and original/interpolated frame phases must
not be mixed in quality comparisons.
