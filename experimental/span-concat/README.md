# Experimental SPAN concat/projection fusion

This optional TensorRT 11 path fuses the final four-way feature concatenation
and 1x1 projection in the **V3.1 non-sharp Balanced and Performance FP16 models**.
It keeps the original learned weights, removes the intermediate concatenation,
and uses FP16 tensor-core accumulation. It is not pixel-identical to TensorRT's
stock tactics. It does not replace the default profiles or quantize the models.

## Build and try

Currently supported: Windows x64, TensorRT 11, CUDA 13.x, RTX 4090 (SM89) and
5090 (SM120). The shipped kernel source produces native code for both GPUs.
TensorRT-RTX, DirectML, Linux and other architectures are outside this experiment.
Use the regular models on those backends.

From an x64 MSVC developer prompt with Python, `onnx` and `numpy` installed:

```bat
python experimental\span-concat\build.py --kind balanced --model "C:\models\2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" --cuda "C:\CUDA" --trt-include "C:\TensorRT\include" --output "C:\span-build\balanced"
python experimental\span-concat\build.py --kind performance --model "C:\models\2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx" --cuda "C:\CUDA" --trt-include "C:\TensorRT\include" --output "C:\span-build\performance"
```

Use a CUDA-supported compiler; `--allow-unsupported-compiler` is available for
local toolchains needing NVCC's override. The scripts verify the exact source
model SHA-256 before generating the fixed-weight plugin. No weights, engines,
fatbins or DLLs are checked into the repository.

Build `aji_trt` normally with TensorRT 11. Copy the generated
`balanced-next-plugin.dll` and `performance-plugin.dll` beside `aji_trt.dll` in
an **isolated test installation**. Copy the generated ONNX files into its ONNX
directory alongside the original models. Use the accompanying `profiles.conf`
as a separate test configuration, preserving your regular configuration.

The backend registers the matching plugin before engine loading and supplies
`--dynamicPlugins` to the ordinary asynchronous trtexec builder. Original models
continue to use the global/Manager builder settings and work without either
plugin DLL. Transformed model names give the engine cache a separate identity.
Direct engine-file mode is a low-level API: callers must register the plugin
themselves; the automatic registration here is for named configuration models.

Supported batch-one static inputs:
480x360, 640x480, 768x576, 960x540, 1280x720 and 1920x1080. Other shapes fail
explicitly; switch to an Original profile for those. Engines use optimization
level 5, `--optShapes=input:1x3xHxW`, and skip inference during building. Balanced
uses tiling level 3 at every supported shape; Performance uses it at 1080p.
Both use a 64 MiB tiling L2 budget at 1080p. These are the tested shared presets,
not a claim that the same cache budget is optimal for every GPU.

## Measurement and quality scope

Historical controlled 5090 native-4K-rendered Balanced throughput improved
135.359 -> 157.052 fps with the tuned concat engine (+16.02%). Separate direct
buffer graph replay raised that to 158.228 fps; that additional backend change
is independent of this PR. Performance with direct graphs on both sides improved
225.36 -> 256.85 fps (+13.98%) at 1080p. These are different experiments, not a
single combined cross-model benchmark.

The five-resolution 4090 comparison showed much smaller 1080p gains: Balanced
+3.42%, Performance +5.50%. Some lower-resolution cases tied or regressed.
Do not extrapolate the 5090 gains to all GPUs or all input sizes.

Three 5090 4K rendered frames for tuned Balanced versus stock measured
73.389374, 73.975172 and 77.638722 dB PSNR. Performance concat versus stock was
66.028–66.515 dB at 1080p, 66.373–67.095 dB at 720p and 66.826–67.722 dB at 540p
on three frames per shape. The weights are unchanged, but accumulation/tactic
choices change rounding. Finite-frame PSNR is not a proof of universal visual
equivalence. Review representative content before opting in.

Use the native-display playback benchmark, report its display dimensions, and
run A/B in isolation with identical input, renderer, graph mode and warmup.
Keep Vulkan queue count at three. Engine-only timings cannot substitute for
playback FPS. The five public benchmark input sizes exclude 540p; 540p remains
an additional supported development shape.

## Review status

The kernel bodies are the versions exercised in the 4090/5090 experiments.
This source packaging and current-upstream builder integration are new. They
remain opt-in pending broader install/build, driver and content validation.
The later Ada grouped-convolution and 48 MiB Performance-cache experiments are
separate changes; this directory does not silently enable them.
