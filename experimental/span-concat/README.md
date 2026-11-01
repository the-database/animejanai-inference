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
Named experimental profiles automatically use their corresponding regular model
on those backends/architectures. Keep the original ONNX files installed.

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

The same bundle can be built and installed with CMake (Python must have `onnx`
and `numpy`): add `-DAJI_BUILD_SPAN_PLUGINS=ON -DAJI_SPAN_MODEL_DIR=C:/models`
to the normal TensorRT 11 configure command, build `aji_trt`, then run
`cmake --install <build> --component experimental-span --prefix <test>/animejanai`.
This stages the backend, both plugin DLLs, both exports and both stock models.
The Windows release script accepts `AJI_SPAN_MODEL_DIR` and stages that optional
component in `build-win-release/span-package/animejanai`. Normal releases keep
the option off; no model files are downloaded or added to source control.

The backend registers the matching plugin before engine loading and supplies
`--dynamicPlugins` to the ordinary asynchronous trtexec builder. Original models
continue to use the global/Manager builder settings and work without either
plugin DLL. Transformed model names and the plugin binary's content fingerprint
give the engine cache a separate identity. Rebuilding/replacing a DLL selects a
new cache; restart the player to load that binary. Loaded modules remain resident
even after failed registration because TensorRT can retain creator pointers.
Direct engine-file mode is a low-level API: callers must register the plugin
themselves; the automatic registration here is for named configuration models.

Supported batch-one static inputs:
480x360, 640x480, 768x576, 960x540, 1280x720 and 1920x1080. Unsupported shapes,
dynamic profiles, missing DLLs/exports and failed optional engine builds log a
reason and fall back to the matching stock model, in blocking and asynchronous
build modes. A failed optional build is not retried on every profile switch.
Stock builds use the user's unchanged builder settings. Experimental builds also
preserve those settings, adding tiling level 3 for Balanced (or Performance at
1080p) and a 64 MiB L2 budget at 1080p only when the user has not set that option.
The accompanying profiles use builder optimization level 5 and static optShapes.
These are the tested shared presets,
not a claim that the same cache budget is optimal for every GPU.

## Measurement and quality scope

The combined recipe includes both TensorRT tiling settings and the concat plugin.
Historical stock-to-concat gains below do not isolate the plugin's contribution.
The review's 5090 control found stock + tiling already reduced engine time by
about 5–6.5%; concat then reduced it by another 6–7%. Those are device-latency
reductions, not playback FPS percentages. Do not enable tiling globally from
those results alone: gains depend on GPU and resolution.

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

The review follow-up builds the complete optional CMake install component from
the original ONNX files. Focused runtime regressions on a 5090 pass for missing
DLL/export, unsupported 1918x1078 input, failed plugin registration and failed
optional engine builds, including blocking/asynchronous operation and reverse
profile switching. Fallback outputs match stock byte-for-byte. Explicit custom
builder flags survive; a changed plugin binary gets a distinct engine cache;
a failed registration leaves its DLL loaded and is attempted only once.
The SM and option policy also has a CPU regression. The non-classic backends'
fallback routes are source changes; their execution was not validated locally.

The kernel bodies are the versions exercised in the 4090/5090 experiments.
The source packaging and named-model builder integration were compiled with
TensorRT 11.3/CUDA 13.3. All four static 1080p engines were built from empty caches
on both a 4090 and a 5090. Both optimized models were also built at the remaining
five supported sizes on the 5090. Original and experimental profiles coexist in one context, and reverse
profile switching checks that the first frame remains byte-identical to its
earlier output within each profile. This is not equality between different models.
The corrected upload check produced 48 frames per GPU (12 per profile). Worst
sample PSNR against stock was 72.431 dB Balanced / 70.719 dB Performance on the
5090, and 77.615 / 70.025 dB on the 4090. These are direct YUV444P16 comparisons
with peak 65535, not the historical rendered RGB48 measurements above.
`validation.json` records the integration results and individual playback timings.

`sanity/named_profiles.py` reproduces the direct API build/cache/switch checks.
It requires Windows, NumPy, the built backend/runtime DLLs and raw NV12 clips.
For example, from the repository root, with an isolated directory containing the
original and generated models:

```bat
python sanity\named_profiles.py --backend "C:\test\aji_trt.dll" --runtime "C:\test" --conf experimental\span-concat\profiles.conf --models "C:\test\models" --frames "C:\clips\1080p.nv12" --output "C:\checks\span" --clean-cache
```

The test uploads frames on the inference stream. An earlier default-stream
upload raced the nonblocking inference stream on the 4090; that harness error
reproduced with both the old and new backend and disappeared with ordered uploads.

Fresh 5090 native 3840x2160 playback completed for all four profiles before
playback testing was paused. Vulkan queue count remained three, output was 2x,
and the checks recorded no decoder/render drops. Timing was not stable across
rounds (Balanced concat: 110.14 then 79.63 fps), so these runs establish playback
integration, not a new speedup. Historical gains above remain separate evidence.

The completed 4090 check used the bundled 1920x1080 benchmark MP4, driver
610.88, and the static engines from the clean-cache integration checks. Two
forward/reverse rounds used 10 seconds warmup and 20 seconds measurement per
profile. The 3840x2160 model output rendered to the native 3840x2160, 60 Hz
display through NVDEC/gpu-next with three Vulkan queues and no dropped frames
or concurrent mpv sessions. FPS is uncapped native `end video-flip` throughput
(`--untimed=yes`, immediate Vulkan presentation), not the panel refresh rate.

| Profile | Median FPS | Versus corresponding Original |
| --- | ---: | ---: |
| Original Balanced | 84.425 | reference |
| Balanced concat | 87.011 | +3.063% |
| Original Performance | 113.985 | reference |
| Performance concat | 118.219 | +3.714% |

Both paired rounds were positive: Balanced +1.927%/+4.209%, Performance
+3.683%/+3.746%. These are two-round, workload-specific results; broader
content/driver coverage remains outstanding. Output differences documented
above still apply. The additional grouped-convolution result is in #37.

A completed 5090 540p check used a looped 24-frame real-video fixture, 10 seconds
warmup and 20 seconds measurement in each of two reversed-order rounds. The 2x
model output was 1920x1080, rendered to the native 3840x2160 display with NVDEC,
three Vulkan queues and no dropped frames. Median Original/concat throughput
was 417.424/417.123 fps (-0.072%); paired changes had opposite signs. This is a
tie within observed variation. The separate NativeTail7 result is in #37.

These paths remain explicitly opt-in.
The later Ada grouped-convolution and 48 MiB Performance-cache experiments are
separate changes; this directory does not silently enable them.
