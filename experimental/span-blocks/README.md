# Experimental grouped Balanced convolutions and Ada tuning

This work builds on `experimental/span-concat` (PR #35). It preserves the later
4090 tuning and the small 5090 540p gain as reproducible source experiments.
It does **not** replace the ordinary profiles or select these engines by default.

The plugin groups each selected three-convolution block behind one TensorRT
plugin boundary, using CUTLASS FP16 NHWC convolutions with the original FP16
weights, bias and ReLU epilogues. Each convolution still launches its own kernel;
the grouping gives explicit control over the intermediate buffer/layout.

Two exports are supported:

- `block-concat.onnx`: all eight blocks plus the custom concat/projection.
  Selected for the 4090 1080p experiment.
- `native-tail7.onnx`: only the first seven blocks, with the eighth block and
  native concat/projection retained. Selected for the 5090 540p experiment.

The attempted all-eight-block **native** tail produced incorrect images because
of a tensor-view/physical-stride mismatch. It is intentionally not an export
option here. Neither the invalid engine's speed nor the 720p regression is a win.

## Reproduce the source build

Requirements: Windows x64, MSVC developer prompt, CUDA 13.x with matching CCCL,
TensorRT 11 headers, Python with ONNX/numpy, and CUTLASS headers. The measured
CUTLASS checkout was `e13f5cd4142c43efc37c2f6339923eb4a57afb99`. Use its existing
license when redistributing dependencies; the vendor source is not copied here.

```bat
python experimental\span-blocks\build.py --kind block-concat --model "C:\models\2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" --cuda "C:\CUDA" --trt-include "C:\TensorRT\include" --cutlass-include "C:\CUTLASS\include" --cccl-include "C:\CCCL\include" --output "C:\blocks-build"
python experimental\span-blocks\prepare.py --kind native-tail7 --model "C:\models\2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" --output "C:\native7-build"
```

The plugin binary supports both exports. The transform verifies the original
model's SHA-256 through the concat exporter, validates every replaced convolution,
and extracts the original weights locally. No weights, DLLs or engines are
committed. The build generates native SM89 and SM120 code and selects the tested
256x64x32, three-stage convolution schedule. `--allow-unsupported-compiler` is
available when explicitly needed by the local NVCC/MSVC combination.

Build the selected 4090 Balanced static engine:

```bat
python experimental\span-blocks\build-engine.py --preset balanced-block-4090-1080 --trtexec "C:\TensorRT\trtexec.exe" --onnx "C:\blocks-build\block-concat.onnx" --plugin "C:\blocks-build\native-block.dll" --engine "C:\blocks-build\balanced-block-1080.engine"
```

For the 5090 540p export, use `--preset balanced-native7-5090-540`, the
`native-tail7.onnx` file and the same plugin. For the later Performance cache
tuning, build Performance's concat model/plugin using `span-concat/build.py`,
then use `--preset performance-cache48-4090-1080` with that model and
`performance-plugin.dll`. This selects 48 MiB (50,331,648 bytes), not the earlier
64 MiB tiling budget. It does not change Performance's kernel or weights.

All presets use batch-one static engines. Run them on the target GPU, in an
isolated output directory, without concurrent player/benchmark work.

## Results and quality boundaries

Three alternating 4090 native-4K-rendered rounds, two-second warmup and
twelve-second measurements; values below are medians from that same final run:

| Model | Original | Previous optimized | New | New vs previous |
| --- | ---: | ---: | ---: | ---: |
| Balanced, 1080p | 90.407180 | 94.056981 | 95.687403 | +1.733% |
| Performance, 1080p | 125.397564 | 130.864512 | 131.921768 | +0.808% |

Every paired final round was positive, but these are small workload-specific
gains. `measurements-4090.json` preserves the actual samples and paired changes.
Three real-video 4K outputs for each new engine were byte-identical to the
**previous optimized** engine (six comparisons; `quality-4090.json`). That is not
a claim that the earlier optimized engine was identical to the stock model.

Separately, the 5090 540p native-tail7 experiment improved pipeline throughput
811.748 -> 837.473 fps (+3.17%) and native-rendered throughput
423.935 -> 428.317 fps (+1.03%). Its three RGB48 frames matched the **original**
model, unlike the reference used for the 4090 block-concat experiment.
The 720p grouped schedule regressed engine QPS 455.764 -> 444.863 (-2.39%).
The later interrupted 4090 lower-resolution screen did not establish a win.

Fresh source-packaging checks compiled both architectures, reproduced the
convolution/bias/ReLU/block parameter headers, and reproduced native-tail7 ONNX
byte-for-byte. The concat export keeps the two now-unused projection initializers
and uses the same custom domain as the other packaged SPAN export; learned
parameter bits are unchanged. Eight paired 5090 native-tail7 runtime outputs
matched the previous plugin binary exactly. This is not a fresh 4090 playback
benchmark or a newly searched engine.

## Remaining integration work

This is a draft research package. The block plugin registers the earlier concat
creator as well as 24 convolution and eight block creators. Load it in a fresh
process with the selected engine; do not load it alongside the separate Balanced
concat plugin, whose creator name overlaps. A direct-engine API caller must load
the DLL and call `initBalancedPlugin` before deserialization. `trtexec` does this
through `--dynamicPlugins`.

Before normal named-profile integration, give the block variants distinct
creator/model/cache identities and validate profile switching, clean-cache
building and native playback on the target GPUs. Do not rename these files over
the stock model or the earlier concat model. This PR deliberately leaves those
default routes untouched while exposing the tested kernel source and presets.
