# Experimental grouped Balanced convolutions and Ada tuning

This work builds on `experimental/span-concat` (PR #35). It preserves the later
4090 tuning and the small 5090 540p gain as reproducible source experiments.
Its `profiles.conf` keeps the four original/concat profiles and adds slots 5-7
for block-concat, NativeTail7 and Performance Cache48. The ordinary models remain
available and slot 1 (Original Balanced) stays the default.

The plugin groups each selected three-convolution block behind one TensorRT
plugin boundary, using CUTLASS FP16 NHWC convolutions with the original FP16
weights, bias and ReLU epilogues. Each convolution still launches its own kernel;
the grouping gives explicit control over the intermediate buffer/layout.

Two exports are supported:

- `2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_Concat_v1.onnx`: all eight blocks plus the custom concat/projection.
  Selected for the 4090 1080p experiment.
- `2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_NativeTail7_v1.onnx`: only the first seven blocks, with the eighth block and
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
python experimental\span-blocks\build-engine.py --preset balanced-block-4090-1080 --trtexec "C:\TensorRT\trtexec.exe" --onnx "C:\blocks-build\2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_Concat_v1.onnx" --plugin "C:\blocks-build\balanced-blocks-plugin.dll" --engine "C:\blocks-build\balanced-block-1080.engine"
```

For the 5090 540p export, use `--preset balanced-native7-5090-540`, the
`2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_NativeTail7_v1.onnx` file and the same plugin. For the later Performance cache
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

Initial source-packaging checks compiled both architectures and reproduced the
convolution/bias/ReLU/block parameter headers. Eight paired 5090 native-tail7
runtime outputs matched the previous plugin binary exactly. The subsequent
named-profile integration changes ONNX plugin namespace attributes and model
filenames; these exports intentionally differ from the historical ONNX bytes.
Learned parameter bits remain unchanged. Historical equality refers to the
specific engines above, not every new TensorRT tactic search.

## Named-profile integration

Copy `balanced-blocks-plugin.dll` beside the built `aji_trt.dll` and the normal
TensorRT/CUDA runtime DLLs in an isolated installation. Put the generated ONNX
exports beside the original/concat models. Both block exports use the dedicated
`aji.span.blocks.v1` TensorRT namespace. Their creators coexist with the standalone
concat plugin, so profile changes no longer require a new process.

The backend registers the correct DLL before deserializing an engine and passes
it to the asynchronous trtexec builder. Distinct model basenames and builder
settings produce distinct cache keys. Slot 5 selects block-concat at 1080p with
optimization level 5 and zero auxiliary streams. Slot 6 selects NativeTail7 at
960x540 with tiling level 3. Other sizes fail explicitly for these two exports.

To create the separately named 1080p Performance Cache48 model, run:

```bat
python experimental\span-blocks\prepare.py --kind performance-cache48 --model "C:\models\2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx" --output "C:\cache48-build"
```

Copy `2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_Cache48_v1.onnx` into the model
directory and retain `performance-plugin.dll` from the concat build. Slot 7 uses
that same creator with its own model/cache identity and 48 MiB tiling budget.
The ordinary Performance and earlier 64 MiB concat profiles are unchanged.
The supplied profiles use static engines; no automatic GPU-specific replacement
of the user's regular model is performed.

The direct-engine API still requires the caller to register the DLL before
loading an engine; its entry point is now `initBalancedBlocksPlugin`. Historical
unnamespaced experimental block engines must be rebuilt from the new exports.

## Integration validation

The complete classic backend and namespaced plugin compile with TensorRT 11.3,
CUDA 13.3 and native SM89/SM120 code. New block-concat/Cache48 engines built
successfully on both the 4090 and 5090 at 1080p; NativeTail7 built on the 5090 at
540p. The six 1080p profiles load in one context, including both standalone
concat plugins and the namespaced block plugin. Each GPU produced 72 frames and
passed exact reverse-profile switching. A separate 16-frame 540p sequence checks
Original, concat and NativeTail7 on the 5090.

Fresh engine searches do not reproduce every historical equality result:
block-concat versus prior concat has worst sampled PSNR 75.070 dB on the 5090 and
70.034 dB on the 4090. Cache48 versus prior Performance concat is 70.724 dB on the
5090 and identical for all 12 sampled 4090 frames. NativeTail7 versus Original
is 69.156 dB on the four sampled 5090 frames, rather than bit-identical. These
are direct YUV444P16 measurements with peak 65535, not rendered RGB48. See
`validation.json`; the historical speed and equality records above remain
specific to their saved engines.

Use `sanity/named_profiles.py` from #35 to repeat the direct API checks, selecting
`--slots 1,2,3,4,5,7` at 1080p or `--slots 1,2,4,6 --width 960 --height 540` with
a raw 540p NV12 clip. A new output directory records every frame and reverse
switching check. Different model basenames and settings isolate each engine.

Fresh 5090 native 3840x2160 playback completed for block-concat and Cache48 with
three Vulkan queues and no recorded drops. The controls varied considerably
across the 1080p suite; it does not establish a new 5090 speedup.

The completed 4090 check used driver 610.88, the bundled 1920x1080 benchmark
MP4 and the static engines from the clean-cache integration checks. Two
forward/reverse rounds used 10 seconds warmup and 20 seconds measurement per
profile. Model output and native display were both 3840x2160 (60 Hz), with
NVDEC/gpu-next, three Vulkan queues and no dropped frames or concurrent mpv
sessions. FPS is uncapped native `end video-flip` throughput (`--untimed=yes`,
immediate Vulkan presentation).

| Profile | Median FPS | Versus corresponding Original |
| --- | ---: | ---: |
| Original Balanced | 84.425 | reference |
| Balanced concat | 87.011 | +3.063% |
| Balanced block-concat | 88.792 | +5.173% |
| Original Performance | 113.985 | reference |
| Performance concat | 118.219 | +3.714% |
| Performance Cache48 | 118.243 | +3.735% |

Block-concat added +2.047% over Balanced concat, positive in both paired
rounds (+2.445%, +1.654%). Cache48 added only +0.020% over Performance concat,
with opposite paired signs (-0.041%, +0.081%): this run does not reproduce a
meaningful extra cache-preset gain. Keep that negative result separate from
the historical improvement above. All values are specific to this workload,
driver and saved engines; the numerical-output differences still apply.

The completed 5090 540p comparison used a looped 24-frame real-video fixture,
10 seconds warmup and 20 seconds measurement, with two reversed-order rounds.
The model output was 1920x1080, rendered to the native 3840x2160 display through
NVDEC/gpu-next with three Vulkan queues and no dropped frames. Medians:

| Profile | Native-display FPS | Versus Original |
| --- | ---: | ---: |
| Original Balanced | 417.424 | reference |
| Balanced concat | 417.123 | -0.072% |
| Balanced NativeTail7 | 422.709 | +1.266% |

NativeTail7 improved both paired rounds (+1.500% and +1.033% versus Original).
Concat's paired changes had opposite signs and are a tie within variation.
This small gain applies to this fixture, GPU, engine and display. It does not
change the numerical-output differences documented above.

These remain optional experiments with shape restrictions, not automatic
replacements or a new claim of bit-exact stock output.
