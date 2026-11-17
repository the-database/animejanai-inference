# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`libaji` — a standalone C-ABI inference engine for real-time anime upscaling and RIFE frame
interpolation. It takes GPU-resident YUV frames, runs the chain described by `animejanai.conf`
(resize + ONNX models + RIFE) entirely on the GPU, and returns YUV.

The strict C ABI in `include/aji.h` is the point: `vf_animejanai` in the
[`the-database/mpv`](https://github.com/the-database/mpv) fork loads the library at **runtime**,
so mpv never links TensorRT. The backend and the TRT version can be swapped without rebuilding
the player, and on Windows an MSVC-built shim coexists with a mingw-built mpv.

`libaji` has **zero mpv dependency** — `aji_encode` (offline CLI) and `VideoJaNai` use the same
ABI.

## Documentation map

- **[`docs/BUILD-WINDOWS.md`](docs/BUILD-WINDOWS.md)** — the Windows dependency setup and
  release build. **The Windows release is not built in CI**: `aji-windows-x64.zip` is built and
  uploaded by hand with the scripts in `ci/windows/`, against SDKs in an out-of-repo working area.
  CI does build and test Windows on pull requests (see [CI](#ci)).
- **[`docs/BUILD-LINUX.md`](docs/BUILD-LINUX.md)** — the Linux/WSL build, the Linux release
  workflow, and the parity harness.

`README.md` is the public-facing overview. Its "Build (Linux)" section is still accurate; the
prose describing the engine as "Phase 0 spike scope" predates chain selection, RIFE, and the
DirectML backend.

## Architecture

```
vf_animejanai (mpv)  ──dlopen──►  aji.dll / libaji.so        (aji_dispatch.cpp — thin dispatcher)
                                        │ forwards over aji.h
                  ┌─────────────────────┼─────────────────────┬──────────────────────┐
            aji_trt.dll          aji_trtrtx.dll          aji_dml.dll          aji_vk.dll   (Windows only)
          TensorRT + CUDA     TensorRT for RTX + CUDA   DirectML + D3D12   aji_dml + Vulkan kernels
```

`[global] backend=` picks the library: `DirectML` (or the retired `NCNN`) -> `aji_dml`,
`TensorRT-RTX` (also `TensorRT_RTX`, `TRT-RTX`, `TRTRTX`; case-insensitive) -> `aji_trtrtx`,
`Vulkan` (also `VK`) -> `aji_vk` (built with `AJI_VULKAN` only; in development, see "The Vulkan
backend"), anything else -> `aji_trt`. The image ABI selects it with `AJI_IMG_TENSORRT_RTX`.

**`aji_trtrtx` is `aji_trt`'s sources compiled with `AJI_TRTRTX`** against the TensorRT for
RTX SDK; every difference lives in `core/trt_engine` (plus a few `#ifdef`s for log wording and
runtime-cache saves). Engines build **in-process** through the builder API (about a second, no
`trtexec`; the trtexec-style settings string is parsed for its shape/optimization-level/
workspace flags), are cached as `....trtrtx-<ver>.gpu-<name-smN>.rtxengine` — deliberately not
`*.engine`, which `aji_trt`'s stale cleanup (including released builds) would delete — and run
with a TensorRT-RTX runtime cache saved beside each engine as `<engine>.jitcache`. Each backend's
stale cleanup only touches its own extension, so both share a model directory.

**Engines remember their ONNX.** The cache key holds the model's file name, not its contents,
so every engine (both backends, both front-ends) gets `<engine>.onnxinfo` beside it: the ONNX's
size, last-write time and CRC-32 at build time (`engine_matches_onnx` in `core/trt_engine`). A
different size, or a different time *and* CRC, means the model under that name changed (a
re-export, or VideoJaNai's `StageModels` copying another model over it), so the engine is deleted
and rebuilt; a new time with the same CRC (the same file copied again) only refreshes the record.
An engine without a record (built by an older libaji) gets one from the current ONNX and is
reused, so updating libaji rebuilds nothing. Stale cleanup removes an engine's `.onnxinfo` (and
`.jitcache`) with it.

TensorRT-RTX compiles kernels at the first enqueue (seconds, into the driver's compute cache),
so every engine gets a **warm-up**: one throwaway inference on zeroed scratch buffers, on the
build thread right after a build (`run_build_inprocess`, so with `async_build` it is part of the
background build, before `aji_poll` fires) and again in `load_engine` (cheap once the caches are
warm). With `async_build`, a *cached* engine not yet warmed by this context also goes through the
worker first (a warm-only `AsyncBuild` job, logged as `Preparing TensorRT-RTX engine ...`;
passthrough until `aji_poll`), because after a driver update its JIT takes seconds: ~0.25-0.4 s
of extra passthrough per play when the caches are warm. The first `aji_infer` is then as fast as
TensorRT's; ~0.2-0.35 s (context creation + the warm enqueue) stays in `aji_configure`.
Synchronous in-process builds run on a thread of their own, so the caller's CUDA context and
runtime device are never touched. A `.jitcache` is rewritten only when its contents changed.
A cached engine that deserializes but then fails (context, warm-up) is kept and reported, not
deleted and rebuilt. TensorRT (`aji_trt`) has no JIT and gets none of this.

**TensorRT-RTX falls back to TensorRT** (TensorRT-RTX is opt-in and experimental; TensorRT stays
the default). The dispatcher keeps a copy of the create params for an `aji_trtrtx` context; if
`aji_create` fails, or `aji_configure` returns `AJI_ERR_ENGINE` (build or context failure,
blocking or after a background build — `aji_trtrtx` reports a failed background build as an
error instead of passthrough, once), it logs one line, `TensorRT-RTX failed (...); using TensorRT
instead`, re-creates the context on `aji_trt` (same params, current slot) and repeats the call.
Sticky for the context's life; conf mode only. If `aji_trt` can't be loaded either, it logs that
and repeats the call on `aji_trtrtx`, which then plays passthrough like `aji_trt` does after a
failed background build (blocking mode keeps returning the error). The image ABI has no fallback: its caller picks
the backend explicitly and can retry with `AJI_IMG_TENSORRT`. Test hook:
`AJI_TRTRTX_FAIL=create|build|context` forces that step to fail in `aji_trtrtx`
(`~/aji-win/trtx-work/fallback-test.cmd` drives all cases).

`aji` is deliberately dependency-free — no CUDA, no TensorRT, no ORT — so the player can load it
on any machine and the backend resolves later. The backends load `onnxruntime.dll` /
`DirectML.dll` from their own directory at runtime, which is why nothing is import-linked for
the DML path.

**Built-in slots are code, not config.** `add_builtin_slots` in `src/aji_conf.cpp` hardcodes
slots 1001–1003 (Quality/Balanced/Performance) and 1010–1013 (benchmark / RIFE-order templates),
**including the ONNX model filenames** via the `HD_BAL` / `HD_PERF` / `SD` constants. A model
rename has to be made here, in the package's committed `animejanai/onnx/`, and in the AnimeJaNai
Manager's default profiles.

User slots 1–9 come from `animejanai.conf`, which the Manager writes.

### Two front-ends over one internal core

`aji.h` (video) is one front-end; `include/aji_image.h` (still images: 8-bit gray/RGB in host
memory, any size, any ONNX SR model — no conf, slots, fps or YUV) is the other. Both are
exported by `aji.dll` (`aji_img_dispatch.cpp` forwards the image ABI to the backend named at
`aji_img_create`) and implemented inside the same backend DLLs: `aji_img_trt.cpp` in `aji_trt`,
`aji_img_dml.cpp` in `aji_dml`.

What both front-ends share lives in `src/core/`, compiled into each library with hidden
visibility (an internal static "library", no ABI of its own):

| File | What |
|---|---|
| `core/fileio` | UTF-8/long-path file helpers, `own_dir()` |
| `core/trt_engine` | engine cache naming, the `.onnxinfo` model records, stale-engine cleanup, trtexec build specs/subprocesses, `AsyncBuild`, engine loading; under `AJI_TRTRTX` the in-process builder and the runtime (JIT) cache |
| `core/ort_dml` | ORT + DirectML loading (DirectML.dll before onnxruntime.dll) |
| `core/tiling` | the image path's tile cover / reflect pad / overlap blend on the host (backend-agnostic; DirectML uses it, and the CUDA backends with `AJI_IMG_CPU_TILING=1`). The band is a ring of rows; blend and flush are split across a few threads with `parallel_for`, per element, so the result doesn't depend on the split. **Seams:** overlaps default to 128 px and blend only across a 16 px band at their centre (`DEFAULT_OVERLAP`, `BLEND_WIDTH`). MangaJaNai's ESRGAN models generate screentone whose phase drifts near a tile's edge; the former 32 px linear ramp mixed the drifted patterns into visible interference patches. Torture test: `bwgradient_autolevels.png` (flat gradient, so the model invents all halftones) against a single-tile run. Also `even_origins`, the layout for `AJI_IMG_LOAD_EVEN_TILES` |
| `core/tiling_gpu` | the same tiling on the GPU for the CUDA backends (`upscale_gpu` in `aji_img_trt.cpp`): one upload, tile extract / blend / 8-bit flush kernels on the model's stream. **Bit-identical to `core/tiling`**: every multiply and add is `__fmul_rn`/`__fadd_rn` (never fused) in the host's order, with the host's fp16/bf16 rounding. Keep it that way, or the md5 gate below stops working |
| `core/half` | fp16/bf16 conversions |

Core functions take explicit paths/devices and return errors as strings; each front-end reports
them through its own context. **A change to engine caching or builds goes in the core**, so both
front-ends get it. The image front-end does inference only — model selection, grayscale
detection, resizes and alpha are the caller's (yosh / MangaJaNaiConverterGui keep that policy).

**The image path shares GPU working memory per context** (`aji_img_trt.cpp`). Its execution
contexts are created `kUSER_MANAGED` (`load_engine`'s `bind`), and one activation buffer per
`aji_img_ctx`, sized to the largest loaded model's `getDeviceMemorySizeV2()`, serves them all,
with the GPU tiling scratch; so runs on one context are serialized (`run_mu`; lock order `mu`
then `run_mu`). With TensorRT's default each model kept its own activations (~340 MB for a 2x
ESRGAN at 512 tiles, ~1.35 GB for 4x), which piled up in apps that never unload. The video path
keeps the default: its chains run upscale, RIFE and temporal steps that may overlap. Out of
memory at load is reported as such and keeps the engine (TensorRT's own allocation failures are
caught through `TrtLogger::oom_errors`), instead of deleting it as stale and rebuilding.

**`aji_img_memory_get`** (0.28.0, probed like `aji_img_release_device`) reports the device's free
memory, the context's shared activations and scratch, and a model's weights (its engine file's
size), activations and tile buffers, so apps size tiles to fit (yosh-image's `tiles::Budget`).
Activations grow linearly with the tile's area (TensorRT-RTX: ~1.1 KB per input pixel for 2x
ESRGAN, ~4.5 KB for 4x), so one engine predicts another's. DirectML can't report per buffer: a
model's activations are this process's local video memory growth (`QueryVideoMemoryInfo`) across
its load and warm-up runs, less the ONNX file's size. On Windows the free memory leaves out other
processes' (WDDM pages them out instead of failing, at a heavy cost in speed), so apps take NVML's
device-wide figure where lower.

**`aji_img_native_probe`** (0.29.0, probed the same way) answers, before a load, whether
`aji_img_load` with those params would run the model on native kernels, and fills the memory it
would use there (weights; its own tile buffers as `model_activations`, `activations_shared` 0). It
builds the native model and frees it, building no engine; `AJI_IMG_ERR` means a load would build
a TensorRT engine (or the backend has none: DirectML). Apps size a native model's tiles from it on
a first run instead of guessing, and keep their TensorRT guess for the rest. `aji_image --probe`
prints it.

### Native kernels (`src/native`, on by default; `[global] native=off` disables)

`aji_trt` and `aji_trtrtx` run every model the native path recognises on hand-written CUDA
kernels instead of TensorRT, on both the video and the image path (sm_80+; older GPUs and other
models use TensorRT, with the reason logged). Video: `native=off` (or `no`) in `[global]` turns it
off; any other value, or no key, is on. Image path: `AJI_NATIVE=off` (or `0`, `no`, `false`).

| Models | Video path (`aji_trt.cpp`) | Image path (`aji_img_trt.cpp`) |
|---|---|---|
| SPAN, SPANF3, span_s, Compact (SRVGG) | `native/lower` + `native/executor` at the chain's size | the same plan and executor behind `ImgNet` (`build_lowered`, `native/framenet.cpp`), per tile |
| ESRGAN, FDAT (M, XL), HAT-L, DAT-2 | the `native/imgnet` builders on the whole frame (`FrameNet::image`), when it fits in memory (ESRGAN, FDAT, HAT and DAT also on Vulkan: `vk/esrgan.cpp`, `vk/fdat.cpp`, `vk/hat.cpp`, `vk/dat.cpp`) | the `native/imgnet` builders, per tile |
| TFDAT, TemporalFix | `native/vidnet` in the temporal stage (also on Vulkan: `vk/tfdat.cpp`, `vk/tfix`) | — |
| RIFE | `native/rife` (the graph read by `native/rife_plan`, which Vulkan's `vk/rife` shares) | — |

Video, single-frame models: in `aji_configure` (`try_native` in `aji_trt.cpp`) the model's ONNX
is parsed (`native/onnx_graph`, no onnx dependency) and lowered (`native/lower`); a lowered model
is instantiated for the chain's size (`native/executor`, `FrameNet::lowered`). A model the
lowering refuses goes to the image builders on the whole frame (`FrameNet::image`,
`native/framenet.h`). They size their buffers for their largest input (per input pixel: HAT-L /
DAT-2 ~10 KB, ESRGAN 4x ~6.5 KB, ESRGAN 2x ~1.6 KB, FDAT_M 2x ~1.4 KB), so the model is first built at a 128x128
probe for its size rule and per-pixel buffers, and the frame is refused (TensorRT) when weights
plus buffers scaled to it exceed the free memory less 1 GiB + 10% of the device; otherwise it is
built at the frame size, reflect-padded at the right and bottom to the model's size rule (HAT: 16,
as its own graph pads) and cropped back. Whole frames rather than tiles: no seams and no overlap
work, the same computation as TensorRT's whole-frame engine; the cost is memory, hence the
budget. The image builders keep it down the way an engine's memory planner does: their per-pixel
buffers share one arena by lifetime (`Lifetimes` in `native/netres.h`, each buffer listing the
steps it holds data in; ESRGAN, HAT, DAT), and FDAT's upsampler reuses its body's scratch as
TFDAT's does. On the 5090 that puts them under TensorRT's whole-frame engines (720p: DAT-2 5.2 vs
6.0 GB, HAT-L 4.9 vs 7.9, FDAT 3.3 vs 3.7, ESRGAN 2x 1.5 vs 2.1); before v0.31.1 each buffer had
its own allocation and they used up to 1.8x TensorRT's. `AJI_NATIVE_POISON` fills those arenas
with 0xFF (NaN) instead of zeros, so a step reading another buffer's leftovers shows in the
output; with it set the output must match a normal run. Either way it becomes a `Step::NATIVE`, run by `run_steps` like a TensorRT `MODEL` step
(same fp16 NCHW RGB in and out), so graph capture, RIFE, temporal steps and mixed chains are
unchanged. Nothing is built: configure takes tens of milliseconds for a lowered model, 0.1–0.5 s
for FDAT / ESRGAN and ~1 s for HAT-L / DAT-2 (reading and parsing the file, the probe, timing the
kernel variants). The log gets
`Native kernels: <model> (<what>)` before the usual `Applied Model` line (`..., whole frame
[padded to WxH], <n> MB` for the image builders), or `Native kernels not used for <model>: <why>;
using TensorRT` (e.g. `a whole 2560x1440 frame needs ~N GB, M GB free (keeping K GB)`), and the
model goes through `ensure_engine` as before. The lowering, and an image-builder refusal that is
not about the frame size or memory, are cached per ONNX path for the context's life. Temporal
models: `try_native_temporal` (`native/vidnet`).

Image path: `load_native` builds `ImgNet::create` at the tile size, which tries ESRGAN, FDAT,
HAT, DAT, then the lowered conv nets (`LoweredImg`: the executor at the tile; even tiles smaller
than it run from the same buffers, `Executor::run(in, out, w, h, s)`; sides must be multiples of
the pixel-unshuffle factor). `aji_img_memory_get` and `aji_img_native_probe` see every family
through the same `ImgNet` interface (`weight_bytes`, `device_bytes`).

The lowering works op by op on the graph's data path and folds into each Conv what follows it (activation,
SPAN gate with the residual, depth-to-space and SRVGG's upsampled-input add); `Concat` + 1x1 `Conv`
is the concat kernel; pixel-unshuffle (reshape/transpose or `SpaceToDepth`) and `DepthToSpace`
(CRD or DCR, by reordering weights) fold into the first and last convs. Shape-only plumbing,
casts, and pads/slices that align to the unshuffle factor are ignored; any other op refuses the
model with its reason. `aji_native_lower_test` (cpu) checks the plans for SPANF3-, Compact- and
span_s-shaped graphs written in memory, and the refusals.

**What's host-only** (no CUDA, shared by any backend that runs a `Plan`): `native/ops.h` (the
`Input` / `Act` / `Gate` / `Output` values a lowered conv step stores), `onnx_graph`, `lower`,
and `native/layout`. `layout()` instantiates a plan's sizes for one input: the even-size
checks, each step's padded channel counts, and the feature buffers (one per live tensor,
reused once its last reader has run). `Executor::create` allocates and uploads from it.
`pack_fragments` is host code too, but its layout is the `mma.sync` B fragment, so it's CUDA's
packing, not a shared one.

The kernels: one family, `k_conv3x3` (`native/conv_kernels.cuh`), a 3x3 conv on fp16 NHWC activations with
`mma.sync` m16n8k16 and **fp16 accumulation** (fp32 accumulation was slower than TensorRT), A
tiles from padded shared memory via `ldmatrix`, and weights packed on the host into B-fragment
order (`native/pack.cpp`; the layout is in `pack.h` and checked by `aji_native_pack_test` against
the PTX ISA's table). The model's elementwise work is folded into the conv: activations (ReLU,
PReLU, SiLU; LeakyReLU = PReLU with a constant slope), SPAN gates with the residual (tanh of a 1x1
conv fed straight from the accumulators, sigmoid−0.5, plain residual), NCHW / pixel-unshuffled
model input, and depth-to-space output with SRVGG's upsampled-input add. `k_concat1x1` is
`conv_cat` without the concat buffer. `native/conv.h` is the API; `conv.cu`'s `kConv` table is the
set of instantiated shapes (32/48/64 channels), each with two tile variants that `pick_variant`
times on the running GPU. Needs sm_80+ (the kernels trap below it; `device_supported` says so).

`aji_native_conv_test` (gpu) checks every table entry and variant at a ragged and an aligned size
against a CPU reference, with guard bands. `aji_native_bench` (built on request) assembles whole
SPANF3 / Compact models from `conv.h` calls on data from `experimental/native-span/v2/prep_*.py`;
on an RTX 5090 it runs SPANF3 Performance 1080p→4K in ~2.0 ms vs ~3.6 ms on TensorRT 11.4, at
58.6 dB vs ORT fp16 (`experimental/native-span/README.md` has the numbers and the NeoRTX
comparison). Tile variants are only tuned on the 5090.

### The Vulkan backend (`aji_vk`, in development)

The native kernels on Vulkan, for GPUs without the CUDA native path (AMD, Intel, RTX 20, later
Linux and Android). Plan, phases and measurements: animejan.ai-site
`plans/vulkan-backend/plan.md`. Built with `AJI_VULKAN=ON`; not shipped yet.

- **`src/vk/vk.h`:** the Vulkan entry points, loaded at run time from `vulkan-1.dll` /
  `libvulkan.so.1` / Android's `libvulkan.so` (nothing links a Vulkan SDK), on one process-wide
  instance.
- **`vk/device`:** a device by index or by the D3D11 adapter's LUID; which kernel family it runs
  (`AJI_VK_FAMILY=cm16|sg16|plain` overrides); `capability_report()`, everything a device offers
  the families; buffers; one-shot and frame submits; pipelines from the
  embedded SPIR-V. Its variant and pipeline caches go in the model directory as
  `aji-vk-<vendor>-<device>-<driver>.variants` / `.vkcache` (`AJI_VK_RETUNE=1` times again).
- **`vk/kernels`, `src/vk/shaders`:** the fused conv and concat kernels of a `Plan`, with the CUDA
  kernels' semantics, in three families:
  - **cm16:** `VK_KHR_cooperative_matrix` 16x16x16 with fp16 accumulate (NVIDIA Ampere+, AMD
    RDNA3/4). The epilogue runs on the accumulators: bias, slopes and the attention bias load as
    matrices with every row the same vector, a gate's block input loads from its feature buffer,
    and activations are element-wise. That needs the slack after each feature buffer.
  - **sg16:** register-blocked with the subgroup pinned to 16 lanes (Intel Xe, Mali); fp16 per tap,
    then fp32. Inputs travel between lanes as packed fp16 pairs in 32-bit values, so no fp16
    subgroup operations are needed.
  - **plain:** sg16's blocking with no subgroup operations and no subgroup size: a thread owns two
    output channels and reads its inputs from shared memory. For what has neither of the others
    (Adreno's 64/128-lane waves); within 1 level of sg16, about 2x slower than it on a UHD 770.

  `shaders.txt` lists the modules built: the CUDA native kernel table (`native/conv.cu` kConv,
  kConcat: 32/48/64 channels, every activation and gate, the heads and tails, concats of 2..4) in
  each family; a step whose module isn't there refuses the model. Where 64-pixel concat blocks
  pass the 32 KB of shared memory of Intel and phones (64 channels x 4 sources), sg16 and plain
  take 32-pixel ones (`_b2`). `aji_vk_conv_test` (`ctest -L gpu`) checks every entry against the
  CPU reference on the device's family, both tile variants.
- **`vk/executor`:** the `Plan` on the device, from the shared `native/layout`. Each kernel's tile
  variant is timed once per size (identical steps share the timing) and stored. `bind_io` points
  the model input and output at caller-owned buffers.
- **`aji_vk` (Windows) is `aji_dml.cpp` built with `AJI_VULKAN`,** as `aji_trtrtx` is `aji_trt`'s
  sources with a define. The D3D12 frame pipeline is DirectML's: plane copies, pre/post, resizes,
  the ticket API. Each model step that lowers runs on the Vulkan executor, and an ESRGAN step on
  the whole frame (`ImgNet::frame`, the memory rule of CUDA's `FrameNet::image`: refused unless
  weights plus buffers fit in free memory less 1 GiB + 10% of the device); the others run on
  DirectML, so the fallback is per model. The log line is `Native kernels: <model> (...; Vulkan
  cm16)`, or `Native kernels not used for <model>: <why>; using DirectML`.
  - The chain buffers are shared D3D12 buffers imported into Vulkan (`import_d3d12_buffer`).
  - Hand-offs use a shared D3D12 fence imported as a timeline semaphore: D3D12 signals, Vulkan
    waits, runs the model and signals, and D3D12 waits.
  - Frames are never imported as textures: a shared D3D11 NV12 texture imported straight into
    Vulkan reads wrong chroma on NVIDIA, and Intel allows no arrays.
- **`vk/gconv`, `vk/esrgan.cpp`, `vk/imgnet.h`:** the general 3x3 conv (`gconv_*.comp`, the
  counterpart of CUDA's gconv): a channel prefix of a wider NHWC buffer in, a channel slice out, at
  strides and offsets per dispatch (a parameter buffer); the input channels and the halo chunk
  (16 / 32 / 64 channels in shared memory at a time) per dispatch too; bias, two scaled residuals
  and LeakyReLU per element in fp32 in the writer; nearest 2x upsampling of the input on the fly;
  stride 2; NCHW in (three planes at an offset, or their unshuffle) and NCHW out; phase-scattered
  outputs (a transposed conv, or one followed by depth-to-space, as 2 or 4 row-phase convs,
  `phase_weights`). A dispatch's z index is an output slice, so a conv wider than 64 channels is
  one dispatch, and narrow slices give small low-resolution convs enough workgroups. The shapes
  built are in `shaders.txt` (`g3_...`). `GConvOp` + `gconv_prepare` / `gconv_record` /
  `gconv_choose` are what a network builder records with (variants timed once per module, size
  and slice count). `aji_vk_gconv_test` (`ctest -L gpu`) checks each mode against a CPU reference.
  ESRGAN (RRDBNet 64/32, 2x unshuffle and 4x) follows `native/esrgan.cpp`'s plan on it, with its
  intermediates in one arena by lifetime; a tile whose buffers exceed `maxStorageBufferRange` is
  refused. `ImgNet::create` tries ESRGAN, then the lowering.
- **`vk/tnet`, `vk/fdat.cpp`, `vk/hat.cpp`, `vk/dat.cpp`, `vk/tfdat.cpp`, `tfm.comp`: the
  transformer models on Vulkan** (FDAT, HAT, DAT; TFDAT in the temporal stage). `TNet` is one arena of maps (as `tfix.comp`'s), fp32 / fp16 weight
  stores, a partial-sums buffer and a list of steps: gconv dispatches, `tfm.comp` passes, copies,
  fills. `tfm.comp` ports `native/tfm.cu` and `fdat_kernels.cu`: the GEMM (fp32 sums, tfm's
  epilogue: bias, act, column scale, two residuals, act last), LayerNorm (32 lanes a token),
  depthwise 3x3 (an 8-channel group of a column over an 8-row strip, the rows reused; channel sums
  as per-workgroup partials, no float atomics), window attention (a workgroup per window, head and
  64 queries; keys in blocks of 32, P rounded to fp16 as tfm's), the channel-attention statistics,
  softmax and fold (into fp16 GEMM weights on the GPU), and FDAT's gates. On cm16 GPUs with fp32-sum
  cooperative matrices (`DeviceInfo::cm16_acc32`) the GEMM is `tfm_gemm_cm16.comp`. gconv gained
  GELU (ACT 1) and phase scatters split into output slices (FDAT's 2x C-channel transposed conv).
  `ImgNet::create` and `ImgNet::frame` (video) try ESRGAN, FDAT, HAT, DAT, then the lowering.
  FDAT_M and FDAT_XL (the shipped IllustrationJaNai exports; the op23 "dynamo" exports aren't read,
  as on CUDA) match CUDA native at 69-71 dB; 720x480 2x page on the 5090: 46 ms (DirectML 2238, CUDA
  23). HAT-L (`vk/hat.cpp`, native/hat.cpp's plan) keeps the residual stream fp32; its window
  attention is the general form (a relative-position table instead of a full bias, Swin's mask in
  the shifted blocks, the OCAB's overlapping zero-padded key windows); the CAB's channel gate runs
  on the GPU (`TNet::chan_gate`, ReLU and biases) and scales the CAB in the proj GEMM's epilogue
  (the add2 scale is an arena offset); the group conv is the GEMM on an implicit im2col (params
  31-34) for fp32 sums, output and residual. The input is reflect-padded to the window multiple
  inside the net (`imgin`) and the output cropped (`imgout`), so any tile or frame size runs. Both
  shipped exports match CUDA native at 72 dB; 512x512 on the 5090: 846 ms (CUDA 365; DirectML can't
  load them). Most of the gap is window attention on scalar math (CUDA's is on tensor cores).
  `TNet::winattn` (a `TWin`) sets pass 5's parameters; its relative-position table (42) must be
  -1 when there is none, as 0 is an offset. DAT-2 (`vk/dat.cpp`) reads the data path op by op
  as native/dat.cpp does and evaluates the position-bias MLPs on the host
  (`native/static_eval.h`, shared with CUDA) into tables; the attention pads to its windows (Hp,
  Wp), so any size runs; the token gate (`tgate`, a thread a token) and the mix are tfm.comp
  passes 16 and 17, the depthwise conv takes a map to multiply by (the SGFN). Both shipped exports
  72 dB from CUDA native; 720x480 page 0.63 s (CUDA 0.32; DirectML can't load them).
  TFDAT v2 (`vk/tfdat.cpp`, a `TemporalNet` with `scale()` and no frame cache, tried before
  TemporalFix) follows native/tfdat.cpp; its alignment kernels are tfm.comp passes 18-25, the FDAT
  body is `vk/fdat_body.h` (shared with vk/fdat.cpp), and the alignment, the body's scratch and the
  tail share one arena region (`TCarve`), as CUDA's phases do. bndl animefilm v3 2x: 73 dB from
  CUDA native; 1080p window 245 ms (CUDA 120), 720x480 in aji_vk 46 ms a frame (DirectML 1490).
  The v1 exports (no flow_net) aren't read, as on CUDA. Convs can write at a channel offset
  (`ConvIO::dst_off`).
  `AJI_TNET_SKIP=<step label prefix>` records no dispatch for those steps: per-step timestamps
  attribute time to the wrong step on NVIDIA, so this is how to see what a kind of step costs.
  `aji_vk_model_test image <onnx> W H` (any image model vs CUDA native), `gemm M K N [act]`,
  `attn H W heads hd [ws]` benchmark the kernels.
- **`vk/tfix`: TemporalFix on Vulkan** (`TemporalNet`), native/tfix.cpp's graph reading and step
  order: the 3x3 convs on gconv, the depthwise / pointwise convs (fp32 sums, as CUDA's tfm) and
  the resampling, warps and blend in `tfix.comp`, the six supports one after another. Its
  intermediates are maps into one arena (fp16 or fp32, offset / stride / channel offset, as
  `TxMap`); a map can name a window position and read entry `sel[pos]` of a stack. The frame cache
  (as aji_trt's): `record_encode` puts one input frame's encoder output into an entry,
  `record_cached` runs a window from entries, each position's entry picked on the GPU under the
  gather's scene-cut rule (`tfix_select`, from the stage's diff sums). In `aji_vk` the temporal
  stage keeps a gen per ring slot and per entry, gathers the window's missing frames into the
  model input, then encodes them and runs the window in one submit (`AJI_NATIVE_FRAME_CACHE=off`
  encodes the whole window, bit-identical). 1080p on an RTX 5090: 14.3 ms per frame (DirectML
  18.5, CUDA native 8.1); 69.3 dB from CUDA native over 16 frames (DirectML: 51.2 dB);
  `aji_vk_model_test temporal` checks the cache against the whole window bit for bit.
- **`vk/rife`: RIFE v4 on Vulkan,** from the graph `native/rife_plan` reads (shared with CUDA's
  `native/rife`): the encoder, each block's convs on gconv, and the resampling (block inputs
  with their warps, the ensemble merge, the output blend) in `rife.comp`, ports of
  `native/rife_kernels.cu` with the same fp16 roundings, the gather specialized like CUDA's
  template (`rife_gather_f<F>_m<NM>_k<KV>_t<NT>_b<B0>_r<REV>`). The timestep comes from the input's
  plane 6. In `aji_vk`, `setup_rife` opens it unless `native=off` (log: `Native kernels:
  rife_v4.14 (...; Vulkan cm16)`; else DirectML, with the reason); the 11-channel input and the
  output tensors are shared D3D12 buffers, and the step goes through the same fence hand-off
  (`vk_submit`). 1080p on an RTX 5090: 4.1 ms of GPU time (CUDA native 2.6-2.9); 7.2 ms per
  interpolated frame through `aji_infer_rife` (DirectML 10.6). Output vs CUDA native: 74.6 dB,
  max 5 levels over 47 frames (DirectML is 49.6 dB from CUDA). `aji_vk_model_test rife <onnx> W
  H` times it and compares with CUDA native RIFE (`AJI_VK_PROFILE=1`: time per kind of step).
- **`vk/frame`: a 4:2:0 frame on Vulkan alone, no D3D** (for Linux and Android): raw NV12 / P010
  planes in buffers, pre (`frame_pre.comp`), the executor, post (`frame_post.comp`) in one
  command buffer. The kernels are GLSL ports of `dml_shaders.h`'s pre/post with the same
  `resample.h` tables; within one level of the DirectML path (division rounds differently
  across APIs, so not bit-identical). `aji_harness_vk` runs raw files through it (`--format
  p010`, `--matrix`, `--range`, `--siting`) and builds for Android. `FrameIO` is the two ends
  around any RGB buffers (input and output formats apart); `FramePipeline` is FrameIO around one
  lowered model.
- **Vulkan frames: `aji_create_vk` (`vk/video`, additive ABI, no version bump).** A context on the
  caller's `VkDevice` (`aji_vk_device`: handles, the queue and its lock, the enabled extensions
  and features chain) with frames as multi-planar VkImages on it (`aji_frame.plane[0]` =
  `aji_vk_image`: image, layout, queue family, access, timeline semaphore and value, FFmpeg's
  `AVVkFrame` rule: wait for value, signal value + 1, write back what libaji left). For mpv with
  `gpu-api=vulkan` + `hwdec=vulkan` on Linux and Windows. `Device::create_external` wraps the
  caller's device: the kernel family is what the device was *created with* (cm16 needs
  `VK_KHR_cooperative_matrix`, the feature, the memory model and subgroup size control enabled;
  else sg16, else plain), every submit runs inside the caller's queue lock, and destroy leaves the
  device alone. `VkVideo` runs the conf's chain (lowered models and the `ImgNet::frame` ones,
  ping-ponging two chain buffers) between `FrameIO`'s pre and post, one submit a frame: the input
  planes copied out of the image (`vkCmdCopyImageToBuffer`, per plane aspect), the output copied in,
  the images' semaphores and the context's own timeline (aji_flush's tickets) signalled. Resizes
  before upscale run as `RgbResize` (`frame_resize.comp`, the HLSL cs_rs_h / cs_rs_v; the factor,
  then the height fitted to 16:9, as aji_dml.cpp's). RIFE as aji_dml.cpp's: padded to 64 and
  centred (studio-black borders filled once), two `FrameIO`s with bilinear chroma and BT.709 into
  the model input at element offsets 0 and 3 * plane, the timestep plane by `vkCmdFillBuffer`
  (truncating fp16), channels 7..10 (mesh) once by `fr_rife_consts`, the scene detection
  (`frame_scd.comp`, SCDetect's metric over the padded luma, divided by the padded area) read back
  between two submits, post and a cropped copy into the output image; synchronous like the
  other backends'. A RIFE-only chain configures to 0 with RIFE active, as the D3D12 path.
  RIFE-first hoists the first model's explicit resize ahead of RIFE, as aji_dml.cpp:
  `aji_pre_resize` reports the work size, `aji_resize` runs pre, `RgbResize` and post on a
  source frame, and the chain then starts at the work size. With a temporal model, RIFE runs
  after the upscale, as on D3D12. Temporal models
  (`TemporalNet`: TFDAT, TemporalFix) as aji_dml.cpp's: one ring per temporal step at its input
  size, the cut metric per stored frame (`frame_temporal.comp` PASS 1, then `fr_scd2`), the
  window gathered padded to 32 (PASS 2) with the same clamping and cut rules, the model's frame
  cache, stacked stages through `rec_ensure`; `aji_ingest` and `aji_infer_seq` are one submit
  each, their per-dispatch parameters written in the command buffer (`vkCmdUpdateBuffer`).
  `aji_frame_diff` (so the dispatcher's held-frame engine) on two Vulkan frames:
  `frame_diff.comp`, the HLSL cs_frame_diff with the same weights table and atomics, synchronous,
  its own scratch. Not yet there, passed through or left out with the reason in the log: models
  without Vulkan kernels (no DirectML on this path). On
  Windows `aji_vk.dll`'s exports (aji_dml.cpp) forward a Vulkan-frames context (`c->vkv`); on Linux
  `aji_vk_frames.cpp` is the video ABI. The dispatcher's `aji_create_vk` always loads aji_vk.
  `aji_vk_frames_test` plays the player (a device of its own, NV12 / P010 images, `--no-coopmat`
  reports a device without cooperative matrices): SPANF3 720x480 0.64 ms a frame on the 5090,
  within 1 level of the D3D12 path. `--rife --rife-model-dir` is aji_harness_dml `--rife`'s loop:
  rife_v4.14 1080p 4.3 ms, 74.6 dB against the D3D12 path; same scene skips on cut clips. A
  temporal chain runs aji_harness_dml's ingest loop (`--reset-at K`): TFDAT 2x 720x480 within 2
  levels of the D3D12 path, TemporalFix 62.9 dB (5.4 ms a frame). `--diff` prints aji_frame_diff
  of consecutive frames (exact against frame_diff.h's arithmetic; 1.8 ms a pair at 720x480).
- **`aji_img_vk.cpp`: the image ABI on Vulkan (`AJI_IMG_VULKAN`, `aji_image --backend vulkan`).**
  A `core/tiling` `TileModel` over the executor at the model's tile: the same cover, reflect pad
  and blend as the other backends. Every model must lower; one that doesn't is refused with the
  reason (no engine fallback, since it also runs where there is no DirectML). Device ids are the
  PCI address as the other backends report it (`VK_EXT_pci_bus_info`), else the LUID, else the
  list position plus one (phones). `aji_img_memory_get` uses `VK_EXT_memory_budget` when there;
  `aji_img_native_probe` is a load and a release. One upscale at a time per context. It's in
  `aji_vk.dll` on Windows (in place of `aji_img_dml.cpp`) and is `libaji_vk.so` elsewhere,
  Android included. Tiling runs on the GPU, as `core/tiling_gpu` does for CUDA: one upload of
  the 8-bit image, then per tile row one command buffer of `tile_extract` -> model ->
  `tile_blend` per tile and a `band_flush` into a host-visible buffer (`src/vk/shaders/tile_*`,
  `band_flush.comp`; parameters per dispatch in a buffer, since the push constants are 16 bytes).
  Bit-identical to the host path (`AJI_IMG_CPU_TILING=1`): `precise` arithmetic and integer fp16
  conversions.
- **`aji_vk_model_test`** (by hand): one model on the executor, timed, against the fp32 CPU
  reference of its plan (`sanity/native_ref.h`), and against the CUDA executor when built with CUDA.
  `aji_vk_model_test list` prints the devices and families, `info [N]` the full capability
  report, and `--seconds S` keeps the model running and prints its time per second (thermal
  throttling). It builds for android-arm64 (NDK, `-DANDROID_ABI=arm64-v8a`, `AJI_CUDA=OFF`
  `AJI_VULKAN=ON`) and runs from an `adb shell`. `aji_harness_dml --adapter N` runs a chain on
  another GPU, e.g. the iGPU.

## Build system

CMake only (`cmake_minimum_required(VERSION 3.24)`, C11 / C++17, `LANGUAGES C CXX`, with CUDA
enabled by `AJI_CUDA`). There is no Cargo, meson, vcpkg, or conan manifest anywhere. The single
`find_package` is `CUDAToolkit REQUIRED`, under `AJI_CUDA`.

**`AJI_CUDA` (default ON)** builds the CUDA backends (`aji_trt`, `aji_trtrtx`), the native CUDA
kernels, `aji_harness`, `aji_encode` and the CUDA tests. **`AJI_CUDA=OFF`** builds what needs no
CUDA toolkit: the dispatcher, `aji_dml` (Windows), `aji_image`, `aji_harness_dml`,
`aji_native_host` (the native path's host half) and the CPU tests. It's for the Vulkan
backend's development and for machines without CUDA (Linux on AMD/Intel, Android). Without CUDA
the dispatcher's runtime-pack names (`AJI_PACK_CUDART`, `AJI_PACK_NVINFER`) are empty, which
`core/runtime_packs` reads as nothing to preload.

**`AJI_VULKAN` (default OFF)** compiles the Vulkan compute shaders in `src/vk/shaders` with
`glslc` (`AJI_GLSLC`; found on `PATH` or under `$VULKAN_SDK/bin`; a missing glslc is a configure
error, not a skip) and embeds them as SPIR-V (`cmake/aji_spirv.cmake`). The modules are listed
in `src/vk/shaders/shaders.txt`, one per line: name, shader file, `-D` defines. One shader
file makes many modules (its structural variants), and the generated table is `vk/spirv.h`'s
`kSpirvModules`. On Windows, MSYS2's `mingw-w64-ucrt-x86_64-shaderc` glslc works; the
Android NDK's shaderc (2022) predates `GL_KHR_cooperative_matrix`. It also builds `aji_vk_core`,
`aji_vk_model_test` and, on Windows, `aji_vk`. Those need the Vulkan headers: from the Vulkan
SDK, or `AJI_VK_INCLUDE` (a directory holding `vulkan/vulkan_core.h`). Don't point it at all of
MSYS2's `include`, which would shadow MSVC's C headers; copy `vulkan/` and `vk_video/` out. CI
doesn't build `AJI_VULKAN` yet. A build without CUDA and with the shaders:

```
cmake -S . -B build-nocuda -G Ninja -DAJI_CUDA=OFF -DAJI_VULKAN=ON -DAJI_GLSLC=<glslc> \
  -DAJI_ORT_ROOT=... -DAJI_DML_ROOT=...   # Windows, for aji_dml
```

### Cache options

| Option | Type | Default (Windows) | Default (Linux) |
|---|---|---|---|
| `AJI_TRT_ROOT` | PATH | `""` | `$ENV{HOME}/sdk/tensorrt/usr` |
| `AJI_NVINFER` | STRING | `nvinfer_10` | `nvinfer` (plain `set`, not cached) |
| `AJI_TRTRTX_ROOT` | PATH | `""` (target skipped) | `""` (target skipped) |
| `AJI_FFMPEG_ROOT` | PATH | `""` | *(unused; pkg-config instead)* |
| `AJI_ORT_ROOT` | PATH | `""` | *(Windows only)* |
| `AJI_DML_ROOT` | PATH | `""` | *(Windows only)* |

The include/lib layout under `AJI_TRT_ROOT` differs by platform: Windows expects headers in
`<root>/include` and libs at `<root>` itself; Linux expects `<root>/include/x86_64-linux-gnu`
and `<root>/lib/x86_64-linux-gnu`.

`AJI_TRTRTX_ROOT` takes the same flat layout on Windows (`<root>/include`, libs at `<root>`;
Linux: `<root>/lib`). The import-library names carry the SDK's major/minor on Windows
(`tensorrt_rtx_1_7.lib`, `tensorrt_onnxparser_rtx_1_7.lib`); CMake reads them from the SDK's
`NvInferVersion.h`, so there is no `AJI_NVINFER`-style name option to forget.

> **`AJI_NVINFER` defaults to `nvinfer_10` but the project now builds against TensorRT 11.**
> Every current build passes `-DAJI_NVINFER=nvinfer_11` explicitly. Omitting it on Windows
> produces a link error for a library that is not there.

### Targets

| Target | Kind | Links | Notes |
|---|---|---|---|
| `aji` | SHARED | `${CMAKE_DL_LIBS}` | the dispatcher; `CXX_VISIBILITY_PRESET hidden`, PIC on |
| `aji_trt_common` | OBJECT | — | conf, ONNX shape, fileio, tiling, `kernels.cu`: the TensorRT-independent sources, compiled once for both TensorRT backends |
| `aji_trt` | SHARED | `${AJI_NVINFER}`, `CUDA::cuda_driver`, `CUDA::cudart` | + `aji_trt_common`; non-Windows adds `-Wl,-Bsymbolic` |
| `aji_trtrtx` | SHARED | `tensorrt_rtx_<M>_<m>`, `tensorrt_onnxparser_rtx_<M>_<m>`, CUDA | `aji_trt`'s sources + `AJI_TRTRTX`, + `aji_trt_common`; **only with `AJI_TRTRTX_ROOT`** |
| `aji_dml` | SHARED | `d3d12 d3d11 dxgi d3dcompiler` | **WIN32 only**; ORT/DirectML loaded at runtime, not import-linked |
| `aji_harness` | EXE | `aji`, `CUDA::cudart` | depends on `aji_trt` (and `aji_trtrtx`); prints wall-clock create/configure/first-frame `timing:` lines; `--async-build` = the player's build mode (poll + reconfigure) |
| `aji_harness_dml` | EXE | `aji`, `d3d11` | **WIN32 only** |
| `aji_encode` | EXE | `aji`, CUDA, ffmpeg libs | **conditional** — see below |
| `aji_kernel_test` | EXE | `CUDA::cudart` | kernel unit tests |
| `aji_image` | EXE | `aji` | image-ABI CLI (PNM in/out); the reference consumer of `aji_image.h` |
| `aji_tiling_test` | EXE | — | CPU-only checks of `core/tiling` and the fp16/bf16 conversions |
| `aji_held_test` | EXE | — | CPU-only checks of `core/held`, the held-frame reuse rule, and its C ABI (`aji_held_*`) over a stub context |
| `aji_native` | OBJECT | `CUDA::cudart` | `src/native`: kernels, ONNX lowering, executor; linked into `aji_trt` and `aji_trtrtx` |
| `aji_native_host` | OBJECT | — | the host-only sources of `src/native` (ONNX reader, lowering, `layout`, packing; `AJI_NATIVE_HOST_SOURCES`, also compiled into `aji_native`), built with or without CUDA |
| `aji_native_pack_test` | EXE | — | CPU-only check of the native weight packing and fp16 conversion |
| `aji_native_lower_test` | EXE | — | CPU-only check of the ONNX reader and lowering on in-memory graphs |
| `aji_native_layout_test` | EXE | — | CPU-only check of `native/layout`: size checks, padding, feature-buffer reuse |
| `aji_vk_shaders` | OBJECT | — | **`AJI_VULKAN` only**: the embedded SPIR-V table (`vk/spirv.h`) |
| `aji_vk_spirv_test` | EXE | — | **`AJI_VULKAN` only**: CPU check that each embedded module is well-formed SPIR-V with its family's capabilities |
| `aji_vk_core` | OBJECT | — (Vulkan loaded at run time) | **`AJI_VULKAN` only**: `src/vk` device, kernels, executor |
| `aji_vk` | SHARED | `d3d12 d3d11 dxgi d3dcompiler` | **`AJI_VULKAN` + WIN32 only**: `aji_dml.cpp` with `AJI_VULKAN`, model steps on the Vulkan executor |
| `aji_vk_model_test` | EXE | — | **`AJI_VULKAN` only**, by hand: one model on the Vulkan executor against the fp32 CPU reference (and the CUDA executor with `AJI_CUDA`); `rife <onnx> W H`: Vulkan RIFE, timed, against CUDA native RIFE; `temporal <onnx> W H`: TemporalFix (whole window and cached), against CUDA native; `image <onnx> W H`: an image model against CUDA native; `gemm` / `attn`: kernel benchmarks |
| `aji_vk_gconv_test` | EXE | — | **`AJI_VULKAN` only** (`ctest -L gpu`): the general conv's modes against a CPU reference, each tile variant |
| `aji_native_conv_test` | EXE | `aji_native`, `CUDA::cudart` | every native kernel against a CPU reference (sm_80+, skips below) |
| `aji_native_bench` | EXE | `aji_native`, `CUDA::cudart` | **EXCLUDE_FROM_ALL**; whole-model native timing on `experimental/native-span` data |

The self-contained tests are registered with CTest under two labels: `ctest --test-dir build -L
cpu` (`aji_half_test`, `aji_tiling_test`, `aji_runtime_packs_test`, `aji_rife_conf_test`,
`aji_held_test`, `aji_native_pack_test`, `aji_native_lower_test`, `aji_native_layout_test`, and
with `AJI_CUDA` `aji_native_onnx_test`, with `AJI_VULKAN` `aji_vk_spirv_test`; what CI runs) and `-L gpu` (`aji_pre_fusion_test`, `aji_msb444_test`,
`aji_frame_diff_test`, `aji_native_conv_test`, `aji_native_gconv_test`, `aji_native_tfm_test`;
`AJI_CUDA` only; needs an
NVIDIA GPU; on Windows also `aji_frame_diff_dml_test`, which runs on any DirectML GPU from
`dml-test/`, beside copies of `aji.dll`, `aji_dml.dll`, ONNX Runtime and DirectML). A new test
that needs no GPU, models or fixtures goes in the `cpu` list at the end of `CMakeLists.txt`, and
CI runs it on both platforms.

**`aji_encode` is skipped silently** unless its ffmpeg dependency resolves. On Windows that
means `AJI_FFMPEG_ROOT` must be set; on Linux, pkg-config must find `libavformat libavcodec
libavutil libavfilter libswscale`. The configure output says which:

```
-- AJI_FFMPEG_ROOT not set; skipping aji_encode target
-- ffmpeg/libav not found via pkg-config; skipping aji_encode target
```

If a release build comes out without `aji_encode.exe`, this is why.

### The CUDA-architecture trap

`CMakeLists.txt:9-17` sets a full arch list **only** `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 100-real 120-real 120-virtual)
```

> **That fallback never fires.** CMake pre-defines `CMAKE_CUDA_ARCHITECTURES` during
> `project()`, so omitting `-DCMAKE_CUDA_ARCHITECTURES=...` yields a **single default arch**
> (e.g. `sm_75`) that runs on one GPU generation and dies on every other. The Windows release
> script, `tools/build-windows-test.ps1` and the Linux build action therefore pass the full list
> explicitly. Never drop it from a build that ships.

For local dev on one machine, `-DCMAKE_CUDA_ARCHITECTURES=120` (or `native`) is much faster to
compile — the release list builds eight architectures.

## CI

`.github/workflows/ci.yml` runs on pull requests, and on pushes to `main`, that touch `src/`,
`include/`, `sanity/`, `tools/`, `ci/`, `CMakeLists.txt`, `.github/actions/` or `ci.yml` itself
(doc-only changes build nothing). It publishes nothing. Dispatch it on a branch to get Windows
test binaries.

| Job | Runner | Builds | Runs |
|---|---|---|---|
| `windows` | `windows-2022` + MSVC | `tools/build-windows-test.ps1`: every arch, `AJI_BUILD_ANALYSIS=ON`; no `aji_encode` or `aji_trtrtx` (no FFmpeg / TensorRT-RTX SDK there) | `ctest -L cpu`, `sanity/half_isa_check.py`, `tools/test-stream-policy-dispatch.py`, `tools/test-release-configuration.py`, `tools/test-release-packaging.ps1`; uploads `aji-windows-test` (7 days) |
| `linux` | `ubuntu-latest`, in the build image | `.github/actions/linux-build`: exactly the release build, `aji_encode` included | `ctest -L cpu` |

Hosted runners have no GPU, so the GPU tests (`ctest -L gpu`, `aji_kernel_test`, the harnesses
and the md5 regression gate below) are compiled but not run: run them locally. The SDKs are
cached: Windows `build-deps/` under a key from `ci/windows/dependencies.json`, Linux
`/opt/aji-deps` under a key from the pins in the action. A cold run downloads ~2 GB of
TensorRT per platform.

`build-linux.yml` (the Linux release) runs the same action and `ctest -L cpu` before it
publishes, so a Linux release can't ship a build that fails them.

## Where the artifacts go

| Platform | Built by | Asset |
|---|---|---|
| Linux | `.github/workflows/build-linux.yml`, dispatched with `release_tag` (no default; refuses a release that already has Linux assets unless `replace_assets=true`) | `aji-linux-x64.tar.zst`, `aji-encode-linux-x64.tar.zst` (aji_encode + libav, for VideoJaNai) |
| Windows | **locally, by hand** (`docs/BUILD-WINDOWS.md`) | `aji-windows-x64.zip`, `aji-trtrtx-windows-x64.zip` (opt-in: `aji_trtrtx.dll` + TensorRT-RTX runtime, ~91 MB), `aji-trtrtx-backend-windows-x64.zip` (`aji_trtrtx.dll` alone, for apps that take the runtime from a shared pack) |

All are attached to the **same release tag**, because the consumer derives every URL from one
`AjiVersion` constant. `the-database/mpv-AnimeJaNai`'s assembler downloads them in `InstallAji`
and extracts them flat into `animejanai/inference/`. It does not take the TensorRT-RTX asset
for now (mpv's background builds start each play with a short warm-up passthrough on
TensorRT-RTX); VideoJaNai, MangaJaNaiConverterGui and yosh are the intended takers. Without
`aji_trtrtx.dll`, `backend=TensorRT-RTX` falls back to TensorRT.

Once all four assets are on the tag, `gh workflow run notify-downstream.yml -f tag=<tag>` asks
MangaJaNaiCLI, VideoJaNai, MangaJaNaiGUI and mpv-AnimeJaNai to open PRs that pin it (their
`bump-pins.yml`; needs the `JANAI_BUMP_TOKEN` secret). It refuses while an asset is missing.

**Shared runtime packs** ship from this repo too, under their own tag (`runtime-<n>`, `TAG` in
`scripts/runtime_packs.py`), published by `.github/workflows/runtime-packs.yml` (manual dispatch):
TensorRT split into `trt-runtime` + one builder pack per GPU generation, the TensorRT-RTX runtime,
ONNX Runtime + DirectML, and index entries for the MangaJaNai model archives. Each platform's
`packs-<platform>.json` (format 1: version, kind, SHA-256 and per-file SHA-256s) is what apps pin
by SHA-256 and install with [aji-runtime](https://github.com/the-database/aji-runtime) into the
per-user pack store; libaji then loads from it through `aji-runtime.json`. To change a pinned
version (a new TensorRT, say):

1. Bump it and `TAG` (`runtime-<n+1>`).
2. Run the script locally (`--platform windows|linux --out DIR`).
3. Run the workflow. Its last job asks the apps that pin the packs for bump PRs (an
   `upstream-release` dispatch, `source=runtime-packs`; MangaJaNaiCLI, MangaJaNai GUI, VideoJaNai and mpv-AnimeJaNai). Add an app there once
   its `bump_pins.py` knows that source.

A store folder (`packs/<name>/<version>`) never changes, so a name/version must always mean the same
files. The script compares each pack's files with the earlier `runtime-<k>` releases' indexes
(it downloads them; `--previous-index` for local runs): a pack whose files didn't change keeps its
earlier version, archive, URL and SHA-256 and isn't uploaded again, so the stores keep it and apps
don't download it again. A changed pack whose version an earlier release used gets `-r2`, `-r3`, ...
Apps must take each pack's `url` from the index, not build it from the tag. A libaji release
built against the new TensorRT usually goes with it. (`runtime-1` is MangaJaNaiCLI's older,
CLI-only release.)

RIFE model weights ship from this repo too, under a separate tag
(`RifeModelsVersion`, currently `models-rife-fp16-2`), converted with
`tools/convert_rife_fp16.py`. Set 2 differs from set 1 only in naming the input's H/W dims,
which the DirectML backend needs to pin them (~10% faster RIFE there); outputs are
bit-identical, and TensorRT reuses existing engines (cache names come from the file name).

## Shared runtime packs (`aji-runtime.json`)

Apps can keep the TensorRT / TensorRT-RTX runtimes in a per-user store shared between apps
(design: `plans/shared-runtime-packs.md` in `the-database/animejan.ai`). The app's installer then
writes `aji-runtime.json` beside `aji.dll` / `libaji.so`:

```json
{ "format": 1,
  "trt": "C:\\...\\packs\\trt-runtime\\11.4.0.106-cuda13.4",
  "trt_builder": ["C:\\...\\packs\\trt-sm120\\11.4.0.106"],
  "trtrtx": "C:\\...\\packs\\trtrtx-runtime\\1.7.1.116-cuda13.4" }
```

`core/runtime_packs` reads it once per library. Where libaji's folder is read-only (an AppImage), the
`AJI_RUNTIME_JSON` environment variable names the file instead. Keys are optional, and relative
paths are relative to the file; unknown keys, like aji-runtime's `models` for the app, are skipped; any other `format` is an
error):

- Before loading `aji_trt`, both dispatchers load cudart + nvinfer from `trt` by full path (Linux:
  `RTLD_GLOBAL`), so the backend's imports bind to them. Before `aji_trtrtx`: cudart (from
  `trtrtx`, else `trt`) + the TensorRT-RTX runtime. The names come from the SDKs at configure time
  (`AJI_PACK_*` in `CMakeLists.txt`). `aji_dml` ignores the file.
- `aji_trt` runs `trt`'s trtexec, **ignoring the host's `trtexec`**, with the `trt_builder`
  folders on its search path: Windows prepends them to the child's `PATH` (environment block);
  Linux appends `LD_LIBRARY_PATH='<trt>:<builders>'` after the host's `trtexec_env`. nvinfer loads
  the per-GPU builder resource by file name, which preloading can't satisfy on Linux, and which
  only builds need.
- A file that names a missing pack makes `aji_trt` / `aji_trtrtx` fail to load with a clear error.
  Without the file, everything loads from libaji's own folder as before.

Tests: `aji_runtime_packs_test` (parser, CPU-only). Hosts' own executables (`aji_harness`,
`aji_encode`) still import cudart themselves.

## The ABI contract

`include/aji.h`'s `AJI_API_VERSION` is shared with `video/filter/aji.h` in the mpv fork —
**the two files must agree**. When the ABI changes:

1. update both headers,
2. rebuild and release both the engine and the mpv builds,
3. bump `AjiVersion` **and** `MpvForkVersion`/`MpvForkLinuxVersion` in the assembler.

The filter lives on the mpv fork's `master` (aji ABI v9). The old standalone `vf-animejanai`
branch is stale (ABI v4) and must not be used.

`include/aji_image.h` has its own `AJI_IMAGE_API_VERSION`, independent of the video ABI: an
image-ABI change never touches the mpv fork. Its consumers (e.g. yosh's upscaling add-on) load
`aji.dll` at runtime and check the version through `aji_img_create`.

**Regression gate for core changes:** the video harnesses must stay byte-identical — run
`aji_harness` / `aji_harness_dml` (slots 1 and 3, `--output`) before and after, with cached
*and* freshly built engines, and compare md5s. Image path: `aji_tiling_test`, then `aji_image`
on a test page with both backends; for a tiling change, also run each CUDA backend with and without
`AJI_IMG_CPU_TILING=1` and compare md5s. They must match, on a set that covers 1- and 3-channel
images and engines, fp32/fp16/bf16 engine I/O, and images smaller than a tile.

## Temporal models

A chain may hold several temporal (multi-frame) models, e.g. a temporal upscale followed by
vs_temporalfix. Each is a stage with its own history ring: stage 0 is filled by `aji_ingest`,
later stages are computed on demand (once per frame) from the previous stage's ring inside
`aji_infer_seq`. `aji_temporal_radius` returns the summed radius, so consumers need no
changes. Stage 0's ring holds 2R+5 frames (the `aji.h` ingest-ahead contract); on DirectML the
gather shader's 5-bit slot indices cap that at a 27-frame combined window.

Two chain keys add or remove steps without listing them as models (`parse_chain` in
`src/aji_conf.cpp`; the backends only ever see the resulting `models` list):

- `chain_N_upscale=no` (default yes) skips the chain's `chain_N_model_M_*` models; they stay in
  the conf so the Manager can turn upscaling back on.
- `chain_N_temporal_fix_model=<onnx name>` (+ `chain_N_temporal_fix_scene_threshold`, default
  0.150) appends a vs_temporalfix model after the upscale models, or alone when `upscale=no`.
  The Manager picks the file (the newest `temporalfix_s<k>_v*`); the engine just loads it.
  `chain_N_temporal_fix_before_upscale=yes` (default no) puts it before the upscale models
  instead, at the source resolution: roughly 2-3x cheaper for a 2x upscale, but it can't steady
  shimmer the upscaler itself adds.

`regress/temporal_regress.sh` compares a reference build with a candidate: bit-exact output
and timing for chains without stacking, plus stacked-chain correctness. Run it before
releasing any change to the temporal or chain code (environment variables in its header).

## RIFE factors in `aji_encode`

Any factor of 1 or more works, fractions included (`5/2` for 23.976 -> 59.94 fps, `3/2`); it
is reduced first (`4/2` runs as `2/1`). The schedule is mpv's `vf_animejanai` accumulator:
output frames sit on a grid `den/num` of a source frame apart from the first frame, source
frames are put out where the grid lands on them, and the output time base is the source's
divided by `num`, so a constant-rate source gets exact, evenly spaced timestamps. The stream
ends at the last grid point at or before the last source frame: n frames give
`floor((n-1)*num/den) + 1`, which for an integer factor is the old `(n-1)*F + 1`. The comment
above `rife_step` in `src/encode.c` has the details. Unlike mpv, `aji_encode` has no 8x cap.
Held-frame reuse (with `--reuse-held-frames-rife`) leaves the grid alone: a pair that ends on a
held frame repeats its first frame at each of its grid points instead of interpolating ("RIFE
chains" under held-frame reuse).
A factor below 1 fails before any engine is built: `aji_rife_factor_for_config` (dispatcher
only) reads the chain's factor from the conf without configuring.

## Untagged colour in `aji_encode`

A YUV source without colour tags is treated the way mpv treats it (`mp_image_params_guess_csp`,
which runs before `vf_animejanai`): BT.601 when the frame is narrower than 1280 and at most 576
lines tall, else BT.709; limited range; primaries and transfer guessed as mpv does. Only missing
fields are filled, and the output is tagged with all four, since libaji writes the upscaled
frames in the source's matrix and range. `guess_colorimetry` in `src/encode.c` has the rule.

## Frame formats in `aji_encode`

A frame reaches the encoder in one of two ways (`emit_output`). It goes zero-copy into NVENC
only if it is already a CUDA frame (or DirectML's D3D11 texture) in exactly the 4:2:0 format the
output asks for (`nvenc_takes`). libaji's output always is. In passthrough (no chain active), the
frame is the decoder's own and only sometimes qualifies: a CPU-decoded frame is in host memory,
and an NVDEC frame may have another bit depth or be 4:2:2 / 4:4:4. Every other frame is
downloaded if need be, converted (and resized) with swscale, and NVENC uploads it. The first
frame decides, since the encoder is opened for it; a zero-copy encode that later meets a frame
of another format or size fails with a message instead of feeding NVENC a frame it was not
opened for. The final resize (`--final-resize-*`, `--square-pixels after`) is Spline36 like every
other resize: on the zero-copy path `aji_resize_frame` (TensorRT backends; scale_cuda only if it
fails), else swscale's `SWS_LANCZOS`, the closest of its filters to Spline36 on anime video
(scale_cuda's bicubic has no prefilter when downscaling: ~50 dB from Spline36 against libaji's
~69). `tools/test-encode-formats.py` (NVIDIA GPU) runs every source layout (8/10-bit
4:2:0, 4:2:2, 4:4:4, odd width) through NVDEC and the CPU decoder, NVENC and the CPU encoders,
both bit depths, `--final-resize-height` and `--deinterlace`, in passthrough and (given a model)
upscaling, and checks the output format, frame count and PSNR against the source.

## Held-frame reuse (`aji_held_*`, used by `aji_encode`)

Players read `[global] reuse_held_frames` (on by default; `no`/`off`/`false`/`0` turns it off)
through `aji_reuse_held_frames_for_config(conf_path)` (0.30.0, additive); aji_encode ignores the
key and keeps its own switch.

On by default for the plain upscale chain (`--no-reuse-held-frames` turns it off, for debugging
and A/B tests; `--reuse-held-frames` is still accepted); RIFE chains need
`--reuse-held-frames-rife` (see "RIFE chains" below). A frame that is the same picture as the last frame
that was upscaled (the anchor) skips inference: the anchor's output is encoded again under the
frame's timestamp. Anime holds drawings for 2-3 frames; slides, screen recordings and still
shots repeat frames exactly. The comparison is `aji_frame_diff` (every backend; additive
export): D is the largest 8x8-tile mean |diff| over all planes, in
8-bit levels, and the fade measure the largest signed mean of a 4x4 luma region grid. Frame N is
reused when D(anchor, N) <= 2.5 and fade <= 0.15, fewer than 24 frames in a row were reused, the
two frames after it exist (the last two frames are always upscaled) and are undamaged, and a
two-frame look-ahead sees no motion starting: D from the anchor crossing 2.5 while N is already
close to the later frame, or growing past 1.5x (floor 0.5). The thresholds can be overridden
(`--reuse-held-threshold`, `--reuse-held-fade-threshold`; tuning/debug only). Plain upscale
chains and RIFE chains without a temporal model use it (see "RIFE chains" below), on CUDA and
DirectML alike; temporal chains (TemporalFix or a temporal upscale model, with or without RIFE),
passthrough and `--preview-session` encode every frame, with a log line saying why.

Evidence (RTX 5090): full episodes reuse 16-57% (a 1999 DVD 16%, 1080p web 30-57%); frame
counts and timestamps match an encode without reuse; with a lossless NVENC every upscaled frame
is bit-identical and reused frames are 48.8 dB or more from a fresh upscale; a blind A/B of the
closest calls (clouds, a title-card fade, smoke, a blurred concert background) showed no stall;
synthetic slow pans (0.1-1 px a frame, sharp and blurred) and drifting patches reuse nothing
inside the motion when clean. Speed: where inference is the limit (`-preset p1` on the 5090) wall time fell
24-31%, about in proportion to the reuse rate; at VideoJaNai's default p7 on a fast GPU NVENC is
the limit and wall time barely changes (CUDA: not at all; DirectML: 1-4% faster). Known limits:
slow motion buried in codec noise can still be held for single frames (a sub-pixel shift; with x264 noise added, a blurred 0.1 px a
frame pan held 26 of 120 frames and a 0.25 px drifting patch 5, each alone); a faint one-pixel change at
half contrast or less (an anti-aliased caret in a screen recording) can be held for up to 24
frames.

**The engine (`aji_held_*`).** The rule is libaji's, in the dispatcher: additive exports of
`aji.dll` / `libaji.so` only (`AJI_API_VERSION` stays 9; `aji_held_version()` is the rule
version, 2), so mpv and other callers can share it. `src/core/held.{h,cpp}` (`HeldRule`,
backend-neutral, with the rule's reasoning and evidence) sits behind a comparison functor;
`src/aji_held.cpp` is the C ABI and binds it to the dispatcher's own `aji_frame_diff_marked`
(the later of the two frames' marks; TensorRT gets the plain call on the caller's stream), so it
reaches every backend the dispatcher loads and follows a TensorRT-RTX context that fell back to
TensorRT. Backends export nothing for it. A handle owns the rule state (the anchor's descriptor,
the run, D(anchor, N+1) cached from the last look-ahead), makes the comparisons in a fixed order
(anchor-N unless cached, anchor-N+1, N-N+1, anchor-N+2, N-N+2, with the rule's short circuits),
counts them and writes the trace. The caller owns every frame, the queue and the anchor copy: it
passes N, N+1, N+2 (NULL at the end of the stream) to `aji_held_decide`; on UPSCALE (which drops
the anchor) it upscales N and hands a copy of N to `aji_held_set_anchor`, on REUSE it emits the
anchor's output; `aji_held_set_anchor` starts a new run. One handle is one decision point;
`aji_held_reset` is for a seek, a discontinuity or a change of device or context. A change of
size or format without one is no error: an anchor unlike N counts as none, and a look-ahead
frame unlike N means an upscale without comparisons. Frames carry `flags`
(`AJI_HELD_FRAME_DAMAGED`) and `reserved[3]`, so `aji_held_frame` can grow: unknown flags or
nonzero reserved words are refused with `AJI_ERR`. A comparison error (`decision.error`) turns
the engine off until a reset. `aji_held_create` returns NULL for a backend without
`aji_frame_diff` (an `aji_dml.dll` older than its port beside a new `aji.dll`): `aji_encode` then
logs `held frames: not reused (the backend can't compare frames)` and encodes every frame.
`aji_held_test` (`ctest -L cpu`) drives the rule with synthetic comparisons: every veto alone,
the 0.5 floor, damage, the look-ahead's cached D and fade, size and format changes, the run after
a new anchor, the comparison order and counts, errors, reset, and the trace text (with cached
decisions) against a golden string. It also runs `src/aji_held.cpp` over a stub context (mark
selection, `cu_stream`, argument and flag checks, partial `get_stats`).

**In `aji_encode`.** `src/encode.c` keeps what touches frames (the comment above `push_reuse`):
frames are decided `ring_depth + 2` frames late and compared on a non-blocking stream that waits
for each frame's arrival event (queued for N, N+1 and N+2 before each decision), because libav's
decoder copies run on the legacy default stream behind the queued inference. Only three steps
differ per backend (`reuse_arrived`, the event waits in `reuse_decide`, `reuse_anchor_copy`).
`reuse_decide` only decides; what a decision does is the active path's (`reuse_consume`: the
ring for a plain chain, the RIFE paths below), so every path shares the queue and the rule.
Moving the rule into the engine changed nothing: the four traces of the evidence clips below
(CUDA and DirectML) are byte-identical to `aji_encode`'s own from before, with the same counts,
frame counts and timestamps, and CUDA at `-preset p1` is as fast as before (median wall time of
seven alternating runs 23.37 / 23.31 s and 18.16 / 18.16 s; 0.1 ms a comparison either way).

**DirectML.** `aji_frame_diff` in `aji_dml` is `cs_frame_diff` (`dml_shaders.h`), a port of
`k_tile_diff` that gives **bit-identical** results: integer sums, the per-tile `scale / n` table
computed on the host exactly as the CUDA kernel computes it (`src/frame_diff.h`, shared by both
backends with the region finish), and the 64-bit region sums carried as two 32-bit halves
(`cs_5_0` has no 64-bit atomics). It takes NV12, P010 and RGB10A2 (R, G, B as three components,
divided by 4). It runs on a high-priority D3D12 COMPUTE queue of its own, which waits on the
shared fence, copies both frames' planes to staging buffers and runs the kernel; then the CPU waits
for that queue alone, so a compare never waits for queued inference. Textures are read through
implicit promotion (COMMON -> COPY_SOURCE, decaying back), never with explicit transitions that
could race the inference queue. The opened textures are cached with a reference to the D3D11
texture: a cache keyed by address alone (like `open_shared`'s) misreads once a freed texture's
address comes back as another's. Frames may sit in larger textures (decoder surfaces are padded,
1920x1088 for 1080p): every DirectML read of a frame (`copy_plane`) copies a box of the frame's
plane size, never the whole subresource. The staging buffers carry slack past each plane (4 KiB,
or 8 rows if more): drivers widen the kernel's dword loads past the plane's end (a page fault on
the UHD 770, a hang on the 5090, where nothing else was mapped after the buffer).

**Marks.** libav uploads frames on the D3D11 immediate context, and NVENC's zero-copy encodes of
D3D11 frames are on it too. The plain `aji_frame_diff` signals the fence on that context at call
time, so it waits for every encode queued so far: ~15-18 ms a compare at p7 4K on the RTX 5090,
which made reuse 6-8% slower there. So `aji_encode` uses marks (additive exports, no API_VERSION
bump): `aji_d3d11_mark` signals the fence with the context's next value, flushes and returns it
(0 on TensorRT), and `aji_frame_diff_marked(..., mark)` waits only for that value (0: like the plain
call; a value the context never issued: `AJI_ERR`; the dispatcher falls back to the plain call on
backends without marks). Every frame, as it arrives, is copied on the context into a texture of
the anchor's pool (`reuse_arrived`) and then marked; the copy becomes the anchor if the frame is
upscaled and is dropped otherwise, so the anchor's mark is its frame's. A compare waits for the
later of its two frames' marks; frames are decided `ring_depth + 2` late, so those have passed and
nothing queued since is waited for. Copying the anchor at decision time, as on CUDA, would mark it
behind every encode queued by then; keeping a reference to the uploaded frame instead of a copy
would have the compare queue read a texture the inference queue may still be reading. The plain
call stays for other callers (mpv).

`aji_frame_diff_dml_test` checks the cases of `aji_frame_diff_test` on DirectML, each both plain
and marked, plus the marks themselves (one increasing counter, unknown marks refused, a marked
compare sees writes before its mark while copies are queued after it, and prints how long it
took next to a plain one) and frames in padded textures (equal to exact-size ones bit for bit);
with CUDA present, every result must be bit-identical to `kernels.cu`'s. `--raw FILE W H
nv12|p010` runs that cross-check on real frames.

Evidence (first 2878 frames of tongari11 and testclip, 2x Balanced to 4K): the DirectML and
CUDA traces are byte-identical (238 and 904 reused); frame counts and timestamps match
`--no-reuse-held-frames`. On the RTX 5090 a compare takes 1.5-2.7 ms at NVENC p7 (15-18 ms
without marks) and 0.7-1.1 ms at p1 (CUDA: ~0.1 ms). Wall time, RTX 5090: 1% and 4% faster at
NVENC p7 (NVENC the limit; 6-8% slower without marks), 7% and 23% faster at p1. Intel UHD 770
(Performance model, QSV as VideoJaNai sets it, 480 frames, 15% and 32% reused): 12% and 26%
faster (11% and 24% without marks), same decisions as on the 5090; a compare takes ~6 ms there
(~14 ms without marks).
`AJI_REUSE_TRACE=<file>` (`aji_encode` hands it to `aji_held_set_trace`) writes every comparison
(frame, pair, D, the 16 region means) and decision (frame, `reuse` or `upscale`) as CSV. At the
end `aji_encode` logs `held frames: reused N of M (P%)` and `held frames: K comparisons, X ms each
on average` (from `aji_held_get_stats`).

**RIFE chains.** Opt-in for now: `--reuse-held-frames-rife` (off by default until the user has
watched an A/B of it; without it a RIFE chain logs `held frames: not reused (RIFE chain;
--reuse-held-frames-rife turns it on)`). RIFE only, upscale then RIFE, and RIFE first
(VideoJaNai's default; with or without a pre-resize), at any factor. The decisions are made on the source frames by the same
queue and rule, before any pre-resize, so the trace is byte-identical to the plain chain's on
the same clip. If N is held, N-1 shows the same anchor's picture (the anchor changes only on an
upscale, and a damaged frame drops it), so the pair (N-1, N) has nothing to interpolate: every
grid point in [N-1, N) puts out N-1's frame under its own timestamp, and the pair gets no RIFE
call, no scene detection, no pre-resize and no upscale. The grid is untouched, so the frame
count and timestamps are those of `--no-reuse-held-frames`. Per path (`rife_up_consume`,
`rife_first_consume`, `rife_only_consume` in `src/encode.c`): upscale then RIFE takes the
anchor's output for N; RIFE first interpolates the pair after a held frame from the anchor's
work-size frame (`anchor_work`, or the anchor's source copy without a pre-resize) and upscales
the anchor once, on first need (with a fractional factor the anchor may sit off the grid); RIFE
only puts the frames out as they are, so only the in-betweens of a held pair change (they repeat
N-1). The log adds `rife: H held pairs not interpolated (I in-betweens repeat their first
frame)` after the scene duplicates. RIFE only compares frames in the output format (they are
converted on upload): an 8-bit source at 10-bit output converts exactly (x4), so its trace is
still the plain chain's; a 10-bit source at 8-bit output compares the rounded frames, so its
decisions can differ slightly. Temporal chains stay off (`held frames: not reused (the chain is
temporal)`): reuse ahead of a temporal stage isn't bit-neutral, and deciding on a temporal
stage's output needs new libaji exports; the simulation put their saving at 1-9%.
`tools/test-encode-reuse-chains.py` is the regression test (every RIFE chain and factor against
`--no-reuse-held-frames`: packets and pts, the trace against the plain chain's, with
`--lossless` every frame no held frame touches bit-identical and the rest above a PSNR floor;
`--damage`: damaged frames and the frames after them are upscaled).

Evidence (RTX 5090; tongari11 and testclip, 2x Balanced to 4K, RIFE 4.22): on 2878 frames at 2x
and 5/2, CUDA and DirectML, every RIFE chain's packets and timestamps match
`--no-reuse-held-frames` and its trace is the plain chain's (238 and 904 held), and CUDA's traces
equal DirectML's. Lossless, at 2, 5/2, 3 and 3/2: every frame no held frame touches is
bit-identical; held frames are the plain chain's reused frames (48.7 dB and up from a fresh
upscale); a held pair's in-betweens are 45-49 dB from RIFE's (CUDA 46 and up, DirectML 45; on
testclip RIFE's own in-between of two near-identical frames is about as far, 48.6 dB and up,
from the frame before it, which is what reuse repeats); in-betweens after a held frame, interpolated from the anchor, 51.6 dB and up
(RIFE first). Every touched frame on CUDA is 46.08 dB or more.
Known defect, not reuse's: DirectML RIFE at 3840x2160 (upscale then RIFE) puts out a warped
in-between even for two identical frames (21 dB from CUDA's), so there the no-reuse encode is the
wrong one. Wall time (median of seven alternating runs, reuse against `--no-reuse-held-frames`,
2878 frames at 2x, 8% and 31% held): RTX 5090 CUDA at NVENC p1, RIFE first 7.3% and 28.0% faster,
upscale then RIFE 7.7% and 29.2%; at p7 (NVENC the limit) 0.6% and 1.9%, and 0.9% and 3.0%;
DirectML at p1, RIFE first, 7.2% and 28.2%. Intel UHD 770 (DirectML, the Performance model, QSV,
480 frames, 15% and 32% held), RIFE first: 13.7% and 30.2%.

## Damaged input in `aji_encode`

`--on-damage fail|continue` (default `fail`). Damage is detected the same way for NVDEC and the
CPU decoder: the decoder rejecting data (`AVERROR_INVALIDDATA` from send/receive), a frame
flagged `decode_error_flags` / `AV_FRAME_FLAG_CORRUPT`, a video packet the demuxer flags
`AV_PKT_FLAG_CORRUPT` (MPEG-TS continuity errors, a sample cut short by the end of the file) or
`AVERROR_INVALIDDATA` from `av_read_frame`, and errors the video decoder or the demuxer *log*
(`av_log_cb` counts them): HEVC drops a slice or a frame without references, and Matroska resyncs
past a broken element, with nothing but a log line. Other errors (ENOMEM, a device failure, the
encoder) stay fatal. Errors before the decoder's first frame are not damage (a recording that
starts mid-GOP). NVDEC decodes macroblocks in hardware and libavcodec (9.0.2) never asks it for
its decode status, so damage it conceals there goes unseen; `--decoder cpu` sees it.

Each damaged spot prints one stdout line, next to `PROGRESS`, in both modes (in `--progress
json`, `{"damage":{"t":...,"frame":...}}`):

```
DAMAGE t=<seconds since the video stream's start, 3 decimals> frame=<source frame index | -1>
```

Finds less than a second apart are one spot (an error, its concealed frame and the frames left
without references are one line); `frame` is the first frame of the spot the decoder put out, or
-1 when the spot starts with a packet that produced none. A packet-level find waits a few frames
for a flagged frame to report instead (frame threads report errors late), so its time can be off
by about that much. `fail` stops at the first spot with exit status **3** and says where on stderr
(`damaged video at 1:23.4 (frame 2001)`); `continue` keeps concealed frames, drops what the
decoder rejects and prints the total. A read error that isn't EOF (or `pb->error`) fails with
exit status 1 in both modes and in `--preview-session`; damage never stops a preview session (it
has no `--on-damage`), which prints no `DAMAGE` lines. `tools/test-encode-damage.py` is the
regression test (Windows + NVIDIA GPU; it simulates the read error with a byte-range lock).

## HDR in `aji_encode`

Colour tags (BT.2020 matrix and primaries, PQ or HLG transfer) carry over like any others, and
a 10-bit output (`--pix-fmt yuv420p10`, the default) keeps the depth. There is no tone mapping:
an 8-bit `--pix-fmt` on a PQ/HLG source writes an 8-bit HDR file (it bands), which is allowed but
gets a `warning:` line on stderr (`resolve_pixfmt`). The HDR10 static metadata
(mastering display, content light level) is taken from the first decoded frame (the decoder
puts the bitstream's SEI/OBUs or the container's side data on every frame) by `hdr_capture`,
since libaji's output frames don't carry it. It goes to the encoder as `decoded_side_data`
(x265, SVT-AV1, x264 and NVENC HEVC/AV1 write it into the bitstream; libavcodec copies it to the
stream's coded side data, which Matroska and MP4 write) and onto every frame sent (QSV and AMF
read it there; not tested, no such GPU here). Dolby Vision is dropped with a note: its RPU
describes the source encode. A profile 8.1 source plays as HDR10 from its base layer; profile 5
has no compatible base layer, so its colours come out wrong.

## Conventions

- Commits: author `the-database`, short imperative subject, no co-author trailers.
- `.gitignore` covers `build/`, `models/`, `*.engine`, and fixture binaries. The out-of-repo
  Windows build dirs (`build-win-release/` from `ci/windows/build-aji-release.bat`) are not in
  `.gitignore`; exclude them locally with `build-win-*/` in `.git/info/exclude`.
- ONNX models for the DirectML backend must be **opset ≤ 21**: the bundled ORT DirectML EP only
  registers `Conv`/`PReLU` kernels through opset 21, and a model exported at opset ≥ 22 silently
  falls back to the CPU EP (roughly 2000× slower per frame, which looks like a hang). Verify
  placement with `AJI_ORT_VERBOSE=1`.
