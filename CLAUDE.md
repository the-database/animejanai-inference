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
  release build. **There is no Windows CI**: `aji-windows-x64.zip` is built and uploaded by hand
  with the scripts in `ci/windows/`, against SDKs in an out-of-repo working area.
- **[`docs/BUILD-LINUX.md`](docs/BUILD-LINUX.md)** — the Linux/WSL build, the CI workflow, and
  the parity harness.

`README.md` is the public-facing overview. Its "Build (Linux)" section is still accurate; the
prose describing the engine as "Phase 0 spike scope" predates chain selection, RIFE, and the
DirectML backend.

## Architecture

```
vf_animejanai (mpv)  ──dlopen──►  aji.dll / libaji.so        (aji_dispatch.cpp — thin dispatcher)
                                        │ forwards over aji.h
                  ┌─────────────────────┼─────────────────────┐
            aji_trt.dll          aji_trtrtx.dll          aji_dml.dll        (Windows only)
          TensorRT + CUDA     TensorRT for RTX + CUDA   DirectML + D3D12
```

`[global] backend=` picks the library: `DirectML` (or the retired `NCNN`) -> `aji_dml`,
`TensorRT-RTX` (also `TensorRT_RTX`, `TRT-RTX`, `TRTRTX`; case-insensitive) -> `aji_trtrtx`,
anything else -> `aji_trt`. The image ABI selects it with `AJI_IMG_TENSORRT_RTX`.

**`aji_trtrtx` is `aji_trt`'s sources compiled with `AJI_TRTRTX`** against the TensorRT for
RTX SDK; every difference lives in `core/trt_engine` (plus a few `#ifdef`s for log wording and
runtime-cache saves). Engines build **in-process** through the builder API (about a second, no
`trtexec`; the trtexec-style settings string is parsed for its shape/optimization-level/
workspace flags), are cached as `....trtrtx-<ver>.gpu-<name-smN>.rtxengine` — deliberately not
`*.engine`, which `aji_trt`'s stale cleanup (including released builds) would delete — and run
with a TensorRT-RTX runtime cache saved beside each engine as `<engine>.jitcache`. Each backend's
stale cleanup only touches its own extension, so both share a model directory.

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
| `core/trt_engine` | engine cache naming, stale-engine cleanup, trtexec build specs/subprocesses, `AsyncBuild`, engine loading; under `AJI_TRTRTX` the in-process builder and the runtime (JIT) cache |
| `core/ort_dml` | ORT + DirectML loading (DirectML.dll before onnxruntime.dll) |
| `core/tiling` | the image path's tile cover / reflect pad / overlap blend on the host (backend-agnostic; DirectML uses it, and the CUDA backends with `AJI_IMG_CPU_TILING=1`). The band is a ring of rows; blend and flush are split across a few threads with `parallel_for`, per element, so the result doesn't depend on the split. **Seams:** overlaps default to 128 px and blend only across a 16 px band at their centre (`DEFAULT_OVERLAP`, `BLEND_WIDTH`). MangaJaNai's ESRGAN models generate screentone whose phase drifts near a tile's edge; the former 32 px linear ramp mixed the drifted patterns into visible interference patches. Torture test: `bwgradient_autolevels.png` (flat gradient, so the model invents all halftones) against a single-tile run. Also `even_origins`, the layout for `AJI_IMG_LOAD_EVEN_TILES` |
| `core/tiling_gpu` | the same tiling on the GPU for the CUDA backends (`upscale_gpu` in `aji_img_trt.cpp`): one upload, tile extract / blend / 8-bit flush kernels on the model's stream. **Bit-identical to `core/tiling`**: every multiply and add is `__fmul_rn`/`__fadd_rn` (never fused) in the host's order, with the host's fp16/bf16 rounding. Keep it that way, or the md5 gate below stops working |
| `core/half` | fp16/bf16 conversions |

Core functions take explicit paths/devices and return errors as strings; each front-end reports
them through its own context. **A change to engine caching or builds goes in the core**, so both
front-ends get it. The image front-end does inference only — model selection, grayscale
detection, resizes and alpha are the caller's (yosh / MangaJaNaiConverterGui keep that policy).

## Build system

CMake only (`cmake_minimum_required(VERSION 3.24)`, C11 / C++17, `LANGUAGES C CXX CUDA`).
There is no Cargo, meson, vcpkg, or conan manifest anywhere. The single `find_package` is
`CUDAToolkit REQUIRED`.

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
> (e.g. `sm_75`) that runs on one GPU generation and dies on every other. Both the release
> script and the Linux CI workflow therefore pass the full list explicitly. Never drop it from
> a build that ships.

For local dev on one machine, `-DCMAKE_CUDA_ARCHITECTURES=120` (or `native`) is much faster to
compile — the release list builds eight architectures.

## Where the artifacts go

| Platform | Built by | Asset |
|---|---|---|
| Linux | `.github/workflows/build-linux.yml` | `aji-linux-x64.tar.zst`, `aji-encode-linux-x64.tar.zst` (aji_encode + libav, for VideoJaNai) |
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
   `upstream-release` dispatch, `source=runtime-packs`; MangaJaNaiCLI and MangaJaNai GUI). Add an app there once
   its `bump_pins.py` knows that source.

Packs whose contents didn't change keep their version, so the stores keep them. A libaji release
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

`core/runtime_packs` reads it once per library (keys optional; relative paths are relative to the
file; unknown keys, like aji-runtime's `models` for the app, are skipped; any other `format` is an
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

## Conventions

- Commits: author `the-database`, short imperative subject, no co-author trailers.
- `.gitignore` covers `build/`, `models/`, `*.engine`, and fixture binaries. The out-of-repo
  Windows build dirs (`build-win-release/` from `ci/windows/build-aji-release.bat`) are not in
  `.gitignore`; exclude them locally with `build-win-*/` in `.git/info/exclude`.
- ONNX models for the DirectML backend must be **opset ≤ 21**: the bundled ORT DirectML EP only
  registers `Conv`/`PReLU` kernels through opset 21, and a model exported at opset ≥ 22 silently
  falls back to the CPU EP (roughly 2000× slower per frame, which looks like a hang). Verify
  placement with `AJI_ORT_VERBOSE=1`.
