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
  from this workstation, so this doc is the only record of how.
- **[`docs/BUILD-LINUX.md`](docs/BUILD-LINUX.md)** — the Linux/WSL build, the CI workflow, and
  the parity harness.

`README.md` is the public-facing overview. Its "Build (Linux)" section is still accurate; the
prose describing the engine as "Phase 0 spike scope" predates chain selection, RIFE, and the
DirectML backend.

## Architecture

```
vf_animejanai (mpv)  ──dlopen──►  aji.dll / libaji.so        (aji_dispatch.cpp — thin dispatcher)
                                        │ forwards over aji.h
                          ┌─────────────┴─────────────┐
                    aji_trt.dll                  aji_dml.dll        (Windows only)
                  TensorRT + CUDA               DirectML + D3D12
```

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

## Build system

CMake only (`cmake_minimum_required(VERSION 3.24)`, C11 / C++17, `LANGUAGES C CXX CUDA`).
There is no Cargo, meson, vcpkg, or conan manifest anywhere. The single `find_package` is
`CUDAToolkit REQUIRED`.

### Cache options

| Option | Type | Default (Windows) | Default (Linux) |
|---|---|---|---|
| `AJI_TRT_ROOT` | PATH | `""` | `$ENV{HOME}/sdk/tensorrt/usr` |
| `AJI_NVINFER` | STRING | `nvinfer_10` | `nvinfer` (plain `set`, not cached) |
| `AJI_FFMPEG_ROOT` | PATH | `""` | *(unused; pkg-config instead)* |
| `AJI_ORT_ROOT` | PATH | `""` | *(Windows only)* |
| `AJI_DML_ROOT` | PATH | `""` | *(Windows only)* |

The include/lib layout under `AJI_TRT_ROOT` differs by platform: Windows expects headers in
`<root>/include` and libs at `<root>` itself; Linux expects `<root>/include/x86_64-linux-gnu`
and `<root>/lib/x86_64-linux-gnu`.

> **`AJI_NVINFER` defaults to `nvinfer_10` but the project now builds against TensorRT 11.**
> Every current build passes `-DAJI_NVINFER=nvinfer_11` explicitly. Omitting it on Windows
> produces a link error for a library that is not there.

### Targets

| Target | Kind | Links | Notes |
|---|---|---|---|
| `aji` | SHARED | `${CMAKE_DL_LIBS}` | the dispatcher; `CXX_VISIBILITY_PRESET hidden`, PIC on |
| `aji_trt` | SHARED | `${AJI_NVINFER}`, `CUDA::cuda_driver`, `CUDA::cudart` | non-Windows adds `-Wl,-Bsymbolic` |
| `aji_dml` | SHARED | `d3d12 d3d11 dxgi d3dcompiler` | **WIN32 only**; ORT/DirectML loaded at runtime, not import-linked |
| `aji_harness` | EXE | `aji`, `CUDA::cudart` | depends on `aji_trt` |
| `aji_harness_dml` | EXE | `aji`, `d3d11` | **WIN32 only** |
| `aji_encode` | EXE | `aji`, CUDA, ffmpeg libs | **conditional** — see below |
| `aji_kernel_test` | EXE | `CUDA::cudart` | kernel unit tests |

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
| Linux | `.github/workflows/build-linux.yml` | `aji-linux-x64.tar.zst` |
| Windows | **locally, by hand** (`docs/BUILD-WINDOWS.md`) | `aji-windows-x64.zip` |

Both are attached to the **same release tag**, because the consumer derives both URLs from one
`AjiVersion` constant. `the-database/mpv-AnimeJaNai`'s assembler downloads them in `InstallAji`
and extracts them flat into `animejanai/inference/`.

RIFE model weights ship from this repo too, under a separate tag
(`RifeModelsVersion`, currently `models-rife-fp16-1`), converted with
`tools/convert_rife_fp16.py`.

## The ABI contract

`include/aji.h`'s `AJI_API_VERSION` is shared with `video/filter/aji.h` in the mpv fork —
**the two files must agree**. When the ABI changes:

1. update both headers,
2. rebuild and release both the engine and the mpv builds,
3. bump `AjiVersion` **and** `MpvForkVersion`/`MpvForkLinuxVersion` in the assembler.

The filter lives on the mpv fork's `master` (aji ABI v8). The old standalone `vf-animejanai`
branch is stale (ABI v4) and must not be used.

## Conventions

- Commits: author `the-database`, short imperative subject, no co-author trailers.
- `.gitignore` covers `build/`, `models/`, `*.engine`, and fixture binaries. The out-of-repo
  Windows build dirs are excluded locally via `.git/info/exclude`
  (`build-win/`, `build-win-trt11/`, `aji-build*/`) — note `build-win-release/` is in **neither**,
  so it shows up as untracked.
- ONNX models for the DirectML backend must be **opset ≤ 21**: the bundled ORT DirectML EP only
  registers `Conv`/`PReLU` kernels through opset 21, and a model exported at opset ≥ 22 silently
  falls back to the CPU EP (roughly 2000× slower per frame, which looks like a hang). Verify
  placement with `AJI_ORT_VERBOSE=1`.
