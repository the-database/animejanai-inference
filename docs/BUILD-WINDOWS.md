# Building libaji on Windows

**There is no Windows CI for this repo.** `.github/workflows/` contains exactly one workflow,
`build-linux.yml`. The `aji-windows-x64.zip` release asset that
`the-database/mpv-AnimeJaNai` downloads is built and uploaded **by hand**, with the two scripts
in [`ci/windows/`](../ci/windows/):

| Script | Does |
|---|---|
| `ci\windows\build-aji-release.bat` | the release build of this checkout (all architectures, clean) |
| `ci\windows\package-aji-release.ps1` | zips the seven engine artifacts into `aji-windows-x64.zip`, and `aji_trtrtx.dll` with the TensorRT-RTX runtime into `aji-trtrtx-windows-x64.zip` |

The SDKs they build against live **outside** the repo, in a working area. Linux build:
[`BUILD-LINUX.md`](BUILD-LINUX.md). Engine overview: [`../CLAUDE.md`](../CLAUDE.md).

### Paths in this document

| Placeholder | Means |
|---|---|
| `%AJI_WIN%` | the SDK working area. The scripts read the `AJI_WIN` environment variable and default to `%USERPROFILE%\aji-win`. |
| `%REPO%` | your checkout of this repository |

## Target layout

The build script passes every dependency path **explicitly and absolutely**, so the layout is
part of the contract:

```
%AJI_WIN%\
├── cuda134\                       CUDA 13.4 from NVIDIA's redistributable archives (see 1.)
│   ├── bin\                       nvcc.exe, ptxas.exe, ...
│   ├── include\  lib\x64\  nvvm\
├── trt114\                        TensorRT 11.4.0.106  <- the release TRT root (see 2.)
│   ├── include\                   NvInfer*.h + impl\
│   ├── nvinfer_11.lib  nvonnxparser_11.lib  nvinfer_plugin_11.lib  (+ lean/dispatch/vc)
│   ├── nvinfer_11.dll  nvinfer_plugin_11.dll  nvonnxparser_11.dll  trtexec.exe
│   ├── nvinfer_builder_resource_{ptx,sm75,sm80,sm86,sm89,sm90,sm100,sm120}_11.dll
│   └── Acknowledgements.txt
├── trtrtx171\                     TensorRT for RTX 1.7.1.116 (optional, aji_trtrtx; see 2b.)
│   ├── include\                   NvInfer*.h, NvOnnxParser.h (TRT_MAJOR_RTX 1, TRT_MINOR_RTX 7)
│   ├── tensorrt_rtx_1_7.lib  tensorrt_onnxparser_rtx_1_7.lib
│   ├── tensorrt_rtx_1_7.dll  tensorrt_onnxparser_rtx_1_7.dll  tensorrt_rtx.exe
│   └── doc\  cmake\  python\
├── ort\
│   ├── ort-dml-1.24.4.nupkg       +  Microsoft.ML.OnnxRuntime.DirectML\   (extracted)
│   └── directml-1.15.4.nupkg      +  Microsoft.AI.DirectML\               (extracted)
├── ffmpeg-shared\ffmpeg-9.0.2-full_build-shared\{include,lib,bin}
└── dist\<tag>\aji-windows-x64.zip      packaging output
             aji-trtrtx-windows-x64.zip
%REPO%\build-win-release\               release build directory  [created by the build]
```

The release build directory sits in the checkout. It is not in `.gitignore`, so exclude it
locally: add `build-win-*/` to `.git/info/exclude`.

> Earlier releases were built from a junction, `%AJI_WIN%\animejanai-inference -> %REPO%`, with
> the scripts living only in `%AJI_WIN%`. Since the scripts moved into `ci/windows/`, they build
> the checkout they sit in, and the junction is not needed.

## Toolchain

| Tool | Version in use | Where |
|---|---|---|
| Visual Studio | 2026 (18.11) Build Tools, C++ workload | found with `vswhere` — Build Tools and Community both work |
| MSVC toolset | 14.51.36231 (`cl` 19.51.36260) | `.../VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe` |
| CMake | 4.3.1 | **bundled with VS**: `.../Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe` |
| Ninja | 1.13.2, bundled with VS | `.../CommonExtensions/Microsoft/CMake/Ninja/ninja.exe` |
| CUDA | 13.4 (nvcc 13.4.59) | `%AJI_WIN%\cuda134` — not a system install (see 1.) |
| PowerShell 7 (`pwsh`) | — | only needed for `package-aji-release.ps1`; must be on `PATH` |

Install Visual Studio with:

```
winget install --id Microsoft.VisualStudio.BuildTools -e --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

`--includeRecommended` brings the Windows SDK (`aji_dml` links `d3d12`, `d3d11`, `dxgi`,
`d3dcompiler`) and the CMake/Ninja component. CMake and Ninja come from the Visual Studio
install — there is no standalone CMake on `PATH`. The build script locates VS 18 with
`vswhere`, sources `VsDevCmd.bat`, and puts `%AJI_WIN%\cuda134\bin` first on `PATH`.

> `VsDevCmd.bat` prints `'vswhere.exe' is not recognized as an internal or external command`
> because the VS installer directory is not on `PATH`. It is **harmless noise** — it appears at
> the top of every successful release build log. Do not chase it.

> `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` stays in the configure line for older
> toolkits: CUDA 13.3's `nvcc` did not recognise MSVC 14.51. CUDA 13.4's does — a trivial
> kernel compiles without the flag.

## Dependency setup, from scratch

### 1. CUDA 13.4 — from NVIDIA's redistributable archives

The build needs only the compiler and the runtime, so instead of running the CUDA installer
(which needs admin rights and installs a display driver by default), `cuda134` is assembled from
five archives listed in NVIDIA's public manifest, which publishes a SHA-256 for every file:

```
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json
  cuda_nvcc    13.4.59    nvcc, ptxas, fatbinary, nvlink, ...
  cuda_crt     13.4.59    nvcc's crt headers
  libnvvm      13.4.59    nvvm\ (cicc, libdevice)
  cuda_cudart  13.4.49    cuda_runtime.h, cuda_fp16.h, cudart.lib, cuda.lib
  cccl         13.3.4.2.1 include\cccl (FindCUDAToolkit adds it to the include path)
```

Download each `windows-x86_64` `relative_path` from `.../compute/cuda/redist/`, check it
against the manifest's `sha256`, and extract all five into `cuda134\`, merging their top-level
`bin\`, `include\`, `lib\` and `nvvm\` directories. `nvcc --version` should then report
`release 13.4, V13.4.59`. The build points CMake at it with `CMAKE_CUDA_COMPILER` and
`CUDAToolkit_ROOT`. This is the same manifest the package assembler pins `cudart` from
(`CudartVersion` in `the-database/mpv-AnimeJaNai`).

TensorRT 11.x requires CUDA 13.2 Update 2 or later; keep this on the CUDA flavour of the
TensorRT zip below.

### 2. TensorRT — one zip from NVIDIA

`trt114` is populated wholesale from NVIDIA's public redistributable archive — no login, and no
third-party redistributor:

```
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.4.0/zip/
  TensorRT-Enterprise-11.4.0.106-Windows-amd64-cuda-13.4-Release-external.zip
  (sha256 b6fd86301ad803492adf7a06de29ce45e3968d29637a1437d0fb0994f412c980)
```

The `-external` suffix marks the publicly redistributable build. It carries everything this
build needs in one versioned place: `include/` headers, the real `lib/*.lib` import libraries,
the runtime DLLs, every per-SM builder resource, a prebuilt `trtexec.exe`, and
`doc/Acknowledgements.txt`. Extract it into `trt114` **flat** — `include/` as is, the contents
of `lib/` and `bin/` at the root — which is the layout `CMakeLists.txt` expects on Windows
(`AJI_TRT_INCLUDE = <root>/include`, `AJI_TRT_LIB = <root>`).

**Keep this, the Linux workflow's `TRT_VERSION`, and `TrtVersion` in the package assembler on
the same version.** `aji_trt` links `nvinfer_11`, which every TensorRT 11.x release ships under
the same name, so the DLL runs on any 11.x runtime. **Engines do not carry over**: a serialized
engine only loads on the TensorRT version that built it (the 11.4 runtime rejects 11.0 engines,
serialization version 243 vs 245). That is why the engine cache name carries the version
(`...trt-11.4.0.gpu-...engine`): a runtime upgrade rebuilds every engine once on first use, and
the stale-engine cleanup deletes the old version's files. Release builds are made against the
TensorRT they ship with.

> **The CUDA flavour is part of the URL and is not derivable from the TensorRT version.**
> NVIDIA's pairing wanders: 11.0 -> cuda-13.2, 11.1 -> 13.3, 11.2 -> 13.3, 11.3 -> 13.4,
> 11.4 -> 13.4. Guessing it yields a 404.

NVIDIA retains old versions (10.13.3.9 through 11.4.0.106 are all still live), so an older
release stays reproducible. Two archival fallbacks exist if a URL ever 404s:
`pypi.nvidia.com/tensorrt-cu13-libs/` (every version, sha256 in the index, but no `trtexec`)
and the CUDA apt repo (including `trtexec` via `libnvinfer-bin`).

Verify the end state:

```c
/* trt114\include\NvInferVersion.h */
#define TRT_MAJOR_ENTERPRISE 11
#define TRT_MINOR_ENTERPRISE 4
#define TRT_PATCH_ENTERPRISE 0
#define TRT_BUILD_ENTERPRISE 106
```

Previous releases assembled the headers and an `nvinfer_11.lib` (synthesized from the DLL's
export table) by hand. Both are obsolete — the zip ships NVIDIA's own headers and import
libraries.

### 2b. TensorRT for RTX — optional, for `aji_trtrtx`

`aji_trtrtx` (selected by `backend=TensorRT-RTX`) builds only when `AJI_TRTRTX_ROOT` is set.
`trtrtx171` comes from NVIDIA's public download (the `secure/` path redirects to a tokenized
CDN URL without a login):

```
https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.7/
  tensorrt-rtx-1.7.1.116-windows-amd64-cuda-13.4-release-external.zip
  (served as TensorRT-RTX-1.7.1.116-Windows-amd64-cuda-13.4-Release-external.zip,
   99,783,796 bytes,
   sha256 2002294a19f81135e4cfd074faddf1f842eb46979f36f7e6db57ff83016ba71d)
```

The release list is on <https://developer.nvidia.com/tensorrt-rtx>; each version comes in a
CUDA 12.9 and a CUDA 13.x flavour, and the CUDA 13.4 one matches `cuda134`. Extract it like
TensorRT, **flat**: `include/` as is, the contents of `lib/` and `bin/` at the root. Check
`include\NvInferVersion.h` for `TRT_MAJOR_RTX 1` / `TRT_MINOR_RTX 7` / `TRT_PATCH_RTX 1`.
Then add `-DAJI_TRTRTX_ROOT=%AJI_WIN%/trtrtx171` to the configure line; it prints
`aji_trtrtx: TensorRT-RTX 1.7 (tensorrt_rtx_1_7;tensorrt_onnxparser_rtx_1_7)`, or
`AJI_TRTRTX_ROOT not set; skipping aji_trtrtx target` without it. A root that holds no SDK
stops the configure with `AJI_TRTRTX_ROOT=...: no TensorRT-RTX SDK there (missing
include/NvInferVersion.h)`.

**The release build requires it**: `build-aji-release.bat` always passes `AJI_TRTRTX_ROOT`
(default `%AJI_WIN%\trtrtx171`) because the release ships `aji_trtrtx.dll` and its runtime, so a
missing SDK fails the release configure with that error instead of silently producing a package
without the backend.

What differs from TensorRT:

- **Runtime is two DLLs, ~232 MB** (`tensorrt_rtx_1_7.dll` 229 MB +
  `tensorrt_onnxparser_rtx_1_7.dll` 2.3 MB), against ~2.4 GB for TensorRT 11.4's DLLs. There are
  no per-SM builder resources and no `trtexec`: `aji_trtrtx` builds engines in-process. The DLLs
  import only Windows system libraries — no CUDA DLL (the driver is reached at run time).
  `aji_trtrtx.dll` itself links `cudart64_13.dll`, like `aji_trt`.
- **GPUs**: compute capability 7.5+ per NVIDIA's support matrix (Turing through Blackwell; the
  header's build targets are sm75/80/86/89/120/121). Engines are built for the current GPU
  (`ComputeCapability::kCURRENT`), so Turing works too.
- **Licence**: the NVIDIA TensorRT for RTX SLA
  (<https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html>) allows
  distributing the libraries only incorporated into an application with material added
  functionality; developer tools (`tensorrt_rtx.exe`) are internal use only, so do not ship it.

### 3. `trtexec.exe` — NVIDIA's, not a patched fork

The package ships the `trtexec.exe` from this zip. Earlier packages took vs-mlrt's rebuild of
it, which carried a patch for long paths and a `CreateFileA` -> `CreateFileW` fix in the
timing-cache file lock. Neither is needed now:

- **Long paths** — NVIDIA's stock binary already embeds a `longPathAware` manifest
  (`<description>trtexec with long path support</description>`).
- **Non-ASCII paths** — fixed on *our* side instead, which is where it belonged.
  `run_build_process` used to spawn with `CreateProcessA`, so the UTF-8 paths in the command
  line were reinterpreted through the process ANSI code page and trtexec received a mangled
  path. It now widens the command line and calls `CreateProcessW`, so the child gets the real
  path regardless of system locale — and regardless of which `trtexec` is in use.

### 4. ONNX Runtime DirectML + DirectML — NuGet packages

Both are plain NuGet packages, extracted in place (a `.nupkg` is a zip). Verified from their
nuspecs: `Microsoft.ML.OnnxRuntime.DirectML` **1.24.4** and `Microsoft.AI.DirectML` **1.15.4** —
the same versions the assembler pins as `OrtDmlVersion` / `DirectMLVersion` and fetches from:

```
https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.24.4
https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4
```

Required layout after extraction:

```
ort\Microsoft.ML.OnnxRuntime.DirectML\build\native\include\      (onnxruntime_c_api.h, ORT_API_VERSION 24)
ort\Microsoft.ML.OnnxRuntime.DirectML\runtimes\win-x64\native\   (onnxruntime.dll/.lib, onnxruntime_providers_shared.dll)
ort\Microsoft.AI.DirectML\include\                               (DirectML.h, DirectMLConfig.h)
ort\Microsoft.AI.DirectML\bin\x64-win\                           (DirectML.dll/.lib)
```

`AJI_ORT_ROOT` and `AJI_DML_ROOT` point at the two package roots.

> **These are the last DirectML-flavoured releases.** Microsoft moved DirectML to sustained
> engineering, so 1.24.x is the ORT ceiling until the WinML migration. Do not expect a newer
> pin to exist.

### 5. ffmpeg dev libraries — for `aji_encode` only

`AJI_FFMPEG_ROOT` needs a root containing `include/` and `lib/`. Only `aji_encode`, the offline
CLI encoder, links ffmpeg; the player path does not (mpv brings its own). In use:
`ffmpeg-shared\ffmpeg-9.0.2-full_build-shared\` — gyan.dev's shared "full build" of
**ffmpeg 9.0.2**, the same ffmpeg release the Linux mpv bundle builds against, carrying
`avcodec-63`, `avformat-63`, `avutil-61`, `swscale-10` plus the matching `.lib` import
libraries. gyan.dev publishes its release builds on GitHub:

```
gh release download 9.0.2 -R GyanD/codexffmpeg -p ffmpeg-9.0.2-full_build-shared.7z
```

Without `AJI_FFMPEG_ROOT`, `aji_encode` is silently skipped.

## The release build

```bat
%REPO%\ci\windows\build-aji-release.bat
```

It configures and builds this checkout into `%REPO%\build-win-release` with (`AJI_TRTRTX_ROOT`
defaults to `%AJI_WIN%\trtrtx171`):

```
cmake -S %REPO% -B %REPO%\build-win-release -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" ^
      "-DCMAKE_CUDA_COMPILER=%AJI_WIN%/cuda134/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%AJI_WIN%/cuda134" ^
      -DAJI_TRT_ROOT=%AJI_WIN%/trt114 -DAJI_NVINFER=nvinfer_11 ^
      "-DAJI_TRTRTX_ROOT:PATH=%AJI_TRTRTX_ROOT%" ^
      -DAJI_ORT_ROOT=%AJI_WIN%/ort/Microsoft.ML.OnnxRuntime.DirectML ^
      -DAJI_DML_ROOT=%AJI_WIN%/ort/Microsoft.AI.DirectML ^
      -DAJI_FFMPEG_ROOT=%AJI_WIN%/ffmpeg-shared/ffmpeg-9.0.2-full_build-shared ^
      "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
cmake --build %REPO%\build-win-release --clean-first
```

Three things in there are deliberate and easy to lose:

- **Every dependency is passed explicitly**, so the build never silently reuses a stale cache
  value.
- **`--clean-first`**, because this is a release.
- **The full architecture list** — see "The CUDA-architecture trap" in `CLAUDE.md`.

### Expected output

```
'vswhere.exe' is not recognized as an internal or external command,
operable program or batch file.
-- The CUDA compiler identification is NVIDIA 13.4.59 with host compiler MSVC 19.51.36260.0
-- Found CUDAToolkit: %AJI_WIN%/cuda134/include;%AJI_WIN%/cuda134/include/cccl (found version "13.4.59")
-- aji_trtrtx: TensorRT-RTX 1.7 (tensorrt_rtx_1_7;tensorrt_onnxparser_rtx_1_7)
-- Configuring done
...
[48/48] Linking CUDA executable aji_msb444_test.exe
AJI-RELEASE-BUILD-OK
```

**48 build edges, and all eight release artifacts plus `aji_msb444_test.exe`.** If the edge
count is lower, a target was skipped — check the configure output for the
`skipping aji_encode target` and `skipping aji_trtrtx target` messages. `AJI-RELEASE-BUILD-OK` on the last line is the success
marker; a MSVC warning C4819 about code page 932 is expected noise on a Japanese ANSI code page.

Quick GPU self-check after a build (TensorRT and CUDA on `PATH`):

```bat
set PATH=%AJI_WIN%\cuda134\bin;%AJI_WIN%\trt114;%PATH%
%REPO%\build-win-release\aji_msb444_test.exe
```

prints `MSB 4:4:4 GPU normalization and quantization passed`.

### Dev build (much faster)

For iterating on one machine, configure a separate directory with
`-DCMAKE_CUDA_ARCHITECTURES=native` (or `120` for RTX 50-series) and without
`AJI_FFMPEG_ROOT`, so `aji_encode.exe` is skipped. Keep it out of `build-win-release`, so a
single-arch build can never pollute the release cache, and be clear which TRT root a dev
artifact was built against before comparing it to a release.

## Packaging and release

**Choosing the tag.** The tag is whatever value `AjiVersion` will hold in the package assembler
(`BuildMpvUpscale2xAnimeJaNai/Program.cs` in `the-database/mpv-AnimeJaNai`) — that one constant
builds *both* asset URLs. In practice the Linux CI job runs first and creates the release, so
use the tag you passed to its `release_tag` input; this step then only attaches a second asset
to the release that already exists. Build the Windows asset from the same commit the Linux job
built.

```powershell
pwsh %REPO%\ci\windows\package-aji-release.ps1 -Tag v0.11.2
```

It builds two zips, **throwing `missing build artifact: <path>` if any artifact is absent**,
and prints each one's entry list and size:

| Asset | Contents | Size |
|---|---|---|
| `aji-windows-x64.zip` | `aji.dll`, `aji_dml.dll`, `aji_trt.dll`, `aji_encode.exe`, `aji_harness.exe`, `aji_harness_dml.exe`, `aji_kernel_test.exe` | ~0.8 MB |
| `aji-trtrtx-windows-x64.zip` | `aji_trtrtx.dll` plus the TensorRT-RTX runtime from `-TrtRtxRoot` (default `%AJI_TRTRTX_ROOT%`, else `%AJI_WIN%\trtrtx171`): exactly one `tensorrt_rtx_*.dll` and one `tensorrt_onnxparser_rtx_*.dll`, or it throws | ~91 MB |

Output:

```
%AJI_WIN%\dist\<tag>\aji-windows-x64.zip
%AJI_WIN%\dist\<tag>\aji-trtrtx-windows-x64.zip
```

The contents are flat because consumers extract them straight into their inference directory
(the assembler: `animejanai/inference/`); the TensorRT-RTX zip goes on top of the engine zip.

**TensorRT-RTX is a separate asset so consumers opt in to its ~232 MB runtime.** It is
experimental and not offered by every consumer: mpv AnimeJaNai leaves it out for now (its
background-build mode starts each play with a short passthrough while the engine warms up),
while the offline and image consumers (VideoJaNai, MangaJaNaiConverterGui, yosh) block on the
load and gain the most from its fast builds. A consumer that skips the asset loses nothing: a
conf asking for `backend=TensorRT-RTX` without `aji_trtrtx.dll` logs `TensorRT-RTX failed to
start; using TensorRT instead` and plays on TensorRT. The runtime ships here because nothing
else pins it yet; **never `tensorrt_rtx.exe`**, which the SDK licence keeps for internal use.
Moving it to the consumers' own installers (from NVIDIA's zip, like `TrtVersion`) would restore
the rule below for this asset too.

The engine zip contains no TensorRT DLLs, no
`trtexec.exe`, no `onnxruntime.dll` and no `DirectML.dll`, even though the backends load those
from their own directory at runtime. The package assembler drops them into that same
`animejanai/inference/` directory from its own independently pinned sources:
`InstallInferenceRuntime()` takes the TensorRT runtime, the per-SM builder resources and
`trtexec.exe` from the same NVIDIA zip this build uses (`TrtVersion`) plus `cudart` from the
CUDA redistributable, and `InstallOrtDml()` pulls
`onnxruntime.dll` + `DirectML.dll` from the two NuGet packages (`OrtDmlVersion`,
`DirectMLVersion`). That is why this build needs those SDKs only to **compile and link**
against — it never ships them.

`-BuildDir` and `-OutDir` are overridable.

Then attach both to the release that the Linux CI job created for the same tag:

```bash
gh release upload v0.11.2 %AJI_WIN%/dist/v0.11.2/aji-windows-x64.zip \
  %AJI_WIN%/dist/v0.11.2/aji-trtrtx-windows-x64.zip \
  -R the-database/animejanai-inference
```

**Both platform assets must sit on the same tag** — the package assembler derives both URLs
from one `AjiVersion`.

To test the zip in a package build before releasing it, point the assembler at it with
`AJI_LOCAL_ZIP` instead of uploading.

## Testing

Nothing automates validation and nothing gates a release. The practical check is to play
content through an installed package with the new engine dropped in: copy `aji.dll`,
`aji_trt.dll` and `aji_dml.dll` into a copy of an install's `animejanai\inference\`, run its
`mpv.exe --config-dir=<copy>\portable_config`, and confirm in the log (`--msg-level=all=v`)
that `[animejanai] Configured slot ...` reports the upscaled size. First play of each model and
resolution builds a TensorRT engine, which takes a few seconds to a minute.

> The previous build workstation also carried a set of unversioned test rigs in its working
> area (engine prebuild, benchmark, screenshot A/B, installer simulation, and `run-mock.cmd`,
> which kept the build dir off `PATH` to catch DLL-resolution failures). They were never
> committed and are not available on the current workstation.
