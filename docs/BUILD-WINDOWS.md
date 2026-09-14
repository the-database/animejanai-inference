# Building libaji on Windows

**There is no Windows CI for this repo.** `.github/workflows/` contains exactly one workflow,
`build-linux.yml`. The `aji-windows-x64.zip` release asset that
`the-database/mpv-AnimeJaNai` downloads is built and uploaded **by hand from this
workstation**, so this document is the only record of how to reproduce it.

Everything below is driven from a working area **outside** the repo that holds the
hand-assembled SDKs, the build scripts, and the test rigs. Linux build:
[`BUILD-LINUX.md`](BUILD-LINUX.md). Engine overview: [`../CLAUDE.md`](../CLAUDE.md).

### Paths in this document

Two placeholders stand in for machine-specific locations. Substitute your own, or set them as
real environment variables and the commands below work as written:

| Placeholder | Means |
|---|---|
| `%AJI_WIN%` | the working area holding the SDK roots, build scripts and test rigs (conventionally a directory named `aji-win`) |
| `%REPO%` | your checkout of this repository |

**The scripts under `%AJI_WIN%` hardcode absolute paths.** They are reproduced here with the
placeholders substituted in, so they are faithful in structure but not byte-identical — adapt
the paths when adopting them on another machine.

## Target layout

The build scripts pass every dependency path **explicitly and absolutely**, so the layout is
part of the contract:

```
%AJI_WIN%\
├── animejanai-inference\          NTFS junction -> %REPO%
│   ├── build-win-release\         release build (all arches)   [created by the build]
│   └── build-win-trt11\           dev build    (single arch)   [created by the build]
├── trt113\                        TensorRT 11.3.0.99  <- the release TRT root
│   ├── include\                   NvInfer*.h + impl\
│   ├── nvinfer_11.lib  nvonnxparser_11.lib  nvinfer_plugin_11.lib  (+ lean/dispatch/vc)
│   ├── nvinfer_11.dll  nvinfer_plugin_11.dll  nvonnxparser_11.dll  trtexec.exe
│   ├── nvinfer_builder_resource_{ptx,sm75,sm80,sm86,sm89,sm90,sm100,sm120}_11.dll
│   └── Acknowledgements.txt
├── trt111\ , trt11\               TRT 11.1.0.106 / 11.0.0.114 — reference only
├── ort\
│   ├── ort-dml-1.24.4.nupkg       +  Microsoft.ML.OnnxRuntime.DirectML\   (extracted)
│   └── directml-1.15.4.nupkg      +  Microsoft.AI.DirectML\               (extracted)
├── ffmpeg-shared\ffmpeg-8.1.1-full_build-shared\{include,lib,bin}
└── dist\<tag>\aji-windows-x64.zip      packaging output
```

The **junction** matters. Both `CMakeCache.txt` files record
`CMAKE_HOME_DIRECTORY=%AJI_WIN%/animejanai-inference`, and the repo's
`.git/info/exclude` says: "local build dirs (out-of-repo build env in ~/aji-win uses these via
junction)". Create it with:

```bat
mklink /J %AJI_WIN%\animejanai-inference %REPO%
```

(`Get-Item` on that path reports `LinkType: Junction` with the repo as its `Target`.)

Because it is a junction, `%REPO%\build-win-release` and `%AJI_WIN%\animejanai-inference\build-win-release`
are the same directory — which is why `build-aji-release.bat` and `package-aji-release.ps1`
spell the path differently and still agree.

## Toolchain

| Tool | Version in use | Where |
|---|---|---|
| Visual Studio | 18 Community | `C:\Program Files\Microsoft Visual Studio\18\Community` |
| MSVC toolset | 14.51.36231 | `.../VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe` |
| CMake | 4.3.1 | **bundled with VS**: `.../Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe` |
| Ninja | bundled with VS | `.../CommonExtensions/Microsoft/CMake/Ninja/ninja.exe` |
| CUDA Toolkit | 13.3 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` |
| PowerShell 7 (`pwsh`) | — | only needed for `package-aji-release.ps1`; must be on `PATH` |

CMake and Ninja come from the Visual Studio install — there is no standalone CMake on `PATH`
here. Every build script starts by sourcing the VS environment. This is `build-aji-release.bat`'s
prologue — you do **not** run it yourself, the script does:

```bat
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -no_logo || exit /b 1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "SRC=%AJI_WIN%\animejanai-inference"
set "BLD=%SRC%\build-win-release"
```

CUDA 13.3 was installed from `cuda_13.3.0_windows_network.exe`, which is still in `aji-win`.

> `VsDevCmd.bat` prints `'vswhere.exe' is not recognized as an internal or external command`
> in this environment. It is **harmless noise** — it appears at the top of every successful
> release build log. Do not chase it.

> `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` is required: CUDA 13.3's `nvcc` does not
> recognise MSVC 14.51 as a supported host compiler. `nvcc-test.cmd` in `aji-win` is the
> minimal reproducer — it compiles a trivial kernel with and without the flag.

## Dependency setup, from scratch

### 1. TensorRT — one zip from NVIDIA

`trt113` is populated wholesale from NVIDIA's public redistributable archive — no login, and no
third-party redistributor:

```
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.3.0/zip/
  TensorRT-Enterprise-11.3.0.99-Windows-amd64-cuda-13.4-Release-external.zip
```

The `-external` suffix marks the publicly redistributable build. It carries everything this
build needs in one versioned place: `include/` headers, the real `lib/*.lib` import libraries,
the runtime DLLs, every per-SM builder resource, a prebuilt `trtexec.exe`, and
`doc/Acknowledgements.txt`. Extract it into `trt113` **flat** — headers under `include/`, the
`.lib` and `.dll` files at the root — which is the layout `CMakeLists.txt` expects on Windows
(`AJI_TRT_INCLUDE = <root>/include`, `AJI_TRT_LIB = <root>`).

**Keep this and `TrtVersion` in the package assembler on the same version.** `aji_trt` links
`nvinfer_11` and must be built against the TensorRT it runs on.

> **The CUDA flavour is part of the URL and is not derivable from the TensorRT version.**
> NVIDIA's pairing wanders: 11.0 -> cuda-13.2, 11.1 -> 13.3, 11.2 -> 13.3, 11.3 -> 13.4.
> Guessing it yields a 404.

NVIDIA retains old versions (10.13.3.9 through 11.3.0.99 are all still live), so an older
release stays reproducible. Two archival fallbacks exist if a URL ever 404s:
`pypi.nvidia.com/tensorrt-cu13-libs/` (every version, sha256 in the index, but no `trtexec`)
and the CUDA apt repo (including `trtexec` via `libnvinfer-bin`).

### 2. Headers and the import library — both from that zip

Previous releases assembled these by hand: headers of unrecorded provenance, and an
`nvinfer_11.lib` synthesized from the DLL's export table by `make-trt111-implib.cmd` (a
`dumpbin /exports` scrape with a hardcoded `skip=19` that broke silently on a toolchain
change). Both are obsolete — the zip ships NVIDIA's own headers and import libraries, so
`make-trt111-implib.cmd` / `make-trt11-implib.cmd` are no longer part of any build.

Verify the end state:

```c
/* trt113\include\NvInferVersion.h */
#define TRT_MAJOR_ENTERPRISE 11
#define TRT_MINOR_ENTERPRISE 3
#define TRT_PATCH_ENTERPRISE 0
#define TRT_BUILD_ENTERPRISE 99
```

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

`AJI_FFMPEG_ROOT` needs a root containing `include/` and `lib/`. In use:
`ffmpeg-shared\ffmpeg-8.1.1-full_build-shared\` — a shared "full build" of **ffmpeg 8.1.1**,
carrying `avcodec-62`, `avformat-62`, `avutil-60`, `avfilter-11`, `swscale-9`, `swresample-6`
plus the matching `.lib` import libraries.

> **Unverified:** where `ffmpeg-shared.7z` was downloaded from — no script in `aji-win`
> references it. The naming matches the common `*-full_build-shared` Windows builds. Any
> shared ffmpeg dev tree with `include/` + `lib/` for those libraries satisfies the build;
> without it, `aji_encode` is silently skipped.

### 6. Inno Setup (only for installer testing)

`install-inno.cmd` installs it silently from the bundled `is-setup.exe`:

```bat
%AJI_WIN%\is-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /NORESTART /DIR="%LOCALAPPDATA%\Programs\Inno Setup 6"
```

Not needed to build libaji — only for `build-installer.cmd` and the install/upgrade rigs.

## The release build

One script, `%AJI_WIN%\build-aji-release.bat`. Its own header calls it "the single
canonical release build of the aji engine". Just run it:

```bat
%AJI_WIN%\build-aji-release.bat
```

The configure line it runs. **This is the script's single `cmake` line re-wrapped with `^`
continuations for readability, not a byte-exact quote** — `build-aji-release.bat` itself is
the source of truth if you need exact spelling:

```
cmake -S %AJI_WIN%\animejanai-inference ^
      -B %AJI_WIN%\animejanai-inference\build-win-release ^
      -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" ^
      -DAJI_TRT_ROOT=%AJI_WIN%/trt113 ^
      -DAJI_NVINFER=nvinfer_11 ^
      -DAJI_ORT_ROOT=%AJI_WIN%/ort/Microsoft.ML.OnnxRuntime.DirectML ^
      -DAJI_DML_ROOT=%AJI_WIN%/ort/Microsoft.AI.DirectML ^
      -DAJI_FFMPEG_ROOT=%AJI_WIN%/ffmpeg-shared/ffmpeg-8.1.1-full_build-shared ^
      "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
cmake --build %AJI_WIN%\animejanai-inference\build-win-release --clean-first
```

Three things in there are deliberate and easy to lose:

- **Every dependency is passed explicitly**, so the build never silently reuses a stale cache
  value.
- **`--clean-first`**, because this is a release.
- **Its own build directory**, `build-win-release`, separate from the dev build's
  `build-win-trt11`, so a single-arch dev build can never pollute the release cache.

### Expected output

```
'vswhere.exe' is not recognized as an internal or external command,
operable program or batch file.
-- Configuring done (1.6s)
-- Generating done (0.1s)
-- Build files have been written to: %AJI_WIN%/animejanai-inference/build-win-release
[1/1] Cleaning all built files...
Cleaning... 22 files.
[1/19] Building C object CMakeFiles\aji_kernel_test.dir\src\kernel_test.c.obj
...
[19/19] Linking C executable aji_encode.exe
AJI-RELEASE-BUILD-OK
```

**19 build edges and all seven artifacts.** If the edge count is lower, a target was skipped —
check the configure output for the `skipping aji_encode target` message. `AJI-RELEASE-BUILD-OK`
on the last line is the success marker; a MSVC warning C4819 about code page 932 is expected
noise on a Japanese ANSI code page.

Reference logs from real releases: `%AJI_WIN%\aji-release-v080.log` and `aji-release-v070.log`.

### Dev build (much faster)

For iterating, build one architecture into a separate directory:

```bat
%AJI_WIN%\build-shim-trt111.cmd      REM TRT 11.1 root, -DCMAKE_CUDA_ARCHITECTURES=120
%AJI_WIN%\build-shim-trt11.cmd       REM TRT 11.0 root, same arch
```

Both set `-DCMAKE_CUDA_ARCHITECTURES=120` (sm120 / RTX 50-series) and omit `AJI_FFMPEG_ROOT`,
so **`aji_encode.exe` is not built** by a dev build — consistent with `build-win-trt11`'s
`CMakeCache.txt` recording `AJI_FFMPEG_ROOT:PATH=` empty. They print `SHIM-TRT111-BUILD-OK` /
`SHIM-TRT11-BUILD-OK`.

> `build-shim-trt111.cmd` targets `build-win-trt111`, which does not currently exist on disk —
> the dev builds that were actually done used `build-win-trt11` (TensorRT 11.0). Either is fine;
> just be clear which TRT root a dev artifact was built against before comparing it to a release.

## Packaging and release

**Choosing the tag.** The tag is whatever value `AjiVersion` will hold in the package assembler
(`BuildMpvUpscale2xAnimeJaNai/Program.cs` in `the-database/mpv-AnimeJaNai`) — that one constant
builds *both* asset URLs. In practice the Linux CI job runs first and creates the release, so
use the tag you passed to its `release_tag` input; this step then only attaches a second asset
to the release that already exists.

```powershell
pwsh %AJI_WIN%\package-aji-release.ps1 -Tag v0.8.0
```

It stages exactly seven files — `aji.dll`, `aji_dml.dll`, `aji_trt.dll`, `aji_encode.exe`,
`aji_harness.exe`, `aji_harness_dml.exe`, `aji_kernel_test.exe` — **throwing
`missing build artifact: <path>` if any is absent**, zips them flat, and prints the entry list
and size. Output:

```
%AJI_WIN%\dist\<tag>\aji-windows-x64.zip
```

The contents are flat because the assembler extracts them straight into
`animejanai/inference/`.

**The zip carries the engine only — no runtime dependencies.** It contains no TensorRT DLLs, no
`trtexec.exe`, no `onnxruntime.dll` and no `DirectML.dll`, even though the backends load those
from their own directory at runtime. The package assembler drops them into that same
`animejanai/inference/` directory from its own independently pinned sources:
`InstallInferenceRuntime()` takes the TensorRT runtime, the per-SM builder resources and
`trtexec.exe` from the same NVIDIA zip this build uses (`TrtVersion`) plus `cudart` from the
CUDA redistributable, and `InstallOrtDml()` pulls
`onnxruntime.dll` + `DirectML.dll` from the two NuGet packages (`OrtDmlVersion`,
`DirectMLVersion`). That is why this build needs those SDKs only to **compile and link**
against — it never ships them. `pkg-mock\animejanai\inference\` mirrors the assembled result.

Parameters `-BuildDir` and `-OutDir` are overridable; the default
`-BuildDir` defaults to the repo-side spelling of the build directory while the build script
writes through the junction — the same directory either way.

Then attach it to the release that the Linux CI job created for the same tag:

```bash
gh release upload v0.8.0 %AJI_WIN%/dist/v0.8.0/aji-windows-x64.zip \
  -R the-database/animejanai-inference
```

**Both platform assets must sit on the same tag** — the package assembler derives both URLs
from one `AjiVersion`.

To test the zip in a package build before releasing it, point the assembler at it with
`AJI_LOCAL_ZIP` instead of uploading.

## Testing rigs in `aji-win`

None of these are needed to build; they are how a build gets validated. **Nothing automates
them and no rig gates a release** — picking which to run after a change is a judgement call,
so the table below says what each one actually proves.

**Prebuild TensorRT engines** (first-play engine builds are slow, so do them up front):

```bat
prebuild-bench.cmd        REM bench models, slots 1010/1011, all five benchmark resolutions
prebuild-display.cmd      REM slots 1..5 at 1920x1080 against test-animejanai.conf
prebuild-rife16.cmd       REM slots 4,5 against models\rife-fp16
```

All shell out to `aji_harness.exe --input <raw> --width W --height H --frames 1 --fps 23.976
--conf <conf> --model-dir <dir> --trtexec <trtexec.exe> --slot N`.

**Playback:**

| Script | What it proves |
|---|---|
| `aji-play.cmd` | drag-and-drop playback through `%AJI_WIN%\mpv\build\mpv.exe` + `portable_config` |
| `run-mpv.cmd` | staging launcher with the shim build dir on `PATH` |
| `run-mock.cmd` | the **DLL-resolution test** — deliberately keeps the build dir off `PATH` and uses a neutral cwd, so `aji.dll` must resolve its TensorRT deps via `LOAD_WITH_ALTERED_SEARCH_PATH` alone. This is what catches "works on my machine, broken once installed". |

**Benchmarks:** `bench-native.cmd` (TensorRT, five resolutions × slots 1010/1011),
`bench-native-dml.cmd` (DirectML via `--hwdec=d3d11va --gpu-api=d3d11`),
`bench-native-pipe.cmd` / `bench-dml-pipe.cmd` (queue-depth sweeps, 3 repeats),
`bench-one.cmd` / `bench-dbg.cmd` (single cells).

**Output correctness:** `shot-interop.cmd`, `shot-bisect.cmd`, `shot-regress.cmd`,
`shot-fixtest.cmd`, `shot-charts.cmd` capture PNG screenshots into `shots-*` dirs for A/B
comparison; `dml-md5.cmd` writes `framemd5` output for bit-exactness checks across queue depths;
`verify-rife.cmd` checks RIFE slots 4/5.

**Installer / updater simulation:** `run-installer.cmd`, `test-install-silent.cmd`,
`test-uninstall.cmd`, `clean-reinstall.cmd`, `test-recommend.cmd`, `sim*.cmd`. These set
`ANIMEJANAI_PACKS_DIR=%AJI_WIN%\draft-packs` so the updater resolves components
from a local directory — necessary while a release is still a draft, because draft assets 404
for the unauthenticated releases API.

**Runtime layout these rigs assume:** `pkg-mock\animejanai\inference\` holds `aji.dll` +
backends + the TRT runtime DLLs + `trtexec.exe`, with `animejanai.conf` at
`pkg-mock\animejanai\`, and models in `bench-models\` / `models\`. That mirrors an installed
package, which is the point.

> **These scripts are not version-controlled.** They live only in `%AJI_WIN%`, with
> no backup. `build-aji-release.bat` and `package-aji-release.ps1` in particular are the sole
> definition of how the Windows asset is produced — worth copying into this repo under
> `ci/windows/` so the release process survives the machine.
