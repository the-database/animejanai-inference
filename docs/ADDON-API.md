# Optional addon interfaces: review and integration

This PR supplies the inference side of AJN addon API 1.12. The public guest API is implemented in `mpv-AnimeJaNai/addons`; addons do not call these C functions or load these DLLs themselves. See the main PR's `addons/REVIEW-GUIDE.md` and `addons/DEVELOPER-GUIDE.md` for the complete system and creator contract.

## Interfaces to review

| Interface | Purpose | Files |
| --- | --- | --- |
| `aji_rife_scene_supported`, `aji_infer_rife_with_scene` | Optional cut/continuous decision supplied by the trusted player; preserves default detection when no override is supplied | `include/aji.h`, dispatcher and both backends |
| `aji_stream_policy_version_for_config`, `async_build=2` | Cached-only streaming; fail on missing/incompatible engines instead of silently building or sending passthrough output | `include/aji.h`, dispatcher, `src/aji_trt.cpp`, `src/aji_dml.cpp` |
| `aji_observer_*`, analysis ABI 1 | Optional worker-owned CUDA source reduction and fixed-model character masks | `include/aji_analysis.h`, `src/aji_analysis_loader.cpp`, `src/aji_analysis.cpp`, `src/analysis_kernels.*` |

These are additive to AJI ABI 9. Old backend support is queried, not assumed. Invalid explicit scene decisions fail rather than silently changing semantics. The ordinary inference path does not call the observation API or import `aji_analysis.dll`.

The analysis module borrows the CUDA context, retains work through its private stream, and returns only bounded BGRA8/unorm16 samples. The host verifies the approved model's digest; this API is not an arbitrary-model guest capability. `aji_observer_run` can wait on its optional worker, not the playback callback. Cancellation and destruction belong to the caller's lifetime protocol.

Observer creation rejects a null CUDA context. After successfully opening the current analysis engine, cleanup removes unlocked older analysis variants for the same model and GPU; other models, GPUs and active caches are left alone.

Windows model/cache/lock/temp paths are normalized consistently into extended-path form. Analysis caches use `.analysisengine`, independently from the upscaler's `.engine` cleanup. Their identity includes the SHA-256 of the bytes actually parsed, TensorRT version, driver and GPU UUID. Cache creation uses a lock and temporary file, checks completed writes, and reports Windows error codes. Lock acquisition is cancellable and limited to ten minutes even without a cancellation callback. Earlier mpv.net qualification covered paths beyond MAX_PATH; see the PR for checks of the current revision.

## Reconciliation with current upstream

The review branch includes the upstream shared TensorRT engine utilities and TensorRT-RTX work. It preserves upstream's `ajicore::AsyncBuild` ownership and ordinary-playback fallback. The addon streaming policy is checked before `be.create`; a cached-only request never falls back to a different backend. TensorRT-RTX reports policy version 0 until its build/warmup/fallback behavior has separate addon qualification. Classic TensorRT and DirectML retain policy version 1.

Cached-only requests do not delete stale or zero-byte engines. Missing/incompatible errors identify the model and expected engine path. `scene=0` supplied through the explicit scene hook skips the built-in detector and its scalar readback; `scene=-1` preserves default detection. Builder subprocesses receive an explicit inherited-handle list.

This is not a new TensorRT-RTX addon feature. Source for the same scene hooks is shared by TensorRT variants, but the current Windows addon qualification covers classic TensorRT and DirectML. The normal RTX backend behavior is otherwise upstream's.

## Build and evidence

`AJI_BUILD_ANALYSIS` defaults to `OFF`. Ordinary Windows and Linux builds do not need `NvOnnxParser.h` or the analysis runtime. Explicitly enabling it currently requires Windows. The optional release packaging adds `aji_analysis.dll` and `aji_analysis_trt.dll`. The loader `aji_analysis.dll` imports no CUDA or TensorRT library: it loads cudart, nvinfer and the ONNX parser from the shared runtime packs when `aji-runtime.json` names them (otherwise the observer's imports resolve beside it), then the observer, and points TensorRT's in-process builder at the GPU's builder pack. No TensorRT file is bundled; consumers supply them like they do for `aji_trt.dll`.

`tools/build-windows-test.ps1` explicitly enables analysis and builds dispatcher, classic TensorRT, DirectML and regression harnesses. It and `ci/windows/build-aji-release.bat` read SDK versions from `ci/windows/dependencies.json`; downloaded NuGet dependencies are SHA-256 pinned. The script records its producing commit in `build/build-info.json`. CI (`ci.yml`) runs it on pull requests and pushes to `main` and uploads those artifacts as `aji-windows-test`. It does not enable or certify TensorRT-RTX without its separate SDK.

The coordinated PR descriptions link source-specific build and test results. Do not treat an earlier preview result, compile success, a symbol export or a capability manifest as proof of current real-time behavior. No model, GPU-specific engine or user media is committed here.

The dispatcher has no selected backend for the legacy no-argument `aji_stream_policy_version()` query and returns 0. Use `aji_stream_policy_version_for_config(conf_path)` to check the requested backend: classic TensorRT/DirectML return 1, TensorRT-RTX returns 0. This query does not load a backend or promise engine readiness; creation still verifies the actual backend and cache. Fixed backend DLLs retain their legacy query. A null config path uses the ordinary TensorRT default.

Release packaging reads both SDK paths from `dependencies.json` and the current `AJI_BUILD_ANALYSIS` setting from the build's CMake cache, so stale analysis outputs cannot leak into an OFF release. `AJI_TRT_LIB` is an optional explicit override; its default follows `AJI_TRT_ROOT` on every reconfiguration. The release script supplies the matching root and library together. `tools/test-release-configuration.py` and `tools/test-release-packaging.ps1` cover those paths without requiring a GPU.

`python tools/test-stream-policy-dispatch.py build/Release/aji.dll` is a no-GPU Windows regression check: a missing RTX backend preserves ordinary playback's fallback attempt, while a cached-only addon request fails without changing backend. The Windows workflow runs it after compilation.

`python tools/test-cached-only-files.py <runtime-dir> <small-model.onnx> <new-output-dir>` runs the focused GPU test for missing and zero-byte engine failures. It verifies that both requests leave the model/cache files unchanged and does not compile an engine.

Downstream mpv must retain matching private headers and capability discovery; winbuild must package the produced native metadata; the host must preserve grants, worker ownership and resource limits. The optional analysis DLL is installed beside the other inference DLLs, with matching TensorRT/CUDA dependencies available on first use. No ordinary-playback sampling, model load or readback is added when there is no observer.
