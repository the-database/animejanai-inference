# Optional addon interfaces: review and integration

This PR supplies the inference side of AJN addon API 1.12. The public guest API is implemented in `mpv-AnimeJaNai/addons`; addons do not call these C functions or load these DLLs themselves. See the main PR's `addons/REVIEW-GUIDE.md` and `addons/DEVELOPER-GUIDE.md` for the complete system and creator contract.

## Interfaces to review

| Interface | Purpose | Files |
| --- | --- | --- |
| `aji_rife_scene_supported`, `aji_infer_rife_with_scene` | Optional cut/continuous decision supplied by the trusted player; preserves default detection when no override is supplied | `include/aji.h`, dispatcher and both backends |
| `aji_stream_policy_version`, `async_build=2` | Cached-only streaming; fail on missing/incompatible engines instead of silently building or sending passthrough output | `include/aji.h`, dispatcher, `src/aji_trt.cpp`, `src/aji_dml.cpp` |
| `aji_observer_*`, analysis ABI 1 | Optional worker-owned CUDA source reduction and fixed-model character masks | `include/aji_analysis.h`, `src/aji_analysis.cpp`, `src/analysis_kernels.*` |

These are additive to AJI ABI 9. Old backend support is queried, not assumed. Invalid explicit scene decisions fail rather than silently changing semantics. The ordinary inference path does not call the observation API or import `aji_analysis.dll`.

The analysis module borrows the CUDA context, retains work through its private stream, and returns only bounded BGRA8/unorm16 samples. The host verifies the approved model's digest; this API is not an arbitrary-model guest capability. `aji_observer_run` can wait on its optional worker, not the playback callback. Cancellation and destruction belong to the caller's lifetime protocol.

Windows model/cache/lock/temp paths are normalized consistently into extended-path form. Analysis caches use `.analysisengine`, independently from the upscaler's `.engine` cleanup. Their identity includes the SHA-256 of the bytes actually parsed, TensorRT version, driver and GPU UUID. Cache creation uses a lock and temporary file, checks completed writes, and reports Windows error codes. Lock acquisition is cancellable and limited to ten minutes even without a cancellation callback. Earlier mpv.net qualification covered paths beyond MAX_PATH; see the PR for checks of the current revision.

## Reconciliation with current upstream

The review branch includes the upstream shared TensorRT engine utilities and TensorRT-RTX work. It preserves upstream's `ajicore::AsyncBuild` ownership and ordinary-playback fallback. The addon streaming policy is checked before `be.create`; a cached-only request never falls back to a different backend. TensorRT-RTX reports policy version 0 until its build/warmup/fallback behavior has separate addon qualification. Classic TensorRT and DirectML retain policy version 1.

Cached-only requests do not delete stale or zero-byte engines. Missing/incompatible errors identify the model and expected engine path. `scene=0` supplied through the explicit scene hook skips the built-in detector and its scalar readback; `scene=-1` preserves default detection. Builder subprocesses receive an explicit inherited-handle list.

This is not a new TensorRT-RTX addon feature. Source for the same scene hooks is shared by TensorRT variants, but the current Windows addon qualification covers classic TensorRT and DirectML. The normal RTX backend behavior is otherwise upstream's.

## Build and evidence

`AJI_BUILD_ANALYSIS` defaults to `OFF`. Ordinary Windows and Linux builds do not need `NvOnnxParser.h` or the analysis runtime. Explicitly enabling it currently requires Windows. The optional release packaging includes `aji_analysis.dll`, the ONNX parser, builder resources and lean runtime from the matching TensorRT installation.

`tools/build-windows-test.ps1` explicitly enables analysis and builds dispatcher, classic TensorRT, DirectML and regression harnesses. It and `ci/windows/build-aji-release.bat` read SDK versions from `ci/windows/dependencies.json`; downloaded NuGet dependencies are SHA-256 pinned. The script records its producing commit in `build/build-info.json`. The manual `build-windows-test.yml` workflow uploads those artifacts. It does not enable or certify TensorRT-RTX without its separate SDK.

The coordinated PR descriptions link source-specific build and test results. Do not treat an earlier preview result, compile success, a symbol export or a capability manifest as proof of current real-time behavior. No model, GPU-specific engine or user media is committed here.

`python tools/test-stream-policy-dispatch.py build/Release/aji.dll` is a no-GPU Windows regression check: a missing RTX backend preserves ordinary playback's fallback attempt, while a cached-only addon request fails without changing backend. The Windows workflow runs it after compilation.

`python tools/test-cached-only-files.py <runtime-dir> <small-model.onnx> <new-output-dir>` runs the focused GPU test for missing and zero-byte engine failures. It verifies that both requests leave the model/cache files unchanged and does not compile an engine.

Downstream mpv must retain matching private headers and capability discovery; winbuild must package the produced native metadata; the host must preserve grants, worker ownership and resource limits. The optional analysis DLL is installed beside the other inference DLLs, with matching TensorRT/CUDA dependencies available on first use. No ordinary-playback sampling, model load or readback is added when there is no observer.
