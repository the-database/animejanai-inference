# Plan v2: Native mpv inference filter + C-ABI backend shim (no-host-round-trip GPU pipeline)

*Supersedes v1. All mpv file:line references verified against this checkout (the-database/mpv fork).*

## Changes from v1

1. **New: C-ABI inference shim (`libaji`) in a dedicated repo** — fixes the unaddressed mingw-mpv ↔ MSVC-TensorRT toolchain clash, kills the TRT version lock structurally, collapses vf_trt/vf_dml/vf_ncnn into ONE thin mpv filter (`vf_animejanai`), and enables a standalone CLI test harness.
2. **Phase 4 (NCNN) demoted** from scheduled phase to "deferred pending demand," and its premise corrected: `hwdec_vulkan.c` exists in-tree (v1 claimed mpv has no Vulkan decode). GLSL shaders documented as the non-NVIDIA fallback meanwhile.
3. **RIFE split into its own milestone (Phase 1.5)**, still inside the NVIDIA release gate.
4. **Numeric parity harness** (PSNR/SSIM vs goldens captured from the current vsmlrt pipeline) replaces eyeball-only verification — the pre-kernel reimplements a slice of zimg (chroma upsampling, matrix, range) and that's where silent quality bugs live.
5. **Bloat attack added:** evaluate TRT ≥ 8.6 hardware-compatible (Ampere+) + version-compatible prebuilt engines so most users never download `nvinfer_builder_resource` (~1.3 GB); builder becomes an optional download for Turing/Pascal.
6. **Output-format decision pinned to Phase 0** (RGB fp16 CUDA hwframe vs convert-back-to-YUV), with the FFmpeg `hwcontext_cuda` sw_format caveat made explicit.
7. **Sync-copy rationale corrected** (decoder pool lifetime/exhaustion, not "tearing").
8. **lavfi alternative documented** as a considered escape hatch — `f_lavfi.c` already passes hw frames, devices, and `vf-command` into libavfilter graphs; mpv already auto-inserts `bwdif_cuda` today, proving the CUDA-filter→render handoff in-tree.
9. **Windows frontend decided: keep mpv.net** (libmpv swap-in). Linux ships standalone forked mpv.
10. **Timelines made realistic:** NVIDIA milestone ~3–4 months (was 2–2.5); reference implementation reframed as feasibility evidence (screenshots), not a recipe.

## Context

Today's runtime pipeline is `mpv (mpv.net) → vf=vapoursynth → .vpy → Python (animejanai_core) → vsmlrt → vstrt.dll (TensorRT)`. Motivations to leave vs-mlrt:

1. **TensorRT version lock** — vstrt.dll is ABI-linked to a specific TRT build.
2. **Bloat** — the vs-mlrt CUDA archive is several GB.
3. **No host round-trips** — decode, upscale, and render must stay in GPU memory.

**Decisive finding (verified):** VapourSynth's frame model is host-memory only. `vf_vapoursynth.c:141-172` (`mp_to_vs`/`mp_from_vs`) maps only software YUV formats, and `filters/f_autoconvert.c:276` force-inserts `mp_hwdownload_create` when a hw frame meets a software-only filter. No VapourSynth-based design can satisfy the requirement. The inference must move into mpv as a native `mp_filter` operating on GPU hwframes (the `vf_d3d11vpp.c`/`vf_vavpp.c` pattern).

**Reference "implementation" ("mpv-v6", `C:\Users\<user>\Downloads\mpvzero`):** inspected 2026-06-10 — the six PNGs are an *architecture document* (diagrams, build-flag tables, perf claims), not screenshots of a working build, and some numbers are implausible (~0.15 ms for a 2x TRT upscale). Its own memory layout shows **pinned host buffers + DMA + `nvdec-copy`** — i.e. that design routes every frame through host memory, exactly what this plan eliminates. Treat it as inspiration only: keep its mpv.conf tuning candidates (`swapchain-depth=3`, `display-resample`, `vulkan-async-compute/transfer`), ignore its perf table, do not cite it as validation. The real in-tree validation is mpv's existing lavfi-CUDA path (`f_auto_filters.c:108` auto-inserts `bwdif_cuda` on CUDA frames). Two ideas retained on their own merits: (a) a synchronized GPU-side copy out of the decoder ring; (b) CUDA Graphs capturing the whole pre→infer→post chain (~0.05 ms launch vs ~1–2 ms).

**Why the sync copy (corrected rationale):** the NVDEC surface pool is fixed-size; an inference filter that holds frames for pipelining (and RIFE's N/N+1 buffering) starves or deadlocks the decoder. Copy-on-arrival decouples frame lifetime from the decoder ring. "No host round-trips + synchronized" is the target, not literal zero-copy. The invariant to preserve: pre-processing, inference, and post-processing all happen on the same accelerator with no CPU intervention — never "convert format in C real quick."

**In-tree proof the handoff works today:** mpv already auto-inserts the libavfilter CUDA filter `bwdif_cuda` (`filters/f_auto_filters.c:108`) — CUDA frames already flow filter→vo_gpu-next in production. The generic pool helper `mp_update_av_hw_frames_pool` (`video/mp_image_pool.c:364`) is already CUDA-aware (`filters/f_hwtransfer.c:246`). The render side maps any sw_format generically (`video/out/hwdec/hwdec_cuda.c:178-187`) via the CUDA↔Vulkan interop (`hwdec_cuda_vk.c`).

**Decisions locked:**
- Synchronized sync-fence copy model; NVIDIA/TensorRT first, DirectML later, NCNN deferred pending demand.
- RIFE stays a release requirement, sequenced as its own milestone (Phase 1.5).
- **Single mpv filter (`vf_animejanai`) + C-ABI inference shim (`libaji`) in a dedicated repo** (e.g. `the-database/animejanai-inference`).
- **Windows keeps mpv.net** (forked libmpv swapped in); **Linux ships standalone forked mpv**.

## Architecture: one thin filter + a C-ABI shim

```
FFmpeg demuxer → NVDEC decode → IMGFMT_CUDA frame
  → vf_animejanai (mpv fork, mingw-built):
      sync-fence copy out of decoder ring
      → calls libaji shim via C ABI (dlopen/LoadLibrary)
         [shim: NV12/P010→RGB norm NCHW kernel → TRT infer → NCHW→output kernel,
          all captured in one CUDA Graph, multi-stream, fp16]
      → emits IMGFMT_CUDA hwframe from its own AVHWFramesContext pool
  → vo_gpu-next / libplacebo (Vulkan) via hwdec_cuda_vk: HDR, color, final scaling
  → display
```

**Why a runtime C-ABI boundary (not just a shared source module):**
1. **Toolchain:** mpv Windows builds are mingw-w64/clang (GNU ABI); TensorRT is MSVC C++ ABI. They cannot link directly. CUDA handles (`CUcontext`, `CUstream`, `CUdeviceptr`) cross a C ABI trivially; TRT's C++ classes don't cross a GNU/MSVC boundary at all.
2. **Kills the TRT version lock structurally** — swap shim+TRT without rebuilding mpv.
3. **One mpv filter forever** — DML/NCNN become new shim implementations (ORT already has a C API); the mpv-side surface exposed to upstream rebase churn stays minimal.
4. **Standalone CLI test harness** — raw YUV in → shim → raw out, diffable against the current pipeline without a player in the loop.

**Shim API sketch (C, versioned):**
```c
aji_ctx *aji_create(const aji_create_params *p);   // CUcontext, animejanai.conf path, cache dir, log callback
int  aji_select_chain(aji_ctx*, int w, int h, double fps, int slot); // conf chain match; builds/loads engine
int  aji_infer(aji_ctx*, const aji_frame *in, aji_frame *out, void *cu_stream);      // device ptrs + strides
int  aji_infer_pair(aji_ctx*, const aji_frame *in[2], double t, aji_frame *out);     // RIFE
const char *aji_stats_json(aji_ctx*);              // for the stats overlay
void aji_destroy(aji_ctx*);
```

**Division of labor.** The shim owns everything CLI-testable: `animejanai.conf` parsing, chain selection (`min_px/max_px/min_fps/max_fps` port of `run_animejanai_with_keybinding`), engine cache (CRC-of-settings filename **+ TRT version string in the hash**), trtexec orchestration, pre/post CUDA kernels, CUDA Graph capture, stats. The filter owns everything mpv-shaped: registration, options, `vf-command` handling, hwframe pools, the sync-fence copy, frame buffering for RIFE, PTS fabrication, hr-seek skip, publishing stats to Lua.

**The pre-kernel is the quality hotspot.** It is not "RGB→NCHW" — it's NV12/P010 → chroma upsampling (filter quality matters for an upscaler's input) → BT.601/709/2020 matrix, limited/full range → normalize → NCHW. This reimplements a slice of zimg in CUDA; verified numerically (see Verification), not by eye.

**Alternatives considered (kept for the record):**
- *Self-build vstrt.dll + strip downloads (~1 week):* solves version-lock + bloat, stays host-round-trip. Remains the Phase 0 fallback.
- *Custom libavfilter CUDA filter:* mpv's lavfi bridge already passes `hw_frames_ctx` (`filters/f_lavfi.c:489`), loads hwdec devices into graphs (`f_lavfi.c:553-570`), and forwards `vf-command` (`f_lavfi.c:810-834`). Pros: ffmpeg-CLI testability, stable public API, reusable in transcode pipelines. Cons: maintain an FFmpeg fork instead. Decision: mp_filter in the mpv fork (already an accepted cost, carrying `150a4b6dba`); the shim keeps a later lavfi port cheap — this is the escape hatch if mpv-internal churn ever bites.
- *GLSL user shaders (Anime4K/ArtCNN-class):* documented fallback for non-NVIDIA users until/unless a DML/NCNN backend ships. No TRT-class models, no RIFE.
- *Cheap in-place win:* `use_cuda_graph=True` in `core.trt.Model` (animejanai_core.py:143) — **do immediately regardless**, independent of this plan.

## Where the work happens (three repos)

1. **The mpv fork — this checkout** (`the-database/mpv`): `video/filter/vf_animejanai.c`, registration, build wiring. Already carries the hr-seek patch `150a4b6dba` ("vf_vapoursynth: fast seek for ML upscalers"), whose `get_hrseek` infrastructure in `filters/f_output_chain.c/h`, `filters/filter.h`, `player/playloop.c` is filter-agnostic — **reuse it in vf_animejanai** for the same drop-pre-target-frames behavior.
2. **New shim repo** (`the-database/animejanai-inference`): `libaji` core + `aji-trt` backend (MSVC + TRT + CUDA kernels), CLI harness, MSVC/CUDA CI, versioned releases bundling shim + TRT runtime.
3. **BuildMpvUpscale2xAnimeJaNai**: assembler rewrite (Phase 2), config/keybinding migration.

## Phases

### Phase 0 — De-risk spike (GO/NO-GO), ~2–3 weeks
Day 1, before any code: smoke-test the handoff with `--vf=lavfi=[scale_cuda=1920:1080]` on an nvdec-decoded file — this exercises the exact CUDA-frame-filter→vo_gpu-next path.
- Skeleton `vf_animejanai.c` from the `vf_d3d11vpp.c` template: accept `IMGFMT_CUDA`, allocate output pool via `mp_update_av_hw_frames_pool` (`vf_d3d11vpp.c:473` pattern), run a hardcoded prebuilt engine single-stream, emit `IMGFMT_CUDA`. Register in `vf_list[]` (`filters/user_filters.c:83-109`), add to `meson.build` behind a feature flag (pattern at `meson.build:1409-1412`).
- **Prove the toolchain boundary:** mingw-built mpv `dlopen`s an MSVC-built shim DLL sharing `CUcontext`/`CUstream` across the C ABI. (Strict C ABI: no C++ types, no CRT ownership crossing the boundary; allocator stays on one side.)
- **Pin the output format:** (a) RGB(A) fp16 CUDA hwframe — verify FFmpeg `hwcontext_cuda` supports it as sw_format (8-bit RGB0/BGR0 are supported; fp16 likely needs a one-line FFmpeg-fork patch — cheap, the Windows build compiles FFmpeg from source anyway), mpv-side mapping is format-generic (`hwdec_cuda.c:178-187`); or (b) post-kernel back to NV12/P010/YUV444P16. Decide with 10-bit/HDR sources in mind; 8-bit RGB silently caps them.
- **Capture golden outputs now** from the current vsmlrt pipeline (per model × a few sources incl. 10-bit) while it still runs — these become the parity-harness reference.
- Measure ms/frame 1080p→4K with a Compact engine; confirm no `hwdownload`/autoconvert insertion (verbose log), GPU residence throughout, no tearing.
- **Gate:** if the output-hwframe/render handoff or the toolchain boundary can't be made clean, stop and fall back to self-built vstrt.dll + stripped downloads.
- **GATE CLOSED 2026-06-10: GO.** Both slices passed. WSL: shim chain 3.77 ms/frame (1080p→4K fp16, RTX 5090), mpv headless decode→infer→emit ~160 fps, parity PNGs correct. Windows: mingw mpv.exe + MSVC aji.dll crossed the C ABI in production shape (TRT 10.8 from the v3.2.0 package), 2.78 ms/frame harness, Win↔Linux output PSNR 69.3 dB, fullscreen display playback via gpu-next/Vulkan with zero hwdownload in the log, user-confirmed smooth/clean/upscaled (1920×1080 NV12→4K and 1400×1080 P010→2800×2160 both exercised).

### Phase 1 — Production filter + shim (NVIDIA), ~5–7 weeks
- Shim: pre/post CUDA kernels (matrix/range/chroma-siting correct per frame metadata), CUDA Graph capture of the full chain, multi-stream context pool, conf parsing + chain selection port, engine cache (CRC + TRT version), on-first-play `trtexec` build, stats JSON.
- **PHASE 1 CLOSED 2026-06-11.** All of the above landed except multi-stream, which is **deliberately deferred with data**: the chain runs 3.2 ms/frame (1080p→4K, RTX 5090) ≈ 310 fps against a 24–60 fps playback budget, and unlike vsmlrt (which used `num_streams` to hide VapourSynth/Python per-frame latency) the native path has no host hops to overlap. mpv's filter model is single-frame pull; N-in-flight would add N frames of latency for zero realtime benefit at current model costs. Revisit for RIFE (doubled output rate) or if a future model exceeds ~30 ms/frame. CUDA Graphs: captured (bit-identical output, auto-fallback, `AJI_NO_GRAPH=1` opt-out); perf-neutral on the 5090 (~0.2 ms of kernels in a 3.2 ms chain; launch overhead invisible), kept for CPU-bound setups. Kernel parity vs VS/zimg: pre 107.8–123.3 dB, post bit-exact–120 dB, resize 78–83 dB; end-to-end vs vsmlrt on the same engine 70.5–75.7 dB (max 1 LSB). Slot 0 = bypass (filter never leaves the chain — profile rebuilds race with queued vf-commands); keybindings migrated on sibling branch `native-filter-migration`; hr-seek pre-target skip via refqueue drop gate; dedicated non-blocking stream (ffmpeg's CUDA ctx leaves stream NULL → TRT default-stream syncs, no graph capture).
- **CLI parity harness (deliverable):** Y4M/raw in → shim → PSNR/SSIM vs Phase 0 goldens. Catches colorimetry bugs eyeballs can't.
- Filter: options via the standard priv-struct + `m_option` table (pattern `vf_d3d11vpp.c:86-95`, `796-812`, entry `814-829`); runtime slot switching via the `command` hook (`filters/filter_internal.h:98`, `MP_FILTER_COMMAND_TEXT` arrives from `vf-command`, `player/command.c:7019-7043`; example handler `f_lavfi.c:810-834`); sync-fence copy; hr-seek skip reusing `150a4b6dba` infra.
- Keybindings: `input.conf` profiles move from `vf=vapoursynth=…` to `vf=animejanai=slot=N` / `vf-command`. Preserve Shift+1/2/3, Ctrl+1–9, Ctrl+0 off.
- Stats overlay: filter publishes shim stats via an mpv property (or keeps writing `currentanimejanai.log`) so `animejanaistats.lua` / Ctrl+J keep working.

### Phase 1.5 — RIFE milestone, ~2–3 weeks
- Filter mode buffering N/N+1, emitting multiple output frames per input. In-tree precedent for >1-out-per-1-in with fabricated PTS: deinterlace field doubling via `MP_MODE_OUTPUT_FIELDS` (`video/filter/refqueue.c:175-202`, second field at `pts + frametime/2`).
- Shim `aji_infer_pair` with RIFE shapes (pad-to-mod-32), engine via the same trtexec path. Replaces `rife_cuda.py`/`vsmlrt.py`.
- The hard part is VFR timestamps, PTS monotonicity, and interaction with hr-seek soft reset — that's why it's its own milestone. **The Phase 2 release gate includes RIFE parity.**

### Phase 2 — Build & distribution overhaul (BuildMpvUpscale2xAnimeJaNai), ~2–3 weeks
- `Program.cs`: **remove** `InstallPortableVapourSynth`, `FixPythonPth`, `InstallPythonDependencies`, `InstallPythonVapourSynthPlugins`, `InstallVapourSynthMiscFilters`, `InstallVapourSynthAkarin`, `InstallVsmlrt`, `VsMlrtVersion`/`VapourSynthVersion` constants, `vsmlrtModelsPath` cleanup. **Add:** fetch forked mpv build + shim release + TRT runtime + CUDA redistributables. **Keep:** `InstallRife` (models move under `animejanai/onnx/rife/`) and `InstallYtDlp`.
- **Windows frontend: keep mpv.net** — swap in the forked `libmpv-2.dll` (the filter ships inside libmpv). Linux: standalone forked mpv.
- **Bloat attack:** evaluate TRT ≥ 8.6 hardware-compatible (`ampere+`) + version-compatible prebuilt engines shipped per model, so most users skip `nvinfer_builder_resource` (~1.3 GB). Measure the few-% perf cost first. Builder + trtexec become an optional download (Turing/Pascal, custom models).
- Rework `deploy.yml` (pre-existing `dotnet-version: '8.x'` vs `net10.0` mismatch).

**NVIDIA milestone = Phases 0 + 1 + 1.5 + 2 ≈ 3–4 months realistic for one person.**

### Phase 3 — DirectML backend (Windows non-NVIDIA), ~4–6 weeks, after the NVIDIA build ships
- `aji-dml` shim implementation via ONNX Runtime's C API; mpv filter unchanged except accepting `IMGFMT_D3D11`.
- The real work is unchanged from v1: D3D11↔D3D12 shared handles + fence sync inside the shim. No host round-trip.

### Deferred pending demand — NCNN/Vulkan backend
- Not scheduled. v1's premise was outdated: `video/out/hwdec/hwdec_vulkan.c` exists (FFmpeg ≥ 6.1 Vulkan decode, all vendors) — if this is ever built, it's `hwdec=vulkan` + NCNN sharing the device/images, not a decode bridge. Audience is Linux ∩ non-NVIDIA ∩ realtime-AI-upscaling; until demand shows up, document GLSL shaders as the non-NVIDIA fallback.

### Phase 5 — Config editor & Linux packaging, ~1–3 weeks, after NVIDIA milestone
- Linux distribution: self-contained tarball/AppImage built on the oldest supported Ubuntu LTS; documented build-from-source path; skip Flatpak/Snap/Docker for v1; never bundle the NVIDIA kernel driver; verify TRT redistribution under NVIDIA's SLA (vs-mlrt publicly redistributing TRT DLLs in GitHub releases is precedent it's tolerated — still verify).
- Config editor: document config-file editing for Linux first; an Avalonia rewrite of `AnimeJaNaiConfEditor` is **not committed** until the NVIDIA build has shipped and demand is shown.

## Critical files

mpv fork (this checkout):
- `video/filter/vf_d3d11vpp.c` — primary template (pool at :473, options :86-95/:796-812, entry :814-829); `video/filter/vf_vavpp.c` secondary.
- `filters/user_filters.c:83-109` (`vf_list[]`) — registration; `meson.build:1409-1412` — conditional build pattern.
- `video/mp_image_pool.c:364` (`mp_update_av_hw_frames_pool`, already CUDA-aware via `filters/f_hwtransfer.c:246`).
- `video/out/hwdec/hwdec_cuda.c`, `hwdec_cuda_vk.c` — render handoff.
- `filters/filter_internal.h:98` + `player/command.c:7019-7043` — `vf-command` plumbing; `filters/f_lavfi.c:810-834` example handler.
- `video/filter/refqueue.c:175-202` — multi-output PTS precedent for RIFE.
- Commit `150a4b6dba` — hr-seek infra to reuse (`filters/f_output_chain.c/h`, `filters/filter.h`, `player/playloop.c`).
- Anti-templates: `video/filter/vf_vapoursynth.c:141-172`, `filters/f_autoconvert.c:276`.
- New: `video/filter/vf_animejanai.c`.

Shim repo (new): `aji.h` (C ABI), `aji-trt` backend, pre/post kernels, CLI harness.

BuildMpvUpscale2xAnimeJaNai: `Program.cs`; `animejanai/core/animejanai_core.py`, `animejanai_config.py`, `rife_cuda.py` (port → delete); `profiles/*.vpy` (delete); `portable_config/input.conf`, `mpv.conf`, `scripts/animejanaistats.lua` (migrate).

## Verification

- **Day 1:** `--vf=lavfi=[scale_cuda]` handoff smoke test passes before writing code.
- **Phase 0:** verbose log shows no `hwdownload`/autoconvert insertion; GPU memory residence end-to-end; no tearing; ms/frame at 1080p→4K Compact ≈ current vsmlrt path; toolchain boundary proven.
- **Parity harness (Phase 1, continuous):** PSNR/SSIM vs vsmlrt goldens per model incl. 10-bit/BT.2020 sources — colorimetry, range, chroma siting.
- **Phase 1:** Shift+1/2/3 and Ctrl+1–9 cycle correctly; chain selection matches old behavior per resolution/fps; first-play engine build works; Ctrl+J stats populate.
- **Phase 1.5:** frame count multiplies correctly; PTS monotonic; A/V sync drift ≈ 0 over 10 min; seeking clean with hr-seek skip.
- **Phase 2:** clean-machine install; A/B watch vs current shipped release (smoothness, HDR, 10-bit); per-frame budget vs the reference's ~7–18 ms; headroom at 60 fps.

## Risks

- **Phase 0 unknowns** (narrower than v1, given the in-tree `bwdif_cuda` precedent): our own output pool + the toolchain boundary. Gate fallback: self-built vstrt.
- **C-ABI discipline:** no C++/CRT objects across the mingw↔MSVC boundary; one side owns each allocation.
- **FFmpeg fork patch** may be needed for fp16 RGB sw_format in `hwcontext_cuda` (only if RGB-out is chosen).
- **RIFE** VFR/PTS complexity — contained as its own milestone.
- **Prebuilt hw-compatible engines** are Ampere+ only — keep the optional builder download for older GPUs.
- **mpv internal API churn** on rebases — exposure minimized to one thin filter + the small hr-seek infra patch.
- **mpv.net is in maintenance mode** — coupling is only the libmpv swap; revisit if it breaks against newer libmpv.
- **TRT redistribution license** — verify before Phase 2 despite the vs-mlrt precedent.
- **No working reference exists** — "mpv-v6" turned out to be a design document whose memory layout is host-roundtrip (`nvdec-copy` + pinned DMA); estimates assume building and proving everything ourselves.

## Removed at the end

VapourSynth runtime, Portable Python, `vsmlrt.py`, `vstrt.dll`/`vsort.dll`/`vsov.dll`/`vsncnn.dll`, the `vsmlrt-cuda/` blob, MiscFilters/Akarin/ffms2, all `.vpy` shims, the `animejanai/core/` Python tree. Replaced by: forked mpv (libmpv for mpv.net on Windows, standalone on Linux) + `libaji` shim + TRT runtime + the native filter.
