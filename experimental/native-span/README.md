# Native SPAN-family kernels (spike)

Go/no-go spike for running SPAN-family (and later Compact) models on hand-written CUDA kernels inside
libaji instead of TensorRT. Not built by CMake; nothing here ships.

## Why

NeoRTX (the closed NVIDIA pack for cHiDeScaler-Neo) runs `2x_AnimeJaNai_HD_V3.1_Performance` in
2.37–2.54 ms at 1080p→4K on an RTX 5090, against 3.42–3.72 ms for TensorRT 11.4, with output within
1 level of ONNX Runtime fp16. It does this with fused CUDA kernels (residual, tanh(1x1) gate and concat
folded into the convs, pixel-unshuffle/format conversion folded into the first and last kernels).

## Results so far (RTX 5090, 402 W power limit, 1080p input → 960x540x48 features)

| SPANF3 body (5 blocks) | ms/frame |
|---|---|
| TensorRT 11.4, same subgraph | 2.87–3.04 |
| `proto/` fused, fp16 accumulation (WMMA, untuned) | 2.18–2.28 |
| `proto/` fused, fp32 accumulation | 3.37 |

`sweep/` generalises the kernels over channel count, activation and attention type and times
random-init SPANF3 variants against TensorRT:

| Variant | TRT body | fused body |
|---|---|---|
| C32 B5 | 1.45 | 1.19 |
| C48 B5 | 2.90 | 2.24 |
| C64 B5 | 4.68 | 5.72 |
| C96 B5 | 9.34 | 9.65 |
| C64 B8 | 7.27 | 9.18 |
| C48 B5 SiLU | 3.11 | 2.40 |
| C48 B5 sigmoid−0.5 gate | 2.54 | 2.16 |
| C48 B5 no gate | 2.58 | 2.16 |

The untuned kernels lose at 64+ channels; NeoRTX still gets ~1.35× over TensorRT on its 64-channel
Compact model (`2x_AnimeJaNai_SD_V1beta34_Compact`), so that is a tuning gap.

## Phase 0 gate: passed (2026-11-13)

`v2/` runs whole models on one shared kernel family (`native_kernels.cuh`): `mma.sync` m16n8k16 with fp16
accumulation, A tiles from padded shared memory via `ldmatrix`, weights pre-packed in fragment order (one
16-byte `__ldg` per lane per fragment pair), `cp.async` halo loads, the SPANF3 1x1 attention fed straight from
the conv accumulators, and the input/output layout work (pixel-unshuffle, depth-to-space, Compact's
upsampled-input add) folded into the first and last convs. Input and output are NCHW fp16, the same as
NeoRTX's GPU-resident path. RTX 5090, 402 W power limit, three interleaved rounds with the GPU otherwise idle:

| Model | TensorRT 11.4 | NeoRTX | native | output vs ORT-CUDA fp16 |
|---|---|---|---|---|
| SPANF3 Performance, 1920x1080 → 4K | 3.62–3.67 | 2.38–2.45 | **2.00–2.03** | 58.6 dB, max 1 level (NeoRTX: 58.6 dB, 1) |
| SD V1beta34 Compact (64 ch), 720x480 | 2.42–2.45 | 1.79–1.82 | **1.71–1.74** | 61.6 dB, max 2 levels |
| SD V1beta34 Compact (64 ch), 1280x720 | 7.38–7.42 | 5.43–5.46 | **5.23–5.24** | 62.8 dB, max 1 level |

Best tile shapes: 8 warps x 2 rows for SPANF3 (48 ch), 4 warps x 4 rows for Compact (64 ch). Only tuned and
measured on the 5090 (sm_120).

Build: `v2\build.bat` (MSVC + CUDA 13.3). Data: `v2\prep_full.py` (SPANF3) and `v2\prep_compact.py` (Compact)
write weights, a real-frame input and the ORT reference into `v2\data\<name>`; `v2\cmp_out.py` compares.

## Phase 1: kernel library (`src/native`)

The v2 kernels, generalised and moved into libaji as the `aji_native` object library: 32/48/64
channels, NHWC / NCHW / pixel-unshuffled inputs, none / ReLU / PReLU / SiLU, residual / tanh(1x1) /
sigmoid−0.5 gates, NHWC or depth-to-space 2/4 outputs (with the upsampled-input add), arbitrary
widths and heights, two tile variants chosen at load time by `pick_variant`, and the multi-source
1x1 for `conv_cat`. Not linked into a backend yet. Tests: `aji_native_pack_test` (cpu),
`aji_native_conv_test` (gpu, 198 op/variant/size checks against a CPU reference, worst error
0.0056 against a 0.02 limit; a deliberately broken depth-to-space mapping fails it).

`lib/native_bench.cu` (`aji_native_bench`, built on request) assembles the gate models from
`conv.h` calls only, with variants picked by `pick_variant`. RTX 5090, two runs:

| Model | ms/frame | output vs ORT-CUDA fp16 |
|---|---|---|
| SPANF3 Performance 1920x1080 → 4K | 1.98–2.01 | 58.6 dB, max 1 level |
| SD V1beta34 Compact 720x480 | 1.64–1.69 | 61.6 dB, max 2 levels |
| SD V1beta34 Compact 1280x720 | 5.02–5.13 | 62.8 dB, max 1 level |

`pick_variant` chose 8x2 for every 48-channel conv and 4x4 for the 64-channel body, the shapes the
v2 spike had settled on by hand.

### Against libaji's TensorRT path

`lib/compare_libaji.sh`: `aji_harness` from the mpv-AnimeJaNai 3.10.0 install (TensorRT 11.4,
libaji's own engine settings, same weights and frames; its time includes the NV12→RGB→NV12
conversions, which cost ~0.05 ms at 1080p) alternated with `aji_native_bench` (model only, NCHW fp16
in/out). RTX 5090 at a 402 W power limit, three rounds, GPU otherwise idle:

| Model | libaji TensorRT | native | speed-up | native vs ORT fp16 |
|---|---|---|---|---|
| SD V1beta34 Compact (64 ch), 720x480 | 2.36–2.40 | 1.67–1.71 | 1.40× | 61.6 dB, max 2 levels |
| SD V1beta34 Compact (64 ch), 1280x720 | 7.06–7.17 | 5.13–5.19 | 1.38× | 62.8 dB, max 1 level |
| SPANF3 Balanced (64 ch x 8), 1280x720 | 2.73–2.76 | 1.92–1.94 | 1.43× | 58.1 dB, max 1 level |
| SPANF3 Balanced (64 ch x 8), 1920x1080 | 7.46–7.55 | 4.82–4.86 | 1.55× | 58.1 dB, max 2 levels |
| SPANF3 Performance (48 ch x 5), 1920x1080 | 3.64–3.65 | 2.07–2.09 | 1.76× | 58.6 dB, max 1 level |

```
aji_native_bench spanf3 v2\data\perf1080 1920 1080
aji_native_bench compact v2\data\compact480 720 480
```

## Phase 2: in libaji (`[global] native=on`)

`aji_trt` / `aji_trtrtx` lower each chain model's ONNX at configure and run it as a native step
when they can (see CLAUDE.md, "Native kernels"); otherwise they log why and use TensorRT. End to
end through `aji_harness` (NV12 in, pre-kernel, model, post-kernel, NV12 out, CUDA graph), the same
build with `native=off` and `native=on`, `lib/phase2_e2e.sh`. RTX 5090 at 402 W, three idle rounds:

| Model | TensorRT | native | speed-up | output vs TensorRT (NV12) |
|---|---|---|---|---|
| SD V1beta34 Compact, 720x480 | 2.33–2.38 | 1.60–1.64 | 1.45× | Y 62.4 dB, UV 67.6 dB, max 1 level |
| SD V1beta34 Compact, 1280x720 | 6.93–7.10 | 4.63–5.05 | 1.4–1.5× | Y 63.4 dB, UV 67.8 dB, max 1 level |
| SPANF3 Balanced, 1280x720 | 2.67–2.73 | 1.87–1.90 | 1.43× | Y 60.3 dB, UV 66.0 dB, max 1 level |
| SPANF3 Balanced, 1920x1080 | 7.33–7.45 | 4.50–4.60 | 1.62× | Y 58.9 dB, UV 64.3 dB, max 2 levels |
| SPANF3 Performance, 1920x1080 | 3.60–3.63 | 2.04–2.06 | 1.76× | Y 59.1 dB, UV 64.5 dB, max 1 level |
| V2 UltraCompact (fp32 export), 1280x720 | 12.17–12.34 | 2.56–2.61 | 4.7×* | Y 65.1 dB, UV 69.7 dB, max 1 level |

\* TensorRT 11 runs an fp32 export in fp32; the native kernels compute in fp16 (logged as "fp32
model computed in fp16"). The other rows are fp16 models on both sides.

Configure with native kernels takes ~50 ms (parse, lower, upload, time the tile variants) against
~85 ms to load a cached TensorRT engine; there is no build. A MangaJaNai ESRGAN model is refused
("unsupported op Unsqueeze on the data path") and runs on TensorRT; temporal models are not
attempted. NV12/P010 conversion is not folded into the native kernels: libaji's pre/post kernels
resample chroma with Spline36 (needs a halo) and cost ~0.05 ms at 1080p, so that is deferred.

## Phase 3: regression (`phase3/`)

Everything on the 5090, the same build with `native=off` and `native=on`. Numbers are in
`BENCHMARKS.md` ("Native kernels vs TensorRT").

| Checked | Result |
|---|---|
| Output, `aji_harness` (`matrix.sh`): 15 cases, 12 frames each, every frame compared | all within 1–2 8-bit levels |
| Sizes: 1918x1078 (959x539 inside), 1440x1080, 1280x642, 854x480, 768x576 | pass |
| P010 input; 4:4:4 10-bit source and output (`aji_encode`) | pass |
| Live action (concert, event), DVD, anime | pass |
| Chains: two native models; resize before upscale; temporal model then native; RIFE after native | pass |
| `--async-build`: a native model is active at once | pass |
| mpv playback (NVDEC, `lib=` the branch's `aji.dll`) | plays; +37% (Balanced), +41% (Compact) |
| `aji_encode` 4:2:0 / 4:4:4 FFV1 output vs TensorRT | 65–69 dB (10-bit) |
| TensorRT-RTX (`aji_trtrtx`) | native used; 1.55–1.58x; output within 1 level |
| Unsupported model (MangaJaNai ESRGAN), temporal model | TensorRT, with the reason logged |

Not covered: slot or resolution changes during playback (each configure rebuilds the native
steps, like TensorRT's), odd frame sizes (unshuffle models refuse them with a reason), GPUs other
than the 5090 (Phase 4).

## Phase 4: other GPUs (`phase4/`)

The Phase 3 matrix on Linux (the user's Ubuntu 26.04 box, a second 5090) and on rented Runpod
pods (`phase4/matrix-pod.sh`): RTX 4090 (Ada) and RTX A5000 (Ampere). Everywhere the native
tests pass and every case is within 1–2 levels of TensorRT; nothing is slower. Performance
(48 ch, memory-bound, which the fusion removes) gains ~1.7x on all three generations. The
64-channel models are compute-bound: the native kernels reach ~60% of fp16 tensor peak on every
card, TensorRT the same on Ampere but less on Ada and Blackwell, so Balanced/Compact gain
1.4–1.6x on the 5090, 1.15–1.4x on the 4090 and 1.0–1.2x on the A5000. Numbers in `BENCHMARKS.md`.
With that, native is on by default (`native=off` disables). Not measured: a small card
(3050/4060 class; none available to rent).

Mid-playback changes in mpv (`phase4/midplay.sh` + `switch.lua`, 5090, the app's own
`vf-command aji slot N`): slot switches every ~190 frames through Balanced, Compact, Performance,
off, Balanced+Performance, TemporalFix+Performance, Performance+RIFE and back each reconfigure in
0.05–0.3 s with no errors; 16 switches between native slots leave GPU memory where the first cycle
was. A stream that changes resolution four times (1080p, 768x576, 1280x642, 1080p live, 768x576)
is upscaled throughout on native with Performance, Compact and Balanced, where TensorRT passes
through each size it has no engine for until a background build finishes.

## Notes

- SPANF3's `SPABV2` uses `nn.ReLU(inplace=True)`, so the exported `last_mid` fed to `conv_cat` is
  `relu(out1)` (`block_N/act1/Relu`), not the raw conv output.
- Scripts carry absolute scratch paths from the spike session; `sweep/sweep_prep.py` imports
  `spanf3_arch.py` from traiNNer-redux through a stub `traiNNer.utils.registry`.
