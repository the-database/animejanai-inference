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

## Notes

- SPANF3's `SPABV2` uses `nn.ReLU(inplace=True)`, so the exported `last_mid` fed to `conv_cat` is
  `relu(out1)` (`block_N/act1/Relu`), not the raw conv output.
- Scripts carry absolute scratch paths from the spike session; `sweep/sweep_prep.py` imports
  `spanf3_arch.py` from traiNNer-redux through a stub `traiNNer.utils.registry`.
