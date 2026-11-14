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
| SD V1beta34 Compact (64 ch), 1280x720 | 7.38–7.41 | 5.43–5.45 | **5.23–5.24** | 62.8 dB, max 1 level |

Best tile shapes: 8 warps x 2 rows for SPANF3 (48 ch), 4 warps x 4 rows for Compact (64 ch). Only tuned and
measured on the 5090 (sm_120).

Build: `v2\build.bat` (MSVC + CUDA 13.3). Data: `v2\prep_full.py` (SPANF3) and `v2\prep_compact.py` (Compact)
write weights, a real-frame input and the ORT reference into `v2\data\<name>`; `v2\cmp_out.py` compares.

## Notes

- SPANF3's `SPABV2` uses `nn.ReLU(inplace=True)`, so the exported `last_mid` fed to `conv_cat` is
  `relu(out1)` (`block_N/act1/Relu`), not the raw conv output.
- Scripts carry absolute scratch paths from the spike session; `sweep/sweep_prep.py` imports
  `spanf3_arch.py` from traiNNer-redux through a stub `traiNNer.utils.registry`.
