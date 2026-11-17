# SST on native kernels

SST (Scalable SR Transformer, "Rank-Factorized Implicit Neural Bias", arXiv 2603.06738,
github.com/dslisleedh/SST, MIT) on libaji's image path, to compare it with HAT-L before training
one. `src/native/sst.cpp` builds it from the ONNX of `export.py`; `src/native/sst_kernels.cu` adds
the one new piece, a flash-style window attention for windows up to 96x96:

- a CTA per (window, head, 128 queries) streams 64-key blocks through shared memory (cp.async,
  double buffered), online softmax in the log2 domain, fp16 MMA with fp32 sums;
- the RIB position bias is two small per-layer tables (evaluated at load from the graph's MLP)
  concatenated to q and k as 16/32/48 extra dims, so it costs MMA k-steps, not a bias matrix;
- SST reflect-pads each layer to its window multiple: padded keys read the reflected pixels;
  query blocks wholly in the padding are skipped (row order, or column order in right-edge windows);
- the gate (sigmoid of a depthwise + 1x1 conv) multiplies the output before it is stored.

Everything else is the HAT path's kernels (tfm GEMMs and LayerNorm, gconv, fp32 residual stream),
plus a depthwise conv with residual (ConvFFN) and the 4x pixel-shuffle tail with the nearest skip.

## Files

- `sst_ref.py`: SST re-implemented for inference and export, same parameter names as upstream
  (its checkpoints load with `load_state_dict`); presets `sst`, `sst_plus`, `sst_l`, `sst_l_plus`,
  `sst_xl_plus` from upstream's test configs.
- `export.py PRESET OUTDIR [--ckpt PATH]`: fp32 opset-17 export (seeded weights without `--ckpt`);
  `--ref IN.ppm OUT.ppm` writes the PyTorch fp32 reference upscale (TF32 off).

## Accuracy (seeded weights, vs PyTorch fp32, 8-bit output)

| Preset | 512x512 tile | 300x212 (tile = image, every window padded) |
|---|---|---|
| SST-L+ | 61.1 dB, max 1 level | 60.8 dB, max 1 |
| SST (hd 30, rank 34) | 61.3 dB, max 1 | 60.9 dB, max 1 |
| SST-XL+ (7 heads) | 60.0 dB, max 1 | 59.7 dB, max 1 |

The network's residual on these weights has a 24.6-level std; native differs by 0.9% of it.

## Speed and memory (RTX 5090, aji_image, 1200x1000 page, warm, native)

| Model | Params | ms/page, 512 tiles | vs HAT-L | ms/page, 576 tiles | vs HAT-L | weights + activations (512 / 576) |
|---|---|---|---|---|---|---|
| HAT-L (IllustrationJaNai V3) | 40.8M | 3440 | 1.00x | 2949 | 1.00x | 88 + 1136 / 1438 MB |
| SST | 12.1M | 2992 | 0.87x | 2553 | 0.87x | 85 + 1160 / 1468 MB |
| SST+ | 12.1M | 4222 | 1.23x | 3588 | 1.22x | 124 + 1160 / 1468 MB |
| SST-L | 21.3M | 3487 | 1.01x | 2958 | 1.00x | 88 + 1352 / 1711 MB |
| SST-L+ | 21.3M | 4489 | 1.30x | 3797 | 1.29x | 125 + 1352 / 1711 MB |
| SST-XL+ | 34.9M | 6793 | 1.97x | 5733 | 1.94x | 189 + 1384 / 1752 MB |

Per 512 tile (AJI_NATIVE_PROFILE): HAT-L 378 ms (window attention 100 ms); SST-L 385 ms
(attention 209); SST-L+ 495 ms (attention 319, ~120 TFLOPS). SST's FLOPs (1.33x / 2.1x HAT-L's
for L / L+) mostly land in large-window attention, which runs near the tensor cores' rate; HAT-L's
mix of small GEMMs, 16x16 windows and the CAB convs runs at about half that.

576 tiles divide every SST window (16..96), so nothing is padded; HAT-L gains as much from the
larger tile (fewer tiles per page), and the ratios stay the same.

Memory: SST's activations are larger mostly because of the upsampler: gconv's 4x phase scatter
needs each output phase group to fill whole 64-channel launches, so the 96-channel tail is padded
to 128 (a 1.07 GB buffer at 512 tiles, vs 0.8 GB unpadded). The body's peak is ~1.0 GB.

Not measured: real checkpoints (the weights don't change the speed or memory), GPUs other than the
5090, the video path, TensorRT.
