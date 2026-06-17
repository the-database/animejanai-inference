# Benchmark ledger

Exact measured numbers, kept verbatim. All runs: RTX 5090, Windows 11
host (WSL2 noted where used), driver CUDA 13.3. Two methodologies:

- **In-package benchmark tool** (`animejanai/benchmarks/benchmark.ps1`,
  what the ConfEditor Benchmark button runs): inference-only device
  throughput (pre-processing + model + post-processing; decode
  excluded), 120 frames per cell over bundled testsrc2 seeds, built-in
  Balanced (slot 1010) / Performance (slot 1011) templates.
- **End-to-end playback**: 500 frames of the 3.3.0 package's bundled
  benchmark clips. 3.3.0 = ffms2 software decode + VapourSynth graph,
  fps as reported by vspipe. Native = hardware decode (NVDEC/D3D11VA)
  through mpv `--untimed --vo=null`, fps = 500 / (log timestamp at
  "Exiting" − timestamp at "Configured slot").

Models: `2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16`
("Balanced") and `2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16`
("Performance") — identical files on every side of every comparison.

## TensorRT 10.16 vs 11.0 — in-package tool (2026-06-12)

Both sweeps through the same tool; engines built fresh by each runtime
(TRT 11 cold builds ≈ 10 s vs ≈ 60 s on TRT 10).

Balanced (slot 1010):

| fps | 480x360 | 640x480 | 768x576 | 1280x720 | 1920x1080 |
|---|---|---|---|---|---|
| TRT 10.16 | 1904.8 | 1168.2 | 811.7 | 406.8 | 150.0 |
| TRT 11.0 | 1960.8 | 1242.2 | 865.1 | 427.4 | 155.9 |
| delta | +2.9% | +6.3% | +6.6% | +5.1% | +3.9% |

Performance (slot 1011):

| fps | 480x360 | 640x480 | 768x576 | 1280x720 | 1920x1080 |
|---|---|---|---|---|---|
| TRT 10.16 | 2994.0 | 1897.5 | 1246.9 | 652.3 | 280.9 |
| TRT 11.0 | 3164.6 | 1938.0 | 1379.3 | 709.2 | 286.0 |
| delta | +5.7% | +2.1% | +10.6% | +8.7% | +1.8% |

Counterpoint kept for honesty: a WSL2/Linux isolation on the older
`2x_AnimeJaNai_HD_V3_Performance` model measured TRT 11 ~6% *slower*
at the bare-engine level (trtexec --loadEngine: 3.42 vs 3.21 ms mean).
The in-package numbers above on the shipping V3.1 models are the
release-relevant measurement.

## 3.3.0 (VapourSynth) vs native — end-to-end playback (2026-06-12)

TensorRT backend (3.3.0 on its bundled TRT 10.16; native on TRT 10.16
the same day — predates the TRT 11 switch):

| fps | 480x360 | 640x480 | 768x576 | 1280x720 | 1920x1080 |
|---|---|---|---|---|---|
| Balanced 3.3.0 | 886.1 | 481.2 | 312.2 | 175.7 | 73.7 |
| Balanced native | 936.3 | 651.0 | 491.6 | 277.8 | 111.9 |
| delta | +5.7% | +35.3% | +57.5% | +58.1% | +51.8% |
| Performance 3.3.0 | 1108.9 | 507.3 | 337.8 | 176.2 | 77.2 |
| Performance native | 1075.3 | 818.3 | 647.7 | 378.8 | 174.7 |
| delta | −3.0% | +61.3% | +91.7% | +115.0% | +126.3% |

DirectML backend (3.3.0 via vsort; native via aji_dml, fp32 RIFE-era
models, 720p/1080p only):

| fps | 1280x720 | 1920x1080 |
|---|---|---|
| Balanced 3.3.0 | 104.0 | 43.4 |
| Balanced native | 122.9 | 55.3 |
| Performance 3.3.0 | 112.1 | 49.8 |
| Performance native | 184.8 | 91.9 |

Note: the native side runs a single frame in flight (no pipelining yet)
against VapourSynth's 8-deep request pipeline; the ~480x360 parity is
per-frame fixed cost, the target of the pipelining backlog item.

## In-package tool, DirectML (2026-06-12, fp32-era rife irrelevant here)

| fps | 480x360 | 640x480 | 768x576 | 1280x720 | 1920x1080 |
|---|---|---|---|---|---|
| Balanced | 442.7 | 389.7 | 291.7 | 157.2 | 65.7 |
| Performance | 667.1 | 588.6 | 467.5 | 265.9 | 126.2 |

## Pipelined inference (queue-depth 3) — 2026-06-12

Shim ABI v7 + filter pipelining (in-flight frame ring; see PLAN). The
in-package tool is unchanged by design (synchronous harness, one frame
in flight) — re-run with the v7 shim it reproduces the TRT 11.0 rows
within noise (Balanced 1920x1080 159.3 vs 155.9; Performance 293.4 vs
286.0), confirming no regression on the synchronous path.

End-to-end playback, same methodology as the 3.3.0-vs-native table but
on TRT 11 / live desktop, best of 3 runs per cell, `output-444=no` so
the `--vo=null` implicit hw-download stays NV12-sized and comparable
with the earlier rows (real playback maps output natively and has no
such download). depth 1 = the old synchronous behavior; depth 3 = the
shipping default.

TensorRT backend:

| fps | 480x360 | 640x480 | 768x576 | 1280x720 | 1920x1080 |
|---|---|---|---|---|---|
| Balanced depth 1 | 874.1 | 613.5 | 459.6 | 248.3 | 101.1 |
| Balanced depth 3 | 1012.1 | 679.3 | 509.7 | 269.8 | 107.1 |
| delta | +16% | +11% | +11% | +9% | +6% |
| Performance depth 1 | 1052.6 | 792.4 | 628.9 | 341.5 | 151.1 |
| Performance depth 3 | 1412.4 | 963.4 | 715.3 | 395.3 | 168.4 |
| delta | +34% | +22% | +14% | +16% | +11% |

DirectML backend (largest win — its per-frame CPU fence wait used to
serialize ORT dispatch with the device):

| fps | 1280x720 | 1920x1080 |
|---|---|---|
| Balanced depth 1 | 115.5 | 52.0 |
| Balanced depth 3 | 156.6 | 67.8 |
| Performance depth 1 | 177.1 | 87.5 |
| Performance depth 3 | 240.2 | 117.7 |
| Performance delta | +36% | +35% |

Correctness: framemd5 bit-identical depth 1 vs depth 3 on both
backends (TRT incl. RIFE slots and CUDA-graph replay, WSL; DML 2168
frames, Windows host). WSL2 isolation of the same sweep (V3
Performance, 1080p->4K, yuv444p16 + download): 100.7 -> 129.8 fps
(+29%) at depth 3.

## 3.3.0 (VapourSynth) fresh baseline vs pipelined native (2026-06-12)

User-run 3.3.0 benchmark the same day on the same machine (verbatim;
stronger than the older 3.3.0 rows above by 10-27%, so this is the
baseline of record):

```
1920x1080 2x (2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16)   :   78.99 fps
1920x1080 2x (2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16):   79.46 fps
1280x720  2x (2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16)   :  182.79 fps
1280x720  2x (2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16):  188.07 fps
768x576   2x (2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16)   :  368.96 fps
768x576   2x (2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16):  377.40 fps
640x480   2x (2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16)   :  539.62 fps
640x480   2x (2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16):  557.25 fps
480x360   2x (2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16)   : 1122.03 fps
480x360   2x (2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16): 1204.47 fps
```

Against the native depth-3 rows above (note the native side carries
the `--vo=null` hw-download handicap that the vspipe baseline does not):

| fps | 480x360 | 640x480 | 768x576 | 1280x720 | 1920x1080 |
|---|---|---|---|---|---|
| Balanced 3.3.0 | 1122.0 | 539.6 | 369.0 | 182.8 | 79.0 |
| Balanced native d3 | 1012.1 | 679.3 | 509.7 | 269.8 | 107.1 |
| delta | -10% | +26% | +38% | +48% | +36% |
| Performance 3.3.0 | 1204.5 | 557.3 | 377.4 | 188.1 | 79.5 |
| Performance native d3 | 1412.4 | 963.4 | 715.3 | 395.3 | 168.4 |
| delta | +17% | +73% | +90% | +110% | +112% |

Reading: the VS baseline plateaus at ~79 fps for BOTH models at 1080p
(Balanced == Performance), i.e. it is CPU-bound (ffms2 software decode
+ Python/zimg), not GPU-bound; the native path differentiates the
models properly and wins 9 of 10 cells. The sole loss (480x360
Balanced, -10%) is the per-frame fixed-cost corner at >1000 fps,
42x the realtime budget.

## RIFE interpolation cost, DirectML (aji_harness_dml --rife, ms/interp)

| | 1920x1080 | 3840x2160 |
|---|---|---|
| fp32 models | 13.8 | 55.4 |
| fp16 models (`models-rife-fp16-1`) | 11.1 | 41.0 |

TensorRT RIFE at 4K runs in single-digit ms (engine-dominated) on both
TRT versions.

## RIFE/upscale order — `aji_harness --rife-chain` (2026-06-17)

Built-in slots 1012/1013 are the same HD Balanced + RIFE 2x chain in the two
orders (1012 = upscale then RIFE; 1013 = RIFE then upscale, the new default).
`aji_harness --rife-chain` times the full chain per source frame:

```
aji_harness --conf test-animejanai.conf --model-dir onnx --rife-model-dir rife-fp16 \
    --trtexec trtexec.exe --slot 1012 \
    --input 1920x1080.raw --width 1920 --height 1080 --format nv12 --frames 240 --rife-chain
# then --slot 1013
```

RTX 5090, TRT 11, CUDA 13.3; 1920x1080 -> 3840x2160, RIFE v4.14 2x; device
chain time per source frame, 240-frame loop (237 timed):

| order | ms/source-frame | source fps | output fps |
|---|---|---|---|
| 1012 upscale -> RIFE | 31.2 | 32.1 | 64.2 |
| 1013 RIFE -> upscale (default) | 21.7 | 46.0 | 92.0 |

**RIFE-first wins ~1.4x — the opposite of the naive expectation.** The cost is
dominated by RIFE, not the upscaler: the SPAN Balanced upscale (1920->3840) is
~5 ms, RIFE at 1080p ~6 ms, RIFE at 4K ~25 ms. Upscale-first pays 4K RIFE every
frame; RIFE-first pays 1080p RIFE plus a second cheap upscale pass. The RIFE
factor scales the margin (5x widens it) but never flips the winner; a much
heavier upscale model eventually would. Measured synchronous/single-stream (the
harness model); real playback pipelines, but the per-frame ordering cost carries.
