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

## Native kernels vs TensorRT (`[global] native=on`, 2026-11-14)

Branch `experimental/native-span`. RTX 5090 at a 402 W power limit (default 575 W), driver 621.14,
TensorRT 11.4.0.106 and TensorRT-RTX 1.7.1, every run with no other process on the GPU. The same
build with `native=off` and `native=on`; engines cached. Models: the shipped V3.1 non-sharp
`Performance` (SPANF3 b5f48) and `Balanced` (b8f64), and `SD_V1beta34_Compact` (SRVGG 64 ch).

`aji_harness`, device chain time (NV12 in → pre → model → post → NV12 out, CUDA graph), ms/frame,
1000 frames, two or three interleaved rounds:

| Case | TensorRT | native | |
|---|---|---|---|
| Performance 1920x1080 | 3.60–3.63 | 2.04–2.06 | 1.76x |
| Performance 1918x1078 | 3.55–3.58 | 2.13–2.14 | 1.67x |
| Performance 1280x642 (live action) | 1.24–1.25 | 0.86–0.87 | 1.44x |
| Performance 960x540 (50% resize first) | 0.84–0.85 | 0.67 | 1.26x |
| Performance 854x480 | 0.68–0.69 | 0.58 | 1.18x |
| Balanced 1920x1080 | 7.33–7.45 | 4.50–4.60 | 1.62x |
| Balanced 1918x1078 | 7.25–7.36 | 4.53–4.61 | 1.60x |
| Balanced 1440x1080 | 4.83–4.91 | 3.08–3.15 | 1.57x |
| Balanced 1280x720 | 2.67–2.73 | 1.87–1.90 | 1.43x |
| Balanced then Performance, 854x480 → 3416x1920 | 3.99–4.01 | 2.57–2.59 | 1.55x |
| Compact 1920x1080 (live action) | 16.0–16.2 | 11.4–11.5 | 1.41x |
| Compact 1280x720 | 6.93–7.10 | 4.63–5.05 | 1.4–1.5x |
| Compact 854x480 | 2.83–2.87 | 1.95–1.99 | 1.44x |
| Compact 768x576 (DVD) | 3.25–3.31 | 2.10–2.15 | 1.54x |
| Compact 720x480 | 2.33–2.38 | 1.60–1.64 | 1.45x |
| TensorRT-RTX: Performance 1920x1080 | 3.11–3.14 | 2.02 | 1.55x |
| TensorRT-RTX: Balanced 1920x1080 | 6.97–7.06 | 4.43–4.48 | 1.58x |
| TensorRT-RTX: Compact 768x576 | 3.36–3.38 | 2.15–2.16 | 1.56x |

Every case's output frames (12 per case, plus P010 input, a temporal + Performance chain and
4:2:0/4:4:4 `aji_encode` output) are within 1–2 8-bit levels of TensorRT's (Y 58–65 dB).

Playback, mpv-AnimeJaNai 3.10.0's mpv with the branch's `aji.dll` (`lib=`), NVDEC,
`--vo=null --untimed`, fps = frames / (Exiting − Configured slot), two rounds:

| 1080p source → 4K | TensorRT | native |
|---|---|---|
| Balanced, anime (1200 frames) | 129.2–130.8 | 177.8–179.2 |
| Compact, live action (600 frames) | 59.1–59.7 | 83.2–83.9 |
| Performance, anime (1200 frames) | 173.9–184.3 | 192.6–195.2 (decode-bound) |

`aji_encode`, 928-frame 1080p live-action clip → 4K `hevc_nvenc`, wall seconds, two rounds:
Compact 12.4–12.6 → 9.6–9.7; Balanced 8.0–8.1 → 7.6; Performance 7.3 → 7.2 (NVENC-bound at ~129 fps).
RIFE 2x after Performance (`--rife-chain`): 19.68 → 18.03 ms per source frame.

Linux (Ubuntu 26.04, gcc 15.2, CUDA 13.2 nvcc, TensorRT 11.4.0.106, driver 615.71, a second RTX 5090
at a 400 W limit), `experimental/native-span/phase3/matrix-linux.sh`, one run on cached engines: the
same 15 cases all within 1–2 8-bit levels of TensorRT, the native tests pass, and the speed-ups match
Windows (TensorRT itself is ~6% faster on Linux):

| Case | TensorRT | native | speed-up |
|---|---|---|---|
| Performance 1920x1080 | 3.40 | 2.02 | 1.68x |
| Balanced 1920x1080 | 6.80 | 4.29 | 1.58x |
| Balanced 1440x1080 | 4.65 | 3.04 | 1.53x |
| Performance 1280x642 | 1.19 | 0.83 | 1.43x |
| Performance 854x480 | 0.64 | 0.52 | 1.22x |
| Compact 1920x1080 (live action) | 14.52 | 10.15 | 1.43x |
| Compact 768x576 (DVD) | 3.00 | 2.06 | 1.46x |
| Balanced then Performance, 854x480 | 3.66 | 2.42 | 1.51x |
| RIFE 2x after Performance, per source frame | 19.01 | 17.55 | |

### RTX 4090 (Phase 4)

Runpod pod: RTX 4090 at its 450 W default, driver 595.91 (CUDA 13.2), Ubuntu 24.04, gcc 13.3,
nvcc 13.2, TensorRT 11.4.0.106 (CUDA 13.4 build, running under minor-version compatibility).
`experimental/native-span/phase4/matrix-pod.sh`; the P010 and 1918x1078 / 1440x1080 inputs are
generated on the pod from the 1080p NV12 frames (8-bit shifted up; top-left crops). The native
tests pass (worst conv error 0.0077 against 0.02), and all 15 cases are within 1–2 8-bit levels of
TensorRT. Two rounds on cached engines:

| Case | TensorRT | native | speed-up |
|---|---|---|---|
| Performance 1920x1080 | 4.90–5.05 | 2.87 | 1.71–1.76x |
| Performance 1918x1078 | 5.03 | 3.02–3.08 | 1.63–1.67x |
| Balanced 1920x1080 | 8.15–8.17 | 5.74 | 1.42x |
| Balanced 1440x1080 | 6.19–6.21 | 4.41–4.53 | 1.37–1.40x |
| Performance 1280x642 | 1.67 | 1.01 | 1.65x |
| Performance 854x480 | 0.81 | 0.62 | 1.31x |
| Compact 1920x1080 (live action) | 15.02–15.04 | 11.13–11.14 | 1.35x |
| Compact 854x480 | 3.16–3.17 | 2.74 | 1.15x |
| Compact 768x576 (DVD) | 3.36 | 2.85–2.86 | 1.18x |
| Balanced then Performance, 854x480 | 5.20–5.21 | 3.15 | 1.65x |
| RIFE 2x after Performance, per source frame | 37.18–37.24 | 35.20–35.30 | |

Faster everywhere, but the 64-channel models (Balanced, Compact) gain less than on the 5090
(Compact at SD sizes 1.15–1.18x against 1.44–1.54x): the two tile variants were tuned on sm_120,
so this is the first place to look when tuning for Ada.
