# Native kernels for more networks (spikes)

Which networks beyond SPAN/SPANF3/Compact (`src/native`) are worth native kernels: ESRGAN
(MangaJaNai 2x/4x), TFDAT v2 (bndl animefilm), FDAT_M (IllustrationJaNai), RIFE, TemporalFix.
Each spike measures TensorRT on the shipped model, works out where its time goes, and times the
existing native kernel template on the network's shapes. Nothing here ships. RTX 5090, 402 W
limit, TensorRT 11.4.0.106 with libaji's image-path build settings (`--builderOptimizationLevel=3`;
level 5 measured the same), cached engines, a 512x512 input tile (the image path's default tile).

## Summary (RTX 5090, TensorRT 11.4)

| Model | Path | TensorRT | Where TensorRT loses | Native estimate | Effort |
|---|---|---|---|---|---|
| RIFE v4.26 | video, every RIFE chain | 19.3 ms @4K, 4.06 @1080p | 76% full-res warp/resize kernels; thin convs at ~9 TFLOPS | 4–6x @4K, ~3x @1080p | medium |
| TemporalFix | video (temporal) | 11.7 ms @1080p, 4.76 @720p | tiny model run as ~126 kernels; convs at ~8 TFLOPS | 5–8x @1080p | medium, one model |
| FDAT_M 2x/4x | image | 14.75 / 73.8 ms per 512 tile | ~550 small kernels, ~8% of peak | 2–3x / 3–4x | high |
| TFDAT (bndl) | video (temporal) | 87 ms @720p | 71% is the FDAT body | ~2x (body) to ~3x | high, shares FDAT |
| DAT-2 4x (bf16) | image | 331 ms per 512 tile | >25% pure data layout; ~19 TFLOPS | ~4x | highest, follows FDAT |
| HAT-L 4x (bf16) | image | 572 ms per 512 tile | >30% data layout; ~44 TFLOPS | ~3x | highest, follows FDAT |
| ESRGAN 2x/4x | image | 19.1 / 82.5 ms per 512 tile | generic sm80 convs; 276 concat copies | 1.5–1.9x (kernels measured) | medium |

Only ESRGAN's estimate rests on measured native kernels; the rest come from profiles and FLOP
counts. All image-path models also need native support in `aji_img_trt`, which has none today.

## ESRGAN (MangaJaNai V1, `esrgan/`)

RRDBNet: 23 RRDB x 3 dense blocks, 64 features, growth 32, LeakyReLU 0.2, residual scale 0.2.
2x has a 2x pixel-unshuffle in front (body at half resolution), 4x runs the body at full
resolution; both end with two nearest-x2 upsample + conv stages. Per input pixel: 2x ~9.0 MFLOP,
4x ~35.9 MFLOP (Compact is ~1.2), so it is compute-bound.

TensorRT runs every conv on generic `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16` kernels (fp16
accumulation, not Blackwell-specific) and materialises each dense-block concat as its own copy
kernel (`__myl_Conc`, 276 of them):

| 512x512 tile | TensorRT total | convs | concat copies | overall rate |
|---|---|---|---|---|
| 2x | 19.1 ms | 16.6 ms (~141 TFLOPS) | 3.1 ms (15%) | ~123 TFLOPS |
| 4x | 82.5 ms | 57.4 ms (~164 TFLOPS) | 20.9 ms (26%) | ~114 TFLOPS |

(Per-layer times from `--separateProfileRun`; they sum to a little more than the graph time.)

The existing `k_conv3x3` template, instantiated at ESRGAN's shapes with no code changes
(`densebench.cu`; shared memory opted in above 48 KB), best tile shape per conv, 256x256 (the 2x
body of a 512 tile), TFLOPS:

| conv | share of body FLOPs | native | 
|---|---|---|
| 64→32 | 8% | 170 |
| 96→32 | 12% | 177 |
| 128→32 | 15% | 197 |
| 160→32 | 19% | 149 |
| 192→64 | 46% | 191 |
| 64→64 (upsampler, `convbench.cu`) | — | 236 (283 at 850x600) |

Estimate for a native ESRGAN, concats free (each dense conv writes its 32 channels into a slice of
one 192+32-channel buffer and the next reads a prefix of it) and the residuals folded into the
epilogues: 2x ~12.8 ms (1.5x), 4x ~51 ms (1.6x) on the 5090 with today's kernels. A K-chunked main
loop (so cin 160/192 can use the larger tiles that shared memory rules out now) is the obvious
tuning step; at ~220 TFLOPS for the dense convs it would be ~1.9x. On Ampere, TensorRT's sm80
kernels are native, so the conv part probably ties (as Compact did on the A5000) and the gain is the
removed concat copies (~15–26%).

What it needs: an input pixel stride and an output channel offset (the concat buffer); cin up to
192 and cout 32 in the table (K-chunking or opted-in shared memory); epilogues `x + 0.2·conv` and
the RRDB's `x + 0.2·(y + 0.2·conv)`; a nearest-x2 input mode for the upsampler convs; an NCHW RGB
output with cout 3; lowering for the dense-block pattern; and native support in the image path
(`aji_img_trt`, tiled), which today has none. That last part is what MangaJaNai GUI, the CLI and
yosh would use.

## FDAT_M (IllustrationJaNai V3, `fdat/`)

traiNNer-redux `fdat_medium`: 120 channels, 4 groups x 6 blocks, alternating 8x8-window spatial
attention (4 heads, head dim 30, learned relative bias) and channel attention (per head a 30x30
Gram matrix over all tokens, L2-normalised q/k, learned temperature). Each block also has a
depthwise 3x3 + GELU branch mixed in by an AIM gate (a 1x1 sigmoid spatial gate, or a global-pool
channel gate), LayerNorms, and an FFN (120→240, GELU, depthwise 3x3 on 240, 240→120). The 2x model
pixel-unshuffles by 2 and upsamples by 4; the 4x model runs the body at full resolution.
~7.75 MFLOP per body token, 71% of it in the four GEMMs per block (qkv, proj, fc1, fc2).

TensorRT, 512x512 tile: **2x 14.75 ms (~34 TFLOPS, ~8% of fp16 tensor peak), 4x 73.8 ms
(~28 TFLOPS)**. Unlike ESRGAN this is nowhere near compute-bound. Where the 2x time goes
(`profile_cats.py`, 553 layers, ~23 kernels per block):

| | ms | share |
|---|---|---|
| GEMMs (`sm80_xmma_gemm`, 256x128 tiles; K = 120) | 6.55 | 39% (~55 TFLOPS) |
| depthwise convs (`implicit_gemm_indexed_wo_smem`) | 2.22 | 13% |
| attention as slice/reshape/mul/sum (`SlicReshMoveMulSum`, 24) | 1.98 | 12% |
| GELU + LayerNorm + gate fusions, transposes, copies (Myelin) | ~5.9 | 36% |

So most of the time is memory traffic between ~550 small kernels (one 120-channel activation of
a 512 tile is 15.7 MB; at 4x it is 63 MB and leaves L2) and GEMMs too thin for the 256x128 tiles
TensorRT picks. A native FDAT would fuse each block into ~5 kernels: LayerNorm in the prologue of
the GEMM that follows it (K = 120, so a row fits), GELU and residuals in epilogues, the window
attention as one kernel per 8x8 window (64x64 scores in registers), the channel attention as a
split reduction for the Gram matrices then an apply, and the FFN's fc1 + GELU written once for its
depthwise conv. Rough ceiling: 2x ~3–5 ms, 4x ~12–16 ms (GEMMs at 150–200 TFLOPS plus the
remaining passes); realistically **2–3x (2x model) and 3–4x (4x model)**, more on every GPU
generation than for ESRGAN since TensorRT's problem here is traffic, not conv kernel tuning.
Not measured with a prototype: these are estimates from the profile and the FLOP count.

Cost: the largest of the candidates. New kernel kinds (fused LayerNorm-GEMM, window attention,
global reductions for the channel attention and channel gate, depthwise convs), lowering for a
~2,400-node graph whose data path is mostly reshapes and transposes, and FDAT-specific patterns
(a new transformer variant would need its own work). Like ESRGAN it runs in the image path.

## TFDAT (bndl animefilm v3, `tfdat/`)

`2x_bndl_animefilm_v3_1x27xHxW`: a 9-frame temporal model. Not traiNNer-redux's `tfdat` exactly:
its alignment is heavier (a 3-level 32/64/96-channel per-frame encoder `enc*`, per-neighbour
flow heads, an `align_refine` step and an occlusion mask, bilinear warping as `GatherElements`),
but the backbone is the same FDAT body as FDAT_M (120 channels, 24 blocks, window + channel
attention) on the pixel-unshuffled centre frame. In libaji it is a temporal model (video path,
`t > 1`), which the native path does not attempt today.

TensorRT (`--builderOptimizationLevel=5`, libaji's video default), per output frame: **1280x736
87.3 ms (11 fps), 1920x1088 205 ms (5 fps)**. By sub-network (`split.py`, from the layer metadata):

| | 1280x736 | 1920x1088 |
|---|---|---|
| FDAT body (24 blocks) | 62.0 ms (71%) | 146.2 ms (72%) |
| alignment (flow, warp, refine) | 9.0 ms (10%) | 19.2 ms (10%) |
| temporal fusion + conv_first | 7.1 ms (8%) | 16.6 ms (8%) |
| upsampler + hr_refine | 3.8 ms (4%) | 8.8 ms (4%) |
| per-frame encoder | 1.5 ms (2%) | 2.8 ms (1%) |
| other | 3.9 ms | 8.8 ms |

The body is FDAT_M's at 3.6x the tokens (640x368 vs a 256x256 tile body): 62 ms against FDAT_M's
14.75 ms x 3.6 = 53 ms, the same ~8%-of-peak profile, with 56 MB activations that leave L2. So a
native FDAT body serves both IllustrationJaNai and bndl. With the body at the FDAT estimate (3–4x
at this size) and the rest left on TensorRT, 720p would go from ~87 to ~40 ms (~2x); with the
alignment and fusion native too, ~25–30 ms (~3x), i.e. from 11 fps to 25–35 fps at 720p. The
per-frame encoder's features could be cached across frames (8 of 9 inputs repeat), but that is
only ~2%.

## RIFE (v4.14, v4.26; `rife/`)

IFNet: 4 (v4.14) or 5 (v4.26) IFBlocks, coarse to fine. Each block warps both frames by the
current flow (`GridSample`), resizes the warped frames (and the features) down to its scale, runs
a stride-2 encoder and 8 residual 3x3 convs (`conv·β + x`, LeakyReLU; 32–192 channels), predicts a
flow/mask delta with a transposed conv + depth-to-space, and resizes it back up to full resolution.
The end warps both frames once more and blends them by a sigmoid mask. libaji feeds it 11 channels
(both frames, the timestep, flow-scale and grid planes) at the padded output size; the ~40 shipped
RIFE ONNX files (4.7–4.26, lite, ensemble) are all this family.

TensorRT (libaji's RIFE settings: strongly typed, fp16 I/O), per interpolated frame:

| | 1920x1088 | 3840x2176 |
|---|---|---|
| v4.14 | 4.8 ms | 21.0 ms |
| v4.26 | 4.06 ms | 19.3 ms |

At 4K, v4.26 spends 4.3 ms in convs (0.29 TFLOP, ~68 TFLOPS) and **14.0 ms (76%) in everything
else**: Myelin kernels around `GridSample` + slice + resize at 2–3.5 ms each, i.e. the full-
resolution warp → downsample → flow-upsample chain every block repeats. The convs are uneven: the
deep 64–128-channel ones reach 110–250 TFLOPS, but the small-channel ones at full or half
resolution are memory-bound and slow (v4.26's two 16→16 convs at 1920x1088: 2.08 ms at ~9 TFLOPS,
where reading and writing their data takes ~0.1 ms).

Why native should win big here: a block only needs its warped, downsampled input at its own
resolution, and a bilinear downsample of a warp is just the warp evaluated at the downsample's
sample points (identical maths, fewer pixels), with the flow itself bilinearly upsampled at those
same points. So the full-resolution passes collapse to the final warp + blend (~0.1–0.2 ms at 4K),
the small-channel convs become memory-bound fused kernels, and the deep convs run on the existing
kernels. Rough estimate for v4.26 at 4K: **~3–5 ms against 19.3 (4–6x)**; at 1080p ~1–1.5 ms
against 4.06 (~3x, where launch overheads matter more). Estimate only, not prototyped.

In mpv that is the interpolated frame in every RIFE chain: Phase 3's Performance + RIFE 2x at 4K
spent ~16 of its ~18 ms per source frame in RIFE.

Cost: medium. Small graphs (~250 nodes) that are regular across versions; new pieces are a
stride-2 conv input mode, the transposed conv + depth-to-space tail, the per-channel β residual
epilogue, a fused warp/resample kernel and the final blend, plus wiring a native RIFE step next
to the TensorRT RIFE engine in the video path.

## TemporalFix (`temporalfix_s1_v1.1_op18_fp16`)

A 7-frame (21-channel) 1x model that runs before the upscale: a per-frame `encode`, an `aligner`
(5 `GridSample` warps, resizes, cumulative sums) and a `conf_head`, 44 convs of 1–32 channels.
TensorRT (level 5), per frame: **1280x736 4.76 ms, 1920x1088 11.7 ms**. At 1080p that is only
~0.033 TFLOP of convs, run at ~8 TFLOPS (16→16 at 960x544: 1.12 ms for two convs; 1→3 and 1→1
convs at full resolution at 0–2 TFLOPS), plus 7.5 ms of Myelin aligner/confidence/warp kernels:
by module, aligner 3.8 ms, conf_head 2.5 ms, encode 1.7 ms, unlabelled 3.7 ms.

So it is bound by memory traffic and kernel count, not maths. The 7 input frames are 88 MB at
1080p and the features 16–32 channels at half resolution (~33 MB each); a native version that
reads the frames once and fuses the thin convs with the elementwise work around them should be
in the ~1.5–2.5 ms range at 1080p (**~5–8x**) and ~1 ms at 720p (~4–5x). Estimate only. It is one
model rather than a family, so the lowering would be specific to it (or written by hand, as
NeoRTX does per model), and it needs the temporal (multi-frame) input path.

## DAT-2 and HAT-L (IllustrationJaNai V3 4x, bf16; `dathat/`)

Both ship in the `models-illustrationjanai-bf16` pack as bf16 ONNX (fp32 input, bf16 weights,
casts in the graph): these archs are in traiNNer's `ARCHS_WITHOUT_FP16`. That matters for native
kernels: bf16 tensor-core maths only exists with fp32 accumulation, which GeForce runs at half the
fp16-accumulate rate, and it would be a second kernel family (bf16 fragments, fp32 accumulators).

- DAT-2: 180 channels, 6 groups x 6 blocks alternating rectangle-window spatial attention (8x32 /
  32x8, dynamic position bias) and channel attention, each mixed with a depthwise-conv branch, and
  an SGFN feed-forward; ~24 MFLOP per token, ~6.2 TFLOP per 4x 512 tile.
- HAT-L: 180 channels, 12 groups x 6 Swin blocks (16x16 shifted windows) each with a channel-
  attention conv branch (3x3 180→60→180), plus an overlapping cross-attention block per group;
  ~95 MFLOP per token, ~25 TFLOP per tile.

TensorRT (level 3), 4x, 512x512 tile: **DAT-2 331 ms (~19 TFLOPS), HAT-L 572 ms (~44 TFLOPS)**,
against a bf16/fp32-accumulate peak of ~210 TFLOPS. Profiles (2,200–2,400 layers each):

| | DAT-2 | HAT-L |
|---|---|---|
| GEMMs (`sm80_xmma_gemm_bf16bf16_bf16f32_f32`) | 60.6 ms (18%) | 166 ms (30%, ~60 TFLOPS) |
| convs (`sm86_xmma_fprop_..._bf16`) | 32.9 ms (10%) | 70 ms (12.5%, ~105 TFLOPS) |
| reshape/transpose/slice/concat (window partition, shift as slice + concat) | ~90 ms (27%) | ~175 ms (31%) |
| LayerNorm, GELU, softmax and gate fusions | the rest | the rest |

Over a quarter of each is pure data layout (window partition and reverse, HAT's shifted windows
exported as slices and concats); a native kernel addresses windows directly, so that disappears.
Rough native estimate with bf16 GEMMs at ~150 TFLOPS plus the memory passes: DAT-2 ~70–90 ms
(**~4x**), HAT-L ~170–200 ms (**~3x**). Not prototyped. Cost: the highest here; two more transformer
variants (rectangle windows, shifted windows, overlapping cross-attention, dynamic position bias)
on top of FDAT's pieces, a bf16 kernel family, and the image path. They share FDAT's building
blocks (LayerNorm-fused GEMM, window attention, channel attention, depthwise convs), so they are
cheapest as a follow-on to FDAT.
