# Native kernels for more networks (spikes)

Which networks beyond SPAN/SPANF3/Compact (`src/native`) are worth native kernels: ESRGAN
(MangaJaNai 2x/4x), TFDAT v2 (bndl animefilm), FDAT_M (IllustrationJaNai), RIFE, TemporalFix.
Each spike measures TensorRT on the shipped model, works out where its time goes, and times the
existing native kernel template on the network's shapes. Nothing here ships. RTX 5090, 402 W
limit, TensorRT 11.4.0.106 with libaji's image-path build settings (`--builderOptimizationLevel=3`;
level 5 measured the same), cached engines, a 512x512 input tile (the image path's default tile).

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
