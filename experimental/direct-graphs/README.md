# Historical direct-buffer graph experiment

The original opt-in path cached an executable per caller buffer pair, up to 128
pairs. The measurements below cover that implementation on a static engine.
The current backend updates one executable when caller surfaces rotate.
For current behavior and validation commands, see [CUDA graphs for regular frame inference](../../docs/DIRECT-GRAPHS.md).

Three additional paired 400-frame CUDA-event measurements used the normal DLL,
the same saved static stock engine per model, a real 1080p NV12 frame, 60 warmup
calls and 4K YUV444P16 output. Order was staged/direct, direct/staged, staged/direct.
No concurrent mpv session was present. This is device-chain timing including
pre/postprocessing, not decode or native-display playback FPS.

| Model | Staged ms/frame (rounds 1/2/3) | Direct ms/frame (rounds 1/2/3) |
| --- | --- | --- |
| Balanced | 6.3711 / 6.4206 / 6.2807 | 6.3464 / 6.3422 / 6.1709 |
| Performance | 3.2446 / 3.2030 / 3.1605 | 3.1971 / 3.0850 / 3.0902 |

Paired throughput changes were +0.39/+1.24/+1.78% and +1.49/+3.82/+2.28%,
respectively. Across-round drift and the reviewer's TensorRT 11.4 results (no
resolved gain) limit the claim: this demonstrates a small local benefit, not a
universal speedup or a reason to change the default. Wider decoder/output pools
may fall back entirely. `validation.json` retains the samples and scope.
