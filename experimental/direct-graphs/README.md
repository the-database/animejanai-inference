# Opt-in direct-buffer CUDA graphs

`AJI_DIRECT_GRAPH=1` captures the preprocessing/model/postprocessing chain against
the caller's actual input/output buffers, avoiding the staging plane copies of
the ordinary graph path. The default remains the staged path. Layout, stride,
color tags and pointer identity are part of the cache key; reconfiguration and
format changes invalidate graphs that refer to the former conversion plans.

Capture is optional. After a successful, synchronized warmup, failed begin/body/
end/instantiate operations preserve that frame, terminate any started capture,
clear capture errors and disable direct capture for the context. Later calls use
the ordinary path. Warmup, synchronization and graph-launch errors still propagate.
At 128 buffer pairs, a new pair disables direct capture and frees that cache.
This bounds recapture cost for rotating pools larger than the cache; switching
FIFO to LRU would not solve a cyclic working set of 129 or more pairs.

## Review validation

Build the normal `aji_trt` target. Run `sanity/direct_graph.py` with `--backend`,
`--engine` (a static 1080p 2x engine), `--cuda-runtime` and a new `--output` directory.
It checks NV12/P010 capture/replay and chroma-siting transitions, then cycles 130
output-buffer addresses twice and checks the cache disables once and output stays
identical. Input and output allocations remain valid until execution completes.

For failure checks, compile a separate test backend with
`sanity/direct_graph_faults.h` forced-included into the C++ compilation (`/FI` on
MSVC or `-include`), and pass it as `--fault-backend`. Never ship this DLL. The
fixture forces real CUDA errors at begin, captured body, end and instantiate;
all preserve the stock hashes, clear the runtime error and latch the fallback.
Separate warmup-sync and graph-launch faults must return errors. The normal DLL
has neither the fault wrappers nor their exports. All these checks passed on an
RTX 5090 with TensorRT 11.3 / CUDA 13.3 after merging upstream `a120c32`.

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
