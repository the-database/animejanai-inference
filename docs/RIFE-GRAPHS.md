# Experimental RIFE CUDA graphs

The existing RIFE path submits preprocessing, TensorRT inference and
postprocessing on each call. Two opt-in modes preserve the existing operations
while reusing their CUDA launch graph:

| `AJI_RIFE_GRAPH` | Captured work |
| --- | --- |
| Unset | No RIFE graph; existing submission path |
| `engine` | TensorRT `enqueueV3` only |
| `chain` | Both input conversions, TensorRT, output conversion |

Set the variable before creating the backend context; it is read once and
retained across reconfiguration. Unset it to choose the ordinary path. An
invalid value logs a warning and chooses off. A default (null) CUDA stream
disables graphs for that context and uses the ordinary path.
This is separate from spatial-model graphs and the RIFE warp plugin. It does
not change the public ABI, models, weights, engine cache keys, interpolation
factor/order, scene detector or Vulkan queue settings.

## Lifetime and correctness

The captured tensors belong to the RIFE context. Copies of current caller
frames into padded storage, timestep updates, scene decisions and crop/output
copies remain outside capture. Input data and timestep therefore change on
every replay; this is not a frame-content cache.

The first invocation executes normally to warm TensorRT, synchronizes, and
captures a second submission without executing it. Its already computed output
is returned. Later calls launch the graph. If capture fails, its valid warmup
output is retained, including the remaining post-processing in engine mode.
If launch fails, the current frame runs through ordinary submission instead.
Either failure logs a diagnostic, clears the CUDA last error and disables RIFE
graphs for the lifetime of the context, including across reconfiguration.
Failed captures are always ended before the stream is reused. Actual normal
inference or warmup synchronization errors still propagate.

The chain graph bakes format/range arguments and references preprocessing
tables. A format/range transition synchronizes the caller's stream and destroys
that graph before replacing tables or buffers. Reconfigure/destruction destroys
graphs before engine tensors. The existing serialized-context and frame-lifetime
contract still applies; this does not make concurrent calls on one context safe.

## Evidence and limits

Earlier research on RTX 5090, TensorRT 11.3, RIFE 4.22 2x before optimized
Balanced, native 3840x2160 and Vulkan queues=3 found:

- Engine-only graphs matched all 73 generated direct-RIFE outputs checked.
- Chain capture matched one generated full-4K output exactly.
- Pilot rendered FPS was 115.035 baseline, 115.853 engine, 115.840 chain.
- Two later engine-mode samples averaged 114.845 versus 114.200 baseline,
  within the observed run variation. **No dependable native-FPS gain was
  established.** These are historical research results, not timings of this port.

Keep this opt-in while evaluating its value. Wider models, TensorRT versions,
GPUs, HDR/4:4:4 formats and real playback sequences need their own checks.
The warp experiment's provisional 2.73% gain does not belong to this change.

Fresh validation of this upstream port on Windows/RTX 5090 with TensorRT 11.3
and CUDA 13.3 passed 85 calls in each mode: 83 output hashes and two scene
decisions were identical. The original 75-call sequence also matched saved
stock-backend results independently. Capture counts were 0/2/7 for
off/engine/chain, including reset and format/range invalidation. The complete
TensorRT backend compiled.

Native mpv playback of this port also passed with the public API, RIFE 4.22 2x
before original Balanced 3.1, native 3840x2160 and Vulkan queues fixed at 3.
All three modes had zero rendered/decoder drops. Both graph modes' generated
4K captures at the same interpolated timestamp were byte-identical to graphs
off (zero changed pixels). Logs confirmed the requested graph mode was captured.
One smoke sample per mode measured 102.97 / 104.01 / 104.00 rendered FPS for
off / engine / chain. These single samples establish working playback, not a
dependable speedup. The earlier historical timings used optimized Balanced
and must not be compared as a before/after result for this port.

Linux and TensorRT-RTX builds/execution and RTX 4090 execution were not tested
here. No Vulkan queue change or image-quality tolerance was introduced.

The failure-handling revision passed the 85-call sequence in all four modes
with normal streams, the default stream, and forced failures at capture begin,
inside captured work, capture end, instantiation and replay. Outputs and scene
decisions matched the saved baseline; failure cases left no context/CUDA error
and attempted capture only once, including across reset.

One initial chain/reset comparison with a reordered clip fixture failed and
could not be reproduced. A follow-up with 20 resets each on the pre-fix and
fixed backends passed and matched across all 123 calls per backend. The cause
of that initial observation is unresolved; no comparison tolerance was added.

## Focused regression

Build the normal TensorRT backend. Supply a compatible, already built static
1080p RIFE 4.22 engine/model directory and three 24-frame decoded NV12 clips.
The padded model input is `1x11x1088x1920`. Do not run alongside another GPU test.

```text
python sanity/rife_graph.py --backend <aji_trt.dll-or-so> --cuda-runtime <cudart-library> --rife-dir <model-and-engine-directory> --frames <clip1.nv12> <clip2.nv12> <clip3.nv12> --output <new-directory>
```

The test uses API version 9 and requires numpy. It compares hashes and scene
decisions for four independent processes (off/engine/chain/invalid), changing inputs
and timesteps, a black/white cut, reconfiguration, and repeated NV12/P010 plus
limited/full-range transitions. It verifies actual capture counts, rather than
passing if graphs were silently bypassed. Source clips and engines are not
committed. This is a correctness check, not a throughput benchmark.
Use a continuous final clip around frames 8-10 so its timestep/replay cases
actually exercise interpolation instead of returning a scene-cut decision.

The regression also changes `AJI_RIFE_GRAPH` after context creation to verify
that configuration retains the initial choice. Add `--default-stream` to test
ordinary submission on the null stream. For forced failures on Windows, make
a separate test-only backend build with MSVC force-including
`sanity/rife_graph_faults.h` (`/FI<absolute-header-path>` in its C++ flags), then
run the same command with each of `--fault begin`, `--fault body`, `--fault end`,
`--fault instantiate` and `--fault launch`. Do not ship the instrumented DLL.
The header injects real CUDA error states into the dedicated test build;
production builds have no fault hooks. Each case checks output parity, cleared
last errors and a single capture attempt even after reconfiguration.
