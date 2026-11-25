# CUDA graphs for regular frame inference

On a non-default CUDA stream, the TensorRT backend can capture preprocessing,
the spatial model chain and postprocessing in one executable. This works with
native models, TensorRT engines and mixed chains. The first call executes and
synchronizes an eager warmup. Later calls update the captured conversion nodes
with the current device addresses and byte pitches for every input and output
plane. The caller must keep those allocations valid until the stream completes.

The graph belongs to the current chain configuration. Its key includes input
and output dimensions and formats, plus input matrix, range and chroma siting.
Postprocessing uses the input matrix and range and places subsampled output
chroma on the left. Callers can vary the output color tags, surfaces and pitches
while reusing the executable. Selecting a slot takes effect at the next
`aji_configure`, which invalidates the graph before replacing its dependencies.

If capture, conversion-node identification or instantiation fails, the warmup
has already produced the requested frame. The next call uses staging. If an
update fails during replay, the backend discards the executable and handles
that same frame through the fallback. Staging buffers are allocated on first
use of the fallback. The backend can run eagerly if staged capture is unavailable.
Execution errors during direct warmup or launch reach the caller.

Set `AJI_NO_GRAPH=1` before creating the backend to run eagerly. Default-stream
calls run eagerly. The former `AJI_DIRECT_GRAPH` opt-in is no longer needed.
The updatable graph was validated for regular spatial inference. Validation
with interleaved RIFE interpolation is still pending. RIFE-enabled chains
therefore keep staging for graphed spatial inference and receive no staging
VRAM savings. Temporal sequence entry points use their existing paths.

## Checking replay

Build `aji_trt`. The regression uses Python's standard library, a compatible CUDA
runtime and either a chain configuration with its model directory or an engine
built for the selected input dimensions. Supply a new output directory.

```text
python sanity/direct_graph.py --backend BUILD/aji_trt.dll --cuda-runtime RUNTIME/cudart64_13.dll --engine MODEL.engine --width 1920 --height 1080 --output NEW_RESULTS
python sanity/direct_graph.py --backend BUILD/aji_trt.dll --cuda-runtime RUNTIME/cudart64_13.dll --conf CONFIG --models MODEL_DIR --slot 1 --width 320 --height 180 --output OTHER_RESULTS
```

For a mixed configuration that needs to build a TensorRT engine, pass
`--trtexec PATH_TO_TRTEXEC`. Use the appropriate shared-library names on Linux.
The test compares each output plane's active bytes with eager execution in a
separate process. It rotates different pitches and addresses, queues frames
before synchronization, changes color metadata and reconfigures at the same shape.

To check failure handling, build a second test backend with
`sanity/direct_graph_faults.h` forced into the `aji_trt.cpp` translation unit
(`/FI` on MSVC, `-include` on GCC or Clang) and pass it as `--fault-backend`.
The `aji_frame_graph_test` CTest also checks metadata keys and plane binding on a CPU.
