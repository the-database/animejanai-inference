# CUDA graphs for regular frame inference

On a non-default CUDA stream, the TensorRT backend can capture preprocessing,
the spatial model chain and postprocessing in one executable. This works with
native models, TensorRT engines and mixed chains. The first call executes and
synchronizes an eager warmup. Later calls update the captured conversion nodes
with the current device addresses and byte pitches for every input and output
plane. The caller must keep those allocations valid until the stream completes.

The graph belongs to the current chain configuration. Input and output shape,
format, matrix, range and chroma siting determine whether the conversion plans
can be reused. A change to that information, the selected slot or the configured
chain invalidates the graph before its dependencies can be replaced. Rotating
surfaces or changing pitches reuses the executable.

If capture, conversion-node identification or instantiation fails, the warmup
has already produced the requested frame. The next call uses staging. If an
update fails during replay, the backend discards the executable and handles
that same frame through the fallback. Staging buffers are allocated on first
use of the fallback. The backend can run eagerly if staged capture is unavailable.
Execution errors during direct warmup or launch reach the caller.

Set `AJI_NO_GRAPH=1` before creating the backend to run eagerly. Default-stream
calls run eagerly. The former `AJI_DIRECT_GRAPH` opt-in is no longer needed.
This change applies to regular spatial inference. RIFE and temporal sequence
entry points use their existing paths.

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
