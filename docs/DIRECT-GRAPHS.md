# Direct-buffer CUDA graph replay

Set `AJI_DIRECT_GRAPH=1` before creating a TensorRT backend to capture spatial
preprocessing, inference and postprocessing against the caller's reusable frame
buffers. This avoids the input/output staging copies used by the default graph
path. Unset the variable to retain the default path; `AJI_NO_GRAPH` still disables
both graph modes. No model, builder setting, engine-cache key or ABI changes.

The cache holds at most 128 input/output buffer pairs, keyed by pointers, pitches,
dimensions, formats and color metadata. These are executable graphs, not cached
frame contents. Reusing an address with new pixels still runs inference. Callers
must retain their frame allocations through completion, as required by the ABI.

A new pair executes normally to initialize plans/TensorRT, then captures later
replay. Conversion-plan changes drain the stream and clear graphs before the
old device tables can be freed. Reconfiguration/destruction also releases the
graphs. Temporal and RIFE entry points are unchanged. Capture/launch failures
on this opt-in path are reported to the caller; the ordinary staged mode remains
available by leaving the variable unset.

The local research implementation matched three RGB48 movie frames and eight
format-transition outputs. Its native 4K result was about +0.75% over the same
optimized engine with staging, on one RTX 5090 setup. This is a small historical
measurement, not a guaranteed gain for arbitrary workloads, unique frame buffers,
other GPUs, or TensorRT-RTX. Frequent new buffer pairs pay capture/setup costs.

The focused regression compares staged and direct capture/replay while changing
NV12/P010 and chroma siting on reused buffer addresses. It requires NumPy, a built
backend, matching CUDA/TensorRT libraries and a regular static 1080p-to-4K engine:

```text
python sanity/direct_graph.py --backend BUILD/aji_trt.dll --engine MODEL.engine --cuda-runtime RUNTIME/cudart64_13.dll --output NEW_RESULTS_DIR
```

Use the corresponding library filenames/search path on Linux. GPU execution of
this PR was checked on Windows/SM120; SM89 code support alone is not a 4090 test.
Run without concurrent playback or benchmarks. The output directory must be new.
