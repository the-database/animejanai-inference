/*
 * core/trt_engine.h — TensorRT engine plumbing shared by the video
 * (aji.h) and image (aji_image.h) front-ends of aji_trt.
 *
 * Engine cache naming (Python-pipeline compatible), trtexec build
 * subprocesses (sync or on a background thread), and engine loading.
 * Nothing here knows about conf slots, chains or frame formats: callers
 * pass explicit paths/settings and get errors back as strings, which
 * they report through their own context.
 */

#ifndef AJI_CORE_TRT_ENGINE_H
#define AJI_CORE_TRT_ENGINE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace ajicore {

typedef void (*log_fn)(void *opaque, int level, const char *msg);

class TrtLogger : public nvinfer1::ILogger {
public:
    log_fn fn = nullptr;
    void *opaque = nullptr;
    void log(Severity sev, const char *msg) noexcept override;
};

// Pushes a CUcontext current for the guard's lifetime.
struct CtxGuard {
    CUcontext ctx;
    bool ok;
    explicit CtxGuard(CUcontext c) : ctx(c), ok(cuCtxPushCurrent(c) == CUDA_SUCCESS) {}
    ~CtxGuard() {
        if (ok) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
        }
    }
};

struct CudaFree {
    void operator()(void *p) const { cudaFree(p); }
};
using CudaPtr = std::unique_ptr<void, CudaFree>;

struct ModelEngine {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> exec;
    const char *in_name = nullptr;
    const char *out_name = nullptr;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    // The video chain's tensors are fp16. A model exported with fp32 I/O
    // (strongly typed, so the engine keeps it) runs through these fp32
    // staging buffers, cast on the way in and out (enqueue_model).
    bool in_fp32 = false, out_fp32 = false;
    CudaPtr in32, out32;
    size_t in32_elems = 0, out32_elems = 0;
};

// zlib-compatible CRC-32 (matches Python zlib.crc32 used for engine names).
uint32_t crc32_z(const std::string &data);
std::string sanitize_token(std::string s);

// Engine cache key: {onnx}.{crc32(settings)}.trt-{ver}.gpu-{name-smN}.engine.
// `dev` is the CUDA runtime ordinal of the GPU that runs the engine.
std::string trt_version_token();
std::string gpu_token(int dev);
std::string engine_suffix(int dev);
std::string engine_path_for(const std::string &model_dir,
                            const std::string &onnx_name,
                            const std::string &settings, int dev);
// Short form for new builds: aji-{crc32(onnx)}.{crc32(settings)}{suffix},
// avoiding MAX_PATH failures in deep portable installs.
std::string short_engine_path_for(const std::string &model_dir,
                                  const std::string &onnx_name,
                                  const std::string &settings, int dev);

// Deletes *.engine files in `dir` built for a TensorRT version or GPU not
// in this system (engines of every present GPU are kept, so moving work
// between GPUs never rebuilds; `fallback_dev` names the one to keep when
// the devices can't be enumerated). Appends removed names to *removed.
void clean_stale_engines(const std::string &dir, int fallback_dev,
                         std::vector<std::string> *removed);

// Human-readable build resolution from the trtexec shape flags.
std::string shapes_label(const std::string &settings);
// W,H of a trtexec shape flag (input:NxCxHxW) in `settings`.
bool parse_shape_wh(const std::string &settings, const char *key, int *w, int *h);

// Everything a build needs, with no context references, so it can run on
// a background thread.
struct BuildSpec {
    std::string cmdline;      // "trtexec" --onnx=... --saveEngine=...
    std::string env_prefix;   // POSIX only, e.g. "LD_LIBRARY_PATH=..."
    std::string engine_path;
    std::string tmp_path;     // what trtexec writes; renamed to engine_path
                              // only once the build has succeeded
    std::string log_path;     // trtexec output, kept for diagnostics
    std::string name;
    std::string res;          // shapes_label() of the build settings
};

// Build spec for {dir}/{onnx_name}.onnx with its timing cache beside it,
// building on CUDA device `dev` (the one that will run the engine).
bool make_build_spec(const std::string &trtexec, const std::string &trtexec_env,
                     const std::string &dir, const std::string &onnx_name,
                     const std::string &settings, const std::string &engine_path,
                     int dev, BuildSpec *spec, std::string *err);
// Same with explicit model and timing-cache paths (`name` labels logs).
bool make_build_spec_paths(const std::string &trtexec,
                           const std::string &trtexec_env,
                           const std::string &onnx_path,
                           const std::string &timing_cache,
                           const std::string &name, const std::string &settings,
                           const std::string &engine_path, int dev,
                           BuildSpec *spec, std::string *err);

// Run the build subprocess (no console window on win32) with output to
// spec.log_path; the child handle/pid is published to *child while it runs.
int run_build_process(const BuildSpec &spec, std::atomic<intptr_t> *child);
// Run a build to completion. trtexec writes spec.tmp_path, renamed to
// spec.engine_path only on success, so the cache only ever holds complete
// engines; on failure the leftovers are removed and the exit code appended
// to the build log. Thread-safe.
bool run_build_spec(const BuildSpec &spec, std::atomic<intptr_t> *child);

// One background trtexec at a time. The worker touches only these
// atomics; the owning thread reads `epath` and reaps via done_flag.
struct AsyncBuild {
    std::thread thread;
    std::atomic<bool> running{false};
    std::atomic<int> done_flag{0};
    std::atomic<bool> ok{false};
    std::atomic<intptr_t> child{0};
    std::string epath;

    void start(const BuildSpec &spec);
    // Kill an in-flight trtexec (the worker's failure path removes the
    // partial engine) and join the worker.
    void stop();
};

// Deserialize `engine_path`, bind its single input/output and set the
// input shape: [1, in_ch, in_h, in_w], or [1, rank5_t, 3, in_h, in_w]
// when rank5_t > 0. Fills me->out_w/out_h from the resolved output and
// me->in_fp32/out_fp32 from the I/O tensor types.
bool load_engine(nvinfer1::IRuntime *runtime, const std::string &engine_path,
                 ModelEngine *me, int in_w, int in_h, int in_ch, int rank5_t,
                 std::string *err);

} // namespace ajicore

#endif
