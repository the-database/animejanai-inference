/*
 * aji_trt — TensorRT backend for the aji C ABI (see include/aji.h).
 * Also built as aji_trtrtx (AJI_TRTRTX, TensorRT for RTX): the engine
 * plumbing differences live in core/trt_engine.
 *
 * Conf mode executes animejanai.conf chains: per-model Catmull-Rom resize
 * steps + 2x model steps, all on fp16 NCHW RGB ping-pong buffers, with
 * static per-resolution engines built on first use via trtexec (cache key
 * compatible with the Python pipeline: {onnx}.{crc32(settings)}.trt-{ver}
 * .gpu-{name-smN}.engine, plus stale-engine cleanup).
 *
 * Context model: all CUDA work happens with the caller's CUcontext pushed
 * current; the CUDA runtime binds to the current driver context.
 */

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "aji.h"
#include "aji_conf.h"
#include "kernels.h"
#include "onnx_shape.h"
#include "core/fileio.h"
#include "core/trt_engine.h"
#include "rife_warp_plugin.h"

namespace fs = std::filesystem;

namespace {

// TRT 11: strongly-typed is the default (--stronglyTyped is a no-op), and sanitize_settings_trt11
// strips --inputIOFormats/--outputIOFormats/--tacticSources anyway (types come from the network;
// the cuDNN/cuBLAS tactic sources are gone). So the only flags worth carrying are the builder
// optimization level, the build shape profile, and skip-inference (we do the inference ourselves).
const char *DEFAULT_TRT_ENGINE_SETTINGS =
    "--builderOptimizationLevel=5 --optShapes=input:%video_resolution% "
    "--skipInference";

using ajicore::BuildSpec;
using ajicore::CtxGuard;
using ajicore::CudaPtr;
using ajicore::ModelEngine;
using ajicore::engine_path_for;
using ajicore::file_exists;
using ajicore::file_size;
using ajicore::parse_shape_wh;
using ajicore::read_file_bytes;
using ajicore::remove_file;
using ajicore::run_build_spec;
using ajicore::shapes_label;
using ajicore::short_engine_path_for;
using Logger = ajicore::TrtLogger;


struct Step {
    enum Kind { RESIZE, MODEL, TEMPORAL } kind;
    int out_w, out_h;
    int engine_idx;             // MODEL, TEMPORAL
    aji_plan *plan = nullptr;   // RESIZE
};

} // namespace

struct aji_ctx {
    Logger logger;
    CUcontext cu_ctx = nullptr;
    bool owns_primary = false;
    // The device cu_ctx lives on, which need not be device 0 (mpv adopts
    // the hwdec device, i.e. the Vulkan VO's GPU). cuda_dev is the runtime
    // API ordinal: it names engines and is trtexec's --device.
    CUdevice cu_dev = 0;
    int cuda_dev = 0;

    std::unique_ptr<nvinfer1::IRuntime> runtime;

    // conf mode
    bool conf_mode = false;
    std::string conf_path, model_dir, trtexec, trtexec_env, rife_model_dir;
    AjiConf conf;
    int slot = 1;
    bool engines_cleaned = false;
    std::vector<ModelEngine> engines;
    std::vector<Step> steps;
    bool active = false;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    std::string current_log;
    std::vector<std::string> log_info, log_steps;

    // direct mode
    ModelEngine direct;
    int scale = 0;
    int max_w = 0, max_h = 0;

    void *buf[2] = {nullptr, nullptr};
    size_t buf_bytes = 0;

    // Background engine builds (async_build): one trtexec at a time on a
    // worker thread that touches no ctx state beyond build's atomics. The
    // caller thread owns build.epath/failed_builds (written only while no
    // worker is running / in aji_poll).
    bool async_build = false;
    ajicore::AsyncBuild build;
    std::set<std::string> failed_builds;
#ifdef AJI_TRTRTX
    // engines built by this context: one that then fails to load is an
    // error, not a stale cache entry to rebuild (which would loop)
    std::set<std::string> built_engines;
    // async: engines warmed up on the worker this session (built or
    // warm-only jobs), so their caller-thread load is cheap
    std::set<std::string> warmed;
    bool build_warm_only = false;   // the running AsyncBuild job's kind
    // failures already reported as an error (for the dispatcher's
    // TensorRT fallback); if it comes back anyway (no fallback possible),
    // the next configure degrades to passthrough as aji_trt does
    std::set<std::string> reported_failures;
#endif

    // resampling plans: per-step resize plans live in steps[]; pre/post are
    // (re)built lazily per frame key since siting arrives with frames
    aji_plan *pre_plan = nullptr, *post_plan = nullptr;
    int pre_key[4] = {0}, post_key[4] = {0};

    // Pre-RIFE downscale (rife-first only): the first model's "resize before
    // upscale" is hoisted ahead of RIFE so interpolation runs on the smaller
    // frame. work_w/h is the post-resize ("work") resolution at which RIFE and
    // the upscale chain run; the filter applies the downscale to source frames
    // via aji_resize. aji_resize keeps its OWN pre/post plans so it doesn't
    // thrash run_chain's per-frame plan keys.
    bool has_pre_resize = false;
    int pre_src_w = 0, pre_src_h = 0;               // aji_resize input dims
    int work_w = 0, work_h = 0;                      // aji_resize output dims
    aji_plan *pre_resize_plan = nullptr;            // source -> work, Spline36
    aji_plan *pr_pre_plan = nullptr, *pr_post_plan = nullptr;
    int pr_pre_key[4] = {0}, pr_post_key[4] = {0};

    // RIFE (phase 1.5): interpolation between already-upscaled frames,
    // replicating animejanai_core's pad/crop + rife_cuda.py's bilinear 709
    // conversions + the vsmlrt v1 11-channel model input. Geometry state
    // is set at configure; format-dependent staging/plans build lazily on
    // the first aji_infer_rife (frames carry the format).
    struct {
        bool enabled = false;
        bool before_upscale = false; // interpolate at source res, before the
                                     // upscale models (else: after, the default)
        int num = 1, den = 1;
        double scd_threshold = 0.150;
        ModelEngine engine;
        int w = 0, h = 0;            // unpadded; chain output dims, or the
                                     // source dims when before_upscale
        int pw = 0, ph = 0;          // padded to mod-64
        int pad_l = 0, pad_t = 0;    // centered, rounded down to even
        int format = 0;              // staging/plans built for this format
        aji_plan *pre = nullptr;     // padded YUV -> RGB, bilinear, 709
        aji_plan *post = nullptr;    // RGB -> padded YUV, bilinear, 709
        void *pad_a = nullptr, *pad_b = nullptr, *pad_o = nullptr;
        void *in_tensor = nullptr;   // 11 * pw*ph fp16
        void *out_tensor = nullptr;  // 3 * pw*ph fp16
        float *scd = nullptr;        // device diff accumulator
    } rife;

    // CUDA graph replay of the whole chain (pre -> steps -> post). The
    // captured graph bakes device pointers, so frames are staged through
    // stable buffers and the per-frame plane copies stay outside the graph.
    // graph_ok latches false on the first capture/launch failure.
    bool graph_ok = true;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_key[9] = {0};
    void *stage_in = nullptr, *stage_out = nullptr;
    size_t stage_in_bytes = 0, stage_out_bytes = 0;

    // Completion tickets (aji_flush/done/wait): an event ring on the
    // caller's stream. Tickets complete in submission order, so any
    // fired event also retires every earlier ticket (tick_completed
    // watermark). The ring bounds in-flight tickets; reusing a slot
    // synchronizes its previous recording first, which never blocks in
    // practice (callers queue 2-4 frames, the ring holds 8).
    static constexpr int TICK_RING = 8;

    // Temporal (multi-frame) model steps, one stage per temporal model. Each
    // stage keeps a history ring of frames at its step's input resolution:
    // stage 0's ring is filled by aji_ingest (which runs the steps before it
    // once per source frame), stage i+1's by running stage i (and the steps
    // between the two) once per frame, on demand. aji_infer_seq gathers the
    // last stage's 2r+1 window into the padded model input, runs the model,
    // crops, and runs the steps after it.
    struct TemporalStage {
        int t = 1, r = 0;
        size_t step_idx = 0;         // index of the TEMPORAL step in steps[]
        int w = 0, h = 0;            // ring frame dims (step input)
        int pw = 0, ph = 0;          // model input dims, padded
        int ow = 0, oh = 0;          // step output dims, cropped
        int pow = 0, poh = 0;        // model output dims
        float threshold = 0.0f;
        int nslots = 0;
        void *ring = nullptr;        // nslots * 3*w*h fp16
        float *diff = nullptr;       // per slot: rgb diff vs seq-1
        std::vector<int64_t> seq;    // per slot, -1 = empty
        std::vector<int> matrix, range;
        void *in_tensor = nullptr;   // 3t * pw*ph fp16
        void *out_tensor = nullptr;  // 3 * pow*poh fp16
        uint64_t runs = 0;           // model runs (AJI_TEMPORAL_STATS)
    };
    struct {
        bool enabled = false;
        int r = 0;                   // total radius: the sum over stages
        uint64_t ingests = 0;        // AJI_TEMPORAL_STATS
        std::vector<TemporalStage> st;
    } temporal;

    // ONNX input frame counts, keyed by path, valid while size/mtime match
    struct OnnxFrames {
        uintmax_t size = 0;
        int64_t mtime = 0;
        int frames = 1;
        bool rank5 = false;
        int elem_type = 0;
        bool dynamic_input = false;
    };
    std::map<std::string, OnnxFrames> onnx_frames;
    cudaEvent_t tick_ev[TICK_RING] = {};
    uint64_t tick_next = 1;
    uint64_t tick_completed = 0;

    char errbuf[512] = {0};

    void set_error(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(errbuf, sizeof(errbuf), fmt, ap);
        va_end(ap);
        if (logger.fn)
            logger.fn(logger.opaque, /*kERROR*/ 1, errbuf);
    }
    void verbose(const char *fmt, ...) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (logger.fn)
            logger.fn(logger.opaque, /*kINFO*/ 3, buf);
    }
};

namespace {

aji_csp make_csp(const aji_frame *f)
{
    return aji_make_csp(f->format, f->matrix, f->range);
}

void clean_stale_engines(aji_ctx *c)
{
    std::vector<std::string> removed;
    ajicore::clean_stale_engines(c->model_dir, c->cuda_dev, &removed);
    for (const std::string &name : removed) {
        c->log_steps.push_back(
            "Removed stale " AJI_TRT_PRODUCT " engine (different GPU or "
            AJI_TRT_PRODUCT " version): " + name);
        c->verbose("removed stale engine %s", name.c_str());
    }
}

bool make_build_spec(aji_ctx *c, const std::string &onnx_name,
                     const std::string &settings,
                     const std::string &engine_path, const std::string *dir,
                     BuildSpec *spec)
{
    std::string err;
    if (!ajicore::make_build_spec(c->trtexec, c->trtexec_env,
                                  dir ? *dir : c->model_dir, onnx_name,
                                  settings, engine_path, c->cuda_dev, spec,
                                  &err)) {
        c->set_error("%s", err.c_str());
        return false;
    }
#if defined(_WIN32) && !defined(AJI_TRTRTX) && NV_TENSORRT_MAJOR == 11
    if (onnx_name == AJI_RIFE_WARP_MODEL)
        spec->cmdline += " --dynamicPlugins=\"" + aji_rife_warp_plugin_path() + "\"";
#endif
    return true;
}

bool build_engine(aji_ctx *c, const std::string &onnx_name,
                  const std::string &settings, const std::string &engine_path,
                  const std::string *dir = nullptr)
{
    BuildSpec spec;
    if (!make_build_spec(c, onnx_name, settings, engine_path, dir, &spec))
        return false;
    c->verbose("building engine: %s", spec.cmdline.c_str());
    c->log_steps.push_back("Building " AJI_TRT_PRODUCT " engine for " + onnx_name +
                           " for " + spec.res +
                           " (first play at this resolution)");
    if (!run_build_spec(spec, &c->build.child)) {
#ifdef AJI_TRTRTX
        c->set_error("TensorRT-RTX engine build failed for %s (see %s)",
#else
        c->set_error("trtexec failed building engine for %s (see %s)",
#endif
                     onnx_name.c_str(), spec.log_path.c_str());
        return false;
    }
#ifdef AJI_TRTRTX
    c->built_engines.insert(engine_path);
#endif
    return true;
}

// rank5_t > 0: the input is [1, rank5_t, 3, H, W] (in_ch is then ignored).
// *keep: see ajicore::load_engine.
bool load_engine(aji_ctx *c, const std::string &engine_path, ModelEngine *me,
                 int in_w, int in_h, int in_ch = 3, int rank5_t = 0,
                 bool *keep = nullptr)
{
    std::string err;
    if (!ajicore::load_engine(c->runtime.get(), engine_path, me, in_w, in_h,
                              in_ch, rank5_t, &err, keep)) {
        c->set_error("%s", err.c_str());
        return false;
    }
    if (me->in_fp32 || me->out_fp32)
        c->verbose("%s: fp32 model I/O, cast to/from the fp16 chain",
                   engine_path.c_str());
    return true;
}


int round_even(double v)
{
    int i = (int)(v + 0.5);
    return i & ~1;
}

// scale_to_1080: fit into (box_w, box_h) preserving aspect.
void fit_box(int w, int h, double box_w, double box_h, int *ow, int *oh)
{
    if ((double)w / h > 16.0 / 9.0) {
        *ow = round_even(box_w);
        *oh = round_even(box_w * h / w);
    } else {
        *ow = round_even(box_h * w / h);
        *oh = round_even(box_h);
    }
}

// Fit (w,h) within an arbitrary (box_w, box_h) box preserving aspect, flooring
// to even so the result never EXCEEDS the box. Unlike fit_box this does not
// assume a 16:9 box — used to clamp to a dynamic engine's max build shape,
// where any overshoot would fail enqueue.
void fit_within(int w, int h, int box_w, int box_h, int *ow, int *oh)
{
    double s = std::min((double)box_w / w, (double)box_h / h);
    *ow = (int)(w * s) & ~1;
    *oh = (int)(h * s) & ~1;
}

// Compose the FIRST model's "resize before upscale" target from (w,h): factor
// then explicit height, exactly as the configure model loop applies them.
// Returns true + fills (ow,oh) when an explicit resize is set and changes the
// size (the maxShapes/auto cap is deliberately NOT part of this — only an
// explicit resize is hoisted ahead of RIFE).
bool first_resize_target(const AjiModelConf &m, int w, int h, int *ow, int *oh)
{
    int cw = w, ch = h;
    double factor = m.resize_factor_before_upscale;
    if (m.resize_height_before_upscale != 0)
        factor = 100;
    if (factor != 100) {
        int nw = round_even(cw * factor / 100.0);
        int nh = round_even(ch * factor / 100.0);
        if (nw >= 2 && nh >= 2) {
            cw = nw;
            ch = nh;
        }
    }
    if (m.resize_height_before_upscale != 0 &&
        (int)m.resize_height_before_upscale != ch) {
        int nw, nh;
        fit_box(cw, ch, m.resize_height_before_upscale * 16.0 / 9.0,
                m.resize_height_before_upscale, &nw, &nh);
        cw = nw;
        ch = nh;
    }
    *ow = cw;
    *oh = ch;
    return cw != w || ch != h;
}

bool ensure_buffers(aji_ctx *c, size_t bytes)
{
    if (bytes <= c->buf_bytes && c->buf[0] && c->buf[1])
        return true;
    for (auto &b : c->buf) {
        cudaFree(b);
        b = nullptr;
    }
    if (cudaMalloc(&c->buf[0], bytes) != cudaSuccess ||
        cudaMalloc(&c->buf[1], bytes) != cudaSuccess) {
        c->set_error("cudaMalloc(%zu) failed for chain buffers", bytes);
        c->buf_bytes = 0;
        return false;
    }
    c->buf_bytes = bytes;
    return true;
}

// Kick a background build of one engine; the configure that called this
// returns passthrough, and aji_poll() tells the caller when to reconfigure.
void start_async_build(aji_ctx *c, const BuildSpec &spec)
{
    c->log_steps.push_back("Building " AJI_TRT_PRODUCT " engine for " + spec.name +
                           " for " + spec.res +
                           " (first play at this resolution)");
    c->verbose("background engine build: %s", spec.cmdline.c_str());
#ifdef AJI_TRTRTX
    c->build_warm_only = false;
#endif
    c->build.start(spec);
}

#ifdef AJI_TRTRTX
// TensorRT-RTX, async: a cached engine not yet warmed this session is
// loaded and warmed on the worker first (after a driver update the JIT can
// take seconds); the configure that called this plays passthrough until
// aji_poll() fires. With warm caches the job is a fraction of a second.
// Returns true if the caller must defer (0).
bool defer_to_warm_up(aji_ctx *c, const std::string &name,
                      const std::string &settings, const std::string &path)
{
    if (c->warmed.count(path) || c->built_engines.count(path) ||
        file_size(path) <= 0)
        return false;
    const std::string line = "Preparing " AJI_TRT_PRODUCT " engine for " +
                             name + " for " + shapes_label(settings) +
                             " (first use this session)";
    c->log_steps.push_back(line);
    if (c->build.running.load())
        return true;  // one job at a time; poll() -> reconfigure cascades
    BuildSpec spec;
    spec.warm_only = true;
    spec.engine_path = path;
    spec.log_path = path + ".warm.log";
    spec.name = name;
    spec.res = shapes_label(settings);
    spec.dev = c->cuda_dev;
    spec.cmdline = "warm-up of " + path;
    c->verbose("background engine warm-up: %s", path.c_str());
    c->build_warm_only = true;
    c->build.start(spec);
    return true;
}
#endif

// Build-if-missing + load, with a self-healing retry: a cached engine that
// fails to deserialize (e.g. built by a different trtexec version than the
// loaded TensorRT runtime) is deleted and rebuilt once instead of wedging
// playback until someone clears the cache by hand.
// Returns 1 loaded, 0 deferred to a background build (async_build; the
// chain stays inactive and the build is logged), -1 failure.
int ensure_engine(aji_ctx *c, const std::string &name,
                  const std::string &settings, const std::string &epath,
                  ModelEngine *me, int w, int h, int ch,
                  const std::string *dir = nullptr, int rank5_t = 0)
{
    const std::string model_dir = dir ? *dir : c->model_dir;
    const std::string short_epath =
        short_engine_path_for(model_dir, name, settings, c->cuda_dev);
#ifdef AJI_TRTRTX
    // a failure already reported (and not replaced by the TensorRT
    // fallback): passthrough, as aji_trt does for a failed build
    for (const std::string *p : {&epath, &short_epath}) {
        if (c->async_build && c->failed_builds.count(*p) &&
            c->reported_failures.count(*p)) {
            c->log_steps.push_back(
                "Engine FAILED for " + name + " for " + shapes_label(settings) +
                " (see " + *p + ".build.log); playing without this chain");
            return 0;
        }
    }
    if (c->async_build) {
        const std::string &cached_path = file_exists(epath) ? epath : short_epath;
        if (file_exists(cached_path) &&
            defer_to_warm_up(c, name, settings, cached_path))
            return 0;
    }
#endif

    // Backwards compatibility: 3.5.0 and earlier cached engines used the
    // full ONNX basename in the filename. Try that legacy path first so
    // upgraders do not rebuild engines that already exist.
    auto try_cached = [&](const std::string &path) -> int {
        if (!file_exists(path))
            return 0;
        if (file_size(path) <= 0) {
            // an interrupted build from before builds went through a temp
            // file: not an error, just rebuild (deserializing it would only
            // print a TensorRT API error first)
            c->verbose("removing incomplete cached engine: %s", path.c_str());
            remove_file(path);
            return -1;
        }
#ifdef AJI_TRTRTX
        bool keep = false;
        if (load_engine(c, path, me, w, h, ch, rank5_t, &keep))
            return 1;
        if (keep || c->built_engines.count(path)) {
            // the file is fine (context/warm-up failed) or was just built:
            // an error to report, not a stale entry to delete and rebuild
            if (c->async_build) {
                c->failed_builds.insert(path);
                c->reported_failures.insert(path);
            }
            return -2;  // errbuf has why
        }
#else
        if (load_engine(c, path, me, w, h, ch, rank5_t))
            return 1;
#endif
        c->verbose("cached engine unusable, rebuilding: %s", path.c_str());
        remove_file(path);
        return -1;
    };

    int cached = try_cached(epath);
    if (cached == 1)
        return 1;
    if (cached == -2)
        return -1;
    if (short_epath != epath) {
        cached = try_cached(short_epath);
        if (cached == 1)
            return 1;
        if (cached == -2)
            return -1;
    }

    // New builds use a short filename. This avoids MAX_PATH failures for
    // portable installs while keeping the model/settings/GPU/TRT cache key.
    const std::string &build_epath = short_epath;
    if (c->async_build) {
        if (c->failed_builds.count(build_epath)) {
#ifdef AJI_TRTRTX
            // an error once, so the dispatcher can fall back to TensorRT;
            // after that (no fallback possible) the passthrough below
            if (c->reported_failures.insert(build_epath).second) {
                c->set_error("TensorRT-RTX engine build failed for %s (see "
                             "%s.build.log)", name.c_str(), build_epath.c_str());
                return -1;
            }
#endif
            // permanent (until reconfigure changes the cache key):
            // play passthrough instead of failing the filter
            c->log_steps.push_back(
                "Engine build FAILED for " + name + " for " +
                shapes_label(settings) + " (see " + build_epath +
                ".build.log); playing without this chain");
            return 0;
        }
        if (c->build.running.load()) {
            // one build at a time; multi-engine chains cascade through
            // repeated poll() -> reconfigure cycles
            c->log_steps.push_back("Building " AJI_TRT_PRODUCT " engine for " +
                                   name + " for " +
                                   shapes_label(settings) +
                                   " (first play at this resolution)");
            return 0;
        }
        BuildSpec spec;
        if (!make_build_spec(c, name, settings, build_epath, dir, &spec))
            return -1;
        start_async_build(c, spec);
        return 0;
    }
    if (!build_engine(c, name, settings, build_epath, dir))
        return -1;
    return load_engine(c, build_epath, me, w, h, ch, rank5_t) ? 1 : -1;
}

// rife model code -> file basename: 414 -> rife_v4.14, 4141 -> _lite,
// ensemble appends _ensemble (rife_cuda.py's mapping).
std::string rife_model_name(int code, bool ensemble)
{
    std::string s = std::to_string(code);
    if (s.size() < 2)
        return "";
    std::string dec = s.size() == 2 ? s.substr(1, 1) : s.substr(1, 2);
    std::string name = "rife_v" + s.substr(0, 1) + "." + dec;
    if (s.size() == 4 && s.back() == '1')
        name += "_lite";
    if (ensemble)
        name += "_ensemble";
    return name;
}

void rife_teardown(aji_ctx *c)
{
    auto &R = c->rife;
    aji_plan_destroy(R.pre);
    aji_plan_destroy(R.post);
    cudaFree(R.pad_a);
    cudaFree(R.pad_b);
    cudaFree(R.pad_o);
    cudaFree(R.in_tensor);
    cudaFree(R.out_tensor);
    cudaFree(R.scd);
    R.engine.exec.reset();
    R.engine.engine.reset();
    R = {};
}

// Configure-time RIFE setup: geometry, engine (build on first play), the
// constant input planes. Format-dependent staging builds lazily in
// aji_infer_rife. Requires the CUDA context to be current.
bool setup_rife(aji_ctx *c, const AjiChainConf *chain, int w, int h,
                double fps, bool before_upscale)
{
    auto &R = c->rife;
    R.w = w;
    R.h = h;
    R.pw = (w + 63) / 64 * 64;
    R.ph = (h + 63) / 64 * 64;
    // centered like animejanai_core's AddBorders split, but rounded down
    // to even so 4:2:0 chroma stays aligned (odd borders are an error in
    // the reference pipeline)
    R.pad_l = ((R.pw - w) / 2) & ~1;
    R.pad_t = ((R.ph - h) / 2) & ~1;
    R.num = chain->rife_factor_num;
    R.den = chain->rife_factor_den;
    R.scd_threshold = chain->rife_scd_threshold;

    std::string model = rife_model_name(chain->rife_model, chain->rife_ensemble);
    const char *warp = std::getenv("AJI_RIFE_WARP");
    if (warp && std::strcmp(warp, "1") == 0) {
#if defined(_WIN32) && !defined(AJI_TRTRTX) && NV_TENSORRT_MAJOR == 11
        if (model != "rife_v4.22" || w != 1920 || h != 1080) {
            c->set_error("AJI_RIFE_WARP requires non-ensemble RIFE 4.22 at 1920x1080");
            return false;
        }
        if (!aji_register_rife_warp_plugin()) {
            c->set_error("Cannot register rife-warp-plugin.dll beside the backend");
            return false;
        }
        // Distinct ONNX and engine-cache identity: never load a stock engine
        // when the experiment is selected, or vice versa.
        model = AJI_RIFE_WARP_MODEL;
#else
        c->set_error("AJI_RIFE_WARP currently requires Windows and TensorRT 11");
        return false;
#endif
    }
    char dims[64];
    snprintf(dims, sizeof(dims), "1x11x%dx%d", R.ph, R.pw);
    // strongly-typed build from the fp16-typed rife onnx (the shipped
    // models keep the sampling-grid math fp32 in-graph); no cuDNN/cuBLAS
    const std::string settings = std::string(
        "--stronglyTyped --optShapes=input:") + dims +
        " --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"
        " --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference";
    const std::string epath =
        engine_path_for(c->rife_model_dir, model, settings, c->cuda_dev);
    int er = ensure_engine(c, model, settings, epath, &R.engine, R.pw, R.ph,
                           11, &c->rife_model_dir);
    if (er < 0)
        return false;
    if (er == 0)
        return true;  // building in the background; chain runs without
                      // rife until aji_poll() triggers a reconfigure

    const size_t plane = (size_t)R.pw * R.ph;
    if (cudaMalloc(&R.in_tensor, plane * 11 * 2) != cudaSuccess ||
        cudaMalloc(&R.out_tensor, plane * 3 * 2) != cudaSuccess ||
        cudaMalloc(&R.scd, sizeof(float)) != cudaSuccess) {
        c->set_error("rife buffer allocation failed");
        return false;
    }
    // constant channels 7..10 (mesh + multipliers); one-time fill, default
    // stream + device sync so non-blocking infer streams see it
    char *t = (char *)R.in_tensor;
    aji_rife_fill_consts(t + plane * 7 * 2, t + plane * 8 * 2,
                         t + plane * 9 * 2, t + plane * 10 * 2,
                         R.pw, R.ph, nullptr);
    cudaDeviceSynchronize();

    char buf[200];
    const double factor = (double)R.num / R.den;
    if (R.pw != w || R.ph != h) {
        snprintf(buf, sizeof(buf),
                 "Padded to %dx%d, applied RIFE v%d Interpolation %.3fx, "
                 "cropped back to %dx%d;    New Video FPS: %.3f",
                 R.pw, R.ph, chain->rife_model, factor, w, h, fps * factor);
    } else {
        snprintf(buf, sizeof(buf),
                 "Applied RIFE v%d Interpolation %.3fx;    "
                 "New Video FPS: %.3f",
                 chain->rife_model, factor, fps * factor);
    }
    // RIFE-first runs after any hoisted pre-RIFE resize but before the upscale
    // models, so its step goes right after the pre-resize line (if one was
    // pushed at index 0); the default (rife-after) appends it last.
    if (before_upscale)
        c->log_steps.insert(c->log_steps.begin() + (c->has_pre_resize ? 1 : 0),
                            buf);
    else
        c->log_steps.push_back(buf);
    R.enabled = true;
    return true;
}

// Append a RESIZE step with its Spline36 plan; advances *cw/*ch to nw x nh.
// Requires the CUDA context to be current (plan creation allocates).
bool push_resize_step(aji_ctx *c, int *cw, int *ch, int nw, int nh)
{
    Step st{Step::RESIZE, nw, nh, -1};
    st.plan = aji_resize_plan_create(*cw, *ch, nw, nh);
    if (!st.plan) {
        c->set_error("resize plan %dx%d -> %dx%d allocation failed",
                     *cw, *ch, nw, nh);
        return false;
    }
    c->steps.push_back(st);
    *cw = nw;
    *ch = nh;
    return true;
}

void finalize_log(aji_ctx *c)
{
    std::string out;
    for (auto &l : c->log_info)
        out += l + "\n";
    out += "\n";
    for (size_t i = 0; i < c->log_steps.size(); i++)
        out += std::to_string(i + 1) + ". " + c->log_steps[i] + "\n";
    c->current_log = out;
}

// Input frame count of <model_dir>/<name>.onnx (1 = single-frame model),
// cached while the file's size/mtime are unchanged.
aji_ctx::OnnxFrames onnx_frames(aji_ctx *c, const std::string &name)
{
    const std::string path =
        (fs::path(c->model_dir) / (name + ".onnx")).string();
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    const int64_t mtime =
        ec ? 0
           : (int64_t)fs::last_write_time(path, ec).time_since_epoch().count();
    auto it = c->onnx_frames.find(path);
    if (it != c->onnx_frames.end() && it->second.size == size &&
        it->second.mtime == mtime)
        return it->second;

    aji_ctx::OnnxFrames f;
    f.size = size;
    f.mtime = mtime;
    std::vector<char> blob;
    AjiOnnxInput in;
    if (read_file_bytes(path, &blob) &&
        aji_onnx_input(blob.data(), blob.size(), &in)) {
        f.frames = aji_onnx_temporal_frames(in);
        f.rank5 = in.dims.size() == 5;
        f.elem_type = in.elem_type;
        f.dynamic_input = std::any_of(in.dims.begin(), in.dims.end(),
                                     [](int64_t d) { return d < 0; });
    }
    c->onnx_frames[path] = f;
    return f;
}

void temporal_teardown(aji_ctx *c)
{
    auto &TT = c->temporal;
    if (TT.enabled && getenv("AJI_TEMPORAL_STATS")) {
        fprintf(stderr, "aji temporal stats: %llu ingested",
                (unsigned long long)TT.ingests);
        for (size_t i = 0; i < TT.st.size(); i++)
            fprintf(stderr, "; stage %zu (%d frames): %llu runs", i,
                    TT.st[i].t, (unsigned long long)TT.st[i].runs);
        fprintf(stderr, "\n");
    }
    for (auto &T : TT.st) {
        cudaFree(T.ring);
        cudaFree(T.diff);
        cudaFree(T.in_tensor);
        cudaFree(T.out_tensor);
    }
    TT = {};
}

// Allocate the history rings and model tensors for the configured temporal
// stages. Requires the CUDA context to be current.
bool setup_temporal(aji_ctx *c)
{
    auto &TT = c->temporal;
    TT.r = 0;
    for (const auto &T : TT.st)
        TT.r += T.r;
    for (auto &T : TT.st) {
        // Stage 0 keeps the last 2R+5 frames ingested (R: the total radius;
        // callers ingest up to R+4 ahead of the next output, see aji.h).
        // Later stages are filled in order, on demand, and only ever need
        // their own window held.
        T.nslots = &T == &TT.st[0] ? 2 * TT.r + 5 : T.t + 4;
        const size_t frame = (size_t)3 * T.w * T.h * 2;
        if (cudaMalloc(&T.ring, frame * T.nslots) != cudaSuccess ||
            cudaMalloc((void **)&T.diff, sizeof(float) * T.nslots) !=
                cudaSuccess ||
            cudaMalloc(&T.in_tensor, (size_t)3 * T.t * T.pw * T.ph * 2) !=
                cudaSuccess ||
            cudaMalloc(&T.out_tensor, (size_t)3 * T.pow * T.poh * 2) !=
                cudaSuccess) {
            c->set_error("temporal buffer allocation failed");
            return false;
        }
        cudaMemset(T.diff, 0, sizeof(float) * T.nslots);
        T.seq.assign(T.nslots, -1);
        T.matrix.assign(T.nslots, AJI_MATRIX_BT709);
        T.range.assign(T.nslots, AJI_RANGE_LIMITED);
    }
    TT.enabled = true;
    return true;
}

std::string fmt_num(double v)
{
    if (v >= 1e300)
        return "inf";
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

} // namespace

extern "C" AJI_EXPORT aji_ctx *aji_create(const aji_create_params *params)
{
    if (!params || params->api_version != AJI_API_VERSION)
        return nullptr;

    auto c = std::make_unique<aji_ctx>();
    c->logger.fn = params->log;
    c->logger.opaque = params->log_opaque;
    c->graph_ok = !getenv("AJI_NO_GRAPH");  // debug/benchmark escape hatch

    if (cuInit(0) != CUDA_SUCCESS) {
        c->set_error("cuInit failed");
        return nullptr;
    }
    if (params->cuda_context) {
        c->cu_ctx = (CUcontext)params->cuda_context;
    } else {
        CUdevice dev;
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&c->cu_ctx, dev) != CUDA_SUCCESS) {
            c->set_error("failed to retain primary CUDA context");
            return nullptr;
        }
        c->owns_primary = true;
        c->cu_dev = dev;
    }
    {
        CtxGuard guard(c->cu_ctx);
        // cudaGetDevice reports the current (pushed) context's device as a
        // runtime ordinal, the numbering cudaGetDeviceProperties and
        // trtexec --device use
        if (!guard.ok || cuCtxGetDevice(&c->cu_dev) != CUDA_SUCCESS ||
            cudaGetDevice(&c->cuda_dev) != cudaSuccess) {
            c->set_error("failed to query the CUDA context's device");
            if (c->owns_primary)
                cuDevicePrimaryCtxRelease(c->cu_dev);
            return nullptr;
        }
    }

    c->runtime.reset(nvinfer1::createInferRuntime(c->logger));
#ifdef AJI_TRTRTX
    if (ajicore::forced_failure("create"))
        c->runtime.reset();
#endif
    if (!c->runtime) {
        c->set_error("createInferRuntime failed");
        return nullptr;
    }

    if (params->conf_path && params->conf_path[0]) {
        c->conf_mode = true;
        c->conf_path = params->conf_path;
        c->model_dir = params->model_dir ? params->model_dir : ".";
        c->trtexec = params->trtexec ? params->trtexec : "trtexec";
        c->trtexec_env = params->trtexec_env ? params->trtexec_env : "";
        c->slot = params->slot >= 0 ? params->slot : 1;
        c->rife_model_dir =
            params->rife_model_dir ? params->rife_model_dir : "";
        c->async_build = params->async_build != 0;
        return c.release();
    }

    // direct mode
    if (!params->engine_path || params->max_width < 2 || params->max_height < 2) {
        c->set_error("direct mode requires engine_path and max dims");
        return nullptr;
    }
    c->max_w = params->max_width;
    c->max_h = params->max_height;
    {
        CtxGuard guard(c->cu_ctx);
        if (!guard.ok) {
            c->set_error("cuCtxPushCurrent failed");
            return nullptr;
        }
        // Probe at max dims, not a fixed small shape: static engines
        // (the default trt_engine_settings build) accept only their one
        // resolution.
        const int pw = c->max_w >= 2 ? c->max_w : 64;
        const int ph = c->max_h >= 2 ? c->max_h : 64;
        if (!load_engine(c.get(), params->engine_path, &c->direct, pw, ph))
            return nullptr;
        c->scale = c->direct.out_w / pw;
        const size_t bytes =
            (size_t)3 * c->max_w * c->max_h * 2 * c->scale * c->scale;
        if (!ensure_buffers(c.get(), bytes))
            return nullptr;
    }
    return c.release();
}

extern "C" AJI_EXPORT int aji_set_slot(aji_ctx *c, int slot)
{
    // slot 0 = bypass: configure() finds no such slot and reports
    // passthrough, which is exactly the wanted "off" behavior.
    if (!c || slot < 0)
        return AJI_ERR;
    c->slot = slot;
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_configure(aji_ctx *c, int w, int h, double fps,
                                        int *out_w, int *out_h)
{
    if (!c || w < 2 || h < 2 || (w & 1) || (h & 1))
        return AJI_ERR_SHAPE;

    if (!c->conf_mode) {
        if (w > c->max_w || h > c->max_h) {
            c->set_error("input %dx%d exceeds direct-mode max %dx%d", w, h,
                         c->max_w, c->max_h);
            return AJI_ERR_SHAPE;
        }
        if (!c->direct.exec->setInputShape(c->direct.in_name,
                                           nvinfer1::Dims4{1, 3, h, w})) {
            c->set_error("engine rejects input %dx%d", w, h);
            return AJI_ERR_SHAPE;
        }
        nvinfer1::Dims od = c->direct.exec->getTensorShape(c->direct.out_name);
        c->in_w = w; c->in_h = h;
        c->out_w = (int)od.d[3]; c->out_h = (int)od.d[2];
        c->active = true;
        if (out_w) *out_w = c->out_w;
        if (out_h) *out_h = c->out_h;
        return 1;
    }

    // Conf mode: re-read the conf each configure so edits apply without
    // restarting the player (matches the Python init() behavior).
    std::string err;
    aji_conf_load(c->conf_path.c_str(), &c->conf, &err);

    // Held from here: clearing steps frees their resize plans (cudaFree),
    // and plan creation below allocates.
    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }

    // Ticketed (pipelined) callers drain before reconfiguring, but the
    // teardown below frees device state in-flight work may reference -
    // make that unconditional.
    cudaDeviceSynchronize();

    c->engines.clear();
    for (Step &st : c->steps)
        aji_plan_destroy(st.plan);
    c->steps.clear();
    aji_plan_destroy(c->pre_resize_plan);
    aji_plan_destroy(c->pr_pre_plan);
    aji_plan_destroy(c->pr_post_plan);
    c->pre_resize_plan = c->pr_pre_plan = c->pr_post_plan = nullptr;
    c->has_pre_resize = false;
    c->pre_src_w = c->pre_src_h = 0;
    c->work_w = c->work_h = 0;
    rife_teardown(c);
    temporal_teardown(c);
    if (c->graph_exec) {
        cudaGraphExecDestroy(c->graph_exec);
        c->graph_exec = nullptr;
    }
    c->log_info.clear();
    c->log_steps.clear();
    c->active = false;
    c->in_w = w; c->in_h = h;
    c->out_w = w; c->out_h = h;

    auto sit = c->conf.slots.find(c->slot);
    std::string profile = sit != c->conf.slots.end()
                              ? sit->second.profile_name : "(missing slot)";
    if (c->slot == 0)
        profile = "Off";  // slot 0 = bypass, deliberately matches nothing
    else if (c->slot < 10)
        profile = std::to_string(c->slot) + ". " + profile;
    c->log_info.push_back("Upscale Profile: " + profile);
    {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "Original Video Resolution: %dx%d;    Original Video FPS: %.3f",
                 w, h, fps);
        c->log_info.push_back(buf);
    }

    const AjiChainConf *chain = nullptr;
    if (sit != c->conf.slots.end()) {
        const double px = (double)w * h;
        for (const auto &ch : sit->second.chains) {
            if (ch.min_px <= px && px <= ch.max_px &&
                ch.min_fps <= fps && fps <= ch.max_fps) {
                // Skip a matching chain whose model file is missing (e.g. a
                // stale config reference after a model rename): degrade to the
                // next chain / passthrough instead of failing playback.
                std::string missing = aji_chain_missing_model(c->model_dir, ch);
                if (!missing.empty()) {
                    c->log_info.push_back(
                        "Chain " + std::to_string(ch.index) +
                        " skipped: model not found: " + missing + ".onnx");
                    continue;
                }
                chain = &ch;
                break;
            }
        }
    }
    if (!chain) {
        c->log_info.push_back("No Chains Activated");
        finalize_log(c);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;
    }

    c->log_info.push_back(
        "Active Upscale Chain: " + std::to_string(chain->index) +
        ";    Resolution Range: " + chain->min_resolution + " - " +
        chain->max_resolution + ";    FPS Range: " + fmt_num(chain->min_fps) +
        " - " + fmt_num(chain->max_fps));

    if (!c->engines_cleaned) {
        clean_stale_engines(c);
        c->engines_cleaned = true;
    }

    std::string settings_tpl = c->conf.trt_engine_settings.empty()
                                   ? DEFAULT_TRT_ENGINE_SETTINGS
                                   : c->conf.trt_engine_settings;

    int cw = w, ch = h;

    // Temporal models: frame counts per model step (conf override, else the
    // onnx input shape). Each one becomes a temporal stage.
    std::vector<aji_ctx::OnnxFrames> model_frames(chain->models.size());
    int n_temporal = 0;
    for (size_t mi = 0; mi < chain->models.size(); mi++) {
        const auto &m = chain->models[mi];
        if (m.name.empty())
            continue;
        model_frames[mi] = onnx_frames(c, m.name);
        if (m.frames > 0)
            model_frames[mi].frames = m.frames;
        const int t = model_frames[mi].frames;
        // Without a shape profile trtexec replaces dynamic dimensions with
        // 1, creating an unusable engine that gets rebuilt on every retry.
        // Temporal models get an explicit shape below regardless of preset.
        if (t <= 1 && model_frames[mi].dynamic_input &&
            settings_tpl.find("--minShapes=") == std::string::npos &&
            settings_tpl.find("--optShapes=") == std::string::npos &&
            settings_tpl.find("--maxShapes=") == std::string::npos) {
            c->set_error("%s: Static ONNX requires a model with fixed input "
                         "dimensions. This model has dynamic dimensions; "
                         "select Static or Dynamic in the TensorRT engine "
                         "settings (or set --optShapes=input:%%video_resolution%%).",
                         m.name.c_str());
            finalize_log(c);
            return AJI_ERR_CONF;
        }
        if (t <= 1)
            continue;
        if (t % 2 == 0 || t > AJI_TEMPORAL_MAX) {
            c->set_error("%s: temporal models need an odd frame count up to "
                         "%d (got %d)", m.name.c_str(), AJI_TEMPORAL_MAX, t);
            finalize_log(c);
            return AJI_ERR_CONF;
        }
        if (model_frames[mi].elem_type == 1 /* FLOAT */) {
            c->set_error("%s: model input is fp32; temporal models need fp16 "
                         "I/O (traiNNer-redux convert_to_onnx: io_dtype: fp16)",
                         m.name.c_str());
            finalize_log(c);
            return AJI_ERR_CONF;
        }
        n_temporal++;
    }
    // RIFE-first would feed interpolated frames into the temporal model's
    // window; interpolate the upscaled frames instead.
    const bool rife_before = chain->rife_before_upscale && n_temporal == 0;
    if (chain->rife && chain->rife_before_upscale && n_temporal)
        c->log_steps.push_back("RIFE before upscale is not supported with a "
                               "temporal model; interpolating after upscale");

    // Pre-RIFE downscale: in rife-first mode, hoist the FIRST model's explicit
    // "resize before upscale" ahead of RIFE so interpolation runs on the
    // smaller frame (order: resize -> RIFE -> upscale). The filter applies it
    // to source frames via aji_resize; the upscale chain then starts here at
    // the work resolution. Rife-after and no-RIFE paths are unaffected.
    const bool rife_first_mode = chain->rife && !c->rife_model_dir.empty() &&
                                 rife_before;
    if (rife_first_mode && !chain->models.empty()) {
        int ww, wh;
        if (first_resize_target(chain->models[0], w, h, &ww, &wh)) {
            c->pre_resize_plan = aji_resize_plan_create(w, h, ww, wh);
            if (!c->pre_resize_plan) {
                c->set_error("pre-rife resize plan %dx%d -> %dx%d allocation "
                             "failed", w, h, ww, wh);
                return AJI_ERR_CUDA;
            }
            c->has_pre_resize = true;
            c->pre_src_w = w;
            c->pre_src_h = h;
            c->work_w = ww;
            c->work_h = wh;
            cw = ww;
            ch = wh;
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "Applied Resize Before Upscale: %dx%d (before RIFE)",
                     ww, wh);
            c->log_steps.push_back(buf);
        }
    }
    // RIFE-first interpolates at this resolution (work res if a pre-RIFE
    // downscale was hoisted, else source); the model loop advances cw/ch.
    const int rife_in_w = cw, rife_in_h = ch;

    size_t max_bytes = (size_t)3 * cw * ch * 2;
    if (c->has_pre_resize)  // source-res RGB also passes through aji_resize
        max_bytes = std::max(max_bytes, (size_t)3 * w * h * 2);
    bool any_model = false;

    for (size_t mi = 0; mi < chain->models.size(); mi++) {
        const auto &m = chain->models[mi];
        // model[0]'s explicit "resize before upscale" is hoisted ahead of RIFE
        // (rife-first) and applied by the filter via aji_resize; suppress it
        // here so it isn't re-applied as an upscale step. The maxShapes cap
        // below still guards model[0]'s (work-res) input on dynamic engines.
        const bool skip_resize = (mi == 0 && c->has_pre_resize);
        double factor = m.resize_factor_before_upscale;
        if (m.resize_height_before_upscale != 0 || skip_resize)
            factor = 100;

        char buf[160];
        if (factor != 100) {
            int nw = round_even(cw * factor / 100.0);
            int nh = round_even(ch * factor / 100.0);
            if (nw >= 2 && nh >= 2 && (nw != cw || nh != ch)) {
                if (!push_resize_step(c, &cw, &ch, nw, nh))
                    return AJI_ERR_CUDA;
                snprintf(buf, sizeof(buf),
                         "Applied Resize Factor Before Upscale: %g%%;    "
                         "New Video Resolution: %dx%d", factor, cw, ch);
                c->log_steps.push_back(buf);
            }
        }
        if (!skip_resize && m.resize_height_before_upscale != 0 &&
            (int)m.resize_height_before_upscale != ch) {
            int nw, nh;
            fit_box(cw, ch, m.resize_height_before_upscale * 16.0 / 9.0,
                    m.resize_height_before_upscale, &nw, &nh);
            if (!push_resize_step(c, &cw, &ch, nw, nh))
                return AJI_ERR_CUDA;
            snprintf(buf, sizeof(buf),
                     "Applied Resize Height Before Upscale: %gpx;    "
                     "New Video Resolution: %dx%d",
                     m.resize_height_before_upscale, cw, ch);
            c->log_steps.push_back(buf);
        } else {
            // Dynamic engines (custom trt_engine_settings carrying --maxShapes)
            // can't accept an input larger than their build shape, so clamp to
            // it. Static engines — the default, --optShapes only — are built per
            // resolution and need no cap; large sources upscale natively. (This
            // replaces a hard-coded "downscale anything over 1080p", which only
            // mattered back when every engine was dynamic.)
            int maxW, maxH;
            if (parse_shape_wh(settings_tpl, "--maxShapes", &maxW, &maxH) &&
                (cw > maxW || ch > maxH)) {
                int nw, nh;
                fit_within(cw, ch, maxW, maxH, &nw, &nh);
                if (nw >= 2 && nh >= 2 && (nw != cw || nh != ch)) {
                    if (!push_resize_step(c, &cw, &ch, nw, nh))
                        return AJI_ERR_CUDA;
                    snprintf(buf, sizeof(buf),
                             "Applied Resize to fit dynamic engine max shape;"
                             "    New Video Resolution: %dx%d", cw, ch);
                    c->log_steps.push_back(buf);
                }
            }
        }
        max_bytes = std::max(max_bytes, (size_t)3 * cw * ch * 2);

        if (m.name.empty())
            continue;

        // Temporal models are built static at the input padded to mod 32:
        // the exported graphs only accept their alignment (TFDAT: 8-32 by
        // scale), and the output is cropped back.
        const int t = model_frames[mi].frames;
        const bool rank5 = t > 1 && model_frames[mi].rank5;
        const int bw = t > 1 ? (cw + 31) / 32 * 32 : cw;
        const int bh = t > 1 ? (ch + 31) / 32 * 32 : ch;
        char dims[64];
        if (rank5)
            snprintf(dims, sizeof(dims), "1x%dx3x%dx%d", t, bh, bw);
        else
            snprintf(dims, sizeof(dims), "1x%dx%dx%d", 3 * t, bh, bw);
        std::string settings = settings_tpl;
        if (t > 1 && settings.find("%video_resolution%") == std::string::npos)
            settings = DEFAULT_TRT_ENGINE_SETTINGS;  // custom 1x3 shapes
        size_t pos;
        while ((pos = settings.find("%video_resolution%")) != std::string::npos)
            settings.replace(pos, strlen("%video_resolution%"), dims);

        const std::string epath =
            engine_path_for(c->model_dir, m.name, settings, c->cuda_dev);
        ModelEngine me;
        int er = ensure_engine(c, m.name, settings, epath, &me, bw, bh, 3 * t,
                               nullptr, rank5 ? t : 0);
        if (er < 0) {
            finalize_log(c);
            return AJI_ERR_ENGINE;
        }
        if (er == 0) {
            // building in the background: passthrough until aji_poll()
            // says to reconfigure
            finalize_log(c);
            if (out_w) *out_w = w;
            if (out_h) *out_h = h;
            return 0;
        }
        if (t > 1) {
            c->temporal.st.emplace_back();
            auto &T = c->temporal.st.back();
            const int sx = me.out_w / bw, sy = me.out_h / bh;
            if (sx < 1 || sy < 1 || me.out_w != sx * bw ||
                me.out_h != sy * bh) {
                c->set_error("%s: output %dx%d is not an integer scale of the "
                             "input %dx%d", m.name.c_str(), me.out_w,
                             me.out_h, bw, bh);
                finalize_log(c);
                return AJI_ERR_ENGINE;
            }
            T.t = t;
            T.r = (t - 1) / 2;
            T.w = cw; T.h = ch;
            T.pw = bw; T.ph = bh;
            T.pow = me.out_w; T.poh = me.out_h;
            T.ow = cw * sx; T.oh = ch * sy;
            T.threshold = (float)m.temporal_scene_threshold;
            T.step_idx = c->steps.size();
            c->engines.push_back(std::move(me));
            c->steps.push_back({Step::TEMPORAL, T.ow, T.oh,
                                (int)c->engines.size() - 1});
            max_bytes = std::max(max_bytes, (size_t)3 * T.ow * T.oh * 2);
            cw = T.ow; ch = T.oh;
            if (T.threshold > 0)
                snprintf(buf, sizeof(buf),
                         "Applied Temporal Model: %s (%d frames, scene "
                         "threshold %g);    New Video Resolution: %dx%d",
                         m.name.c_str(), t, T.threshold, cw, ch);
            else
                snprintf(buf, sizeof(buf),
                         "Applied Temporal Model: %s (%d frames);    "
                         "New Video Resolution: %dx%d",
                         m.name.c_str(), t, cw, ch);
        } else {
            cw = me.out_w; ch = me.out_h;
            c->engines.push_back(std::move(me));
            c->steps.push_back({Step::MODEL, cw, ch,
                                (int)c->engines.size() - 1});
            snprintf(buf, sizeof(buf),
                     "Applied Model: %s;    New Video Resolution: %dx%d",
                     m.name.c_str(), cw, ch);
        }
        any_model = true;
        c->log_steps.push_back(buf);
        max_bytes = std::max(max_bytes, (size_t)3 * cw * ch * 2);
    }

    if (chain->rife) {
        if (!c->rife_model_dir.empty()) {
            // RIFE-first interpolates at the work resolution (source, or the
            // hoisted pre-RIFE downscale); the upscale models then run on every
            // frame. The default (rife-after) interpolates the already-upscaled
            // frames (cw, ch). aji_infer_rife validates its inputs against these
            // configured dims either way.
            const int rw = rife_before ? rife_in_w : cw;
            const int rh = rife_before ? rife_in_h : ch;
            if (!setup_rife(c, chain, rw, rh, fps, rife_before)) {
                finalize_log(c);
                return AJI_ERR_ENGINE;
            }
            c->rife.before_upscale = rife_before;
        } else {
            c->log_steps.push_back(
                "RIFE requested by the chain but no rife model dir is "
                "configured; interpolation disabled");
        }
    }

    finalize_log(c);

    if (!any_model && c->steps.empty() && !c->has_pre_resize) {
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        // chain selected no scaling work and no pre-RIFE downscale; rife (if
        // enabled above) still interpolates between the passthrough frames
        return 0;
    }

    if (!ensure_buffers(c, max_bytes))
        return AJI_ERR_CUDA;
    if (!c->temporal.st.empty() && !setup_temporal(c))
        return AJI_ERR_CUDA;

    // In hoist mode the filter downscales source -> work via aji_resize and
    // feeds work-res frames to the upscale chain, so the chain's input geometry
    // is the work resolution (aji_infer validates in->w/h against c->in_w/in_h).
    c->in_w = c->has_pre_resize ? c->work_w : w;
    c->in_h = c->has_pre_resize ? c->work_h : h;
    c->out_w = cw;
    c->out_h = ch;
    c->active = true;
    if (out_w) *out_w = cw;
    if (out_h) *out_h = ch;
    return 1;
}

static int run_chain(aji_ctx *c, const aji_frame *in, const aji_frame *out,
                     cudaStream_t stream);

static bool ensure_stage(void **p, size_t *cur, size_t need)
{
    if (*p && *cur >= need)
        return true;
    cudaFree(*p);
    *p = nullptr;
    *cur = 0;
    if (cudaMalloc(p, need) != cudaSuccess)
        return false;
    *cur = need;
    return true;
}

static bool copy_plane(void *dst, size_t dpitch, const void *src,
                       size_t spitch, size_t row_bytes, size_t rows,
                       cudaStream_t s)
{
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, row_bytes, rows,
                             cudaMemcpyDeviceToDevice, s) == cudaSuccess;
}

// Run the chain via a captured CUDA graph. Returns true if the frame was
// handled (*ret holds the status); false means fall back to the plain path.
static bool infer_via_graph(aji_ctx *c, const aji_frame *in,
                            const aji_frame *out, cudaStream_t stream,
                            int *ret)
{
    const bool out444 = aji_format_is_444(out->format);
    const bool in444 = aji_format_is_444(in->format);
    const int bpp = (in->format == AJI_FMT_P010 || in444) ? 2 : 1;
    // output bytes/sample is the OUTPUT format's (NV12 is the only 8-bit one);
    // input bpp must not leak in here, or NV12->P010 staging is half-sized.
    const int obpp = out->format == AJI_FMT_NV12 ? 1 : 2;
    const size_t in_row = (size_t)in->width * bpp;
    const size_t out_row = (size_t)out->width * obpp;
    const size_t in_y = in_row * in->height;
    const size_t out_y = out_row * out->height;

    if (!ensure_stage(&c->stage_in, &c->stage_in_bytes,
                      in444 ? in_y * 3 : in_y * 3 / 2) ||
        !ensure_stage(&c->stage_out, &c->stage_out_bytes,
                      out444 ? out_y * 3 : out_y * 3 / 2)) {
        c->graph_ok = false;
        return false;
    }

    // staging descriptors: same colorimetry tags, packed strides
    aji_frame sin = *in, sout = *out;
    sin.plane[0] = c->stage_in;
    sin.plane[1] = (char *)c->stage_in + in_y;
    sin.stride[0] = sin.stride[1] = (ptrdiff_t)in_row;
    if (in444) {
        sin.plane[2] = (char *)c->stage_in + 2 * in_y;
        sin.stride[2] = (ptrdiff_t)in_row;
    }
    sout.plane[0] = c->stage_out;
    sout.plane[1] = (char *)c->stage_out + out_y;
    sout.stride[0] = sout.stride[1] = (ptrdiff_t)out_row;
    if (out444) {
        sout.plane[2] = (char *)c->stage_out + 2 * out_y;
        sout.stride[2] = (ptrdiff_t)out_row;
    }

    // 4:2:0 chroma is half-height (one interleaved CbCr plane); 4:4:4 is three
    // full-height planes.
    if (!copy_plane(sin.plane[0], in_row, in->plane[0], in->stride[0],
                    in_row, in->height, stream) ||
        !copy_plane(sin.plane[1], in_row, in->plane[1], in->stride[1],
                    in_row, in444 ? in->height : in->height / 2, stream) ||
        (in444 &&
         !copy_plane(sin.plane[2], in_row, in->plane[2], in->stride[2],
                     in_row, in->height, stream))) {
        c->graph_ok = false;
        return false;
    }

    const int key[9] = {in->format, in->width, in->height, in->siting,
                        in->matrix, in->range, out->width, out->height,
                        out->format};
    if (!c->graph_exec || memcmp(key, c->graph_key, sizeof(key)) != 0) {
        if (c->graph_exec) {
            cudaGraphExecDestroy(c->graph_exec);
            c->graph_exec = nullptr;
        }
        // Warmup run on the staging buffers: performs all lazy allocations
        // (plans, TRT internals) so the capture below is allocation-free,
        // and already computes this frame's output.
        *ret = run_chain(c, &sin, &sout, stream);
        if (*ret != AJI_OK)
            return true;
        cudaStreamSynchronize(stream);

        cudaGraph_t graph = nullptr;
        bool ok = cudaStreamBeginCapture(stream,
                      cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) {
            ok = run_chain(c, &sin, &sout, stream) == AJI_OK;
            // EndCapture must run even on failure to unstick the stream.
            cudaError_t ce = cudaStreamEndCapture(stream, &graph);
            ok = ok && ce == cudaSuccess && graph;
        }
        if (ok)
            ok = cudaGraphInstantiate(&c->graph_exec, graph, 0) == cudaSuccess;
        if (graph)
            cudaGraphDestroy(graph);
        if (ok) {
            memcpy(c->graph_key, key, sizeof(key));
            c->verbose("chain captured as CUDA graph");
        } else {
            cudaGetLastError();  // clear sticky capture errors
            c->graph_ok = false;
            c->verbose("CUDA graph capture unavailable; per-call launches");
            // the warmup output on staging is still valid - fall through
        }
    } else {
        if (cudaGraphLaunch(c->graph_exec, stream) != cudaSuccess) {
            c->graph_ok = false;
            return false;  // plain path recomputes from the real frame
        }
    }

    if (!copy_plane(out->plane[0], out->stride[0], sout.plane[0], out_row,
                    out_row, out->height, stream) ||
        !copy_plane(out->plane[1], out->stride[1], sout.plane[1], out_row,
                    out_row, out444 ? out->height : out->height / 2,
                    stream) ||
        (out444 &&
         !copy_plane(out->plane[2], out->stride[2], sout.plane[2], out_row,
                     out_row, out->height, stream))) {
        c->set_error("staging copy-out failed");
        *ret = AJI_ERR_CUDA;
        return true;
    }
    *ret = AJI_OK;
    return true;
}

extern "C" AJI_EXPORT int aji_infer(aji_ctx *c, const aji_frame *in,
                                    const aji_frame *out, void *cu_stream)
{
    if (!c || !in || !out)
        return AJI_ERR;
    if (!c->active) {
        c->set_error("aji_infer without an active configuration");
        return AJI_ERR;
    }
    if (c->temporal.enabled) {
        c->set_error("the active chain has a temporal model: use "
                     "aji_ingest + aji_infer_seq");
        return AJI_ERR;
    }
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010 &&
        !aji_format_is_444(in->format)) {
        c->set_error("unsupported input format %d", in->format);
        return AJI_ERR_FORMAT;
    }
    // Output is independent of input: any 4:2:0 (NV12/P010) or 4:4:4
    // (planar 4:4:4). The post-kernel quantizes to the output format's depth,
    // so NV12->P010 (8-bit source, 10-bit output) and P010->NV12 are on-GPU.
    if (out->format != AJI_FMT_NV12 && out->format != AJI_FMT_P010 &&
        !aji_format_is_444(out->format)) {
        c->set_error("unsupported output format %d (nv12/p010/planar 444)",
                     out->format);
        return AJI_ERR_FORMAT;
    }
    if (in->width != c->in_w || in->height != c->in_h ||
        out->width != c->out_w || out->height != c->out_h) {
        c->set_error("frame dims %dx%d->%dx%d do not match configured %dx%d->%dx%d",
                     in->width, in->height, out->width, out->height,
                     c->in_w, c->in_h, c->out_w, c->out_h);
        return AJI_ERR_SHAPE;
    }

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }
    cudaStream_t stream = (cudaStream_t)cu_stream;

    // Graph replay needs a non-default stream (capture restriction).
    if (c->graph_ok && stream) {
        int ret;
        if (infer_via_graph(c, in, out, stream, &ret))
            return ret;
    }
    return run_chain(c, in, out, stream);
}

// Source frame -> fp16 NCHW RGB in dst (the chain's first buffer).
static int run_pre(aji_ctx *c, const aji_frame *in, void *dst,
                   cudaStream_t stream)
{
    const aji_csp csp = make_csp(in);

    // 4:4:4 input is already full-resolution: pure matrix convert, no chroma
    // upsample and no plan (aji_pre_plan_create hardcodes w>>1/h>>1 chroma and
    // would silently corrupt 444). Mirrors the RIFE f444 branch.
    const bool in444 = aji_format_is_444(in->format);
    if (!in444) {
        const int pkey[4] = {in->format, in->width, in->height, in->siting};
        if (!c->pre_plan || memcmp(pkey, c->pre_key, sizeof(pkey)) != 0) {
            aji_plan_destroy(c->pre_plan);
            c->pre_plan = aji_pre_plan_create(in->format, in->width, in->height,
                                              in->siting, AJI_FILTER_SPLINE36);
            if (!c->pre_plan) {
                c->set_error("pre plan allocation failed");
                return AJI_ERR_CUDA;
            }
            memcpy(c->pre_key, pkey, sizeof(pkey));
        }
    }

    int err = in444
        ? aji_run_pre444(in->width, in->height, in->plane[0], in->stride[0],
                         in->plane[1], in->stride[1], in->plane[2],
                         in->stride[2], &csp, dst, stream)
        : aji_run_pre(c->pre_plan, in->plane[0], in->stride[0],
                      in->plane[1], in->stride[1], &csp, dst, stream);
    if (err) {
        c->set_error("pre-kernel failed: %s", cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
    return AJI_OK;
}

static size_t tensor_elems(const nvinfer1::Dims &d)
{
    size_t n = 1;
    for (int i = 0; i < d.nbDims; i++)
        n *= (size_t)(d.d[i] > 0 ? d.d[i] : 0);
    return n;
}

// Grow an fp32 staging buffer to hold n elements.
static bool ensure_f32(aji_ctx *c, CudaPtr *buf, size_t *have, size_t n)
{
    if (*buf && *have >= n)
        return true;
    void *p = nullptr;
    if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) {
        c->set_error("cudaMalloc(%zu) failed for fp32 model I/O",
                     n * sizeof(float));
        return false;
    }
    buf->reset(p);
    *have = n;
    return true;
}

// Run one model from the fp16 tensor `in` into the fp16 tensor `out`. Models
// with fp32 inputs/outputs are bound to fp32 staging buffers and cast around
// the enqueue; binding the fp16 tensors to them directly is what turned their
// output into noise.
static int enqueue_model(aji_ctx *c, ModelEngine &me, void *in, void *out,
                         cudaStream_t stream)
{
    void *bind_in = in, *bind_out = out;
    size_t n_in = 0, n_out = 0;
    if (me.in_fp32) {
        n_in = tensor_elems(me.exec->getTensorShape(me.in_name));
        if (!ensure_f32(c, &me.in32, &me.in32_elems, n_in))
            return AJI_ERR_CUDA;
        int err = aji_f16_to_f32(in, me.in32.get(), n_in, stream);
        if (err) {
            c->set_error("fp32 input cast failed: %s",
                         cudaGetErrorString((cudaError_t)err));
            return AJI_ERR_CUDA;
        }
        bind_in = me.in32.get();
    }
    if (me.out_fp32) {
        n_out = tensor_elems(me.exec->getTensorShape(me.out_name));
        if (!ensure_f32(c, &me.out32, &me.out32_elems, n_out))
            return AJI_ERR_CUDA;
        bind_out = me.out32.get();
    }
    if (!me.exec->setTensorAddress(me.in_name, bind_in) ||
        !me.exec->setTensorAddress(me.out_name, bind_out)) {
        c->set_error("setTensorAddress failed");
        return AJI_ERR_ENGINE;
    }
    if (!me.exec->enqueueV3(stream)) {
        c->set_error("enqueueV3 failed");
        return AJI_ERR_ENGINE;
    }
    if (me.out_fp32) {
        int err = aji_f32_to_f16(me.out32.get(), out, n_out, stream);
        if (err) {
            c->set_error("fp32 output cast failed: %s",
                         cudaGetErrorString((cudaError_t)err));
            return AJI_ERR_CUDA;
        }
    }
    return AJI_OK;
}

// Run steps [first, last) over the ping-pong buffers, starting in buf[*cur];
// *cur/*cw/*ch track the current buffer and dims.
static int run_steps(aji_ctx *c, size_t first, size_t last, int *cur,
                     int *cw, int *ch, cudaStream_t stream)
{
    for (size_t i = first; i < last; i++) {
        const Step &st = c->steps[i];
        if (st.kind == Step::RESIZE) {
            int err = aji_run_resize(st.plan, c->buf[*cur], c->buf[*cur ^ 1],
                                     stream);
            if (err) {
                c->set_error("resize kernel failed: %s",
                             cudaGetErrorString((cudaError_t)err));
                return AJI_ERR_CUDA;
            }
        } else {
            int err = enqueue_model(c, c->engines[st.engine_idx],
                                    c->buf[*cur], c->buf[*cur ^ 1], stream);
            if (err != AJI_OK)
                return err;
        }
        *cur ^= 1;
        *cw = st.out_w;
        *ch = st.out_h;
    }
    return AJI_OK;
}

// fp16 NCHW RGB (cw x ch) in src -> the output frame, in the source's
// matrix/range.
static int run_post(aji_ctx *c, int matrix, int range, const void *src,
                    int cw, int ch, const aji_frame *out, cudaStream_t stream)
{
    if (aji_format_is_444(out->format)) {
        // full-resolution chroma: pure matrix + output-depth quantize, no
        // resampling and no plan (this exceeds the reference pipeline,
        // which always subsampled back to 4:2:0)
        const aji_csp ocsp = aji_make_csp(out->format, matrix, range);
        int err4 = aji_run_post444(out->format, cw, ch, src, &ocsp,
                                   out->plane[0], out->stride[0],
                                   out->plane[1], out->stride[1],
                                   out->plane[2], out->stride[2], stream);
        if (err4) {
            c->set_error("post444 kernel failed: %s",
                         cudaGetErrorString((cudaError_t)err4));
            return AJI_ERR_CUDA;
        }
        return AJI_OK;
    }

    // The reference pipeline's final resize.Spline36(format=YUV420...) call
    // always subsamples with LEFT placement: VS/zimg only propagates chroma
    // location between subsampled formats, and RGB sources have none, so the
    // frame prop is ignored and no chromaloc argument is passed. Match it
    // regardless of the frame's tagged siting.
    const int qkey[4] = {out->format, cw, ch, AJI_SITING_LEFT};
    if (!c->post_plan || memcmp(qkey, c->post_key, sizeof(qkey)) != 0) {
        aji_plan_destroy(c->post_plan);
        c->post_plan = aji_post_plan_create(out->format, cw, ch,
                                            AJI_SITING_LEFT,
                                            AJI_FILTER_SPLINE36);
        if (!c->post_plan) {
            c->set_error("post plan allocation failed");
            return AJI_ERR_CUDA;
        }
        memcpy(c->post_key, qkey, sizeof(qkey));
    }

    // Quantize to the OUTPUT format's depth/range (NV12 8-bit vs P010 10-bit),
    // not the input's. Same matrix/range as the input (the model preserves
    // them).
    const aji_csp ocsp = aji_make_csp(out->format, matrix, range);
    int err = aji_run_post(c->post_plan, src, &ocsp,
                           out->plane[0], out->stride[0], out->plane[1],
                           out->stride[1], stream);
    if (err) {
        c->set_error("post-kernel failed: %s", cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
    return AJI_OK;
}

// The chain body: pre -> (model/resize steps) -> post, launched on `stream`
// against whatever plane pointers the frame descriptors carry (real frames
// on the plain path, stable staging buffers on the graph path). Everything
// in here must be capture-safe once warm: allocations only happen on plan
// key changes, which a warmup run performs before any capture.
static int run_chain(aji_ctx *c, const aji_frame *in, const aji_frame *out,
                     cudaStream_t stream)
{
    int cur = 0;
    int err = run_pre(c, in, c->buf[cur], stream);
    if (err != AJI_OK)
        return err;

    int cw = in->width, ch = in->height;

    if (!c->conf_mode) {
        err = enqueue_model(c, c->direct, c->buf[cur], c->buf[cur ^ 1], stream);
        if (err != AJI_OK)
            return err;
        cur ^= 1;
        cw = c->out_w; ch = c->out_h;
    } else {
        err = run_steps(c, 0, c->steps.size(), &cur, &cw, &ch, stream);
        if (err != AJI_OK)
            return err;
    }
    return run_post(c, in->matrix, in->range, c->buf[cur], cw, ch, out,
                    stream);
}

// Frame format/dims checks for the temporal calls (aji_infer's rules).
static int check_frame(aji_ctx *c, const aji_frame *f, bool input)
{
    if (f->format != AJI_FMT_NV12 && f->format != AJI_FMT_P010 &&
        !aji_format_is_444(f->format)) {
        c->set_error("unsupported %s format %d (nv12/p010/planar 444)",
                     input ? "input" : "output", f->format);
        return AJI_ERR_FORMAT;
    }
    const int w = input ? c->in_w : c->out_w;
    const int h = input ? c->in_h : c->out_h;
    if (f->width != w || f->height != h) {
        c->set_error("%s dims %dx%d do not match configured %dx%d",
                     input ? "input" : "output", f->width, f->height, w, h);
        return AJI_ERR_SHAPE;
    }
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_temporal_radius(aji_ctx *c)
{
    return (c && c->active && c->temporal.enabled) ? c->temporal.r : 0;
}

extern "C" AJI_EXPORT void aji_temporal_reset(aji_ctx *c)
{
    if (c && c->temporal.enabled)
        for (auto &T : c->temporal.st)
            std::fill(T.seq.begin(), T.seq.end(), -1);
}

static int temporal_slot(aji_ctx *c, size_t lv, int64_t seq)
{
    auto &T = c->temporal.st[lv];
    if (seq < 0)
        return -1;
    const int slot = (int)(seq % T.nslots);
    return T.seq[slot] == seq ? slot : -1;
}

static void *temporal_frame(aji_ctx *c, size_t lv, int slot)
{
    auto &T = c->temporal.st[lv];
    return (char *)T.ring + (size_t)slot * 3 * T.w * T.h * 2;
}

// Store the frame in src (the stage's ring frame dims) as `seq` in stage
// lv's ring and record its scene-change metric against seq-1, if held.
static int temporal_store(aji_ctx *c, size_t lv, const void *src,
                          int64_t seq, int matrix, int range,
                          cudaStream_t stream)
{
    auto &T = c->temporal.st[lv];
    const int slot = (int)(seq % T.nslots);
    void *dst = temporal_frame(c, lv, slot);
    if (cudaMemcpyAsync(dst, src, (size_t)3 * T.w * T.h * 2,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        c->set_error("temporal ingest copy failed");
        return AJI_ERR_CUDA;
    }
    // scene-change metric against the previous frame, if held
    cudaMemsetAsync(&T.diff[slot], 0, sizeof(float), stream);
    const int prev = temporal_slot(c, lv, seq - 1);
    if (prev >= 0) {
        int err = aji_rgb_diff(temporal_frame(c, lv, prev), dst, T.w, T.h,
                               &T.diff[slot], stream);
        if (err) {
            c->set_error("temporal diff kernel failed: %s",
                         cudaGetErrorString((cudaError_t)err));
            return AJI_ERR_CUDA;
        }
    }
    T.seq[slot] = seq;
    T.matrix[slot] = matrix;
    T.range[slot] = range;
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_ingest(aji_ctx *c, const aji_frame *in,
                                     uint64_t seq, void *cu_stream)
{
    if (!c || !in)
        return AJI_ERR;
    auto &TT = c->temporal;
    if (!c->active || !TT.enabled) {
        c->set_error("aji_ingest without an active temporal configuration");
        return AJI_ERR;
    }
    if (seq > (uint64_t)INT64_MAX)
        return AJI_ERR;
    int err = check_frame(c, in, true);
    if (err != AJI_OK)
        return err;
    if (temporal_slot(c, 0, (int64_t)seq) >= 0)
        return AJI_OK;  // already held

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }
    cudaStream_t stream = (cudaStream_t)cu_stream;

    // the steps before the first temporal model, once per source frame
    int cur = 0, cw = in->width, ch = in->height;
    err = run_pre(c, in, c->buf[cur], stream);
    if (err == AJI_OK)
        err = run_steps(c, 0, TT.st[0].step_idx, &cur, &cw, &ch, stream);
    if (err != AJI_OK)
        return err;
    TT.ingests++;
    return temporal_store(c, 0, c->buf[cur], (int64_t)seq, in->matrix,
                          in->range, stream);
}

// Run stage lv's model on the window around `center` (which its ring
// holds) and crop the output into buf[0]. Window frames that are not held
// (stream start, after a reset, end of stream) or lie across a scene cut
// from the center are replaced by the nearest frame on the center's side;
// a cut pointer exists only between two consecutive held frames.
static int run_temporal_stage(aji_ctx *c, size_t lv, int64_t center,
                              int cslot, cudaStream_t stream)
{
    auto &T = c->temporal.st[lv];
    aji_gather_args args = {};
    int slots[AJI_TEMPORAL_MAX];
    bool held[AJI_TEMPORAL_MAX];
    slots[T.r] = cslot;
    held[T.r] = true;
    for (int dir = -1; dir <= 1; dir += 2) {
        bool clamped = false;
        for (int k = 1; k <= T.r; k++) {
            const int j = T.r + dir * k;
            const int s = clamped ? -1 : temporal_slot(c, lv, center + dir * k);
            clamped = s < 0;
            slots[j] = clamped ? slots[j - dir] : s;
            held[j] = !clamped;
        }
    }
    for (int j = 0; j < T.t; j++) {
        args.src[j] = temporal_frame(c, lv, slots[j]);
        if (j + 1 < T.t && held[j] && held[j + 1])
            args.cut[j] = &T.diff[slots[j + 1]];
    }
    int err = aji_temporal_gather(&args, T.t, T.threshold, T.w, T.h, T.pw,
                                  T.ph, T.in_tensor, stream);
    if (err) {
        c->set_error("temporal gather kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }

    const Step &st = c->steps[T.step_idx];
    ModelEngine &me = c->engines[st.engine_idx];
    if (!me.exec->setTensorAddress(me.in_name, T.in_tensor) ||
        !me.exec->setTensorAddress(me.out_name, T.out_tensor)) {
        c->set_error("setTensorAddress failed");
        return AJI_ERR_ENGINE;
    }
    if (!me.exec->enqueueV3(stream)) {
        c->set_error("enqueueV3 failed");
        return AJI_ERR_ENGINE;
    }
    T.runs++;

    // crop the padded model output into the chain buffer
    const size_t splane = (size_t)T.pow * T.poh * 2;
    const size_t dplane = (size_t)T.ow * T.oh * 2;
    for (int pl = 0; pl < 3; pl++) {
        if (cudaMemcpy2DAsync((char *)c->buf[0] + pl * dplane,
                              (size_t)T.ow * 2,
                              (char *)T.out_tensor + pl * splane,
                              (size_t)T.pow * 2, (size_t)T.ow * 2, T.oh,
                              cudaMemcpyDeviceToDevice, stream) !=
            cudaSuccess) {
            c->set_error("temporal crop copy failed");
            return AJI_ERR_CUDA;
        }
    }
    return AJI_OK;
}

// Make stage lv's ring hold frame p if it can: run stage lv-1 on its window
// around p (making those frames held first, oldest first) and the steps
// between the two stages. A frame is computed once and then served from the
// ring. Not possible when the source frame was never ingested (before the
// stream start, past its end, a gap, or after a reset): the frame stays
// absent and windows clamp around it.
static int temporal_ensure(aji_ctx *c, size_t lv, int64_t p,
                           cudaStream_t stream)
{
    if (p < 0 || temporal_slot(c, lv, p) >= 0 || lv == 0)
        return AJI_OK;
    const auto &S = c->temporal.st[lv - 1];
    for (int64_t q = p - S.r; q <= p + S.r; q++) {
        int err = temporal_ensure(c, lv - 1, q, stream);
        if (err != AJI_OK)
            return err;
    }
    const int cslot = temporal_slot(c, lv - 1, p);
    if (cslot < 0)
        return AJI_OK;
    int err = run_temporal_stage(c, lv - 1, p, cslot, stream);
    if (err != AJI_OK)
        return err;
    int cur = 0, cw = S.ow, ch = S.oh;
    err = run_steps(c, S.step_idx + 1, c->temporal.st[lv].step_idx, &cur,
                    &cw, &ch, stream);
    if (err != AJI_OK)
        return err;
    return temporal_store(c, lv, c->buf[cur], p, S.matrix[cslot],
                          S.range[cslot], stream);
}

extern "C" AJI_EXPORT int aji_infer_seq(aji_ctx *c, uint64_t seq,
                                        const aji_frame *out, void *cu_stream)
{
    if (!c || !out)
        return AJI_ERR;
    auto &TT = c->temporal;
    if (!c->active || !TT.enabled) {
        c->set_error("aji_infer_seq without an active temporal configuration");
        return AJI_ERR;
    }
    int err = check_frame(c, out, false);
    if (err != AJI_OK)
        return err;
    const int64_t center = (int64_t)seq;
    if (seq > (uint64_t)INT64_MAX || temporal_slot(c, 0, center) < 0) {
        c->set_error("aji_infer_seq: frame %llu was not ingested",
                     (unsigned long long)seq);
        return AJI_ERR;
    }

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }
    cudaStream_t stream = (cudaStream_t)cu_stream;

    // stacked temporal models: bring the last stage's window up to date
    const size_t last = TT.st.size() - 1;
    const auto &T = TT.st[last];
    for (int64_t q = center - T.r; last > 0 && q <= center + T.r; q++) {
        err = temporal_ensure(c, last, q, stream);
        if (err != AJI_OK)
            return err;
    }
    const int cslot = temporal_slot(c, last, center);
    if (cslot < 0) {
        c->set_error("aji_infer_seq: frame %llu is not available",
                     (unsigned long long)seq);
        return AJI_ERR;
    }

    err = run_temporal_stage(c, last, center, cslot, stream);
    if (err != AJI_OK)
        return err;
    int cur = 0;
    int cw = T.ow, ch = T.oh;
    err = run_steps(c, T.step_idx + 1, c->steps.size(), &cur, &cw, &ch,
                    stream);
    if (err != AJI_OK)
        return err;
    return run_post(c, T.matrix[cslot], T.range[cslot], c->buf[cur], cw, ch,
                    out, stream);
}

extern "C" AJI_EXPORT int aji_pre_resize(aji_ctx *c, int *work_w, int *work_h)
{
    if (!c || !c->has_pre_resize)
        return 0;
    if (work_w)
        *work_w = c->work_w;
    if (work_h)
        *work_h = c->work_h;
    return 1;
}

// Downscale one source-resolution frame to the work resolution (the rife-first
// pre-RIFE step), reusing the pre -> resize -> post kernels over c->buf with
// aji_resize's OWN pre/post plans so it doesn't thrash run_chain's per-frame
// keys. A trimmed run_chain: no model steps, a single fixed resize.
extern "C" AJI_EXPORT int aji_resize(aji_ctx *c, const aji_frame *in,
                                     const aji_frame *out, void *cu_stream)
{
    if (!c || !in || !out)
        return AJI_ERR;
    if (!c->has_pre_resize || !c->pre_resize_plan) {
        c->set_error("aji_resize without a configured pre-RIFE downscale");
        return AJI_ERR;
    }
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010 &&
        !aji_format_is_444(in->format)) {
        c->set_error("unsupported input format %d", in->format);
        return AJI_ERR_FORMAT;
    }
    if (out->format != in->format) {
        c->set_error("aji_resize requires matching in/out formats");
        return AJI_ERR_FORMAT;
    }
    if (in->width != c->pre_src_w || in->height != c->pre_src_h ||
        out->width != c->work_w || out->height != c->work_h) {
        c->set_error("aji_resize dims %dx%d->%dx%d do not match configured "
                     "%dx%d->%dx%d", in->width, in->height, out->width,
                     out->height, c->pre_src_w, c->pre_src_h, c->work_w,
                     c->work_h);
        return AJI_ERR_SHAPE;
    }

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }
    cudaStream_t stream = (cudaStream_t)cu_stream;

    const aji_csp csp = make_csp(in);
    const bool in444 = aji_format_is_444(in->format);
    if (!in444) {
        const int pkey[4] = {in->format, in->width, in->height, in->siting};
        if (!c->pr_pre_plan || memcmp(pkey, c->pr_pre_key, sizeof(pkey)) != 0) {
            aji_plan_destroy(c->pr_pre_plan);
            c->pr_pre_plan = aji_pre_plan_create(in->format, in->width,
                                                 in->height, in->siting,
                                                 AJI_FILTER_SPLINE36);
            if (!c->pr_pre_plan) {
                c->set_error("pre-resize pre plan allocation failed");
                return AJI_ERR_CUDA;
            }
            memcpy(c->pr_pre_key, pkey, sizeof(pkey));
        }
    }

    int err = in444
        ? aji_run_pre444(in->width, in->height, in->plane[0], in->stride[0],
                         in->plane[1], in->stride[1], in->plane[2],
                         in->stride[2], &csp, c->buf[0], stream)
        : aji_run_pre(c->pr_pre_plan, in->plane[0], in->stride[0],
                      in->plane[1], in->stride[1], &csp, c->buf[0], stream);
    if (err) {
        c->set_error("pre-resize pre-kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }

    err = aji_run_resize(c->pre_resize_plan, c->buf[0], c->buf[1], stream);
    if (err) {
        c->set_error("pre-resize kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }

    if (aji_format_is_444(out->format)) {
        const aji_csp ocsp = aji_make_csp(out->format, in->matrix,
                                          in->range);
        int err4 = aji_run_post444(out->format, c->work_w, c->work_h, c->buf[1], &ocsp,
                                   out->plane[0], out->stride[0],
                                   out->plane[1], out->stride[1],
                                   out->plane[2], out->stride[2], stream);
        if (err4) {
            c->set_error("pre-resize post444 kernel failed: %s",
                         cudaGetErrorString((cudaError_t)err4));
            return AJI_ERR_CUDA;
        }
        return AJI_OK;
    }

    const int qkey[4] = {out->format, c->work_w, c->work_h, AJI_SITING_LEFT};
    if (!c->pr_post_plan || memcmp(qkey, c->pr_post_key, sizeof(qkey)) != 0) {
        aji_plan_destroy(c->pr_post_plan);
        c->pr_post_plan = aji_post_plan_create(out->format, c->work_w,
                                               c->work_h, AJI_SITING_LEFT,
                                               AJI_FILTER_SPLINE36);
        if (!c->pr_post_plan) {
            c->set_error("pre-resize post plan allocation failed");
            return AJI_ERR_CUDA;
        }
        memcpy(c->pr_post_key, qkey, sizeof(qkey));
    }

    const aji_csp ocsp = aji_make_csp(out->format, in->matrix, in->range);
    err = aji_run_post(c->pr_post_plan, c->buf[1], &ocsp,
                       out->plane[0], out->stride[0], out->plane[1],
                       out->stride[1], stream);
    if (err) {
        c->set_error("pre-resize post-kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
    return AJI_OK;
}

extern "C" AJI_EXPORT uint64_t aji_flush(aji_ctx *c, void *cu_stream)
{
    if (!c)
        return 0;
    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return 0;
    }
    cudaEvent_t &ev = c->tick_ev[c->tick_next % aji_ctx::TICK_RING];
    if (!ev && cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)
                   != cudaSuccess) {
        c->set_error("ticket event creation failed");
        return 0;
    }
    if (c->tick_next > aji_ctx::TICK_RING) {
        // slot reuse: retire the ring-old ticket this event tracked
        if (cudaEventSynchronize(ev) != cudaSuccess) {
            c->set_error("ticket event sync failed");
            return 0;
        }
        c->tick_completed = std::max(c->tick_completed,
                                     c->tick_next - aji_ctx::TICK_RING);
    }
    if (cudaEventRecord(ev, (cudaStream_t)cu_stream) != cudaSuccess) {
        c->set_error("ticket event record failed");
        return 0;
    }
    return c->tick_next++;
}

extern "C" AJI_EXPORT int aji_done(aji_ctx *c, uint64_t ticket)
{
    if (!c)
        return AJI_ERR;
    if (ticket <= c->tick_completed)
        return 1;
    if (ticket >= c->tick_next)
        return AJI_ERR;
    CtxGuard guard(c->cu_ctx);
    if (!guard.ok)
        return AJI_ERR_CUDA;
    cudaError_t e = cudaEventQuery(c->tick_ev[ticket % aji_ctx::TICK_RING]);
    if (e == cudaSuccess) {
        c->tick_completed = std::max(c->tick_completed, ticket);
        return 1;
    }
    if (e == cudaErrorNotReady)
        return 0;
    c->set_error("ticket query failed: %s", cudaGetErrorString(e));
    return AJI_ERR_CUDA;
}

extern "C" AJI_EXPORT int aji_wait(aji_ctx *c, uint64_t ticket)
{
    if (!c)
        return AJI_ERR;
    if (ticket <= c->tick_completed)
        return AJI_OK;
    if (ticket >= c->tick_next)
        return AJI_ERR;
    CtxGuard guard(c->cu_ctx);
    if (!guard.ok)
        return AJI_ERR_CUDA;
    if (cudaEventSynchronize(c->tick_ev[ticket % aji_ctx::TICK_RING])
            != cudaSuccess) {
        c->set_error("ticket wait failed");
        return AJI_ERR_CUDA;
    }
    c->tick_completed = std::max(c->tick_completed, ticket);
    return AJI_OK;
}

extern "C" AJI_EXPORT const char *aji_current_log(aji_ctx *c)
{
    return c ? c->current_log.c_str() : "";
}

extern "C" AJI_EXPORT int aji_scale_factor(aji_ctx *c)
{
    return c ? c->scale : 0;
}

extern "C" AJI_EXPORT int aji_poll(aji_ctx *c)
{
    if (!c || !c->build.done_flag.exchange(0))
        return 0;
    if (c->build.thread.joinable())
        c->build.thread.join();
#ifdef AJI_TRTRTX
    if (c->build_warm_only) {
        // ok or not: the caller-thread load comes next either way and
        // reports a real problem (the warm log has the worker's view)
        c->warmed.insert(c->build.epath);
        if (!c->build.ok.load())
            c->verbose("background engine warm-up failed: %s (see .warm.log)",
                       c->build.epath.c_str());
        return 1;
    }
#endif
    if (!c->build.ok.load()) {
        c->failed_builds.insert(c->build.epath);
        c->verbose("background engine build failed: %s (see .build.log)",
                   c->build.epath.c_str());
    }
#ifdef AJI_TRTRTX
    else
        c->built_engines.insert(c->build.epath);
#endif
    return 1;  // reconfigure either way; failures latch to passthrough
}

extern "C" AJI_EXPORT int aji_rife_factor(aji_ctx *c, int *num, int *den)
{
    if (!c || !c->rife.enabled)
        return 0;
    if (num)
        *num = c->rife.num;
    if (den)
        *den = c->rife.den;
    return 1;
}

extern "C" AJI_EXPORT int aji_rife_before_upscale(aji_ctx *c)
{
    return (c && c->rife.enabled && c->rife.before_upscale) ? 1 : 0;
}

extern "C" AJI_EXPORT int aji_infer_rife(aji_ctx *c, const aji_frame *a,
                                         const aji_frame *b, double t,
                                         const aji_frame *out, void *cu_stream)
{
    if (!c || !a || !b || !out)
        return AJI_ERR;
    auto &R = c->rife;
    if (!R.enabled) {
        c->set_error("aji_infer_rife without an active RIFE configuration");
        return AJI_ERR;
    }
    if (a->format != b->format || a->format != out->format ||
        (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010 &&
         !aji_format_is_444(a->format))) {
        c->set_error("rife frame formats must match (nv12/p010/planar 444)");
        return AJI_ERR_FORMAT;
    }
    if (a->width != R.w || a->height != R.h || b->width != R.w ||
        b->height != R.h || out->width != R.w || out->height != R.h) {
        c->set_error("rife frame dims do not match configured %dx%d",
                     R.w, R.h);
        return AJI_ERR_SHAPE;
    }

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }
    cudaStream_t stream = (cudaStream_t)cu_stream;
    const int fmt = a->format;
    const bool f444 = aji_format_is_444(fmt);
    const int bpp = fmt == AJI_FMT_NV12 ? 1 : 2;
    const size_t prow = (size_t)R.pw * bpp;
    const size_t py = prow * R.ph;
    const size_t plane = (size_t)R.pw * R.ph;          // fp16 elements

    // format-dependent staging + plans, built on first use
    if (R.format != fmt) {
        aji_plan_destroy(R.pre);
        aji_plan_destroy(R.post);
        R.pre = R.post = nullptr;
        if (!f444) {
            R.pre = aji_pre_plan_create(fmt, R.pw, R.ph, AJI_SITING_LEFT,
                                        AJI_FILTER_BILINEAR);
            R.post = aji_post_plan_create(fmt, R.pw, R.ph, AJI_SITING_LEFT,
                                          AJI_FILTER_BILINEAR);
        }
        cudaFree(R.pad_a);
        cudaFree(R.pad_b);
        cudaFree(R.pad_o);
        R.pad_a = R.pad_b = R.pad_o = nullptr;
        // 4:2:0: Y plane + interleaved CbCr; 4:4:4 16-bit: three planes
        // (no resampling plans needed - conversions are planless)
        const size_t pad_bytes = f444 ? 3 * py : py + py / 2;
        if ((!f444 && (!R.pre || !R.post)) ||
            cudaMalloc(&R.pad_a, pad_bytes) != cudaSuccess ||
            cudaMalloc(&R.pad_b, pad_bytes) != cudaSuccess ||
            cudaMalloc(&R.pad_o, pad_bytes) != cudaSuccess) {
            c->set_error("rife staging allocation failed");
            R.format = 0;
            return AJI_ERR_CUDA;
        }
        // pad borders are studio black like std.AddBorders; interiors get
        // overwritten per frame, so fill whole planes once
        const unsigned yblack = fmt == AJI_FMT_NV12 ? 16 : 16 * 256;
        const unsigned cblack = fmt == AJI_FMT_NV12 ? 128 : 128 * 256;
        for (void *p : {R.pad_a, R.pad_b}) {
            aji_fill_plane(fmt, p, prow, R.pw, R.ph, yblack, stream);
            if (f444) {
                aji_fill_plane(fmt, (char *)p + py, prow, R.pw, R.ph,
                               cblack, stream);
                aji_fill_plane(fmt, (char *)p + 2 * py, prow, R.pw, R.ph,
                               cblack, stream);
            } else {
                aji_fill_plane(fmt, (char *)p + py, prow, R.pw, R.ph / 2,
                               cblack, stream);
            }
        }
        R.format = fmt;
    }

    // scene detection on the unpadded luma (misc.SCDetect's metric)
    cudaMemsetAsync(R.scd, 0, sizeof(float), stream);
    int err = aji_scd_diff(fmt, a->plane[0], a->stride[0], b->plane[0],
                           b->stride[0], R.w, R.h, R.scd, stream);
    if (err) {
        c->set_error("scd kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
    float sum = 0.0f;
    cudaMemcpyAsync(&sum, R.scd, sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        c->set_error("scd sync failed");
        return AJI_ERR_CUDA;
    }
    // The reference runs SCDetect on the padded clip; the constant borders
    // contribute zero difference, so its mean divides by the padded area.
    if (sum / ((double)R.pw * R.ph) > R.scd_threshold)
        return AJI_SCENE;

    // stage both frames into the padded buffers (interior only)
    const size_t doff_y = (size_t)R.pad_t * prow + (size_t)R.pad_l * bpp;
    const size_t doff_uv = py + (size_t)(R.pad_t / 2) * prow +
                           (size_t)R.pad_l * bpp;
    const size_t doff_p = (size_t)R.pad_t * prow + (size_t)R.pad_l * bpp;
    const aji_frame *src[2] = {a, b};
    void *pad[2] = {R.pad_a, R.pad_b};
    for (int i = 0; i < 2; i++) {
        bool ok;
        if (f444) {
            ok = true;
            for (int pl = 0; pl < 3 && ok; pl++)
                ok = copy_plane((char *)pad[i] + pl * py + doff_p, prow,
                                src[i]->plane[pl], src[i]->stride[pl],
                                (size_t)R.w * bpp, R.h, stream);
        } else {
            ok = copy_plane((char *)pad[i] + doff_y, prow, src[i]->plane[0],
                            src[i]->stride[0], (size_t)R.w * bpp, R.h,
                            stream) &&
                 copy_plane((char *)pad[i] + doff_uv, prow,
                            src[i]->plane[1], src[i]->stride[1],
                            (size_t)R.w * bpp, R.h / 2, stream);
        }
        if (!ok) {
            c->set_error("rife staging copy failed");
            return AJI_ERR_CUDA;
        }
    }

    // padded YUV -> RGB into input channels 0-2 (frame a) and 3-5 (b);
    // 4:2:0 upsamples chroma bilinearly like rife_cuda.py, 4:4:4 needs
    // no resampling at all. Matrix hardcoded BT.709 like the reference.
    const aji_csp csp = aji_make_csp(fmt, AJI_MATRIX_BT709, a->range);
    char *tin = (char *)R.in_tensor;
    for (int i = 0; i < 2; i++) {
        if (f444) {
            err = aji_run_pre444(R.pw, R.ph,
                                 pad[i], prow,
                                 (char *)pad[i] + py, prow,
                                 (char *)pad[i] + 2 * py, prow,
                                 &csp, tin + plane * 2 * (i * 3), stream);
        } else {
            err = aji_run_pre(R.pre, pad[i], prow, (char *)pad[i] + py,
                              prow, &csp, tin + plane * 2 * (i * 3),
                              stream);
        }
        if (err) {
            c->set_error("rife pre kernel failed: %s",
                         cudaGetErrorString((cudaError_t)err));
            return AJI_ERR_CUDA;
        }
    }
    // channel 6: the timestep plane
    err = aji_fill_f16(tin + plane * 2 * 6, plane, (float)t, stream);
    if (err) {
        c->set_error("timestep fill failed");
        return AJI_ERR_CUDA;
    }

    ModelEngine &me = R.engine;
    if (!me.exec->setTensorAddress(me.in_name, R.in_tensor) ||
        !me.exec->setTensorAddress(me.out_name, R.out_tensor)) {
        c->set_error("rife setTensorAddress failed");
        return AJI_ERR_ENGINE;
    }
    if (!me.exec->enqueueV3(stream)) {
        c->set_error("rife enqueueV3 failed");
        return AJI_ERR_ENGINE;
    }

    // RGB -> padded YUV, then crop the window out
    if (f444) {
        err = aji_run_post444(fmt, R.pw, R.ph, R.out_tensor, &csp,
                              R.pad_o, prow,
                              (char *)R.pad_o + py, prow,
                              (char *)R.pad_o + 2 * py, prow, stream);
    } else {
        err = aji_run_post(R.post, R.out_tensor, &csp, R.pad_o, prow,
                           (char *)R.pad_o + py, prow, stream);
    }
    if (err) {
        c->set_error("rife post kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
    bool ok;
    if (f444) {
        ok = true;
        for (int pl = 0; pl < 3 && ok; pl++)
            ok = copy_plane(out->plane[pl], out->stride[pl],
                            (char *)R.pad_o + pl * py + doff_p, prow,
                            (size_t)R.w * bpp, R.h, stream);
    } else {
        ok = copy_plane(out->plane[0], out->stride[0],
                        (char *)R.pad_o + doff_y, prow, (size_t)R.w * bpp,
                        R.h, stream) &&
             copy_plane(out->plane[1], out->stride[1],
                        (char *)R.pad_o + doff_uv, prow,
                        (size_t)R.w * bpp, R.h / 2, stream);
    }
    if (!ok) {
        c->set_error("rife crop copy failed");
        return AJI_ERR_CUDA;
    }
    return AJI_OK;
}

extern "C" AJI_EXPORT const char *aji_last_error(aji_ctx *c)
{
    return c ? c->errbuf : "no context";
}

extern "C" AJI_EXPORT void aji_destroy(aji_ctx **pc)
{
    if (!pc || !*pc)
        return;
    aji_ctx *c = *pc;
    // Stop any in-flight engine build (player quit shouldn't wait minutes
    // for trtexec); the worker's failure path removes the partial engine.
    c->build.stop();
    {
        CtxGuard guard(c->cu_ctx);
        cudaDeviceSynchronize();    // in-flight ticketed work
        for (cudaEvent_t &ev : c->tick_ev) {
            if (ev)
                cudaEventDestroy(ev);
        }
        for (auto &b : c->buf)
            cudaFree(b);
        for (Step &st : c->steps)
            aji_plan_destroy(st.plan);
        c->steps.clear();
        aji_plan_destroy(c->pre_plan);
        aji_plan_destroy(c->post_plan);
        aji_plan_destroy(c->pre_resize_plan);
        aji_plan_destroy(c->pr_pre_plan);
        aji_plan_destroy(c->pr_post_plan);
        rife_teardown(c);
        temporal_teardown(c);
        if (c->graph_exec)
            cudaGraphExecDestroy(c->graph_exec);
        cudaFree(c->stage_in);
        cudaFree(c->stage_out);
        c->engines.clear();
        c->direct.exec.reset();
        c->direct.engine.reset();
#ifdef AJI_TRTRTX
        c->direct.rt.save();  // needs the context
#endif
        c->runtime.reset();
    }
    if (c->owns_primary)
        cuDevicePrimaryCtxRelease(c->cu_dev);
    delete c;
    *pc = nullptr;
}
