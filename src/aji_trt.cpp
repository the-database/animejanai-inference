/*
 * aji_trt — TensorRT backend for the aji C ABI (see include/aji.h).
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

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "aji.h"
#include "aji_conf.h"
#include "kernels.h"

namespace fs = std::filesystem;

namespace {

const char *DEFAULT_TRT_ENGINE_SETTINGS =
    "--stronglyTyped --optShapes=input:%video_resolution% "
    "--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw "
    "--builderOptimizationLevel=5 "
    "--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference";

class Logger : public nvinfer1::ILogger {
public:
    aji_log_fn fn = nullptr;
    void *opaque = nullptr;
    void log(Severity sev, const char *msg) noexcept override {
        if (fn)
            fn(opaque, (int)sev, msg);
        else if (sev <= Severity::kWARNING)
            fprintf(stderr, "[aji-trt] %s\n", msg);
    }
};

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

// zlib-compatible CRC-32 (matches Python zlib.crc32 used for engine names).
uint32_t crc32_z(const std::string &data)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (unsigned char ch : data)
        c = table[(c ^ ch) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::string sanitize_token(std::string s)
{
    for (auto &ch : s) {
        if (ch == ' ')
            ch = '-';
    }
    std::string out;
    for (char ch : s) {
        if (isalnum((unsigned char)ch) || ch == '.' || ch == '_' || ch == '-')
            out += ch;
    }
    return out.empty() ? "device0" : out;
}

struct ModelEngine {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> exec;
    const char *in_name = nullptr;
    const char *out_name = nullptr;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
};

struct Step {
    enum Kind { RESIZE, MODEL } kind;
    int out_w, out_h;
    int engine_idx;             // MODEL
    aji_plan *plan = nullptr;   // RESIZE
};

} // namespace

struct aji_ctx {
    Logger logger;
    CUcontext cu_ctx = nullptr;
    bool owns_primary = false;

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

    // resampling plans: per-step resize plans live in steps[]; pre/post are
    // (re)built lazily per frame key since siting arrives with frames
    aji_plan *pre_plan = nullptr, *post_plan = nullptr;
    int pre_key[4] = {0}, post_key[4] = {0};

    // RIFE (phase 1.5): interpolation between already-upscaled frames,
    // replicating animejanai_core's pad/crop + rife_cuda.py's bilinear 709
    // conversions + the vsmlrt v1 11-channel model input. Geometry state
    // is set at configure; format-dependent staging/plans build lazily on
    // the first aji_infer_rife (frames carry the format).
    struct {
        bool enabled = false;
        int num = 1, den = 1;
        double scd_threshold = 0.150;
        ModelEngine engine;
        int w = 0, h = 0;            // unpadded = chain output dims
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
    int graph_key[8] = {0};
    void *stage_in = nullptr, *stage_out = nullptr;
    size_t stage_in_bytes = 0, stage_out_bytes = 0;

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

std::string trt_version_token()
{
    int32_t v = getInferLibVersion();
    int major, minor, patch;
    if (v < 10000) {
        major = v / 1000; minor = (v / 100) % 10; patch = v % 100;
    } else {
        major = v / 10000; minor = (v / 100) % 100; patch = v % 100;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d", major, minor, patch);
    return buf;
}

std::string gpu_token()
{
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return "unknown";
    std::string tok = sanitize_token(prop.name);
    return tok + "-sm" + std::to_string(prop.major);
}

std::string engine_suffix()
{
    return ".trt-" + trt_version_token() + ".gpu-" + gpu_token() + ".engine";
}

std::string engine_path_for(const std::string &model_dir,
                            const std::string &onnx_name,
                            const std::string &settings)
{
    return (fs::path(model_dir) /
            (onnx_name + "." + std::to_string(crc32_z(settings)) + engine_suffix()))
        .string();
}

void clean_stale_engines(aji_ctx *c)
{
    const std::string suffix = engine_suffix();
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(c->model_dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.size() < 7 || name.substr(name.size() - 7) != ".engine")
            continue;
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            continue;
        std::error_code rec;
        if (fs::remove(e.path(), rec)) {
            c->log_steps.push_back(
                "Removed stale TensorRT engine (different GPU or TensorRT version): " + name);
            c->verbose("removed stale engine %s", name.c_str());
        }
    }
}

bool build_engine(aji_ctx *c, const std::string &onnx_name,
                  const std::string &settings, const std::string &engine_path,
                  const std::string *dir = nullptr)
{
    const std::string onnx_path =
        (fs::path(dir ? *dir : c->model_dir) / (onnx_name + ".onnx")).string();
    if (!fs::exists(onnx_path)) {
        c->set_error("model not found: %s", onnx_path.c_str());
        return false;
    }

    std::string cmd;
#ifdef _WIN32
    cmd = "\"\"" + c->trtexec + "\" --onnx=\"" + onnx_path + "\" --saveEngine=\"" +
          engine_path + "\" " + settings + "\"";
#else
    if (!c->trtexec_env.empty())
        cmd = c->trtexec_env + " ";
    cmd += "\"" + c->trtexec + "\" --onnx=\"" + onnx_path + "\" --saveEngine=\"" +
           engine_path + "\" " + settings + " >/dev/null 2>&1";
#endif
    c->verbose("building engine: %s", cmd.c_str());
    c->log_steps.push_back("Building TensorRT engine for " + onnx_name +
                           " (first play at this resolution)");
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(engine_path)) {
        std::error_code ec;
        fs::remove(engine_path, ec);  // don't leave a half-written engine
        c->set_error("trtexec failed (exit %d) building engine for %s", rc,
                     onnx_name.c_str());
        return false;
    }
    return true;
}

bool load_engine(aji_ctx *c, const std::string &engine_path, ModelEngine *me,
                 int in_w, int in_h, int in_ch = 3)
{
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) {
        c->set_error("cannot open engine: %s", engine_path.c_str());
        return false;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    me->engine.reset(c->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!me->engine) {
        c->set_error("engine deserialization failed: %s", engine_path.c_str());
        return false;
    }
    for (int i = 0; i < me->engine->getNbIOTensors(); i++) {
        const char *name = me->engine->getIOTensorName(i);
        if (me->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            me->in_name = name;
        else
            me->out_name = name;
    }
    if (!me->in_name || !me->out_name) {
        c->set_error("engine I/O tensors not found");
        return false;
    }
    me->exec.reset(me->engine->createExecutionContext());
    if (!me->exec) {
        c->set_error("createExecutionContext failed");
        return false;
    }
    if (!me->exec->setInputShape(me->in_name,
                                 nvinfer1::Dims4{1, in_ch, in_h, in_w})) {
        c->set_error("engine rejects input %dx%d (%s)", in_w, in_h,
                     engine_path.c_str());
        return false;
    }
    nvinfer1::Dims od = me->exec->getTensorShape(me->out_name);
    if (od.nbDims != 4) {
        c->set_error("unexpected output rank from %s", engine_path.c_str());
        return false;
    }
    me->in_w = in_w;
    me->in_h = in_h;
    me->out_w = (int)od.d[3];
    me->out_h = (int)od.d[2];
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
                double fps)
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

    const std::string model =
        rife_model_name(chain->rife_model, chain->rife_ensemble);
    char dims[64];
    snprintf(dims, sizeof(dims), "1x11x%dx%d", R.ph, R.pw);
    // vsmlrt's RIFE TRT backend: fp16 build with fp16 I/O, no cuDNN/cuBLAS
    const std::string settings = std::string("--fp16 --optShapes=input:") +
        dims + " --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"
        " --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference";
    const std::string epath =
        engine_path_for(c->rife_model_dir, model, settings);
    if (!fs::exists(epath)) {
        if (!build_engine(c, model, settings, epath, &c->rife_model_dir))
            return false;
    }
    if (!load_engine(c, epath, &R.engine, R.pw, R.ph, 11))
        return false;

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
    }

    c->runtime.reset(nvinfer1::createInferRuntime(c->logger));
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

    c->engines.clear();
    for (Step &st : c->steps)
        aji_plan_destroy(st.plan);
    c->steps.clear();
    rife_teardown(c);
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
    size_t max_bytes = (size_t)3 * cw * ch * 2;
    bool any_model = false;

    for (const auto &m : chain->models) {
        double factor = m.resize_factor_before_upscale;
        if (m.resize_height_before_upscale != 0)
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
        if (m.resize_height_before_upscale != 0 &&
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
        } else if (ch > 1080) {
            int nw, nh;
            fit_box(cw, ch, 1920, 1080, &nw, &nh);
            if (!push_resize_step(c, &cw, &ch, nw, nh))
                return AJI_ERR_CUDA;
            snprintf(buf, sizeof(buf),
                     "Applied Resize to Video Larger than 1080p;    "
                     "New Video Resolution: %dx%d", cw, ch);
            c->log_steps.push_back(buf);
        }
        max_bytes = std::max(max_bytes, (size_t)3 * cw * ch * 2);

        if (m.name.empty())
            continue;

        char dims[48];
        snprintf(dims, sizeof(dims), "1x3x%dx%d", ch, cw);
        std::string settings = settings_tpl;
        size_t pos;
        while ((pos = settings.find("%video_resolution%")) != std::string::npos)
            settings.replace(pos, strlen("%video_resolution%"), dims);

        const std::string epath = engine_path_for(c->model_dir, m.name, settings);
        if (!fs::exists(epath)) {
            if (!build_engine(c, m.name, settings, epath)) {
                finalize_log(c);
                return AJI_ERR_ENGINE;
            }
        }
        ModelEngine me;
        if (!load_engine(c, epath, &me, cw, ch)) {
            finalize_log(c);
            return AJI_ERR_ENGINE;
        }
        cw = me.out_w; ch = me.out_h;
        c->engines.push_back(std::move(me));
        c->steps.push_back({Step::MODEL, cw, ch, (int)c->engines.size() - 1});
        any_model = true;
        snprintf(buf, sizeof(buf),
                 "Applied Model: %s;    New Video Resolution: %dx%d",
                 m.name.c_str(), cw, ch);
        c->log_steps.push_back(buf);
        max_bytes = std::max(max_bytes, (size_t)3 * cw * ch * 2);
    }

    if (chain->rife) {
        if (!c->rife_model_dir.empty()) {
            if (!setup_rife(c, chain, cw, ch, fps)) {
                finalize_log(c);
                return AJI_ERR_ENGINE;
            }
        } else {
            c->log_steps.push_back(
                "RIFE requested by the chain but no rife model dir is "
                "configured; interpolation disabled");
        }
    }

    finalize_log(c);

    if (!any_model && c->steps.empty()) {
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        // chain selected no scaling work; rife (if enabled above) still
        // interpolates between the passthrough frames
        return 0;
    }

    if (!ensure_buffers(c, max_bytes))
        return AJI_ERR_CUDA;

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
    const int bpp = in->format == AJI_FMT_P010 ? 2 : 1;
    const size_t in_row = (size_t)in->width * bpp;
    const size_t out_row = (size_t)out->width * bpp;
    const size_t in_y = in_row * in->height;
    const size_t out_y = out_row * out->height;

    if (!ensure_stage(&c->stage_in, &c->stage_in_bytes, in_y * 3 / 2) ||
        !ensure_stage(&c->stage_out, &c->stage_out_bytes, out_y * 3 / 2)) {
        c->graph_ok = false;
        return false;
    }

    // staging descriptors: same colorimetry tags, packed strides
    aji_frame sin = *in, sout = *out;
    sin.plane[0] = c->stage_in;
    sin.plane[1] = (char *)c->stage_in + in_y;
    sin.stride[0] = sin.stride[1] = (ptrdiff_t)in_row;
    sout.plane[0] = c->stage_out;
    sout.plane[1] = (char *)c->stage_out + out_y;
    sout.stride[0] = sout.stride[1] = (ptrdiff_t)out_row;

    if (!copy_plane(sin.plane[0], in_row, in->plane[0], in->stride[0],
                    in_row, in->height, stream) ||
        !copy_plane(sin.plane[1], in_row, in->plane[1], in->stride[1],
                    in_row, in->height / 2, stream)) {
        c->graph_ok = false;
        return false;
    }

    const int key[8] = {in->format, in->width, in->height, in->siting,
                        in->matrix, in->range, out->width, out->height};
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
                    out_row, out->height / 2, stream)) {
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
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010) {
        c->set_error("unsupported input format %d", in->format);
        return AJI_ERR_FORMAT;
    }
    if (out->format != in->format) {
        c->set_error("output format must match input");
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

// The chain body: pre -> (model/resize steps) -> post, launched on `stream`
// against whatever plane pointers the frame descriptors carry (real frames
// on the plain path, stable staging buffers on the graph path). Everything
// in here must be capture-safe once warm: allocations only happen on plan
// key changes, which a warmup run performs before any capture.
static int run_chain(aji_ctx *c, const aji_frame *in, const aji_frame *out,
                     cudaStream_t stream)
{
    const aji_csp csp = make_csp(in);

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

    int cur = 0;
    int err = aji_run_pre(c->pre_plan, in->plane[0], in->stride[0],
                          in->plane[1], in->stride[1], &csp, c->buf[cur],
                          stream);
    if (err) {
        c->set_error("pre-kernel failed: %s", cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }

    int cw = in->width, ch = in->height;

    if (!c->conf_mode) {
        ModelEngine &me = c->direct;
        if (!me.exec->setTensorAddress(me.in_name, c->buf[cur]) ||
            !me.exec->setTensorAddress(me.out_name, c->buf[cur ^ 1])) {
            c->set_error("setTensorAddress failed");
            return AJI_ERR_ENGINE;
        }
        if (!me.exec->enqueueV3(stream)) {
            c->set_error("enqueueV3 failed");
            return AJI_ERR_ENGINE;
        }
        cur ^= 1;
        cw = c->out_w; ch = c->out_h;
    } else {
        for (const Step &st : c->steps) {
            if (st.kind == Step::RESIZE) {
                err = aji_run_resize(st.plan, c->buf[cur], c->buf[cur ^ 1],
                                     stream);
                if (err) {
                    c->set_error("resize kernel failed: %s",
                                 cudaGetErrorString((cudaError_t)err));
                    return AJI_ERR_CUDA;
                }
            } else {
                ModelEngine &me = c->engines[st.engine_idx];
                if (!me.exec->setTensorAddress(me.in_name, c->buf[cur]) ||
                    !me.exec->setTensorAddress(me.out_name, c->buf[cur ^ 1])) {
                    c->set_error("setTensorAddress failed");
                    return AJI_ERR_ENGINE;
                }
                if (!me.exec->enqueueV3(stream)) {
                    c->set_error("enqueueV3 failed");
                    return AJI_ERR_ENGINE;
                }
            }
            cur ^= 1;
            cw = st.out_w; ch = st.out_h;
        }
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

    err = aji_run_post(c->post_plan, c->buf[cur], &csp,
                       out->plane[0], out->stride[0], out->plane[1],
                       out->stride[1], stream);
    if (err) {
        c->set_error("post-kernel failed: %s", cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
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
        (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010)) {
        c->set_error("rife frame formats must match (nv12/p010)");
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
    const int bpp = fmt == AJI_FMT_P010 ? 2 : 1;
    const size_t prow = (size_t)R.pw * bpp;
    const size_t py = prow * R.ph;
    const size_t plane = (size_t)R.pw * R.ph;          // fp16 elements

    // format-dependent staging + plans, built on first use
    if (R.format != fmt) {
        aji_plan_destroy(R.pre);
        aji_plan_destroy(R.post);
        R.pre = aji_pre_plan_create(fmt, R.pw, R.ph, AJI_SITING_LEFT,
                                    AJI_FILTER_BILINEAR);
        R.post = aji_post_plan_create(fmt, R.pw, R.ph, AJI_SITING_LEFT,
                                      AJI_FILTER_BILINEAR);
        cudaFree(R.pad_a);
        cudaFree(R.pad_b);
        cudaFree(R.pad_o);
        R.pad_a = R.pad_b = R.pad_o = nullptr;
        const size_t pad_bytes = py + py / 2;
        if (!R.pre || !R.post ||
            cudaMalloc(&R.pad_a, pad_bytes) != cudaSuccess ||
            cudaMalloc(&R.pad_b, pad_bytes) != cudaSuccess ||
            cudaMalloc(&R.pad_o, pad_bytes) != cudaSuccess) {
            c->set_error("rife staging allocation failed");
            R.format = 0;
            return AJI_ERR_CUDA;
        }
        // pad borders are studio black like std.AddBorders; interiors get
        // overwritten per frame, so fill whole planes once
        const unsigned yblack = fmt == AJI_FMT_P010 ? 16 * 256 : 16;
        const unsigned cblack = fmt == AJI_FMT_P010 ? 128 * 256 : 128;
        for (void *p : {R.pad_a, R.pad_b}) {
            aji_fill_plane(fmt, p, prow, R.pw, R.ph, yblack, stream);
            aji_fill_plane(fmt, (char *)p + py, prow, R.pw, R.ph / 2, cblack,
                           stream);
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
    const aji_frame *src[2] = {a, b};
    void *pad[2] = {R.pad_a, R.pad_b};
    for (int i = 0; i < 2; i++) {
        if (!copy_plane((char *)pad[i] + doff_y, prow, src[i]->plane[0],
                        src[i]->stride[0], (size_t)R.w * bpp, R.h, stream) ||
            !copy_plane((char *)pad[i] + doff_uv, prow, src[i]->plane[1],
                        src[i]->stride[1], (size_t)R.w * bpp, R.h / 2,
                        stream)) {
            c->set_error("rife staging copy failed");
            return AJI_ERR_CUDA;
        }
    }

    // padded YUV -> RGB into input channels 0-2 (frame a) and 3-5 (b),
    // bilinear chroma, hardcoded BT.709 like rife_cuda.py
    const aji_csp csp = aji_make_csp(fmt, AJI_MATRIX_BT709, a->range);
    char *tin = (char *)R.in_tensor;
    for (int i = 0; i < 2; i++) {
        err = aji_run_pre(R.pre, pad[i], prow, (char *)pad[i] + py, prow,
                          &csp, tin + plane * 2 * (i * 3), stream);
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

    // RGB -> padded YUV (bilinear, 709), then crop the window out
    err = aji_run_post(R.post, R.out_tensor, &csp, R.pad_o, prow,
                       (char *)R.pad_o + py, prow, stream);
    if (err) {
        c->set_error("rife post kernel failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }
    if (!copy_plane(out->plane[0], out->stride[0],
                    (char *)R.pad_o + doff_y, prow, (size_t)R.w * bpp, R.h,
                    stream) ||
        !copy_plane(out->plane[1], out->stride[1],
                    (char *)R.pad_o + doff_uv, prow, (size_t)R.w * bpp,
                    R.h / 2, stream)) {
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
    {
        CtxGuard guard(c->cu_ctx);
        for (auto &b : c->buf)
            cudaFree(b);
        for (Step &st : c->steps)
            aji_plan_destroy(st.plan);
        c->steps.clear();
        aji_plan_destroy(c->pre_plan);
        aji_plan_destroy(c->post_plan);
        rife_teardown(c);
        if (c->graph_exec)
            cudaGraphExecDestroy(c->graph_exec);
        cudaFree(c->stage_in);
        cudaFree(c->stage_out);
        c->engines.clear();
        c->direct.exec.reset();
        c->direct.engine.reset();
        c->runtime.reset();
    }
    if (c->owns_primary) {
        CUdevice dev;
        if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS)
            cuDevicePrimaryCtxRelease(dev);
    }
    delete c;
    *pc = nullptr;
}
