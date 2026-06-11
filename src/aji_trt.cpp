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
    std::string conf_path, model_dir, trtexec, trtexec_env;
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
                  const std::string &settings, const std::string &engine_path)
{
    const std::string onnx_path =
        (fs::path(c->model_dir) / (onnx_name + ".onnx")).string();
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
                 int in_w, int in_h)
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
    if (!me->exec->setInputShape(me->in_name, nvinfer1::Dims4{1, 3, in_h, in_w})) {
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
        c->slot = params->slot > 0 ? params->slot : 1;
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
        if (!load_engine(c.get(), params->engine_path, &c->direct, 64, 64))
            return nullptr;
        c->scale = c->direct.out_w / 64;
        const size_t bytes =
            (size_t)3 * c->max_w * c->max_h * 2 * c->scale * c->scale;
        if (!ensure_buffers(c.get(), bytes))
            return nullptr;
    }
    return c.release();
}

extern "C" AJI_EXPORT int aji_set_slot(aji_ctx *c, int slot)
{
    if (!c || slot <= 0)
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
    c->log_info.clear();
    c->log_steps.clear();
    c->active = false;
    c->in_w = w; c->in_h = h;
    c->out_w = w; c->out_h = h;

    auto sit = c->conf.slots.find(c->slot);
    std::string profile = sit != c->conf.slots.end()
                              ? sit->second.profile_name : "(missing slot)";
    if (c->slot < 10)
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

    finalize_log(c);

    if (!any_model && c->steps.empty()) {
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;  // chain matched but does nothing -> passthrough
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
    const aji_csp csp = make_csp(in);

    const int pkey[4] = {in->format, in->width, in->height, in->siting};
    if (!c->pre_plan || memcmp(pkey, c->pre_key, sizeof(pkey)) != 0) {
        aji_plan_destroy(c->pre_plan);
        c->pre_plan = aji_pre_plan_create(in->format, in->width, in->height,
                                          in->siting);
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
                                            AJI_SITING_LEFT);
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
