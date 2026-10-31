/*
 * aji_img_trt — TensorRT implementation of the image ABI (aji_image.h),
 * compiled into aji_trt alongside the video front-end.
 *
 * One static engine per (model, tile, GPU, TensorRT version), cached via
 * the core's engine naming; missing engines build with trtexec, inside
 * aji_img_load unless AJI_IMG_LOAD_ASYNC_BUILD. Tiles travel through pinned
 * host staging in the engine's own I/O dtype (fp32/fp16/bf16) on a
 * per-model stream; core/tiling does the cover/pad/blend on the host.
 *
 * Also built into aji_trtrtx (AJI_TRTRTX) as the AJI_IMG_TENSORRT_RTX
 * backend: engines then build in-process (core/trt_engine).
 */

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "aji_image.h"
#include "onnx_shape.h"
#include "core/fileio.h"
#include "core/half.h"
#include "core/tiling.h"
#include "core/trt_engine.h"

namespace fs = std::filesystem;
using namespace ajicore;

#ifdef AJI_TRTRTX
#define AJI_IMG_THIS_BACKEND AJI_IMG_TENSORRT_RTX
#else
#define AJI_IMG_THIS_BACKEND AJI_IMG_TENSORRT
#endif

namespace {

// Static tile engines: builder level 3 keeps first-use builds short; the
// shape flags are added per model when its ONNX input is dynamic.
const char *IMG_ENGINE_SETTINGS = "--builderOptimizationLevel=3 --skipInference";

uint64_t pci_id(const cudaDeviceProp &p)
{
    // nonzero by construction, so 0 can mean "first device"
    return (1ull << 63) | ((uint64_t)(uint32_t)p.pciDomainID << 32) |
           ((uint64_t)(uint32_t)p.pciBusID << 8) | (uint64_t)(p.pciDeviceID & 0xff);
}

size_t dtype_size(nvinfer1::DataType t)
{
    switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16: return 2;
    default: return 0;
    }
}

void pack(const float *src, void *dst, size_t n, nvinfer1::DataType t)
{
    if (t == nvinfer1::DataType::kFLOAT) {
        memcpy(dst, src, n * 4);
    } else if (t == nvinfer1::DataType::kHALF) {
        uint16_t *d = (uint16_t *)dst;
        for (size_t i = 0; i < n; i++)
            d[i] = f32_to_f16(src[i]);
    } else {
        uint16_t *d = (uint16_t *)dst;
        for (size_t i = 0; i < n; i++)
            d[i] = f32_to_bf16(src[i]);
    }
}

void unpack(const void *src, float *dst, size_t n, nvinfer1::DataType t)
{
    if (t == nvinfer1::DataType::kFLOAT) {
        memcpy(dst, src, n * 4);
    } else if (t == nvinfer1::DataType::kHALF) {
        const uint16_t *s = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++)
            dst[i] = f16_to_f32(s[i]);
    } else {
        const uint16_t *s = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++)
            dst[i] = bf16_to_f32(s[i]);
    }
}

} // namespace

struct aji_img_model;

struct aji_img_ctx {
    TrtLogger logger;
    int device = 0;  // CUDA ordinal
    CUdevice cu_dev = 0;
    CUcontext cu_ctx = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::string cache_dir, trtexec, trtexec_env;

    // Guards everything below. Engine builds run one at a time: `building`
    // is the model whose trtexec is running, the rest wait in build_queue.
    std::mutex mu;
    AsyncBuild build;
    aji_img_model *building = nullptr;
    std::deque<aji_img_model *> build_queue;
    std::set<aji_img_model *> models;
    std::set<std::string> cleaned_dirs;
    std::string err;

    void set_error(const std::string &e) {
        err = e;
        if (logger.fn)
            logger.fn(logger.opaque, /*kERROR*/ 1, e.c_str());
    }
    void info(const std::string &msg) {
        if (logger.fn)
            logger.fn(logger.opaque, /*kINFO*/ 3, msg.c_str());
    }
};

struct aji_img_model : TileModel {
    aji_img_ctx *c = nullptr;
    std::string onnx_path, name, settings, engine_path, timing_cache;
    int state = AJI_IMG_STATE_BUILDING;
    int overlap = DEFAULT_OVERLAP;

    std::mutex run_mu;  // serializes aji_img_upscale on this model
    ModelEngine me;
    nvinfer1::DataType in_dt = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType out_dt = nvinfer1::DataType::kFLOAT;
    cudaStream_t stream = nullptr;
    void *d_in = nullptr, *d_out = nullptr;  // device tensors
    void *h_in = nullptr, *h_out = nullptr;  // pinned staging
    size_t in_elems = 0, out_elems = 0;

    int run_tile(const float *in, float *out, std::string *err) override
    {
        pack(in, h_in, in_elems, in_dt);
        const size_t in_bytes = in_elems * dtype_size(in_dt);
        const size_t out_bytes = out_elems * dtype_size(out_dt);
        cudaError_t e = cudaMemcpyAsync(d_in, h_in, in_bytes,
                                        cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess && !me.exec->enqueueV3(stream)) {
            *err = AJI_TRT_PRODUCT " enqueue failed for " + name;
            return AJI_IMG_ERR;
        }
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(h_out, d_out, out_bytes,
                                cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess)
            e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            *err = std::string("CUDA error running ") + name + ": " +
                   cudaGetErrorString(e);
            return AJI_IMG_ERR;
        }
        unpack(h_out, out, out_elems, out_dt);
        return AJI_IMG_OK;
    }

    void free_buffers()
    {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
        if (stream)
            cudaStreamDestroy(stream);
        d_in = d_out = h_in = h_out = nullptr;
        stream = nullptr;
        me.exec.reset();
        me.engine.reset();
#ifdef AJI_TRTRTX
        me.rt.save();  // needs the context, which the callers hold
#endif
    }
};

namespace {

// Deserialize the model's engine and set up tile buffers. Context pushed.
bool finalize(aji_img_model *m, std::string *err)
{
    aji_img_ctx *c = m->c;
    if (!load_engine(c->runtime.get(), m->engine_path, &m->me, m->tile_w,
                     m->tile_h, m->in_ch, 0, err))
        return false;
    nvinfer1::Dims od = m->me.exec->getTensorShape(m->me.out_name);
    m->out_ch = (int)od.d[1];
    if (m->me.out_w % m->tile_w || m->me.out_h % m->tile_h ||
        m->me.out_w / m->tile_w != m->me.out_h / m->tile_h ||
        (m->out_ch != 1 && m->out_ch != 3)) {
        *err = m->name + ": output " + std::to_string(m->me.out_w) + "x" +
               std::to_string(m->me.out_h) + "x" + std::to_string(m->out_ch) +
               " is not an integer-scale 1/3-channel image of the " +
               std::to_string(m->tile_w) + "x" + std::to_string(m->tile_h) +
               " tile";
        return false;
    }
    m->scale = m->me.out_w / m->tile_w;
    m->in_dt = m->me.engine->getTensorDataType(m->me.in_name);
    m->out_dt = m->me.engine->getTensorDataType(m->me.out_name);
    if (!dtype_size(m->in_dt) || !dtype_size(m->out_dt)) {
        *err = m->name + ": engine I/O must be fp32, fp16 or bf16";
        return false;
    }
    m->in_elems = (size_t)m->in_ch * m->tile_w * m->tile_h;
    m->out_elems = (size_t)m->out_ch * m->me.out_w * m->me.out_h;
    const size_t ib = m->in_elems * dtype_size(m->in_dt);
    const size_t ob = m->out_elems * dtype_size(m->out_dt);
    if (cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&m->d_in, ib) != cudaSuccess ||
        cudaMalloc(&m->d_out, ob) != cudaSuccess ||
        cudaMallocHost(&m->h_in, ib) != cudaSuccess ||
        cudaMallocHost(&m->h_out, ob) != cudaSuccess) {
        *err = m->name + ": out of memory for tile buffers";
        m->free_buffers();
        return false;
    }
    if (!m->me.exec->setTensorAddress(m->me.in_name, m->d_in) ||
        !m->me.exec->setTensorAddress(m->me.out_name, m->d_out)) {
        *err = m->name + ": setTensorAddress failed";
        m->free_buffers();
        return false;
    }
    return true;
}

void fail(aji_img_model *m, const std::string &e)
{
    m->state = AJI_IMG_STATE_FAILED;
    m->c->set_error(e);
}

// Load a freshly built (or cached) engine; on failure the model fails.
void become_ready(aji_img_model *m)
{
    std::string e;
    CtxGuard guard(m->c->cu_ctx);
    if (finalize(m, &e))
        m->state = AJI_IMG_STATE_READY;
    else
        fail(m, e);
}

// Advance background builds. Caller holds c->mu.
void pump(aji_img_ctx *c)
{
    if (c->building && c->build.done_flag.exchange(0)) {
        if (c->build.thread.joinable())
            c->build.thread.join();
        aji_img_model *m = c->building;
        c->building = nullptr;
        if (c->build.ok.load())
            become_ready(m);
        else
            fail(m, AJI_TRT_PRODUCT " engine build failed for " + m->name + " (see " +
                        m->engine_path + ".build.log)");
    }
    while (!c->building && !c->build_queue.empty()) {
        aji_img_model *m = c->build_queue.front();
        c->build_queue.pop_front();
        BuildSpec spec;
        std::string e;
        if (!make_build_spec_paths(c->trtexec, c->trtexec_env, m->onnx_path,
                                   m->timing_cache, m->name, m->settings,
                                   m->engine_path, c->device, &spec, &e)) {
            fail(m, e);
            continue;
        }
        c->info("building " AJI_TRT_PRODUCT " engine for " + m->name + " (" + spec.res +
                " tiles): " + spec.cmdline);
        c->building = m;
        c->build.start(spec);
    }
}

std::string stem_of(const std::string &path)
{
    return fs::u8path(path).stem().u8string();
}

void copy_err(const std::string &e, char *err, size_t n)
{
    if (err && n) {
        snprintf(err, n, "%s", e.c_str());
    }
}

} // namespace

extern "C" AJI_IMG_EXPORT int aji_img_enum_devices(int backend,
                                                   aji_img_device *out, int max)
{
    if (backend != AJI_IMG_THIS_BACKEND)
        return -1;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return -1;
    for (int i = 0; i < n && out && i < max; i++) {
        cudaDeviceProp p = {};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess)
            return -1;
        aji_img_device &d = out[i];
        memset(&d, 0, sizeof(d));
        d.id = pci_id(p);
        snprintf(d.name, sizeof(d.name), "%s", p.name);
        d.vram_bytes = p.totalGlobalMem;
        d.sm_major = p.major;
        d.sm_minor = p.minor;
    }
    return n;
}

extern "C" AJI_IMG_EXPORT aji_img_ctx *aji_img_create(const aji_img_create_params *p,
                                                      char *err, size_t err_size)
{
    if (!p || p->api_version != AJI_IMAGE_API_VERSION ||
        p->backend != AJI_IMG_THIS_BACKEND) {
        copy_err("aji_img_create: bad params or API version mismatch", err, err_size);
        return nullptr;
    }
    int n = 0;
    if (cuInit(0) != CUDA_SUCCESS || cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        copy_err("no CUDA device (NVIDIA driver missing?)", err, err_size);
        return nullptr;
    }
    int ordinal = -1;
    for (int i = 0; i < n; i++) {
        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess &&
            (p->device_id == 0 || pci_id(prop) == p->device_id)) {
            ordinal = i;
            break;
        }
    }
    if (ordinal < 0) {
        copy_err("requested CUDA device not found", err, err_size);
        return nullptr;
    }

    auto c = std::make_unique<aji_img_ctx>();
    c->logger.fn = p->log;
    c->logger.opaque = p->log_opaque;
    c->device = ordinal;
    if (cuDeviceGet(&c->cu_dev, ordinal) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&c->cu_ctx, c->cu_dev) != CUDA_SUCCESS) {
        copy_err("cannot retain the CUDA primary context", err, err_size);
        return nullptr;
    }
    {
        CtxGuard guard(c->cu_ctx);
        c->runtime.reset(nvinfer1::createInferRuntime(c->logger));
    }
    if (!c->runtime) {
        cuDevicePrimaryCtxRelease(c->cu_dev);
        copy_err("createInferRuntime failed (" AJI_TRT_PRODUCT " runtime missing?)", err, err_size);
        return nullptr;
    }
    c->cache_dir = p->cache_dir ? p->cache_dir : "";
    c->trtexec_env = p->trtexec_env ? p->trtexec_env : "";
#ifdef _WIN32
    c->trtexec = p->trtexec ? p->trtexec : own_dir() + "\\trtexec.exe";
#else
    c->trtexec = p->trtexec ? p->trtexec : own_dir() + "/trtexec";
#endif
    return c.release();
}

extern "C" AJI_IMG_EXPORT aji_img_model *aji_img_load(aji_img_ctx *c,
                                                      const aji_img_model_params *p)
{
    if (!c)
        return nullptr;
    std::lock_guard<std::mutex> lock(c->mu);
    if (!p || !p->onnx_path) {
        c->set_error("aji_img_load: no model path");
        return nullptr;
    }
    auto m = std::make_unique<aji_img_model>();
    m->c = c;
    m->onnx_path = p->onnx_path;
    m->name = stem_of(m->onnx_path);
    m->overlap = p->overlap > 0 ? p->overlap : DEFAULT_OVERLAP;

    std::vector<char> blob;
    AjiOnnxInput oin;
    if (!read_file_bytes(m->onnx_path, &blob)) {
        c->set_error("cannot read model: " + m->onnx_path);
        return nullptr;
    }
    if (!aji_onnx_input(blob.data(), blob.size(), &oin) || oin.dims.size() != 4) {
        c->set_error(m->name + ": expected one NCHW graph input");
        return nullptr;
    }
    blob.clear();
    blob.shrink_to_fit();
    m->in_ch = oin.dims[1] > 0 ? (int)oin.dims[1] : 3;
    if (m->in_ch != 1 && m->in_ch != 3) {
        c->set_error(m->name + ": input must have 1 or 3 channels");
        return nullptr;
    }
    m->settings = IMG_ENGINE_SETTINGS;
    if (oin.dims[2] > 0 && oin.dims[3] > 0) {
        m->tile_h = (int)oin.dims[2];  // fixed-shape model: its shape is the tile
        m->tile_w = (int)oin.dims[3];
    } else {
        m->tile_w = p->tile_w > 0 ? p->tile_w : DEFAULT_TILE;
        m->tile_h = p->tile_h > 0 ? p->tile_h : DEFAULT_TILE;
        const std::string shape = oin.name + ":1x" + std::to_string(m->in_ch) +
                                  "x" + std::to_string(m->tile_h) + "x" +
                                  std::to_string(m->tile_w);
        m->settings += " --minShapes=" + shape + " --optShapes=" + shape +
                       " --maxShapes=" + shape;
    }
    m->overlap = std::min(m->overlap, std::min(m->tile_w, m->tile_h) / 2);

    const std::string dir = c->cache_dir.empty()
                                ? fs::u8path(m->onnx_path).parent_path().u8string()
                                : c->cache_dir;
    std::error_code ec;
    fs::create_directories(fs::u8path(dir), ec);
    if (!c->cleaned_dirs.count(dir)) {
        std::vector<std::string> removed;
        clean_stale_engines(dir, c->device, &removed);
        for (const std::string &r : removed)
            c->info("removed stale engine " + r);
        c->cleaned_dirs.insert(dir);
    }
    m->engine_path = short_engine_path_for(dir, m->name, m->settings, c->device);
    m->timing_cache = (fs::u8path(dir) / fs::u8path(m->name + ".timing.cache")).u8string();

    if (file_exists(m->engine_path)) {
        std::string e;
        CtxGuard guard(c->cu_ctx);
        if (finalize(m.get(), &e)) {
            m->state = AJI_IMG_STATE_READY;
            c->models.insert(m.get());
            return m.release();
        }
        c->info("cached engine unusable, rebuilding: " + e);
        m->free_buffers();
        remove_file(m->engine_path);
    }

    if (!(p->flags & AJI_IMG_LOAD_ASYNC_BUILD)) {
        BuildSpec spec;
        std::string e;
        if (!make_build_spec_paths(c->trtexec, c->trtexec_env, m->onnx_path,
                                   m->timing_cache, m->name, m->settings,
                                   m->engine_path, c->device, &spec, &e)) {
            c->set_error(e);
            return nullptr;
        }
        c->info("building " AJI_TRT_PRODUCT " engine for " + m->name + ": " + spec.cmdline);
        std::atomic<intptr_t> child{0};
        if (!run_build_spec(spec, &child)) {
            c->set_error(AJI_TRT_PRODUCT " engine build failed for " + m->name + " (see " +
                         spec.log_path + ")");
            return nullptr;
        }
        become_ready(m.get());
        if (m->state != AJI_IMG_STATE_READY)
            return nullptr;
    } else {
        m->state = AJI_IMG_STATE_BUILDING;
        c->build_queue.push_back(m.get());
        pump(c);
    }
    c->models.insert(m.get());
    return m.release();
}

extern "C" AJI_IMG_EXPORT int aji_img_model_state(aji_img_model *m)
{
    if (!m)
        return AJI_IMG_STATE_FAILED;
    std::lock_guard<std::mutex> lock(m->c->mu);
    pump(m->c);
    return m->state;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_info_get(aji_img_model *m,
                                                     aji_img_model_info *info)
{
    if (!m || !info || aji_img_model_state(m) != AJI_IMG_STATE_READY)
        return AJI_IMG_NOT_READY;
    info->scale = m->scale;
    info->in_channels = m->in_ch;
    info->out_channels = m->out_ch;
    info->tile_w = m->tile_w;
    info->tile_h = m->tile_h;
    info->overlap = m->overlap;
    return AJI_IMG_OK;
}

extern "C" AJI_IMG_EXPORT int aji_img_upscale(aji_img_model *m,
                                              const aji_img_buffer *in,
                                              const aji_img_buffer *out,
                                              aji_img_progress_fn progress,
                                              void *opaque)
{
    if (!m)
        return AJI_IMG_ERR_ARGS;
    if (aji_img_model_state(m) != AJI_IMG_STATE_READY)
        return AJI_IMG_NOT_READY;
    std::lock_guard<std::mutex> run(m->run_mu);
    std::string e;
    int rc;
    {
        CtxGuard guard(m->c->cu_ctx);
        rc = upscale_tiled(*m, m->overlap, in, out, progress, opaque, &e);
    }
    if (rc != AJI_IMG_OK && rc != AJI_IMG_CANCELLED) {
        std::lock_guard<std::mutex> lock(m->c->mu);
        m->c->set_error(e);
    }
    return rc;
}

extern "C" AJI_IMG_EXPORT void aji_img_release(aji_img_model **pm)
{
    if (!pm || !*pm)
        return;
    aji_img_model *m = *pm;
    aji_img_ctx *c = m->c;
    {
        std::lock_guard<std::mutex> lock(c->mu);
        if (c->building == m) {
            c->build.stop();  // kills trtexec; the partial engine is removed
            c->build.done_flag = 0;
            c->building = nullptr;
        }
        c->build_queue.erase(std::remove(c->build_queue.begin(),
                                         c->build_queue.end(), m),
                             c->build_queue.end());
        c->models.erase(m);
        pump(c);
    }
    {
        std::lock_guard<std::mutex> run(m->run_mu);
        CtxGuard guard(c->cu_ctx);
        m->free_buffers();
    }
    delete m;
    *pm = nullptr;
}

extern "C" AJI_IMG_EXPORT const char *aji_img_last_error(aji_img_ctx *c)
{
    return c ? c->err.c_str() : "no context";
}

extern "C" AJI_IMG_EXPORT void aji_img_destroy(aji_img_ctx **pc)
{
    if (!pc || !*pc)
        return;
    aji_img_ctx *c = *pc;
    std::vector<aji_img_model *> left;
    {
        std::lock_guard<std::mutex> lock(c->mu);
        c->build.stop();
        c->building = nullptr;
        c->build_queue.clear();
        left.assign(c->models.begin(), c->models.end());
    }
    for (aji_img_model *m : left)
        aji_img_release(&m);
    {
        CtxGuard guard(c->cu_ctx);
        c->runtime.reset();
    }
    cuDevicePrimaryCtxRelease(c->cu_dev);
    delete c;
    *pc = nullptr;
}
