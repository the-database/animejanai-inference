/*
 * aji_img_vk.cpp — the image ABI (aji_image.h) on Vulkan: AJI_IMG_VULKAN.
 *
 * Each model runs on libaji's native kernels through the Vulkan executor (vk/executor.h): the ONNX
 * graph is lowered (native/lower.h) and instantiated for the model's tile; core/tiling covers the
 * image with tiles, reflect-pads small images and blends the overlaps, as on the other backends.
 * A model that doesn't lower, or a device without a kernel family, is refused with the reason:
 * there is no fallback inside this backend (it also runs where there is no DirectML: Linux,
 * Android), so a caller that wants one picks another backend.
 *
 * Portable C++ with the Vulkan loader opened at run time. On Windows it is part of aji_vk.dll,
 * elsewhere libaji_vk.so. A context runs one aji_img_upscale at a time, whatever the model (its
 * models share the device's one-shot queue).
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "aji_image.h"
#include "core/fileio.h"
#include "core/half_array.h"
#include "core/tiling.h"
#include "native/lower.h"
#include "native/onnx_graph.h"
#include "onnx_shape.h"
#include "vk/device.h"
#include "vk/executor.h"

using namespace ajicore;

namespace {

// The image ABI's device id: the PCI address as the CUDA and DirectML backends report it (bit 63
// set), else the LUID (DirectML's fallback), else the device's place in the list plus one (GPUs
// that are not on PCI, e.g. a phone's).
uint64_t image_device_id(const ajivk::DeviceInfo &d)
{
    if (d.pci_valid)
        return (1ull << 63) | ((uint64_t)d.pci_domain << 32) | ((uint64_t)d.pci_bus << 8) | (uint64_t)(d.pci_device & 0xff);
    if (d.luid_valid) {
        uint32_t lo, hi;
        std::memcpy(&lo, d.luid, 4);
        std::memcpy(&hi, d.luid + 4, 4);
        return ((uint64_t)hi << 32) | lo;
    }
    return d.index + 1;
}

// The devices a kernel family runs on, discrete GPUs first (id 0 takes the first).
std::vector<ajivk::DeviceInfo> usable_devices()
{
    std::string why;
    std::vector<ajivk::DeviceInfo> out;
    for (const auto &d : ajivk::list_devices(&why))
        if (d.family != ajivk::Family::None) out.push_back(d);
    std::stable_sort(out.begin(), out.end(), [](const ajivk::DeviceInfo &a, const ajivk::DeviceInfo &b) {
        return a.discrete > b.discrete;
    });
    return out;
}

void copy_err(const std::string &e, char *err, size_t n)
{
    if (err && n) std::snprintf(err, n, "%s", e.c_str());
}

std::string stem_of(const std::string &path)
{
    size_t s = path.find_last_of("/\\");
    std::string f = s == std::string::npos ? path : path.substr(s + 1);
    size_t dot = f.find_last_of('.');
    return dot == std::string::npos ? f : f.substr(0, dot);
}

std::string dir_of(const std::string &path)
{
    size_t s = path.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : path.substr(0, s);
}

}  // namespace

struct aji_img_ctx {
    aji_img_log_fn log = nullptr;
    void *log_opaque = nullptr;
    std::unique_ptr<ajivk::Device> dev;
    std::string cache_dir;  // empty: each model's own directory

    std::mutex mu;      // models and err
    std::mutex run_mu;  // one upscale at a time on the device
    std::set<aji_img_model *> models;
    std::string err;

    void set_error(const std::string &e)
    {
        err = e;
        if (log) log(log_opaque, 1, e.c_str());
    }
    void info(const std::string &m)
    {
        if (log) log(log_opaque, 3, m.c_str());
    }
};

struct aji_img_model : TileModel {
    aji_img_ctx *c = nullptr;
    std::string name;
    int overlap = DEFAULT_OVERLAP;
    std::unique_ptr<ajivk::Executor> ex;
    std::vector<uint16_t> in16, out16;

    int run_tile(const float *in, float *out, std::string *err) override
    {
        using clk = std::chrono::steady_clock;
        const auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        auto t0 = clk::now();
        f32_to_f16_array(in, in16.data(), in16.size());
        if (prof.on) {
            const auto t1 = clk::now();
            prof.pack += ms(t0, t1);
            t0 = t1;
        }
        ajivk::Device &dev = *c->dev;
        if (!dev.upload(ex->input(), in16.data(), in16.size() * 2, err) || !ex->run(tile_w, tile_h, err) ||
            !dev.download(ex->output(), out16.data(), out16.size() * 2, err)) {
            *err = name + ": Vulkan: " + *err;
            return AJI_IMG_ERR;
        }
        if (prof.on) {
            const auto t1 = clk::now();
            prof.wait += ms(t0, t1);  // upload, inference, download
            t0 = t1;
        }
        // only the planes the blend reads (a gray page: the first)
        const size_t nu = out16.size() / out_ch * std::min(out_used, out_ch);
        parallel_for(nu, (size_t)1 << 18, [&](size_t b, size_t e) { f16_to_f32_array(out16.data() + b, out + b, e - b); });
        if (prof.on) prof.unpack += ms(t0, clk::now());
        return AJI_IMG_OK;
    }
};

namespace {

// The model at its tile on the context's device, without registering it. Null and *e on failure.
std::unique_ptr<aji_img_model> build_model(aji_img_ctx *c, const aji_img_model_params *p, std::string *e)
{
    auto m = std::make_unique<aji_img_model>();
    m->c = c;
    m->name = stem_of(p->onnx_path);
    std::vector<char> blob;
    if (!read_file_bytes(p->onnx_path, &blob)) {
        *e = "failed to read " + std::string(p->onnx_path);
        return nullptr;
    }
    AjiOnnxInput oin;
    if (!aji_onnx_input(blob.data(), blob.size(), &oin) || oin.dims.size() != 4) {
        *e = m->name + ": expected one NCHW input";
        return nullptr;
    }
    if (oin.dims[1] > 0 && oin.dims[1] != 3) {
        *e = m->name + ": native kernels take 3-channel models only";
        return nullptr;
    }
    if (oin.dims[2] > 0 && oin.dims[3] > 0) {
        m->tile_h = (int)oin.dims[2];
        m->tile_w = (int)oin.dims[3];
    } else {
        m->tile_w = p->tile_w > 0 ? p->tile_w : DEFAULT_TILE;
        m->tile_h = p->tile_h > 0 ? p->tile_h : DEFAULT_TILE;
    }
    m->overlap = std::min(p->overlap > 0 ? p->overlap : DEFAULT_OVERLAP, std::min(m->tile_w, m->tile_h) / 2);
    ajinative::OnnxGraph g;
    ajinative::Plan plan;
    std::string why;
    if (!ajinative::parse_onnx(blob.data(), blob.size(), &g, &why) || !ajinative::lower(g, &plan, &why)) {
        *e = m->name + ": not on native kernels: " + why;
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> run(c->run_mu);  // the device's one-shot queue times the variants
        c->dev->set_cache_dir(c->cache_dir.empty() ? dir_of(p->onnx_path) : c->cache_dir);
        m->ex = ajivk::Executor::create(*c->dev, plan, m->tile_w, m->tile_h, &why);
        if (m->ex) c->dev->save_caches();
    }
    if (!m->ex) {
        *e = m->name + ": not on native kernels at " + std::to_string(m->tile_w) + "x" + std::to_string(m->tile_h) + ": " + why;
        return nullptr;
    }
    if (m->ex->out_w() % m->tile_w || m->ex->out_h() % m->tile_h || m->ex->out_w() / m->tile_w != m->ex->out_h() / m->tile_h) {
        *e = m->name + ": output is not an integer-scale image of the tile";
        return nullptr;
    }
    m->scale = m->ex->out_w() / m->tile_w;
    m->in_ch = m->out_ch = 3;
    try {
        m->in16.resize((size_t)3 * m->tile_w * m->tile_h);
        m->out16.resize((size_t)3 * m->ex->out_w() * m->ex->out_h());
    } catch (const std::bad_alloc &) {
        *e = m->name + ": out of host memory for tile buffers";
        return nullptr;
    }
    return m;
}

void fill_memory(aji_img_ctx *c, const aji_img_model *m, aji_img_memory *out)
{
    *out = {};
    c->dev->memory(&out->device_free, &out->device_total);
    if (m && m->ex) {
        const uint64_t io = m->ex->input().size + m->ex->output().size;
        out->model_weights = m->ex->weight_bytes();
        out->model_buffers = io;
        const uint64_t all = m->ex->device_bytes();
        out->model_activations = all > out->model_weights + io ? all - out->model_weights - io : 0;
        out->activations_shared = 0;
    }
}

}  // namespace

extern "C" AJI_IMG_EXPORT int aji_img_enum_devices(int backend, aji_img_device *out, int max)
{
    if (backend != AJI_IMG_VULKAN) return -1;
    std::string why;
    if (!ajivk::instance(&why)) return -1;
    const auto devs = usable_devices();
    for (int i = 0; i < (int)devs.size() && out && i < max; i++) {
        aji_img_device &o = out[i];
        std::memset(&o, 0, sizeof(o));
        o.id = image_device_id(devs[i]);
        std::snprintf(o.name, sizeof(o.name), "%s", devs[i].name.c_str());
        o.vram_bytes = devs[i].device_local_bytes;
    }
    return (int)devs.size();
}

extern "C" AJI_IMG_EXPORT aji_img_ctx *aji_img_create(const aji_img_create_params *p, char *err, size_t err_size)
{
    if (!p || p->api_version != AJI_IMAGE_API_VERSION || p->backend != AJI_IMG_VULKAN) {
        copy_err("aji_img_create: bad params or API version mismatch", err, err_size);
        return nullptr;
    }
    std::string why;
    if (!ajivk::instance(&why)) {
        copy_err(why, err, err_size);
        return nullptr;
    }
    const auto devs = usable_devices();
    const ajivk::DeviceInfo *pick = nullptr;
    for (const auto &d : devs)
        if (p->device_id == 0 || image_device_id(d) == p->device_id) {
            pick = &d;
            break;
        }
    if (!pick) {
        copy_err(devs.empty() ? "no Vulkan device runs aji_vk's kernels (aji_vk_model_test info says why)"
                              : "the requested Vulkan device is gone",
                 err, err_size);
        return nullptr;
    }
    auto c = std::make_unique<aji_img_ctx>();
    c->log = p->log;
    c->log_opaque = p->log_opaque;
    if (p->cache_dir) c->cache_dir = p->cache_dir;
    c->dev = ajivk::Device::create((int)pick->index, nullptr, &why);
    if (!c->dev) {
        copy_err("Vulkan device: " + why, err, err_size);
        return nullptr;
    }
    c->info("Vulkan device: " + c->dev->info().name + ", kernel family " + ajivk::family_name(c->dev->family()));
    return c.release();
}

extern "C" AJI_IMG_EXPORT aji_img_model *aji_img_load(aji_img_ctx *c, const aji_img_model_params *p)
{
    if (!c) return nullptr;
    std::lock_guard<std::mutex> lock(c->mu);
    if (!p || !p->onnx_path) {
        c->set_error("aji_img_load: no model path");
        return nullptr;
    }
    std::string e;
    std::unique_ptr<aji_img_model> m = build_model(c, p, &e);
    if (!m) {
        c->set_error(e);
        return nullptr;
    }
    c->info("native kernels: " + m->name + " (Vulkan " + ajivk::family_name(c->dev->family()) + ", tile " +
            std::to_string(m->tile_w) + "x" + std::to_string(m->tile_h) + ")");
    aji_img_model *raw = m.release();
    c->models.insert(raw);
    return raw;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_state(aji_img_model *m)
{
    return m ? AJI_IMG_STATE_READY : AJI_IMG_STATE_FAILED;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_info_get(aji_img_model *m, aji_img_model_info *info)
{
    if (!m || !info) return AJI_IMG_ERR_ARGS;
    info->scale = m->scale;
    info->in_channels = m->in_ch;
    info->out_channels = m->out_ch;
    info->tile_w = m->tile_w;
    info->tile_h = m->tile_h;
    info->overlap = m->overlap;
    return AJI_IMG_OK;
}

extern "C" AJI_IMG_EXPORT int aji_img_upscale(aji_img_model *m, const aji_img_buffer *in, const aji_img_buffer *out,
                                              aji_img_progress_fn progress, void *opaque)
{
    if (!m) return AJI_IMG_ERR_ARGS;
    std::lock_guard<std::mutex> run(m->c->run_mu);
    std::string e;
    const int rc = upscale_tiled(*m, m->overlap, in, out, progress, opaque, &e);
    if (rc != AJI_IMG_OK && rc != AJI_IMG_CANCELLED) {
        std::lock_guard<std::mutex> lock(m->c->mu);
        m->c->set_error(e);
    }
    return rc;
}

extern "C" AJI_IMG_EXPORT void aji_img_release(aji_img_model **pm)
{
    if (!pm || !*pm) return;
    aji_img_model *m = *pm;
    {
        std::lock_guard<std::mutex> lock(m->c->mu);
        m->c->models.erase(m);
    }
    {
        std::lock_guard<std::mutex> run(m->c->run_mu);  // let a running upscale finish
        delete m;
    }
    *pm = nullptr;
}

extern "C" AJI_IMG_EXPORT const char *aji_img_last_error(aji_img_ctx *c)
{
    return c ? c->err.c_str() : "no context";
}

// Vulkan keeps nothing on the device once its contexts are destroyed (each has its own VkDevice).
extern "C" AJI_IMG_EXPORT int aji_img_release_device(int, uint64_t)
{
    return 0;
}

extern "C" AJI_IMG_EXPORT void aji_img_destroy(aji_img_ctx **pc)
{
    if (!pc || !*pc) return;
    aji_img_ctx *c = *pc;
    std::vector<aji_img_model *> left(c->models.begin(), c->models.end());
    for (aji_img_model *m : left) aji_img_release(&m);
    delete c;
    *pc = nullptr;
}

extern "C" AJI_IMG_EXPORT int aji_img_memory_get(aji_img_ctx *c, aji_img_model *m, aji_img_memory *out)
{
    if (!c || !out || (m && m->c != c)) return AJI_IMG_ERR_ARGS;
    fill_memory(c, m, out);
    return AJI_IMG_OK;
}

// Every model this backend loads is on native kernels, so the probe is a load and a release.
extern "C" AJI_IMG_EXPORT int aji_img_native_probe(aji_img_ctx *c, const aji_img_model_params *p, aji_img_memory *out)
{
    if (!c || !p || !p->onnx_path || !out) return AJI_IMG_ERR_ARGS;
    std::lock_guard<std::mutex> lock(c->mu);
    std::string e;
    std::unique_ptr<aji_img_model> m = build_model(c, p, &e);
    if (!m) {
        c->set_error(e);
        return AJI_IMG_ERR;
    }
    fill_memory(c, m.get(), out);
    m.reset();
    uint64_t total = 0;
    c->dev->memory(&out->device_free, &total);
    return AJI_IMG_OK;
}
