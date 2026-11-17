/*
 * aji_img_vk.cpp — the image ABI (aji_image.h) on Vulkan: AJI_IMG_VULKAN.
 *
 * Each model runs on libaji's native kernels on Vulkan for the model's tile (vk/imgnet.h): ESRGAN on
 * the general conv, the conv nets the video path lowers (SPAN, SPANF3, Compact) on the executor;
 * core/tiling's cover, reflect pad and overlap blend, as on the other backends.
 * A model that doesn't lower, or a device without a kernel family, is refused with the reason:
 * there is no fallback inside this backend (it also runs where there is no DirectML: Linux,
 * Android), so a caller that wants one picks another backend.
 *
 * Tiling runs on the GPU by default (upscale_gpu, the Vulkan counterpart of the CUDA backends'
 * core/tiling_gpu): the 8-bit image is uploaded once, and per tile row one command buffer cuts
 * each tile out (tile_extract), runs the model and blends its output into a float band
 * (tile_blend), then flushes the finished rows to 8-bit (band_flush) into a host-visible buffer
 * the host copies out. Bit-identical to the host path, which AJI_IMG_CPU_TILING=1 selects.
 *
 * Portable C++ with the Vulkan loader opened at run time. On Windows it is part of aji_vk.dll,
 * elsewhere libaji_vk.so. A context runs one aji_img_upscale at a time, whatever the model (its
 * models share the device's one-shot queue).
 */
#include <algorithm>
#include <chrono>
#include <cstdlib>
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
#include "native/onnx_graph.h"
#include "onnx_shape.h"
#include "vk/device.h"
#include "vk/imgnet.h"

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

// upscale_gpu's device buffers, kept across calls (grown, never shrunk).
struct VkTileScratch {
    ajivk::Buffer img, band, wx, wy, params;  // device-local
    ajivk::Buffer rows;                       // host-visible: flushed 8-bit rows
};

struct aji_img_ctx {
    aji_img_log_fn log = nullptr;
    void *log_opaque = nullptr;
    std::unique_ptr<ajivk::Device> dev;
    VkTileScratch gpu;
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
    std::unique_ptr<ajivk::ImgNet> net;
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
        if (!dev.upload(net->input(), in16.data(), in16.size() * 2, err) || !net->run(dev, tile_w, tile_h, err) ||
            !dev.download(net->output(), out16.data(), out16.size() * 2, err)) {
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
    std::string why;
    if (!ajinative::parse_onnx(blob.data(), blob.size(), &g, &why)) {
        *e = m->name + ": not on native kernels: " + why;
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> run(c->run_mu);  // the device's one-shot queue times the variants
        c->dev->set_cache_dir(c->cache_dir.empty() ? dir_of(p->onnx_path) : c->cache_dir);
        m->net = ajivk::ImgNet::create(*c->dev, g, m->tile_w, m->tile_h, &why);
        if (m->net) c->dev->save_caches();
    }
    if (!m->net) {
        *e = m->name + ": not on native kernels at " + std::to_string(m->tile_w) + "x" + std::to_string(m->tile_h) + ": " + why;
        return nullptr;
    }
    if (m->net->out_w() % m->tile_w || m->net->out_h() % m->tile_h || m->net->out_w() / m->tile_w != m->net->out_h() / m->tile_h) {
        *e = m->name + ": output is not an integer-scale image of the tile";
        return nullptr;
    }
    m->scale = m->net->out_w() / m->tile_w;
    m->in_ch = m->out_ch = 3;
    try {
        m->in16.resize((size_t)3 * m->tile_w * m->tile_h);
        m->out16.resize((size_t)3 * m->net->out_w() * m->net->out_h());
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
    // the GPU tiling scratch (rows live in host memory): grows to the largest image yet
    const VkTileScratch &g = c->gpu;
    out->ctx_scratch = g.img.size + g.band.size + g.wx.size + g.wy.size + g.params.size;
    if (m && m->net) {
        const uint64_t io = m->net->input().size + m->net->output().size;
        out->model_weights = m->net->weight_bytes();
        out->model_buffers = io;
        const uint64_t all = m->net->device_bytes();
        out->model_activations = all > out->model_weights + io ? all - out->model_weights - io : 0;
        out->activations_shared = 0;
    }
}

}  // namespace

namespace {

using namespace ajivk;  // the Vulkan entry points (vk/vk.h)

// Grow b to at least bytes (contents not kept).
bool grow(Device &dev, Buffer *b, size_t bytes, bool host, std::string *why)
{
    bytes = (bytes + 15) & ~(size_t)15;
    if (b->buf && b->size >= bytes) return true;
    dev.destroy(*b);
    return dev.make_buffer(b, (VkDeviceSize)bytes, host, why);
}

void barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = sa;
    mb.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void compute_barrier(VkCommandBuffer cb)
{
    barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

// n invocations of a 256-wide kernel, on a 2-D grid of groups (one dimension caps at 65535).
void dispatch_flat(VkCommandBuffer cb, uint32_t n)
{
    const uint32_t groups = (n + 255) / 256, gx = groups < 65535 ? groups : 65535, gy = (groups + gx - 1) / gx;
    vkCmdDispatch(cb, gx ? gx : 1, gy ? gy : 1, 1);
}

// One descriptor set of the device's layout: bindings[k] for slot k, the dummy buffer elsewhere.
struct TileSets {
    Device &dev;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    explicit TileSets(Device &d) : dev(d) {}
    ~TileSets()
    {
        if (pool) vkDestroyDescriptorPool(dev.device(), pool, nullptr);
    }
    bool init(uint32_t sets, std::string *why)
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * Device::kBindings};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = sets;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev.device(), &pi, nullptr, &pool) != VK_SUCCESS) {
            *why = "descriptor pool for tiling";
            return false;
        }
        return true;
    }
    VkDescriptorSet make(const std::vector<std::pair<uint32_t, VkBuffer>> &bindings, std::string *why)
    {
        VkDescriptorSetLayout l = dev.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(dev.device(), &ai, &set) != VK_SUCCESS) {
            *why = "descriptor set for tiling";
            return VK_NULL_HANDLE;
        }
        VkDescriptorBufferInfo bi[Device::kBindings];
        VkWriteDescriptorSet wr[Device::kBindings];
        for (uint32_t j = 0; j < (uint32_t)Device::kBindings; j++) {
            VkBuffer b = dev.dummy();
            for (const auto &p : bindings)
                if (p.first == j) b = p.second;
            bi[j] = {b, 0, VK_WHOLE_SIZE};
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = set;
            wr[j].dstBinding = j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev.device(), Device::kBindings, wr, 0, nullptr);
        return set;
    }
};

int float_bits(float f)
{
    int i;
    std::memcpy(&i, &f, 4);
    return i;
}

// upscale_tiled's cover / pad / blend with the image on the GPU (see the top of this file).
// c->run_mu held. Bit-identical to upscale_tiled.
int upscale_gpu(aji_img_model &m, const aji_img_buffer *in, const aji_img_buffer *out, aji_img_progress_fn progress,
                void *opaque, std::string *err)
{
    if (!in || !out || !in->data || !out->data) {
        *err = "null image buffer";
        return AJI_IMG_ERR_ARGS;
    }
    const int W = in->width, H = in->height, s = m.scale;
    const int Ci = in->channels, Co = out->channels;
    if (W < 1 || H < 1 || (Ci != 1 && Ci != 3) || Co != Ci) {
        *err = "image must be 1 or 3 channels, same on both sides";
        return AJI_IMG_ERR_ARGS;
    }
    if (out->width != W * s || out->height != H * s) {
        *err = "output must be " + std::to_string(W * s) + "x" + std::to_string(H * s) + " (scale " + std::to_string(s) + ")";
        return AJI_IMG_ERR_ARGS;
    }
    if (in->stride < (ptrdiff_t)W * Ci || out->stride < (ptrdiff_t)W * s * Co) {
        *err = "stride smaller than a row";
        return AJI_IMG_ERR_ARGS;
    }
    using clk = std::chrono::steady_clock;
    const bool prof = std::getenv("AJI_IMG_PROFILE") != nullptr;
    const auto t_start = clk::now();
    double t_wait = 0;

    aji_img_ctx *c = m.c;
    Device &dev = *c->dev;
    VkTileScratch &g = c->gpu;
    const int Tw = m.tile_w, Th = m.tile_h;
    const std::vector<int> ox = tile_origins(W, Tw, m.overlap), oy = tile_origins(H, Th, m.overlap);
    const int nx = (int)ox.size(), ny = (int)oy.size(), total = nx * ny;
    const int spw = std::min(Tw, W), sph = std::min(Th, H);
    const int OW = W * s, OH = H * s, tow = Tw * s, toh = Th * s;
    const int cols = spw * s, band_rows = sph * s;

    std::vector<float> wxf, wyf;
    std::vector<int32_t> prm((size_t)(2 * total + ny) * 16, 0);
    std::vector<uint8_t> img;
    try {
        for (const auto &w : axis_weights(ox, W, Tw, s)) wxf.insert(wxf.end(), w.begin(), w.end());
        for (const auto &w : axis_weights(oy, H, Th, s)) wyf.insert(wyf.end(), w.begin(), w.end());
        img.resize(((size_t)W * H * Ci + 3) & ~(size_t)3);
    } catch (const std::bad_alloc &) {
        *err = "out of host memory for tiling buffers";
        return AJI_IMG_ERR_NO_MEM;
    }
    for (int y = 0; y < H; y++) std::memcpy(img.data() + (size_t)y * W * Ci, in->data + (ptrdiff_t)y * in->stride, (size_t)W * Ci);
    // parameter entries (tile_common.glsl): extract per tile, blend per tile, flush per tile row
    const auto entry = [&](int e) { return prm.data() + (size_t)e * 16; };
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            int32_t *x = entry(j * nx + i);
            const int32_t ev[9] = {W, H, Ci, ox[i], oy[j], Tw, Th, m.in_ch, float_bits(1.0f / 255.0f)};
            std::memcpy(x, ev, sizeof ev);
            int32_t *b = entry(total + j * nx + i);
            const int32_t bv[12] = {m.out_ch, tow, toh, cols, band_rows, ox[i] * s, oy[j] * s, Co, band_rows, OW, i * cols, j * band_rows};
            std::memcpy(b, bv, sizeof bv);
        }
    int band_y0 = 0;
    std::vector<int> flush_y0(ny), flush_n(ny);
    for (int j = 0; j < ny; j++) {
        const int next_y0 = j + 1 < ny ? oy[j + 1] * s : OH;
        flush_y0[j] = band_y0;
        flush_n[j] = next_y0 - band_y0;
        const int32_t fv[5] = {Co, band_rows, OW, band_y0, next_y0 - band_y0};
        std::memcpy(entry(2 * total + j), fv, sizeof fv);
        band_y0 = next_y0;
    }
    int max_n = 0;
    for (int n : flush_n) max_n = std::max(max_n, n);
    const size_t band_bytes = (size_t)Co * band_rows * OW * 4;
    std::string why;
    if (!grow(dev, &g.img, img.size(), false, &why) || !grow(dev, &g.band, band_bytes, false, &why) ||
        !grow(dev, &g.wx, wxf.size() * 4, false, &why) || !grow(dev, &g.wy, wyf.size() * 4, false, &why) ||
        !grow(dev, &g.params, prm.size() * 4, false, &why) || !grow(dev, &g.rows, (size_t)max_n * OW * Co + 4, true, &why)) {
        *err = "out of GPU memory: " + m.name + " for a " + std::to_string(W) + "x" + std::to_string(H) + " image (" + why + ")";
        return AJI_IMG_ERR_NO_MEM;
    }
    if (!dev.upload(g.img, img.data(), img.size(), &why) || !dev.upload(g.wx, wxf.data(), wxf.size() * 4, &why) ||
        !dev.upload(g.wy, wyf.data(), wyf.size() * 4, &why) || !dev.upload(g.params, prm.data(), prm.size() * 4, &why)) {
        *err = "Vulkan upload: " + why;
        return AJI_IMG_ERR;
    }
    VkPipeline p_ext = dev.pipeline("tile_extract", 0, 0, &why), p_blend = p_ext ? dev.pipeline("tile_blend", 0, 0, &why) : nullptr,
               p_flush = p_blend ? dev.pipeline("band_flush", 0, 0, &why) : nullptr;
    TileSets sets(dev);
    VkDescriptorSet s_ext = VK_NULL_HANDLE, s_blend = VK_NULL_HANDLE, s_flush = VK_NULL_HANDLE;
    if (!p_flush || !sets.init(3, &why) ||
        !(s_ext = sets.make({{0, g.img.buf}, {1, m.net->input().buf}, {7, g.params.buf}}, &why)) ||
        !(s_blend = sets.make({{0, m.net->output().buf}, {1, g.wx.buf}, {2, g.wy.buf}, {3, g.band.buf}, {7, g.params.buf}}, &why)) ||
        !(s_flush = sets.make({{0, g.band.buf}, {1, g.rows.buf}, {7, g.params.buf}}, &why))) {
        *err = "Vulkan tiling: " + why;
        return AJI_IMG_ERR;
    }
    const auto run_kernel = [&](VkCommandBuffer cb, VkPipeline pipe, VkDescriptorSet set, int e, uint32_t n) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev.pipeline_layout(), 0, 1, &set, 0, nullptr);
        const int32_t pc[4] = {e, 0, 0, 0};
        vkCmdPushConstants(cb, dev.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        dispatch_flat(cb, n);
    };

    int done = 0;
    for (int j = 0; j < ny; j++) {
        VkCommandBuffer cb = dev.begin();
        if (j == 0) {
            vkCmdFillBuffer(cb, g.band.buf, 0, VK_WHOLE_SIZE, 0);
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        bool ok = true;
        for (int i = 0; i < nx && ok; i++) {
            run_kernel(cb, p_ext, s_ext, j * nx + i, (uint32_t)((m.in_ch * Tw * Th + 1) / 2));
            compute_barrier(cb);
            ok = m.net->record(cb, Tw, Th);
            compute_barrier(cb);
            run_kernel(cb, p_blend, s_blend, total + j * nx + i, (uint32_t)(cols * band_rows));
            compute_barrier(cb);
        }
        run_kernel(cb, p_flush, s_flush, 2 * total + j, (uint32_t)((flush_n[j] * OW * Co + 3) / 4));
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        const auto t0 = clk::now();
        const bool sub = dev.submit_wait(cb, &why);
        t_wait += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (!ok || !sub) {
            *err = m.name + ": Vulkan: " + (ok ? why : "the executor refused the tile size");
            return AJI_IMG_ERR;
        }
        const uint8_t *src = (const uint8_t *)g.rows.map;
        const size_t row = (size_t)OW * Co;
        for (int k = 0; k < flush_n[j]; k++) std::memcpy(out->data + (ptrdiff_t)(flush_y0[j] + k) * out->stride, src + k * row, row);
        for (int i = 0; i < nx; i++) {
            done++;
            if (progress && progress(opaque, done, total)) {
                *err = "cancelled";
                return AJI_IMG_CANCELLED;
            }
        }
    }
    if (prof) {
        const double total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_start).count();
        std::fprintf(stderr, "aji-profile %dx%d x%d tiles=%d tile=%dx%d total=%.1fms | gpu tiling (Vulkan), image %dch | wait=%.1f\n", W,
                     H, s, done, Tw, Th, total_ms, Ci, t_wait);
    }
    return AJI_IMG_OK;
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
    c->info("native kernels: " + m->name + " (" + m->net->describe() + ", tile " + std::to_string(m->tile_w) + "x" +
            std::to_string(m->tile_h) + ")");
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
    const int rc = std::getenv("AJI_IMG_CPU_TILING") ? upscale_tiled(*m, m->overlap, in, out, progress, opaque, &e)
                                                     : upscale_gpu(*m, in, out, progress, opaque, &e);
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
    for (ajivk::Buffer *b : {&c->gpu.img, &c->gpu.band, &c->gpu.wx, &c->gpu.wy, &c->gpu.params, &c->gpu.rows}) c->dev->destroy(*b);
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
