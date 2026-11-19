// vk/video.cpp — see video.h.
#include "vk/video.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/fileio.h"
#include "frame_diff.h"
#include "native/lower.h"
#include "native/onnx_graph.h"
#include "onnx_shape.h"

namespace ajivk {

constexpr int AJI_TEMPORAL_MAX_FRAMES = 15;  // kernels.h's AJI_TEMPORAL_MAX (the gather's 3 x 6 slots of 5 bits)

namespace {

// Two aji_vk_images of one VkImage (one wrapper twice, or two wrappers of it)
bool same_image(const aji_vk_image *a, const aji_vk_image *b) { return a == b || a->image == b->image; }

// Queue-family ownership of a caller's image (aji.h): read from an owner other than libaji's
// family, it is acquired from it (the caller released it to libaji's family); an output's
// contents are replaced, so it moves without a transfer. VK_QUEUE_FAMILY_IGNORED (a CONCURRENT
// image, or one used on libaji's family alone) transfers nothing.
void set_owner(VkImageMemoryBarrier &b, const aji_vk_image *im, uint32_t ours, bool read)
{
    const bool acquire = read && im->queue_family != VK_QUEUE_FAMILY_IGNORED && im->queue_family != ours;
    b.srcQueueFamilyIndex = acquire ? im->queue_family : VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = acquire ? ours : VK_QUEUE_FAMILY_IGNORED;
}
// the owner to write back: libaji's family, unless the image had none (it keeps none)
uint32_t owner_after(const aji_vk_image *im, uint32_t ours) { return im->queue_family == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : ours; }

// One submit's frame semaphores (AVVkFrame's rule: wait for value, signal value + 1) after libaji's
// own timeline at sig[0]. Frames on one timeline semaphore (a frame twice, or two images that share
// it) wait once for the later value and signal it once: Vulkan forbids signalling a semaphore twice
// in a submission. commit() writes each frame's new value back.
struct FrameSems {
    Device::SemValue waits[2], sig[3];
    uint32_t nwait = 0;
    FrameSems(VkSemaphore tl, uint64_t ticket, aji_vk_image *const *im, int n)
    {
        sig[0] = {tl, ticket};
        for (int i = 0; i < n; i++) {
            const VkSemaphore s = static_cast<VkSemaphore>(im[i]->semaphore);
            uint32_t j = 0;
            while (j < nwait && waits[j].sem != s) j++;
            if (j == nwait) waits[nwait++] = {s, im[i]->value};
            else waits[j].value = std::max(waits[j].value, im[i]->value);
        }
        for (uint32_t j = 0; j < nwait; j++) sig[1 + j] = {waits[j].sem, waits[j].value + 1};
    }
    uint32_t nsig() const { return 1 + nwait; }
    void commit(aji_vk_image *const *im, int n) const
    {
        for (int i = 0; i < n; i++)
            for (uint32_t j = 0; j < nwait; j++)
                if (sig[1 + j].sem == static_cast<VkSemaphore>(im[i]->semaphore)) im[i]->value = sig[1 + j].value;
    }
};

std::string fmt_num(double v)
{
    if (v >= 1e300) return "inf";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

int round_even(double v)
{
    const int i = (int)(v + 0.5);
    return i & ~1;
}

// fit into (box_w, box_h) preserving the aspect (aji_dml.cpp's fit_box)
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

constexpr uint64_t kWaitNs = 10ull * 1000 * 1000 * 1000;  // a ticket wait or a slot reuse: a lost device after this

}  // namespace

void VkVideo::logf(int level, const char *fmt, ...)
{
    if (!log_fn_) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_fn_(log_opaque_, level, buf);
}

void VkVideo::fail(const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    error_ = buf;
    if (log_fn_) log_fn_(log_opaque_, 0, buf);
}

void VkVideo::finalize_log()
{
    std::string out;
    for (auto &l : log_info_) out += l + "\n";
    out += "\n";
    for (size_t i = 0; i < log_steps_.size(); i++) out += std::to_string(i + 1) + ". " + log_steps_[i] + "\n";
    log_ = out;
}

std::unique_ptr<VkVideo> VkVideo::create(const aji_create_params *p, const aji_vk_device *d, std::string *why)
{
    const auto refuse = [&](const std::string &m) -> std::unique_ptr<VkVideo> {
        if (why) *why = m;
        return nullptr;
    };
    if (!p || !d || d->struct_size < sizeof(aji_vk_device)) return refuse("no Vulkan device given");
    if (!p->conf_path) return refuse("Vulkan frames need conf mode (conf_path)");
    ExternalDevice e;
    e.phys = static_cast<VkPhysicalDevice>(d->physical_device);
    e.device = static_cast<VkDevice>(d->device);
    e.queue = static_cast<VkQueue>(d->queue);
    e.family = d->queue_family;
    e.index = d->queue_index;
    e.lock = d->lock_queue;
    e.unlock = d->unlock_queue;
    e.opaque = d->lock_opaque;
    for (uint32_t i = 0; i < d->num_enabled_extensions && d->enabled_extensions; i++)
        if (d->enabled_extensions[i]) e.extensions.push_back(d->enabled_extensions[i]);
    e.features = d->enabled_features;
    std::unique_ptr<VkVideo> v(new VkVideo());
    v->log_fn_ = p->log;
    v->log_opaque_ = p->log_opaque;
    v->conf_path_ = p->conf_path;
    v->model_dir_ = p->model_dir ? p->model_dir : "";
    v->rife_dir_ = p->rife_model_dir ? p->rife_model_dir : "";
    v->slot_ = p->slot;
    v->dev_ = Device::create_external(e, why);
    if (!v->dev_) return nullptr;
    if (!v->model_dir_.empty()) v->dev_->set_cache_dir(v->model_dir_);
    VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st};
    if (vkCreateSemaphore(v->dev_->device(), &si, nullptr, &v->tl_) != VK_SUCCESS) return refuse("timeline semaphore");
    v->logf(2, "Vulkan frames on %s (%s kernels)", v->dev_->info().name.c_str(), family_name(v->dev_->family()));
    return v;
}

VkVideo::~VkVideo()
{
    if (!dev_) return;
    wait(ticket_);
    teardown();
    diff_teardown();
    if (tl_) vkDestroySemaphore(dev_->device(), tl_, nullptr);
    dev_->save_caches();
}

void VkVideo::rife_teardown()
{
    RifeState &R = rife_;
    R.io[0].reset();
    R.io[1].reset();
    R.net.reset();
    if (R.pool) vkDestroyDescriptorPool(dev_->device(), R.pool, nullptr);
    for (Buffer *b : {&R.part, &R.sum, &R.read, &R.params}) dev_->destroy(*b);
    R = RifeState{};
}

// "414" -> rife_v4.14, a trailing 1 on four digits -> _lite (aji_dml.cpp's rife_model_name)
static std::string rife_name(int code, bool ensemble)
{
    std::string s = std::to_string(code);
    if (s.size() < 2) return "";
    const std::string dec = s.size() == 2 ? s.substr(1, 1) : s.substr(1, 2);
    std::string name = "rife_v" + s.substr(0, 1) + "." + dec;
    if (s.size() == 4 && s.back() == '1') name += "_lite";
    if (ensemble) name += "_ensemble";
    return name;
}

bool VkVideo::rife_setup(const AjiChainConf &chain, int w, int h, double fps)
{
    RifeState &R = rife_;
    R.w = w;
    R.h = h;
    R.pw = (w + 63) / 64 * 64;
    R.ph = (h + 63) / 64 * 64;
    // centred like animejanai_core's AddBorders split, rounded down to even (4:2:0 chroma)
    R.pad_l = ((R.pw - w) / 2) & ~1;
    R.pad_t = ((R.ph - h) / 2) & ~1;
    R.num = chain.rife_factor_num;
    R.den = chain.rife_factor_den;
    R.threshold = chain.rife_scd_threshold;
    R.enhanced = chain.rife_scene_detect_enhanced;
    const std::string model = rife_name(chain.rife_model, chain.rife_ensemble);
    std::string why;
    std::vector<char> bytes;
    ajinative::OnnxGraph g;
    if (model.empty()) why = "invalid rife model code " + std::to_string(chain.rife_model);
    else if (!ajicore::read_file_bytes(rife_dir_ + "/" + model + ".onnx", &bytes)) why = model + ".onnx can't be read";
    else if (ajinative::parse_onnx(bytes.data(), bytes.size(), &g, &why)) R.net = Rife::create(*dev_, g, R.pw, R.ph, &why);
    if (!R.net) {
        log_steps_.push_back("RIFE not used: " + why + " (no DirectML on Vulkan frames)");
        rife_teardown();
        return false;
    }
    // the scene detection: partials over the padded luma, their sum, a readback
    R.groups_x = (uint32_t)(R.pw + 31) / 32;
    R.groups_y = (uint32_t)(R.ph + 7) / 8;
    if (!dev_->make_buffer(&R.part, (VkDeviceSize)R.groups_x * R.groups_y * 4, false, &why) || !dev_->make_buffer(&R.sum, 4, false, &why) ||
        !dev_->make_buffer(&R.read, 4, true, &why)) {
        log_steps_.push_back("RIFE not used: " + why);
        rife_teardown();
        return false;
    }
    char buf[256];
    const double factor = (double)R.num / R.den;
    if (R.pw != w || R.ph != h)
        std::snprintf(buf, sizeof buf, "Padded to %dx%d, applied RIFE v%d Interpolation %.3fx, cropped back to %dx%d;    New Video FPS: %.3f",
                      R.pw, R.ph, chain.rife_model, factor, w, h, fps * factor);
    else
        std::snprintf(buf, sizeof buf, "Applied RIFE v%d Interpolation %.3fx;    New Video FPS: %.3f", chain.rife_model, factor, fps * factor);
    const std::string kernels = "Native kernels: " + model + " (" + R.net->describe() + ")";
    if (R.before) log_steps_.insert(log_steps_.begin() + (pre_.on ? 1 : 0), {buf, kernels});
    else log_steps_.insert(log_steps_.end(), {buf, kernels});
    R.on = true;
    return true;
}

// The pre / post of both frames and the scene detection's sets, for a frame format and range.
bool VkVideo::rife_formats(int fmt, int range, std::string *why)
{
    RifeState &R = rife_;
    if (R.io[0] && R.fmt == fmt && R.range == range) return true;
    if (ticket_) wait(ticket_);
    R.io[0].reset();
    R.io[1].reset();
    if (R.pool) vkDestroyDescriptorPool(dev_->device(), R.pool, nullptr);
    R.pool = VK_NULL_HANDLE;
    dev_->destroy(R.params);
    const FrameFormat f{fmt, AJI_MATRIX_BT709, range, AJI_SITING_LEFT};  // BT.709, as rife_cuda.py
    const uint32_t plane = (uint32_t)R.pw * R.ph;
    for (int i = 0; i < 2; i++) {
        R.io[i] = FrameIO::create(*dev_, R.pw, R.ph, R.pw, R.ph, f, f, R.net->input().buf, R.net->output().buf, why, 1 /* AJI_FILTER_BILINEAR */,
                                  i * 3 * plane);
        if (!R.io[i]) return false;
    }
    const bool t16 = fmt == AJI_FMT_P010;
    // descriptor sets: the scene detection's partial (a, b luma -> part) and final (part -> sum),
    // the constant channels 7..10 of the model input
    struct KA {
        int32_t di[4], dj[4];
        float kr, kb, yoff, yscale, coff, cscale, qdiv, qmax;
    } ka[3] = {};
    ka[0].di[0] = R.pw;
    ka[0].di[1] = R.ph;
    ka[0].di[2] = (int32_t)R.groups_x;
    ka[0].dj[0] = ka[0].dj[1] = (int32_t)R.io[0]->in_pitch();
    ka[0].kr = t16 ? 1.0f / 65472.0f : 1.0f / 255.0f;
    ka[1].di[0] = (int32_t)(R.groups_x * R.groups_y);
    ka[2].di[0] = R.pw;
    ka[2].di[1] = R.ph;
    ka[2].dj[0] = (int32_t)(plane * 7);
    if (!dev_->make_buffer(&R.params, sizeof ka, false, why) || !dev_->upload(R.params, ka, sizeof ka, why)) return false;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * Device::kBindings};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 3;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(dev_->device(), &pi, nullptr, &R.pool) != VK_SUCCESS) return *why = "descriptor pool", false;
    const VkBuffer d[3][3] = {{R.io[0]->in_y().buf, R.io[1]->in_y().buf, R.part.buf},
                              {R.part.buf, R.sum.buf, VK_NULL_HANDLE},
                              {R.net->input().buf, VK_NULL_HANDLE, VK_NULL_HANDLE}};
    const char *mods[3] = {t16 ? "fr_scd1_u16" : "fr_scd1_u8", "fr_scd2", "fr_rife_consts"};
    for (int k = 0; k < 3; k++) {
        R.pipes[k] = dev_->pipeline(mods[k], 0, 0, why);
        if (!R.pipes[k]) return false;
        VkDescriptorSetLayout l = dev_->set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = R.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev_->device(), &ai, &R.sets[k]) != VK_SUCCESS) return *why = "descriptor set", false;
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev_->dummy(), 0, VK_WHOLE_SIZE};
        for (int j = 0; j < 3; j++)
            if (d[k][j]) bi[2 + j] = {d[k][j], 0, VK_WHOLE_SIZE};
        bi[7] = {R.params.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = R.sets[k];
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev_->device(), Device::kBindings, wr, 0, nullptr);
    }
    // once: the padding in studio black (the frames overwrite their window every call), the
    // constant channels (the mesh and its multipliers, as the other backends fill them)
    VkCommandBuffer cb = dev_->begin();
    for (int i = 0; i < 2; i++) {
        vkCmdFillBuffer(cb, R.io[i]->in_y().buf, 0, VK_WHOLE_SIZE, t16 ? 0x10001000u : 0x10101010u);
        vkCmdFillBuffer(cb, R.io[i]->in_uv().buf, 0, VK_WHOLE_SIZE, t16 ? 0x80008000u : 0x80808080u);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, R.pipes[2]);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_->pipeline_layout(), 0, 1, &R.sets[2], 0, nullptr);
    const int32_t pc[4] = {2, 0, 0, 0};
    vkCmdPushConstants(cb, dev_->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (uint32_t)(R.pw / 2 + 31) / 32, (uint32_t)(R.ph + 7) / 8, 1);
    if (!dev_->submit_wait(cb, why)) return false;
    R.fmt = fmt;
    R.range = range;
    return true;
}

int VkVideo::rife_factor(int *num, int *den) const
{
    if (!rife_.on) return 0;
    if (num) *num = rife_.num;
    if (den) *den = rife_.den;
    return 1;
}

int VkVideo::infer_rife(const aji_frame *a, const aji_frame *b, double t, const aji_frame *out, int scene)
{
    RifeState &R = rife_;
    if (!R.on) {
        fail("aji_infer_rife without an active RIFE configuration");
        return AJI_ERR;
    }
    if (!a || !b || !out || !a->plane[0] || !b->plane[0] || !out->plane[0] || scene < -1 || scene > 1) return AJI_ERR;
    if (a->format != b->format || a->format != out->format || (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010)) {
        fail("rife frame formats must match (nv12/p010)");
        return AJI_ERR_FORMAT;
    }
    if (a->width != R.w || a->height != R.h || b->width != R.w || b->height != R.h || out->width != R.w || out->height != R.h) {
        fail("rife frame dims do not match configured %dx%d", R.w, R.h);
        return AJI_ERR_SHAPE;
    }
    if (scene == 1) return AJI_SCENE;
    std::string why;
    if (!rife_formats(a->format, a->range, &why)) {
        fail("aji_infer_rife: %s", why.c_str());
        return AJI_ERR;
    }
    auto *ia = static_cast<aji_vk_image *>(a->plane[0]);
    auto *ib = static_cast<aji_vk_image *>(b->plane[0]);
    auto *io = static_cast<aji_vk_image *>(out->plane[0]);
    const uint32_t bps = a->format == AJI_FMT_P010 ? 2 : 1, pitch = R.io[0]->in_pitch(), plane = (uint32_t)R.pw * R.ph;
    const auto copy_region = [&](VkImageAspectFlagBits aspect, int w, int h, VkDeviceSize off, uint32_t texel) {
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.bufferRowLength = pitch / texel;
        r.imageSubresource = {(VkImageAspectFlags)aspect, 0, 0, 1};
        r.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
        return r;
    };
    const VkDeviceSize off_y = (VkDeviceSize)R.pad_t * pitch + (VkDeviceSize)R.pad_l * bps;
    const VkDeviceSize off_uv = (VkDeviceSize)(R.pad_t / 2) * pitch + (VkDeviceSize)R.pad_l * bps;
    const VkBufferImageCopy ry = copy_region(VK_IMAGE_ASPECT_PLANE_0_BIT, R.w, R.h, off_y, bps);
    const VkBufferImageCopy ruv = copy_region(VK_IMAGE_ASPECT_PLANE_1_BIT, R.w / 2, R.h / 2, off_uv, 2 * bps);
    const auto barrier_image = [&](VkCommandBuffer cb, aji_vk_image *im, VkImageLayout to, VkAccessFlags acc, bool discard) {
        VkImageMemoryBarrier ib2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib2.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        ib2.dstAccessMask = acc;
        ib2.oldLayout = discard ? VK_IMAGE_LAYOUT_UNDEFINED : static_cast<VkImageLayout>(im->layout);
        ib2.newLayout = to;
        set_owner(ib2, im, dev_->queue_family(), !discard);
        ib2.image = static_cast<VkImage>(im->image);
        ib2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib2);
    };
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    const auto mem_barrier = [&](VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da) {
        mb.srcAccessMask = sa;
        mb.dstAccessMask = da;
        vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    // 1: both frames into the padded inputs (and, when asked, the scene detection's sum to the host)
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_infer_rife: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    // a and b can be one frame (a held frame repeated, a clip's last frame twice), through one
    // aji_vk_image or two: each image is transitioned once, each semaphore waited and signalled once
    aji_vk_image *in2[2] = {ia, ib};
    const bool one_image = same_image(ia, ib);
    for (int i = 0; i < 2; i++) {
        if (i == 0 || !one_image) barrier_image(cb, in2[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, false);
        vkCmdCopyImageToBuffer(cb, static_cast<VkImage>(in2[i]->image), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, R.io[i]->in_y().buf, 1, &ry);
        vkCmdCopyImageToBuffer(cb, static_cast<VkImage>(in2[i]->image), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, R.io[i]->in_uv().buf, 1, &ruv);
    }
    mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (scene < 0) {
        for (int k = 0; k < 2; k++) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, R.pipes[k]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_->pipeline_layout(), 0, 1, &R.sets[k], 0, nullptr);
            const int32_t pc[4] = {k, 0, 0, 0};
            vkCmdPushConstants(cb, dev_->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            if (k == 0) vkCmdDispatch(cb, R.groups_x, R.groups_y, 1);
            else vkCmdDispatch(cb, 1, 1, 1);
            mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        }
        const VkBufferCopy c4{0, 0, 4};
        vkCmdCopyBuffer(cb, R.sum.buf, R.read.buf, 1, &c4);
        mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    }
    {
        const uint64_t tk = ticket_ + 1;
        const FrameSems fs(tl_, tk, in2, 2);
        if (!dev_->frame_submit(slot, fs.waits, fs.nwait, fs.sig, fs.nsig(), &why)) {
            fail("aji_infer_rife: %s", why.c_str());
            return AJI_ERR;
        }
        ticket_ = tk;
        fs.commit(in2, 2);
        for (int i = 0; i < 2; i++) {
            aji_vk_image *im = in2[i];
            im->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            im->access = VK_ACCESS_2_TRANSFER_READ_BIT;
            im->queue_family = owner_after(im, dev_->queue_family());
        }
    }
    if (scene < 0) {
        if (wait(ticket_) != AJI_OK) return AJI_ERR;
        float sum = 0.0f;
        std::memcpy(&sum, R.read.map, 4);
        // the reference runs SCDetect on the padded clip: the borders add nothing, the mean divides by the padded area
        if (sum / ((double)R.pw * R.ph) > R.threshold) return AJI_SCENE;
    }
    // 2: pre of both, the timestep plane, the model, post, the window into the output frame
    cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_infer_rife: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    mem_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    R.io[0]->record_pre(cb);
    R.io[1]->record_pre(cb);
    {
        // channel 6, the timestep: fp16 by truncation, as the other backends fill it
        uint32_t x;
        const float ft = (float)t;
        std::memcpy(&x, &ft, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        const int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
        const uint32_t half = e <= 0 ? sign : e >= 31 ? (sign | 0x7c00u) : (sign | ((uint32_t)e << 10) | ((x & 0x7fffffu) >> 13));
        vkCmdFillBuffer(cb, R.net->input().buf, (VkDeviceSize)plane * 6 * 2, (VkDeviceSize)plane * 2, half | (half << 16));
        mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    R.net->record(cb);
    mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    R.io[0]->record_post(cb);
    mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    barrier_image(cb, io, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, true);
    const uint32_t opitch = R.io[0]->out_pitch();
    VkBufferImageCopy oy = copy_region(VK_IMAGE_ASPECT_PLANE_0_BIT, R.w, R.h, (VkDeviceSize)R.pad_t * opitch + (VkDeviceSize)R.pad_l * bps, bps);
    VkBufferImageCopy ouv = copy_region(VK_IMAGE_ASPECT_PLANE_1_BIT, R.w / 2, R.h / 2,
                                        (VkDeviceSize)(R.pad_t / 2) * opitch + (VkDeviceSize)R.pad_l * bps, 2 * bps);
    oy.bufferRowLength = opitch / bps;
    ouv.bufferRowLength = opitch / (2 * bps);
    vkCmdCopyBufferToImage(cb, R.io[0]->out_y().buf, static_cast<VkImage>(io->image), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &oy);
    vkCmdCopyBufferToImage(cb, R.io[0]->out_uv().buf, static_cast<VkImage>(io->image), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ouv);
    {
        const Device::SemValue w1{static_cast<VkSemaphore>(io->semaphore), io->value};
        const uint64_t tk = ticket_ + 1;
        const Device::SemValue sig[2] = {{tl_, tk}, {static_cast<VkSemaphore>(io->semaphore), io->value + 1}};
        if (!dev_->frame_submit(slot, &w1, 1, sig, 2, &why)) {
            fail("aji_infer_rife: %s", why.c_str());
            return AJI_ERR;
        }
        ticket_ = tk;
        io->value += 1;
        io->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        io->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        io->queue_family = owner_after(io, dev_->queue_family());
    }
    // synchronous like the other backends' aji_infer_rife: the interpolated frame is complete on return
    return wait(ticket_);
}

void VkVideo::teardown()
{
    if (dev_ && ticket_) wait(ticket_);
    rife_teardown();
    temporal_teardown();
    pre_.io.reset();
    pre_.rs.reset();
    dev_->destroy(pre_.a);
    dev_->destroy(pre_.b);
    pre_ = PreResize{};
    io_.reset();
    steps_.clear();
    for (Buffer &b : chain_) dev_->destroy(b);
    active_ = false;
}

int VkVideo::set_slot(int slot)
{
    slot_ = slot;
    return AJI_OK;
}

int VkVideo::configure(int w, int h, double fps, int *out_w, int *out_h)
{
    if (w < 2 || h < 2 || (w & 1) || (h & 1)) return AJI_ERR_SHAPE;
    std::string err;
    aji_conf_load(conf_path_.c_str(), &conf_, &err);  // re-read: edits apply without a restart
    teardown();
    log_info_.clear();
    log_steps_.clear();
    in_w_ = out_w_ = w;
    in_h_ = out_h_ = h;
    const auto passthrough = [&](const std::string &step) {
        if (!step.empty()) log_steps_.push_back(step);
        teardown();
        out_w_ = in_w_;
        out_h_ = in_h_;
        finalize_log();
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;
    };

    auto sit = conf_.slots.find(slot_);
    std::string profile = sit != conf_.slots.end() ? sit->second.profile_name : "(missing slot)";
    if (slot_ == 0) profile = "Off";
    else if (slot_ < 10) profile = std::to_string(slot_) + ". " + profile;
    log_info_.push_back("Upscale Profile: " + profile);
    log_info_.push_back("GPU: " + dev_->info().name);
    {
        char buf[128];
        std::snprintf(buf, sizeof buf, "Original Video Resolution: %dx%d;    Original Video FPS: %.3f", w, h, fps);
        log_info_.push_back(buf);
    }
    const AjiChainConf *chain = aji_select_chain(conf_, slot_, model_dir_, w, h, fps, &log_info_);
    if (!chain) {
        log_info_.push_back("No Chains Activated");
        return passthrough("");
    }
    log_info_.push_back("Active Upscale Chain: " + std::to_string(chain->index) + ";    Resolution Range: " + chain->min_resolution + " - " +
                        chain->max_resolution + ";    FPS Range: " + fmt_num(chain->min_fps) + " - " + fmt_num(chain->max_fps));
    if (!conf_.native) return passthrough("Vulkan frames run on the native kernels, which [global] native=off turns off: passthrough");

    int cw = w, ch = h;
    int first_rs_log = -1;  // the first model's explicit resize: its log line (steps_[0])
    for (const AjiModelConf &m : chain->models) {
        if (m.name.empty()) continue;
        const std::string onnx = model_dir_ + "/" + m.name + ".onnx";
        std::vector<char> bytes;
        if (!ajicore::read_file_bytes(onnx, &bytes)) return passthrough(m.name + ": the model file can't be read: passthrough");
        AjiOnnxInput oin;
        int t = aji_onnx_input(bytes.data(), bytes.size(), &oin) ? aji_onnx_temporal_frames(oin) : 1;
        if (m.frames > 0) t = m.frames;
        if (t > 1 && (t % 2 == 0 || t > AJI_TEMPORAL_MAX_FRAMES))
            return passthrough(m.name + ": temporal models need an odd frame count up to 15: passthrough");
        // resizes before upscale: the factor, then the height (fitted to 16:9), as aji_dml.cpp's
        const auto resize_to = [&](int nw, int nh, const char *what) {
            Step r;
            r.in_w = cw;
            r.in_h = ch;
            r.out_w = nw;
            r.out_h = nh;
            if (steps_.empty() && &m == &chain->models.front()) first_rs_log = (int)log_steps_.size();
            steps_.push_back(std::move(r));
            cw = nw;
            ch = nh;
            char buf[200];
            std::snprintf(buf, sizeof buf, what, m.resize_height_before_upscale > 0 ? m.resize_height_before_upscale : m.resize_factor_before_upscale,
                          cw, ch);
            log_steps_.push_back(buf);
        };
        if (m.resize_height_before_upscale == 0 && m.resize_factor_before_upscale != 100.0) {
            const int nw = round_even(cw * m.resize_factor_before_upscale / 100.0), nh = round_even(ch * m.resize_factor_before_upscale / 100.0);
            if (nw >= 2 && nh >= 2 && (nw != cw || nh != ch))
                resize_to(nw, nh, "Applied Resize Factor Before Upscale: %g%%;    New Video Resolution: %dx%d");
        }
        if (m.resize_height_before_upscale != 0 && (int)m.resize_height_before_upscale != ch) {
            int nw, nh;
            fit_box(cw, ch, m.resize_height_before_upscale * 16.0 / 9.0, m.resize_height_before_upscale, &nw, &nh);
            resize_to(nw, nh, "Applied Resize Height Before Upscale: %gpx;    New Video Resolution: %dx%d");
        }
        std::string why;
        ajinative::OnnxGraph g;
        if (t > 1) {
            // opened at the input padded to 32 (the exported graphs' alignment), the output cropped
            const int bw = (cw + 31) / 32 * 32, bh = (ch + 31) / 32 * 32;
            std::unique_ptr<TemporalNet> tn;
            if (ajinative::parse_onnx(bytes.data(), bytes.size(), &g, &why)) tn = TemporalNet::create(*dev_, g, bw, bh, &why);
            if (tn && tn->frames() != t) {
                why = "the graph's window is not the model input's";
                tn.reset();
            }
            if (!tn)
                return passthrough("Native kernels not used for " + m.name + ": " + why + "; Vulkan frames have no DirectML to fall back to: passthrough");
            log_steps_.push_back("Native kernels: " + m.name + " (" + tn->describe() + ")");
            Stage T;
            T.t = t;
            T.r = (t - 1) / 2;
            T.step = steps_.size();
            T.w = cw;
            T.h = ch;
            T.pw = bw;
            T.ph = bh;
            T.pow = bw * tn->scale();
            T.poh = bh * tn->scale();
            T.ow = cw * tn->scale();
            T.oh = ch * tn->scale();
            T.threshold = (float)m.temporal_scene_threshold;
            Step s;
            s.in_w = cw;
            s.in_h = ch;
            s.out_w = T.ow;
            s.out_h = T.oh;
            s.tnet = std::move(tn);
            s.stage = (int)stages_.size();
            cw = T.ow;
            ch = T.oh;
            char buf[256];
            if (T.threshold > 0)
                std::snprintf(buf, sizeof buf, "Applied Temporal Model: %s (%d frames, scene threshold %g);    New Video Resolution: %dx%d",
                              m.name.c_str(), t, T.threshold, cw, ch);
            else
                std::snprintf(buf, sizeof buf, "Applied Temporal Model: %s (%d frames);    New Video Resolution: %dx%d", m.name.c_str(), t, cw, ch);
            log_steps_.push_back(buf);
            stages_.push_back(std::move(T));
            steps_.push_back(std::move(s));
            continue;
        }
        std::unique_ptr<ImgNet> net;
        if (ajinative::parse_onnx(bytes.data(), bytes.size(), &g, &why)) {
            ajinative::Plan plan;
            if (ajinative::lower(g, &plan, &why)) {
                net = ImgNet::lowered(*dev_, plan, cw, ch, &why);
            } else {
                const std::string wl = why;
                net = ImgNet::frame(*dev_, g, cw, ch, &why);
                if (!net && why.rfind("not an ESRGAN", 0) == 0) why = wl;  // neither: the lowering's reason
            }
        }
        if (!net)
            return passthrough("Native kernels not used for " + m.name + ": " + why + "; Vulkan frames have no DirectML to fall back to: passthrough");
        log_steps_.push_back("Native kernels: " + m.name + " (" + net->describe() + ")");
        Step s;
        s.in_w = cw;
        s.in_h = ch;
        s.out_w = net->out_w();
        s.out_h = net->out_h();
        s.net = std::move(net);
        cw = s.out_w;
        ch = s.out_h;
        char buf[256];
        std::snprintf(buf, sizeof buf, "Applied Model: %s;    New Video Resolution: %dx%d", m.name.c_str(), cw, ch);
        log_steps_.push_back(buf);
        steps_.push_back(std::move(s));
    }
    if (steps_.empty()) {
        if (!chain->rife) return passthrough("");
        if (rife_dir_.empty()) return passthrough("RIFE requested by the chain but no rife model dir is configured; interpolation disabled");
        // RIFE alone: no upscale (0, as the D3D12 path), the interpolation at the source size
        teardown();
        rife_.before = chain->rife_before_upscale;
        rife_setup(*chain, w, h, fps);
        finalize_log();
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;
    }
    if ((cw & 1) || (ch & 1)) return passthrough("the chain's output size is odd: passthrough");
    if (chain->rife) {
        if (rife_dir_.empty()) {
            log_steps_.push_back("RIFE requested by the chain but no rife model dir is configured; interpolation disabled");
        } else {
            // RIFE-first would feed interpolated frames into a temporal model's window: interpolate
            // the upscaled frames instead (as aji_dml.cpp)
            rife_.before = chain->rife_before_upscale && stages_.empty();
            if (chain->rife_before_upscale && !stages_.empty())
                log_steps_.push_back("RIFE before upscale is not supported with a temporal model; interpolating after upscale");
            // RIFE-first: the first model's explicit resize is hoisted ahead of RIFE (the caller
            // applies it to source frames with aji_resize), so RIFE and the rest run at the work size
            if (rife_.before && first_rs_log >= 0 && steps_.size() > 1) {
                PreResize &P = pre_;
                P.sw = w;
                P.sh = h;
                P.ww = steps_[0].out_w;
                P.wh = steps_[0].out_h;
                std::string why2;
                if (!dev_->make_buffer(&P.a, (VkDeviceSize)3 * P.sw * P.sh * 2, false, &why2) ||
                    !dev_->make_buffer(&P.b, (VkDeviceSize)3 * P.ww * P.wh * 2, false, &why2) ||
                    !(P.rs = RgbResize::create(*dev_, P.sw, P.sh, P.ww, P.wh, P.a.buf, P.b.buf, &why2))) {
                    fail("pre-RIFE resize %dx%d -> %dx%d: %s", P.sw, P.sh, P.ww, P.wh, why2.c_str());
                    teardown();
                    return AJI_ERR;
                }
                P.on = true;
                steps_.erase(steps_.begin());
                log_steps_.erase(log_steps_.begin() + first_rs_log);
                char buf[160];
                std::snprintf(buf, sizeof buf, "Applied Resize Before Upscale: %dx%d (before RIFE)", P.ww, P.wh);
                log_steps_.insert(log_steps_.begin(), buf);
                in_w_ = P.ww;
                in_h_ = P.wh;
            }
            rife_setup(*chain, rife_.before ? in_w_ : cw, rife_.before ? in_h_ : ch, fps);
        }
    }
    // the chain buffers: every step reads one and writes the other
    VkDeviceSize most = 0;
    for (const Step &s : steps_) most = std::max({most, (VkDeviceSize)3 * s.in_w * s.in_h * 2, (VkDeviceSize)3 * s.out_w * s.out_h * 2});
    std::string why;
    if (!dev_->make_buffer(&chain_[0], most, false, &why) || !dev_->make_buffer(&chain_[1], most, false, &why)) {
        fail("chain buffers: %s", why.c_str());
        teardown();
        return AJI_ERR;
    }
    for (size_t k = 0; k < steps_.size(); k++) {
        Step &s = steps_[k];
        if (s.net) {
            s.net->bind_io(chain_[k % 2].buf, chain_[(k + 1) % 2].buf);
        } else if (s.tnet) {
            // its own input (the gathered window) and output (cropped into chain_[(k + 1) % 2])
        } else if (!(s.rs = RgbResize::create(*dev_, s.in_w, s.in_h, s.out_w, s.out_h, chain_[k % 2].buf, chain_[(k + 1) % 2].buf, &why))) {
            fail("resize %dx%d -> %dx%d: %s", s.in_w, s.in_h, s.out_w, s.out_h, why.c_str());
            teardown();
            return AJI_ERR;
        }
    }
    if (!stages_.empty() && !temporal_setup(&why)) {
        fail("temporal setup: %s", why.c_str());
        teardown();
        return AJI_ERR;
    }
    dev_->save_caches();
    active_ = true;
    out_w_ = cw;
    out_h_ = ch;
    finalize_log();
    if (out_w) *out_w = cw;
    if (out_h) *out_h = ch;
    return 1;
}

int VkVideo::infer(const aji_frame *in, const aji_frame *out)
{
    if (!active_) {
        fail("aji_infer: no chain is configured");
        return AJI_ERR;
    }
    if (!stages_.empty()) {
        fail("aji_infer: the chain has a temporal model: drive it with aji_ingest + aji_infer_seq");
        return AJI_ERR;
    }
    if (!in || !out || !in->plane[0] || !out->plane[0]) {
        fail("aji_infer: a Vulkan frame needs an aji_vk_image in plane[0]");
        return AJI_ERR;
    }
    if (in->width != in_w_ || in->height != in_h_ || out->width != out_w_ || out->height != out_h_) {
        fail("aji_infer: frame sizes %dx%d -> %dx%d, configured %dx%d -> %dx%d", in->width, in->height, out->width, out->height, in_w_,
             in_h_, out_w_, out_h_);
        return AJI_ERR_SHAPE;
    }
    for (const aji_frame *f : {in, out})
        if (f->format != AJI_FMT_NV12 && f->format != AJI_FMT_P010) {
            fail("aji_infer: Vulkan frames are NV12 or P010");
            return AJI_ERR_FORMAT;
        }
    auto *ii = static_cast<aji_vk_image *>(in->plane[0]);
    auto *oi = static_cast<aji_vk_image *>(out->plane[0]);
    if (same_image(ii, oi)) {  // the output is discarded before the input is read
        fail("aji_infer: the input and output must be different images");
        return AJI_ERR;
    }
    const FrameFormat fi{in->format, in->matrix, in->range, in->siting}, fo{out->format, out->matrix, out->range, AJI_SITING_LEFT};
    std::string why;
    if (!io_ || !(io_->in_format() == fi) || !(io_->out_format() == fo)) {
        if (io_ && ticket_) wait(ticket_);
        io_ = FrameIO::create(*dev_, in_w_, in_h_, out_w_, out_h_, fi, fo, chain_[0].buf, chain_[steps_.size() % 2].buf, &why);
        if (!io_) {
            fail("aji_infer: %s", why.c_str());
            return AJI_ERR;
        }
    }
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_infer: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    // the images into transfer layouts (the output's contents are replaced: from UNDEFINED)
    VkImageMemoryBarrier ib[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    ib[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ib[0].oldLayout = static_cast<VkImageLayout>(ii->layout);
    ib[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ib[0].image = static_cast<VkImage>(ii->image);
    ib[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib[1].image = static_cast<VkImage>(oi->image);
    for (auto &b : ib) {
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    set_owner(ib[0], ii, dev_->queue_family(), true);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, ib);
    const uint32_t ibps = in->format == AJI_FMT_P010 ? 2 : 1, obps = out->format == AJI_FMT_P010 ? 2 : 1;
    const auto region = [](uint32_t row_texels, VkImageAspectFlagBits plane, int w, int h) {
        VkBufferImageCopy r{};
        r.bufferRowLength = row_texels;
        r.imageSubresource = {(VkImageAspectFlags)plane, 0, 0, 1};
        r.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
        return r;
    };
    const VkBufferImageCopy ry = region(io_->in_pitch() / ibps, VK_IMAGE_ASPECT_PLANE_0_BIT, in_w_, in_h_);
    const VkBufferImageCopy ruv = region(io_->in_pitch() / (2 * ibps), VK_IMAGE_ASPECT_PLANE_1_BIT, in_w_ / 2, in_h_ / 2);
    vkCmdCopyImageToBuffer(cb, ib[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, io_->in_y().buf, 1, &ry);
    vkCmdCopyImageToBuffer(cb, ib[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, io_->in_uv().buf, 1, &ruv);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
    // pre, the chain, post
    io_->record_pre(cb);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    for (const Step &s : steps_) {
        if (s.rs) {
            s.rs->record(cb);  // ends with its barrier
            continue;
        }
        (void)s.net->record(cb, s.in_w, s.in_h);  // the sizes it was created for: it records
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    io_->record_post(cb);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const VkBufferImageCopy oy = region(io_->out_pitch() / obps, VK_IMAGE_ASPECT_PLANE_0_BIT, out_w_, out_h_);
    const VkBufferImageCopy ouv = region(io_->out_pitch() / (2 * obps), VK_IMAGE_ASPECT_PLANE_1_BIT, out_w_ / 2, out_h_ / 2);
    vkCmdCopyBufferToImage(cb, io_->out_y().buf, ib[1].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &oy);
    vkCmdCopyBufferToImage(cb, io_->out_uv().buf, ib[1].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ouv);
    // the images' semaphores and the ticket (FrameSems: one signal for an input and output on one timeline)
    aji_vk_image *const both[2] = {ii, oi};
    const uint64_t t = ticket_ + 1;
    const FrameSems fs(tl_, t, both, 2);
    if (!dev_->frame_submit(slot, fs.waits, fs.nwait, fs.sig, fs.nsig(), &why)) {
        fail("aji_infer: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_ = t;
    fs.commit(both, 2);
    ii->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ii->access = VK_ACCESS_2_TRANSFER_READ_BIT;
    ii->queue_family = owner_after(ii, dev_->queue_family());
    oi->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    oi->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    oi->queue_family = owner_after(oi, dev_->queue_family());
    return AJI_OK;
}

// ---- temporal models ----------------------------------------------------------------------------

namespace {

struct KArgs {
    int32_t di[4], dj[4];
    float kr, kb, yoff, yscale, coff, cscale, qdiv, qmax;
};

void mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = sa;
    mb.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

constexpr VkPipelineStageFlags kCT = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr VkAccessFlags kRW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

// everything before (this and earlier submits) visible to everything after
void full_barrier(VkCommandBuffer cb)
{
    mem_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, kCT, kRW);
}

void step_barrier(VkCommandBuffer cb) { mem_barrier(cb, kCT, kRW, kCT, kRW); }

// an image's planes into a FrameIO's input buffers, or its output buffers into an image
void copy_planes(VkCommandBuffer cb, aji_vk_image *im, VkBuffer y, VkBuffer uv, uint32_t pitch, int w, int h, uint32_t bps, bool in,
                 uint32_t family)
{
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.dstAccessMask = in ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.oldLayout = in ? static_cast<VkImageLayout>(im->layout) : VK_IMAGE_LAYOUT_UNDEFINED;  // the output is replaced
    ib.newLayout = in ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    set_owner(ib, im, family, in);
    ib.image = static_cast<VkImage>(im->image);
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    VkBufferImageCopy r[2] = {};
    r[0].bufferRowLength = pitch / bps;
    r[0].imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
    r[0].imageExtent = {(uint32_t)w, (uint32_t)h, 1};
    r[1].bufferRowLength = pitch / (2 * bps);
    r[1].imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
    r[1].imageExtent = {(uint32_t)w / 2, (uint32_t)h / 2, 1};
    const VkImage img = static_cast<VkImage>(im->image);
    if (in) {
        vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, y, 1, &r[0]);
        vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, uv, 1, &r[1]);
    } else {
        vkCmdCopyBufferToImage(cb, y, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r[0]);
        vkCmdCopyBufferToImage(cb, uv, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r[1]);
    }
}

}  // namespace

void VkVideo::temporal_teardown()
{
    io_pre_.reset();
    for (Stage &T : stages_)
        for (Buffer *b : {&T.ring, &T.diff, &T.part}) dev_->destroy(*b);
    stages_.clear();
    if (tpool_) vkDestroyDescriptorPool(dev_->device(), tpool_, nullptr);
    tpool_ = VK_NULL_HANDLE;
    dev_->destroy(tparams_);
    temporal_r_ = 0;
}

// The rings, metrics, parameter entries and descriptor sets of the configured temporal steps.
bool VkVideo::temporal_setup(std::string *why)
{
    temporal_r_ = 0;
    for (const Stage &T : stages_) temporal_r_ += T.r;
    if (2 * temporal_r_ + 5 > 32) return *why = "the temporal models' combined window is too large (ring slots are 5 bits)", false;
    static const bool fcache = !getenv("AJI_NATIVE_FRAME_CACHE") || std::strcmp(getenv("AJI_NATIVE_FRAME_CACHE"), "off") != 0;
    for (size_t lv = 0; lv < stages_.size(); lv++) {
        Stage &T = stages_[lv];
        TemporalNet &net = *steps_[T.step].tnet;
        // stage 0 keeps the last 2R+5 ingested (callers ingest up to R+4 ahead); later stages
        // are filled on demand and need only their window
        T.nslots = lv == 0 ? 2 * temporal_r_ + 5 : T.t + 4;
        T.part_groups = ((T.w + 31) / 32) * ((T.h + 7) / 8);
        T.seq.assign(T.nslots, -1);
        T.format.assign(T.nslots, 0);
        T.matrix.assign(T.nslots, 0);
        T.range.assign(T.nslots, 0);
        if (!dev_->make_buffer(&T.ring, (VkDeviceSize)T.nslots * 3 * T.w * T.h * 2, false, why) ||
            !dev_->make_buffer(&T.diff, (VkDeviceSize)T.nslots * 4, false, why) ||
            !dev_->make_buffer(&T.part, (VkDeviceSize)T.part_groups * 4, false, why))
            return false;
        if (fcache && net.cache_entries() >= T.t) {
            net.bind_diff(T.diff.buf);
            T.cache = true;
            T.gen.assign(T.nslots, 0);
            T.cgen.assign(net.cache_entries(), 0);
        }
    }
    const char *mods[3] = {"fr_tdiff", "fr_scd2", "fr_tgather"};
    for (int k = 0; k < 3; k++)
        if (!(tpipes_[k] = dev_->pipeline(mods[k], 0, 0, why))) return false;
    if (!dev_->make_buffer(&tparams_, (VkDeviceSize)stages_.size() * 3 * sizeof(KArgs), false, why)) return false;
    const uint32_t nsets = (uint32_t)stages_.size() * 3;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nsets * Device::kBindings};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = nsets;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(dev_->device(), &pi, nullptr, &tpool_) != VK_SUCCESS) return *why = "descriptor pool", false;
    for (Stage &T : stages_) {
        const VkBuffer d[3][3] = {{T.ring.buf, VK_NULL_HANDLE, T.part.buf},
                                  {T.part.buf, T.diff.buf, VK_NULL_HANDLE},
                                  {T.ring.buf, T.diff.buf, steps_[T.step].tnet->input().buf}};
        for (int k = 0; k < 3; k++) {
            VkDescriptorSetLayout l = dev_->set_layout();
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = tpool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &l;
            if (vkAllocateDescriptorSets(dev_->device(), &ai, &T.sets[k]) != VK_SUCCESS) return *why = "descriptor set", false;
            VkDescriptorBufferInfo bi[Device::kBindings];
            for (auto &b : bi) b = {dev_->dummy(), 0, VK_WHOLE_SIZE};
            for (int j = 0; j < 3; j++)
                if (d[k][j]) bi[2 + j] = {d[k][j], 0, VK_WHOLE_SIZE};
            bi[7] = {tparams_.buf, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet wr[Device::kBindings];
            for (int j = 0; j < Device::kBindings; j++) {
                wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                wr[j].dstSet = T.sets[k];
                wr[j].dstBinding = (uint32_t)j;
                wr[j].descriptorCount = 1;
                wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wr[j].pBufferInfo = &bi[j];
            }
            vkUpdateDescriptorSets(dev_->device(), Device::kBindings, wr, 0, nullptr);
        }
    }
    return true;
}

int VkVideo::tslot(size_t lv, int64_t seq) const
{
    const Stage &T = stages_[lv];
    if (seq < 0) return -1;
    const int slot = (int)(seq % T.nslots);
    return T.seq[slot] == seq ? slot : -1;
}

// One parameter entry, in the command buffer: after every earlier read of it, before what follows.
void VkVideo::set_param(VkCommandBuffer cb, uint32_t entry, const void *ka) const
{
    mem_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cb, tparams_.buf, (VkDeviceSize)entry * sizeof(KArgs), sizeof(KArgs), ka);
    mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// steps [from, to) of the chain (no temporal step among them), each after a barrier
void VkVideo::rec_steps(VkCommandBuffer cb, size_t from, size_t to) const
{
    for (size_t k = from; k < to; k++) {
        const Step &s = steps_[k];
        step_barrier(cb);
        if (s.rs) s.rs->record(cb);
        else (void)s.net->record(cb, s.in_w, s.in_h);
    }
    step_barrier(cb);
}

// The step's input (chain_[step % 2]) as `seq` into stage lv's ring, and its cut metric against
// seq - 1 when that is held.
void VkVideo::rec_store(VkCommandBuffer cb, size_t lv, int64_t seq, int format, int matrix, int range)
{
    Stage &T = stages_[lv];
    const int slot = (int)(seq % T.nslots);
    const uint32_t fe = 3u * (uint32_t)T.w * (uint32_t)T.h;  // frame elements
    const VkBufferCopy c{0, (VkDeviceSize)slot * fe * 2, (VkDeviceSize)fe * 2};
    vkCmdCopyBuffer(cb, chain_[T.step % 2].buf, T.ring.buf, 1, &c);
    step_barrier(cb);
    const int prev = tslot(lv, seq - 1);
    const uint32_t base = (uint32_t)lv * 3;
    if (prev >= 0) {
        KArgs a{};
        a.di[0] = T.w;
        a.di[1] = T.h;
        a.di[2] = (T.w + 31) / 32;
        a.dj[0] = (int32_t)((uint32_t)prev * fe);
        a.dj[1] = (int32_t)((uint32_t)slot * fe);
        set_param(cb, base, &a);
        KArgs f{};
        f.di[0] = T.part_groups;
        f.dj[0] = slot;
        set_param(cb, base + 1, &f);
        for (int k = 0; k < 2; k++) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tpipes_[k]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_->pipeline_layout(), 0, 1, &T.sets[k], 0, nullptr);
            const int32_t pc[4] = {(int32_t)(base + k), 0, 0, 0};
            vkCmdPushConstants(cb, dev_->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            if (k == 0) vkCmdDispatch(cb, (uint32_t)(T.w + 31) / 32, (uint32_t)(T.h + 7) / 8, 1);
            else vkCmdDispatch(cb, 1, 1, 1);
            step_barrier(cb);
        }
    } else {
        vkCmdFillBuffer(cb, T.diff.buf, (VkDeviceSize)slot * 4, 4, 0);
        step_barrier(cb);
    }
    T.seq[slot] = seq;
    if (T.cache) T.gen[slot] = gen_next_++;
    T.format[slot] = format;
    T.matrix[slot] = matrix;
    T.range[slot] = range;
}

// Stage lv's model on the window around `center` (held in cslot), its output cropped into
// chain_[(step + 1) % 2]. Window frames not held take the nearest held one toward the center; a
// cut metric counts only between two consecutive held frames (aji_dml.cpp's rules).
void VkVideo::rec_stage(VkCommandBuffer cb, size_t lv, int64_t center, int cslot)
{
    Stage &T = stages_[lv];
    TemporalNet &net = *steps_[T.step].tnet;
    int slots[AJI_TEMPORAL_MAX_FRAMES];
    bool held[AJI_TEMPORAL_MAX_FRAMES];
    slots[T.r] = cslot;
    held[T.r] = true;
    for (int dir = -1; dir <= 1; dir += 2) {
        bool clamped = false;
        for (int k = 1; k <= T.r; k++) {
            const int j = T.r + dir * k;
            const int s = clamped ? -1 : tslot(lv, center + dir * k);
            clamped = s < 0;
            slots[j] = clamped ? slots[j - dir] : s;
            held[j] = !clamped;
        }
    }
    uint32_t packed[3] = {0, 0, 0}, cut_mask = 0;
    for (int j = 0; j < T.t; j++) {
        packed[j / 6] |= (uint32_t)slots[j] << ((j % 6) * 5);
        if (j + 1 < T.t && held[j] && held[j + 1]) cut_mask |= 1u << j;
    }
    KArgs a{};
    a.di[0] = T.w;
    a.di[1] = T.h;
    a.di[2] = T.pw;
    a.di[3] = T.ph;
    a.kb = 1.0f / ((float)T.w * (float)T.h);
    const uint32_t entry = (uint32_t)lv * 3 + 2;
    const auto gather = [&](int n) {
        set_param(cb, entry, &a);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tpipes_[2]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_->pipeline_layout(), 0, 1, &T.sets[2], 0, nullptr);
        const int32_t pc[4] = {(int32_t)entry, 0, 0, 0};
        vkCmdPushConstants(cb, dev_->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, (uint32_t)(T.pw / 2 + 31) / 32, (uint32_t)(T.ph + 7) / 8, (uint32_t)n);
        step_barrier(cb);
    };
    if (T.cache) {
        // per-frame work kept across windows: each window frame not in the model's cache is gathered
        // into the next free frame of its input and encoded into an entry this window doesn't hold;
        // the model then picks each position's entry under the gather's cut rule
        int ent[AJI_TEMPORAL_MAX_FRAMES], cut[AJI_TEMPORAL_MAX_FRAMES], miss_e[AJI_TEMPORAL_MAX_FRAMES];
        uint32_t mpack[3] = {0, 0, 0};
        int nm = 0;
        for (int j = 0; j < T.t; j++) {
            const uint64_t g = T.gen[slots[j]];
            int e = -1;
            for (int k = 0; k < (int)T.cgen.size() && e < 0; k++)
                if (T.cgen[k] == g) e = k;
            if (e < 0) {
                uint64_t oldest = UINT64_MAX;
                for (int k = 0; k < (int)T.cgen.size(); k++) {
                    bool used = false;
                    for (int i = 0; i < T.t; i++) used = used || T.cgen[k] == T.gen[slots[i]];
                    if (!used && T.cgen[k] < oldest) {
                        oldest = T.cgen[k];
                        e = k;
                    }
                }
                T.cgen[e] = g;
                mpack[nm / 6] |= (uint32_t)slots[j] << ((nm % 6) * 5);
                miss_e[nm++] = e;
            }
            ent[j] = e;
            cut[j] = (cut_mask >> j) & 1u ? slots[j + 1] : -1;
        }
        if (nm) {
            a.dj[0] = nm;  // the misses in order, no cut substitution
            a.dj[1] = (int32_t)mpack[0];
            a.dj[2] = (int32_t)mpack[1];
            a.dj[3] = (int32_t)mpack[2];
            a.kr = 0.0f;
            gather(nm);
            for (int i = 0; i < nm; i++) net.record_encode(cb, miss_e[i], i);
        }
        step_barrier(cb);
        net.record_cached(cb, ent, cut, T.threshold, 1.0f / ((float)T.w * (float)T.h));
    } else {
        a.dj[0] = T.t;
        a.dj[1] = (int32_t)packed[0];
        a.dj[2] = (int32_t)packed[1];
        a.dj[3] = (int32_t)packed[2];
        a.kr = T.threshold;
        std::memcpy(&a.qdiv, &cut_mask, 4);
        gather(T.t);
        net.record(cb);
    }
    step_barrier(cb);
    // crop the padded output (pow x poh) into the step's output (ow x oh), fp16 planes
    const VkBuffer dst = chain_[(T.step + 1) % 2].buf;
    std::vector<VkBufferCopy> rc;
    if (T.pow == T.ow) {
        for (int pl = 0; pl < 3; pl++)
            rc.push_back({(VkDeviceSize)pl * T.pow * T.poh * 2, (VkDeviceSize)pl * T.ow * T.oh * 2, (VkDeviceSize)T.ow * T.oh * 2});
    } else {
        for (int pl = 0; pl < 3; pl++)
            for (int y = 0; y < T.oh; y++)
                rc.push_back({((VkDeviceSize)pl * T.poh + y) * T.pow * 2, ((VkDeviceSize)pl * T.oh + y) * T.ow * 2, (VkDeviceSize)T.ow * 2});
    }
    vkCmdCopyBuffer(cb, net.output().buf, dst, (uint32_t)rc.size(), rc.data());
    step_barrier(cb);
}

// Stage lv's ring to hold frame p if it can: stage lv-1 on its window around p (those frames made
// held first, oldest first) and the steps between the two. Not when the source frame was never
// ingested: the frame stays absent and windows clamp around it.
void VkVideo::rec_ensure(VkCommandBuffer cb, size_t lv, int64_t p)
{
    if (p < 0 || lv == 0 || tslot(lv, p) >= 0) return;
    const Stage &S = stages_[lv - 1];
    for (int64_t q = p - S.r; q <= p + S.r; q++) rec_ensure(cb, lv - 1, q);
    const int cslot = tslot(lv - 1, p);
    if (cslot < 0) return;
    rec_stage(cb, lv - 1, p, cslot);
    rec_steps(cb, S.step + 1, stages_[lv].step);
    rec_store(cb, lv, p, S.format[cslot], S.matrix[cslot], S.range[cslot]);
}

void VkVideo::temporal_reset()
{
    for (Stage &T : stages_) std::fill(T.seq.begin(), T.seq.end(), -1);
}

int VkVideo::ingest(const aji_frame *in, uint64_t seq)
{
    if (!active_ || stages_.empty()) {
        fail("aji_ingest without an active temporal configuration");
        return AJI_ERR;
    }
    if (!in || !in->plane[0] || seq > (uint64_t)INT64_MAX) return AJI_ERR;
    if (in->width != in_w_ || in->height != in_h_) {
        fail("aji_ingest: frame size %dx%d, configured %dx%d", in->width, in->height, in_w_, in_h_);
        return AJI_ERR_SHAPE;
    }
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010) {
        fail("aji_ingest: Vulkan frames are NV12 or P010");
        return AJI_ERR_FORMAT;
    }
    if (tslot(0, (int64_t)seq) >= 0) return AJI_OK;  // already held
    std::string why;
    const FrameFormat fi{in->format, in->matrix, in->range, in->siting};
    if (!io_pre_ || !(io_pre_->in_format() == fi)) {
        if (io_pre_ && ticket_) wait(ticket_);
        const FrameFormat fo{in->format, in->matrix, in->range, AJI_SITING_LEFT};
        io_pre_ = FrameIO::create(*dev_, in_w_, in_h_, 2, 2, fi, fo, chain_[0].buf, chain_[0].buf, &why);
        if (!io_pre_) {
            fail("aji_ingest: %s", why.c_str());
            return AJI_ERR;
        }
    }
    auto *ii = static_cast<aji_vk_image *>(in->plane[0]);
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_ingest: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    full_barrier(cb);
    copy_planes(cb, ii, io_pre_->in_y().buf, io_pre_->in_uv().buf, io_pre_->in_pitch(), in_w_, in_h_, in->format == AJI_FMT_P010 ? 2 : 1, true, dev_->queue_family());
    step_barrier(cb);
    io_pre_->record_pre(cb);
    rec_steps(cb, 0, stages_[0].step);
    rec_store(cb, 0, (int64_t)seq, in->format, in->matrix, in->range);
    const Device::SemValue w1{static_cast<VkSemaphore>(ii->semaphore), ii->value};
    const uint64_t t = ticket_ + 1;
    const Device::SemValue sig[2] = {{tl_, t}, {static_cast<VkSemaphore>(ii->semaphore), ii->value + 1}};
    if (!dev_->frame_submit(slot, &w1, 1, sig, 2, &why)) {
        fail("aji_ingest: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_ = t;
    ii->value += 1;
    ii->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ii->access = VK_ACCESS_2_TRANSFER_READ_BIT;
    ii->queue_family = owner_after(ii, dev_->queue_family());
    return AJI_OK;
}

int VkVideo::infer_seq(uint64_t seq, const aji_frame *out)
{
    if (!active_ || stages_.empty()) {
        fail("aji_infer_seq without an active temporal configuration");
        return AJI_ERR;
    }
    if (!out || !out->plane[0]) return AJI_ERR;
    const int64_t center = (int64_t)seq;
    const int islot = seq > (uint64_t)INT64_MAX ? -1 : tslot(0, center);
    if (islot < 0) {
        fail("aji_infer_seq: frame %llu was not ingested", (unsigned long long)seq);
        return AJI_ERR;
    }
    if (out->width != out_w_ || out->height != out_h_) {
        fail("aji_infer_seq: frame size %dx%d, configured %dx%d", out->width, out->height, out_w_, out_h_);
        return AJI_ERR_SHAPE;
    }
    if (out->format != AJI_FMT_NV12 && out->format != AJI_FMT_P010) {
        fail("aji_infer_seq: Vulkan frames are NV12 or P010");
        return AJI_ERR_FORMAT;
    }
    const Stage &S0 = stages_[0];
    const FrameFormat fi{S0.format[islot], S0.matrix[islot], S0.range[islot], AJI_SITING_LEFT};
    const FrameFormat fo{out->format, out->matrix, out->range, AJI_SITING_LEFT};
    std::string why;
    if (!io_ || !(io_->out_format() == fo)) {
        if (io_ && ticket_) wait(ticket_);
        io_ = FrameIO::create(*dev_, in_w_, in_h_, out_w_, out_h_, fi, fo, chain_[0].buf, chain_[steps_.size() % 2].buf, &why);
        if (!io_) {
            fail("aji_infer_seq: %s", why.c_str());
            return AJI_ERR;
        }
    }
    auto *oi = static_cast<aji_vk_image *>(out->plane[0]);
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_infer_seq: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    full_barrier(cb);
    // stacked temporal models: the last stage's window brought up to date
    const size_t last = stages_.size() - 1;
    const Stage &T = stages_[last];
    for (int64_t q = center - T.r; last > 0 && q <= center + T.r; q++) rec_ensure(cb, last, q);
    const int cslot = tslot(last, center);
    if (cslot < 0) {
        // what was recorded (the earlier stages' stores) still runs: the rings' state says it did
        const Device::SemValue s1{tl_, ticket_ + 1};
        if (dev_->frame_submit(slot, nullptr, 0, &s1, 1, &why)) ticket_++;
        fail("aji_infer_seq: frame %llu is not available", (unsigned long long)seq);
        return AJI_ERR;
    }
    rec_stage(cb, last, center, cslot);
    rec_steps(cb, T.step + 1, steps_.size());
    io_->record_post(cb);
    mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    copy_planes(cb, oi, io_->out_y().buf, io_->out_uv().buf, io_->out_pitch(), out_w_, out_h_, out->format == AJI_FMT_P010 ? 2 : 1, false, dev_->queue_family());
    const Device::SemValue w1{static_cast<VkSemaphore>(oi->semaphore), oi->value};
    const uint64_t t = ticket_ + 1;
    const Device::SemValue sig[2] = {{tl_, t}, {static_cast<VkSemaphore>(oi->semaphore), oi->value + 1}};
    if (!dev_->frame_submit(slot, &w1, 1, sig, 2, &why)) {
        fail("aji_infer_seq: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_ = t;
    oi->value += 1;
    oi->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    oi->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    oi->queue_family = owner_after(oi, dev_->queue_family());
    return AJI_OK;
}

// ---- aji_frame_diff ------------------------------------------------------------------------------

void VkVideo::diff_teardown()
{
    DiffState &D = diff_;
    if (D.pool) vkDestroyDescriptorPool(dev_->device(), D.pool, nullptr);
    for (Buffer *b : {&D.y[0], &D.y[1], &D.uv[0], &D.uv[1], &D.acc, &D.read, &D.weights, &D.params}) dev_->destroy(*b);
    D = DiffState{};
}

bool VkVideo::diff_setup(int w, int h, int fmt, std::string *why)
{
    DiffState &D = diff_;
    if (D.pool && D.w == w && D.h == h && D.fmt == fmt) return true;
    if (ticket_) wait(ticket_);
    diff_teardown();
    const uint32_t bps = fmt == AJI_FMT_P010 ? 2 : 1;
    D.pitch = ((uint32_t)w * bps + 255) & ~255u;
    for (int i = 0; i < 2; i++)
        if (!dev_->make_buffer(&D.y[i], (VkDeviceSize)D.pitch * h, false, why) ||
            !dev_->make_buffer(&D.uv[i], (VkDeviceSize)D.pitch * ((h + 1) / 2), false, why))
            return false;
    // weights[n] = scale / n for a tile of n pixels, as every backend computes it
    float inv[65] = {0.0f};
    const float scale = aji_frame_diff_scale(fmt);
    for (int n = 1; n <= 64; n++) inv[n] = aji_frame_diff_tile_inv(scale, n);
    struct KA {
        int32_t di[4], dj[4];
        float f[8];
    } ka[2] = {};
    for (int p = 0; p < 2; p++) {
        // 4:2:0's second plane is CbCr interleaved at half size
        ka[p].di[0] = p ? (w + 1) / 2 : w;
        ka[p].di[1] = p ? (h + 1) / 2 : h;
        ka[p].di[2] = p ? 2 : 1;
        ka[p].di[3] = p == 0;
        ka[p].dj[0] = ka[p].dj[1] = (int32_t)D.pitch;
        ka[p].dj[2] = bps == 2 ? 1 : 0;
    }
    if (!dev_->make_buffer(&D.acc, 33 * 4, false, why) || !dev_->make_buffer(&D.read, 33 * 4, true, why) ||
        !dev_->make_buffer(&D.weights, sizeof inv, false, why) || !dev_->upload(D.weights, inv, sizeof inv, why) ||
        !dev_->make_buffer(&D.params, sizeof ka, false, why) || !dev_->upload(D.params, ka, sizeof ka, why))
        return false;
    if (!(D.pipe = dev_->pipeline("fr_diff", 0, 0, why))) return false;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * Device::kBindings};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 2;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(dev_->device(), &pi, nullptr, &D.pool) != VK_SUCCESS) return *why = "descriptor pool", false;
    for (int p = 0; p < 2; p++) {
        VkDescriptorSetLayout l = dev_->set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = D.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev_->device(), &ai, &D.sets[p]) != VK_SUCCESS) return *why = "descriptor set", false;
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev_->dummy(), 0, VK_WHOLE_SIZE};
        bi[1] = {D.weights.buf, 0, VK_WHOLE_SIZE};
        bi[2] = {p ? D.uv[0].buf : D.y[0].buf, 0, VK_WHOLE_SIZE};
        bi[3] = {p ? D.uv[1].buf : D.y[1].buf, 0, VK_WHOLE_SIZE};
        bi[4] = {D.acc.buf, 0, VK_WHOLE_SIZE};
        bi[7] = {D.params.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = D.sets[p];
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev_->device(), Device::kBindings, wr, 0, nullptr);
    }
    D.w = w;
    D.h = h;
    D.fmt = fmt;
    return true;
}

int VkVideo::frame_diff(const aji_frame *a, const aji_frame *b, aji_frame_diff_result *out)
{
    if (!a || !b || !out || !a->plane[0] || !b->plane[0]) return AJI_ERR;
    const int fmt = a->format;
    if (fmt != AJI_FMT_NV12 && fmt != AJI_FMT_P010) {
        fail("aji_frame_diff: unsupported format %d (Vulkan frames are NV12 or P010)", fmt);
        return AJI_ERR_FORMAT;
    }
    if (b->format != fmt) {
        fail("aji_frame_diff requires matching formats");
        return AJI_ERR_FORMAT;
    }
    if (a->width != b->width || a->height != b->height || a->width < 2 || a->height < 2 || (a->width & 1) || (a->height & 1)) {
        fail("aji_frame_diff dims %dx%d vs %dx%d", a->width, a->height, b->width, b->height);
        return AJI_ERR_SHAPE;
    }
    if (same_image(static_cast<const aji_vk_image *>(a->plane[0]), static_cast<const aji_vk_image *>(b->plane[0]))) {
        // one image: no difference (and no second transition of it)
        *out = aji_frame_diff_result{};
        return AJI_OK;
    }
    std::string why;
    if (!diff_setup(a->width, a->height, fmt, &why)) {
        fail("aji_frame_diff: %s", why.c_str());
        diff_teardown();
        return AJI_ERR;
    }
    DiffState &D = diff_;
    aji_vk_image *im[2] = {static_cast<aji_vk_image *>(a->plane[0]), static_cast<aji_vk_image *>(b->plane[0])};
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_frame_diff: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    full_barrier(cb);
    const uint32_t bps = fmt == AJI_FMT_P010 ? 2 : 1;
    const int nimg = 2;
    for (int i = 0; i < 2; i++) copy_planes(cb, im[i], D.y[i].buf, D.uv[i].buf, D.pitch, a->width, a->height, bps, true, dev_->queue_family());
    vkCmdFillBuffer(cb, D.acc.buf, 0, 33 * 4, 0);
    step_barrier(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pipe);
    for (int p = 0; p < 2; p++) {
        const int pw = p ? (a->width + 1) / 2 : a->width, ph = p ? (a->height + 1) / 2 : a->height;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_->pipeline_layout(), 0, 1, &D.sets[p], 0, nullptr);
        const int32_t pc[4] = {p, 0, 0, 0};
        vkCmdPushConstants(cb, dev_->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        // a tile per invocation: (tiles + 31) / 32 x (tiles + 7) / 8 groups
        vkCmdDispatch(cb, (uint32_t)(((pw + 7) / 8 + 31) / 32), (uint32_t)(((ph + 7) / 8 + 7) / 8), 1);
    }
    mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const VkBufferCopy c{0, 0, 33 * 4};
    vkCmdCopyBuffer(cb, D.acc.buf, D.read.buf, 1, &c);
    mem_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    const FrameSems fs(tl_, ticket_ + 1, im, nimg);
    if (!dev_->frame_submit(slot, fs.waits, fs.nwait, fs.sig, fs.nsig(), &why)) {
        fail("aji_frame_diff: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_++;
    fs.commit(im, nimg);
    for (int i = 0; i < nimg; i++) {
        im[i]->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        im[i]->access = VK_ACCESS_2_TRANSFER_READ_BIT;
        im[i]->queue_family = owner_after(im[i], dev_->queue_family());
    }
    if (wait(ticket_) != AJI_OK) return AJI_ERR;
    uint32_t acc[33];
    std::memcpy(acc, D.read.map, sizeof acc);
    std::memcpy(&out->max_tile, &acc[0], sizeof(float));
    long long sum[16];
    for (int i = 0; i < 16; i++) sum[i] = (long long)(int32_t)acc[17 + i] * 65536 + (long long)acc[1 + i];
    aji_frame_diff_regions(a->width, a->height, fmt, 1, sum, out->region_mean);
    return AJI_OK;
}

// ---- RIFE-first's pre-RIFE resize -----------------------------------------------------------------

int VkVideo::pre_resize(int *work_w, int *work_h) const
{
    if (!active_ || !pre_.on) return 0;
    if (work_w) *work_w = pre_.ww;
    if (work_h) *work_h = pre_.wh;
    return 1;
}

int VkVideo::resize(const aji_frame *in, const aji_frame *out)
{
    PreResize &P = pre_;
    if (!active_ || !P.on) {
        fail("aji_resize without a configured pre-RIFE downscale");
        return AJI_ERR;
    }
    if (!in || !out || !in->plane[0] || !out->plane[0]) return AJI_ERR;
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010) {
        fail("unsupported input format %d", in->format);
        return AJI_ERR_FORMAT;
    }
    if (out->format != in->format) {
        fail("output format must match input");
        return AJI_ERR_FORMAT;
    }
    if (in->width != P.sw || in->height != P.sh || out->width != P.ww || out->height != P.wh) {
        fail("aji_resize dims %dx%d->%dx%d do not match configured %dx%d->%dx%d", in->width, in->height, out->width, out->height, P.sw, P.sh,
             P.ww, P.wh);
        return AJI_ERR_SHAPE;
    }
    const FrameFormat fi{in->format, in->matrix, in->range, in->siting}, fo{out->format, in->matrix, in->range, AJI_SITING_LEFT};
    std::string why;
    if (!P.io || !(P.io->in_format() == fi) || !(P.io->out_format() == fo)) {
        if (P.io && ticket_) wait(ticket_);
        P.io = FrameIO::create(*dev_, P.sw, P.sh, P.ww, P.wh, fi, fo, P.a.buf, P.b.buf, &why);
        if (!P.io) {
            fail("aji_resize: %s", why.c_str());
            return AJI_ERR;
        }
    }
    auto *ii = static_cast<aji_vk_image *>(in->plane[0]);
    auto *oi = static_cast<aji_vk_image *>(out->plane[0]);
    if (same_image(ii, oi)) {
        fail("aji_resize: the input and output must be different images");
        return AJI_ERR;
    }
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_resize: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    const uint32_t bps = in->format == AJI_FMT_P010 ? 2 : 1;
    full_barrier(cb);
    copy_planes(cb, ii, P.io->in_y().buf, P.io->in_uv().buf, P.io->in_pitch(), P.sw, P.sh, bps, true, dev_->queue_family());
    step_barrier(cb);
    P.io->record_pre(cb);
    step_barrier(cb);
    P.rs->record(cb);
    P.io->record_post(cb);
    mem_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    copy_planes(cb, oi, P.io->out_y().buf, P.io->out_uv().buf, P.io->out_pitch(), P.ww, P.wh, bps, false, dev_->queue_family());
    // the images' semaphores and the ticket (FrameSems: one signal for an input and output on one timeline)
    aji_vk_image *const both[2] = {ii, oi};
    const uint64_t t = ticket_ + 1;
    const FrameSems fs(tl_, t, both, 2);
    if (!dev_->frame_submit(slot, fs.waits, fs.nwait, fs.sig, fs.nsig(), &why)) {
        fail("aji_resize: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_ = t;
    fs.commit(both, 2);
    ii->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ii->access = VK_ACCESS_2_TRANSFER_READ_BIT;
    ii->queue_family = owner_after(ii, dev_->queue_family());
    oi->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    oi->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    oi->queue_family = owner_after(oi, dev_->queue_family());
    return AJI_OK;
}

int VkVideo::copy_frame(const aji_frame *in, const aji_frame *out)
{
    if (!in || !out || !in->plane[0] || !out->plane[0]) return AJI_ERR;
    if (in->format != out->format || (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010)) {
        fail("aji_copy_frame: the frames must be NV12 or P010, the same format");
        return AJI_ERR_FORMAT;
    }
    if (in->width != out->width || in->height != out->height || (in->width & 1) || (in->height & 1)) {
        fail("aji_copy_frame dims %dx%d vs %dx%d", in->width, in->height, out->width, out->height);
        return AJI_ERR_SHAPE;
    }
    auto *ii = static_cast<aji_vk_image *>(in->plane[0]);
    auto *oi = static_cast<aji_vk_image *>(out->plane[0]);
    if (same_image(ii, oi)) return AJI_OK;
    int slot = 0;
    VkCommandBuffer cb = dev_->frame_begin(&slot);
    if (!cb) {
        fail("aji_copy_frame: an earlier Vulkan submit has not finished after 10 s");
        return AJI_ERR;
    }
    VkImageMemoryBarrier ib[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    ib[0].srcAccessMask = ib[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ib[0].oldLayout = static_cast<VkImageLayout>(ii->layout);
    ib[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ib[0].image = static_cast<VkImage>(ii->image);
    ib[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // replaced
    ib[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib[1].image = static_cast<VkImage>(oi->image);
    for (auto &b : ib) {
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    set_owner(ib[0], ii, dev_->queue_family(), true);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, ib);
    VkImageCopy r[2] = {};
    for (int p = 0; p < 2; p++) {
        const VkImageAspectFlags a = p ? VK_IMAGE_ASPECT_PLANE_1_BIT : VK_IMAGE_ASPECT_PLANE_0_BIT;
        r[p].srcSubresource = r[p].dstSubresource = {a, 0, 0, 1};
        r[p].extent = {(uint32_t)(p ? in->width / 2 : in->width), (uint32_t)(p ? in->height / 2 : in->height), 1};
    }
    vkCmdCopyImage(cb, ib[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ib[1].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, r);
    // the images' semaphores and the ticket (FrameSems: one signal for an input and output on one timeline)
    aji_vk_image *const both[2] = {ii, oi};
    const uint64_t t = ticket_ + 1;
    const FrameSems fs(tl_, t, both, 2);
    std::string why;
    if (!dev_->frame_submit(slot, fs.waits, fs.nwait, fs.sig, fs.nsig(), &why)) {
        fail("aji_copy_frame: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_ = t;
    fs.commit(both, 2);
    ii->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ii->access = VK_ACCESS_2_TRANSFER_READ_BIT;
    ii->queue_family = owner_after(ii, dev_->queue_family());
    oi->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    oi->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    oi->queue_family = owner_after(oi, dev_->queue_family());
    return AJI_OK;
}

int VkVideo::done(uint64_t t)
{
    if (t == 0) return 1;
    uint64_t v = 0;
    if (vkGetSemaphoreCounterValue(dev_->device(), tl_, &v) != VK_SUCCESS) return AJI_ERR;
    return v >= t ? 1 : 0;
}

int VkVideo::wait(uint64_t t)
{
    if (t == 0) return AJI_OK;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &tl_;
    wi.pValues = &t;
    const VkResult r = vkWaitSemaphores(dev_->device(), &wi, kWaitNs);
    if (r != VK_SUCCESS) {
        fail("aji_wait: %s", r == VK_TIMEOUT ? "the GPU work has not finished after 10 s" : "the device was lost");
        return AJI_ERR;
    }
    return AJI_OK;
}

}  // namespace ajivk
