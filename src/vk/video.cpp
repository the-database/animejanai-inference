// vk/video.cpp — see video.h.
#include "vk/video.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "core/fileio.h"
#include "native/lower.h"
#include "native/onnx_graph.h"
#include "onnx_shape.h"

namespace ajivk {

namespace {

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
    if (tl_) vkDestroySemaphore(dev_->device(), tl_, nullptr);
    dev_->save_caches();
}

void VkVideo::teardown()
{
    if (dev_ && ticket_) wait(ticket_);
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
    for (const AjiModelConf &m : chain->models) {
        if (m.name.empty()) continue;
        const std::string onnx = model_dir_ + "/" + m.name + ".onnx";
        std::vector<char> bytes;
        if (!ajicore::read_file_bytes(onnx, &bytes)) return passthrough(m.name + ": the model file can't be read: passthrough");
        AjiOnnxInput oin;
        int t = aji_onnx_input(bytes.data(), bytes.size(), &oin) ? aji_onnx_temporal_frames(oin) : 1;
        if (m.frames > 0) t = m.frames;
        if (t > 1) return passthrough(m.name + ": temporal models are not on Vulkan frames yet: passthrough");
        // resizes before upscale: the factor, then the height (fitted to 16:9), as aji_dml.cpp's
        const auto resize_to = [&](int nw, int nh, const char *what) {
            Step r;
            r.in_w = cw;
            r.in_h = ch;
            r.out_w = nw;
            r.out_h = nh;
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
    if (steps_.empty()) return passthrough("");
    if ((cw & 1) || (ch & 1)) return passthrough("the chain's output size is odd: passthrough");
    if (chain->rife) log_steps_.push_back("RIFE is not on Vulkan frames yet: left out");
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
        } else if (!(s.rs = RgbResize::create(*dev_, s.in_w, s.in_h, s.out_w, s.out_h, chain_[k % 2].buf, chain_[(k + 1) % 2].buf, &why))) {
            fail("resize %dx%d -> %dx%d: %s", s.in_w, s.in_h, s.out_w, s.out_h, why.c_str());
            teardown();
            return AJI_ERR;
        }
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
    // the images' semaphores (AVVkFrame's rule: wait for value, signal value + 1) and the ticket
    const Device::SemValue waits[2] = {{static_cast<VkSemaphore>(ii->semaphore), ii->value},
                                       {static_cast<VkSemaphore>(oi->semaphore), oi->value}};
    const uint64_t t = ticket_ + 1;
    const Device::SemValue signals[3] = {
        {tl_, t}, {static_cast<VkSemaphore>(ii->semaphore), ii->value + 1}, {static_cast<VkSemaphore>(oi->semaphore), oi->value + 1}};
    if (!dev_->frame_submit(slot, waits, 2, signals, 3, &why)) {
        fail("aji_infer: %s", why.c_str());
        return AJI_ERR;
    }
    ticket_ = t;
    ii->value += 1;
    ii->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ii->access = VK_ACCESS_2_TRANSFER_READ_BIT;
    ii->queue_family = dev_->queue_family();
    oi->value += 1;
    oi->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    oi->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    oi->queue_family = dev_->queue_family();
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
