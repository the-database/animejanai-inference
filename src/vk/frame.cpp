// vk/frame.cpp — see frame.h. Module names must match src/vk/shaders/shaders.txt.
#include "vk/frame.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "resample.h"

namespace ajivk {
namespace {

bool fail(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

// One parameter entry, the HLSL KernArgs layout (frame_common.glsl).
struct KArgs {
    int32_t di[4] = {};
    int32_t dj[4] = {};
    float kr = 0, kb = 0, yoff = 0, yscale = 0;
    float coff = 0, cscale = 0, qdiv = 0, qmax = 0;
};
static_assert(sizeof(KArgs) == 64, "one Params entry");

KArgs csp_args(const aji_csp &c, float qdiv, float qmax)
{
    KArgs a;
    a.kr = c.kr;
    a.kb = c.kb;
    a.yoff = c.yoff;
    a.yscale = c.yscale;
    a.coff = c.coff;
    a.cscale = c.cscale;
    a.qdiv = qdiv;
    a.qmax = qmax;
    return a;
}

uint32_t up4(uint32_t n) { return (n + 3) & ~3u; }
uint32_t groups(int n, int per) { return (uint32_t)((n + per - 1) / per); }

}  // namespace

FramePipeline::~FramePipeline()
{
    if (pool_) vkDestroyDescriptorPool(dev_.device(), pool_, nullptr);
    for (Buffer *b : {&in_y_, &in_uv_, &out_y_, &out_uv_, &tmp_h_, &tmp_v_, &ctmp_, &tables_, &params_}) dev_.destroy(*b);
}

std::unique_ptr<FramePipeline> FramePipeline::create(Device &dev, const ajinative::Plan &plan, int w, int h, const FrameFormat &f,
                                                     std::string *why)
{
    if (w < 2 || h < 2 || (w & 1) || (h & 1)) return fail(why, "4:2:0 frames need even sizes"), nullptr;
    if (f.format != AJI_FMT_NV12 && f.format != AJI_FMT_P010) return fail(why, "NV12 or P010 frames only"), nullptr;
    std::unique_ptr<FramePipeline> p(new FramePipeline(dev));
    p->w_ = w;
    p->h_ = h;
    p->ex_ = Executor::create(dev, plan, w, h, why);
    if (!p->ex_) return nullptr;
    p->ow_ = p->ex_->out_w();
    p->oh_ = p->ex_->out_h();
    if ((p->ow_ & 1) || (p->oh_ & 1)) return fail(why, "the model's output size is odd"), nullptr;
    const bool t16 = f.format == AJI_FMT_P010;
    const int bps = t16 ? 2 : 1;
    const int cw = w / 2, ch = h / 2, ow = p->ow_, oh = p->oh_, ocw = ow / 2, och = oh / 2;
    // pitches: whole dwords (the post kernels store dwords that spill into the row padding)
    p->in_pitch_ = up4((uint32_t)(w * bps));
    p->out_pitch_ = up4((uint32_t)(ow * bps));

    // weight tables: pre upsamples chroma (the input's siting), post downsamples (left-sited)
    double sx, sy, dx, dy;
    aji_resample::chroma_shifts(f.siting, true, &sx, &sy);
    aji_resample::chroma_shifts(AJI_SITING_LEFT, false, &dx, &dy);
    const aji_resample::weights tbl[4] = {
        aji_resample::compute(cw, w, sx, AJI_FILTER_SPLINE36), aji_resample::compute(ch, h, sy, AJI_FILTER_SPLINE36),
        aji_resample::compute(ow, ocw, dx, AJI_FILTER_SPLINE36), aji_resample::compute(oh, och, dy, AJI_FILTER_SPLINE36)};
    // the fused post pass's tile (frame_post.comp): 64 outputs per group row, span <= 192, taps <= 16
    {
        int span = 0;
        for (int g = 0; g < ocw; g += 64) {
            const int last = std::min(g + 64, ocw) - 1;
            span = std::max(span, tbl[2].start[last] + tbl[2].taps - tbl[2].start[g]);
        }
        if (tbl[2].taps > 16 || span > 192) return fail(why, "chroma filter wider than the post kernel's tile"), nullptr;
    }
    std::vector<uint8_t> tables;
    VkDeviceSize toff[4][2] = {}, tsize[4][2] = {};
    const VkDeviceSize align = std::max<VkDeviceSize>(dev.info().storage_align, 16);
    const auto put = [&](const void *src, size_t bytes, VkDeviceSize *off, VkDeviceSize *size) {
        tables.resize((tables.size() + align - 1) / align * align);
        *off = tables.size();
        *size = std::max<size_t>(bytes, 4);
        tables.resize(tables.size() + *size, 0);
        std::memcpy(tables.data() + *off, src, bytes);
    };
    for (int i = 0; i < 4; i++) {
        put(tbl[i].start.data(), tbl[i].start.size() * 4, &toff[i][0], &tsize[i][0]);
        put(tbl[i].wt.data(), tbl[i].wt.size() * 4, &toff[i][1], &tsize[i][1]);
    }

    // parameters
    const aji_csp csp = aji_resample::make_csp(f.format, f.matrix, f.range);
    const float qdiv = t16 ? 64.0f : 1.0f, qmax = t16 ? 1023.0f : 255.0f;
    KArgs a[5];
    a[0] = csp_args(csp, 0, 0);
    a[0].di[0] = cw; a[0].di[1] = w; a[0].di[2] = tbl[0].taps; a[0].di[3] = ch;
    a[0].dj[0] = (int)p->in_pitch_;
    a[1] = csp_args(csp, 0, 0);
    a[1].di[0] = w; a[1].di[1] = ch; a[1].di[2] = tbl[1].taps; a[1].di[3] = h;
    a[2] = csp_args(csp, 0, 0);
    a[2].di[0] = w; a[2].di[1] = h;
    a[2].dj[0] = (int)p->in_pitch_; a[2].dj[1] = 0;
    a[3] = csp_args(csp, qdiv, qmax);
    a[3].di[0] = ow; a[3].di[1] = oh; a[3].di[2] = tbl[2].taps; a[3].di[3] = ocw;
    a[3].dj[0] = (int)p->out_pitch_;
    a[4] = csp_args(csp, qdiv, qmax);
    a[4].di[0] = ocw; a[4].di[1] = oh; a[4].di[2] = tbl[3].taps; a[4].di[3] = och;
    a[4].dj[0] = (int)p->out_pitch_;

    if (!dev.make_buffer(&p->in_y_, (VkDeviceSize)p->in_pitch_ * h, false, why) ||
        !dev.make_buffer(&p->in_uv_, (VkDeviceSize)p->in_pitch_ * ch, false, why) ||
        !dev.make_buffer(&p->out_y_, (VkDeviceSize)p->out_pitch_ * oh, false, why) ||
        !dev.make_buffer(&p->out_uv_, (VkDeviceSize)p->out_pitch_ * och, false, why) ||
        !dev.make_buffer(&p->tmp_h_, (VkDeviceSize)2 * w * ch * 4, false, why) ||
        !dev.make_buffer(&p->tmp_v_, (VkDeviceSize)2 * w * h * 4, false, why) ||
        !dev.make_buffer(&p->ctmp_, (VkDeviceSize)2 * ocw * oh * 4, false, why) ||
        !dev.make_buffer(&p->tables_, tables.size(), false, why) || !dev.make_buffer(&p->params_, sizeof a, false, why) ||
        !dev.upload(p->tables_, tables.data(), tables.size(), why) || !dev.upload(p->params_, a, sizeof a, why))
        return nullptr;

    // pipelines and descriptor sets: pre uv_h, pre v, pre combine, post mh, post uv_v
    const char *sfx = t16 ? "_u16" : "_u8";
    const std::string mod[5] = {std::string("fr_pre1") + sfx, std::string("fr_pre2") + sfx, std::string("fr_pre3") + sfx,
                                std::string("fr_post1") + sfx, std::string("fr_post2") + sfx};
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * Device::kBindings};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 5;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev.device(), &pi, nullptr, &p->pool_) != VK_SUCCESS) return fail(why, "descriptor pool"), nullptr;
    }
    struct B {
        VkBuffer buf;
        VkDeviceSize off, size;
    };
    const B none{VK_NULL_HANDLE, 0, 0};
    const auto T = [&](int i, int k) { return B{p->tables_.buf, toff[i][k], tsize[i][k]}; };
    const auto W = [](const Buffer &b) { return B{b.buf, 0, VK_WHOLE_SIZE}; };
    const B binds[5][5] = {
        {T(0, 0), T(0, 1), W(p->in_uv_), W(p->tmp_h_), none},
        {T(1, 0), T(1, 1), W(p->tmp_h_), W(p->tmp_v_), none},
        {none, none, W(p->in_y_), W(p->tmp_v_), W(p->ex_->input())},
        {T(2, 0), T(2, 1), W(p->ex_->output()), W(p->out_y_), W(p->ctmp_)},
        {T(3, 0), T(3, 1), W(p->ctmp_), W(p->out_uv_), none},
    };
    for (int k = 0; k < 5; k++) {
        p->pipe_[k] = dev.pipeline(mod[k].c_str(), 0, 0, why);
        if (!p->pipe_[k]) return nullptr;
        VkDescriptorSetLayout l = dev.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = p->pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev.device(), &ai, &p->set_[k]) != VK_SUCCESS) return fail(why, "descriptor set"), nullptr;
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev.dummy(), 0, VK_WHOLE_SIZE};
        for (int j = 0; j < 5; j++)
            if (binds[k][j].buf) bi[j] = {binds[k][j].buf, binds[k][j].off, binds[k][j].size};
        bi[7] = {p->params_.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = p->set_[k];
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev.device(), Device::kBindings, wr, 0, nullptr);
    }
    const int tpx_y = t16 ? 2 : 4, tpx_uv = t16 ? 1 : 2;
    (void)tpx_y;
    const uint32_t g[5][3] = {{groups(w, 32), groups(ch, 8), 1},
                              {groups(w, 32), groups(h, 8), 2},
                              {groups(w / 2, 32), groups(h, 8), 1},
                              {groups(ocw, 64), groups(oh, 8), 1},
                              {groups((ocw + tpx_uv - 1) / tpx_uv, 32), groups(och, 8), 1}};
    std::memcpy(p->groups_, g, sizeof g);
    return p;
}

void FramePipeline::record(VkCommandBuffer cb) const
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    const auto run = [&](int k) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_[k]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_.pipeline_layout(), 0, 1, &set_[k], 0, nullptr);
        const int32_t pc[4] = {k, 0, 0, 0};
        vkCmdPushConstants(cb, dev_.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, groups_[k][0], groups_[k][1], groups_[k][2]);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    run(0);
    run(1);
    run(2);
    ex_->record(cb, w_, h_);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    run(3);
    run(4);
}

}  // namespace ajivk
