// vk/executor.cpp — see executor.h.
#include "vk/executor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "native/layout.h"

namespace ajivk {

using ajinative::kModelInput;
using ajinative::kModelOutput;
using ajinative::Plan;
using ajinative::PlanStep;
using ajinative::PlanWeights;

namespace {

void compute_barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags src_access, VkPipelineStageFlags dst,
                     VkAccessFlags dst_access)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = src_access;
    mb.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void step_barrier(VkCommandBuffer cb)
{
    compute_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

}  // namespace

Executor::~Executor()
{
    VkDevice d = dev_.device();
    vkDeviceWaitIdle(d);
    if (pool_) vkDestroyDescriptorPool(d, pool_, nullptr);
    for (Buffer &b : owned_) dev_.destroy(b);
    dev_.destroy(in_);
    dev_.destroy(out_);
}

std::unique_ptr<Executor> Executor::create(Device &dev, const Plan &plan, int in_w, int in_h, std::string *why)
{
    auto fail = [&](const std::string &m) -> std::unique_ptr<Executor> {
        if (why) *why = m;
        return nullptr;
    };
    std::string reason;
    const bool trace = std::getenv("AJI_VK_TRACE") != nullptr;
    auto t_mark = std::chrono::steady_clock::now();
    auto mark = [&](const char *what) {
        const auto now = std::chrono::steady_clock::now();
        if (trace) std::fprintf(stderr, "aji_vk: create: %s %.1f ms\n", what, std::chrono::duration<double, std::milli>(now - t_mark).count());
        t_mark = now;
    };
    ajinative::Layout L;
    if (!ajinative::layout(plan, in_w, in_h, &L, &reason)) return fail(reason);
    const Family fam = dev.family();

    std::unique_ptr<Executor> e(new Executor(dev));
    e->u_ = L.u;
    e->W_ = L.W;
    e->H_ = L.H;
    e->out_w_ = L.out_w;
    e->out_h_ = L.out_h;
    if (!dev.make_buffer(&e->in_, (VkDeviceSize)3 * in_w * in_h * 2, false, &reason) ||
        !dev.make_buffer(&e->out_, (VkDeviceSize)3 * L.out_w * L.out_h * 2, false, &reason))
        return fail(reason);
    e->bytes_ += e->in_.size + e->out_.size;

    mark("input/output buffers");
    // feature buffers, as the layout assigned them
    std::vector<int> fbuf;
    // Slack after each feature map: the cm16 kernels load a gate's block input 16 pixels at a
    // time, and the last tile's columns past W run off the end.
    constexpr size_t kSlack = 16 * 64 * 2;
    for (size_t need : L.buffer_bytes) {
        Buffer b;
        if (!dev.make_buffer(&b, need + kSlack, false, &reason)) return fail("feature buffer: " + reason);
        fbuf.push_back((int)e->owned_.size());
        e->owned_.push_back(b);
        e->bytes_ += b.size;
    }
    auto tensor = [&](int t) -> VkBuffer {
        if (t == kModelInput) return e->in_.buf;
        if (t == kModelOutput) return e->out_.buf;
        if (t >= 0 && t < (int)L.buffer.size() && L.buffer[t] >= 0) return e->owned_[fbuf[L.buffer[t]]].buf;
        return dev.dummy();
    };
    auto upload = [&](const std::vector<uint16_t> &h, VkBuffer *out) -> bool {
        Buffer b;
        const size_t bytes = h.size() * 2;
        if (!dev.make_buffer(&b, bytes, false, &reason) || !dev.upload(b, h.data(), bytes, &reason)) {
            dev.destroy(b);
            return false;
        }
        e->owned_.push_back(b);
        e->bytes_ += b.size;
        e->weight_bytes_ += b.size;
        *out = b.buf;
        return true;
    };
    auto upload_conv = [&](const PlanWeights &pw, int cin_pad, int cout_pad, VkBuffer *w, VkBuffer *bias, VkBuffer *slope) -> bool {
        if (!upload(pack_weights(fam, pw.w.data(), pw.taps, pw.cin, cin_pad, pw.cout, cout_pad), w)) return false;
        if (!upload(pack_vector(pw.bias.data(), pw.cout, cout_pad), bias)) return false;
        if (slope && !pw.slope.empty() && !upload(pack_vector(pw.slope.data(), pw.cout, cout_pad), slope)) return false;
        return true;
    };

    mark("feature buffers");
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(plan.steps.size() * Device::kBindings)};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = (uint32_t)plan.steps.size();
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VkResult r = vkCreateDescriptorPool(dev.device(), &dpi, nullptr, &e->pool_);
    if (r != VK_SUCCESS) return fail(std::string("vkCreateDescriptorPool failed: ") + result_name(r));

    for (size_t i = 0; i < plan.steps.size(); i++) {
        const PlanStep &p = plan.steps[i];
        Step st;
        VkBuffer b[Device::kBindings];
        for (VkBuffer &x : b) x = dev.dummy();
        if (p.kind == PlanStep::CONCAT) {
            st.concat = true;
            if (!concat_dispatch(dev, p.c, p.nsrc, &st.disp[0], &reason)) return fail("concat: " + reason);
            for (int k = 0; k < p.nsrc && k < 4; k++) b[k] = tensor(p.srcs[k]);
            b[4] = tensor(p.dst);
            if (!upload_conv(p.w, L.cin_pad[i], L.cout_pad[i], &b[5], &b[6], nullptr)) return fail("weights: " + reason);
        } else {
            ConvKey k;
            k.cinp = L.cin_pad[i];
            k.coutp = L.cout_pad[i];
            k.in = p.in;
            k.act = p.act;
            k.gate = p.gate;
            k.out = p.out;
            st.nvar = 0;
            std::string first;
            for (int v = 0; v < kConvVariants; v++) {
                std::string w;
                if (conv_dispatch(dev, k, v, &st.disp[st.nvar], &w)) st.nvar++;
                else if (first.empty()) first = w;
            }
            if (st.nvar == 0) return fail("conv " + std::to_string(p.cin) + "->" + std::to_string(p.cout) + ": " + first);
            b[0] = tensor(p.src);
            b[1] = tensor(p.resid);
            b[2] = tensor(p.dst);
            st.has_resid = p.resid != -1;
            if (!upload_conv(p.w, k.cinp, k.coutp, &b[3], &b[4], &b[5])) return fail("weights: " + reason);
            if (p.gate == (int)ajinative::Gate::Tanh1x1 &&
                !upload_conv(p.att, p.att.cin, p.att.cout, &b[6], &b[7], nullptr))
                return fail("attention weights: " + reason);
        }
        VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsa.descriptorPool = e->pool_;
        dsa.descriptorSetCount = 1;
        const VkDescriptorSetLayout sl = dev.set_layout();
        dsa.pSetLayouts = &sl;
        if ((r = vkAllocateDescriptorSets(dev.device(), &dsa, &st.set)) != VK_SUCCESS)
            return fail(std::string("vkAllocateDescriptorSets failed: ") + result_name(r));
        VkDescriptorBufferInfo bi[Device::kBindings];
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            bi[j] = {b[j], 0, VK_WHOLE_SIZE};
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = st.set;
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev.device(), Device::kBindings, wr, 0, nullptr);
        e->steps_.push_back(st);
    }

    mark("weights and descriptor sets");
    // zeroed buffers, so variant timing never runs on stale bits
    {
        VkCommandBuffer cb = dev.begin();
        vkCmdFillBuffer(cb, e->in_.buf, 0, VK_WHOLE_SIZE, 0);
        for (int fi : fbuf) vkCmdFillBuffer(cb, e->owned_[fi].buf, 0, VK_WHOLE_SIZE, 0);
        if (!dev.submit_wait(cb, &reason)) return fail(reason);
    }

    mark("zero fill");
    // Tile variants: AJI_VK_VARIANT=<n> fixes them; else the device's stored pick for the kernel
    // at this size (vk/device.h, set_cache_dir); else timed when the queue has timestamps (once
    // per kernel and size: identical steps share it) and stored. Only the pipelines a step can
    // run are compiled.
    const char *fixed = std::getenv("AJI_VK_VARIANT");
    const double tick_ns = dev.info().timestamp_ns;
    VkQueryPool qp = VK_NULL_HANDLE;
    auto compile = [&](Step &st, int v) -> bool {
        if (!st.pipe[v]) st.pipe[v] = dev.pipeline(st.disp[v].module.c_str(), st.disp[v].local_size, st.disp[v].subgroup, &reason);
        return st.pipe[v] != VK_NULL_HANDLE;
    };
    auto bail = [&](const std::string &m) -> std::unique_ptr<Executor> {
        if (qp) vkDestroyQueryPool(dev.device(), qp, nullptr);
        return fail(m);
    };
    const std::string size = "@" + std::to_string(e->W_) + "x" + std::to_string(e->H_);
    std::map<std::string, std::pair<int, double>> timed;  // this create's timings
    char line[256];
    for (size_t i = 0; i < e->steps_.size(); i++) {
        Step &st = e->steps_[i];
        const std::string &m0 = st.disp[0].module;
        const std::string key = m0.substr(0, m0.rfind('_')) + size;  // the kernel without its tile shape
        double ms = -1.0;
        const char *how = "";
        int v = -1;
        if (fixed) {
            v = std::atoi(fixed);
            how = " (AJI_VK_VARIANT)";
        } else if (st.nvar > 1) {
            const auto it = timed.find(key);
            if (it != timed.end()) {
                v = it->second.first;
                ms = it->second.second;
            } else if ((v = dev.cached_variant(key)) >= 0 && v < st.nvar) {
                how = " (stored)";
            } else if (tick_ns > 0) {
                if (!qp) {
                    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
                    qpi.queryCount = 2;
                    if (vkCreateQueryPool(dev.device(), &qpi, nullptr, &qp) != VK_SUCCESS) qp = VK_NULL_HANDLE;
                }
                v = -1;
                for (int c = 0; qp && c < st.nvar; c++) {
                    if (!compile(st, c)) return bail(reason);
                    constexpr int kWarm = 2, kRuns = 8;
                    VkCommandBuffer cb = dev.begin();
                    vkCmdResetQueryPool(cb, qp, 0, 2);
                    for (int k = 0; k < kWarm; k++) e->record_step(cb, st, c, e->W_, e->H_);
                    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                    for (int k = 0; k < kRuns; k++) e->record_step(cb, st, c, e->W_, e->H_);
                    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                    if (!dev.submit_wait(cb, &reason)) return bail("timing the tile variants: " + reason);
                    uint64_t t[2] = {0, 0};
                    vkGetQueryPoolResults(dev.device(), qp, 0, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                    const double cms = (double)(t[1] - t[0]) * tick_ns * 1e-6 / kRuns;
                    if (v < 0 || cms < ms) {
                        ms = cms;
                        v = c;
                    }
                }
                if (v >= 0) {
                    timed[key] = {v, ms};
                    dev.store_variant(key, v);
                }
            }
        }
        st.variant = v >= 0 && v < st.nvar ? v : 0;
        if (!compile(st, st.variant)) return bail(reason);
        if (ms >= 0)
            std::snprintf(line, sizeof line, "%2zu %s %.3f ms\n", i, st.disp[st.variant].module.c_str(), ms);
        else
            std::snprintf(line, sizeof line, "%2zu %s%s\n", i, st.disp[st.variant].module.c_str(), how);
        e->desc_ += line;
    }
    mark("variants and pipelines");
    if (qp) vkDestroyQueryPool(dev.device(), qp, nullptr);
    return e;
}

void Executor::record_step(VkCommandBuffer cb, const Step &st, int v, int W, int H) const
{
    const Dispatch &d = st.disp[v];
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipe[v]);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_.pipeline_layout(), 0, 1, &st.set, 0, nullptr);
    int32_t pc[4] = {0, 0, 0, 0};
    uint32_t gx, gy;
    if (st.concat) {
        pc[0] = W * H;
        gx = (uint32_t)((W * H + d.tile_px - 1) / d.tile_px);
        gy = 1;
    } else {
        pc[0] = H;
        pc[1] = W;
        pc[2] = st.has_resid ? 1 : 0;
        gx = (uint32_t)((W + 15) / 16);
        gy = (uint32_t)((H + d.tile_h - 1) / d.tile_h);
    }
    vkCmdPushConstants(cb, dev_.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, gx, gy, 1);
    step_barrier(cb);
}

bool Executor::record(VkCommandBuffer cb, int w, int h) const
{
    if (w < 1 || h < 1 || w % u_ || h % u_ || w / u_ > W_ || h / u_ > H_) return false;
    for (const Step &st : steps_) record_step(cb, st, st.variant, w / u_, h / u_);
    return true;
}

bool Executor::run(int w, int h, std::string *why) const
{
    VkCommandBuffer cb = dev_.begin();
    compute_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    const bool ok = record(cb, w, h);
    compute_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
    if (!dev_.submit_wait(cb, why)) return false;
    if (!ok && why) *why = "unsupported input size";
    return ok;
}

}  // namespace ajivk
