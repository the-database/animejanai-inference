// vk/tnet.cpp — see tnet.h.
#include "vk/tnet.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include "native/pack.h"
#include "vk/kernels.h"

namespace ajivk {
namespace {

bool fail(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

const char *kPassName[] = {"", "gemm", "ln", "dw", "colsum", "winattn", "chanstats", "chanreduce", "chansoft", "chanfold", "aimcg", "aimsg"};

void full_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

int pad16i(int c) { return (c + 15) / 16 * 16; }

}  // namespace

TArgs::TArgs()
{
    for (int at : {0, 4, 8, 12, 24}) v[at] = -1;  // every map none until set
}

void TArgs::map(int k, const TMap &m)
{
    const int at = k < 4 ? 4 * k : 24;
    v[at] = m.none ? -1 : (int32_t)(m.f32 ? m.bytes / 4 : m.bytes / 2);
    v[at + 1] = m.stride;
    v[at + 2] = m.off;
    v[at + 3] = m.f32 ? 1 : 0;
}

void TArgs::f(int k, float x) { std::memcpy(&v[k], &x, 4); }

TNet::~TNet()
{
    if (pool_) vkDestroyDescriptorPool(dev_.device(), pool_, nullptr);
    if (!io_external_) {
        dev_.destroy(in_);
        dev_.destroy(out_);
    }
    for (Buffer *b : {&arena_, &wfb_, &whb_, &gwb_, &gbb_, &partb_, &gprm_, &tprm_}) dev_.destroy(*b);
}

VkDeviceSize TNet::place(VkDeviceSize bytes)
{
    const VkDeviceSize align = std::max<VkDeviceSize>(dev_.info().storage_align, 16);
    const VkDeviceSize o = ar_;
    ar_ += (bytes + align - 1) / align * align;
    return o;
}

int TNet::add_wf(const std::vector<float> &v)
{
    while (wf_.size() % 4) wf_.push_back(0.f);
    const int o = (int)wf_.size();
    wf_.insert(wf_.end(), v.begin(), v.end());
    return o;
}

int TNet::add_wh(const std::vector<uint16_t> &h)
{
    while (wh_.size() % 8) wh_.push_back(0);
    const int o = (int)wh_.size();
    wh_.insert(wh_.end(), h.begin(), h.end());
    return o;
}

int TNet::add_wh_zero(size_t halfs) { return add_wh(std::vector<uint16_t>(halfs, 0)); }

int TNet::add_part(size_t floats)
{
    const int o = (int)part_;
    part_ += (floats + 3) / 4 * 4;
    return o;
}

TNet::Mat TNet::add_mat(const std::vector<float> &w, const std::vector<float> &bias, int K, int N)
{
    Mat m;
    m.K = K;
    m.N = N;
    m.Kp = pad16i(K);
    m.Np = pad16i(N);
    std::vector<uint16_t> h((size_t)m.Kp * m.Np, 0);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) h[(size_t)k * m.Np + n] = ajinative::float_to_half_bits(w[(size_t)k * N + n]);
    m.w = add_wh(h);
    if (!bias.empty()) {
        std::vector<float> b(m.Np, 0.f);
        std::copy(bias.begin(), bias.end(), b.begin());
        m.bias = add_wf(b);
    }
    return m;
}

void TNet::conv(const std::vector<uint16_t> &taps, const std::vector<float> &bias, int cin, int cout, const ConvIO &io,
                std::initializer_list<int> widths, const void *key)
{
    const int coutp = gconv_slice_width(cout, io.W, io.H, io.stride, widths);
    const int nz = pad16i(cout) / coutp;
    int wi = -1;
    for (auto &k : wkeys_)
        if (key && k.first == key) wi = k.second;
    if (wi < 0) {
        const std::vector<uint16_t> pw = gconv_pack_slices(dev_.family(), taps.data(), cin, io.cinp, cout, coutp, nz);
        const std::vector<uint16_t> pb = gconv_bias_slices(bias.data(), cout, coutp, nz);
        const VkDeviceSize align = std::max<VkDeviceSize>(dev_.info().storage_align, 16);
        const size_t aw = align / 2;
        W1 w;
        gw_.resize((gw_.size() + aw - 1) / aw * aw);
        w.woff = gw_.size() * 2;
        w.wsize = pw.size() * 2;
        gw_.insert(gw_.end(), pw.begin(), pw.end());
        gb_.resize((gb_.size() + aw - 1) / aw * aw);
        w.boff = gb_.size() * 2;
        w.bsize = pb.size() * 2;
        gb_.insert(gb_.end(), pb.begin(), pb.end());
        wv_.push_back(w);
        wi = (int)wv_.size() - 1;
        if (key) wkeys_.push_back({key, wi});
    }
    Op o;
    o.s.coutp = coutp;
    o.s.in = io.in;
    o.s.act = io.act;
    o.s.nres = io.nres;
    o.s.stride = io.stride;
    o.s.out = io.out;
    o.nz = nz;
    o.p.H = io.H;
    o.p.W = io.W;
    o.p.IH = io.IH;
    o.p.IW = io.IW;
    o.p.cinp = io.cinp;
    o.p.src_off = io.src_off;
    o.p.src_stride = io.src_stride;
    o.p.dst_stride = io.dst_stride;
    o.p.r1_stride = io.r1_stride;
    o.p.sy = io.sy;
    o.w = wi;
    o.nres = io.nres;
    o.src_in = io.src_in;
    o.dst_out = io.dst_out;
    o.src = io.src;
    o.dst = io.dst;
    o.r1 = io.r1;
    o.src_bytes = io.src_bytes;
    o.dst_bytes = io.dst_bytes;
    o.r1_bytes = io.r1_bytes;
    o.entry = (int)ops_.size();
    ops_.push_back(o);
    Step st;
    st.kind = GCONV;
    st.idx = o.entry;
    char lb[96];
    std::snprintf(lb, sizeof lb, "conv %d->%d %dx%d%s%s", io.cinp, coutp * nz, io.W, io.H, io.stride == 2 ? " s2" : "",
                  io.out == 4 ? " phase2" : io.out == 5 ? " phase4" : "");
    st.label = lb;
    steps_.push_back(st);
}

void TNet::pass(int p, const TArgs &a, uint32_t gx, uint32_t gy, const std::string &label)
{
    Step st;
    st.kind = PASS;
    st.pass = p;
    st.idx = (int)ta_.size();
    st.gx = gx;
    st.gy = gy;
    st.label = label.empty() ? kPassName[p] : label;
    ta_.push_back(a);
    steps_.push_back(st);
}

void TNet::pass_threads(int p, const TArgs &a, long long threads, const std::string &label)
{
    const long long groups = std::max<long long>(1, (threads + 255) / 256);
    const uint32_t gx = (uint32_t)std::min<long long>(groups, 4096), gy = (uint32_t)((groups + gx - 1) / gx);
    pass(p, a, gx, gy, label);
}

void TNet::copy(VkDeviceSize src, VkDeviceSize dst, VkDeviceSize bytes)
{
    Step st;
    st.kind = COPY;
    st.a = src;
    st.b = dst;
    st.n = bytes;
    st.label = "copy";
    steps_.push_back(st);
}

void TNet::zero(VkDeviceSize off, VkDeviceSize bytes)
{
    Step st;
    st.kind = ZERO;
    st.a = off;
    st.n = bytes;
    st.label = "zero";
    steps_.push_back(st);
}

void TNet::gemm(const TMap &a, int M, const Mat &w, const TMap &out, int act, bool act_last, int colscale, const TMap &add1,
                const TMap &add2, float slope)
{
    TArgs t;
    t.map(0, a);
    t.map(1, out);
    t.map(2, add1);
    t.map(3, add2);
    t.v[16] = M;
    t.v[17] = w.Kp;
    t.v[18] = w.Np;
    t.v[19] = act;
    t.v[20] = act_last ? 1 : 0;
    t.v[21] = w.w;
    t.v[22] = w.bias;
    t.f(28, slope);
    t.v[29] = colscale;
    t.v[30] = -1;
    char lb[64];
    std::snprintf(lb, sizeof lb, "gemm %dx%d", w.Kp, w.Np);
    pass(1, t, (uint32_t)((w.Np + 63) / 64), (uint32_t)((M + 63) / 64), lb);
}

void TNet::layernorm(const TMap &in, const TMap &out, int M, int C, int cpad, int gamma, int beta, float eps)
{
    if (C > 256 || cpad > 256) {  // the kernel's lanes cover 256 channels
        failed_ = "LayerNorm over more than 256 channels";
        return;
    }
    TArgs t;
    t.map(0, in);
    t.map(1, out);
    t.v[16] = M;
    t.v[17] = C;
    t.v[18] = cpad;
    t.v[19] = gamma;
    t.v[20] = beta;
    t.f(21, eps);
    // 8 tokens a workgroup
    const long long groups = (M + 7) / 8;
    const uint32_t gx = (uint32_t)std::min<long long>(groups, 4096), gy = (uint32_t)((groups + gx - 1) / gx);
    pass(2, t, gx, gy, "ln");
}

void TNet::dwconv(const TMap &in, const TMap &out, int H, int W, int C, int w, int bias, bool gelu, VkDeviceSize sums)
{
    if (C > 2048) {
        failed_ = "a depthwise conv over " + std::to_string(C) + " channels";
        return;
    }
    // a workgroup: 256 / ceil(C/8) columns of 8-row strips (tfm.comp)
    const int np = 256 / ((C + 7) / 8), strips = (H + 7) / 8;
    const int groups = (W * strips + np - 1) / np;
    TArgs t;
    t.map(0, in);
    t.map(1, out);
    t.v[16] = H;
    t.v[17] = W;
    t.v[18] = C;
    t.v[19] = w;
    t.v[20] = bias;
    t.v[21] = gelu ? 1 : 0;
    const bool with_sums = sums != ~0ull;
    const int po = with_sums ? add_part((size_t)groups * C) : -1;
    t.v[22] = po;
    const uint32_t gx = (uint32_t)std::min(groups, 4096);
    pass(3, t, gx, (uint32_t)((groups + gx - 1) / gx), "dw");
    if (with_sums) {
        TArgs r;
        r.v[16] = C;
        r.v[17] = groups;
        r.v[18] = po;
        r.v[19] = (int32_t)(sums / 4);
        pass_threads(4, r, C, "colsum");
    }
}

void TNet::step_sets(VkBuffer in, VkBuffer out)
{
    for (const Step &st : steps_) {
        if (st.kind == COPY || st.kind == ZERO) continue;
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev_.dummy(), 0, VK_WHOLE_SIZE};
        if (st.kind == GCONV) {
            const Op &o = ops_[st.idx];
            const W1 &w = wv_[o.w];
            bi[0] = o.src_in ? VkDescriptorBufferInfo{in, 0, VK_WHOLE_SIZE} : VkDescriptorBufferInfo{arena_.buf, o.src, o.src_bytes};
            if (o.nres) bi[1] = {arena_.buf, o.r1, o.r1_bytes};
            bi[2] = o.dst_out ? VkDescriptorBufferInfo{out, 0, VK_WHOLE_SIZE} : VkDescriptorBufferInfo{arena_.buf, o.dst, o.dst_bytes};
            bi[3] = {gwb_.buf, w.woff, w.wsize};
            bi[4] = {gbb_.buf, w.boff, w.bsize};
            bi[7] = {gprm_.buf, 0, VK_WHOLE_SIZE};
        } else {
            bi[2] = {arena_.buf, 0, VK_WHOLE_SIZE};
            bi[4] = {wfb_.buf, 0, VK_WHOLE_SIZE};
            bi[5] = {whb_.buf, 0, VK_WHOLE_SIZE};
            bi[6] = {partb_.buf, 0, VK_WHOLE_SIZE};
            bi[7] = {tprm_.buf, 0, VK_WHOLE_SIZE};
        }
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = st.set;
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev_.device(), Device::kBindings, wr, 0, nullptr);
    }
}

bool TNet::finish(VkDeviceSize in_bytes, VkDeviceSize out_bytes, std::string *why)
{
    if (!failed_.empty()) return fail(why, failed_);
    const uint64_t lim = dev_.info().max_storage_range;
    if (lim && ar_ > lim)
        return fail(why, "needs a " + std::to_string(ar_ >> 20) + " MB buffer, over this device's " + std::to_string(lim >> 20) +
                             " MB storage-buffer limit");
    if (ar_ >= (1ull << 32)) return fail(why, "needs a buffer over 4 GB");
    if (wf_.empty()) wf_.push_back(0.f);
    if (wh_.empty()) wh_.push_back(0);
    if (gw_.empty()) gw_.push_back(0);
    if (gb_.empty()) gb_.push_back(0);
    if (!part_) part_ = 4;
    if (!dev_.make_buffer(&in_, in_bytes, false, why) || !dev_.make_buffer(&out_, out_bytes, false, why) ||
        !dev_.make_buffer(&arena_, std::max<VkDeviceSize>(ar_, 16), false, why) || !dev_.make_buffer(&wfb_, wf_.size() * 4, false, why) ||
        !dev_.make_buffer(&whb_, wh_.size() * 2, false, why) || !dev_.make_buffer(&gwb_, gw_.size() * 2, false, why) ||
        !dev_.make_buffer(&gbb_, gb_.size() * 2, false, why) || !dev_.make_buffer(&partb_, part_ * 4, false, why) ||
        !dev_.upload(wfb_, wf_.data(), wf_.size() * 4, why) || !dev_.upload(whb_, wh_.data(), wh_.size() * 2, why) ||
        !dev_.upload(gwb_, gw_.data(), gw_.size() * 2, why) || !dev_.upload(gbb_, gb_.data(), gb_.size() * 2, why))
        return false;
    wbytes_ = wfb_.size + whb_.size + gwb_.size + gbb_.size;
    for (Op &o : ops_) {
        std::string w2;
        if (!gconv_prepare(dev_, &o, &w2)) return fail(why, "kernel: " + w2);
    }
    std::vector<GConvParams> gp;
    for (const Op &o : ops_) gp.push_back(o.p);
    if (gp.empty()) gp.push_back(GConvParams{});
    if (ta_.empty()) ta_.push_back(TArgs{});
    if (!dev_.make_buffer(&gprm_, gp.size() * sizeof(GConvParams), false, why) ||
        !dev_.upload(gprm_, gp.data(), gp.size() * sizeof(GConvParams), why) ||
        !dev_.make_buffer(&tprm_, ta_.size() * sizeof(TArgs), false, why) || !dev_.upload(tprm_, ta_.data(), ta_.size() * sizeof(TArgs), why))
        return false;
    size_t nsets = 0;
    for (Step &st : steps_) {
        if (st.kind == COPY || st.kind == ZERO) continue;
        nsets++;
        if (st.kind != PASS) continue;
        // the GEMM on cooperative matrices (fp32 sums) where the family has them
        const bool cm = st.pass == 1 && dev_.family() == Family::CM16 && dev_.info().cm16_acc32;
        const std::string m = cm ? std::string("tfm_gemm_cm16") : std::string("tfm_") + kPassName[st.pass];
        if (!(st.pipe = dev_.pipeline(m.c_str(), 0, cm ? 32 : 0, why))) return false;
        if (cm) st.label += " cm";
    }
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(std::max<size_t>(nsets, 1) * Device::kBindings)};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = (uint32_t)std::max<size_t>(nsets, 1);
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev_.device(), &pi, nullptr, &pool_) != VK_SUCCESS) return fail(why, "descriptor pool");
    }
    for (Step &st : steps_) {
        if (st.kind == COPY || st.kind == ZERO) continue;
        VkDescriptorSetLayout l = dev_.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev_.device(), &ai, &st.set) != VK_SUCCESS) return fail(why, "descriptor set");
        if (st.kind == GCONV) ops_[st.idx].set = st.set;
    }
    step_sets(in_.buf, out_.buf);
    {
        // zero the arena once (padding channels some passes don't write)
        VkCommandBuffer cb = dev_.begin();
        vkCmdFillBuffer(cb, arena_.buf, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cb, partb_.buf, 0, VK_WHOLE_SIZE, 0);
        if (!dev_.submit_wait(cb, why)) return false;
    }
    std::vector<GConvOp *> gops;
    for (Op &o : ops_) gops.push_back(&o);
    if (!gconv_choose(dev_, gops, why)) return false;
    bytes_ = in_.size + out_.size + arena_.size + wbytes_ + partb_.size + gprm_.size + tprm_.size;
    return true;
}

void TNet::bind_io(VkBuffer in, VkBuffer out)
{
    step_sets(in, out);
    if (!io_external_) {
        bytes_ -= in_.size + out_.size;
        dev_.destroy(in_);
        dev_.destroy(out_);
    }
    io_external_ = true;
    in_ = Buffer{};
    out_ = Buffer{};
    in_.buf = in;
    out_.buf = out;
}

void TNet::record_step(VkCommandBuffer cb, const Step &st) const
{
    switch (st.kind) {
    case GCONV: {
        const Op &o = ops_[st.idx];
        gconv_record(dev_, cb, o, o.variant);
        return;
    }
    case COPY: {
        const VkBufferCopy r{st.a, st.b, st.n};
        vkCmdCopyBuffer(cb, arena_.buf, arena_.buf, 1, &r);
        return;
    }
    case ZERO:
        vkCmdFillBuffer(cb, arena_.buf, st.a, st.n, 0);
        return;
    case PASS:
        break;
    }
    {  // AJI_TNET_SKIP=<label prefix> records no dispatch for those steps (to see what they cost)
        static const char *skip = std::getenv("AJI_TNET_SKIP");
        if (skip && st.label.rfind(skip, 0) == 0) return;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_.pipeline_layout(), 0, 1, &st.set, 0, nullptr);
    const int32_t pc[4] = {st.idx, 0, 0, 0};
    vkCmdPushConstants(cb, dev_.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, st.gx, st.gy, 1);
}

void TNet::record(VkCommandBuffer cb) const
{
    for (size_t i = 0; i < steps_.size(); i++) {
        if (i) full_barrier(cb);
        record_step(cb, steps_[i]);
    }
}

std::string TNet::profile(Device &dev) const
{
    if (dev.info().timestamp_ns <= 0) return "no GPU timestamps on this device\n";
    const uint32_t n = (uint32_t)steps_.size() + 1;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = n;
    if (vkCreateQueryPool(dev.device(), &qpi, nullptr, &qp) != VK_SUCCESS) return "no query pool\n";
    VkCommandBuffer cb = dev.begin();
    vkCmdResetQueryPool(cb, qp, 0, n);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    for (size_t i = 0; i < steps_.size(); i++) {
        if (i) full_barrier(cb);
        record_step(cb, steps_[i]);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, (uint32_t)i + 1);
    }
    std::string why, out;
    if (dev.submit_wait(cb, &why)) {
        std::vector<uint64_t> t(n);
        vkGetQueryPoolResults(dev.device(), qp, 0, n, n * 8, t.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        std::map<std::string, double> agg;
        for (size_t i = 0; i < steps_.size(); i++) agg[steps_[i].label] += (double)(t[i + 1] - t[i]) * dev.info().timestamp_ns / 1e6;
        double tot = 0;
        char line[160];
        for (auto &kv : agg) {
            std::snprintf(line, sizeof line, "  %-40s %7.3f ms\n", kv.first.c_str(), kv.second);
            out += line;
            tot += kv.second;
        }
        std::snprintf(line, sizeof line, "  total %.3f ms\n", tot);
        out += line;
    } else {
        out = "profile run failed: " + why + "\n";
    }
    vkDestroyQueryPool(dev.device(), qp, nullptr);
    return out;
}

}  // namespace ajivk
