// vk/rife.cpp — see rife.h. The steps are native/rife.cpp's:
//
//  - the encoder (when the model has one) per frame: a stride-2 conv straight from the frame's NCHW
//    planes, the middle convs, and its transposed conv as 2 row-phase convs into the frame's
//    full-resolution features;
//  - per block: the gather (its input at its scale), conv0 (two stride-2 convs), the residual
//    convs (beta folded into the weights, the block input added back), and lastconv + depth-to-
//    space as 4 row-phase convs into the block's output; an ensemble runs the block again on the
//    time-reversed input and merges the two;
//  - the final blend at full resolution.
//
// Convs wider than 64 output channels run as 64-channel slices. The block outputs and features
// live in one buffer the resampling kernels read (binding 2); the block scratch in another, bound
// per op.
#include "vk/rife.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "native/rife_plan.h"
#include "vk/gconv.h"
#include "vk/kernels.h"
#include "vk/spirv.h"

namespace ajivk {
namespace {

using namespace ajinative::onnxread;
using ajinative::kRifeMaxBlocks;

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

struct Ref {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize off = 0, size = 0;
};

// One entry of the resampling kernels' parameters (rife.comp).
struct RParams {
    int32_t v[64] = {};
};

class VkRife : public Rife {
public:
    explicit VkRife(Device &d) : dev_(d) {}
    ~VkRife() override
    {
        if (pool_) vkDestroyDescriptorPool(dev_.device(), pool_, nullptr);
        if (!io_external_) {
            dev_.destroy(in_);
            dev_.destroy(out_);
        }
        for (Buffer *b : {&pers_, &scr_, &wts_, &bias_, &gprm_, &rprm_}) dev_.destroy(*b);
    }
    static std::unique_ptr<Rife> build(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why);

    const Buffer &input() const override { return in_; }
    const Buffer &output() const override { return out_; }
    void bind_io(VkBuffer in, VkBuffer out) override;
    void record(VkCommandBuffer cb) const override;
    std::string profile(Device &dev) const override;
    std::string describe() const override { return what_; }
    size_t device_bytes() const override { return bytes_; }

private:
    struct Op : GConvOp {
        int w = 0;  // into wv_
        Ref src, dst, r1;
        bool src_in = false;  // src is the 11-channel input (bind_io rebinds it)
    };
    struct W1 {
        VkDeviceSize woff = 0, wsize = 0, boff = 0, bsize = 0;
    };
    enum Kind { GCONV, GATHER, MERGE, FINAL };
    struct Step {
        Kind kind = GCONV;
        int idx = 0;  // GCONV: into ops_; else its RParams entry
        uint32_t gx = 1, gy = 1;
        Ref dst;      // GATHER: the block input
        std::string label;  // profile()
        VkPipeline pipe = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    Device &dev_;
    int W_ = 0, H_ = 0;
    Buffer in_, out_, pers_, scr_, wts_, bias_, gprm_, rprm_;
    std::vector<W1> wv_;
    std::vector<Op> ops_;
    std::vector<RParams> rp_;
    std::vector<Step> steps_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    size_t bytes_ = 0;
    bool io_external_ = false;  // in_ / out_ are the caller's (bind_io): handles only
    std::string what_;

    void write_set(VkDescriptorSet set, const VkDescriptorBufferInfo *bi) const;
    void record_step(VkCommandBuffer cb, const Step &st) const;
    void step_sets(VkBuffer in, VkBuffer out);
};

void VkRife::write_set(VkDescriptorSet set, const VkDescriptorBufferInfo *bi) const
{
    VkWriteDescriptorSet wr[Device::kBindings];
    for (int j = 0; j < Device::kBindings; j++) {
        wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[j].dstSet = set;
        wr[j].dstBinding = (uint32_t)j;
        wr[j].descriptorCount = 1;
        wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[j].pBufferInfo = &bi[j];
    }
    vkUpdateDescriptorSets(dev_.device(), Device::kBindings, wr, 0, nullptr);
}

// Every step's descriptors, with the input and output at in / out.
void VkRife::step_sets(VkBuffer in, VkBuffer out)
{
    for (const Step &st : steps_) {
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev_.dummy(), 0, VK_WHOLE_SIZE};
        if (st.kind == GCONV) {
            const Op &o = ops_[st.idx];
            const W1 &w = wv_[o.w];
            bi[0] = o.src_in ? VkDescriptorBufferInfo{in, 0, VK_WHOLE_SIZE} : VkDescriptorBufferInfo{o.src.buf, o.src.off, o.src.size};
            if (o.r1.buf) bi[1] = {o.r1.buf, o.r1.off, o.r1.size};
            bi[2] = {o.dst.buf, o.dst.off, o.dst.size};
            bi[3] = {wts_.buf, w.woff, w.wsize};
            bi[4] = {bias_.buf, w.boff, w.bsize};
            bi[7] = {gprm_.buf, 0, VK_WHOLE_SIZE};
        } else {
            bi[0] = {in, 0, VK_WHOLE_SIZE};
            bi[2] = {pers_.buf, 0, VK_WHOLE_SIZE};
            if (st.kind == GATHER) bi[3] = {st.dst.buf, st.dst.off, st.dst.size};
            if (st.kind == FINAL) bi[3] = {out, 0, VK_WHOLE_SIZE};
            bi[7] = {rprm_.buf, 0, VK_WHOLE_SIZE};
        }
        write_set(st.set, bi);
    }
}

void VkRife::bind_io(VkBuffer in, VkBuffer out)
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

std::string VkRife::profile(Device &dev) const
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
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    for (size_t i = 0; i < steps_.size(); i++) {
        if (i) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
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

void VkRife::record_step(VkCommandBuffer cb, const Step &st) const
{
    if (st.kind == GCONV) {
        const Op &o = ops_[st.idx];
        gconv_record(dev_, cb, o, o.variant);
        return;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_.pipeline_layout(), 0, 1, &st.set, 0, nullptr);
    const int32_t pc[4] = {st.idx, 0, 0, 0};
    vkCmdPushConstants(cb, dev_.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, st.gx, st.gy, 1);
}

void VkRife::record(VkCommandBuffer cb) const
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (size_t i = 0; i < steps_.size(); i++) {
        if (i) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        record_step(cb, steps_[i]);
    }
}

std::unique_ptr<Rife> VkRife::build(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why)
{
    ajinative::RifePlan plan;
    if (!ajinative::rife_plan(g, W, H, &plan, why)) return nullptr;
    const int nb = (int)plan.blocks.size();
    if (nb < 1 || nb > kRifeMaxBlocks) return refuse_(why, "RIFE block count"), nullptr;
    std::unique_ptr<VkRife> r(new VkRife(dev));
    r->W_ = W;
    r->H_ = H;
    const Family fam = dev.family();
    const bool ens = plan.ens;
    const int F = plan.F, FP = F ? pad8(F) : 0;
    const VkDeviceSize plane = (VkDeviceSize)W * H;
    const VkDeviceSize align = std::max<VkDeviceSize>(dev.info().storage_align, 16);
    const auto up = [&](VkDeviceSize b) { return (b + align - 1) / align * align; };

    // ---- the persistent buffer: features, block outputs, the ensemble's reverse output
    VkDeviceSize pers = 0;
    const auto place = [&](VkDeviceSize bytes) {
        const VkDeviceSize o = pers;
        pers += up(bytes);
        return o;
    };
    VkDeviceSize ofeat[2] = {0, 0};
    if (F)
        for (auto &o : ofeat) o = place(plane * FP * 2);
    std::vector<VkDeviceSize> oblk(nb);
    std::vector<int> kp(nb);
    VkDeviceSize needR = 0;
    for (int b = 0; b < nb; b++) {
        const auto &bp = plan.blocks[b];
        kp[b] = bp.K <= 8 ? 8 : 16;
        const VkDeviceSize px = (VkDeviceSize)(W / bp.s) * (H / bp.s);
        oblk[b] = place(px * kp[b] * 2);
        needR = std::max(needR, px * kp[b] * 2);
    }
    const VkDeviceSize oR = ens ? place(needR) : 0;

    // ---- the scratch buffer: a block's input X, conv0's T1, the residual ping-pong A / B; the
    // encoder (before any block) ping-pongs inside X when it fits
    VkDeviceSize needX = 0, needT1 = 0, needA = 0;
    for (const auto &bp : plan.blocks) {
        const VkDeviceSize px = (VkDeviceSize)(W / bp.s) * (H / bp.s);
        needX = std::max(needX, px * pad16(bp.c0a.cin) * 2);
        needT1 = std::max(needT1, px / 4 * pad16(bp.c0a.cout) * 2);
        needA = std::max(needA, px / 16 * pad16(bp.c0b.cout) * 2);
    }
    int maxc = 0;
    for (const Conv &c : plan.enc) maxc = std::max(maxc, pad16(c.cout));
    const VkDeviceSize hbytes = plane / 4 * maxc * 2;
    VkDeviceSize scr = 0;
    const auto splace = [&](VkDeviceSize bytes) {
        const VkDeviceSize o = scr;
        scr += up(bytes);
        return o;
    };
    const VkDeviceSize oX = splace(needX), oT1 = splace(needT1), oA = splace(needA), oB = splace(needA);
    VkDeviceSize oH[2] = {oX, oX + up(hbytes)};
    if (F && up(hbytes) * 2 > up(needX)) {
        oH[0] = splace(hbytes);
        oH[1] = splace(hbytes);
    }

    const uint64_t lim = dev.info().max_storage_range;
    const uint64_t biggest = std::max<uint64_t>({pers, needX, needT1, needA, plane * 11 * 2});
    if (lim && biggest > lim)
        return refuse_(why, "RIFE at " + std::to_string(W) + "x" + std::to_string(H) + " needs " + std::to_string(biggest >> 20) +
                                " MB buffers, over this device's " + std::to_string(lim >> 20) + " MB storage-buffer limit"),
               nullptr;
    if (!dev.make_buffer(&r->in_, plane * 11 * 2, false, why) || !dev.make_buffer(&r->out_, plane * 3 * 2, false, why) ||
        !dev.make_buffer(&r->pers_, pers, false, why) || !dev.make_buffer(&r->scr_, scr, false, why))
        return nullptr;
    const VkBuffer P = r->pers_.buf, S = r->scr_.buf;
    const auto pref = [&](VkDeviceSize off, VkDeviceSize bytes) { return Ref{P, off, bytes}; };
    const auto sref = [&](VkDeviceSize off, VkDeviceSize bytes) { return Ref{S, off, bytes}; };

    // ---- weights (packed per family, each slice at an aligned offset)
    std::vector<uint16_t> wall, ball;
    const auto add_w = [&](const std::vector<uint16_t> &taps, const std::vector<float> &bias, int cin, int cinp, int cout, int coutp,
                           int nz) {
        const std::vector<uint16_t> pw = gconv_pack_slices(fam, taps.data(), cin, cinp, cout, coutp, nz);
        const std::vector<uint16_t> pb = gconv_bias_slices(bias.data(), cout, coutp, nz);
        W1 w;
        const size_t aw = align / 2;
        wall.resize((wall.size() + aw - 1) / aw * aw);
        w.woff = wall.size() * 2;
        w.wsize = pw.size() * 2;
        wall.insert(wall.end(), pw.begin(), pw.end());
        ball.resize((ball.size() + aw - 1) / aw * aw);
        w.boff = ball.size() * 2;
        w.bsize = pb.size() * 2;
        ball.insert(ball.end(), pb.begin(), pb.end());
        r->wv_.push_back(w);
        return (int)r->wv_.size() - 1;
    };
    // a conv op (and its step), nz output slices of s.coutp channels
    const auto add_op = [&](int w, const GConvShape &s, int nz, Ref src, bool src_in, Ref dst, Ref r1, const GConvParams &p) {
        Op o;
        o.s = s;
        o.nz = nz;
        o.p = p;
        o.w = w;
        o.src = src;
        o.src_in = src_in;
        o.dst = dst;
        o.r1 = r1;
        o.entry = (int)r->ops_.size();
        r->ops_.push_back(o);
        Step st;
        st.kind = GCONV;
        st.idx = o.entry;
        char lb[96];
        std::snprintf(lb, sizeof lb, "conv %d->%d %dx%d%s%s", p.cinp, s.coutp * nz, p.W, p.H, s.stride == 2 ? " s2" : "",
                      s.out == 4 ? " phase2" : s.out == 5 ? " phase4" : "");
        st.label = lb;
        r->steps_.push_back(st);
    };
    // A plain conv, as one dispatch of output slices (gconv.h).
    struct ConvIO {
        Ref src, dst, r1;
        bool src_in = false;
        int in = 0, act = 0, stride = 1;
        int src_off = 0, src_stride = 0, dst_stride = 0, r1_stride = 0;
        int IH = 0, IW = 0, H = 0, W = 0, cinp = 16;
        float slope = 0.2f;
    };
    const auto conv = [&](const std::vector<uint16_t> &taps, const std::vector<float> &bias, int cin, int cout, const ConvIO &io,
                          std::vector<int> *wslices) {
        // the slice width: the widest built (shaders.txt) that still gives the dispatch ~256
        // workgroups, so the small low-resolution convs of the first blocks fill the GPU
        const int CP = pad16(cout);
        const int coutp = io.in == 0 ? gconv_slice_width(cout, io.W, io.H, io.stride, {64, 48, 32, 16})
                                     : gconv_slice_width(cout, io.W, io.H, io.stride, {32, 16});
        const int nz = CP / coutp;
        if (wslices->empty()) wslices->push_back(add_w(taps, bias, cin, io.cinp, cout, coutp, nz));
        {
            GConvShape s;
            s.coutp = coutp;
            s.in = io.in;
            s.act = io.act;
            s.nres = io.r1.buf ? 1 : 0;
            s.stride = io.stride;
            GConvParams p;
            p.H = io.H;
            p.W = io.W;
            p.IH = io.IH;
            p.IW = io.IW;
            p.cinp = io.cinp;
            p.src_off = io.src_off;
            p.src_stride = io.src_stride;
            p.dst_stride = io.dst_stride;
            p.r1_stride = io.r1_stride;
            p.slope = io.slope;
            add_op((*wslices)[0], s, nz, io.src, io.src_in, io.dst, io.r1, p);
        }
    };
    // A transposed conv's row phase: one dispatch of R * CHP channels.
    const auto phase = [&](int w, int R, int CHP, int sy, Ref src, int cinp, int src_stride, int h, int ww, Ref dst) {
        GConvShape s;
        s.coutp = R * CHP;
        s.out = R == 2 ? 4 : 5;
        GConvParams p;
        p.H = p.IH = h;
        p.W = p.IW = ww;
        p.cinp = cinp;
        p.src_stride = src_stride;
        p.dst_stride = CHP;
        p.sy = sy;
        add_op(w, s, 1, src, false, dst, Ref{}, p);
    };
    const auto rstep = [&](Kind k, const RParams &rp, uint32_t gx, uint32_t gy, Ref dst) {
        Step st;
        st.kind = k;
        st.idx = (int)r->rp_.size();
        st.gx = gx;
        st.gy = gy;
        st.dst = dst;
        st.label = k == GATHER ? "gather s" + std::to_string(rp.v[9]) : k == MERGE ? "merge" : "final";
        r->rp_.push_back(rp);
        r->steps_.push_back(st);
    };
    const auto rbase = [&](RParams &rp) {
        rp.v[0] = W;
        rp.v[1] = H;
        rp.v[2] = F;
        rp.v[3] = FP;
        rp.v[4] = (int32_t)(ofeat[0] / 2);
        rp.v[5] = (int32_t)(ofeat[1] / 2);
    };
    const auto rblock = [&](RParams &rp, int slot, int b, VkDeviceSize off) {
        rp.v[16 + 3 * slot] = (int32_t)(off / 2);
        rp.v[17 + 3 * slot] = plan.blocks[b].s;
        rp.v[18 + 3 * slot] = kp[b];
    };

    // ---- the encoder, per frame
    if (F) {
        const std::vector<Conv> &enc = plan.enc;
        std::vector<std::vector<int>> ew(enc.size());
        const Conv &tc = enc.back();
        if (2 * FP > 64) return refuse_(why, "RIFE encoder features"), nullptr;
        int wph[2];
        for (int py = 0; py < 2; py++) {
            std::vector<uint16_t> w;
            std::vector<float> b;
            phase_weights(tc, 2, py, FP, &w, &b);
            wph[py] = add_w(w, b, tc.cin, pad16(tc.cin), 2 * FP, 2 * FP, 1);
        }
        for (int k = 0; k < 2; k++) {
            Ref cur;
            int curc = 0;
            for (size_t i = 0; i + 1 < enc.size(); i++) {
                const Conv &c = enc[i];
                const Ref dst = sref(oH[i % 2], hbytes);
                ConvIO io;
                io.act = c.lrelu ? 2 : 0;
                io.slope = c.slope;
                io.H = H / 2;
                io.W = W / 2;
                io.dst = dst;
                io.dst_stride = pad16(c.cout);
                if (i == 0) {
                    if (c.cin != 3) return refuse_(why, "RIFE encoder input"), nullptr;
                    io.in = 1;
                    io.src_in = true;
                    io.src = Ref{r->in_.buf, 0, VK_WHOLE_SIZE};
                    io.src_off = (int)(k * 3 * plane);
                    io.stride = 2;
                    io.IH = H;
                    io.IW = W;
                    io.cinp = 16;
                } else {
                    io.src = cur;
                    io.src_stride = curc;
                    io.IH = H / 2;
                    io.IW = W / 2;
                    io.cinp = pad16(c.cin);
                }
                conv(conv_taps(c, nullptr), c.b, c.cin, c.cout, io, &ew[i]);
                cur = dst;
                curc = pad16(c.cout);
            }
            for (int py = 0; py < 2; py++)
                phase(wph[py], 2, FP, py, cur, curc, curc, H / 2, W / 2, pref(ofeat[k], plane * FP * 2));
        }
    }

    // ---- the blocks
    int prevK = 0;
    for (int b = 0; b < nb; b++) {
        const ajinative::RifeBlockPlan &bp = plan.blocks[b];
        const Conv &c0a = bp.c0a, &c0b = bp.c0b, &last = bp.last;
        const int c = c0b.cout, s = bp.s, KP = kp[b];
        const int bw = W / s, bh = H / s, cinp = pad16(c0a.cin);
        if (pad8(c0a.cin) > 64) return refuse_(why, "RIFE block input channels"), nullptr;
        if (4 * KP > 64) return refuse_(why, "RIFE block output channels"), nullptr;
        const VkDeviceSize pxb = (VkDeviceSize)bw * bh;
        const Ref X = sref(oX, pxb * cinp * 2), T1 = sref(oT1, pxb / 4 * pad16(c0a.cout) * 2),
                  A = sref(oA, pxb / 16 * pad16(c) * 2), B = sref(oB, pxb / 16 * pad16(c) * 2);
        std::vector<int> w0a, w0b, wl[4];
        std::vector<std::vector<int>> wres(bp.res.size());
        std::vector<uint16_t> tl[4];
        std::vector<float> bl[4];
        for (int sy = 0; sy < 4; sy++) phase_weights(last, 4, sy, KP, &tl[sy], &bl[sy]);
        for (int pass = 0; pass < (ens ? 2 : 1); pass++) {
            const VkDeviceSize odst = pass ? oR : oblk[b];
            // gather
            RParams ga;
            rbase(ga);
            ga.v[6] = b;
            ga.v[7] = bp.feeds ? 1 : 0;
            ga.v[8] = prevK;
            ga.v[9] = s;
            ga.v[10] = 0;
            ga.v[11] = cinp;
            ga.v[12] = c0a.cin;
            ga.v[13] = pass;
            for (int j = 0; j < b; j++) rblock(ga, j, j, oblk[j]);
            rstep(GATHER, ga, (uint32_t)(bw + 15) / 16, (uint32_t)(bh + 7) / 8, X);
            // conv0
            ConvIO i0;
            i0.src = X;
            i0.src_stride = cinp;
            i0.cinp = cinp;
            i0.IH = bh;
            i0.IW = bw;
            i0.H = bh / 2;
            i0.W = bw / 2;
            i0.dst = T1;
            i0.dst_stride = pad16(c0a.cout);
            i0.act = 2;
            i0.stride = 2;
            conv(conv_taps(c0a, nullptr), c0a.b, c0a.cin, c0a.cout, i0, &w0a);
            ConvIO i1;
            i1.src = T1;
            i1.src_stride = pad16(c0a.cout);
            i1.cinp = pad16(c0b.cin);
            i1.IH = bh / 2;
            i1.IW = bw / 2;
            i1.H = bh / 4;
            i1.W = bw / 4;
            i1.dst = A;
            i1.dst_stride = pad16(c);
            i1.act = 2;
            i1.stride = 2;
            conv(conv_taps(c0b, nullptr), c0b.b, c0b.cin, c, i1, &w0b);
            // residual convs: lrelu(conv * beta + x), beta folded
            Ref cur = A, oth = B;
            for (size_t k = 0; k < bp.res.size(); k++) {
                std::vector<float> bb(c);
                for (int o = 0; o < c; o++) bb[o] = bp.res[k].b[o] * bp.beta[k][o];
                ConvIO ir;
                ir.src = cur;
                ir.src_stride = pad16(c);
                ir.cinp = pad16(c);
                ir.IH = ir.H = bh / 4;
                ir.IW = ir.W = bw / 4;
                ir.dst = oth;
                ir.dst_stride = pad16(c);
                ir.r1 = cur;
                ir.r1_stride = pad16(c);
                ir.act = 2;
                conv(conv_taps(bp.res[k], &bp.beta[k]), bb, c, c, ir, &wres[k]);
                std::swap(cur, oth);
            }
            // lastconv + depth-to-space
            for (int sy = 0; sy < 4; sy++) {
                if (wl[sy].empty()) wl[sy].push_back(add_w(tl[sy], bl[sy], c, pad16(c), 4 * KP, 4 * KP, 1));
                phase(wl[sy][0], 4, KP, sy, cur, pad16(c), pad16(c), bh / 4, bw / 4, pref(odst, pxb * KP * 2));
            }
        }
        if (ens) {
            RParams mg;
            rbase(mg);
            rblock(mg, 0, b, oblk[b]);
            mg.v[14] = (int32_t)(oR / 2);
            const uint32_t gy = (uint32_t)((pxb + 4096 * 8 - 1) / (4096 * 8));
            rstep(MERGE, mg, 256, gy, Ref{});
        }
        prevK = bp.K;
    }
    {
        RParams fi;
        rbase(fi);
        fi.v[6] = nb;
        for (int j = 0; j < nb; j++) rblock(fi, j, j, oblk[j]);
        rstep(FINAL, fi, (uint32_t)(W / 2 + 15) / 16, (uint32_t)(H + 7) / 8, Ref{});
    }

    // ---- uploads, pipelines, descriptor sets, variants
    if (!dev.make_buffer(&r->wts_, wall.size() * 2, false, why) || !dev.make_buffer(&r->bias_, ball.size() * 2, false, why) ||
        !dev.upload(r->wts_, wall.data(), wall.size() * 2, why) || !dev.upload(r->bias_, ball.data(), ball.size() * 2, why))
        return nullptr;
    for (Op &o : r->ops_) {
        std::string w2;
        if (!gconv_prepare(dev, &o, &w2)) return refuse_(why, "kernel: " + w2), nullptr;
    }
    std::vector<GConvParams> gp;
    for (const Op &o : r->ops_) gp.push_back(o.p);
    if (!dev.make_buffer(&r->gprm_, gp.size() * sizeof(GConvParams), false, why) ||
        !dev.upload(r->gprm_, gp.data(), gp.size() * sizeof(GConvParams), why) ||
        !dev.make_buffer(&r->rprm_, r->rp_.size() * sizeof(RParams), false, why) ||
        !dev.upload(r->rprm_, r->rp_.data(), r->rp_.size() * sizeof(RParams), why))
        return nullptr;
    for (Step &st : r->steps_) {
        if (st.kind == GCONV) continue;
        std::string m = st.kind == MERGE ? "rife_merge" : "rife_final";
        if (st.kind == GATHER) {
            // the specialization (rife.comp): as native/rife_kernels.cu picks its template
            const RParams &q = r->rp_[st.idx];
            const bool b0 = q.v[6] == 0, rev = q.v[13] != 0;
            const int nm = b0 ? 1 : (q.v[7] ? q.v[8] - 4 : 1), kv = b0 ? 1 : q.v[18 + 3 * (q.v[6] - 1)] / 8;
            m = "rife_gather_f" + std::to_string(F) + "_m" + std::to_string(nm) + "_k" + std::to_string(kv) + "_t" +
                std::to_string(q.v[9] == 1 ? 1 : 2) + "_b" + std::to_string(b0 ? 1 : 0) + "_r" + std::to_string(rev ? 1 : 0);
            if (!spirv_find(m.c_str())) return refuse_(why, "no RIFE gather kernel for this feature layout (" + m + ")"), nullptr;
            st.label += " " + m.substr(12);
        }
        if (!(st.pipe = dev.pipeline(m.c_str(), 0, 0, why))) return nullptr;
    }
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(r->steps_.size() * Device::kBindings)};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = (uint32_t)r->steps_.size();
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev.device(), &pi, nullptr, &r->pool_) != VK_SUCCESS) return refuse_(why, "descriptor pool"), nullptr;
    }
    for (Step &st : r->steps_) {
        VkDescriptorSetLayout l = dev.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = r->pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev.device(), &ai, &st.set) != VK_SUCCESS) return refuse_(why, "descriptor set"), nullptr;
        if (st.kind == GCONV) r->ops_[st.idx].set = st.set;
    }
    r->step_sets(r->in_.buf, r->out_.buf);
    std::vector<GConvOp *> gops;
    for (Op &o : r->ops_) gops.push_back(&o);
    if (!gconv_choose(dev, gops, why)) return nullptr;

    r->bytes_ = r->in_.size + r->out_.size + r->pers_.size + r->scr_.size + r->wts_.size + r->bias_.size + r->gprm_.size + r->rprm_.size;
    std::string sc;
    for (int b = 0; b < nb; b++) sc += (b ? "/" : "") + std::to_string(plan.blocks[b].s);
    char buf[200];
    std::snprintf(buf, sizeof buf, "%d blocks, scales %s, %d feature channels%s; Vulkan %s", nb, sc.c_str(), F,
                  ens ? ", ensemble" : "", family_name(fam));
    r->what_ = buf;
    return std::unique_ptr<Rife>(r.release());
}

}  // namespace

bool Rife::run(Device &dev, std::string *why) const
{
    VkCommandBuffer cb = dev.begin();
    record(cb);
    return dev.submit_wait(cb, why);
}

std::unique_ptr<Rife> Rife::create(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why)
{
    return VkRife::build(dev, g, W, H, why);
}

}  // namespace ajivk
