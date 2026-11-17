// vk/tfix.cpp — see tfix.h. The graph reading and the step order are native/tfix.cpp's (see its
// header comment for the model); the supports run one after another on one queue.
#include "vk/tfix.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "vk/gconv.h"
#include "vk/kernels.h"

namespace ajivk {
namespace {

using namespace ajinative::onnxread;

constexpr int NF = 7, CEN = 3;
const int kSupports[6] = {0, 1, 2, 4, 5, 6};
const float kDt[6] = {-1.f, -2.f / 3.f, -1.f / 3.f, 1.f / 3.f, 2.f / 3.f, 1.f};

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

struct Ref {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize off = 0, size = 0;
};

// A map into the arena (tfix.comp): byte offset of its buffer, elements per pixel, first channel.
struct TMap {
    VkDeviceSize bytes = 0;
    int stride = 0, off = 0;
    bool f32 = false;
    bool none = true;
    int pos = -1;              // a window position: entry sel[pos] of a stack, fbytes apart
    VkDeviceSize fbytes = 0;
};
TMap hm(VkDeviceSize bytes, int stride, int off = 0) { return TMap{bytes, stride, off, false, false, -1, 0}; }
TMap fm(VkDeviceSize bytes, int stride, int off = 0) { return TMap{bytes, stride, off, true, false, -1, 0}; }
// window position pos's entry of the stack at bytes (entries fbytes apart)
TMap hw(VkDeviceSize bytes, VkDeviceSize fbytes, int pos, int stride, int off = 0)
{
    TMap m = hm(bytes, stride, off);
    m.pos = pos;
    m.fbytes = fbytes;
    return m;
}

struct TParams {
    int32_t v[64] = {};
    void map(int k, const TMap &m)
    {
        const int at = k < 4 ? 4 * k : 24;
        v[at] = m.none ? -1 : (int32_t)(m.f32 ? m.bytes / 4 : m.bytes / 2);
        v[at + 1] = m.stride;
        v[at + 2] = m.off;
        v[at + 3] = m.f32 ? 1 : 0;
        v[32 + 2 * k] = m.pos;
        v[33 + 2 * k] = (int32_t)(m.f32 ? m.fbytes / 4 : m.fbytes / 2);
    }
    void f(int k, float x) { std::memcpy(&v[k], &x, 4); }
};

class VkTfix : public TemporalNet {
public:
    explicit VkTfix(Device &d) : dev_(d) {}
    ~VkTfix() override
    {
        if (pool_) vkDestroyDescriptorPool(dev_.device(), pool_, nullptr);
        if (!io_external_) {
            dev_.destroy(in_);
            dev_.destroy(out_);
        }
        for (Buffer *b : {&ar_, &wts_, &bias_, &fw_, &gprm_, &tprm_, &sel_}) dev_.destroy(*b);
    }
    static std::unique_ptr<TemporalNet> build(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why);

    int frames() const override { return NF; }
    const Buffer &input() const override { return in_; }
    const Buffer &output() const override { return out_; }
    void bind_io(VkBuffer in, VkBuffer out) override;
    void record(VkCommandBuffer cb) const override;
    void record_encode(VkCommandBuffer cb, int entry, int frame) const override;
    void record_cached(VkCommandBuffer cb, const int *ent, const int *cut_slot, float thresh, float inv_area) const override;
    void bind_diff(VkBuffer diff) override;
    std::string profile(Device &dev) const override;
    std::string describe() const override { return what_; }
    size_t device_bytes() const override { return bytes_; }

private:
    struct Op : GConvOp {
        int w = 0;
        Ref src, dst, r1;
        bool src_in = false;
    };
    struct W1 {
        VkDeviceSize woff = 0, wsize = 0, boff = 0, bsize = 0;
    };
    enum Kind { GCONV, TX, ZERO };
    struct Step {
        Kind kind = GCONV;
        int idx = 0;  // GCONV: into ops_; TX: its TParams entry
        int pass = 0;
        uint32_t gx = 1, gy = 1;
        VkDeviceSize zoff = 0, zsize = 0;  // ZERO: the arena range
        VkPipeline pipe = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::string label;
    };
    void record_step(VkCommandBuffer cb, const Step &st) const;
    void record_range(VkCommandBuffer cb, size_t b, size_t e) const;
    void step_sets(VkBuffer in, VkBuffer out);

    Device &dev_;
    int W_ = 0, H_ = 0;
    Buffer in_, out_, ar_, wts_, bias_, fw_, gprm_, tprm_, sel_;
    VkBuffer diff_ = VK_NULL_HANDLE;
    bool io_external_ = false;
    // steps: per cache entry its encoder [enc_b, enc_e), whose first conv (op enc_op) and frame
    // map (TParams enc_tx) read the input frame record_encode patches in; the select step; the
    // window [win_b, win_e)
    size_t enc_b[NF] = {}, enc_e[NF] = {}, win_b = 0, win_e = 0, sel_step = 0;
    int enc_op[NF] = {}, enc_tx[NF] = {};
    int plane3 = 0;  // halfs per input frame
    std::vector<W1> wv_;
    std::vector<Op> ops_;
    std::vector<TParams> tp_;
    std::vector<Step> steps_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    size_t bytes_ = 0;
    std::string what_;
};

void VkTfix::step_sets(VkBuffer in, VkBuffer out)
{
    for (const Step &st : steps_) {
        if (st.kind == ZERO) continue;
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
            bi[1] = {sel_.buf, 0, VK_WHOLE_SIZE};
            bi[2] = {ar_.buf, 0, VK_WHOLE_SIZE};
            bi[3] = {out, 0, VK_WHOLE_SIZE};
            if (fw_.buf) bi[4] = {fw_.buf, 0, VK_WHOLE_SIZE};
            if (diff_) bi[5] = {diff_, 0, VK_WHOLE_SIZE};
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

void VkTfix::bind_io(VkBuffer in, VkBuffer out)
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

void VkTfix::record_step(VkCommandBuffer cb, const Step &st) const
{
    if (st.kind == GCONV) {
        const Op &o = ops_[st.idx];
        gconv_record(dev_, cb, o, o.variant);
        return;
    }
    if (st.kind == ZERO) {
        vkCmdFillBuffer(cb, ar_.buf, st.zoff, st.zsize, 0);
        return;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev_.pipeline_layout(), 0, 1, &st.set, 0, nullptr);
    const int32_t pc[4] = {st.idx, 0, 0, 0};
    vkCmdPushConstants(cb, dev_.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, st.gx, st.gy, 1);
}

// a barrier between any compute or transfer work and what follows
void full_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                       VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VkTfix::record_range(VkCommandBuffer cb, size_t b, size_t e) const
{
    for (size_t i = b; i < e; i++) {
        full_barrier(cb);
        record_step(cb, steps_[i]);
    }
}

void VkTfix::record_encode(VkCommandBuffer cb, int entry, int frame) const
{
    const int32_t off = frame * plane3;
    full_barrier(cb);
    vkCmdUpdateBuffer(cb, gprm_.buf, (VkDeviceSize)enc_op[entry] * sizeof(GConvParams) + offsetof(GConvParams, src_off), 4, &off);
    vkCmdUpdateBuffer(cb, tprm_.buf, (VkDeviceSize)enc_tx[entry] * sizeof(TParams) + 29 * 4, 4, &off);
    record_range(cb, enc_b[entry], enc_e[entry]);
}

void VkTfix::record_cached(VkCommandBuffer cb, const int *ent, const int *cut_slot, float thresh, float inv_area) const
{
    int32_t v[19];
    for (int z = 0; z < 8; z++) v[z] = z < NF ? ent[z] : 0;
    for (int z = 0; z < 8; z++) v[8 + z] = z < NF ? cut_slot[z] : -1;
    v[16] = NF;
    std::memcpy(&v[17], &thresh, 4);
    std::memcpy(&v[18], &inv_area, 4);
    full_barrier(cb);
    vkCmdUpdateBuffer(cb, tprm_.buf, (VkDeviceSize)steps_[sel_step].idx * sizeof(TParams) + 40 * 4, sizeof v, v);
    record_range(cb, sel_step, sel_step + 1);
    record_range(cb, win_b, win_e);
}

void VkTfix::record(VkCommandBuffer cb) const
{
    // every frame of the input into the entry of its position, then the window in order
    const int32_t id[NF] = {0, 1, 2, 3, 4, 5, 6};
    full_barrier(cb);
    vkCmdUpdateBuffer(cb, sel_.buf, 0, sizeof id, id);
    for (int i = 0; i < NF; i++) record_encode(cb, i, i);
    record_range(cb, win_b, win_e);
}

void VkTfix::bind_diff(VkBuffer diff)
{
    diff_ = diff;
    step_sets(in_.buf, out_.buf);
}

std::string VkTfix::profile(Device &dev) const
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
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    (void)mb;
    const int32_t id[NF] = {0, 1, 2, 3, 4, 5, 6};
    vkCmdUpdateBuffer(cb, sel_.buf, 0, sizeof id, id);
    for (int i = 0; i < NF; i++) {
        const int32_t off = i * plane3;
        full_barrier(cb);
        vkCmdUpdateBuffer(cb, gprm_.buf, (VkDeviceSize)enc_op[i] * sizeof(GConvParams) + offsetof(GConvParams, src_off), 4, &off);
        vkCmdUpdateBuffer(cb, tprm_.buf, (VkDeviceSize)enc_tx[i] * sizeof(TParams) + 29 * 4, 4, &off);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    for (size_t i = 0; i < steps_.size(); i++) {
        full_barrier(cb);
        if (i != sel_step) record_step(cb, steps_[i]);
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

bool cvt(const Reader &rd, const std::string &name, Conv *c, bool lrelu, int stride, std::string *why)
{
    if (!rd.conv(name, c, why)) return false;
    if (c->lrelu != lrelu || c->stride != stride) return refuse_(why, name + ": unexpected activation or stride");
    return true;
}

// The model's convs: [channels in, out, padded], read and checked as native/tfix.cpp does.
struct Model {
    Conv e0, e1, e2, ep, h0, hh1, h2c;
    struct Stage {
        int c = 0, ca = 0, h = 0, w = 0, ha = 0, wa = 0, hq = 0, wq = 0, xs = 0;
        bool ds = false;
        Conv c00, c01, res[4], out;
        std::vector<float> beta[4];
        std::vector<float> dw[4], dwb[4], pw[4], pwb[4];  // dw [C][9]; pw [K][N] with beta folded
    } st[3];
    float thresh = 0.6f, gslope = 12.f, minsup = 1.f, cslope = 4.f, eps = 1e-6f;
};

bool read_model(const ajinative::OnnxGraph &g, int w, int h, Model *m, std::string *why)
{
    Reader rd(g);
    if (!rd.node("/encode/cnn0/Conv") || !rd.node("/aligner/blocks.0/out_conv/Conv") || !rd.node("/conf_head/net/net.2/Conv"))
        return refuse_(why, "not a TemporalFix graph");
    if (w % 8 || h % 8 || w < 64 || h < 64) return refuse_(why, "TemporalFix needs a frame size that is a multiple of 8");
    if (!cvt(rd, "/encode/cnn0/Conv", &m->e0, true, 2, why) || m->e0.cin != 3 || m->e0.cout != 16) return false;
    if (!cvt(rd, "/encode/cnn1/Conv", &m->e1, true, 1, why) || !cvt(rd, "/encode/cnn2/Conv", &m->e2, true, 1, why)) return false;
    if (!cvt(rd, "/encode/proj/Conv", &m->ep, false, 1, why) || m->ep.cout != 4) return refuse_(why, "TemporalFix encoder");
    const int scales[3] = {8, 4, 2};
    for (int k = 0; k < 3; k++) {
        Model::Stage &S = m->st[k];
        const std::string B = "/aligner/blocks." + std::to_string(k) + "/";
        S.h = (h + scales[k] - 1) / scales[k];
        S.w = (w + scales[k] - 1) / scales[k];
        S.ha = (S.h + 1) / 2;
        S.wa = (S.w + 1) / 2;
        S.hq = (S.ha + 1) / 2;
        S.wq = (S.wa + 1) / 2;
        S.xs = k ? 32 : 16;
        if (!cvt(rd, B + "conv0/conv0.0/conv0.0.0/Conv", &S.c00, true, 2, why) || S.c00.cin != (k ? 26 : 15))
            return refuse_(why, B + "conv0");
        S.ca = S.c00.cout;
        if (!cvt(rd, B + "conv0/conv0.1/conv0.1.0/Conv", &S.c01, true, 2, why)) return false;
        S.c = S.c01.cout;
        if (S.c % 16) return refuse_(why, B + "channels");
        S.ds = rd.node(B + "convblock/convblock.0/dw/Conv") != nullptr;
        for (int j = 0; j < 4; j++) {
            const std::string R = B + "convblock/convblock." + std::to_string(j) + "/";
            std::vector<float> v, bb;
            std::vector<int64_t> d;
            if (!init_input(rd, rd.node(R + "Mul"), &S.beta[j], &d) || (int)S.beta[j].size() != S.c) return refuse_(why, R + "beta");
            if (!rd.node(R + "relu/LeakyRelu")) return refuse_(why, R + "activation");
            if (!S.ds) {
                if (!cvt(rd, R + "conv/Conv", &S.res[j], false, 1, why)) return false;
                continue;
            }
            const ajinative::OnnxNode *dn = rd.node(R + "dw/Conv");
            if (!tensor_floats(rd, dn, 1, &v, &d) || d != std::vector<int64_t>{S.c, 1, 3, 3} || dn->in.size() < 3 || !rd.floats(dn->in[2], &bb))
                return refuse_(why, R + "dw");
            S.dw[j] = v;
            S.dwb[j] = bb;
            std::vector<float> wm, bm;
            int co = 0;
            if (!conv1x1_matrix(rd, R + "pw/Conv", S.c, nullptr, &wm, &bm, &co, why) || co != S.c || (int)bm.size() != S.c)
                return refuse_(why, R + "pw");
            for (int kk = 0; kk < S.c; kk++)
                for (int o = 0; o < S.c; o++) wm[(size_t)kk * S.c + o] *= S.beta[j][o];
            for (int o = 0; o < S.c; o++) bm[o] *= S.beta[j][o];
            S.pw[j] = wm;
            S.pwb[j] = bm;
        }
        if (!cvt(rd, B + "out_conv/Conv", &S.out, false, 1, why) || S.out.cout != 11) return refuse_(why, B + "out_conv");
    }
    if (m->st[2].h != (h + 1) / 2 || m->st[2].w != (w + 1) / 2) return refuse_(why, "TemporalFix stage sizes");
    if (!cvt(rd, "/conf_head/net/net.0/net.0.0/Conv", &m->h0, true, 1, why) || m->h0.cin != 25) return refuse_(why, "TemporalFix conf head");
    if (!cvt(rd, "/conf_head/net/net.1/net.1.0/Conv", &m->hh1, true, 1, why)) return false;
    if (!cvt(rd, "/conf_head/net/net.2/Conv", &m->h2c, false, 1, why) || m->h2c.cout != 1) return refuse_(why, "TemporalFix conf head");
    std::vector<float> v;
    if (init_input(rd, rd.node("/Sub_6"), &v) && v.size() == 1) m->thresh = v[0];
    if (init_input(rd, rd.node("/Mul_16"), &v) && v.size() == 1) m->gslope = v[0];
    if (init_input(rd, rd.node("/Mul_19"), &v) && v.size() == 1) m->cslope = v[0];
    return true;
}

std::unique_ptr<TemporalNet> VkTfix::build(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why)
{
    Model md;
    if (!read_model(g, W, H, &md, why)) return nullptr;
    std::unique_ptr<VkTfix> f(new VkTfix(dev));
    f->W_ = W;
    f->H_ = H;
    const Family fam = dev.family();
    const int h1 = (H + 1) / 2, w1 = (W + 1) / 2;
    const VkDeviceSize M = (VkDeviceSize)W * H, M1 = (VkDeviceSize)h1 * w1;
    const VkDeviceSize align = std::max<VkDeviceSize>(dev.info().storage_align, 16);
    VkDeviceSize ar = 0;
    const auto place = [&](VkDeviceSize bytes) {
        const VkDeviceSize o = ar;
        ar += (bytes + align - 1) / align * align;
        return o;
    };

    // ---- the arena: per frame its 8-channel map and pyramid, the encoder's scratch, one chain
    const VkDeviceSize frb = M * 8 * 2;
    VkDeviceSize FR[NF], PY[3][NF], pyb[3];
    for (auto &o : FR) o = place(frb);
    const VkDeviceSize E1 = place(M1 * 16 * 2), E2 = place(M1 * 16 * 2), EP = place(M1 * 16 * 2);
    for (int k = 0; k < 3; k++) {
        pyb[k] = (VkDeviceSize)md.st[k].h * md.st[k].w * 8 * 2;
        for (auto &o : PY[k]) o = place(pyb[k]);
    }
    struct Sb {
        VkDeviceSize X, A, B, B2, T, O, ST, F;
        VkDeviceSize nX, nA, nB, nO, nST;
    } sb[3];
    for (int k = 0; k < 3; k++) {
        const Model::Stage &S = md.st[k];
        const VkDeviceSize Ms = (VkDeviceSize)S.h * S.w, Ma = (VkDeviceSize)S.ha * S.wa, Mq = (VkDeviceSize)S.hq * S.wq;
        Sb &b = sb[k];
        b.nX = Ms * S.xs * 2;
        b.nA = Ma * pad16(S.ca) * 2;
        b.nB = Mq * S.c * 2;
        b.nO = Mq * 16 * 2;
        b.nST = Ms * 16 * 2;
        b.X = place(b.nX);
        b.A = place(b.nA);
        b.B = place(b.nB);
        b.B2 = place(b.nB);
        b.T = place(b.nB);
        b.O = place(b.nO);
        b.ST = place(b.nST);
        b.F = place(Ms * 2 * 4);
    }
    const VkDeviceSize FULL = place(M * 8 * 2), FF = place(M * 2 * 4), FH = place(M1 * 2 * 4), WH = place(M1 * 8 * 2),
                      RCH = place(M1 * 16 * 2), WF = place(M1 * 8 * 2), CIN = place(M1 * 32 * 2), C1 = place(M1 * 32 * 2),
                      C2 = place(M1 * 32 * 2), C3 = place(M1 * 16 * 2), ACC = place(M * 8 * 4);
    const uint64_t lim = dev.info().max_storage_range;
    if (lim && ar > lim)
        return refuse_(why, "TemporalFix at " + std::to_string(W) + "x" + std::to_string(H) + " needs a " + std::to_string(ar >> 20) +
                                " MB buffer, over this device's " + std::to_string(lim >> 20) + " MB storage-buffer limit"),
               nullptr;
    if (ar >= (1ull << 31)) return refuse_(why, "TemporalFix at this size needs a buffer over 2 GB"), nullptr;
    if (!dev.make_buffer(&f->in_, (VkDeviceSize)3 * NF * M * 2, false, why) || !dev.make_buffer(&f->out_, 3 * M * 2, false, why) ||
        !dev.make_buffer(&f->ar_, ar, false, why) || !dev.make_buffer(&f->sel_, 64, false, why))
        return nullptr;
    const VkBuffer A = f->ar_.buf;
    const auto ref = [&](VkDeviceSize off, VkDeviceSize bytes) { return Ref{A, off, bytes}; };

    // ---- weights: the 3x3 convs packed for the family; the dw / pw ones as fp32
    std::vector<uint16_t> wall, ball;
    std::vector<float> fwall;
    const auto add_w = [&](const std::vector<uint16_t> &taps, const std::vector<float> &bias, int cin, int cinp, int cout, int coutp, int nz) {
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
        f->wv_.push_back(w);
        return (int)f->wv_.size() - 1;
    };
    std::map<const std::vector<float> *, int> fcache;  // each fp32 tensor once
    const auto add_f = [&](const std::vector<float> &v) {
        auto it = fcache.find(&v);
        if (it != fcache.end()) return it->second;
        const int o = (int)fwall.size();
        fwall.insert(fwall.end(), v.begin(), v.end());
        fcache[&v] = o;
        return o;
    };

    // ---- steps
    const auto step_label = [](const GConvShape &s, const GConvParams &p, int nz) {
        char lb[96];
        std::snprintf(lb, sizeof lb, "conv %d->%d %dx%d%s", p.cinp, s.coutp * nz, p.W, p.H, s.stride == 2 ? " s2" : "");
        return std::string(lb);
    };
    struct CIO {
        Ref src, dst, r1;
        bool src_in = false;
        int in = 0, act = 0, stride = 1;
        int src_off = 0, src_stride = 0, dst_stride = 0, r1_stride = 0;
        int IH = 0, IW = 0, H = 0, W = 0, cinp = 16;
    };
    // A 3x3 conv (weights wi, from conv c with scale sc): one dispatch of output slices.
    std::map<const Conv *, int> wcache;
    const auto conv = [&](const Conv &c, const std::vector<float> *sc, const CIO &io) {
        const int coutp = io.in == 0 ? gconv_slice_width(c.cout, io.W, io.H, io.stride, {64, 48, 32, 16})
                                     : gconv_slice_width(c.cout, io.W, io.H, io.stride, {32, 16});
        const int nz = pad16(c.cout) / coutp;
        auto it = wcache.find(&c);
        int wi;
        if (it != wcache.end()) {
            wi = it->second;
        } else {
            std::vector<float> b = c.b;
            if (sc)
                for (size_t i = 0; i < b.size(); i++) b[i] *= (*sc)[i];
            wi = add_w(conv_taps(c, sc), b, c.cin, io.cinp, c.cout, coutp, nz);
            wcache[&c] = wi;
        }
        Op o;
        o.s.coutp = coutp;
        o.s.in = io.in;
        o.s.act = io.act;
        o.s.nres = io.r1.buf ? 1 : 0;
        o.s.stride = io.stride;
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
        o.w = wi;
        o.src = io.src;
        o.src_in = io.src_in;
        o.dst = io.dst;
        o.r1 = io.r1;
        o.entry = (int)f->ops_.size();
        f->ops_.push_back(o);
        Step st;
        st.kind = GCONV;
        st.idx = o.entry;
        st.label = step_label(o.s, o.p, nz);
        f->steps_.push_back(st);
    };
    const auto io3 = [](Ref src, int sstride, int IH, int IW, Ref dst, int dstride, int Ho, int Wo, int act, int stride) {
        CIO io;
        io.src = src;
        io.src_stride = sstride;
        io.cinp = sstride;
        io.IH = IH;
        io.IW = IW;
        io.dst = dst;
        io.dst_stride = dstride;
        io.H = Ho;
        io.W = Wo;
        io.act = act;
        io.stride = stride;
        return io;
    };
    static const char *kPass[] = {"", "dw", "pw", "resize", "down", "warp", "frame", "full", "fill", "conf_in", "accum", "final", "select"};
    const auto tx = [&](int pass, const TParams &p, long long threads) {
        Step st;
        st.kind = TX;
        st.pass = pass;
        st.idx = (int)f->tp_.size();
        const long long groups = (threads + 255) / 256;
        st.gx = (uint32_t)std::min<long long>(groups, 4096);
        st.gy = (uint32_t)((groups + st.gx - 1) / st.gx);
        st.label = kPass[pass];
        f->tp_.push_back(p);
        f->steps_.push_back(st);
    };
    const auto resize = [&](TMap src, int ih, int iw, TMap dst, int oh, int ow, int C, float mx, float my, bool add) {
        TParams p;
        p.map(0, src);
        p.map(1, dst);
        p.v[16] = ih;
        p.v[17] = iw;
        p.v[18] = oh;
        p.v[19] = ow;
        p.v[20] = C;
        p.v[21] = add ? 1 : 0;
        p.f(22, mx);
        p.f(23, my);
        tx(3, p, (long long)oh * ow);
    };
    const auto down = [&](TMap src, int ih, int iw, TMap dst, int oh, int ow, int C, float mx, float my) {
        TParams p;
        p.map(0, src);
        p.map(1, dst);
        p.v[16] = ih;
        p.v[17] = iw;
        p.v[18] = oh;
        p.v[19] = ow;
        p.v[20] = C;
        p.f(22, mx);
        p.f(23, my);
        tx(4, p, (long long)oh * ow);
    };
    const auto warp = [&](TMap src, int C, TMap flow, int h, int w, TMap dst, TMap valid) {
        TParams p;
        p.map(0, src);
        p.map(1, dst);
        p.map(2, flow);
        p.map(3, valid);
        p.v[16] = h;
        p.v[17] = w;
        p.v[20] = C;
        tx(5, p, (long long)h * w);
    };
    const auto fill = [&](TMap dst, int h, int w, float v) {
        TParams p;
        p.map(0, dst);
        p.v[16] = h;
        p.v[17] = w;
        p.f(22, v);
        tx(8, p, (long long)h * w);
    };

    // the encoder of every frame: rgb into FR channels 0..2, features into 3..6, then its pyramid
    const VkDeviceSize plane = M;
    f->plane3 = (int)(3 * plane);
    for (int i = 0; i < NF; i++) {
        f->enc_b[i] = f->steps_.size();
        f->enc_op[i] = (int)f->ops_.size();
        CIO a = io3(Ref{}, 0, H, W, ref(E1, M1 * 16 * 2), 16, h1, w1, 2, 2);
        a.in = 1;
        a.src_in = true;
        a.src = Ref{f->in_.buf, 0, VK_WHOLE_SIZE};
        a.src_off = (int)(i * 3 * plane);
        a.cinp = 16;
        conv(md.e0, nullptr, a);
        conv(md.e1, nullptr, io3(ref(E1, M1 * 32), 16, h1, w1, ref(E2, M1 * 32), 16, h1, w1, 2, 1));
        conv(md.e2, nullptr, io3(ref(E2, M1 * 32), 16, h1, w1, ref(E1, M1 * 32), 16, h1, w1, 2, 1));
        conv(md.ep, nullptr, io3(ref(E1, M1 * 32), 16, h1, w1, ref(EP, M1 * 32), 16, h1, w1, 0, 1));
        {
            TParams p;
            p.map(0, hm(EP, 16));
            p.map(1, hm(FR[i], 8));
            p.v[16] = h1;
            p.v[17] = w1;
            p.v[18] = H;
            p.v[19] = W;
            p.v[29] = (int32_t)(i * 3 * plane);
            f->enc_tx[i] = (int)f->tp_.size();
            tx(6, p, (long long)M);
        }
        for (int k = 0; k < 3; k++) down(hm(FR[i], 8), H, W, hm(PY[k][i], 8), md.st[k].h, md.st[k].w, 7, 1.f, 1.f);
        f->enc_e[i] = f->steps_.size();
    }
    // the select step (record_cached fills its entries and cuts)
    {
        TParams p;
        p.v[56] = NF;
        f->sel_step = f->steps_.size();
        tx(12, p, 32);
    }
    f->win_b = f->steps_.size();
    // the centre's channels of each stage input
    for (int k = 0; k < 3; k++) {
        const Model::Stage &S = md.st[k];
        resize(hw(PY[k][0], pyb[k], CEN, 8, 0), S.h, S.w, hm(sb[k].X, S.xs, 0), S.h, S.w, 3, 1.f, 1.f, false);
        resize(hw(PY[k][0], pyb[k], CEN, 8, 3), S.h, S.w, hm(sb[k].X, S.xs, 6), S.h, S.w, 4, 1.f, 1.f, false);
    }
    {
        Step z;
        z.kind = ZERO;
        z.zoff = ACC;
        z.zsize = M * 8 * 4;
        z.label = "zero";
        f->steps_.push_back(z);
    }
    for (int si = 0; si < 6; si++) {
        const int fr = kSupports[si];
        const float dt = kDt[si];
        for (int k = 0; k < 3; k++) {
            Model::Stage &S = md.st[k];
            const Sb &b = sb[k];
            if (k == 0) {
                resize(hw(PY[k][0], pyb[k], fr, 8, 0), S.h, S.w, hm(b.X, S.xs, 3), S.h, S.w, 3, 1.f, 1.f, false);
                resize(hw(PY[k][0], pyb[k], fr, 8, 3), S.h, S.w, hm(b.X, S.xs, 10), S.h, S.w, 4, 1.f, 1.f, false);
            } else {
                const Model::Stage &P = md.st[k - 1];
                const Sb &pb = sb[k - 1];
                resize(fm(pb.F, 2), P.h, P.w, fm(b.F, 2), S.h, S.w, 2, (float)S.w / P.w, (float)S.h / P.h, false);
                resize(hm(pb.ST, 16), P.h, P.w, hm(b.X, S.xs, 15), S.h, S.w, 9, 1.f, 1.f, false);
                resize(fm(b.F, 2), S.h, S.w, hm(b.X, S.xs, 24), S.h, S.w, 2, 1.f, 1.f, false);
                warp(hw(PY[k][0], pyb[k], fr, 8, 0), 3, fm(b.F, 2), S.h, S.w, hm(b.X, S.xs, 3), TMap{});
                warp(hw(PY[k][0], pyb[k], fr, 8, 3), 4, fm(b.F, 2), S.h, S.w, hm(b.X, S.xs, 10), TMap{});
            }
            fill(hm(b.X, S.xs, 14), S.h, S.w, dt);
            // the MCBlock
            conv(S.c00, nullptr, io3(ref(b.X, b.nX), S.xs, S.h, S.w, ref(b.A, b.nA), pad16(S.ca), S.ha, S.wa, 2, 2));
            {
                CIO a = io3(ref(b.A, b.nA), pad16(S.ca), S.ha, S.wa, ref(b.B, b.nB), S.c, S.hq, S.wq, 2, 2);
                conv(S.c01, nullptr, a);
            }
            VkDeviceSize x = b.B, y = b.B2;
            for (int j = 0; j < 4; j++) {
                if (!S.ds) {
                    CIO a = io3(ref(x, b.nB), S.c, S.hq, S.wq, ref(y, b.nB), S.c, S.hq, S.wq, 2, 1);
                    a.r1 = ref(x, b.nB);
                    a.r1_stride = S.c;
                    conv(S.res[j], &S.beta[j], a);
                } else {
                    TParams p;
                    p.map(0, hm(x, S.c));
                    p.map(1, hm(b.T, S.c));
                    p.v[16] = S.hq;
                    p.v[17] = S.wq;
                    p.v[20] = S.c;
                    p.v[30] = add_f(S.dw[j]);
                    p.v[31] = add_f(S.dwb[j]);
                    tx(1, p, (long long)S.hq * S.wq * (S.c / 8));
                    TParams q;
                    q.map(0, hm(b.T, S.c));
                    q.map(1, hm(y, S.c));
                    q.map(2, hm(x, S.c));
                    q.v[16] = S.hq * S.wq;
                    q.v[20] = S.c;
                    q.v[21] = S.c;
                    q.v[30] = add_f(S.pw[j]);
                    q.v[31] = add_f(S.pwb[j]);
                    tx(2, q, (long long)S.hq * S.wq * (S.c / 8));
                }
                std::swap(x, y);
            }
            conv(S.out, nullptr, io3(ref(x, b.nB), S.c, S.hq, S.wq, ref(b.O, b.nO), 16, S.hq, S.wq, 0, 1));
            resize(hm(b.O, 16, 0), S.hq, S.wq, fm(b.F, 2), S.h, S.w, 2, 1.f, 1.f, k > 0);
            resize(hm(b.O, 16, 2), S.hq, S.wq, hm(b.ST, 16, 0), S.h, S.w, 9, 1.f, 1.f, false);
        }
        // full resolution: flow, logit, the warped support and its validity
        const Model::Stage &L = md.st[2];
        {
            TParams p;
            p.map(0, fm(sb[2].F, 2));
            p.map(1, fm(FF, 2));
            p.map(2, hm(sb[2].ST, 16));
            p.map(3, hw(FR[0], frb, fr, 8));
            p.map(4, hm(FULL, 8));
            p.v[16] = L.h;
            p.v[17] = L.w;
            p.v[18] = H;
            p.v[19] = W;
            tx(7, p, (long long)M);
        }
        // the confidence head at 1/2 on the low-passed results
        down(fm(FF, 2), H, W, fm(FH, 2), h1, w1, 2, (float)w1 / W, (float)h1 / H);
        down(hm(FULL, 8), H, W, hm(WH, 8), h1, w1, 4, 1.f, 1.f);
        resize(hm(WH, 8, 3), h1, w1, hm(RCH, 16, 0), h1, w1, 1, 1.f, 1.f, false);
        warp(hw(PY[2][0], pyb[2], fr, 8, 3), 4, fm(FH, 2), h1, w1, hm(WF, 8, 0), TMap{});
        {
            TParams p;
            p.map(0, hw(PY[2][0], pyb[2], CEN, 8));
            p.map(1, hm(WH, 8));
            p.map(2, hm(WF, 8));
            p.map(3, fm(FH, 2));
            p.map(4, hm(CIN, 32));
            p.v[16] = h1;
            p.v[17] = w1;
            p.f(22, dt);
            tx(9, p, (long long)M1);
        }
        conv(md.h0, nullptr, io3(ref(CIN, M1 * 64), 32, h1, w1, ref(C1, M1 * 64), 32, h1, w1, 2, 1));
        conv(md.hh1, nullptr, io3(ref(C1, M1 * 64), 32, h1, w1, ref(C2, M1 * 64), 32, h1, w1, 2, 1));
        {
            CIO a = io3(ref(C2, M1 * 64), 32, h1, w1, ref(C3, M1 * 32), 16, h1, w1, 0, 1);
            a.r1 = ref(RCH, M1 * 32);
            a.r1_stride = 16;
            conv(md.h2c, nullptr, a);
        }
        {
            TParams p;
            p.map(0, hm(C3, 16, 0));
            p.map(1, hm(FULL, 8));
            p.map(2, fm(ACC, 8));
            p.v[16] = h1;
            p.v[17] = w1;
            p.v[18] = H;
            p.v[19] = W;
            p.v[20] = 4;
            p.f(22, md.thresh);
            p.f(23, md.gslope);
            tx(10, p, (long long)M);
        }
    }
    {
        TParams p;
        p.map(0, hw(FR[0], frb, CEN, 8));
        p.map(2, fm(ACC, 8));
        p.v[18] = H;
        p.v[19] = W;
        p.f(22, md.minsup);
        p.f(23, md.cslope);
        p.f(28, md.eps);
        tx(11, p, (long long)M);
    }
    f->win_e = f->steps_.size();

    // ---- uploads, pipelines, descriptor sets, variants
    if (fwall.empty()) fwall.push_back(0.f);
    if (!dev.make_buffer(&f->wts_, wall.size() * 2, false, why) || !dev.make_buffer(&f->bias_, ball.size() * 2, false, why) ||
        !dev.make_buffer(&f->fw_, fwall.size() * 4, false, why) || !dev.upload(f->wts_, wall.data(), wall.size() * 2, why) ||
        !dev.upload(f->bias_, ball.data(), ball.size() * 2, why) || !dev.upload(f->fw_, fwall.data(), fwall.size() * 4, why))
        return nullptr;
    for (Op &o : f->ops_) {
        std::string w2;
        if (!gconv_prepare(dev, &o, &w2)) return refuse_(why, "kernel: " + w2), nullptr;
    }
    std::vector<GConvParams> gp;
    for (const Op &o : f->ops_) gp.push_back(o.p);
    if (!dev.make_buffer(&f->gprm_, gp.size() * sizeof(GConvParams), false, why) ||
        !dev.upload(f->gprm_, gp.data(), gp.size() * sizeof(GConvParams), why) ||
        !dev.make_buffer(&f->tprm_, f->tp_.size() * sizeof(TParams), false, why) ||
        !dev.upload(f->tprm_, f->tp_.data(), f->tp_.size() * sizeof(TParams), why))
        return nullptr;
    size_t nsets = 0;
    for (Step &st : f->steps_) {
        if (st.kind == ZERO) continue;
        nsets++;
        if (st.kind != TX) continue;
        const std::string m = std::string("tfix_") + kPass[st.pass];
        if (!(st.pipe = dev.pipeline(m.c_str(), 0, 0, why))) return nullptr;
    }
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(nsets * Device::kBindings)};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = (uint32_t)nsets;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev.device(), &pi, nullptr, &f->pool_) != VK_SUCCESS) return refuse_(why, "descriptor pool"), nullptr;
    }
    for (Step &st : f->steps_) {
        if (st.kind == ZERO) continue;
        VkDescriptorSetLayout l = dev.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = f->pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev.device(), &ai, &st.set) != VK_SUCCESS) return refuse_(why, "descriptor set"), nullptr;
        if (st.kind == GCONV) f->ops_[st.idx].set = st.set;
    }
    f->step_sets(f->in_.buf, f->out_.buf);
    std::vector<GConvOp *> gops;
    for (Op &o : f->ops_) gops.push_back(&o);
    if (!gconv_choose(dev, gops, why)) return nullptr;

    f->bytes_ = f->sel_.size + f->in_.size + f->out_.size + f->ar_.size + f->wts_.size + f->bias_.size + f->fw_.size + f->gprm_.size + f->tprm_.size;
    char buf[200];
    std::snprintf(buf, sizeof buf, "TemporalFix, 7 frames, stages %d/%d/%d channels; Vulkan %s", md.st[0].c, md.st[1].c, md.st[2].c,
                  family_name(fam));
    f->what_ = buf;
    return std::unique_ptr<TemporalNet>(f.release());
}

}  // namespace

bool TemporalNet::run(Device &dev, std::string *why) const
{
    VkCommandBuffer cb = dev.begin();
    record(cb);
    return dev.submit_wait(cb, why);
}

std::unique_ptr<TemporalNet> TemporalNet::create(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why)
{
    return VkTfix::build(dev, g, W, H, why);
}

}  // namespace ajivk
