// vk/esrgan.cpp — ESRGAN (RRDBNet, as exported by traiNNer/chaiNNer: /model/model.N node names) on
// the general conv (vk/gconv.h), and ImgNet::create. The plan is native/esrgan.cpp's:
//
// Dense blocks without concatenation: each RDB works in one NHWC buffer of nf + 4*gc channels; conv
// k reads its first nf + (k-1)*gc channels and writes its gc channels into the next slice. The RDB's
// last conv (the export folds the 0.2 into its weights) adds the block input; the third RDB of an
// RRDB also folds the RRDB's own 0.2 and residual (out = 0.2 * (conv + x3) + x) and writes in place
// over the RRDB input, which the next RRDB then reads. The upsampler convs read their input
// nearest-upsampled on the fly. The first conv reads the NCHW tile (or its pixel-unshuffle) and runs
// twice, into the trunk's residual and into the first dense buffer; the last writes NCHW RGB.
//
// Intermediate buffers share one arena by lifetime; weights and biases one buffer each.
#include "vk/imgnet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>

#include "native/lower.h"
#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "native/pack.h"
#include "vk/executor.h"
#include "vk/gconv.h"

namespace ajivk {
namespace {

using namespace ajinative::onnxread;

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

// Buffers laid out in one arena by lifetime (native/netres.h's Lifetimes, on offsets).
struct Arena {
    struct Buf {
        VkDeviceSize *off, bytes;
        int first, last;  // live steps, inclusive
    };
    std::vector<Buf> bufs;
    void add(VkDeviceSize *off, VkDeviceSize bytes, int first, int last, VkDeviceSize align)
    {
        bufs.push_back({off, (bytes + align - 1) / align * align, first, last});
    }
    VkDeviceSize layout()
    {
        std::vector<size_t> order(bufs.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return bufs[a].bytes > bufs[b].bytes; });
        std::vector<size_t> placed;
        VkDeviceSize total = 0;
        for (size_t i : order) {
            Buf &b = bufs[i];
            std::vector<std::pair<VkDeviceSize, VkDeviceSize>> taken;
            for (size_t j : placed)
                if (b.first <= bufs[j].last && bufs[j].first <= b.last) taken.push_back({*bufs[j].off, *bufs[j].off + bufs[j].bytes});
            std::sort(taken.begin(), taken.end());
            VkDeviceSize off = 0;
            for (auto &t : taken) {
                if (off + b.bytes <= t.first) break;
                off = std::max(off, t.second);
            }
            *b.off = off;
            total = std::max(total, off + b.bytes);
            placed.push_back(i);
        }
        return total;
    }
};

// A tensor: a range of the arena (or of an owned buffer).
struct Ref {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize off = 0, size = 0;
};

class Esrgan : public ImgNet {
public:
    explicit Esrgan(Device &d) : dev_(d) {}
    ~Esrgan() override
    {
        if (pool_) vkDestroyDescriptorPool(dev_.device(), pool_, nullptr);
        if (!io_external_) {
            dev_.destroy(in_);
            dev_.destroy(out_);
        }
        for (Buffer *b : {&arena_, &wts_, &bias_, &params_}) dev_.destroy(*b);
    }
    static std::unique_ptr<ImgNet> build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);

    const Buffer &input() const override { return in_; }
    const Buffer &output() const override { return out_; }
    void bind_io(VkBuffer in, VkBuffer out) override;
    bool record(VkCommandBuffer cb, int w, int h) const override;
    void suffix(const std::string &s) override { what_ += s; }
    int out_w() const override { return w_ * scale_; }
    int out_h() const override { return h_ * scale_; }
    size_t device_bytes() const override { return bytes_; }
    size_t weight_bytes() const override { return wbytes_; }
    std::string describe() const override { return what_; }

private:
    struct Conv1 {
        GConvShape s;
        int cinp = 16, nz = 1;
        VkDeviceSize woff = 0, wsize = 0, boff = 0, bsize = 0;
    };
    struct Op : GConvOp {
        int conv = 0;  // into convs_
        Ref src, dst, r1, r2;
    };

    Device &dev_;
    int scale_ = 4, nf = 64, gc = 32, nb = 0, DS = 0, w_ = 0, h_ = 0;
    bool unshuffle_ = false;
    float rrdb_scale_ = 0.2f;
    Buffer in_, out_, arena_, wts_, bias_, params_;
    bool io_external_ = false;  // in_ / out_ are the caller's (bind_io): handles only
    std::vector<Conv1> convs_;
    std::vector<Op> ops_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    size_t bytes_ = 0, wbytes_ = 0;
    std::string what_;
};

void Esrgan::bind_io(VkBuffer in, VkBuffer out)
{
    // the first conv reads the input (binding 0), the last writes the output (binding 2)
    std::vector<VkDescriptorBufferInfo> bi;
    std::vector<VkWriteDescriptorSet> wr;
    bi.reserve(2 * ops_.size());
    for (const Op &o : ops_)
        for (int k = 0; k < 2; k++) {
            const bool hit = k == 0 ? o.src.buf == in_.buf : o.dst.buf == out_.buf;
            if (!hit) continue;
            bi.push_back({k == 0 ? in : out, 0, VK_WHOLE_SIZE});
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = o.set;
            w.dstBinding = k == 0 ? 0u : 2u;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &bi.back();
            wr.push_back(w);
        }
    vkUpdateDescriptorSets(dev_.device(), (uint32_t)wr.size(), wr.data(), 0, nullptr);
    for (Op &o : ops_) {
        if (o.src.buf == in_.buf) o.src = Ref{in, 0, VK_WHOLE_SIZE};
        if (o.dst.buf == out_.buf) o.dst = Ref{out, 0, VK_WHOLE_SIZE};
    }
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

bool Esrgan::record(VkCommandBuffer cb, int w, int h) const
{
    if (w != w_ || h != h_) return false;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (size_t i = 0; i < ops_.size(); i++) {
        if (i) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        gconv_record(dev_, cb, ops_[i], ops_[i].variant);
    }
    return true;
}

std::unique_ptr<ImgNet> Esrgan::build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    Reader rd(g);
    if (!rd.node("/model/model.0/Conv")) return refuse_(why, "not an ESRGAN graph"), nullptr;
    std::unique_ptr<Esrgan> e(new Esrgan(dev));
    Conv c0;
    if (!rd.conv("/model/model.0/Conv", &c0, why)) return nullptr;
    if (c0.cin != 3 && c0.cin != 12) return refuse_(why, "ESRGAN input channels"), nullptr;
    e->unshuffle_ = c0.cin == 12;
    e->nf = c0.cout;
    const auto c5name = [&](const std::string &pre) {
        return rd.node(pre + "conv5/conv5.0/Conv") ? pre + "conv5/conv5.0/Conv" : pre + "Mul";
    };
    int nb = 0;
    while (rd.node(c5name("/model/model.1/sub/sub." + std::to_string(nb) + "/RDB1/"))) nb++;
    if (nb < 1) return refuse_(why, "no RRDB blocks"), nullptr;
    e->nb = nb;
    Conv probe;
    if (!rd.conv("/model/model.1/sub/sub.0/RDB1/conv1/conv1.0/Conv", &probe, why)) return nullptr;
    e->gc = probe.cout;
    // the modules built (shaders.txt): 64 features, growth 32
    if (e->nf != 64 || e->gc != 32) return refuse_(why, "ESRGAN with " + std::to_string(e->nf) + " features, growth " + std::to_string(e->gc) + " (Vulkan has 64 / 32)"), nullptr;
    e->DS = e->nf + 4 * e->gc;
    {
        const ajinative::OnnxNode *m = rd.node("/model/model.1/sub/sub.0/Mul");
        std::vector<float> v;
        if (!m || m->in.size() != 2 || !rd.floats(m->in[1], &v) || v.size() != 1) return refuse_(why, "RRDB residual scale"), nullptr;
        e->rrdb_scale_ = v[0];
    }
    if (!rd.node("/model/model.10/Conv") || (!rd.node("/model/model.2/Resize_output_cast_0") && !rd.node("/model/model.2/Resize")))
        return refuse_(why, "not the two-upsample ESRGAN layout"), nullptr;
    e->scale_ = e->unshuffle_ ? 2 : 4;
    if (w < 1 || h < 1 || (e->unshuffle_ && (w % 2 || h % 2))) return refuse_(why, "this model needs even tile dimensions"), nullptr;
    e->w_ = w;
    e->h_ = h;
    const int bw = e->unshuffle_ ? w / 2 : w, bh = e->unshuffle_ ? h / 2 : h;
    const Family fam = dev.family();

    // weights, packed for the family, into one buffer (each conv's at an aligned offset)
    std::vector<uint16_t> wall, ball;
    const VkDeviceSize align = std::max<VkDeviceSize>(dev.info().storage_align, 16);
    // (output slices don't pay here: on a 360x240 trunk they made ESRGAN 2x a third slower)
    const auto add_conv = [&](const Conv &c, float scale, int cinp, const GConvShape &s) -> int {
        const int nz = 1;
        std::vector<float> sc(c.cout, scale);
        const std::vector<uint16_t> taps = conv_taps(c, scale == 1.f ? nullptr : &sc);
        const std::vector<uint16_t> pw = gconv_pack_slices(fam, taps.data(), c.cin, cinp, c.cout, s.coutp, nz);
        std::vector<float> b(c.cout);
        for (int o = 0; o < c.cout; o++) b[o] = c.b[o] * scale;
        const std::vector<uint16_t> pb = gconv_bias_slices(b.data(), c.cout, s.coutp, nz);
        Conv1 cv;
        cv.s = s;
        cv.cinp = cinp;
        cv.nz = nz;
        const size_t aw = align / 2;
        wall.resize((wall.size() + aw - 1) / aw * aw);
        cv.woff = wall.size() * 2;
        cv.wsize = pw.size() * 2;
        wall.insert(wall.end(), pw.begin(), pw.end());
        ball.resize((ball.size() + aw - 1) / aw * aw);
        cv.boff = ball.size() * 2;
        cv.bsize = pb.size() * 2;
        ball.insert(ball.end(), pb.begin(), pb.end());
        e->convs_.push_back(cv);
        return (int)e->convs_.size() - 1;
    };
    const auto shape = [](int coutp, int in, int act, int nres, int up2, int out) {
        GConvShape s;
        s.coutp = coutp;
        s.in = in;
        s.act = act;
        s.nres = nres;
        s.up2 = up2;
        s.out = out;
        return s;
    };
    const int i_first = add_conv(c0, 1.f, 16, shape(64, e->unshuffle_ ? 2 : 1, 0, 0, 0, 0));
    std::vector<int> i_dense;
    for (int i = 0; i < nb; i++)
        for (int r = 1; r <= 3; r++) {
            const std::string pre = "/model/model.1/sub/sub." + std::to_string(i) + "/RDB" + std::to_string(r) + "/";
            for (int k = 1; k <= 4; k++) {
                Conv c;
                if (!rd.conv(pre + "conv" + std::to_string(k) + "/conv" + std::to_string(k) + ".0/Conv", &c, why)) return nullptr;
                if (c.cin != e->nf + (k - 1) * e->gc || c.cout != e->gc || c.stride != 1 || !c.lrelu)
                    return refuse_(why, pre + ": dense conv shape"), nullptr;
                i_dense.push_back(add_conv(c, 1.f, c.cin, shape(32, 0, 2, 0, 0, 0)));
            }
            Conv c5;
            if (!rd.conv(c5name(pre), &c5, why)) return nullptr;
            if (c5.cin != e->DS || c5.cout != e->nf || c5.lrelu) return refuse_(why, pre + ": last conv shape"), nullptr;
            i_dense.push_back(add_conv(c5, r == 3 ? e->rrdb_scale_ : 1.f, e->DS, shape(64, 0, 0, r == 3 ? 2 : 1, 0, 0)));
        }
    Conv cb, cu1, cu2, chr, cl;
    if (!rd.conv("/model/model.1/sub/sub." + std::to_string(nb) + "/Conv", &cb, why) || !rd.conv("/model/model.3/Conv", &cu1, why) ||
        !rd.conv("/model/model.6/Conv", &cu2, why) || !rd.conv("/model/model.8/Conv", &chr, why) || !rd.conv("/model/model.10/Conv", &cl, why))
        return nullptr;
    if (cb.cin != e->nf || cb.cout != e->nf || cb.lrelu || !cu1.lrelu || !cu2.lrelu || !chr.lrelu || cl.lrelu || cl.cout != 3 ||
        cu1.cin != e->nf || cu2.cin != e->nf || chr.cin != e->nf || cl.cin != e->nf)
        return refuse_(why, "ESRGAN upsampler shape"), nullptr;
    const int i_body = add_conv(cb, 1.f, 64, shape(64, 0, 0, 1, 0, 0));
    const int i_up1 = add_conv(cu1, 1.f, 64, shape(64, 0, 2, 0, 1, 0));
    const int i_up2 = add_conv(cu2, 1.f, 64, shape(64, 0, 2, 0, 1, 0));
    const int i_hr = add_conv(chr, 1.f, 64, shape(64, 0, 2, 0, 0, 0));
    const int i_last = add_conv(cl, 1.f, 64, shape(16, 0, 0, 0, 0, 3));

    // buffers: the tile in and out (owned), the intermediates in the arena
    const VkDeviceSize px = (VkDeviceSize)bw * bh;
    if (!dev.make_buffer(&e->in_, (VkDeviceSize)3 * w * h * 2, false, why) ||
        !dev.make_buffer(&e->out_, (VkDeviceSize)3 * e->out_w() * e->out_h() * 2, false, why) ||
        !dev.make_buffer(&e->wts_, wall.size() * 2, false, why) || !dev.make_buffer(&e->bias_, ball.size() * 2, false, why) ||
        !dev.upload(e->wts_, wall.data(), wall.size() * 2, why) || !dev.upload(e->bias_, ball.data(), ball.size() * 2, why))
        return nullptr;
    e->wbytes_ = e->wts_.size + e->bias_.size;
    // lifetimes (steps): 1 F0 and D (trunk), 2 T, 3 U1, 4 P, 5 Q
    VkDeviceSize oF0 = 0, oD[3] = {}, oT = 0, oU1 = 0, oP = 0, oQ = 0;
    Arena ar;
    ar.add(&oF0, px * e->nf * 2, 1, 2, align);
    for (auto &o : oD) ar.add(&o, px * e->DS * 2, 1, 2, align);
    ar.add(&oT, px * e->nf * 2, 2, 3, align);
    ar.add(&oU1, px * 4 * e->nf * 2, 3, 4, align);
    ar.add(&oP, px * 16 * e->nf * 2, 4, 5, align);
    ar.add(&oQ, px * 16 * e->nf * 2, 5, 6, align);
    const VkDeviceSize total = ar.layout();
    const uint64_t biggest = std::max<uint64_t>(px * 16 * e->nf * 2, (uint64_t)3 * e->out_w() * e->out_h() * 2);
    if (dev.info().max_storage_range && biggest > dev.info().max_storage_range)
        return refuse_(why, "a " + std::to_string(w) + "x" + std::to_string(h) + " tile needs " + std::to_string(biggest >> 20) +
                                " MB buffers, over this device's " + std::to_string(dev.info().max_storage_range >> 20) +
                                " MB storage-buffer limit: use a smaller tile"),
               nullptr;
    if (!dev.make_buffer(&e->arena_, total, false, why)) return nullptr;
    e->bytes_ = e->in_.size + e->out_.size + e->wbytes_ + e->arena_.size;
    const VkBuffer A = e->arena_.buf;
    const Ref F0{A, oF0, px * e->nf * 2}, D[3] = {{A, oD[0], px * e->DS * 2}, {A, oD[1], px * e->DS * 2}, {A, oD[2], px * e->DS * 2}},
              T{A, oT, px * e->nf * 2}, U1{A, oU1, px * 4 * e->nf * 2}, P{A, oP, px * 16 * e->nf * 2}, Q{A, oQ, px * 16 * e->nf * 2};
    const Ref TILE_IN{e->in_.buf, 0, e->in_.size}, TILE_OUT{e->out_.buf, 0, e->out_.size};

    // the ops, in order
    const auto op = [&](int conv, Ref src, int soff, int sstride, Ref dst, int doff, int dstride, int H, int W) {
        Op o;
        o.conv = conv;
        o.s = e->convs_[conv].s;
        o.nz = e->convs_[conv].nz;
        o.entry = (int)e->ops_.size();
        o.src = src;
        o.dst = dst;
        o.p.H = H;
        o.p.W = W;
        o.p.src_off = soff;
        o.p.src_stride = sstride;
        o.p.dst_off = doff;
        o.p.dst_stride = dstride;
        o.p.cinp = e->convs_[conv].cinp;
        o.p.IH = H;  // stride 1: the input is the conv's size (the upsampled one, with UP2)
        o.p.IW = W;
        e->ops_.push_back(o);
        return &e->ops_.back();
    };
    op(i_first, TILE_IN, 0, 0, F0, 0, e->nf, bh, bw);
    op(i_first, TILE_IN, 0, 0, D[0], 0, e->DS, bh, bw);
    int di = 0;
    for (int i = 0; i < nb; i++)
        for (int r = 0; r < 3; r++) {
            const Ref src = D[r], nxt = D[(r + 1) % 3];
            for (int k = 0; k < 4; k++) op(i_dense[di++], src, 0, e->DS, src, e->nf + k * e->gc, e->DS, bh, bw);
            Op *o = op(i_dense[di++], src, 0, e->DS, nxt, 0, e->DS, bh, bw);
            o->r1 = src;  // the RDB input: the first nf channels of its dense buffer
            o->p.r1_stride = e->DS;
            o->p.c1 = 1.f;
            if (r == 2) {
                // out = s * (conv + x3) + x: weights x s (at upload), x3 x s, x (in place, D[0])
                o->p.c1 = e->rrdb_scale_;
                o->r2 = D[0];
                o->p.r2_stride = e->DS;
                o->p.c2 = 1.f;
            }
        }
    {
        Op *o = op(i_body, D[0], 0, e->DS, T, 0, e->nf, bh, bw);
        o->r1 = F0;
        o->p.r1_stride = e->nf;
    }
    op(i_up1, T, 0, e->nf, U1, 0, e->nf, 2 * bh, 2 * bw);
    op(i_up2, U1, 0, e->nf, P, 0, e->nf, 4 * bh, 4 * bw);
    op(i_hr, P, 0, e->nf, Q, 0, e->nf, 4 * bh, 4 * bw);
    op(i_last, Q, 0, e->nf, TILE_OUT, 0, 0, 4 * bh, 4 * bw);

    // each op's modules (and its halo chunk), then parameters, descriptor sets, pipelines
    for (Op &o : e->ops_) {
        std::string w2;
        if (!gconv_prepare(dev, &o, &w2)) return refuse_(why, "kernel: " + w2), nullptr;
    }
    std::vector<GConvParams> prm;
    for (const Op &o : e->ops_) prm.push_back(o.p);
    if (!dev.make_buffer(&e->params_, prm.size() * sizeof(GConvParams), false, why) ||
        !dev.upload(e->params_, prm.data(), prm.size() * sizeof(GConvParams), why))
        return nullptr;
    e->bytes_ += e->params_.size;
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(e->ops_.size() * Device::kBindings)};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = (uint32_t)e->ops_.size();
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(dev.device(), &pi, nullptr, &e->pool_) != VK_SUCCESS) return refuse_(why, "descriptor pool"), nullptr;
    }
    for (Op &o : e->ops_) {
        const Conv1 &cv = e->convs_[o.conv];
        VkDescriptorSetLayout l = dev.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = e->pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        if (vkAllocateDescriptorSets(dev.device(), &ai, &o.set) != VK_SUCCESS) return refuse_(why, "descriptor set"), nullptr;
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev.dummy(), 0, VK_WHOLE_SIZE};
        bi[0] = {o.src.buf, o.src.off, o.src.size};
        if (o.r1.buf) bi[1] = {o.r1.buf, o.r1.off, o.r1.size};
        bi[2] = {o.dst.buf, o.dst.off, o.dst.size};
        bi[3] = {e->wts_.buf, cv.woff, cv.wsize};
        bi[4] = {e->bias_.buf, cv.boff, cv.bsize};
        if (o.r2.buf) bi[5] = {o.r2.buf, o.r2.off, o.r2.size};
        bi[7] = {e->params_.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = o.set;
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev.device(), Device::kBindings, wr, 0, nullptr);
    }

    // tile variants: fixed (AJI_VK_VARIANT), stored, or timed once per distinct (module, size)
    std::vector<GConvOp *> gops;
    for (Op &o : e->ops_) gops.push_back(&o);
    if (!gconv_choose(dev, gops, why)) return nullptr;
    char buf[160];
    std::snprintf(buf, sizeof buf, "ESRGAN %dx, %d RRDB, %d features, growth %d; Vulkan %s", e->scale_, nb, e->nf, e->gc, family_name(fam));
    e->what_ = buf;
    return std::unique_ptr<ImgNet>(e.release());
}

// The conv nets the video path lowers, on the executor.
class Lowered : public ImgNet {
public:
    explicit Lowered(std::unique_ptr<Executor> e) : ex_(std::move(e)) {}
    const Buffer &input() const override { return ex_->input(); }
    const Buffer &output() const override { return ex_->output(); }
    void bind_io(VkBuffer in, VkBuffer out) override { ex_->bind_io(in, out); }
    void suffix(const std::string &s) override { desc_ += s; }
    bool record(VkCommandBuffer cb, int w, int h) const override { return ex_->record(cb, w, h); }
    int out_w() const override { return ex_->out_w(); }
    int out_h() const override { return ex_->out_h(); }
    size_t device_bytes() const override { return ex_->device_bytes(); }
    size_t weight_bytes() const override { return ex_->weight_bytes(); }
    std::string describe() const override { return desc_; }
    std::string desc_;

private:
    std::unique_ptr<Executor> ex_;
};

}  // namespace

bool ImgNet::run(Device &dev, int w, int h, std::string *why) const
{
    VkCommandBuffer cb = dev.begin();
    const bool ok = record(cb, w, h);
    const bool sub = dev.submit_wait(cb, why);
    if (!ok && why) *why = "the network refused the tile size";
    return ok && sub;
}

std::unique_ptr<ImgNet> build_esrgan(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    return Esrgan::build(dev, g, w, h, why);
}

std::unique_ptr<ImgNet> ImgNet::lowered(Device &dev, const ajinative::Plan &plan, int w, int h, std::string *why)
{
    std::unique_ptr<Executor> ex = Executor::create(dev, plan, w, h, why);
    if (!ex) return nullptr;
    auto n = std::make_unique<Lowered>(std::move(ex));
    n->desc_ = ajinative::describe(plan) + "; Vulkan " + family_name(dev.family());
    return n;
}

std::unique_ptr<ImgNet> ImgNet::frame(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    if (w < 1 || h < 1) return refuse_(why, "bad frame size"), nullptr;
    // the probe: buffers per input pixel (they scale with the area; what does not makes the
    // estimate high, not low)
    constexpr int P = 128;
    // the image models: ESRGAN, then FDAT
    using Builder = std::unique_ptr<ImgNet> (*)(Device &, const ajinative::OnnxGraph &, int, int, std::string *);
    Builder bld = &build_esrgan;
    std::string we;
    std::unique_ptr<ImgNet> probe = bld(dev, g, P, P, &we);
    if (!probe && we.rfind("not an ESRGAN", 0) == 0) {
        bld = &build_fdat;
        probe = bld(dev, g, P, P, &we);
        if (!probe && we.rfind("not an FDAT", 0) == 0) we = "not an ESRGAN graph";
    }
    if (!probe) {
        if (why) *why = we;
        return nullptr;
    }
    const double per_px = (double)(probe->device_bytes() - probe->weight_bytes()) / ((double)P * P);
    const double need = (double)probe->weight_bytes() + per_px * w * h;
    probe.reset();
    uint64_t free_b = 0, total_b = 0;
    dev.memory(&free_b, &total_b);
    const double keep = (double)(1ull << 30) + total_b / 10.0;
    const auto gb = [](double b) {
        char t[32];
        std::snprintf(t, sizeof t, "%.1f GB", b / (1u << 30));
        return std::string(t);
    };
    if (need + keep > (double)free_b)
        return refuse_(why, "a whole " + std::to_string(w) + "x" + std::to_string(h) + " frame needs ~" + gb(need) + ", " +
                                gb((double)free_b) + " free (keeping " + gb(keep) + ")"),
               nullptr;
    std::unique_ptr<ImgNet> n = bld(dev, g, w, h, why);
    if (!n) return nullptr;
    n->suffix(", whole frame");
    return n;
}

std::unique_ptr<ImgNet> ImgNet::create(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    std::string we;
    if (auto n = Esrgan::build(dev, g, w, h, &we)) return n;
    if (we.rfind("not an ESRGAN", 0) != 0) {
        if (why) *why = we;
        return nullptr;
    }
    if (auto n = build_fdat(dev, g, w, h, &we)) return n;
    if (we.rfind("not an FDAT", 0) != 0) {
        if (why) *why = we;
        return nullptr;
    }
    ajinative::Plan plan;
    std::string wl;
    if (!ajinative::lower(g, &plan, &wl)) {
        if (why) *why = wl;
        return nullptr;
    }
    std::unique_ptr<Executor> ex = Executor::create(dev, plan, w, h, &wl);
    if (!ex) {
        if (why) *why = wl;
        return nullptr;
    }
    auto n = std::make_unique<Lowered>(std::move(ex));
    n->desc_ = ajinative::describe(plan) + "; Vulkan " + family_name(dev.family());
    return n;
}

}  // namespace ajivk
