// vk/tfdat.cpp — TFDAT v2 (traiNNer-redux dis2-tfdat2; bndl animefilm is 2x, 9 frames) on Vulkan for
// one padded frame size, as native/tfdat.cpp runs it: per frame a 3-level encoder; per neighbour a
// coarse-to-fine flow, the full-resolution warp and occlusion, conv_first on the unshuffled warped
// frame, align_refine and a confidence-weighted slot into the fusion 1x1; the fusion's 3x3 and SE
// gate feed the FDAT body (vk/fdat_body.h); then conv_after, the pixel-shuffle upsampler, the
// bilinearly upscaled centre frame and hr_refine. The alignment kernels are tfm.comp passes 18-25.
// As on CUDA, the alignment, the body's scratch and the tail share one arena region.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "vk/fdat_body.h"
#include "vk/tfix.h"
#include "vk/tnet.h"

namespace ajivk {

using namespace ajinative::onnxread;
using ajinative::OnnxAttr;
using ajinative::OnnxNode;

namespace {

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

// conv weights with the expected LeakyReLU (0 = none)
bool cv(const Reader &rd, const std::string &name, Conv *c, float slope, int cin, int cout, std::string *why)
{
    if (!rd.conv(name, c, why)) return false;
    if ((slope > 0) != c->lrelu || (slope > 0 && std::fabs(c->slope - slope) > 1e-6f) || (cin && c->cin != cin) || (cout && c->cout != cout))
        return refuse_(why, name + ": unexpected shape or activation");
    return true;
}

// A 1x1 conv followed by LeakyRelu(0.2) as the next node's consumer.
bool lrelu_after(const Reader &rd, const std::string &conv_name, const std::string &lrelu_name)
{
    const OnnxNode *c = rd.node(conv_name), *l = rd.node(lrelu_name);
    if (!c || !l || l->op != "LeakyRelu" || l->in.empty() || l->in[0] != c->out[0]) return false;
    const OnnxAttr *a = l->attr("alpha");
    return a && std::fabs(a->f - 0.2f) < 1e-6f;
}

// A conv's weights, kept for the build (their address keys TNet's weight store: a conv used for
// every frame is uploaded once).
struct Cw {
    Conv c;
    std::vector<uint16_t> taps;
    std::vector<float> b;
    int cout = 0;  // as packed (ps_weights: 128)
};

class VkTfdat : public TemporalNet {
public:
    explicit VkTfdat(Device &d) : net_(d) {}
    static std::unique_ptr<TemporalNet> build(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why);
    int frames() const override { return nf_; }
    int scale() const override { return sc_; }
    int cache_entries() const override { return 0; }
    const Buffer &input() const override { return net_.input(); }
    const Buffer &output() const override { return net_.output(); }
    void bind_io(VkBuffer in, VkBuffer out) override { net_.bind_io(in, out); }
    void record(VkCommandBuffer cb) const override { net_.record(cb); }
    void record_encode(VkCommandBuffer, int, int) const override {}
    void record_cached(VkCommandBuffer, const int *, const int *, float, float) const override {}
    void bind_diff(VkBuffer) override {}
    std::string profile(Device &dev) const override { return net_.profile(dev); }
    std::string describe() const override { return what_; }
    size_t device_bytes() const override { return net_.device_bytes(); }

private:
    TNet net_;
    int nf_ = 0, sc_ = 2;
    std::string what_;
};

std::unique_ptr<TemporalNet> VkTfdat::build(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    const std::string P = "/model/";
    if (!rd.node(P + "flow_net/dec3/dec3.0/Conv") || !rd.node(P + "temporal_fusion/fuse/fuse.0/Conv") ||
        !rd.node(P + "groups/groups.0/blocks.0/n1/LayerNormalization"))
        return refuse_(why, "not a TFDAT graph"), nullptr;
    if (W % 32 || H % 32 || W < 64 || H < 64) return refuse_(why, "TFDAT needs a frame size that is a multiple of 32"), nullptr;
    std::unique_ptr<VkTfdat> f(new VkTfdat(dev));
    int nf = rd.node(P + "enc1/enc1.0/Conv") ? 1 : 0;
    while (rd.node(P + "enc1/enc1.0_" + std::to_string(nf) + "/Conv")) nf++;
    if (nf < 3 || nf % 2 == 0) return refuse_(why, "TFDAT frame count"), nullptr;
    f->nf_ = nf;
    std::vector<std::unique_ptr<Cw>> keep;
    const auto load = [&](const std::string &name, float slope, int cin, int cout) -> Cw * {
        std::unique_ptr<Cw> k(new Cw);
        if (!cv(rd, name, &k->c, slope, cin, cout, why)) return nullptr;
        k->taps = conv_taps(k->c, nullptr);
        k->b = k->c.b;
        k->cout = k->c.cout;
        keep.push_back(std::move(k));
        return keep.back().get();
    };
    Cw *enc[6], *dec[3][3];
    const char *encs[6] = {"enc1/enc1.0", "enc1/enc1.2", "enc2/enc2.0", "enc2/enc2.2", "enc3/enc3.0", "enc3/enc3.2"};
    const int ecin[6] = {3, 32, 32, 64, 64, 96}, ecout[6] = {32, 32, 64, 64, 96, 96};
    for (int i = 0; i < 6; i++) {
        if (!(enc[i] = load(P + encs[i] + "/Conv", 0.2f, ecin[i], ecout[i]))) return nullptr;
        if (enc[i]->c.stride != (i % 2 ? 1 : 2)) return refuse_(why, std::string(encs[i]) + ": stride"), nullptr;
    }
    static constexpr int CS1 = 80, CS2 = 144, CS3 = 192, CAT = 256;
    struct Dec {
        const char *name;
        int cin, ch;
    } decs[3] = {{"flow_net/dec3/dec3", 192, 96}, {"flow_net/dec2/dec2", 130, 64}, {"flow_net/dec1/dec1", 66, 32}};
    for (int k = 0; k < 3; k++) {
        const std::string n = P + decs[k].name;
        if (!(dec[k][0] = load(n + ".0/Conv", 0.2f, decs[k].cin, decs[k].ch)) || !(dec[k][1] = load(n + ".2/Conv", 0.2f, decs[k].ch, decs[k].ch)) ||
            !(dec[k][2] = load(n + ".4/Conv", 0, decs[k].ch, 2)))
            return nullptr;
    }
    Cw *cf0 = load(P + "conv_first/conv_first.0/Conv", 0.2f, 0, 0);
    if (!cf0) return nullptr;
    int U = 0;
    if (cf0->c.cin == 3) U = 1;
    else if (cf0->c.cin == 12) U = 2;
    else if (cf0->c.cin == 48) U = 4;
    else return refuse_(why, "TFDAT conv_first input channels"), nullptr;
    f->sc_ = 4 / U;
    const int UC = pad16(cf0->c.cin);
    const int C = cf0->c.cout, CP = pad16(C);
    if (CP > 128) return refuse_(why, "TFDAT channels"), nullptr;
    Cw *cf2 = load(P + "conv_first/conv_first.2/Conv", 0, C, C);
    if (!cf2) return nullptr;
    // align_refine: 1x1 over [nb, ctr] (rows 0.. and 128..), LeakyReLU, 3x3, plus nb
    std::vector<float> wm, bm, emc, emn;
    std::vector<int> map(2 * C);
    for (int i = 0; i < 2 * C; i++) map[i] = i < C ? i : 128 + (i - C);
    int co = 0;
    if (!conv1x1_matrix(rd, P + "align_refine/body/body.0/Conv", CAT, &map, &wm, &bm, &co, why) || co != C ||
        !lrelu_after(rd, P + "align_refine/body/body.0/Conv", P + "align_refine/body/body.1/LeakyRelu"))
        return refuse_(why, "TFDAT align_refine"), nullptr;
    Cw *ar2 = load(P + "align_refine/body/body.2/Conv", 0, C, C);
    if (!ar2) return nullptr;
    // temporal fusion
    std::vector<float> nb_;
    if (!conv1x1_matrix(rd, P + "temporal_fusion/emb_center/Conv", C, nullptr, &emc, &nb_, &co, why) || co != C || !nb_.empty())
        return refuse_(why, "TFDAT emb_center"), nullptr;
    if (!conv1x1_matrix(rd, P + "temporal_fusion/emb_neighbor/Conv", C, nullptr, &emn, &nb_, &co, why) || co != C || !nb_.empty())
        return refuse_(why, "TFDAT emb_neighbor"), nullptr;
    std::vector<float> sc;
    if (!init_input(rd, rd.node(P + "temporal_fusion/Mul_1"), &sc) || sc.size() != 1) return refuse_(why, "TFDAT similarity scale"), nullptr;
    const float sim_scale = sc[0];
    Cw *conf0 = load(P + "temporal_fusion/conf/conf.0/Conv", 0.2f, 2, 16), *conf2 = load(P + "temporal_fusion/conf/conf.2/Conv", 0, 16, 1);
    if (!conf0 || !conf2) return nullptr;
    std::vector<std::vector<float>> fuse(nf);
    std::vector<float> fuseb;
    {
        const OnnxNode *n = rd.node(P + "temporal_fusion/fuse/fuse.0/Conv");
        std::vector<float> v;
        std::vector<int64_t> d;
        if (!n || !tensor_floats(rd, n, 1, &v, &d) || d.size() != 4 || d[0] != C || d[1] != (int64_t)nf * C || d[2] != 1 || n->in.size() < 3 ||
            !rd.floats(n->in[2], &fuseb) || (int)fuseb.size() != C ||
            !lrelu_after(rd, P + "temporal_fusion/fuse/fuse.0/Conv", P + "temporal_fusion/fuse/fuse.1/LeakyRelu"))
            return refuse_(why, "TFDAT fuse"), nullptr;
        for (int j = 0; j < nf; j++) {
            fuse[j].assign((size_t)C * C, 0.f);
            for (int o = 0; o < C; o++)
                for (int k = 0; k < C; k++) fuse[j][(size_t)k * C + o] = v[(size_t)o * nf * C + j * C + k];
        }
    }
    Cw *fuse2 = load(P + "temporal_fusion/fuse/fuse.2/Conv", 0.2f, C, C);
    if (!fuse2) return nullptr;
    std::vector<float> se1, se2;
    int ser = 0;
    {
        std::vector<int64_t> d;
        if (!tensor_floats(rd, rd.node(P + "temporal_fusion/se/se.1/Conv"), 1, &se1, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 ||
            d[3] != 1)
            return refuse_(why, "TFDAT se"), nullptr;
        ser = (int)d[0];
        if (!tensor_floats(rd, rd.node(P + "temporal_fusion/se/se.3/Conv"), 1, &se2, &d) || d != std::vector<int64_t>{C, ser, 1, 1})
            return refuse_(why, "TFDAT se"), nullptr;
        const OnnxNode *s1 = rd.node(P + "temporal_fusion/se/se.1/Conv"), *s3 = rd.node(P + "temporal_fusion/se/se.3/Conv");
        if ((s1->in.size() > 2 && !s1->in[2].empty()) || (s3->in.size() > 2 && !s3->in[2].empty())) return refuse_(why, "TFDAT se bias"), nullptr;
    }
    FdatBody B;
    if (!fdat_parse_body(rd, P, C, &B, why)) return nullptr;
    // tail
    Cw *after = load(P + "conv_after/Conv", 0, C, C), *up0 = load(P + "upsampler/upsampler.0/Conv", 0.01f, C, 64);
    if (!after || !up0) return nullptr;
    Cw *ups[2][2];
    for (int k = 0; k < 2; k++) {
        Conv c;
        if (!cv(rd, P + (k ? "upsampler/upsampler.4/Conv" : "upsampler/upsampler.2/Conv"), &c, 0, 64, 256, why)) return nullptr;
        for (int sy = 0; sy < 2; sy++) {
            std::unique_ptr<Cw> w(new Cw);
            w->c = c;
            ps_weights(c, sy, 64, &w->taps, &w->b);
            w->cout = 128;
            keep.push_back(std::move(w));
            ups[k][sy] = keep.back().get();
        }
    }
    if (!rd.node(P + "upsampler/upsampler.3/DepthToSpace") || !rd.node(P + "upsampler/upsampler.5/DepthToSpace"))
        return refuse_(why, "TFDAT upsampler"), nullptr;
    Cw *up6 = load(P + "upsampler/upsampler.6/Conv", 0, 64, 3), *hr0 = load(P + "hr_refine/hr_refine.0/Conv", 0.2f, 3, 32),
       *hr2 = load(P + "hr_refine/hr_refine.2/Conv", 0, 32, 3);
    if (!up6 || !hr0 || !hr2) return nullptr;

    // ---- the arena
    TNet &n = f->net_;
    const int w1 = W / 2, h1 = H / 2, w2 = W / 4, h2 = H / 4, w3 = W / 8, h3 = H / 8;
    const VkDeviceSize M1 = (VkDeviceSize)w1 * h1, M2 = (VkDeviceSize)w2 * h2, M3 = (VkDeviceSize)w3 * h3;
    const int wf = W / U, hf = H / U, OW = 4 * wf, OH = 4 * hf;
    const VkDeviceSize MF = (VkDeviceSize)wf * hf, MO = 16 * MF, plane = (VkDeviceSize)W * H;
    // persistent: the flows (fp32 pairs), the body's input and output, the SE gate
    const VkDeviceSize F3 = n.place(M3 * 8), U2 = n.place(M2 * 8), F2 = n.place(M2 * 8), U1 = n.place(M1 * 8), F1 = n.place(M1 * 8);
    const VkDeviceSize X0 = n.place(MF * CP * 2), G = n.place(MF * CP * 2), SUMS = n.place(CP * 4), CM = n.place(CP * 4);
    // the shared region: alignment + fusion, the body's scratch, the tail
    TCarve a1, a3;
    a1.align = a3.align = n.align();
    const auto B2 = [](TCarve &c, VkDeviceSize m, int ch) { return c.take(m * ch * 2); };
    const VkDeviceSize E1a = B2(a1, M1, 32), E1N = B2(a1, M1, 32), C1 = B2(a1, M1, CS1), E2a = B2(a1, M2, 64), E2N = B2(a1, M2, 64),
                       C2 = B2(a1, M2, CS2), E3a = B2(a1, M3, 96), C3 = B2(a1, M3, CS3), D3a = B2(a1, M3, 96), D3b = B2(a1, M3, 96),
                       D3 = B2(a1, M3, 16), D2a = B2(a1, M2, 64), D2b = B2(a1, M2, 64), D2 = B2(a1, M2, 16), D1a = B2(a1, M1, 32),
                       D1b = B2(a1, M1, 32), D1 = B2(a1, M1, 16);
    const VkDeviceSize US = B2(a1, MF, UC), WUS = B2(a1, MF, UC), CONF = B2(a1, MF, 16), CF1 = B2(a1, MF, 16), CF2 = B2(a1, MF, 16),
                       T1 = B2(a1, MF, CP), CATB = B2(a1, MF, CAT), AR1 = B2(a1, MF, CP), AL = B2(a1, MF, CP), EC = B2(a1, MF, CP),
                       EN = B2(a1, MF, CP), SLOT = B2(a1, MF, CP), ACC = B2(a1, MF, CP), FU = B2(a1, MF, CP);
    const VkDeviceSize TT = B2(a3, MF, CP), U0 = B2(a3, MF, 64), V1 = B2(a3, 4 * MF, 64), V2 = B2(a3, MO, 64), RZ = B2(a3, MO, 16),
                       S = B2(a3, MO, 16);
    const VkDeviceSize HR = V2, O16 = RZ;  // dead once up6 has run
    const VkDeviceSize R = n.place(std::max({a1.off, a3.off, fdat_body_scratch(n, B, hf, wf)}));

    const auto hm = [](VkDeviceSize o, int s, int off = 0) { return hmap(o, s, off); };
    const auto fm = [](VkDeviceSize o, int s) { return fmap(o, s); };
    const VkDeviceSize all = ~0ull;
    // a conv: src (arena offset, or the input planes at in_off with in = 1), dst at dst_off halfs
    struct Cio {
        int in = 0, stride = 1, act = 0;
        float slope = 0.2f;
        VkDeviceSize src = 0;
        int src_off = 0, sstride = 0, cinp = 16, IH = 0, IW = 0;
        VkDeviceSize dst = 0;
        int dst_off = 0, dstride = 0, H = 0, Wd = 0;
        VkDeviceSize r1 = ~0ull;
        int r1stride = 0, out = 0, sy = 0;
    };
    const auto conv = [&](const Cw *c, const Cio &x, std::initializer_list<int> widths) {
        TNet::ConvIO io;
        io.in = x.in;
        io.stride = x.stride;
        io.act = x.act;
        io.slope = x.slope;
        io.out = x.out;
        io.sy = x.sy;
        if (x.in == 1) {
            io.src_in = true;
        } else {
            io.src = R + x.src;
            io.src_bytes = all;
        }
        io.src_off = x.src_off;
        io.src_stride = x.sstride;
        io.cinp = x.cinp;
        io.IH = x.IH;
        io.IW = x.IW;
        io.dst = R + x.dst;
        io.dst_bytes = all;
        io.dst_off = x.dst_off;
        io.dst_stride = x.dstride;
        io.H = x.H;
        io.W = x.Wd;
        if (x.r1 != ~0ull) {
            io.nres = 1;
            io.r1 = x.r1;
            io.r1_bytes = all;
            io.r1_stride = x.r1stride;
        }
        n.conv(c->taps, c->b, c->c.cin, c->cout, io, widths, &c->taps);
    };
    // the arena offsets above are relative to R, except the persistent ones: an absolute offset for those
    const auto abs_ = [&](VkDeviceSize o) { return o - R; };
    const auto lr = [](Cio x, float s = 0.2f) {
        x.act = 2;
        x.slope = s;
        return x;
    };
    const auto io = [](VkDeviceSize src, int soff, int ss, int cinp, int IH, int IW, VkDeviceSize dst, int doff, int ds, int Hh, int Ww) {
        Cio x;
        x.src = src;
        x.src_off = soff;
        x.sstride = ss;
        x.cinp = cinp;
        x.IH = IH;
        x.IW = IW;
        x.dst = dst;
        x.dst_off = doff;
        x.dstride = ds;
        x.H = Hh;
        x.Wd = Ww;
        return x;
    };
    const std::initializer_list<int> WD = {64, 48, 32, 16};
    const int ctr = nf / 2;
    // the 3-level encoder of frame fr into e1 (+off, stride s1), e2, e3
    const auto encode = [&](int fr, VkDeviceSize e1, int o1, int s1, VkDeviceSize e2, int o2, int s2, VkDeviceSize e3, int o3, int s3) {
        Cio a = lr(io(0, (int)(fr * 3 * plane), 0, 16, H, W, E1a, 0, 32, h1, w1));
        a.in = 1;
        a.stride = 2;
        conv(enc[0], a, {32, 16});
        conv(enc[1], lr(io(E1a, 0, 32, 32, h1, w1, e1, o1, s1, h1, w1)), WD);
        Cio b = lr(io(e1, o1, s1, 32, h1, w1, E2a, 0, 64, h2, w2));
        b.stride = 2;
        conv(enc[2], b, WD);
        conv(enc[3], lr(io(E2a, 0, 64, 64, h2, w2, e2, o2, s2, h2, w2)), WD);
        Cio c = lr(io(e2, o2, s2, 64, h2, w2, E3a, 0, 96, h3, w3));
        c.stride = 2;
        conv(enc[4], c, WD);
        conv(enc[5], lr(io(E3a, 0, 96, 96, h3, w3, e3, o3, s3, h3, w3)), WD);
    };
    const auto decode = [&](Cw *const *ops, VkDeviceSize cin, int cs, VkDeviceSize a, VkDeviceSize b, VkDeviceSize d, int ch, int hh, int ww) {
        conv(ops[0], lr(io(cin, 0, cs, cs, hh, ww, a, 0, ch, hh, ww)), WD);
        conv(ops[1], lr(io(a, 0, ch, ch, hh, ww, b, 0, ch, hh, ww)), WD);
        conv(ops[2], io(b, 0, ch, ch, hh, ww, d, 0, 16, hh, ww), {16});
    };
    // conv_first of an unshuffled frame (stride UC at hf x wf) into CATB at doff
    const auto first = [&](VkDeviceSize us, int doff) {
        conv(cf0, lr(io(us, 0, UC, UC, hf, wf, T1, 0, CP, hf, wf)), WD);
        conv(cf2, io(T1, 0, CP, CP, hf, wf, CATB, doff, CAT, hf, wf), WD);
    };
    const int M = (int)MF;
    const auto pt = [&](int p, const TArgs &a, long long threads, const char *lb) { n.pass_threads(p, a, threads, lb); };
    const auto flow_add = [&](VkDeviceSize Uf, VkDeviceSize D, VkDeviceSize Fo, VkDeviceSize m) {
        TArgs t;
        t.map(0, hm(R + D, 16));
        if (Uf != ~0ull) t.map(1, fm(Uf, 2));
        t.map(2, fm(Fo, 2));
        t.v[16] = (int)m;
        pt(18, t, (long long)m, "tfflow");
    };
    const auto up_warp = [&](VkDeviceSize Fi, int hh, int ww, VkDeviceSize feat, int fs, int Cc, VkDeviceSize dst, int ds, int uoff, VkDeviceSize Uo) {
        TArgs t;
        t.map(0, fm(Fi, 2));
        t.map(1, hm(R + feat, fs));
        t.map(2, hm(R + dst, ds));
        t.map(3, fm(Uo, 2));
        t.v[16] = hh;
        t.v[17] = ww;
        t.v[18] = Cc;
        t.v[19] = uoff;
        pt(19, t, 4ll * hh * ww * (Cc / 8), "tfupwarp");
    };
    const auto gemm = [&](const TNet::Mat &m, const TMap &a, VkDeviceSize o, int act, const TMap &add, bool last) {
        n.gemm(a, M, m, hm(R + o, CP), act, last, -1, add);
    };
    const TNet::Mat ar0 = n.add_mat(wm, bm, CAT, C), embc = n.add_mat(emc, {}, C, C), embn = n.add_mat(emn, {}, C, C);
    std::vector<TNet::Mat> fm_(nf);
    for (int j = 0; j < nf; j++) fm_[j] = n.add_mat(fuse[j], j + 1 == nf ? fuseb : std::vector<float>{}, C, C);

    // ---- centre frame: encoder into the concat buffers' centre slices, features, embedding, slot 0
    encode(ctr, C1, 32, CS1, C2, 64, CS2, C3, 96, CS3);
    {
        TArgs t;
        t.map(0, hm(R + US, UC));
        t.v[16] = H;
        t.v[17] = W;
        t.v[18] = U;
        t.v[19] = (int)(ctr * 3 * plane);
        pt(21, t, (long long)MF, "tfunshuf");
    }
    first(US, 128);
    gemm(embc, hm(R + CATB, CAT, 128), EC, 0, TMap{}, false);
    gemm(fm_[0], hm(R + CATB, CAT, 128), ACC, 0, TMap{}, false);
    int slot = 1;
    for (int fr = 0; fr < nf; fr++) {
        if (fr == ctr) continue;
        encode(fr, E1N, 0, 32, E2N, 0, 64, C3, 0, CS3);
        // flow, coarse to fine
        decode(dec[0], C3, CS3, D3a, D3b, D3, 96, h3, w3);
        flow_add(~0ull, D3, F3, M3);
        up_warp(F3, h3, w3, E2N, 64, 64, C2, CS2, 128, U2);
        decode(dec[1], C2, CS2, D2a, D2b, D2, 64, h2, w2);
        flow_add(U2, D2, F2, M2);
        up_warp(F2, h2, w2, E1N, 32, 32, C1, CS1, 64, U1);
        decode(dec[2], C1, CS1, D1a, D1b, D1, 32, h1, w1);
        flow_add(U1, D1, F1, M1);
        {
            TArgs t;
            t.map(0, fm(F1, 2));
            t.map(1, hm(R + WUS, UC));
            t.map(2, hm(R + CONF, 16));
            t.v[16] = H;
            t.v[17] = W;
            t.v[18] = U;
            t.v[19] = (int)(fr * 3 * plane);
            t.v[20] = (int)(ctr * 3 * plane);
            pt(20, t, (long long)MF, "tfwarp");
        }
        // features, alignment refinement
        first(WUS, 0);
        gemm(ar0, hm(R + CATB, CAT), AR1, 2, TMap{}, false);
        {
            Cio x = io(AR1, 0, CP, CP, hf, wf, AL, 0, CP, hf, wf);
            x.r1 = R + CATB;
            x.r1stride = CAT;
            conv(ar2, x, WD);
        }
        // the confidence-weighted slot into the fusion 1x1
        gemm(embn, hm(R + AL, CP), EN, 0, TMap{}, false);
        {
            TArgs t;
            t.map(0, hm(R + EC, CP));
            t.map(1, hm(R + EN, CP));
            t.map(2, hm(R + CONF, 16));
            t.v[16] = M;
            t.v[17] = C;
            t.f(18, sim_scale);
            const long long g8 = (M + 7) / 8;
            const uint32_t gx = (uint32_t)std::min<long long>(g8, 4096);
            n.pass(22, t, gx, (uint32_t)((g8 + gx - 1) / gx), "tfsim");
        }
        conv(conf0, lr(io(CONF, 0, 16, 16, hf, wf, CF1, 0, 16, hf, wf)), {16});
        conv(conf2, io(CF1, 0, 16, 16, hf, wf, CF2, 0, 16, hf, wf), {16});
        {
            TArgs t;
            t.map(0, hm(R + AL, CP));
            t.map(1, hm(R + CF2, 16));
            t.map(2, hm(R + SLOT, CP));
            t.v[16] = M;
            t.v[17] = C;
            pt(23, t, (long long)M * (CP / 8), "tfgate");
        }
        const bool last = slot + 1 == nf;
        gemm(fm_[slot], hm(R + SLOT, CP), ACC, last ? 2 : 0, hm(R + ACC, CP), last);
        slot++;
    }
    conv(fuse2, lr(io(ACC, 0, CP, CP, hf, wf, FU, 0, CP, hf, wf)), WD);
    n.colsums(hm(R + FU, CP), M, C, SUMS);
    n.chan_gate(SUMS, M, C, ser, n.add_wf(se1), -1, n.add_wf(se2), -1, false, 1.f, CM, CP);
    {
        TArgs t;
        t.map(0, hm(R + FU, CP));
        t.map(1, hm(X0, CP));
        t.v[16] = M;
        t.v[17] = C;
        t.v[18] = (int32_t)(CM / 4);
        pt(24, t, (long long)M * (CP / 8), "tfcolscale");
    }
    // ---- the body
    n.copy(X0, G, MF * CP * 2);
    {
        TCarve bs;
        bs.off = R;
        bs.align = n.align();
        fdat_record_body(n, B, G, hf, wf, &bs);
    }
    // ---- the tail
    {
        Cio x = io(abs_(G), 0, CP, CP, hf, wf, TT, 0, CP, hf, wf);
        x.r1 = X0;
        x.r1stride = CP;
        conv(after, x, WD);
    }
    conv(up0, lr(io(TT, 0, CP, CP, hf, wf, U0, 0, 64, hf, wf), 0.01f), WD);
    for (int sy = 0; sy < 2; sy++) {
        Cio x = io(U0, 0, 64, 64, hf, wf, V1, 0, 64, hf, wf);
        x.out = 4;
        x.sy = sy;
        conv(ups[0][sy], x, WD);
    }
    for (int sy = 0; sy < 2; sy++) {
        Cio x = io(V1, 0, 64, 64, 2 * hf, 2 * wf, V2, 0, 64, 2 * hf, 2 * wf);
        x.out = 4;
        x.sy = sy;
        conv(ups[1][sy], x, WD);
    }
    {
        TArgs t;
        t.map(0, hm(R + RZ, 16));
        t.v[16] = H;
        t.v[17] = W;
        t.v[18] = f->sc_;
        t.v[19] = (int)(ctr * 3 * plane);
        pt(25, t, (long long)OW * OH, "tfresize");
    }
    {
        Cio x = io(V2, 0, 64, 64, OH, OW, S, 0, 16, OH, OW);
        x.r1 = R + RZ;
        x.r1stride = 16;
        conv(up6, x, {16});
    }
    conv(hr0, lr(io(S, 0, 16, 16, OH, OW, HR, 0, 32, OH, OW)), {32, 16});
    {
        Cio x = io(HR, 0, 32, 32, OH, OW, O16, 0, 16, OH, OW);
        x.r1 = R + S;
        x.r1stride = 16;
        conv(hr2, x, {16});
    }
    n.imgout(hm(R + O16, 16), OH, OW, OW);
    if (!n.finish((VkDeviceSize)3 * nf * plane * 2, (VkDeviceSize)3 * OW * OH * 2, why)) return nullptr;
    f->what_ = "TFDAT " + std::to_string(f->sc_) + "x, " + std::to_string(nf) + " frames, " + std::to_string(B.ngroups) + " groups x " +
               std::to_string(B.nblk) + " blocks, " + std::to_string(C) + " channels; Vulkan " + family_name(dev.family());
    return f;
}

}  // namespace

std::unique_ptr<TemporalNet> build_tfdat(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why)
{
    return VkTfdat::build(dev, g, W, H, why);
}

}  // namespace ajivk
