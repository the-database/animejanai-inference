// vk/hat.cpp — HAT (Hybrid Attention Transformer: IllustrationJaNai's HAT-L 4x, bf16 exports) on
// Vulkan for one tile size, as native/hat.cpp runs it: conv_first on the input minus the dataset
// mean, patch_embed's LayerNorm, the residual groups (HABs: window attention with a
// relative-position bias, shifted with Swin's mask in every second block, plus the CAB's convs and
// channel gate; an OCAB over overlapping key windows; each with its MLP; the group conv plus the
// group input), the final LayerNorm, conv_after_body plus conv_first's output and the pixel-shuffle
// upsampler as phase convs. The residual stream is fp32 (it outgrows fp16 in the last group); the
// group conv is an im2col GEMM for its fp32 sums and output. The input is reflect-padded to the
// window multiple and the output cropped, as native/framenet.cpp pads whole frames.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "native/hat_graph.h"
#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "vk/imgnet.h"
#include "vk/tnet.h"

namespace ajivk {

using namespace ajinative::onnxread;

namespace {

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

struct Ln {
    std::vector<float> g, b;
    float eps = 1e-5f;
};
struct Lin {
    std::vector<float> w, b;
    int K = 0, N = 0;
};
struct Attn {
    Lin qkv, proj;
    float scale = 1.f;
    std::vector<float> rtab;
    int rt_w = 0, rt_offy = 0, rt_offx = 0, tn = 0, shift = 0;
};
struct Mlp {
    Lin fc1, fc2;
};
struct Hab {
    Ln n1, n2;
    Attn at;
    Mlp mlp;
    Conv cab0, cab2;
    std::vector<float> ca1, cab1, ca2, cab2b;
    float conv_scale = 1.f;
};
struct Ocab {
    Ln n1, n2;
    Attn at;
    Mlp mlp;
};
struct Group {
    std::vector<Hab> habs;
    Ocab oc;
    Conv conv;
};

Ln to_ln(const ajinative::HatLn &h) { return Ln{h.g, h.b, h.eps}; }
Lin to_lin(const ajinative::HatLin &h) { return Lin{h.w, h.b, h.K, h.N}; }

bool to_attn(const ajinative::HatGraph &hg, const ajinative::HatAttn &h, bool overlap, Attn *a, std::string *why)
{
    a->qkv = to_lin(h.qkv);
    a->proj = to_lin(h.proj);
    a->scale = h.scale;
    a->shift = h.shift;
    if (!bias_table(h.bias, hg.heads, hg.ws, hg.ws, overlap ? hg.kws : 0, overlap ? hg.kws : 0, overlap ? hg.kpad : 0, &a->rtab, &a->rt_w,
                    &a->rt_offy, &a->rt_offx, &a->tn))
        return refuse_(why, "HAT bias is not a relative-position table");
    return true;
}

class VkHat : public ImgNet {
public:
    explicit VkHat(Device &d) : net_(d) {}
    static std::unique_ptr<ImgNet> build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
    const Buffer &input() const override { return net_.input(); }
    const Buffer &output() const override { return net_.output(); }
    void bind_io(VkBuffer in, VkBuffer out) override { net_.bind_io(in, out); }
    bool record(VkCommandBuffer cb, int w, int h) const override
    {
        if (w != w_ || h != h_) return false;
        net_.record(cb);
        return true;
    }
    int out_w() const override { return w_ * scale_; }
    int out_h() const override { return h_ * scale_; }
    size_t device_bytes() const override { return net_.device_bytes(); }
    size_t weight_bytes() const override { return net_.weight_bytes(); }
    std::string describe() const override { return what_; }
    std::string profile(Device &dev) const override { return net_.profile(dev); }
    void suffix(const std::string &s) override { what_ += s; }

private:
    TNet net_;
    int w_ = 0, h_ = 0, scale_ = 4;
    std::string what_;
};

std::unique_ptr<ImgNet> VkHat::build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    ajinative::HatGraph hg;
    if (!ajinative::find_hat(rd, &hg, why)) return nullptr;
    std::unique_ptr<VkHat> f(new VkHat(dev));
    float mean[3];
    std::copy(hg.ir.mean, hg.ir.mean + 3, mean);
    const Conv &cfirst = hg.first;
    const int C = hg.C, CP = pad16(C), heads = hg.heads, ws = hg.ws, kws = hg.kws, kpad = hg.kpad;
    const Ln pe = to_ln(hg.pe), fin = to_ln(hg.fin);
    const int hd = C / heads;
    if (hd > 32 || hd % 2 || ws * ws % 16 || ws * ws > 256) return refuse_(why, "HAT head size / window"), nullptr;
    const int QS = pad16(3 * C), hid = hg.hid, HS = pad16(hid), cr = hg.cr, cs = hg.cs, nup = hg.nup, S_ = hg.scale();
    const int nl = (int)hg.groups.size(), nb = hg.blocks();
    f->scale_ = S_;
    std::vector<Group> groups;
    for (const ajinative::HatGroup &hgr : hg.groups) {
        Group gr;
        for (const ajinative::HatHab &h : hgr.habs) {
            Hab hb;
            hb.n1 = to_ln(h.n1);
            hb.n2 = to_ln(h.n2);
            if (!to_attn(hg, h.at, false, &hb.at, why)) return nullptr;
            hb.mlp = Mlp{to_lin(h.fc1), to_lin(h.fc2)};
            hb.cab0 = h.cab0;
            hb.cab2 = h.cab2;
            hb.ca1 = h.ca1;
            hb.cab1 = h.cab1;
            hb.ca2 = h.ca2;
            hb.cab2b = h.cab2b;
            hb.conv_scale = h.conv_scale;
            gr.habs.push_back(std::move(hb));
        }
        gr.oc.n1 = to_ln(hgr.oc.n1);
        gr.oc.n2 = to_ln(hgr.oc.n2);
        if (!to_attn(hg, hgr.oc.at, true, &gr.oc.at, why)) return nullptr;
        gr.oc.mlp = Mlp{to_lin(hgr.oc.fc1), to_lin(hgr.oc.fc2)};
        gr.conv = hgr.conv;
        groups.push_back(std::move(gr));
    }
    if (cs > 512 || C > 256 || hid > 4096) return refuse_(why, "HAT widths"), nullptr;
    const Conv &cafter = hg.after, &cbu = hg.cbu, &clast = hg.last;
    if (w < 1 || h < 1) return refuse_(why, "bad tile size"), nullptr;
    f->w_ = w;
    f->h_ = h;
    // the windows' multiple, reflect-padded (one period of the reflection at most)
    const int pw = (w + ws - 1) / ws * ws, ph = (h + ws - 1) / ws * ws;
    if (pw - w >= w || ph - h >= h) return refuse_(why, "the tile is smaller than the padding to HAT's windows"), nullptr;
    const bool pad = pw != w || ph != h;
    const int M = pw * ph, crp = pad16(cr);
    TNet &n = f->net_;

    // ---- the arena. The body: S, GIN (fp32 stream, group input), N1, T60 (the CAB's hidden map),
    // Y (the CAB), AT (attention out), QKV (also the MLP's hidden layer); S16 (the group conv's
    // input) is N1. The tail reuses them: TT in Y, U0 in AT, V1 and the padded output in QKV, V2
    // from S on.
    const VkDeviceSize Mb = (VkDeviceSize)M;
    const VkDeviceSize IN16 = n.place(Mb * 16 * 2), XF = n.place(Mb * CP * 2);
    const VkDeviceSize S = n.place(Mb * CP * 4), GIN = n.place(Mb * CP * 4), N1 = n.place(Mb * CP * 2), T60 = n.place(Mb * crp * 2),
                       Y = n.place(Mb * CP * 2), AT = n.place(Mb * CP * 2);
    const VkDeviceSize QKVB = Mb * std::max(std::max(QS, HS), 256) * 2, QKV = n.place(QKVB), H1 = QKV, S16 = N1;
    const VkDeviceSize SUMS = n.place(CP * 4), CM = n.place(CP * 4);
    const VkDeviceSize TT = Y, U0 = CP >= 64 ? AT : n.place(Mb * 64 * 2), V1 = QKV, O16 = QKV;
    const VkDeviceSize V2 = nup < 2 ? 0 : QKV - S >= Mb * 16 * 64 * 2 ? S : n.place(Mb * 16 * 64 * 2);
    const auto hm = [](VkDeviceSize o, int s, int off = 0) { return hmap(o, s, off); };
    const auto fm = [](VkDeviceSize o, int s) { return fmap(o, s); };
    const auto conv = [&](const Conv &c, const std::vector<uint16_t> &taps, const std::vector<float> &b, int cout, VkDeviceSize src,
                          int sstride, VkDeviceSize dst, int dstride, int H, int W, int act, float slope, VkDeviceSize r1, int out, int sy,
                          std::initializer_list<int> widths) {
        TNet::ConvIO io;
        io.act = act;
        io.slope = slope;
        io.out = out;
        io.src = src;
        io.src_bytes = (VkDeviceSize)H * W * sstride * 2;
        io.src_stride = sstride;
        io.cinp = sstride;
        io.IH = io.H = H;
        io.IW = io.W = W;
        io.sy = sy;
        if (out == 3) {
            io.dst_out = true;
            io.clamp = hg.ir.clamp;
        } else {
            io.dst = dst;
            io.dst_bytes = (VkDeviceSize)H * W * dstride * 2 * (out == 4 ? 2 : 1) * (out == 4 ? 2 : 1);
            io.dst_stride = dstride;
        }
        if (r1 != ~0ull) {
            io.nres = 1;
            io.r1 = r1;
            io.r1_bytes = (VkDeviceSize)H * W * dstride * 2;
            io.r1_stride = dstride;
        }
        n.conv(taps, b, c.cin, cout, io, widths, &taps);
    };
    std::vector<std::vector<uint16_t>> keep;  // the taps outlive the build (their address keys the weights)
    keep.reserve(8 + groups.size() * (nb * 2 + 1));
    const auto taps = [&](const Conv &c, const std::vector<float> *sc) -> const std::vector<uint16_t> & {
        keep.push_back(conv_taps(c, sc));
        return keep.back();
    };
    const auto mat = [&](const Lin &l) { return n.add_mat(l.w, l.b, l.K, l.N); };
    const auto lnorm = [&](const Ln &l, const TMap &in, const TMap &out) { n.layernorm(in, out, M, C, CP, n.add_wf(l.g), n.add_wf(l.b), l.eps); };
    const auto attend = [&](const Attn &a, bool overlap) {
        TWin t;
        t.H = ph;
        t.W = pw;
        t.heads = heads;
        t.hd = hd;
        t.wh = t.ww = ws;
        t.sy = t.sx = overlap ? 0 : a.shift;
        t.mask = !overlap && a.shift;
        t.scale = a.scale;
        t.ko = C;
        t.vo = 2 * C;
        if (overlap) {
            t.kwh = t.kww = kws;
            t.kpad = kpad;
        }
        t.rtab = n.add_wf(a.rtab);
        t.rt_w = a.rt_w;
        t.rt_offy = a.rt_offy;
        t.rt_offx = a.rt_offx;
        t.tn = a.tn;
        n.winattn(t, hm(QKV, QS), hm(AT, CP), overlap ? "winattn ocab" : "winattn");
    };
    const auto mlp_step = [&](const Ln &n2, const Mlp &m) {
        lnorm(n2, fm(S, CP), hm(N1, CP));
        n.gemm(hm(N1, CP), M, mat(m.fc1), hm(H1, HS), 1);
        n.gemm(hm(H1, HS), M, mat(m.fc2), fm(S, CP), 0, false, -1, fm(S, CP));
    };

    // ---- conv_first on the input minus the mean, patch_embed's norm
    n.imgin(hm(IN16, 16), h, w, ph, pw, mean);
    conv(cfirst, taps(cfirst, nullptr), cfirst.b, C, IN16, 16, XF, CP, ph, pw, 0, 0.2f, ~0ull, 0, 0, {64, 48, 32, 16});
    lnorm(pe, hm(XF, CP), fm(S, CP));
    for (const Group &gr : groups) {
        n.copy(S, GIN, Mb * CP * 4);
        for (const Hab &b : gr.habs) {
            lnorm(b.n1, fm(S, CP), hm(N1, CP));
            // the CAB on the normalised map (its last conv carries the scale), its channel gate
            conv(b.cab0, taps(b.cab0, nullptr), b.cab0.b, cr, N1, CP, T60, crp, ph, pw, 1, 0.2f, ~0ull, 0, 0, {64, 48, 32, 16});
            std::vector<float> sc(C, b.conv_scale), b2 = b.cab2.b;
            for (float &x : b2) x *= b.conv_scale;
            conv(b.cab2, taps(b.cab2, &sc), b2, C, T60, crp, Y, CP, ph, pw, 0, 0.2f, ~0ull, 0, 0, {64, 48, 32, 16});
            n.colsums(hm(Y, CP), M, C, SUMS);
            n.chan_gate(SUMS, M, C, cs, n.add_wf(b.ca1), n.add_wf(b.cab1), n.add_wf(b.ca2), n.add_wf(b.cab2b), true, 1.f / b.conv_scale, CM, CP);
            n.gemm(hm(N1, CP), M, mat(b.at.qkv), hm(QKV, QS));
            attend(b.at, false);
            // S += proj(attn) + CAB * gate
            n.gemm(hm(AT, CP), M, mat(b.at.proj), fm(S, CP), 0, false, -1, fm(S, CP), hm(Y, CP), 0.2f, (int)(CM / 4));
            mlp_step(b.n2, b.mlp);
        }
        const Ocab &o = gr.oc;
        lnorm(o.n1, fm(S, CP), hm(N1, CP));
        n.gemm(hm(N1, CP), M, mat(o.at.qkv), hm(QKV, QS));
        attend(o.at, true);
        n.gemm(hm(AT, CP), M, mat(o.at.proj), fm(S, CP), 0, false, -1, fm(S, CP));
        mlp_step(o.n2, o.mlp);
        // the group conv plus the group input, fp32 sums and output
        n.convert(fm(S, CP), hm(S16, CP), M, CP);
        n.conv_gemm(hm(S16, CP), ph, pw, CP, n.add_conv_mat(conv_taps(gr.conv, nullptr), gr.conv.b, C, CP, C), fm(S, CP), fm(GIN, CP));
    }
    // ---- the tail
    lnorm(fin, fm(S, CP), hm(N1, CP));
    conv(cafter, taps(cafter, nullptr), cafter.b, C, N1, CP, TT, CP, ph, pw, 0, 0.2f, XF, 0, 0, {64, 48, 32, 16});
    conv(cbu, taps(cbu, nullptr), cbu.b, 64, TT, CP, U0, 64, ph, pw, 2, cbu.slope, ~0ull, 0, 0, {64, 48, 32, 16});
    // the pixel-shuffle steps U0 -> V1 -> V2, then conv_last at the scale
    VkDeviceSize src = U0;
    for (int k = 0; k < nup; k++)
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(hg.ups[k], sy, 64, &wt, &bt);
            keep.push_back(std::move(wt));
            const int s = 1 << k;
            conv(hg.ups[k], keep.back(), bt, 128, k ? V1 : U0, 64, k ? V2 : V1, 64, s * ph, s * pw, 0, 0.2f, ~0ull, 4, sy, {64, 48, 32, 16});
            src = k ? V2 : V1;
        }
    if (pad) {
        // O16 is V1's region: at 2x conv_last reads V1, so the padded output goes after it
        const VkDeviceSize o16 = nup == 1 ? n.place(Mb * S_ * S_ * 16 * 2) : O16;
        conv(clast, taps(clast, nullptr), clast.b, 3, src, 64, o16, 16, S_ * ph, S_ * pw, 0, 0.2f, ~0ull, 0, 0, {16});
        n.imgout(hm(o16, 16), S_ * h, S_ * w, S_ * pw, hg.ir.clamp);
    } else {
        conv(clast, taps(clast, nullptr), clast.b, 3, src, 64, 0, 0, S_ * ph, S_ * pw, 0, 0.2f, ~0ull, 3, 0, {16});
    }
    if (!n.finish((VkDeviceSize)3 * w * h * 2, (VkDeviceSize)3 * f->out_w() * f->out_h() * 2, why)) return nullptr;
    char buf[240];
    std::snprintf(buf, sizeof buf, "HAT %dx, %d groups x %d blocks + OCAB, %d channels, %d heads, %dx%d windows (shifted), fp32 stream%s%s; Vulkan %s",
                  S_, nl, nb, C, heads, ws, ws, hg.ir.clamp ? ", clamped" : "",
                  pad ? (", padded to " + std::to_string(pw) + "x" + std::to_string(ph)).c_str() : "", family_name(dev.family()));
    f->what_ = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

}  // namespace

std::unique_ptr<ImgNet> build_hat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    return VkHat::build(dev, g, w, h, why);
}

}  // namespace ajivk
