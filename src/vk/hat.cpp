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

bool ln(const Reader &rd, const std::string &name, int C, Ln *l, std::string *why)
{
    const ajinative::OnnxNode *n = rd.node(name);
    if (!n || n->op != "LayerNormalization" || !tensor_floats(rd, n, 1, &l->g) || !tensor_floats(rd, n, 2, &l->b) || (int)l->g.size() != C ||
        (int)l->b.size() != C)
        return refuse_(why, "missing " + name);
    l->eps = n->attr("epsilon") ? n->attr("epsilon")->f : 1e-5f;
    return true;
}

// Linear: MatMul [K][N] then Add of the bias.
bool linear(const Reader &rd, const std::string &pre, int K, int N, Lin *l, std::string *why)
{
    std::vector<int64_t> d;
    if (!tensor_floats(rd, rd.node(pre + "/MatMul"), 1, &l->w, &d) || d != std::vector<int64_t>{K, N} ||
        !init_input(rd, rd.node(pre + "/Add"), &l->b) || (int)l->b.size() != N)
        return refuse_(why, "missing linear " + pre);
    l->K = K;
    l->N = N;
    return true;
}

// The scalar constant input of a Mul (attention scales, the CAB scale).
bool scalar(const Reader &rd, const std::string &name, float *v)
{
    std::vector<float> x;
    if (!init_input(rd, rd.node(name), &x) || x.size() != 1) return false;
    *v = x[0];
    return true;
}

// The relative-position table behind a full bias [heads][wh*ww][Tk] (native/tfm.cu's tbias_table):
// bias(q, k) = tab[h][(ky - kpad - qy + offy) * tw + (kx - kpad - qx + offx)].
bool bias_table(const std::vector<float> &bias, int heads, int ws, int kws, int kpad, Attn *a)
{
    const int kh = kws ? kws : ws, kw = kh, kp = kws ? kpad : 0, Tq = ws * ws, Tk = kh * kw;
    const int th = kh + ws - 1, tw = kw + ws - 1, offy = kp + ws - 1, offx = kp + ws - 1;
    if ((int)bias.size() != heads * Tq * Tk) return false;
    a->rtab.assign((size_t)heads * th * tw, 0.f);
    std::vector<char> seen(a->rtab.size(), 0);
    for (int h = 0; h < heads; h++)
        for (int r = 0; r < Tq; r++)
            for (int c = 0; c < Tk; c++) {
                const int dy = c / kw - kp - r / ws + offy, dx = c % kw - kp - r % ws + offx;
                if (dy < 0 || dy >= th || dx < 0 || dx >= tw) return false;
                const size_t ti = ((size_t)h * th + dy) * tw + dx;
                const float v = bias[((size_t)h * Tq + r) * Tk + c];
                if (seen[ti] && a->rtab[ti] != v) return false;
                seen[ti] = 1;
                a->rtab[ti] = v;
            }
    a->rt_w = tw;
    a->rt_offy = offy;
    a->rt_offx = offx;
    a->tn = th * tw;
    return true;
}

bool attn(const Reader &rd, const std::string &pre, const char *scale_mul, int C, int heads, int ws, int kws, int kpad, Attn *a, std::string *why)
{
    if (!linear(rd, pre + "qkv", C, 3 * C, &a->qkv, why) || !linear(rd, pre + "proj", C, C, &a->proj, why)) return false;
    if (!scalar(rd, pre + scale_mul, &a->scale)) return refuse_(why, pre + ": attention scale");
    std::vector<float> bias;
    std::vector<int64_t> d;
    const int Tq = ws * ws, Tk = kws ? kws * kws : Tq;
    if (!init_input(rd, rd.node(pre + "Add"), &bias, &d) || d != std::vector<int64_t>{1, heads, Tq, Tk}) return refuse_(why, pre + ": bias");
    if (!bias_table(bias, heads, ws, kws, kpad, a)) return refuse_(why, pre + ": bias is not a relative-position table");
    return true;
}

bool mlp(const Reader &rd, const std::string &pre, int C, int hid, Mlp *m, std::string *why)
{
    if (!rd.node(pre + "act/Gelu")) return refuse_(why, pre + ": activation");
    return linear(rd, pre + "fc1", C, hid, &m->fc1, why) && linear(rd, pre + "fc2", hid, C, &m->fc2, why);
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
    int out_w() const override { return w_ * 4; }
    int out_h() const override { return h_ * 4; }
    size_t device_bytes() const override { return net_.device_bytes(); }
    size_t weight_bytes() const override { return net_.weight_bytes(); }
    std::string describe() const override { return what_; }
    std::string profile(Device &dev) const override { return net_.profile(dev); }
    void suffix(const std::string &s) override { what_ += s; }

private:
    TNet net_;
    int w_ = 0, h_ = 0;
    std::string what_;
};

std::unique_ptr<ImgNet> VkHat::build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    if (!rd.node("/layers.0/residual_group/blocks.0/attn/qkv/MatMul") || !rd.node("/layers.0/residual_group/overlap_attn/qkv/MatMul"))
        return refuse_(why, "not a HAT graph"), nullptr;
    std::unique_ptr<VkHat> f(new VkHat(dev));
    std::vector<float> v;
    std::vector<int64_t> d;
    float mean[3];
    if (!init_input(rd, rd.node("/Sub"), &v, &d) || v.size() != 3) return refuse_(why, "HAT input mean"), nullptr;
    std::copy(v.begin(), v.end(), mean);
    Conv cfirst;
    if (!rd.conv("/conv_first/Conv", &cfirst, why)) return nullptr;
    if (cfirst.cin != 3 || cfirst.lrelu) return refuse_(why, "HAT conv_first"), nullptr;
    const int C = cfirst.cout, CP = pad16(C);
    Ln pe, fin;
    if (!ln(rd, "/patch_embed/norm/LayerNormalization", C, &pe, why) || !ln(rd, "/norm/LayerNormalization", C, &fin, why)) return nullptr;
    // window sizes and heads from the first block's bias and the OCAB's
    int heads = 0, ws = 16, kws = 24, kpad = 4;
    {
        std::vector<float> b;
        if (!init_input(rd, rd.node("/layers.0/residual_group/blocks.0/attn/Add"), &b, &d) || d.size() != 4) return refuse_(why, "HAT bias"), nullptr;
        heads = (int)d[1];
        ws = (int)std::lround(std::sqrt((double)d[2]));
        if (heads < 1 || ws < 1 || (int64_t)ws * ws != d[2] || ws * ws > 256) return refuse_(why, "HAT bias"), nullptr;
        if (!init_input(rd, rd.node("/layers.0/residual_group/overlap_attn/Add"), &b, &d) || d.size() != 4)
            return refuse_(why, "HAT OCAB bias"), nullptr;
        kws = (int)std::lround(std::sqrt((double)d[3]));
        if (kws < ws || (int64_t)kws * kws != d[3] || (kws - ws) % 2) return refuse_(why, "HAT OCAB bias"), nullptr;
        kpad = (kws - ws) / 2;
        std::vector<int64_t> pads;
        const ajinative::OnnxNode *pn = rd.node("/layers.0/residual_group/overlap_attn/unfold/Pad");
        const ajinative::OnnxTensor *pt = pn && pn->in.size() > 1 ? rd.tensor(pn->in[1]) : nullptr;
        if (!pt || !pt->ints(&pads) || pads.size() != 8 || pads[2] != kpad || pads[3] != kpad) return refuse_(why, "HAT OCAB padding"), nullptr;
    }
    if (C % heads) return refuse_(why, "HAT heads"), nullptr;
    const int hd = C / heads;
    if (hd > 32 || hd % 2 || ws * ws % 16) return refuse_(why, "HAT head size / window"), nullptr;
    const int QS = pad16(3 * C);
    int nl = 0, nb = 0;
    while (rd.node("/layers." + std::to_string(nl) + "/conv/Conv")) nl++;
    while (rd.node("/layers.0/residual_group/blocks." + std::to_string(nb) + "/norm1/LayerNormalization")) nb++;
    if (!nl || !nb) return refuse_(why, "HAT layers"), nullptr;
    int hid = 0;
    if (!tensor_floats(rd, rd.node("/layers.0/residual_group/blocks.0/mlp/fc1/MatMul"), 1, &v, &d) || d.size() != 2 || d[0] != C || d[1] < 1)
        return refuse_(why, "HAT mlp"), nullptr;
    hid = (int)d[1];
    const int HS = pad16(hid);
    int cr = 0, cs = 0;
    std::vector<Group> groups;
    for (int l = 0; l < nl; l++) {
        Group gr;
        const std::string L = "/layers." + std::to_string(l) + "/";
        for (int b = 0; b < nb; b++) {
            const std::string B = L + "residual_group/blocks." + std::to_string(b) + "/";
            Hab hb;
            if (!ln(rd, B + "norm1/LayerNormalization", C, &hb.n1, why) || !ln(rd, B + "norm2/LayerNormalization", C, &hb.n2, why) ||
                !attn(rd, B + "attn/", "Mul", C, heads, ws, 0, 0, &hb.at, why) || !mlp(rd, B + "mlp/", C, hid, &hb.mlp, why))
                return nullptr;
            if (const ajinative::OnnxNode *sl = rd.node(B + "Slice")) {  // the roll: Slice(start = shift) + Slice(0, shift)
                std::vector<int64_t> st;
                const ajinative::OnnxTensor *t = sl->in.size() > 1 ? rd.tensor(sl->in[1]) : nullptr;
                if (!t || !t->ints(&st) || st.size() != 1 || st[0] <= 0 || st[0] >= ws) return refuse_(why, B + "shift"), nullptr;
                hb.at.shift = (int)st[0];
            }
            bool got = false;  // the CAB scale: the block's scalar Mul in (0, 1) (shape Muls carry integers)
            for (const char *m : {"Mul", "Mul_1", "Mul_2"})
                if (!got && scalar(rd, B + m, &hb.conv_scale) && hb.conv_scale > 0.f && hb.conv_scale < 1.f) got = true;
            if (!got) return refuse_(why, B + "conv scale"), nullptr;
            const std::string CB = B + "conv_block/cab/";
            if (!rd.conv(CB + "cab.0/Conv", &hb.cab0, why) || !rd.conv(CB + "cab.2/Conv", &hb.cab2, why)) return nullptr;
            if (hb.cab0.cin != C || hb.cab2.cout != C || hb.cab2.cin != hb.cab0.cout || !rd.node(CB + "cab.1/Gelu") || hb.cab0.lrelu ||
                hb.cab2.lrelu)
                return refuse_(why, B + "CAB"), nullptr;
            if (cr && cr != hb.cab0.cout) return refuse_(why, B + "CAB width"), nullptr;
            cr = hb.cab0.cout;
            const ajinative::OnnxNode *a1 = rd.node(CB + "cab.3/attention/attention.1/Conv"), *a3 = rd.node(CB + "cab.3/attention/attention.3/Conv");
            std::vector<int64_t> d3;
            if (!tensor_floats(rd, a1, 1, &hb.ca1, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 ||
                !tensor_floats(rd, a3, 1, &hb.ca2, &d3) || d3 != std::vector<int64_t>{C, d[0], 1, 1} || a1->in.size() < 3 ||
                !rd.floats(a1->in[2], &hb.cab1) || (int64_t)hb.cab1.size() != d[0] || a3->in.size() < 3 || !rd.floats(a3->in[2], &hb.cab2b) ||
                (int)hb.cab2b.size() != C || !rd.node(CB + "cab.3/attention/attention.2/Relu"))
                return refuse_(why, B + "CAB channel attention"), nullptr;
            if (cs && cs != (int)hb.cab1.size()) return refuse_(why, B + "CAB channel attention width"), nullptr;
            cs = (int)hb.cab1.size();
            gr.habs.push_back(std::move(hb));
        }
        const std::string O = L + "residual_group/overlap_attn/";
        if (!ln(rd, O + "norm1/LayerNormalization", C, &gr.oc.n1, why) || !ln(rd, O + "norm2/LayerNormalization", C, &gr.oc.n2, why) ||
            !attn(rd, O, "Mul_2", C, heads, ws, kws, kpad, &gr.oc.at, why) || !mlp(rd, O + "mlp/", C, hid, &gr.oc.mlp, why))
            return nullptr;
        if (!rd.conv(L + "conv/Conv", &gr.conv, why)) return nullptr;
        if (gr.conv.cin != C || gr.conv.cout != C || gr.conv.lrelu) return refuse_(why, L + "conv"), nullptr;
        groups.push_back(std::move(gr));
    }
    if (cs > 512 || C > 256 || hid > 4096) return refuse_(why, "HAT widths"), nullptr;
    Conv cafter, cbu, cup[2], clast;
    if (!rd.conv("/conv_after_body/Conv", &cafter, why)) return nullptr;
    if (!rd.conv("/conv_before_upsample/conv_before_upsample.0/Conv", &cbu, why)) return nullptr;
    if (!cbu.lrelu || cbu.cin != C || cbu.cout != 64) return refuse_(why, "HAT conv_before_upsample"), nullptr;
    for (int k = 0; k < 2; k++) {
        if (!rd.conv(k ? "/upsample/upsample.2/Conv" : "/upsample/upsample.0/Conv", &cup[k], why)) return nullptr;
        if (cup[k].cin != 64 || cup[k].cout != 256 || cup[k].lrelu) return refuse_(why, "HAT upsample"), nullptr;
    }
    if (!rd.node("/upsample/upsample.1/DepthToSpace") || !rd.node("/upsample/upsample.3/DepthToSpace")) return refuse_(why, "HAT upsample"), nullptr;
    if (!rd.conv("/conv_last/Conv", &clast, why)) return nullptr;
    if (clast.cin != 64 || clast.cout != 3) return refuse_(why, "HAT conv_last"), nullptr;
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
    const VkDeviceSize V2 = QKV - S >= Mb * 16 * 64 * 2 ? S : n.place(Mb * 16 * 64 * 2);
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
    const int nwx = pw / ws, nwy = ph / ws, nqb = (ws * ws + 63) / 64;
    const auto attend = [&](const Attn &a, bool overlap) {
        TArgs t;
        t.map(0, hm(QKV, QS));
        t.map(1, hm(AT, CP));
        t.v[16] = ph;
        t.v[17] = pw;
        t.v[18] = heads;
        t.v[19] = hd;
        t.v[20] = t.v[21] = ws;
        t.v[22] = t.v[23] = overlap ? 0 : a.shift;
        t.v[28] = nwx;
        t.v[29] = ph;
        t.v[30] = pw;
        t.f(31, a.scale);
        t.v[32] = -1;
        t.v[33] = 0;
        t.v[34] = C;
        t.v[35] = 2 * C;
        const long long wg = (long long)nwx * nwy * heads * nqb;
        t.v[36] = (int32_t)wg;
        t.v[37] = nqb;
        t.v[38] = overlap ? kws : 0;
        t.v[39] = overlap ? kws : 0;
        t.v[40] = overlap ? kpad : 0;
        t.v[41] = !overlap && a.shift ? 1 : 0;
        t.v[42] = n.add_wf(a.rtab);
        t.v[43] = a.rt_w;
        t.v[44] = a.rt_offy;
        t.v[45] = a.rt_offx;
        t.v[46] = a.tn;
        const uint32_t gx = (uint32_t)std::min<long long>(wg, 65535), gy = (uint32_t)((wg + gx - 1) / gx);
        n.pass(5, t, gx, gy, overlap ? "winattn ocab" : "winattn");
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
    for (int k = 0; k < 2; k++)
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(cup[k], sy, 64, &wt, &bt);
            keep.push_back(std::move(wt));
            const int s = k ? 2 : 1;
            conv(cup[k], keep.back(), bt, 128, k ? V1 : U0, 64, k ? V2 : V1, 64, s * ph, s * pw, 0, 0.2f, ~0ull, 4, sy, {64, 48, 32, 16});
        }
    if (pad) {
        conv(clast, taps(clast, nullptr), clast.b, 3, V2, 64, O16, 16, 4 * ph, 4 * pw, 0, 0.2f, ~0ull, 0, 0, {16});
        n.imgout(hm(O16, 16), 4 * h, 4 * w, 4 * pw);
    } else {
        conv(clast, taps(clast, nullptr), clast.b, 3, V2, 64, 0, 0, 4 * ph, 4 * pw, 0, 0.2f, ~0ull, 3, 0, {16});
    }
    if (!n.finish((VkDeviceSize)3 * w * h * 2, (VkDeviceSize)3 * f->out_w() * f->out_h() * 2, why)) return nullptr;
    char buf[240];
    std::snprintf(buf, sizeof buf, "HAT 4x, %d groups x %d blocks + OCAB, %d channels, %d heads, %dx%d windows (shifted), fp32 stream%s; Vulkan %s", nl,
                  nb, C, heads, ws, ws, pad ? (", padded to " + std::to_string(pw) + "x" + std::to_string(ph)).c_str() : "",
                  family_name(dev.family()));
    f->what_ = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

}  // namespace

std::unique_ptr<ImgNet> build_hat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    return VkHat::build(dev, g, w, h, why);
}

}  // namespace ajivk
