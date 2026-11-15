// native/hat.cpp — HAT (Hybrid Attention Transformer; IllustrationJaNai's HAT-L 4x, bf16 exports) on
// the tfm and gconv kernels.
//
// conv_first (on the input minus the dataset mean), patch_embed's LayerNorm, then per residual group:
// HABs — x += proj(W-MSA(LN1 x)) + 0.01 * CAB(LN1 x); x += MLP(LN2 x) — where the window attention
// (16x16, shifted by 8 with Swin's mask in every second block) takes its bias from the relative-position
// table and the CAB (3x3 C -> C/3, GELU, 3x3 back, channel attention) folds the 0.01 into its last conv;
// an OCAB (16x16 queries over zero-padded overlapping 24x24 key windows) and its MLP; the group conv
// plus the group input. Then the final LayerNorm, conv_after_body plus conv_first's output, and the
// pixel-shuffle upsampler. The models are bf16-trained and their residual stream outgrows fp16 (the
// last group's conv reaches ~1e6), so the stream is fp32; everything inside the blocks is fp16 with
// fp32 accumulation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "native/fdat_kernels.h"
#include "native/img_kernels.h"
#include "native/imgnet.h"
#include "native/netres.h"

namespace ajinative {
namespace {

using namespace onnxread;

struct Ln {
    float *g = nullptr, *b = nullptr, eps = 1e-5f;
};
struct Attn {
    TMat *qkv = nullptr, *proj = nullptr;
    float scale = 1.f, *rtab = nullptr;
    int rt_w = 0, rt_offy = 0, rt_offx = 0, shift = 0;
};
struct Mlp {
    TMat *fc1 = nullptr, *fc2 = nullptr;
};
struct Hab {
    Ln n1, n2;
    Attn at;
    Mlp mlp;
    GOp cab0, cab2;
    float *ca1 = nullptr, *cab1 = nullptr, *ca2 = nullptr, *cab2b = nullptr, conv_scale = 1.f;
};
struct Ocab {
    Ln n1, n2;
    Attn at;
    Mlp mlp;
};
struct Group {
    std::vector<Hab> habs;
    Ocab oc;
    GOp conv;
};

class Hat : public ImgNet {
public:
    static std::unique_ptr<ImgNet> build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override;
    int scale() const override { return 4; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes; }
    bool accepts(int w, int h) const override { return w > 0 && h > 0 && w <= maxw && h <= maxh && w % ws == 0 && h % ws == 0; }

private:
    DevRes res;
    int C = 0, CP = 0, QS = 0, HS = 0, heads = 0, hd = 0, ws = 16, kws = 24, kpad = 4, cr = 0, crp = 0, cs = 0, maxw = 0, maxh = 0;
    float mean[3] = {}, cbu_slope = 0.01f;
    Ln pe, fin;
    std::vector<Group> groups;
    GOp first, after, cbu, up0[2], up2[2], last;
    void *IN16 = nullptr, *XF = nullptr, *N1 = nullptr, *QKV = nullptr, *AT = nullptr, *T60 = nullptr, *Y = nullptr, *H1 = nullptr;
    void *S16 = nullptr, *TT = nullptr, *U0 = nullptr, *V1 = nullptr, *V2 = nullptr, *O16 = nullptr;
    float *S = nullptr, *GIN = nullptr, *sums = nullptr, *cm = nullptr, *wsp = nullptr;
    size_t ws_floats = 0;
    std::string what;

    cudaError_t exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
                     std::vector<std::pair<GOp *, GConvArgs>> *collect);
};

bool ln(const Reader &rd, DevRes &r, const std::string &name, int C, Ln *l, std::string *why)
{
    const OnnxNode *n = rd.node(name);
    std::vector<float> g, b;
    if (!n || n->op != "LayerNormalization" || !tensor_floats(rd, n, 1, &g) || !tensor_floats(rd, n, 2, &b) || (int)g.size() != C ||
        (int)b.size() != C)
        return refuse(why, "missing " + name);
    l->g = r.dfl(g, C);
    l->b = r.dfl(b, C);
    l->eps = n->attr("epsilon") ? n->attr("epsilon")->f : 1e-5f;
    return true;
}

// Linear: MatMul [K][N] then Add of the bias.
TMat *linear(const Reader &rd, DevRes &r, const std::string &pre, int K, int N, std::string *why)
{
    std::vector<float> w, b;
    std::vector<int64_t> d;
    if (!tensor_floats(rd, rd.node(pre + "/MatMul"), 1, &w, &d) || d != std::vector<int64_t>{K, N} || !init_input(rd, rd.node(pre + "/Add"), &b) ||
        (int)b.size() != N)
        return refuse(why, "missing linear " + pre), nullptr;
    return r.mat(w, &b, K, N);
}

// The scalar constant input of a Mul (attention scales, the CAB scale).
bool scalar(const Reader &rd, const std::string &name, float *v)
{
    std::vector<float> x;
    if (!init_input(rd, rd.node(name), &x) || x.size() != 1) return false;
    *v = x[0];
    return true;
}

bool attn(const Reader &rd, DevRes &r, const std::string &pre, const char *scale_mul, int C, int heads, int ws, int kws, int kpad, Attn *a,
          std::string *why)
{
    if (!(a->qkv = linear(rd, r, pre + "qkv", C, 3 * C, why)) || !(a->proj = linear(rd, r, pre + "proj", C, C, why))) return false;
    if (!scalar(rd, pre + scale_mul, &a->scale)) return refuse(why, pre + ": attention scale");
    std::vector<float> bias;
    std::vector<int64_t> d;
    const int Tq = ws * ws, Tk = kws ? kws * kws : Tq;
    if (!init_input(rd, rd.node(pre + "Add"), &bias, &d) || d != std::vector<int64_t>{1, heads, Tq, Tk}) return refuse(why, pre + ": bias");
    std::vector<float> tab;
    if (!tbias_table(bias, heads, ws, ws, kws, kws, kpad, &tab, &a->rt_w, &a->rt_offy, &a->rt_offx))
        return refuse(why, pre + ": bias is not a relative-position table");
    a->rtab = r.dfl(tab, tab.size());
    return true;
}

bool mlp(const Reader &rd, DevRes &r, const std::string &pre, int C, int hid, Mlp *m, std::string *why)
{
    if (!rd.node(pre + "act/Gelu")) return refuse(why, pre + ": activation");
    return (m->fc1 = linear(rd, r, pre + "fc1", C, hid, why)) && (m->fc2 = linear(rd, r, pre + "fc2", hid, C, why));
}

std::unique_ptr<ImgNet> Hat::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    if (!rd.node("/layers.0/residual_group/blocks.0/attn/qkv/MatMul") || !rd.node("/layers.0/residual_group/overlap_attn/qkv/MatMul"))
        return refuse(why, "not a HAT graph"), nullptr;
    std::unique_ptr<Hat> f(new Hat);
    DevRes &r = f->res;
    std::vector<float> v;
    std::vector<int64_t> d;
    if (!init_input(rd, rd.node("/Sub"), &v, &d) || v.size() != 3) return refuse(why, "HAT input mean"), nullptr;
    std::copy(v.begin(), v.end(), f->mean);
    Conv c;
    if (!rd.conv("/conv_first/Conv", &c, why)) return nullptr;
    if (c.cin != 3 || c.lrelu) return refuse(why, "HAT conv_first"), nullptr;
    const int C = f->C = c.cout, CP = f->CP = pad16(C);
    f->first.w = r.gup(conv_taps(c, nullptr), c.b, 3, 16, C, CP);
    if (!ln(rd, r, "/patch_embed/norm/LayerNormalization", C, &f->pe, why) || !ln(rd, r, "/norm/LayerNormalization", C, &f->fin, why))
        return nullptr;
    // window sizes and heads from the first block's bias and the OCAB's
    {
        std::vector<float> b;
        if (!init_input(rd, rd.node("/layers.0/residual_group/blocks.0/attn/Add"), &b, &d) || d.size() != 4) return refuse(why, "HAT bias"), nullptr;
        f->heads = (int)d[1];
        f->ws = (int)std::lround(std::sqrt((double)d[2]));
        if (f->heads < 1 || f->ws < 1 || (int64_t)f->ws * f->ws != d[2] || f->ws * f->ws > 256) return refuse(why, "HAT bias"), nullptr;
        if (!init_input(rd, rd.node("/layers.0/residual_group/overlap_attn/Add"), &b, &d) || d.size() != 4) return refuse(why, "HAT OCAB bias"), nullptr;
        f->kws = (int)std::lround(std::sqrt((double)d[3]));
        if (f->kws < f->ws || (int64_t)f->kws * f->kws != d[3] || (f->kws - f->ws) % 2) return refuse(why, "HAT OCAB bias"), nullptr;
        f->kpad = (f->kws - f->ws) / 2;
        std::vector<int64_t> pads;
        const OnnxNode *pn = rd.node("/layers.0/residual_group/overlap_attn/unfold/Pad");
        const OnnxTensor *pt = pn && pn->in.size() > 1 ? rd.tensor(pn->in[1]) : nullptr;
        if (!pt || !pt->ints(&pads) || pads.size() != 8 || pads[2] != f->kpad || pads[3] != f->kpad) return refuse(why, "HAT OCAB padding"), nullptr;
    }
    if (C % f->heads) return refuse(why, "HAT heads"), nullptr;
    f->hd = C / f->heads;
    if (f->hd > 32 || f->hd % 2 || f->ws * f->ws % 16 || f->kws * f->kws > 576) return refuse(why, "HAT head size / window"), nullptr;
    f->QS = pad16(3 * C);
    int nl = 0, nb = 0;
    while (rd.node("/layers." + std::to_string(nl) + "/conv/Conv")) nl++;
    while (rd.node("/layers.0/residual_group/blocks." + std::to_string(nb) + "/norm1/LayerNormalization")) nb++;
    if (!nl || !nb) return refuse(why, "HAT layers"), nullptr;
    int hid = 0;
    {
        if (!tensor_floats(rd, rd.node("/layers.0/residual_group/blocks.0/mlp/fc1/MatMul"), 1, &v, &d) || d.size() != 2) return refuse(why, "HAT mlp"), nullptr;
        hid = (int)d[1];
        if (hid < 1 || d[0] != C) return refuse(why, "HAT mlp"), nullptr;
        f->HS = pad16(hid);
    }
    for (int l = 0; l < nl; l++) {
        Group gr;
        const std::string L = "/layers." + std::to_string(l) + "/";
        for (int b = 0; b < nb; b++) {
            const std::string B = L + "residual_group/blocks." + std::to_string(b) + "/";
            Hab hb;
            if (!ln(rd, r, B + "norm1/LayerNormalization", C, &hb.n1, why) || !ln(rd, r, B + "norm2/LayerNormalization", C, &hb.n2, why) ||
                !attn(rd, r, B + "attn/", "Mul", C, f->heads, f->ws, 0, 0, &hb.at, why) || !mlp(rd, r, B + "mlp/", C, hid, &hb.mlp, why))
                return nullptr;
            if (const OnnxNode *sl = rd.node(B + "Slice")) {  // the roll before the partition: Slice(start = shift) + Slice(0, shift)
                std::vector<int64_t> st;
                const OnnxTensor *t = sl->in.size() > 1 ? rd.tensor(sl->in[1]) : nullptr;
                if (!t || !t->ints(&st) || st.size() != 1 || st[0] <= 0 || st[0] >= f->ws) return refuse(why, B + "shift"), nullptr;
                hb.at.shift = (int)st[0];
            }
            // CAB: conv scale (the Mul of the block with a scalar outside attn/conv_block)
            bool got = false;  // (shape arithmetic Muls carry integers >= 1)
            for (const char *m : {"Mul", "Mul_1", "Mul_2"})
                if (!got && scalar(rd, B + m, &hb.conv_scale) && hb.conv_scale > 0.f && hb.conv_scale < 1.f) got = true;
            if (!got) return refuse(why, B + "conv scale"), nullptr;
            const std::string CB = B + "conv_block/cab/";
            Conv c0, c2;
            if (!rd.conv(CB + "cab.0/Conv", &c0, why) || !rd.conv(CB + "cab.2/Conv", &c2, why)) return nullptr;
            if (c0.cin != C || c2.cout != C || c2.cin != c0.cout || !rd.node(CB + "cab.1/Gelu") || c0.lrelu || c2.lrelu)
                return refuse(why, B + "CAB"), nullptr;
            if (f->cr && f->cr != c0.cout) return refuse(why, B + "CAB width"), nullptr;  // the buffers are sized once
            f->cr = c0.cout;
            f->crp = pad16(f->cr);
            hb.cab0.w = r.gup(conv_taps(c0, nullptr), c0.b, C, CP, f->cr, f->crp);
            std::vector<float> sc(C, hb.conv_scale), b2 = c2.b;
            for (float &x : b2) x *= hb.conv_scale;
            hb.cab2.w = r.gup(conv_taps(c2, &sc), b2, f->cr, f->crp, C, CP);
            std::vector<float> w1, bb1, w2, bb2;
            const OnnxNode *a1 = rd.node(CB + "cab.3/attention/attention.1/Conv"), *a3 = rd.node(CB + "cab.3/attention/attention.3/Conv");
            std::vector<int64_t> d3;
            if (!tensor_floats(rd, a1, 1, &w1, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 ||
                !tensor_floats(rd, a3, 1, &w2, &d3) || d3 != std::vector<int64_t>{C, d[0], 1, 1} || a1->in.size() < 3 ||
                !rd.floats(a1->in[2], &bb1) || (int64_t)bb1.size() != d[0] || a3->in.size() < 3 || !rd.floats(a3->in[2], &bb2) ||
                (int)bb2.size() != C || !rd.node(CB + "cab.3/attention/attention.2/Relu"))
                return refuse(why, B + "CAB channel attention"), nullptr;
            if (f->cs && f->cs != (int)bb1.size()) return refuse(why, B + "CAB channel attention width"), nullptr;
            f->cs = (int)bb1.size();
            hb.ca1 = r.dfl(w1, (size_t)f->cs * C);
            hb.cab1 = r.dfl(bb1, f->cs);
            hb.ca2 = r.dfl(w2, (size_t)C * f->cs);
            hb.cab2b = r.dfl(bb2, C);
            gr.habs.push_back(hb);
        }
        const std::string O = L + "residual_group/overlap_attn/";
        if (!ln(rd, r, O + "norm1/LayerNormalization", C, &gr.oc.n1, why) || !ln(rd, r, O + "norm2/LayerNormalization", C, &gr.oc.n2, why) ||
            !attn(rd, r, O, "Mul_2", C, f->heads, f->ws, f->kws, f->kpad, &gr.oc.at, why) || !mlp(rd, r, O + "mlp/", C, hid, &gr.oc.mlp, why))
            return nullptr;
        if (!rd.conv(L + "conv/Conv", &c, why)) return nullptr;
        if (c.cin != C || c.cout != C || c.lrelu) return refuse(why, L + "conv"), nullptr;
        gr.conv.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, C, CP);
        f->groups.push_back(std::move(gr));
    }
    // tail
    if (!rd.conv("/conv_after_body/Conv", &c, why)) return nullptr;
    f->after.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, C, CP);
    if (!rd.conv("/conv_before_upsample/conv_before_upsample.0/Conv", &c, why)) return nullptr;
    if (!c.lrelu || c.cin != C || c.cout != 64) return refuse(why, "HAT conv_before_upsample"), nullptr;
    f->cbu_slope = c.slope;
    f->cbu.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, 64, 64);
    for (int k = 0; k < 2; k++) {
        if (!rd.conv(k ? "/upsample/upsample.2/Conv" : "/upsample/upsample.0/Conv", &c, why)) return nullptr;
        if (c.cin != 64 || c.cout != 256 || c.lrelu) return refuse(why, "HAT upsample"), nullptr;
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(c, sy, 64, &wt, &bt);
            (k ? f->up2 : f->up0)[sy].w = r.gup(wt, bt, 64, 64, 128, 128);
        }
    }
    if (!rd.node("/upsample/upsample.1/DepthToSpace") || !rd.node("/upsample/upsample.3/DepthToSpace")) return refuse(why, "HAT upsample"), nullptr;
    if (!rd.conv("/conv_last/Conv", &c, why)) return nullptr;
    if (c.cin != 64 || c.cout != 3) return refuse(why, "HAT conv_last"), nullptr;
    f->last.w = r.gup(conv_taps(c, nullptr), c.b, 64, 64, 3, 16);
    if (r.failed) return refuse(why, "weight upload failed"), nullptr;

    // buffers for the largest tile
    f->maxw = max_w;
    f->maxh = max_h;
    const size_t M = (size_t)max_w * max_h;
    f->IN16 = r.alloc(M * 32);
    f->XF = r.alloc(M * CP * 2);
    f->N1 = r.alloc(M * CP * 2);
    f->QKV = r.alloc(M * f->QS * 2);
    f->AT = r.alloc(M * CP * 2);
    f->T60 = r.alloc(M * f->crp * 2);
    f->Y = r.alloc(M * CP * 2);
    f->H1 = r.alloc(M * f->HS * 2);
    f->S16 = r.alloc(M * CP * 2);
    f->TT = r.alloc(M * CP * 2);
    f->U0 = r.alloc(M * 64 * 2);
    f->V1 = r.alloc(4 * M * 64 * 2);
    f->V2 = r.alloc(16 * M * 64 * 2);
    f->O16 = r.alloc(16 * M * 16 * 2);
    f->S = (float *)r.alloc(M * CP * 4);
    f->GIN = (float *)r.alloc(M * CP * 4);
    f->sums = (float *)r.alloc(CP * 4);
    f->cm = (float *)r.alloc(CP * 4);
    if (r.failed) return refuse(why, "out of GPU memory"), nullptr;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        ops.clear();
        const int w = sz ? std::min(max_w, 64) / f->ws * f->ws : max_w, h = sz ? std::min(max_h, 64) / f->ws * f->ws : max_h;
        f->exec(nullptr, false, nullptr, false, w, h, s, &ops);
        f->ws_floats = std::max(f->ws_floats, gworkspace_max(ops));
    }
    if (f->ws_floats && !(f->wsp = (float *)r.alloc(f->ws_floats * 4))) return refuse(why, "out of GPU memory"), nullptr;
    ops.clear();
    f->exec(nullptr, false, nullptr, false, max_w, max_h, s, &ops);
    if (!gprepare(ops, s, why)) return nullptr;
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    char buf[200];
    snprintf(buf, sizeof(buf), "HAT 4x, %d groups x %d blocks + OCAB, %d channels, %d heads, %dx%d windows (shifted), fp32 stream", nl, nb, C,
             f->heads, f->ws, f->ws);
    f->what = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

cudaError_t Hat::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    return exec(in, in_f32, out, out_f32, w, h, s, nullptr);
}

cudaError_t Hat::exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
                      std::vector<std::pair<GOp *, GConvArgs>> *collect)
{
    const int M = w * h;
    StepProf pr(s, !collect);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return e_; } while (0)
#define K(x) do { if (!collect) CK(x); } while (0)
    auto Gc = [&](GOp &op, GConvArgs a) -> cudaError_t {
        a.ws = wsp;
        a.ws_floats = ws_floats;
        if (collect) {
            collect->push_back({&op, a});
            return cudaSuccess;
        }
        return glaunch(*op.w, a, op.variant, s);
    };
    auto gemm = [&](const TMat &m, const void *a, int sa, void *o, int so, TAct act, bool f32_stream, const void *add2 = nullptr,
                    const float *add2scale = nullptr) -> cudaError_t {
        if (collect) return cudaSuccess;
        TGemm gm;
        gm.a = a; gm.sa = sa; gm.M = M; gm.out = o; gm.so = so; gm.act = act;
        if (f32_stream) { gm.out_f32 = true; gm.add1 = S; gm.s1 = CP; gm.add1_f32 = true; }
        if (add2) { gm.add2 = add2; gm.s2 = CP; gm.add2scale = add2scale; }
        return tgemm(m, gm, s);
    };
    auto lnorm = [&](const Ln &l, const void *x, bool xf32, void *y, bool yf32) -> cudaError_t {
        if (collect) return cudaSuccess;
        return tlayernorm2(x, xf32, CP, y, yf32, CP, M, C, CP, l.g, l.b, l.eps, s);
    };
    auto attend = [&](const Attn &a, bool overlap) -> cudaError_t {
        if (collect) return cudaSuccess;
        TWinAttn t;
        t.qkv = QKV; t.sq = QS; t.qo = 0; t.ko = C; t.vo = 2 * C; t.H = h; t.W = w; t.heads = heads; t.hd = hd;
        t.wh = t.ww = ws; t.scale = a.scale; t.out = AT; t.so = CP;
        t.rtab = a.rtab; t.rt_w = a.rt_w; t.rt_offy = a.rt_offy; t.rt_offx = a.rt_offx;
        if (overlap) {
            t.kwh = t.kww = kws;
            t.kpad = kpad;
        } else if (a.shift) {
            t.shift_y = t.shift_x = a.shift;
            t.mask = true;
        }
        return twinattn(t, s);
    };
    auto mlp_step = [&](const Ln &n2, const Mlp &m) -> cudaError_t {
        CK(lnorm(n2, S, true, N1, false));
        pr.mark(P_LN);
        CK(gemm(*m.fc1, N1, CP, H1, HS, TAct::Gelu, false));
        pr.mark(P_FC1);
        CK(gemm(*m.fc2, H1, HS, S, CP, TAct::None, true));
        pr.mark(P_FC2);
        return cudaSuccess;
    };

    K(img_to_nhwc(in, in_f32, w, h, false, IN16, 16, s));
    K(tok_sub3(IN16, 16, M, mean[0], mean[1], mean[2], s));
    pr.mark(P_IO);
    CK(Gc(first, ga(IN16, 16, h, w, XF, CP, h, w)));
    CK(lnorm(pe, XF, false, S, true));
    pr.mark(P_CONV);
    for (Group &gr : groups) {
        K(cudaMemcpyAsync(GIN, S, (size_t)M * CP * 4, cudaMemcpyDeviceToDevice, s));
        for (Hab &b : gr.habs) {
            CK(lnorm(b.n1, S, true, N1, false));
            pr.mark(P_LN);
            // CAB on the normalised map (its last conv carries the 0.01), its channel gate
            GConvArgs c0 = ga(N1, CP, h, w, T60, crp, h, w);
            c0.act = GAct::Gelu;
            CK(Gc(b.cab0, c0));
            CK(Gc(b.cab2, ga(T60, crp, h, w, Y, CP, h, w)));
            pr.mark(P_CONV);
            K(cudaMemsetAsync(sums, 0, CP * 4, s));
            K(tcolsum(Y, CP, M, C, sums, s));
            K(chan_gate(sums, M, C, cs, b.ca1, b.cab1, b.ca2, b.cab2b, 1, 1.f / b.conv_scale, cm, CP, s));
            pr.mark(P_GATE);
            CK(gemm(*b.at.qkv, N1, CP, QKV, QS, TAct::None, false));
            pr.mark(P_QKV);
            CK(attend(b.at, false));
            pr.mark(P_WIN);
            CK(gemm(*b.at.proj, AT, CP, S, CP, TAct::None, true, Y, cm));  // S += proj(attn) + CAB * gate
            pr.mark(P_PROJ);
            CK(mlp_step(b.n2, b.mlp));
        }
        Ocab &o = gr.oc;
        CK(lnorm(o.n1, S, true, N1, false));
        pr.mark(P_LN);
        CK(gemm(*o.at.qkv, N1, CP, QKV, QS, TAct::None, false));
        pr.mark(P_QKV);
        CK(attend(o.at, true));
        pr.mark(P_WIN);
        CK(gemm(*o.at.proj, AT, CP, S, CP, TAct::None, true));
        pr.mark(P_PROJ);
        CK(mlp_step(o.n2, o.mlp));
        // the group conv plus the group input, in fp32
        K(tconvert(S, true, CP, S16, false, CP, M, CP, s));
        GConvArgs a = ga(S16, CP, h, w, S, CP, h, w);
        a.acc32 = true;
        a.out_f32 = true;
        a.r1f = GIN;
        a.r1f_stride = CP;
        CK(Gc(gr.conv, a));
        pr.mark(P_CONV);
    }
    CK(lnorm(fin, S, true, N1, false));
    GConvArgs a = ga(N1, CP, h, w, TT, CP, h, w);
    a.r1 = XF;
    a.r1_stride = CP;
    CK(Gc(after, a));
    GConvArgs b = ga(TT, CP, h, w, U0, 64, h, w);
    b.act = GAct::LRelu;
    b.slope = cbu_slope;
    CK(Gc(cbu, b));
    for (int sy = 0; sy < 2; sy++) {
        GConvArgs u = ga(U0, 64, h, w, V1, 64, h, w);
        u.R = 2;
        u.sy = sy;
        CK(Gc(up0[sy], u));
    }
    for (int sy = 0; sy < 2; sy++) {
        GConvArgs u = ga(V1, 64, 2 * h, 2 * w, V2, 64, 2 * h, 2 * w);
        u.R = 2;
        u.sy = sy;
        CK(Gc(up2[sy], u));
    }
    CK(Gc(last, ga(V2, 64, 4 * h, 4 * w, O16, 16, 4 * h, 4 * w)));
    pr.mark(P_UP);
    K(img_from_nhwc(O16, 16, 4 * w, 4 * h, out, out_f32, s));
    pr.mark(P_IO);
#undef K
#undef CK
    if (!collect) {
        char label[48];
        snprintf(label, sizeof(label), "hat %dx%d", w, h);
        pr.report(label);
    }
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<ImgNet> build_hat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    return Hat::build(g, max_w, max_h, s, why);
}

}  // namespace ajinative
