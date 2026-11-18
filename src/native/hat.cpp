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
#include "native/hat_graph.h"
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
    int scale() const override { return 1 << nup; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes; }
    size_t weight_bytes() const override { return res.weight_bytes; }
    bool accepts(int w, int h) const override { return w > 0 && h > 0 && w <= maxw && h <= maxh && w % ws == 0 && h % ws == 0; }

private:
    DevRes res;
    int C = 0, CP = 0, QS = 0, HS = 0, heads = 0, hd = 0, ws = 16, kws = 24, kpad = 4, cr = 0, crp = 0, cs = 0, maxw = 0, maxh = 0;
    float mean[3] = {}, cbu_slope = 0.01f;
    bool clamp = false;
    int nup = 2;  // pixel-shuffle 2 steps
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

void up_ln(DevRes &r, const HatLn &h, int C, Ln *l)
{
    l->g = r.dfl(h.g, C);
    l->b = r.dfl(h.b, C);
    l->eps = h.eps;
}

bool up_attn(DevRes &r, const HatGraph &hg, const HatAttn &h, bool overlap, Attn *a, std::string *why)
{
    const int C = hg.C;
    a->qkv = r.mat(h.qkv.w, &h.qkv.b, C, 3 * C);
    a->proj = r.mat(h.proj.w, &h.proj.b, C, C);
    a->scale = h.scale;
    a->shift = h.shift;
    std::vector<float> tab;
    if (!tbias_table(h.bias, hg.heads, hg.ws, hg.ws, overlap ? hg.kws : 0, overlap ? hg.kws : 0, overlap ? hg.kpad : 0, &tab, &a->rt_w,
                     &a->rt_offy, &a->rt_offx))
        return refuse(why, "HAT bias is not a relative-position table");
    a->rtab = r.dfl(tab, tab.size());
    return true;
}

void up_mlp(DevRes &r, const HatGraph &hg, const HatLin &fc1, const HatLin &fc2, Mlp *m)
{
    m->fc1 = r.mat(fc1.w, &fc1.b, hg.C, hg.hid);
    m->fc2 = r.mat(fc2.w, &fc2.b, hg.hid, hg.C);
}

std::unique_ptr<ImgNet> Hat::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    HatGraph hg;
    if (!find_hat(rd, &hg, why)) return nullptr;
    std::unique_ptr<Hat> f(new Hat);
    DevRes &r = f->res;
    std::copy(hg.ir.mean, hg.ir.mean + 3, f->mean);
    f->clamp = hg.ir.clamp;
    f->nup = hg.nup;
    const int C = f->C = hg.C, CP = f->CP = pad16(C);
    f->heads = hg.heads;
    f->ws = hg.ws;
    f->kws = hg.kws;
    f->kpad = hg.kpad;
    f->hd = C / f->heads;
    if (f->hd > 32 || f->hd % 2 || f->ws * f->ws % 16 || f->ws * f->ws > 256 || f->kws * f->kws > 576) return refuse(why, "HAT head size / window"), nullptr;
    f->QS = pad16(3 * C);
    f->HS = pad16(hg.hid);
    f->cr = hg.cr;
    f->crp = pad16(f->cr);
    f->cs = hg.cs;
    const int nl = (int)hg.groups.size(), nb = hg.blocks();
    f->first.w = r.gup(conv_taps(hg.first, nullptr), hg.first.b, 3, 16, C, CP);
    up_ln(r, hg.pe, C, &f->pe);
    up_ln(r, hg.fin, C, &f->fin);
    for (const HatGroup &hgr : hg.groups) {
        Group gr;
        for (const HatHab &h : hgr.habs) {
            Hab hb;
            up_ln(r, h.n1, C, &hb.n1);
            up_ln(r, h.n2, C, &hb.n2);
            if (!up_attn(r, hg, h.at, false, &hb.at, why)) return nullptr;
            up_mlp(r, hg, h.fc1, h.fc2, &hb.mlp);
            hb.conv_scale = h.conv_scale;
            hb.cab0.w = r.gup(conv_taps(h.cab0, nullptr), h.cab0.b, C, CP, f->cr, f->crp);
            std::vector<float> sc(C, hb.conv_scale), b2 = h.cab2.b;
            for (float &x : b2) x *= hb.conv_scale;
            hb.cab2.w = r.gup(conv_taps(h.cab2, &sc), b2, f->cr, f->crp, C, CP);
            hb.ca1 = r.dfl(h.ca1, (size_t)f->cs * C);
            hb.cab1 = r.dfl(h.cab1, f->cs);
            hb.ca2 = r.dfl(h.ca2, (size_t)C * f->cs);
            hb.cab2b = r.dfl(h.cab2b, C);
            gr.habs.push_back(hb);
        }
        up_ln(r, hgr.oc.n1, C, &gr.oc.n1);
        up_ln(r, hgr.oc.n2, C, &gr.oc.n2);
        if (!up_attn(r, hg, hgr.oc.at, true, &gr.oc.at, why)) return nullptr;
        up_mlp(r, hg, hgr.oc.fc1, hgr.oc.fc2, &gr.oc.mlp);
        const std::vector<float> inv(C, 1.f / kStreamScale);  // its input is the stream's scaled fp16 copy
        gr.conv.w = r.gup(conv_taps(hgr.conv, &inv), hgr.conv.b, C, CP, C, CP);
        f->groups.push_back(std::move(gr));
    }
    // tail
    f->after.w = r.gup(conv_taps(hg.after, nullptr), hg.after.b, C, CP, C, CP);
    f->cbu_slope = hg.cbu.slope;
    f->cbu.w = r.gup(conv_taps(hg.cbu, nullptr), hg.cbu.b, C, CP, 64, 64);
    for (int k = 0; k < hg.nup; k++)
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(hg.ups[k], sy, 64, &wt, &bt);
            (k ? f->up2 : f->up0)[sy].w = r.gup(wt, bt, 64, 64, 128, 128);
        }
    f->last.w = r.gup(conv_taps(hg.last, nullptr), hg.last.b, 64, 64, 3, 16);
    if (r.failed) return refuse(why, "weight upload failed"), nullptr;

    // buffers for the largest tile
    f->maxw = max_w;
    f->maxh = max_h;
    const size_t M = (size_t)max_w * max_h;
    // The per-pixel buffers share one arena by lifetime: the steps of exec() each is live in (1-3 the
    // input conv, 9-28 a group (10-19 a HAB, 20-26 the OCAB), 30-36 the tail). Kept in step with exec().
    Lifetimes lt;
    lt.add(&f->IN16, M * 32, {{1, 2}});
    lt.add(&f->XF, M * CP * 2, {{2, 31}});
    lt.add((void **)&f->S, M * CP * 4, {{3, 30}});
    lt.add((void **)&f->GIN, M * CP * 4, {{9, 28}});
    lt.add(&f->N1, M * CP * 2, {{10, 14}, {17, 19}, {20, 21}, {24, 25}, {30, 31}});  // 19: the MLP makes the next LN1
    lt.add(&f->T60, M * f->crp * 2, {{11, 12}});
    lt.add(&f->Y, M * CP * 2, {{12, 16}});
    lt.add(&f->QKV, M * f->QS * 2, {{14, 15}, {21, 22}});
    lt.add(&f->AT, M * CP * 2, {{15, 16}, {22, 23}});
    lt.add(&f->H1, M * f->HS * 2, {{18, 19}, {25, 26}});
    lt.add(&f->S16, M * CP * 2, {{27, 28}});
    lt.add(&f->TT, M * CP * 2, {{31, 32}});
    // (conv_last reads the last of U0 / V1 / V2 at step 35)
    lt.add(&f->U0, M * 64 * 2, {{32, f->nup == 0 ? 35 : 33}});
    const size_t S2 = (size_t)1 << (2 * f->nup);  // output pixels per input pixel
    if (f->nup >= 1) lt.add(&f->V1, 4 * M * 64 * 2, {{33, f->nup == 1 ? 35 : 34}});
    if (f->nup >= 2) lt.add(&f->V2, 16 * M * 64 * 2, {{34, 35}});
    lt.add(&f->O16, S2 * M * 16 * 2, {{35, 36}});
    if (!lt.assign(r)) return refuse(why, "out of GPU memory"), nullptr;
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
    snprintf(buf, sizeof(buf), "HAT %dx, %d groups x %d blocks + OCAB, %d channels, %d heads, %dx%d windows (shifted), fp32 stream%s",
             f->scale(), nl, nb, C, f->heads, f->ws, f->ws, f->clamp ? ", clamped" : "");
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
    // nxt: the next block's LN1, which the fused MLP also applies to its output (into N1); *made: whether it did
    auto mlp_step = [&](const Ln &n2, const Mlp &m, const Ln *nxt, bool *made) -> cudaError_t {
        *made = false;
        if (collect) return cudaSuccess;
        // S += fc2(GELU(fc1(LN2 S))) in one kernel where there is one (else the LayerNorm, the MLP
        // kernel without it, or the two GEMMs)
        TMlp t;
        t.a = S; t.sa = CP; t.M = M; t.out = S; t.so = CP; t.out_f32 = true; t.add1 = S; t.s1 = CP; t.add1_f32 = true;
        t.ln_gamma = n2.g; t.ln_beta = n2.b; t.ln_eps = n2.eps;
        if (nxt) { t.lno = N1; t.slno = CP; t.lno_gamma = nxt->g; t.lno_beta = nxt->b; t.lno_eps = nxt->eps; }
        cudaError_t e = tmlp(*m.fc1, *m.fc2, t, s);
        if (e != cudaErrorNotSupported) {
            CK(e);
            *made = nxt != nullptr;
            pr.mark(P_MLP);
            return cudaSuccess;
        }
        t.lno = nullptr;
        CK(lnorm(n2, S, true, N1, false));
        pr.mark(P_LN);
        t.a = N1; t.ln_gamma = t.ln_beta = nullptr;
        e = tmlp(*m.fc1, *m.fc2, t, s);
        if (e != cudaErrorNotSupported) {
            CK(e);
            pr.mark(P_MLP);
            return cudaSuccess;
        }
        CK(gemm(*m.fc1, N1, CP, H1, HS, TAct::Gelu, false));
        pr.mark(P_FC1);
        CK(gemm(*m.fc2, H1, HS, S, CP, TAct::None, true));
        pr.mark(P_FC2);
        return cudaSuccess;
    };

    // Steps (the buffers' lifetimes in build()): 1 IN16, 2 XF, 3 S; per group 9 GIN; per HAB 10 N1,
    // 11 T60, 12 Y, 13 the gate, 14 QKV, 15 AT, 16 S, 17 N1, 18 H1, 19 S; the OCAB 20 N1, 21 QKV,
    // 22 AT, 23 S, 24 N1, 25 H1, 26 S; 27 S16, 28 S; then 30 N1, 31 TT, 32 U0, 33 V1, 34 V2, 35 O16, 36 out.
    K(img_to_nhwc(in, in_f32, w, h, false, IN16, 16, s));
    K(tok_sub3(IN16, 16, M, mean[0], mean[1], mean[2], s));
    pr.mark(P_IO);
    CK(Gc(first, ga(IN16, 16, h, w, XF, CP, h, w)));
    CK(lnorm(pe, XF, false, S, true));
    pr.mark(P_CONV);
    for (Group &gr : groups) {
        K(cudaMemcpyAsync(GIN, S, (size_t)M * CP * 4, cudaMemcpyDeviceToDevice, s));
        bool n1_made = false;  // N1 = LN1(S) already, from the previous block's MLP
        for (size_t bi = 0; bi < gr.habs.size(); bi++) {
            Hab &b = gr.habs[bi];
            if (!n1_made) CK(lnorm(b.n1, S, true, N1, false));
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
            CK(mlp_step(b.n2, b.mlp, bi + 1 < gr.habs.size() ? &gr.habs[bi + 1].n1 : &gr.oc.n1, &n1_made));
        }
        Ocab &o = gr.oc;
        if (!n1_made) CK(lnorm(o.n1, S, true, N1, false));
        pr.mark(P_LN);
        CK(gemm(*o.at.qkv, N1, CP, QKV, QS, TAct::None, false));
        pr.mark(P_QKV);
        CK(attend(o.at, true));
        pr.mark(P_WIN);
        CK(gemm(*o.at.proj, AT, CP, S, CP, TAct::None, true));
        pr.mark(P_PROJ);
        CK(mlp_step(o.n2, o.mlp, nullptr, &n1_made));
        // the group conv plus the group input, in fp32
        K(tconvert(S, true, CP, S16, false, CP, M, CP, s, kStreamScale));
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
    // the pixel-shuffle steps U0 -> V1 -> V2, then conv_last at the scale
    void *src = U0;
    for (int k = 0; k < nup; k++) {
        void *dst = k ? V2 : V1;
        const int sc = 1 << k;
        for (int sy = 0; sy < 2; sy++) {
            GConvArgs u = ga(src, 64, sc * h, sc * w, dst, 64, sc * h, sc * w);
            u.R = 2;
            u.sy = sy;
            CK(Gc((k ? up2 : up0)[sy], u));
        }
        src = dst;
    }
    const int S_ = 1 << nup;
    CK(Gc(last, ga(src, 64, S_ * h, S_ * w, O16, 16, S_ * h, S_ * w)));
    pr.mark(P_UP);
    K(img_from_nhwc(O16, 16, S_ * w, S_ * h, out, out_f32, s, clamp));
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
