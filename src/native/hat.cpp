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
#include "native/hat_lifetimes.h"

namespace ajinative {
#ifdef AJI_TEST_ORDERED_HAT
cudaError_t test_ordered_hat_sums(const GConvArgs &args, int channels, int rows, cudaStream_t stream);
#endif
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

class Hat : public ImgNet, private HatBuffers {
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
    float *sums = nullptr, *cm = nullptr;
    StepWorkspaces<36> scratch;
    LifetimeCheck *checks = LifetimeCheck::active;
    std::string what;

    cudaError_t exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
                     std::vector<std::pair<GOp *, GConvArgs>> *collect, std::vector<int> *steps = nullptr);
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
    Lifetimes lt;
    f->declare(lt, (size_t)max_w * max_h, CP, f->crp, f->QS, f->HS, f->nup);
    if (!lt.assign(r, f->checks)) return refuse(why, "out of GPU memory"), nullptr;
    f->sums = (float *)r.alloc(CP * 4);
    f->cm = (float *)r.alloc(CP * 4);
    if (r.failed) return refuse(why, "out of GPU memory"), nullptr;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    std::vector<int> steps;
    size_t stock_limit = 0;
    // Visit the build shape last so preparation can use the same ops, with their final workspace addresses.
    for (bool small : {true, false}) {
        ops.clear();
        const int w = small ? std::min(max_w, 64) / f->ws * f->ws : max_w, h = small ? std::min(max_h, 64) / f->ws * f->ws : max_h;
        const auto err = f->exec(nullptr, false, nullptr, false, w, h, s, &ops, small ? nullptr : &steps);
        if (err != cudaSuccess) return refuse(why, cudaGetErrorString(err)), nullptr;
        stock_limit = std::max(stock_limit, gworkspace_max(ops));
    }
    for (size_t i = 0; i < ops.size(); ++i)
        f->scratch.need(steps[i], gworkspace_for_subtiles(*ops[i].first->w, ops[i].second, stock_limit));
    if (!f->scratch.place(lt, [&](size_t n) { return r.alloc(n); }, f->checks))
        return refuse(why, "out of GPU memory"), nullptr;
    if (f->checks && !f->checks->valid()) return refuse(why, f->checks->error), nullptr;
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto &slot = f->scratch.slots[steps[i]];
        ops[i].second.ws = slot.ptr;
        ops[i].second.ws_floats = slot.floats;
    }
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
                      std::vector<std::pair<GOp *, GConvArgs>> *collect, std::vector<int> *steps)
{
    const int M = w * h;
    StepProf pr(s, !collect);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return e_; } while (0)
#define K(x) do { if (!collect) CK(x); } while (0)
    auto enter = [&](int step, void **forwarded = nullptr) -> cudaError_t {
        return checks->enter(step, [&](void *p, size_t n) { return cudaMemsetAsync(p, 0xff, n, s) == cudaSuccess; },
                             forwarded) ? cudaSuccess : cudaErrorInvalidValue;
    };
#define STEP(n) do { if (checks && !collect) CK(enter(n)); } while (0)
    auto Gc = [&](int step, GOp &op, GConvArgs a) -> cudaError_t {
        if (collect) {
            collect->push_back({&op, a});
            if (steps) steps->push_back(step);
            return cudaSuccess;
        }
        const auto &slot = scratch.slots[step];
        a.ws = slot.ptr;
        a.ws_floats = slot.floats;
        if (checks) {
            CK(enter(step));
            if (!checks->workspace(step, a.ws, a.ws_floats * sizeof(float))) return cudaErrorInvalidValue;
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
        const int first_step = nxt ? 17 : 24;
        STEP(first_step);
        STEP(first_step + 1);
        STEP(first_step + 2);
        cudaError_t e = tmlp(*m.fc1, *m.fc2, t, s);
        if (e != cudaErrorNotSupported) {
            CK(e);
            *made = nxt != nullptr;
            pr.mark(P_MLP);
            return cudaSuccess;
        }
        t.lno = nullptr;
        STEP(first_step);
        CK(lnorm(n2, S, true, N1, false));
        pr.mark(P_LN);
        t.a = N1; t.ln_gamma = t.ln_beta = nullptr;
        e = tmlp(*m.fc1, *m.fc2, t, s);
        if (e != cudaErrorNotSupported) {
            CK(e);
            pr.mark(P_MLP);
            return cudaSuccess;
        }
        STEP(first_step + 1);
        CK(gemm(*m.fc1, N1, CP, H1, HS, TAct::Gelu, false));
        pr.mark(P_FC1);
        STEP(first_step + 2);
        CK(gemm(*m.fc2, H1, HS, S, CP, TAct::None, true));
        pr.mark(P_FC2);
        return cudaSuccess;
    };

    // Steps (the buffers' lifetimes in build()): 1 IN16, 2 XF, 3 S; per group 9 GIN; per HAB 10 N1,
    // 11 T60, 12 Y, 13 the gate, 14 QKV, 15 AT, 16 S, 17 N1, 18 H1, 19 S; the OCAB 20 N1, 21 QKV,
    // 22 AT, 23 S, 24 N1, 25 H1, 26 S; 27 S16, 28 S; then 30 N1, 31 TT, 32 U0, 33 V1, 34 V2, 35 O16, 36 out.
    STEP(1);
    K(img_to_nhwc(in, in_f32, w, h, false, IN16, 16, s));
    K(tok_sub3(IN16, 16, M, mean[0], mean[1], mean[2], s));
    pr.mark(P_IO);
    CK(Gc(2, first, ga(IN16, 16, h, w, XF, CP, h, w)));
    STEP(3);
    CK(lnorm(pe, XF, false, S, true));
    pr.mark(P_CONV);
    for (Group &gr : groups) {
        STEP(9);
        K(cudaMemcpyAsync(GIN, S, (size_t)M * CP * 4, cudaMemcpyDeviceToDevice, s));
        bool n1_made = false;  // N1 = LN1(S) already, from the previous block's MLP
        for (size_t bi = 0; bi < gr.habs.size(); bi++) {
            Hab &b = gr.habs[bi];
            if (checks && !collect) CK(enter(10, n1_made ? &N1 : nullptr));
            if (!n1_made) CK(lnorm(b.n1, S, true, N1, false));
            pr.mark(P_LN);
            // CAB on the normalised map (its last conv carries the 0.01), its channel gate
            GConvArgs c0 = ga(N1, CP, h, w, T60, crp, h, w);
            c0.act = GAct::Gelu;
            CK(Gc(11, b.cab0, c0));
            K(cudaMemsetAsync(sums, 0, CP * 4, s));
            GConvArgs c2 = ga(T60, crp, h, w, Y, CP, h, w);
            c2.sums = sums;  // the channel gate's sums from the conv's epilogue, where there's a kernel for it
            if (!gsupported(*b.cab2.w, c2)) c2.sums = nullptr;
            CK(Gc(12, b.cab2, c2));
#ifdef AJI_TEST_ORDERED_HAT
            if (!c2.sums) return cudaErrorNotSupported; // shipped HAT-L uses the CAB sums epilogue
            K(test_ordered_hat_sums(c2, CP, gvariant_rows(b.cab2.variant), s));
#else
            if (!c2.sums) K(tcolsum(Y, CP, M, C, sums, s));
#endif
            pr.mark(P_CONV);
            STEP(13);
            K(chan_gate(sums, M, C, cs, b.ca1, b.cab1, b.ca2, b.cab2b, 1, 1.f / b.conv_scale, cm, CP, s));
            pr.mark(P_GATE);
            STEP(14);
            CK(gemm(*b.at.qkv, N1, CP, QKV, QS, TAct::None, false));
            pr.mark(P_QKV);
            STEP(15);
            CK(attend(b.at, false));
            pr.mark(P_WIN);
            STEP(16);
            CK(gemm(*b.at.proj, AT, CP, S, CP, TAct::None, true, Y, cm));  // S += proj(attn) + CAB * gate
            pr.mark(P_PROJ);
            CK(mlp_step(b.n2, b.mlp, bi + 1 < gr.habs.size() ? &gr.habs[bi + 1].n1 : &gr.oc.n1, &n1_made));
        }
        Ocab &o = gr.oc;
        if (checks && !collect) CK(enter(20, n1_made ? &N1 : nullptr));
        if (!n1_made) CK(lnorm(o.n1, S, true, N1, false));
        pr.mark(P_LN);
        STEP(21);
        CK(gemm(*o.at.qkv, N1, CP, QKV, QS, TAct::None, false));
        pr.mark(P_QKV);
        STEP(22);
        CK(attend(o.at, true));
        pr.mark(P_WIN);
        STEP(23);
        CK(gemm(*o.at.proj, AT, CP, S, CP, TAct::None, true));
        pr.mark(P_PROJ);
        CK(mlp_step(o.n2, o.mlp, nullptr, &n1_made));
        // the group conv plus the group input, in fp32
        STEP(27);
        K(tconvert(S, true, CP, S16, false, CP, M, CP, s, kStreamScale));
        GConvArgs a = ga(S16, CP, h, w, S, CP, h, w);
        a.acc32 = true;
        a.out_f32 = true;
        a.r1f = GIN;
        a.r1f_stride = CP;
        CK(Gc(28, gr.conv, a));
        pr.mark(P_CONV);
    }
    STEP(30);
    CK(lnorm(fin, S, true, N1, false));
    GConvArgs a = ga(N1, CP, h, w, TT, CP, h, w);
    a.r1 = XF;
    a.r1_stride = CP;
    CK(Gc(31, after, a));
    GConvArgs b = ga(TT, CP, h, w, U0, 64, h, w);
    b.act = GAct::LRelu;
    b.slope = cbu_slope;
    CK(Gc(32, cbu, b));
    // the pixel-shuffle steps U0 -> V1 -> V2, then conv_last at the scale
    void *src = U0;
    for (int k = 0; k < nup; k++) {
        void *dst = k ? V2 : V1;
        const int sc = 1 << k;
        for (int sy = 0; sy < 2; sy++) {
            GConvArgs u = ga(src, 64, sc * h, sc * w, dst, 64, sc * h, sc * w);
            u.R = 2;
            u.sy = sy;
            CK(Gc(33 + k, (k ? up2 : up0)[sy], u));
        }
        src = dst;
    }
    const int S_ = 1 << nup;
    CK(Gc(35, last, ga(src, 64, S_ * h, S_ * w, O16, 16, S_ * h, S_ * w)));
    pr.mark(P_UP);
    STEP(36);
    K(img_from_nhwc(O16, 16, S_ * w, S_ * h, out, out_f32, s, clamp));
    pr.mark(P_IO);
#undef STEP
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
