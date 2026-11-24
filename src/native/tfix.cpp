// native/tfix.cpp — TemporalFix (pifroggi/vs_temporalfix, temporalfix_arch; 7 frames, same size) on native
// kernels.
//
// A small encoder gives every frame 4 feature channels (stride-2 convs, then bilinear x2). Per
// support frame (the 6 around the centre): three coarse-to-fine MCBlocks at 1/8, 1/4, 1/2 (two stride-2
// convs, 4 residual convs — plain at 1/8, depthwise + pointwise after — and an 11-channel head:
// flow delta, a confidence logit, 8 latent channels, upsampled x4) on [centre, warped support,
// features, dt, the previous logit and latent, flow]; the full-resolution warp of the support and
// its validity; the confidence head at 1/2 on the low-passed full-resolution results; and the
// conservative temporal average of the confidence-gated supports. Downsizing resizes low-pass first
// (5x5 binomial), as the model does. The export's last op nudges output pixel (0, 0) by at most
// 2^-11 (a tanh of the warped frames' mean, keeping them alive in the graph); it is not reproduced.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "native/netres.h"
#include "native/tfix_kernels.h"
#include "native/vidnet.h"

namespace ajinative {
namespace {

using namespace onnxread;

constexpr int NF = 7, CEN = 3;
const int kSupports[6] = {0, 1, 2, 4, 5, 6};
const float kDt[6] = {-1.f, -2.f / 3.f, -1.f / 3.f, 1.f / 3.f, 2.f / 3.f, 1.f};

struct Stage {
    int c = 0, ca = 0, h = 0, w = 0, ha = 0, wa = 0, hq = 0, wq = 0, xs = 0;  // channels, sizes, x stride
    bool ds = false;                                                       // depthwise residual convs
    GOp c00, c01, res[4], out;
    float *dw[4] = {}, *dwb[4] = {};
    TMat *pw[4] = {};
    void *pyr[NF] = {};
};

// One support's scratch and stream: the supports run as concurrent chains (their small kernels
// overlap), the shared accumulation in support order.
struct Chain {
    struct Sb {
        void *X = nullptr, *A = nullptr, *B = nullptr, *B2 = nullptr, *T = nullptr, *O = nullptr, *ST = nullptr;
        float *F = nullptr;
    } sb[3];
    void *WH = nullptr, *RCH = nullptr, *CIN = nullptr, *C1 = nullptr, *C2 = nullptr, *C3 = nullptr;
    float *FH = nullptr, *wsp = nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t done = nullptr;
};

class Tfix : public VidNet {
public:
    static std::unique_ptr<VidNet> build(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why);
    ~Tfix() override
    {
        for (Replay *r : {&full, &win})
            if (r->g) cudaGraphExecDestroy(r->g);
        for (Replay &r : enc)
            if (r.g) cudaGraphExecDestroy(r.g);
        for (Chain &c : ch) {
            if (c.st) cudaStreamDestroy(c.st);
            if (c.done) cudaEventDestroy(c.done);
        }
        if (ev) cudaEventDestroy(ev);
    }
    // The whole window from the model input: every frame encoded into the slot of its position.
    cudaError_t run(const void *in, void *out, cudaStream_t s) override
    {
        const void *fr[NF];
        for (int i = 0; i < NF; i++) fr[i] = (const uint16_t *)in + (size_t)i * 3 * W * H;
        return replay(full, in, out, s, [&] { return exec(fr, dsel, out, s, nullptr); });
    }
    int cache_entries() const override { return NF; }
    cudaError_t encode(int entry, const void *frame, cudaStream_t s) override
    {
        if (entry < 0 || entry >= NF) return cudaErrorInvalidValue;
        const void *fr[NF] = {};
        fr[entry] = frame;
        return replay(enc[entry], frame, nullptr, s, [&] { return exec(fr, nullptr, nullptr, s, nullptr); });
    }
    cudaError_t run_cached(const int *sel, void *out, cudaStream_t s) override
    {
        const void *fr[NF] = {};
        return replay(win, sel, out, s, [&] { return exec(fr, sel, out, s, nullptr); });
    }
    int frames() const override { return NF; }
    int scale() const override { return 1; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes; }

private:
    DevRes res;
    int W = 0, H = 0, h1 = 0, w1 = 0;
    float thresh = 0.6f, gslope = 12.f, minsup = 1.f, cslope = 4.f, eps = 1e-6f;
    GOp e0, e1, e2, ep, h0, hh1, h2c;
    Stage st[3];
    std::vector<Chain> ch;
    cudaEvent_t ev = nullptr;  // the encoder done
    // Per frame (one slot each, frb / Stage pyb bytes apart): the 8-channel full-resolution map and
    // its pyramid. run() fills slot i from window position i; encode() fills one slot.
    void *FR[NF] = {}, *E1 = nullptr, *E2 = nullptr, *EP = nullptr;
    size_t frb = 0, pyb[3] = {};
    int *dsel = nullptr;  // 0..NF-1: run()'s window, position i in slot i
    float *wsp = nullptr;
    // Per support at half resolution, what the blend reads: the last stage's flow (PF, fp32 [h1][w1][2])
    // and (raw logit, confidence logit) pairs (PL, fp16 [h1][w1][2]).
    float *PF = nullptr;
    void *PL = nullptr;
    // A window is ~400 small launches, a frame's encoding ~10: each captured once per (key, out) as a
    // CUDA graph and replayed (AJI_NATIVE_PROFILE runs them directly, for the per-phase times).
    struct Replay {
        cudaGraphExec_t g = nullptr;
        const void *key = nullptr;
        void *out = nullptr;
    } full, win, enc[NF];
    size_t ws_floats = 0;
    std::string what;

    template <class Fn>
    static cudaError_t replay(Replay &r, const void *key, void *out, cudaStream_t s, Fn fn)
    {
        static const bool prof = std::getenv("AJI_NATIVE_PROFILE") != nullptr;
        if (prof) return fn();
        if (!r.g || key != r.key || out != r.out) {
            if (r.g) cudaGraphExecDestroy(r.g);
            r.g = nullptr;
            cudaGraph_t g = nullptr;
            cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) return fn();
            const cudaError_t er = fn();
            e = cudaStreamEndCapture(s, &g);
            if (er != cudaSuccess || e != cudaSuccess) {
                if (g) cudaGraphDestroy(g);
                return er != cudaSuccess ? er : e;
            }
            e = cudaGraphInstantiate(&r.g, g, 0);
            cudaGraphDestroy(g);
            if (e != cudaSuccess) {
                r.g = nullptr;
                return fn();
            }
            r.key = key;
            r.out = out;
        }
        return cudaGraphLaunch(r.g, s);
    }

    // Encodes frame fr[i] into slot i for each non-null fr[i]; then, with sel, runs the window whose
    // position j reads slot sel[j] (sel on the device) into out.
    cudaError_t exec(const void *const *fr, const int *sel, void *out, cudaStream_t s, std::vector<std::pair<GOp *, GConvArgs>> *collect);
};

TxMap tm(void *p, int stride, int off = 0, bool f32 = false)
{
    TxMap m;
    m.p = p;
    m.f32 = f32;
    m.stride = stride;
    m.off = off;
    return m;
}

bool cvt(const Reader &rd, const std::string &name, Conv *c, bool lrelu, int stride, std::string *why)
{
    if (!rd.conv(name, c, why)) return false;
    if (c->lrelu != lrelu || c->stride != stride) return refuse(why, name + ": unexpected activation or stride");
    return true;
}

std::unique_ptr<VidNet> Tfix::build(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    if (!rd.node("/encode/cnn0/Conv") || !rd.node("/aligner/blocks.0/out_conv/Conv") || !rd.node("/conf_head/net/net.2/Conv"))
        return refuse(why, "not a TemporalFix graph"), nullptr;
    if (w % 8 || h % 8 || w < 64 || h < 64) return refuse(why, "TemporalFix needs a frame size that is a multiple of 8"), nullptr;
    std::unique_ptr<Tfix> f(new Tfix);
    DevRes &r = f->res;
    f->W = w;
    f->H = h;
    f->h1 = (h + 1) / 2;
    f->w1 = (w + 1) / 2;
    Conv c;
    auto G = [&](GOp &op, const Conv &cc, int cin_pad, int cout_pad, const std::vector<float> *scale = nullptr) {
        std::vector<float> b = cc.b;
        if (scale)
            for (size_t i = 0; i < b.size(); i++) b[i] *= (*scale)[i];
        op.w = r.gup(conv_taps(cc, scale), b, cc.cin, cin_pad, cc.cout, cout_pad);
    };
    if (!cvt(rd, "/encode/cnn0/Conv", &c, true, 2, why) || c.cin != 3 || c.cout != 16) return nullptr;
    G(f->e0, c, 16, 16);
    if (!cvt(rd, "/encode/cnn1/Conv", &c, true, 1, why)) return nullptr;
    G(f->e1, c, 16, 16);
    if (!cvt(rd, "/encode/cnn2/Conv", &c, true, 1, why)) return nullptr;
    G(f->e2, c, 16, 16);
    if (!cvt(rd, "/encode/proj/Conv", &c, false, 1, why) || c.cout != 4) return refuse(why, "TemporalFix encoder"), nullptr;
    G(f->ep, c, 16, 16);
    const int scales[3] = {8, 4, 2};
    for (int k = 0; k < 3; k++) {
        Stage &S = f->st[k];
        const std::string B = "/aligner/blocks." + std::to_string(k) + "/";
        S.h = (h + scales[k] - 1) / scales[k];
        S.w = (w + scales[k] - 1) / scales[k];
        S.ha = (S.h + 1) / 2;
        S.wa = (S.w + 1) / 2;
        S.hq = (S.ha + 1) / 2;
        S.wq = (S.wa + 1) / 2;
        S.xs = k ? 32 : 16;
        if (!cvt(rd, B + "conv0/conv0.0/conv0.0.0/Conv", &c, true, 2, why) || c.cin != (k ? 26 : 15)) return refuse(why, B + "conv0"), nullptr;
        S.ca = c.cout;
        G(S.c00, c, S.xs, pad16(S.ca));
        if (!cvt(rd, B + "conv0/conv0.1/conv0.1.0/Conv", &c, true, 2, why)) return nullptr;
        S.c = c.cout;
        if (S.c % 16) return refuse(why, B + "channels"), nullptr;
        G(S.c01, c, pad16(S.ca), S.c);
        S.ds = rd.node(B + "convblock/convblock.0/dw/Conv") != nullptr;
        for (int j = 0; j < 4; j++) {
            const std::string R = B + "convblock/convblock." + std::to_string(j) + "/";
            std::vector<float> beta, v, bb;
            std::vector<int64_t> d;
            if (!init_input(rd, rd.node(R + "Mul"), &beta, &d) || (int)beta.size() != S.c) return refuse(why, R + "beta"), nullptr;
            if (!rd.node(R + "relu/LeakyRelu")) return refuse(why, R + "activation"), nullptr;
            if (!S.ds) {
                if (!cvt(rd, R + "conv/Conv", &c, false, 1, why)) return nullptr;
                G(S.res[j], c, S.c, S.c, &beta);
            } else {
                const OnnxNode *dn = rd.node(R + "dw/Conv"), *pn = rd.node(R + "pw/Conv");
                if (!tensor_floats(rd, dn, 1, &v, &d) || d != std::vector<int64_t>{S.c, 1, 3, 3} || dn->in.size() < 3 || !rd.floats(dn->in[2], &bb))
                    return refuse(why, R + "dw"), nullptr;
                S.dw[j] = r.dfl(v, (size_t)9 * S.c);
                S.dwb[j] = r.dfl(bb, S.c);
                std::vector<float> wm, bm;
                int co = 0;
                if (!conv1x1_matrix(rd, R + "pw/Conv", S.c, nullptr, &wm, &bm, &co, why) || co != S.c || (int)bm.size() != S.c)
                    return refuse(why, R + "pw"), nullptr;
                for (int kk = 0; kk < S.c; kk++)
                    for (int o = 0; o < S.c; o++) wm[(size_t)kk * S.c + o] *= beta[o];
                for (int o = 0; o < S.c; o++) bm[o] *= beta[o];
                S.pw[j] = r.mat(wm, &bm, S.c, S.c);
            }
        }
        if (!cvt(rd, B + "out_conv/Conv", &c, false, 1, why) || c.cout != 11) return refuse(why, B + "out_conv"), nullptr;
        G(S.out, c, S.c, 16);
    }
    if (f->st[2].h != f->h1 || f->st[2].w != f->w1) return refuse(why, "TemporalFix stage sizes"), nullptr;
    if (!cvt(rd, "/conf_head/net/net.0/net.0.0/Conv", &c, true, 1, why) || c.cin != 25) return refuse(why, "TemporalFix conf head"), nullptr;
    G(f->h0, c, 32, 32);
    if (!cvt(rd, "/conf_head/net/net.1/net.1.0/Conv", &c, true, 1, why)) return nullptr;
    G(f->hh1, c, 32, 32);
    if (!cvt(rd, "/conf_head/net/net.2/Conv", &c, false, 1, why) || c.cout != 1) return refuse(why, "TemporalFix conf head"), nullptr;
    G(f->h2c, c, 32, 16);
    // the blend's constants, where the export names them
    std::vector<float> v;
    if (init_input(rd, rd.node("/Sub_6"), &v) && v.size() == 1) f->thresh = v[0];
    if (init_input(rd, rd.node("/Mul_16"), &v) && v.size() == 1) f->gslope = v[0];
    if (init_input(rd, rd.node("/Mul_19"), &v) && v.size() == 1) f->cslope = v[0];
    if (r.failed) return refuse(why, "weight upload failed"), nullptr;

    // buffers
    const size_t M = (size_t)w * h, M1 = (size_t)f->h1 * f->w1;
    f->frb = M * 8 * 2;
    f->FR[0] = r.alloc(NF * f->frb);
    for (int i = 1; i < NF; i++) f->FR[i] = f->FR[0] ? (char *)f->FR[0] + i * f->frb : nullptr;
    f->E1 = r.alloc(M1 * 16 * 2);
    f->E2 = r.alloc(M1 * 16 * 2);
    f->EP = r.alloc(M1 * 16 * 2);
    for (int k = 0; k < 3; k++) {
        Stage &S = f->st[k];
        f->pyb[k] = (size_t)S.h * S.w * 8 * 2;
        S.pyr[0] = r.alloc(NF * f->pyb[k]);
        for (int i = 1; i < NF; i++) S.pyr[i] = S.pyr[0] ? (char *)S.pyr[0] + i * f->pyb[k] : nullptr;
    }
    f->dsel = (int *)r.alloc(NF * sizeof(int));
    // Concurrent chains (their small kernels overlap). RTX 5090, frame ms by chains 1 / 2 / 3 / 6: DVD 1.73 /
    // 1.31 / 1.15 / 0.99, 720p 2.30 / 1.92 / 1.74 / 1.69, 1080p 3.92 / 3.59 / 3.46 / 3.38. A chain is ~92 MB at
    // 1080p, so 3 there take what 2 did before the blend stopped needing full-resolution buffers.
    int nch = M <= (size_t)500000 ? 6 : M <= (size_t)2200000 ? 3 : 2;
    if (const char *e = std::getenv("AJI_NATIVE_TFIX_CHAINS")) nch = std::max(1, std::min(6, std::atoi(e)));  // for benchmarks
    f->ch.resize(nch);
    // A chain's scratch is one block in two overlapping layouts, as a support runs on its stream: the
    // aligner stages (A), then from the full-resolution pass on the confidence head (B). The last stage's
    // flow and logit, which B's first pass reads, are the support's own (PF, PL), outside the block. In B,
    // C2 takes CIN's place (dead after the head's first conv) and C3 C1's (dead after the second).
    // Every buffer is fully written before it is read, so none relies on the block's zeroing.
    auto bump = [](size_t &off, size_t n) {
        const size_t at = off;
        off += (n + 255) & ~(size_t)255;
        return at;
    };
    size_t oA = 0, at[3][8] = {};
    for (int k = 0; k < 3; k++) {
        const Stage &S = f->st[k];
        const size_t Ms = (size_t)S.h * S.w, Ma = (size_t)S.ha * S.wa, Mq = (size_t)S.hq * S.wq;
        at[k][0] = bump(oA, Ms * S.xs * 2);
        at[k][1] = bump(oA, Ma * pad16(S.ca) * 2);
        at[k][2] = bump(oA, Mq * S.c * 2);
        at[k][3] = bump(oA, Mq * S.c * 2);
        at[k][4] = bump(oA, Mq * S.c * 2);
        at[k][5] = bump(oA, Mq * 16 * 2);
        if (k < 2) {  // the flow and state the next stage reads
            at[k][6] = bump(oA, Ms * 16 * 2);
            at[k][7] = bump(oA, Ms * 8);
        }
    }
    size_t oB = 0;
    const size_t pFH = bump(oB, M1 * 8), pWH = bump(oB, M1 * 8 * 2), pRCH = bump(oB, M1 * 16 * 2);
    const size_t pCIN = bump(oB, M1 * 32 * 2), pC1 = bump(oB, M1 * 32 * 2);
    const size_t chain_bytes = std::max(oA, oB);
    for (Chain &c : f->ch) {
        char *blk = (char *)r.alloc(chain_bytes);
        if (!blk) return refuse(why, "out of GPU memory"), nullptr;
        for (int k = 0; k < 3; k++) {
            Chain::Sb &b = c.sb[k];
            b.X = blk + at[k][0];
            b.A = blk + at[k][1];
            b.B = blk + at[k][2];
            b.B2 = blk + at[k][3];
            b.T = blk + at[k][4];
            b.O = blk + at[k][5];
            b.ST = k < 2 ? blk + at[k][6] : nullptr;
            b.F = k < 2 ? (float *)(blk + at[k][7]) : nullptr;
        }
        c.FH = (float *)(blk + pFH);
        c.WH = blk + pWH;
        c.RCH = blk + pRCH;
        c.CIN = blk + pCIN;
        c.C1 = blk + pC1;
        c.C2 = c.CIN;
        c.C3 = c.C1;
        if (cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&c.done, cudaEventDisableTiming) != cudaSuccess)
            return refuse(why, "cannot create a CUDA stream"), nullptr;
    }
    if (cudaEventCreateWithFlags(&f->ev, cudaEventDisableTiming) != cudaSuccess) return refuse(why, "cannot create a CUDA event"), nullptr;
    f->PF = (float *)r.alloc((NF - 1) * M1 * 8);
    f->PL = r.alloc((NF - 1) * M1 * 4);
    if (r.failed) return refuse(why, "out of GPU memory"), nullptr;
    {
        int id[NF];
        for (int i = 0; i < NF; i++) id[i] = i;
        if (cudaMemcpy(f->dsel, id, sizeof(id), cudaMemcpyHostToDevice) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    }
    void *zin = nullptr;
    if (cudaMalloc(&zin, (size_t)3 * NF * M * 2) != cudaSuccess) return refuse(why, "out of GPU memory"), nullptr;
    cudaMemset(zin, 0, (size_t)3 * NF * M * 2);
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    const void *zf[NF];
    for (int i = 0; i < NF; i++) zf[i] = (const uint16_t *)zin + (size_t)i * 3 * M;
    f->exec(zf, f->dsel, nullptr, s, &ops);
    f->ws_floats = gworkspace_max(ops);
    bool ok = true;
    if (f->ws_floats) {  // one split-K workspace per stream
        ok = (f->wsp = (float *)r.alloc(f->ws_floats * 4)) != nullptr;
        for (Chain &c : f->ch) ok = ok && (c.wsp = (float *)r.alloc(f->ws_floats * 4)) != nullptr;
    }
    if (ok) {
        ops.clear();
        f->exec(zf, f->dsel, nullptr, s, &ops);
        ok = gprepare(ops, s, why);
    } else {
        refuse(why, "out of GPU memory");
    }
    cudaStreamSynchronize(s);
    cudaFree(zin);
    if (!ok) return nullptr;
    if (cudaGetLastError() != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    char buf[160];
    snprintf(buf, sizeof(buf), "TemporalFix, 7 frames, stages %d/%d/%d channels, %d streams", f->st[0].c, f->st[1].c, f->st[2].c, nch);
    f->what = buf;
    return std::unique_ptr<VidNet>(f.release());
}

cudaError_t Tfix::exec(const void *const *fr_in, const int *sel, void *out, cudaStream_t s, std::vector<std::pair<GOp *, GConvArgs>> *collect)
{
    const size_t plane = (size_t)W * H;
    StepProf pr(s, !collect);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return e_; } while (0)
#define K(x) do { if (!collect) CK(x); } while (0)
    auto Gc = [&](GOp &op, GConvArgs a, cudaStream_t q, float *ws) -> cudaError_t {
        a.ws = ws;
        a.ws_floats = ws_floats;
        if (collect) {
            collect->push_back({&op, a});
            return cudaSuccess;
        }
        return glaunch(*op.w, a, op.variant, q);
    };
    auto lrelu = [](GConvArgs a) {
        a.act = GAct::LRelu;
        a.slope = 0.2f;
        return a;
    };
    auto s2 = [](GConvArgs a) {
        a.stride = 2;
        return a;
    };
    // window position j's frame map and pyramid level k: the slot sel[j] names
    auto frm = [&](int j, int off) {
        TxMap m = tm(FR[0], 8, off);
        m.sel = sel;
        m.pos = j;
        m.fbytes = (long long)frb;
        return m;
    };
    auto pym = [&](int k, int j, int off) {
        TxMap m = tm(st[k].pyr[0], 8, off);
        m.sel = sel;
        m.pos = j;
        m.fbytes = (long long)pyb[k];
        return m;
    };
    // the encoder of frame i: rgb into FR channels 0..2, features into 3..6, then its pyramid
    for (int i = 0; i < NF; i++) {
        const void *fp = fr_in[i];
        if (!fp) continue;
        GConvArgs a = s2(lrelu(ga(fp, (int)plane, H, W, E1, 16, h1, w1)));
        a.nchw = true;
        CK(Gc(e0, a, s, wsp));
        CK(Gc(e1, lrelu(ga(E1, 16, h1, w1, E2, 16, h1, w1)), s, wsp));
        CK(Gc(e2, lrelu(ga(E2, 16, h1, w1, E1, 16, h1, w1)), s, wsp));
        CK(Gc(ep, ga(E1, 16, h1, w1, EP, 16, h1, w1), s, wsp));
        K(tx_frame(fp, EP, 16, h1, w1, FR[i], H, W, s));
        for (Stage &S : st) K(tx_down(tm(FR[i], 8), H, W, tm(S.pyr[i], 8), S.h, S.w, 7, 1.f, 1.f, s));
    }
    pr.mark(P_ENC);
    if (!sel) {
        if (!collect) pr.report("temporalfix encode");
        return cudaSuccess;
    }
    pr.mark(P_ALIGN);
    K(cudaEventRecord(ev, s));
    for (Chain &c : ch) K(cudaStreamWaitEvent(c.st, ev, 0));
    auto mcblock = [&](const Stage &Sc, Stage &S, Chain::Sb &b, cudaStream_t q, float *ws) -> cudaError_t {
        (void)Sc;
        CK(Gc(S.c00, s2(lrelu(ga(b.X, S.xs, S.h, S.w, b.A, pad16(S.ca), S.ha, S.wa))), q, ws));
        CK(Gc(S.c01, s2(lrelu(ga(b.A, pad16(S.ca), S.ha, S.wa, b.B, S.c, S.hq, S.wq))), q, ws));
        void *x = b.B, *y = b.B2;
        for (int j = 0; j < 4; j++) {
            if (!S.ds) {
                GConvArgs a = lrelu(ga(x, S.c, S.hq, S.wq, y, S.c, S.hq, S.wq));
                a.r1 = x;
                a.r1_stride = S.c;
                CK(Gc(S.res[j], a, q, ws));
            } else if (!collect) {
                CK(tdwconv2(x, S.c, b.T, S.c, S.hq, S.wq, S.c, S.dw[j], S.dwb[j], false, nullptr, 0, nullptr, q));
                TGemm gm;
                gm.a = b.T; gm.sa = S.c; gm.M = S.hq * S.wq; gm.out = y; gm.so = S.c;
                gm.add1 = x; gm.s1 = S.c; gm.act = TAct::LRelu; gm.slope = 0.2f; gm.act_last = true;
                CK(tgemm(*S.pw[j], gm, q));
            }
            std::swap(x, y);
        }
        CK(Gc(S.out, ga(x, S.c, S.hq, S.wq, b.O, 16, S.hq, S.wq), q, ws));
        return cudaSuccess;
    };
    for (int si = 0; si < 6; si++) {
        const int fr = kSupports[si];
        const float dt = kDt[si];
        Chain &c = ch[si % ch.size()];
        const cudaStream_t q = c.st;
        float *const pf = PF + (size_t)si * h1 * w1 * 2;   // the support's last-stage flow,
        void *const pl = (char *)PL + (size_t)si * h1 * w1 * 4;  // raw and confidence logits
        for (int k = 0; k < 3; k++) {
            Stage &S = st[k];
            Chain::Sb &b = c.sb[k];
            // the whole MCBlock input in one pass: centre, the support (warped from the second stage on),
            // dt, and the previous stage's flow and state upsampled
            const Stage *P = k ? &st[k - 1] : nullptr;
            const Chain::Sb *pb = k ? &c.sb[k - 1] : nullptr;
            float *const F = k < 2 ? b.F : pf;
            K(tx_assemble(pym(k, CEN, 0), pym(k, fr, 0), k ? pb->F : nullptr, k ? pb->ST : nullptr, k ? P->h : 0, k ? P->w : 0, F, b.X,
                          S.xs, S.h, S.w, dt, q));
            CK(mcblock(S, S, b, q, c.wsp));
            if (k < 2)
                K(tx_head_up(b.O, S.hq, S.wq, F, b.ST, S.h, S.w, k > 0, q));
            else
                K(tx_head_up_last(b.O, S.hq, S.wq, F, pl, S.h, S.w, q));
        }
        // full resolution (flow, logit, the warped support and its validity), low-passed to 1/2 for the
        // confidence head in the same pass
        const Stage &L = st[2];
        K(tx_full_down(pf, pl, 2, L.h, L.w, frm(fr, 0), c.FH, c.WH, H, W, h1, w1, q));  // WH: warped 0..2, raw logit 3
        K(tx_conf_in_warp(pym(2, CEN, 0), tm(c.WH, 8, 0), pym(2, fr, 0), c.FH, dt, c.CIN, c.RCH, h1, w1, q));
        CK(Gc(h0, lrelu(ga(c.CIN, 32, h1, w1, c.C1, 32, h1, w1)), q, c.wsp));
        CK(Gc(hh1, lrelu(ga(c.C1, 32, h1, w1, c.C2, 32, h1, w1)), q, c.wsp));
        GConvArgs a = ga(c.C2, 32, h1, w1, c.C3, 16, h1, w1);
        a.r1 = c.RCH;
        a.r1_stride = 16;
        CK(Gc(h2c, a, q, c.wsp));
        K(tx_pick(c.C3, 16, pl, h1 * w1, q));  // the confidence logit next to the raw one
    }
    for (Chain &c : ch) {
        K(cudaEventRecord(c.done, c.st));
        K(cudaStreamWaitEvent(s, c.done, 0));
    }
    pr.mark(P_FUSE);
    // every support's confidence-gated accumulation and the blend, per pixel
    TxBlend bl;
    bl.pf = PF;
    bl.pl = PL;
    for (int si = 0; si < 6; si++) bl.fr[si] = frm(kSupports[si], 0);
    bl.center = frm(CEN, 0);
    bl.thresh = thresh;
    bl.gslope = gslope;
    bl.minsup = minsup;
    bl.cslope = cslope;
    bl.eps = eps;
    K(tx_blend(bl, h1, w1, H, W, out, s));
    pr.mark(P_IO);
#undef K
#undef CK
    if (!collect) {
        char label[48];
        snprintf(label, sizeof(label), "temporalfix %dx%d", W, H);
        pr.report(label);
    }
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<VidNet> build_tfix(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why)
{
    return Tfix::build(g, w, h, s, why);
}

}  // namespace ajinative
