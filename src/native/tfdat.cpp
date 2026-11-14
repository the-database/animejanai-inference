// native/tfdat.cpp — TFDAT v2 (traiNNer-redux dis2-tfdat2 tfdat_arch; bndl animefilm is 2x, 9 frames) on
// native kernels, at 1x, 2x and 4x: the frames are pixel-unshuffled by U = 4, 2 or 1 before conv_first,
// so the features and the body run at H/U x W/U and the 4x upsampler brings them to the output size.
//
// Per frame a 3-level encoder (stride-2 conv + conv, 32/64/96 channels). Per neighbour of the centre
// frame: a coarse-to-fine flow (dec3 on [enc3 nb, enc3 ctr]; at each finer level the flow is
// upsampled x2, warps the neighbour's encoder features, and [warped, enc ctr, flow] predicts a
// delta), the full-resolution warp of the neighbour frame and an occlusion estimate (mean |warped -
// centre|, U x U average), conv_first on the pixel-unshuffled warped frame, align_refine ([nb, ctr]
// 1x1 + 3x3 + nb), and a confidence (embedding similarity and occlusion -> two 3x3 convs ->
// sigmoid) that weights it. The fusion (a 1x1 over the 9 slots, accumulated slot by slot, a 3x3,
// an SE gate) feeds the FDAT body (fdat_body.h); then conv_after plus the fused features, two
// conv + pixel-shuffle stages, a conv to RGB plus the bilinearly upscaled centre frame, and
// hr_refine (two convs plus its input).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "native/fdat_body.h"
#include "native/fdat_kernels.h"
#include "native/img_kernels.h"
#include "native/tfdat_kernels.h"
#include "native/vidnet.h"

namespace ajinative {
namespace {

using namespace onnxread;

class Tfdat : public VidNet {
public:
    static std::unique_ptr<VidNet> build(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why);
    ~Tfdat() override
    {
        if (graph) cudaGraphExecDestroy(graph);
    }
    // A run is several hundred launches, many of them small: captured once per (in, out) as a CUDA
    // graph and replayed (AJI_NATIVE_PROFILE runs it directly, for the per-phase times).
    cudaError_t run(const void *in, void *out, cudaStream_t s) override
    {
        static const bool prof = std::getenv("AJI_NATIVE_PROFILE") != nullptr;
        if (prof) return exec(in, out, s, nullptr);
        if (!graph || in != g_in || out != g_out) {
            if (graph) cudaGraphExecDestroy(graph);
            graph = nullptr;
            cudaGraph_t g = nullptr;
            cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) return exec(in, out, s, nullptr);
            const cudaError_t er = exec(in, out, s, nullptr);
            e = cudaStreamEndCapture(s, &g);
            if (er != cudaSuccess || e != cudaSuccess) {
                if (g) cudaGraphDestroy(g);
                return er != cudaSuccess ? er : e;
            }
            e = cudaGraphInstantiate(&graph, g, 0);
            cudaGraphDestroy(g);
            if (e != cudaSuccess) {
                graph = nullptr;
                return exec(in, out, s, nullptr);
            }
            g_in = in;
            g_out = out;
        }
        return cudaGraphLaunch(graph, s);
    }
    int frames() const override { return nf; }
    int scale() const override { return SC; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes + body.device_bytes(); }

private:
    DevRes res;
    FdatBody body;
    int nf = 0, C = 0, CP = 0, W = 0, H = 0;
    int U = 2, SC = 2, UC = 16;  // unshuffle, scale, unshuffled frame stride
    float sim_scale = 1.f;
    // gconv steps
    GOp enc[6], dec3[3], dec2[3], dec1[3], cf0, cf2, ar2, conf0, conf2, fuse2, after, up0, up2[2], up4[2], up6, hr0, hr2;
    TMat *ar0 = nullptr, *embc = nullptr, *embn = nullptr;
    std::vector<TMat *> fuse;  // per slot (centre first); the last carries the bias
    float *se1 = nullptr, *se2 = nullptr;
    int ser = 0;
    // buffers
    static constexpr int CS1 = 80, CS2 = 144, CS3 = 192, CAT = 256;
    void *E1a, *E1N, *C1, *E2a, *E2N, *C2, *E3a, *C3, *D3a, *D3b, *D3, *D2a, *D2b, *D2, *D1a, *D1b, *D1;
    float *F3, *U2, *F2, *U1, *F1, *sums, *cm, *wsp = nullptr;
    void *US, *WUS, *CONF, *CF1, *CF2, *T1, *CATB, *AR1, *AL, *EC, *EN, *SLOT, *ACC, *FU, *X0, *TT;
    void *U0, *V1, *V2, *RZ, *S, *HR, *O16;
    size_t ws_floats = 0;
    std::string what;
    cudaGraphExec_t graph = nullptr;
    const void *g_in = nullptr, *g_out = nullptr;

    // One run; with collect, the gconv steps are gathered into *collect instead of launched (and the
    // other kernels skipped): setup's workspace sizing and kernel picks.
    cudaError_t exec(const void *in, void *out, cudaStream_t s, std::vector<std::pair<GOp *, GConvArgs>> *collect);
};

// conv weights with the expected LeakyReLU (0 = none)
bool cv(const Reader &rd, const std::string &name, Conv *c, float slope, int cin, int cout, std::string *why)
{
    if (!rd.conv(name, c, why)) return false;
    if ((slope > 0) != c->lrelu || (slope > 0 && std::fabs(c->slope - slope) > 1e-6f) || (cin && c->cin != cin) || (cout && c->cout != cout))
        return refuse(why, name + ": unexpected shape or activation");
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

std::unique_ptr<VidNet> Tfdat::build(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    const std::string P = "/model/";
    if (!rd.node(P + "flow_net/dec3/dec3.0/Conv") || !rd.node(P + "temporal_fusion/fuse/fuse.0/Conv") || !FdatBody::present(rd, P))
        return refuse(why, "not a TFDAT graph"), nullptr;
    if (w % 32 || h % 32 || w < 64 || h < 64) return refuse(why, "TFDAT needs a frame size that is a multiple of 32"), nullptr;
    std::unique_ptr<Tfdat> f(new Tfdat);
    f->W = w;
    f->H = h;
    // frames: one encoder instance per frame
    f->nf = rd.node(P + "enc1/enc1.0/Conv") ? 1 : 0;
    while (rd.node(P + "enc1/enc1.0_" + std::to_string(f->nf) + "/Conv")) f->nf++;
    if (f->nf < 3 || f->nf % 2 == 0) return refuse(why, "TFDAT frame count"), nullptr;
    Conv c;
    auto G = [&](GOp &op, const Conv &cc, int cin_pad, int cout_pad) {
        op.w = f->res.gup(conv_taps(cc, nullptr), cc.b, cc.cin, cin_pad, cc.cout, cout_pad);
    };
    const char *encs[6] = {"enc1/enc1.0", "enc1/enc1.2", "enc2/enc2.0", "enc2/enc2.2", "enc3/enc3.0", "enc3/enc3.2"};
    const int ecin[6] = {3, 32, 32, 64, 64, 96}, ecout[6] = {32, 32, 64, 64, 96, 96};
    for (int i = 0; i < 6; i++) {
        if (!cv(rd, P + encs[i] + "/Conv", &c, 0.2f, ecin[i], ecout[i], why)) return nullptr;
        if (c.stride != (i % 2 ? 1 : 2)) return refuse(why, std::string(encs[i]) + ": stride"), nullptr;
        G(f->enc[i], c, pad16(ecin[i]), ecout[i]);
    }
    struct Dec { const char *name; int cin, ch, cin_pad; GOp *ops; } decs[3] = {
        {"flow_net/dec3/dec3", 192, 96, CS3, f->dec3}, {"flow_net/dec2/dec2", 130, 64, CS2, f->dec2}, {"flow_net/dec1/dec1", 66, 32, CS1, f->dec1}};
    for (auto &d : decs) {
        const std::string n = P + d.name;
        if (!cv(rd, n + ".0/Conv", &c, 0.2f, d.cin, d.ch, why)) return nullptr;
        G(d.ops[0], c, d.cin_pad, d.ch);
        if (!cv(rd, n + ".2/Conv", &c, 0.2f, d.ch, d.ch, why)) return nullptr;
        G(d.ops[1], c, d.ch, d.ch);
        if (!cv(rd, n + ".4/Conv", &c, 0, d.ch, 2, why)) return nullptr;
        G(d.ops[2], c, d.ch, 16);
    }
    if (!cv(rd, P + "conv_first/conv_first.0/Conv", &c, 0.2f, 0, 0, why)) return nullptr;
    if (c.cin == 3) f->U = 1;
    else if (c.cin == 12) f->U = 2;
    else if (c.cin == 48) f->U = 4;
    else return refuse(why, "TFDAT conv_first input channels"), nullptr;
    f->SC = 4 / f->U;
    f->UC = pad16(c.cin);
    const int U = f->U, UC = f->UC;
    const int C = f->C = c.cout, CP = f->CP = pad16(C);
    if (CP > 128) return refuse(why, "TFDAT channels"), nullptr;
    G(f->cf0, c, UC, CP);
    if (!cv(rd, P + "conv_first/conv_first.2/Conv", &c, 0, C, C, why)) return nullptr;
    G(f->cf2, c, CP, CP);
    // align_refine: 1x1 over [nb, ctr] (rows 0.. and 128..), LeakyReLU, 3x3, plus nb
    std::vector<float> wm, bm;
    std::vector<int> map(2 * C);
    for (int i = 0; i < 2 * C; i++) map[i] = i < C ? i : 128 + (i - C);
    int co = 0;
    if (!conv1x1_matrix(rd, P + "align_refine/body/body.0/Conv", CAT, &map, &wm, &bm, &co, why) || co != C ||
        !lrelu_after(rd, P + "align_refine/body/body.0/Conv", P + "align_refine/body/body.1/LeakyRelu"))
        return refuse(why, "TFDAT align_refine"), nullptr;
    f->ar0 = f->res.mat(wm, bm.empty() ? nullptr : &bm, CAT, C);
    if (!cv(rd, P + "align_refine/body/body.2/Conv", &c, 0, C, C, why)) return nullptr;
    G(f->ar2, c, CP, CP);
    // temporal fusion
    if (!conv1x1_matrix(rd, P + "temporal_fusion/emb_center/Conv", C, nullptr, &wm, &bm, &co, why) || co != C || !bm.empty())
        return refuse(why, "TFDAT emb_center"), nullptr;
    f->embc = f->res.mat(wm, nullptr, C, C);
    if (!conv1x1_matrix(rd, P + "temporal_fusion/emb_neighbor/Conv", C, nullptr, &wm, &bm, &co, why) || co != C || !bm.empty())
        return refuse(why, "TFDAT emb_neighbor"), nullptr;
    f->embn = f->res.mat(wm, nullptr, C, C);
    std::vector<float> sc;
    if (!init_input(rd, rd.node(P + "temporal_fusion/Mul_1"), &sc) || sc.size() != 1) return refuse(why, "TFDAT similarity scale"), nullptr;
    f->sim_scale = sc[0];
    if (!cv(rd, P + "temporal_fusion/conf/conf.0/Conv", &c, 0.2f, 2, 16, why)) return nullptr;
    G(f->conf0, c, 16, 16);
    if (!cv(rd, P + "temporal_fusion/conf/conf.2/Conv", &c, 0, 16, 1, why)) return nullptr;
    G(f->conf2, c, 16, 16);
    {
        const OnnxNode *n = rd.node(P + "temporal_fusion/fuse/fuse.0/Conv");
        std::vector<float> v, b;
        std::vector<int64_t> d;
        if (!n || !tensor_floats(rd, n, 1, &v, &d) || d.size() != 4 || d[0] != C || d[1] != (int64_t)f->nf * C || d[2] != 1 ||
            n->in.size() < 3 || !rd.floats(n->in[2], &b) || (int)b.size() != C ||
            !lrelu_after(rd, P + "temporal_fusion/fuse/fuse.0/Conv", P + "temporal_fusion/fuse/fuse.1/LeakyRelu"))
            return refuse(why, "TFDAT fuse"), nullptr;
        for (int j = 0; j < f->nf; j++) {
            std::vector<float> m((size_t)C * C);
            for (int o = 0; o < C; o++)
                for (int k = 0; k < C; k++) m[(size_t)k * C + o] = v[(size_t)o * f->nf * C + j * C + k];
            f->fuse.push_back(f->res.mat(m, j + 1 == f->nf ? &b : nullptr, C, C));
        }
    }
    if (!cv(rd, P + "temporal_fusion/fuse/fuse.2/Conv", &c, 0.2f, C, C, why)) return nullptr;
    G(f->fuse2, c, CP, CP);
    {
        std::vector<float> v;
        std::vector<int64_t> d;
        if (!tensor_floats(rd, rd.node(P + "temporal_fusion/se/se.1/Conv"), 1, &v, &d) || d.size() != 4 || d[1] != C)
            return refuse(why, "TFDAT se"), nullptr;
        f->ser = (int)d[0];
        f->se1 = f->res.dfl(v);
        if (!tensor_floats(rd, rd.node(P + "temporal_fusion/se/se.3/Conv"), 1, &v, &d) || d[0] != C || d[1] != f->ser)
            return refuse(why, "TFDAT se"), nullptr;
        f->se2 = f->res.dfl(v);
        const OnnxNode *s1 = rd.node(P + "temporal_fusion/se/se.1/Conv"), *s3 = rd.node(P + "temporal_fusion/se/se.3/Conv");
        if ((s1->in.size() > 2 && !s1->in[2].empty()) || (s3->in.size() > 2 && !s3->in[2].empty())) return refuse(why, "TFDAT se bias"), nullptr;
    }
    if (!f->body.parse(rd, P, C, why)) return nullptr;
    // tail
    if (!cv(rd, P + "conv_after/Conv", &c, 0, C, C, why)) return nullptr;
    G(f->after, c, CP, CP);
    if (!cv(rd, P + "upsampler/upsampler.0/Conv", &c, 0.01f, C, 64, why)) return nullptr;
    G(f->up0, c, CP, 64);
    for (int k = 0; k < 2; k++) {
        if (!cv(rd, P + (k ? "upsampler/upsampler.4/Conv" : "upsampler/upsampler.2/Conv"), &c, 0, 64, 256, why)) return nullptr;
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(c, sy, 64, &wt, &bt);
            (k ? f->up4 : f->up2)[sy].w = f->res.gup(wt, bt, 64, 64, 128, 128);
        }
    }
    if (!rd.node(P + "upsampler/upsampler.3/DepthToSpace") || !rd.node(P + "upsampler/upsampler.5/DepthToSpace"))
        return refuse(why, "TFDAT upsampler"), nullptr;
    if (!cv(rd, P + "upsampler/upsampler.6/Conv", &c, 0, 64, 3, why)) return nullptr;
    G(f->up6, c, 64, 16);
    if (!cv(rd, P + "hr_refine/hr_refine.0/Conv", &c, 0.2f, 3, 32, why)) return nullptr;
    G(f->hr0, c, 16, 32);
    if (!cv(rd, P + "hr_refine/hr_refine.2/Conv", &c, 0, 32, 3, why)) return nullptr;
    G(f->hr2, c, 32, 16);
    if (f->res.failed) return refuse(why, "weight upload failed"), nullptr;

    // Buffers: the small and persistent ones (flows, gates, X0 = the body input, also conv_after's
    // residual) in res; the rest in one arena shared by the three phases, which never overlap in
    // time: alignment + fusion, the body (its scratch, its output G first), and the tail (after G).
    // M1..M3 the flow pyramid (always H/2, H/4, H/8); MF the features and the body (H/U); MO the output
    const size_t M1 = (size_t)(w / 2) * (h / 2), M2 = M1 / 4, M3 = M2 / 4, MF = (size_t)(w / U) * (h / U), MO = 16 * MF;
    DevRes &r = f->res;
    auto Fl = [&](size_t n) { return (float *)r.alloc(n * 8); };
    f->F3 = Fl(M3); f->U2 = Fl(M2); f->F2 = Fl(M2); f->U1 = Fl(M1); f->F1 = Fl(M1);
    f->X0 = r.alloc(MF * CP * 2);
    f->sums = (float *)r.alloc(CP * 4);
    f->cm = (float *)r.alloc(CP * 4);
    if (r.failed) return refuse(why, "out of GPU memory"), nullptr;
    auto carve = [&](Bump &bp) {
        auto B = [&](size_t n, int ch) { return bp.take(n * ch * 2); };
        // phase 1: alignment and fusion
        f->E1a = B(M1, 32); f->E1N = B(M1, 32); f->C1 = B(M1, CS1);
        f->E2a = B(M2, 64); f->E2N = B(M2, 64); f->C2 = B(M2, CS2);
        f->E3a = B(M3, 96); f->C3 = B(M3, CS3);
        f->D3a = B(M3, 96); f->D3b = B(M3, 96); f->D3 = B(M3, 16);
        f->D2a = B(M2, 64); f->D2b = B(M2, 64); f->D2 = B(M2, 16);
        f->D1a = B(M1, 32); f->D1b = B(M1, 32); f->D1 = B(M1, 16);
        f->US = B(MF, UC); f->WUS = B(MF, UC); f->CONF = B(MF, 16); f->CF1 = B(MF, 16); f->CF2 = B(MF, 16);
        f->T1 = B(MF, CP); f->CATB = B(MF, CAT); f->AR1 = B(MF, CP); f->AL = B(MF, CP); f->EC = B(MF, CP); f->EN = B(MF, CP);
        f->SLOT = B(MF, CP); f->ACC = B(MF, CP); f->FU = B(MF, CP);
        const size_t p1 = bp.off;
        // phase 3: the tail, after the body's output (the first MF x CP of the arena)
        bp.off = 0;
        B(MF, CP);
        f->TT = B(MF, CP); f->U0 = B(MF, 64); f->V1 = B(4 * MF, 64); f->V2 = B(MO, 64); f->RZ = B(MO, 16); f->S = B(MO, 16);
        f->HR = f->V2;   // V2 is dead once up6 has run
        f->O16 = f->RZ;  // RZ is dead once up6 has run
        bp.off = std::max(bp.off, p1);
    };
    Bump cnt;
    carve(cnt);
    const size_t arena = std::max(cnt.off, f->body.scratch_bytes(w / U, h / U));
    Bump bp;
    bp.base = (char *)r.alloc(arena);
    if (!bp.base) return refuse(why, "out of GPU memory"), nullptr;
    carve(bp);
    if (!f->body.setup(w / U, h / U, s, why, bp.base)) return nullptr;
    // kernel picks on a zero input
    void *zin = nullptr;
    if (cudaMalloc(&zin, (size_t)3 * f->nf * w * h * 2) != cudaSuccess) return refuse(why, "out of GPU memory"), nullptr;
    cudaMemset(zin, 0, (size_t)3 * f->nf * w * h * 2);
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    f->exec(zin, nullptr, s, &ops);
    f->ws_floats = gworkspace_max(ops);
    bool ok = !f->ws_floats || (f->wsp = (float *)r.alloc(f->ws_floats * 4));
    if (ok) {
        ops.clear();
        f->exec(zin, nullptr, s, &ops);
        ok = gprepare(ops, s, why);
    } else {
        refuse(why, "out of GPU memory");
    }
    cudaStreamSynchronize(s);
    cudaFree(zin);
    if (!ok) return nullptr;
    if (cudaGetLastError() != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    f->what = "TFDAT " + std::to_string(f->SC) + "x, " + std::to_string(f->nf) + " frames, " + f->body.describe();
    return std::unique_ptr<VidNet>(f.release());
}

cudaError_t Tfdat::exec(const void *in, void *out, cudaStream_t s, std::vector<std::pair<GOp *, GConvArgs>> *collect)
{
    const int w1 = W / 2, h1 = H / 2, w2 = W / 4, h2 = H / 4, w3 = W / 8, h3 = H / 8, M1 = w1 * h1, M2 = w2 * h2, M3 = w3 * h3;
    const int wf = W / U, hf = H / U, MF = wf * hf, OW = 4 * wf, OH = 4 * hf;  // features; output
    const size_t plane = (size_t)W * H;
    const uint16_t *frames = (const uint16_t *)in;
    const int ctr = nf / 2;
    const void *ctrp = frames + (size_t)ctr * 3 * plane;
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
    auto act = [](GConvArgs a, float slope) {
        a.act = GAct::LRelu;
        a.slope = slope;
        return a;
    };
    auto s2 = [](GConvArgs a) {
        a.stride = 2;
        return a;
    };
    auto r1 = [](GConvArgs a, const void *r, int rs) {
        a.r1 = r;
        a.r1_stride = rs;
        return a;
    };
    auto off = [](void *p, int halfs) { return (void *)((uint16_t *)p + halfs); };
    // the 3-level encoder of frame planes fp into e1 (stride s1), e2 (s2s), e3 (s3s)
    auto encode = [&](const void *fp, void *e1, int s1, void *e2, int s2s, void *e3, int s3s) -> cudaError_t {
        GConvArgs a = s2(act(ga(fp, (int)plane, H, W, E1a, 32, h1, w1), 0.2f));
        a.nchw = true;
        CK(Gc(enc[0], a));
        CK(Gc(enc[1], act(ga(E1a, 32, h1, w1, e1, s1, h1, w1), 0.2f)));
        CK(Gc(enc[2], s2(act(ga(e1, s1, h1, w1, E2a, 64, h2, w2), 0.2f))));
        CK(Gc(enc[3], act(ga(E2a, 64, h2, w2, e2, s2s, h2, w2), 0.2f)));
        CK(Gc(enc[4], s2(act(ga(e2, s2s, h2, w2, E3a, 96, h3, w3), 0.2f))));
        CK(Gc(enc[5], act(ga(E3a, 96, h3, w3, e3, s3s, h3, w3), 0.2f)));
        return cudaSuccess;
    };
    auto dec = [&](GOp *ops, const void *cin, int cs, void *a, void *b, void *d, int ch, int hh, int ww) -> cudaError_t {
        CK(Gc(ops[0], act(ga(cin, cs, hh, ww, a, ch, hh, ww), 0.2f)));
        CK(Gc(ops[1], act(ga(a, ch, hh, ww, b, ch, hh, ww), 0.2f)));
        CK(Gc(ops[2], ga(b, ch, hh, ww, d, 16, hh, ww)));
        return cudaSuccess;
    };
    // conv_first of an unshuffled frame (stride UC at hf x wf) into dst (stride CAT)
    auto first = [&](const void *us, void *dst) -> cudaError_t {
        CK(Gc(cf0, act(ga(us, UC, hf, wf, T1, CP, hf, wf), 0.2f)));
        CK(Gc(cf2, ga(T1, CP, hf, wf, dst, CAT, hf, wf)));
        return cudaSuccess;
    };
    auto gemm = [&](const TMat &m, const void *a, int sa, void *o, TAct ac, const void *add, bool last) -> cudaError_t {
        if (collect) return cudaSuccess;
        TGemm g;
        g.a = a; g.sa = sa; g.M = MF; g.out = o; g.so = CP; g.act = ac;
        if (add) { g.add1 = add; g.s1 = CP; }
        g.act_last = last;
        return tgemm(m, g, s);
    };

    // centre frame: encoder into the concat buffers' centre slices, features, embedding, slot 0
    CK(encode(ctrp, off(C1, 32), CS1, off(C2, 64), CS2, off(C3, 96), CS3));
    pr.mark(P_ENC);
    K(tf_unshuffle(ctrp, H, W, U, US, UC, s));
    CK(first(US, off(CATB, 128)));
    CK(gemm(*embc, off(CATB, 128), CAT, EC, TAct::None, nullptr, false));
    CK(gemm(*fuse[0], off(CATB, 128), CAT, ACC, TAct::None, nullptr, false));
    pr.mark(P_ALIGN);
    int slot = 1;
    for (int fr = 0; fr < nf; fr++) {
        if (fr == ctr) continue;
        const void *nbp = frames + (size_t)fr * 3 * plane;
        CK(encode(nbp, E1N, 32, E2N, 64, C3, CS3));
        pr.mark(P_ENC);
        // flow, coarse to fine
        CK(dec(dec3, C3, CS3, D3a, D3b, D3, 96, h3, w3));
        K(tf_flow_add(nullptr, D3, 16, F3, M3, s));
        K(tf_up_warp(F3, h3, w3, E2N, 64, 64, C2, CS2, 128, U2, s));
        CK(dec(dec2, C2, CS2, D2a, D2b, D2, 64, h2, w2));
        K(tf_flow_add(U2, D2, 16, F2, M2, s));
        K(tf_up_warp(F2, h2, w2, E1N, 32, 32, C1, CS1, 64, U1, s));
        CK(dec(dec1, C1, CS1, D1a, D1b, D1, 32, h1, w1));
        K(tf_flow_add(U1, D1, 16, F1, M1, s));
        pr.mark(P_FLOW);
        K(tf_frame_warp(F1, nbp, ctrp, H, W, U, WUS, UC, CONF, s));
        pr.mark(P_WARP);
        // features, alignment refinement
        CK(first(WUS, CATB));
        CK(gemm(*ar0, CATB, CAT, AR1, TAct::LRelu, nullptr, false));
        CK(Gc(ar2, r1(ga(AR1, CP, hf, wf, AL, CP, hf, wf), CATB, CAT)));
        pr.mark(P_ALIGN);
        // confidence-weighted slot into the fusion 1x1
        CK(gemm(*embn, AL, CP, EN, TAct::None, nullptr, false));
        K(tf_sim(EC, EN, CP, C, sim_scale, CONF, MF, s));
        CK(Gc(conf0, act(ga(CONF, 16, hf, wf, CF1, 16, hf, wf), 0.2f)));
        CK(Gc(conf2, ga(CF1, 16, hf, wf, CF2, 16, hf, wf)));
        K(tf_gate(AL, CF2, 16, SLOT, CP, C, MF, s));
        const bool last = slot + 1 == nf;
        CK(gemm(*fuse[slot], SLOT, CP, ACC, last ? TAct::LRelu : TAct::None, ACC, last));
        slot++;
        pr.mark(P_FUSE);
    }
    CK(Gc(fuse2, act(ga(ACC, CP, hf, wf, FU, CP, hf, wf), 0.2f)));
    K(cudaMemsetAsync(sums, 0, CP * 4, s));
    K(tcolsum(FU, CP, MF, C, sums, s));
    K(fdat_aim_cg(sums, MF, C, ser, se1, se2, cm, CP, s));
    K(tf_colscale(FU, cm, X0, CP, C, MF, s));
    pr.mark(P_FUSE);
    if (!collect) CK(body.run(X0, wf, hf, s, &pr));
    // tail
    CK(Gc(after, r1(ga(body.out(), CP, hf, wf, TT, CP, hf, wf), X0, CP)));
    CK(Gc(up0, act(ga(TT, CP, hf, wf, U0, 64, hf, wf), 0.01f)));
    for (int sy = 0; sy < 2; sy++) {
        GConvArgs a = ga(U0, 64, hf, wf, V1, 64, hf, wf);
        a.R = 2;
        a.sy = sy;
        CK(Gc(up2[sy], a));
    }
    for (int sy = 0; sy < 2; sy++) {
        GConvArgs a = ga(V1, 64, 2 * hf, 2 * wf, V2, 64, 2 * hf, 2 * wf);
        a.R = 2;
        a.sy = sy;
        CK(Gc(up4[sy], a));
    }
    K(tf_resize(ctrp, H, W, SC, RZ, s));
    CK(Gc(up6, r1(ga(V2, 64, OH, OW, S, 16, OH, OW), RZ, 16)));
    CK(Gc(hr0, act(ga(S, 16, OH, OW, HR, 32, OH, OW), 0.2f)));
    CK(Gc(hr2, r1(ga(HR, 32, OH, OW, O16, 16, OH, OW), S, 16)));
    pr.mark(P_UP);
    K(img_from_nhwc(O16, 16, OW, OH, out, false, s));
    pr.mark(P_IO);
#undef K
#undef CK
    if (!collect) {
        char label[48];
        snprintf(label, sizeof(label), "tfdat %dx%d", W, H);
        pr.report(label);
    }
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<VidNet> build_tfix(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why);

std::unique_ptr<VidNet> VidNet::create(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why)
{
    std::string w1, w2;
    if (auto n = Tfdat::build(g, w, h, s, &w1)) return n;
    if (auto n = build_tfix(g, w, h, s, &w2)) return n;
    if (why) *why = w2.rfind("not a", 0) ? w2 : w1;
    return nullptr;
}

}  // namespace ajinative
