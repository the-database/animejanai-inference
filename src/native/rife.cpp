// native/rife.cpp — see rife.h.
#include "native/rife.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#include "native/gconv.h"
#include "native/onnx_graph.h"
#include "native/pack.h"
#include "native/onnx_read.h"
#include "native/rife_kernels.h"
#include "native/rife_plan.h"

namespace ajinative {

namespace {

using namespace onnxread;

struct Step {
    enum Kind { GCONV, GATHER, MERGE, FINAL } kind;
    RifeBlockOut mo;
    const void *mr = nullptr;
    GWeights *w = nullptr;
    GConvArgs a;
    int variant = 0;
    RifeGather g;
};

}  // namespace

struct Rife::Impl {
    int W = 0, H = 0, F = 0, FP = 0, nb = 0;
    std::vector<int> scales;
    std::vector<GWeights *> weights;
    std::vector<void *> bufs;
    size_t bytes = 0;
    RifeFrame frame;
    std::vector<RifeBlockOut> outs;
    std::vector<Step> steps;
    std::string what;
    float *dt = nullptr;                  // the timestep, device
    cudaGraphExec_t graph = nullptr;      // replay of steps for (graph_in, graph_out)
    const void *graph_in = nullptr;
    void *graph_out = nullptr;
    bool graph_ok = true;

    cudaError_t enqueue(const void *in11, void *out3, cudaStream_t s, bool prof);

    ~Impl()
    {
        if (graph) cudaGraphExecDestroy(graph);
        for (GWeights *w : weights) {
            grelease(*w);
            delete w;
        }
        for (void *b : bufs) cudaFree(b);
    }
    void *alloc(size_t halfs)
    {
        void *p = nullptr;
        if (cudaMalloc(&p, halfs * 2) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, halfs * 2);
        bufs.push_back(p);
        bytes += halfs * 2;
        return p;
    }
    GWeights *upload(const std::vector<uint16_t> &w, const std::vector<float> &b, int cin, int cin_pad, int cout, int cout_pad)
    {
        GWeights *g = new GWeights;
        if (gupload(*g, w.data(), b.data(), cin, cin_pad, cout, cout_pad) != cudaSuccess) {
            delete g;
            return nullptr;
        }
        weights.push_back(g);
        bytes += (size_t)9 * cin_pad * cout_pad * 2;
        return g;
    }
};

std::unique_ptr<Rife> Rife::create(const OnnxGraph &g, int W, int H, cudaStream_t s, std::string *why)
{
    RifePlan plan;
    if (!rife_plan(g, W, H, &plan, why)) return nullptr;
    const bool ens = plan.ens;
    const std::vector<Conv> &enc = plan.enc;
    const int F = plan.F;

    std::unique_ptr<Rife> r(new Rife);
    r->d.reset(new Impl);
    Impl &d = *r->d;
    d.W = W;
    d.H = H;
    d.F = F;
    d.FP = F ? pad8(F) : 0;
    d.dt = (float *)d.alloc(2);
    if (!d.dt) return refuse(why, "out of GPU memory"), nullptr;
    const size_t plane = (size_t)W * H;

    // ---- blocks: the shared scratch sizes (a block's input and conv buffers are only used inside
    // it; the block outputs and the frame features persist)
    size_t needX = 0, needT1 = 0, needA = 0;
    for (const RifeBlockPlan &bp : plan.blocks) {
        const size_t px = (size_t)(W / bp.s) * (H / bp.s);
        needX = std::max(needX, px * pad16(bp.c0a.cin));
        needT1 = std::max(needT1, px / 4 * pad16(bp.c0a.cout));
        needA = std::max(needA, px / 16 * pad16(bp.c0b.cout));
    }
    void *sX = d.alloc(needX), *sT1 = d.alloc(needT1), *sA = d.alloc(needA), *sB = d.alloc(needA);
    if (!sX || !sT1 || !sA || !sB) return refuse(why, "out of GPU memory"), nullptr;
    void *sR = ens ? d.alloc(plane * 16) : nullptr;  // ensemble: the reverse pass's block output
    if (ens && !sR) return refuse(why, "out of GPU memory"), nullptr;
    // ---- encoder steps (per frame)
    void *feat[2] = {nullptr, nullptr};
    if (F) {
        int maxc = 0;
        for (const Conv &c : enc) maxc = std::max(maxc, pad16(c.cout));
        // the encoder runs before any block: its ping-pong buffers live in the block input scratch
        void *hb0, *hb1;
        if (needX >= 2 * (plane / 4 * maxc)) {
            hb0 = sX;
            hb1 = (uint16_t *)sX + plane / 4 * maxc;
        } else {
            hb0 = d.alloc(plane / 4 * maxc);
            hb1 = d.alloc(plane / 4 * maxc);
        }
        feat[0] = d.alloc(plane * d.FP);
        feat[1] = d.alloc(plane * d.FP);
        if (!hb0 || !hb1 || !feat[0] || !feat[1]) return refuse(why, "out of GPU memory"), nullptr;
        std::vector<GWeights *> ew;
        for (size_t i = 0; i + 1 < enc.size(); i++) {
            const Conv &c = enc[i];
            GWeights *w = d.upload(conv_taps(c, nullptr), c.b, c.cin, pad16(c.cin), c.cout, pad16(c.cout));
            if (!w) return refuse(why, "weight upload failed"), nullptr;
            ew.push_back(w);
        }
        std::vector<GWeights *> ph(2);
        for (int py = 0; py < 2; py++) {
            std::vector<uint16_t> w;
            std::vector<float> b;
            phase_weights(enc.back(), 2, py, d.FP, &w, &b);
            ph[py] = d.upload(w, b, enc.back().cin, pad16(enc.back().cin), 2 * d.FP, 2 * d.FP);
            if (!ph[py]) return refuse(why, "weight upload failed"), nullptr;
        }
        for (int k = 0; k < 2; k++) {
            void *cur = nullptr;
            int curc = 0;
            for (size_t i = 0; i + 1 < enc.size(); i++) {
                Step st{Step::GCONV};
                st.w = ew[i];
                GConvArgs &a = st.a;
                void *dst = (i % 2 == 0) ? hb0 : hb1;
                if (i == 0) {
                    a.src = nullptr;  // the frame plane, set at run
                    a.nchw = true;
                    a.stride = 2;
                    a.src_stride = (int)plane;
                    a.IH = H;
                    a.IW = W;
                    a.H = H / 2;
                    a.W = W / 2;
                } else {
                    a.src = cur;
                    a.src_stride = curc;
                    a.IH = H / 2;
                    a.IW = W / 2;
                    a.H = H / 2;
                    a.W = W / 2;
                }
                a.dst = dst;
                a.dst_stride = pad16(enc[i].cout);
                a.act = enc[i].lrelu ? GAct::LRelu : GAct::None;
                st.variant = k;  // frame index for the input pointer (patched at run); variant picked below
                d.steps.push_back(st);
                cur = dst;
                curc = pad16(enc[i].cout);
            }
            for (int py = 0; py < 2; py++) {
                Step st{Step::GCONV};
                st.w = ph[py];
                GConvArgs &a = st.a;
                a.src = cur;
                a.src_stride = curc;
                a.IH = H / 2;
                a.IW = W / 2;
                a.H = H / 2;
                a.W = W / 2;
                a.dst = feat[k];
                a.dst_stride = d.FP;
                a.R = 2;
                a.sy = py;
                d.steps.push_back(st);
            }
        }
    }
    d.frame.W = W;
    d.frame.H = H;
    d.frame.feat[0] = feat[0];
    d.frame.feat[1] = feat[1];
    d.frame.F = F;
    d.frame.fstride = d.FP;

    const int nb = (int)plan.blocks.size();
    int prevK = 0;
    for (int b = 0; b < nb; b++) {
        const RifeBlockPlan &bp = plan.blocks[b];
        const std::string pre = "/block" + std::to_string(b) + "/";
        const Conv &c0a = bp.c0a, &c0b = bp.c0b, &last = bp.last;
        const std::vector<Conv> &res = bp.res;
        const std::vector<std::vector<float>> &beta = bp.beta;
        const int c = c0b.cout, K = bp.K, s = bp.s;
        const bool feeds = bp.feeds;
        d.scales.push_back(s);

        // buffers
        const int bw = W / s, bh = H / s, cinp = pad16(c0a.cin), KP = K <= 8 ? 8 : 16;
        if ((size_t)bw * bh * cinp > needX || (size_t)bw * bh / 4 * pad16(c0a.cout) > needT1 || (size_t)bw * bh / 16 * pad16(c) > needA)
            return refuse(why, "scratch sizing"), nullptr;
        void *X = sX, *T1 = sT1, *A = sA, *B = sB;
        void *O = d.alloc((size_t)bw * bh * KP);
        if (!O) return refuse(why, "out of GPU memory"), nullptr;
        RifeBlockOut bo;
        bo.b = O;
        bo.s = s;
        bo.kstride = KP;

        GWeights *w0a = nullptr, *w0b = nullptr, *wlast[4] = {};
        std::vector<GWeights *> wres;
        for (int pass = 0; pass < (ens ? 2 : 1); pass++) {
        void *Odst = pass ? sR : O;
        // gather
        Step ga{Step::GATHER};
        ga.g.nprev = b;
        for (int j = 0; j < b; j++) ga.g.prev[j] = d.outs[j];
        ga.g.feeds_feat = feeds;
        ga.g.K = prevK;
        ga.g.s = s;
        ga.g.dst = X;
        ga.g.dstride = cinp;
        ga.g.cin = c0a.cin;
        ga.g.rev = pass == 1;
        d.steps.push_back(ga);
        // conv0
        if (pass == 0) {
            w0a = d.upload(conv_taps(c0a, nullptr), c0a.b, c0a.cin, cinp, c0a.cout, pad16(c0a.cout));
            w0b = d.upload(conv_taps(c0b, nullptr), c0b.b, c0b.cin, pad16(c0b.cin), c, pad16(c));
            if (!w0a || !w0b) return refuse(why, "weight upload failed"), nullptr;
        }
        Step s0{Step::GCONV};
        s0.w = w0a;
        s0.a.src = X;
        s0.a.src_stride = cinp;
        s0.a.IH = bh;
        s0.a.IW = bw;
        s0.a.H = bh / 2;
        s0.a.W = bw / 2;
        s0.a.dst = T1;
        s0.a.dst_stride = pad16(c0a.cout);
        s0.a.act = GAct::LRelu;
        s0.a.stride = 2;
        d.steps.push_back(s0);
        Step s1{Step::GCONV};
        s1.w = w0b;
        s1.a.src = T1;
        s1.a.src_stride = pad16(c0a.cout);
        s1.a.IH = bh / 2;
        s1.a.IW = bw / 2;
        s1.a.H = bh / 4;
        s1.a.W = bw / 4;
        s1.a.dst = A;
        s1.a.dst_stride = pad16(c);
        s1.a.act = GAct::LRelu;
        s1.a.stride = 2;
        d.steps.push_back(s1);
        // residual convs: lrelu(conv * beta + x), beta folded into weights and bias
        void *cur = A, *oth = B;
        for (size_t k = 0; k < res.size(); k++) {
            if (pass == 0) {
                std::vector<float> bb(c);
                for (int o = 0; o < c; o++) bb[o] = res[k].b[o] * beta[k][o];
                wres.push_back(d.upload(conv_taps(res[k], &beta[k]), bb, c, pad16(c), c, pad16(c)));
                if (!wres.back()) return refuse(why, "weight upload failed"), nullptr;
            }
            GWeights *wr = wres[k];
            Step sr{Step::GCONV};
            sr.w = wr;
            sr.a.src = cur;
            sr.a.src_stride = pad16(c);
            sr.a.IH = bh / 4;
            sr.a.IW = bw / 4;
            sr.a.H = bh / 4;
            sr.a.W = bw / 4;
            sr.a.dst = oth;
            sr.a.dst_stride = pad16(c);
            sr.a.r1 = cur;
            sr.a.r1_stride = pad16(c);
            sr.a.c1 = 1.f;
            sr.a.act = GAct::LRelu;
            d.steps.push_back(sr);
            std::swap(cur, oth);
        }
        // lastconv (transposed) + depth-to-space as 4 row-phase convs
        for (int sy = 0; sy < 4; sy++) {
            if (pass == 0) {
                std::vector<uint16_t> w;
                std::vector<float> bias;
                phase_weights(last, 4, sy, KP, &w, &bias);
                wlast[sy] = d.upload(w, bias, c, pad16(c), 4 * KP, 4 * KP);
                if (!wlast[sy]) return refuse(why, "weight upload failed"), nullptr;
            }
            GWeights *wl = wlast[sy];
            Step sl{Step::GCONV};
            sl.w = wl;
            sl.a.src = cur;
            sl.a.src_stride = pad16(c);
            sl.a.IH = bh / 4;
            sl.a.IW = bw / 4;
            sl.a.H = bh / 4;
            sl.a.W = bw / 4;
            sl.a.dst = Odst;
            sl.a.dst_stride = KP;
            sl.a.R = 4;
            sl.a.sy = sy;
            d.steps.push_back(sl);
        }
        }
        if (ens) {
            Step mg{Step::MERGE};
            mg.mo = bo;
            mg.mr = sR;
            d.steps.push_back(mg);
        }
        d.outs.push_back(bo);
        prevK = K;
    }
    d.nb = nb;
    d.steps.push_back(Step{Step::FINAL});

    // check every conv and pick its tile variant on scratch data (the input pointer of the first
    // encoder conv is unknown until run: use a zeroed full-size scratch tensor)
    void *scratch_in = d.alloc(plane * 11);
    if (!scratch_in) return refuse(why, "out of GPU memory"), nullptr;
    size_t wsf = 0;
    for (Step &st : d.steps)
        if (st.kind == Step::GCONV) wsf = std::max(wsf, gworkspace(*st.w, st.a));
    if (wsf) {
        float *ws = (float *)d.alloc(wsf * 2);  // alloc counts halfs
        if (!ws) return refuse(why, "out of GPU memory"), nullptr;
        for (Step &st : d.steps)
            if (st.kind == Step::GCONV) {
                st.a.ws = ws;
                st.a.ws_floats = wsf;
            }
    }
    for (Step &st : d.steps)
        if (st.kind == Step::GATHER && !rife_gather_supported(d.frame, st.g))
            return refuse(why, "no gather kernel for this feature layout"), nullptr;
    for (Step &st : d.steps) {
        if (st.kind != Step::GCONV) continue;
        GConvArgs a = st.a;
        if (a.nchw) a.src = (const uint16_t *)scratch_in + (size_t)st.variant * 3 * plane;
        std::string w;
        if (!gsupported(*st.w, a, &w)) return refuse(why, "kernel: " + w), nullptr;
        const int v = gpick(*st.w, a, s);
        if (v < 0) return refuse(why, "kernel timing failed"), nullptr;
        if (!a.nchw) st.variant = v;
        else st.a.sy = st.variant, st.variant = v;  // remember the frame index in sy (unused for NHWC output)
    }
    cudaFree(scratch_in);
    d.bufs.erase(std::find(d.bufs.begin(), d.bufs.end(), scratch_in));
    d.bytes -= plane * 11 * 2;
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;

    char buf[160];
    std::string sc;
    for (size_t i = 0; i < d.scales.size(); i++) sc += (i ? "/" : "") + std::to_string(d.scales[i]);
    snprintf(buf, sizeof(buf), "%d blocks, scales %s, %d feature channels%s", nb, sc.c_str(), F, ens ? ", ensemble" : "");
    d.what = buf;
    return r;
}

Rife::~Rife() = default;

cudaError_t Rife::Impl::enqueue(const void *in11, void *out3, cudaStream_t s, bool prof)
{
    Impl &d = *this;
    std::vector<cudaEvent_t> ev;
    if (prof) {
        ev.resize(d.steps.size() + 1);
        for (auto &e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], s);
    }
    size_t si = 0;
    const size_t plane = (size_t)d.W * d.H;
    RifeFrame fr = d.frame;
    fr.in11 = in11;
    for (Step &st : d.steps) {
        cudaError_t e = cudaSuccess;
        switch (st.kind) {
        case Step::GCONV: {
            if (st.a.nchw) {
                GConvArgs a = st.a;
                a.src = (const uint16_t *)in11 + (size_t)st.a.sy * 3 * plane;
                a.sy = 0;
                e = glaunch(*st.w, a, st.variant, s);
            } else {
                e = glaunch(*st.w, st.a, st.variant, s);
            }
            break;
        }
        case Step::GATHER: {
            RifeGather g = st.g;
            g.t = d.dt;
            e = rife_gather(fr, g, s);
            break;
        }
        case Step::MERGE:
            e = rife_merge(fr, st.mo, st.mr, s);
            break;
        case Step::FINAL:
            e = rife_final(fr, d.nb, d.outs.data(), out3, s);
            break;
        }
        if (e != cudaSuccess) return e;
        if (prof) cudaEventRecord(ev[++si], s);
    }
    if (prof) {
        cudaEventSynchronize(ev.back());
        std::map<std::string, float> agg;
        for (size_t i = 0; i < d.steps.size(); i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
            const Step &st = d.steps[i];
            std::string k = st.kind == Step::GATHER ? "gather s" + std::to_string(st.g.s)
                          : st.kind == Step::FINAL ? "final"
                          : st.kind == Step::MERGE ? "merge"
                          : "conv " + std::to_string(st.w->cin) + "->" + std::to_string(st.w->cout) + " " +
                                std::to_string(st.a.W) + "x" + std::to_string(st.a.H) + (st.a.stride == 2 ? " s2" : "") +
                                (st.a.R ? " phase" + std::to_string(st.a.R) : "");
            agg[k] += ms;
        }
        float tot = 0;
        for (auto &kv : agg) tot += kv.second;
        for (auto &kv : agg) fprintf(stderr, "  %-34s %7.3f ms\n", kv.first.c_str(), kv.second);
        fprintf(stderr, "  total %.3f ms\n", tot);
        for (auto &e : ev) cudaEventDestroy(e);
    }
    return cudaSuccess;
}

cudaError_t Rife::run(const void *in11, float t, void *out3, cudaStream_t s)
{
    Impl &d = *this->d;
    static const bool prof = getenv("AJI_NATIVE_PROFILE") != nullptr;
    cudaError_t e = cudaMemcpyAsync(d.dt, &t, sizeof(float), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    if (prof || !d.graph_ok) return d.enqueue(in11, out3, s, prof);
    if (!d.graph || d.graph_in != in11 || d.graph_out != out3) {
        if (d.graph) cudaGraphExecDestroy(d.graph);
        d.graph = nullptr;
        cudaGraph_t gr = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            cudaGetLastError();
            d.graph_ok = false;
            return d.enqueue(in11, out3, s, false);
        }
        e = d.enqueue(in11, out3, s, false);
        const cudaError_t ce = cudaStreamEndCapture(s, &gr);
        if (e != cudaSuccess || ce != cudaSuccess || !gr || cudaGraphInstantiate(&d.graph, gr, 0) != cudaSuccess) {
            if (gr) cudaGraphDestroy(gr);
            cudaGetLastError();
            d.graph = nullptr;
            d.graph_ok = false;
            return d.enqueue(in11, out3, s, false);
        }
        cudaGraphDestroy(gr);
        d.graph_in = in11;
        d.graph_out = out3;
    }
    return cudaGraphLaunch(d.graph, s);
}

std::string Rife::describe() const { return d->what; }
size_t Rife::device_bytes() const { return d->bytes; }

}  // namespace ajinative
