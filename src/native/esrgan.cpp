// native/esrgan.cpp — ESRGAN (RRDBNet, found by structure: native/esrgan_graph.h) on gconv, and
// ImgNet::create.
//
// Dense blocks without concatenation: each RDB works in one NHWC buffer of nf + 4*gc channels; conv k
// reads its first nf + (k-1)*gc channels and writes its gc channels into the next slice. The RDB's
// last conv (the export folds the 0.2 into its weights) adds the block input; the third RDB of an
// RRDB also folds the RRDB's own 0.2 and residual (out = 0.2 * (conv + x3) + x) and writes in place
// over the RRDB input, which the next RRDB then reads. The upsampler convs read their input
// nearest-upsampled on the fly.
#include "native/imgnet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "native/gconv.h"
#include "native/img_kernels.h"
#include "native/netres.h"
#include "native/esrgan_graph.h"
#include "native/onnx_read.h"

namespace ajinative {
namespace {

using namespace onnxread;

struct GOp {
    GWeights *w = nullptr;
    int variant = 0;
};

class Esrgan : public ImgNet {
public:
    ~Esrgan() override
    {
        for (GWeights *w : weights) {
            grelease(*w);
            delete w;
        }
        for (void *b : bufs) cudaFree(b);
    }

    static std::unique_ptr<ImgNet> build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override;
    int scale() const override { return scale_; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return bytes; }
    size_t weight_bytes() const override { return wbytes; }
    bool accepts(int w, int h) const override
    {
        return w > 0 && h > 0 && w <= maxw && h <= maxh && w % unshuffle == 0 && h % unshuffle == 0;
    }

private:
    int scale_ = 4, nf = 64, gc = 32, nb = 0, DS = 0, maxw = 0, maxh = 0;
    int unshuffle = 1;   // the input's pixel-unshuffle factor (1, 2, 4)
    int nup = 2;         // nearest 2x upsample stages (S = 2^nup over the body size)
    int in_stride = 16;  // channels of the unshuffled input (16, or 48 for 4x unshuffle)
    bool clamp = false;  // a final Clip(0, 1)
    std::vector<float> rrdb_scale;  // per RRDB
    int acc_mode = 0;
    GOp first, body, hr, last;
    std::vector<GOp> ups;    // nup
    std::vector<GOp> dense;  // nb * 3 RDBs * 5 convs
    std::vector<GWeights *> weights;
    std::vector<void *> bufs;
    size_t bytes = 0, wbytes = 0;  // all device memory; of which the weights
    void *in16 = nullptr, *F0 = nullptr, *D[3] = {}, *T = nullptr, *U[3] = {}, *Q = nullptr, *O16 = nullptr;
    float *ws = nullptr;
    size_t ws_floats = 0;
    std::string what;

    void *alloc(size_t halfs)
    {
        void *p = nullptr;
        if (cudaMalloc(&p, halfs * 2) != cudaSuccess) return nullptr;
        bufs.push_back(p);
        bytes += halfs * 2;
        return p;
    }
    GWeights *upload(const Conv &c, float scale, int cin_pad)
    {
        std::vector<float> sc(c.cout, scale), b(c.cout);
        for (int o = 0; o < c.cout; o++) b[o] = c.b[o] * scale;
        GWeights *g = new GWeights;
        if (gupload(*g, conv_taps(c, scale == 1.f ? nullptr : &sc).data(), b.data(), c.cin, cin_pad, c.cout, pad16(c.cout)) != cudaSuccess) {
            delete g;
            return nullptr;
        }
        weights.push_back(g);
        bytes += (size_t)9 * cin_pad * pad16(c.cout) * 2;
        wbytes += (size_t)9 * cin_pad * pad16(c.cout) * 2;
        return g;
    }
    // The ops of one run at body size bw x bh, in order, as (op, args) pairs.
    void plan(int bw, int bh, std::vector<std::pair<GOp *, GConvArgs>> *ops);
};

GConvArgs args(const void *src, int sstride, int IH, int IW, void *dst, int dstride, int H, int W, GAct act)
{
    GConvArgs a;
    a.src = src;
    a.src_stride = sstride;
    a.IH = IH;
    a.IW = IW;
    a.dst = dst;
    a.dst_stride = dstride;
    a.H = H;
    a.W = W;
    a.act = act;
    a.slope = 0.2f;
    return a;
}

void Esrgan::plan(int bw, int bh, std::vector<std::pair<GOp *, GConvArgs>> *ops)
{
    ops->clear();
    const GAct L = GAct::LRelu, N = GAct::None;
    // first conv -> F0 (also the trunk's global residual); its copy into D[0] is done by run()
    ops->push_back({&first, args(in16, in_stride, bh, bw, F0, nf, bh, bw, N)});
    for (int i = 0; i < nb; i++)
        for (int r = 0; r < 3; r++) {
            void *src = D[r], *nxt = D[(r + 1) % 3];
            for (int k = 0; k < 4; k++)
                ops->push_back({&dense[(i * 3 + r) * 5 + k],
                                args(src, DS, bh, bw, (uint16_t *)src + nf + k * gc, DS, bh, bw, L)});
            GConvArgs a = args(src, DS, bh, bw, nxt, DS, bh, bw, N);
            a.r1 = src;  // the RDB input: the first nf channels of its dense buffer
            a.r1_stride = DS;
            a.c1 = 1.f;
            if (r == 2) {
                // out = s * (s_rdb * conv + x3) + x: weights x s * s_rdb (at upload), x3 x s, x (in place, D[0])
                a.c1 = rrdb_scale[i];
                a.r2 = D[0];
                a.r2_stride = DS;
                a.c2 = 1.f;
            }
            ops->push_back({&dense[(i * 3 + r) * 5 + 4], a});
        }
    GConvArgs b = args(D[0], DS, bh, bw, T, nf, bh, bw, N);
    b.r1 = F0;
    b.r1_stride = nf;
    ops->push_back({&body, b});
    // nup nearest 2x upsample stages, each conv reading its input upsampled on the fly
    void *cur = T;
    int s = 1;
    for (int k = 0; k < nup; k++) {
        s *= 2;
        GConvArgs u = args(cur, nf, s * bh, s * bw, U[k], nf, s * bh, s * bw, L);
        u.up2 = true;
        ops->push_back({&ups[k], u});
        cur = U[k];
    }
    ops->push_back({&hr, args(cur, nf, s * bh, s * bw, Q, nf, s * bh, s * bw, L)});
    ops->push_back({&last, args(Q, nf, s * bh, s * bw, O16, 16, s * bh, s * bw, N)});
    // accumulation: acc_mode 0 fp16 everywhere, 1 fp32 everywhere, 2 fp32 on the deep convs (cin >= 128)
    for (auto &o : *ops) {
        o.second.ws = ws;
        o.second.ws_floats = ws_floats;
        o.second.acc32 = acc_mode == 1 || (acc_mode == 2 && o.first->w->cin_pad >= 128);
    }
}

std::unique_ptr<ImgNet> Esrgan::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    EsrganGraph eg;
    if (!find_esrgan(rd, &eg, why)) return nullptr;
    std::unique_ptr<Esrgan> e(new Esrgan);
    e->unshuffle = eg.unshuffle;
    e->nup = eg.nup;
    e->clamp = eg.clamp;
    e->nf = eg.nf;
    e->gc = eg.gc;
    e->nb = eg.nb;
    e->rrdb_scale = eg.rrdb_scale;
    e->in_stride = pad16(3 * eg.unshuffle * eg.unshuffle);
    if (e->nf % 16 || e->nf > 128) return refuse(why, "ESRGAN feature count"), nullptr;
    if (e->gc % 16) return refuse(why, "ESRGAN growth channels"), nullptr;
    if (e->nup > 3) return refuse(why, "ESRGAN upsample stages"), nullptr;
    e->DS = e->nf + 4 * e->gc;
    e->scale_ = eg.scale();
    // fp16 accumulation by default (what TensorRT does too); AJI_NATIVE_ACC32=1 accumulates in fp32,
    // which this network's last blocks are sensitive to (closer to the fp32 model, ~TensorRT's speed)
    if (const char *am = getenv("AJI_NATIVE_ACC32")) e->acc_mode = atoi(am) ? 1 : 0;
    // weights
    Conv c0;
    if (!rd.conv(eg.first, &c0, why)) return nullptr;
    if (!eg.cin_from.empty()) {  // SpaceToDepth's channel order -> the kernels' (CRD)
        std::vector<float> w(c0.w.size());
        for (int o = 0; o < c0.cout; o++)
            for (int i = 0; i < c0.cin; i++)
                for (int t = 0; t < 9; t++) w[((size_t)o * c0.cin + i) * 9 + t] = c0.w[((size_t)o * c0.cin + eg.cin_from[i]) * 9 + t];
        c0.w.swap(w);
    }
    e->first.w = e->upload(c0, 1.f, e->in_stride);
    for (int i = 0; i < e->nb; i++)
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 5; k++) {
                Conv c;
                if (!rd.conv(eg.dense[(i * 3 + r) * 5 + k], &c, why)) return nullptr;
                if (k < 4 && (!c.lrelu || c.stride != 1)) return refuse(why, "ESRGAN dense conv"), nullptr;
                if (k == 4 && c.lrelu) return refuse(why, "ESRGAN RDB last conv"), nullptr;
                const float sc = k < 4 ? 1.f : eg.rdb_scale[i * 3 + r] * (r == 2 ? eg.rrdb_scale[i] : 1.f);
                GOp op;
                op.w = e->upload(c, sc, c.cin);
                e->dense.push_back(op);
            }
    Conv cb, chr, cl;
    if (!rd.conv(eg.body, &cb, why) || !rd.conv(eg.hr, &chr, why) || !rd.conv(eg.last, &cl, why)) return nullptr;
    if (cb.lrelu || !chr.lrelu || cl.lrelu) return refuse(why, "ESRGAN output activations"), nullptr;
    e->body.w = e->upload(cb, 1.f, e->nf);
    e->hr.w = e->upload(chr, 1.f, e->nf);
    e->last.w = e->upload(cl, 1.f, e->nf);
    for (const OnnxNode *u : eg.ups) {
        Conv cu;
        if (!rd.conv(u, &cu, why)) return nullptr;
        if (!cu.lrelu) return refuse(why, "ESRGAN upsample activation"), nullptr;
        GOp op;
        op.w = e->upload(cu, 1.f, e->nf);
        e->ups.push_back(op);
    }
    for (GWeights *w : e->weights)
        if (!w) return refuse(why, "weight upload failed"), nullptr;

    // buffers for the largest tile
    e->maxw = max_w;
    e->maxh = max_h;
    const int bw = max_w / e->unshuffle, bh = max_h / e->unshuffle;
    if (bw < 1 || bh < 1) return refuse(why, "tile too small"), nullptr;
    const size_t px = (size_t)bw * bh;
    const int S = 1 << e->nup;
    // One arena shared by lifetime (steps of run(): 1 in16, 2 F0 and its copy into D[0], 3 the
    // dense blocks, 4 T, 5.. the upsample stages, then Q, O16, out): the upsampler's buffers reuse
    // the trunk's.
    Lifetimes lt;
    lt.add(&e->in16, px * e->in_stride * 2, {{1, 2}});
    lt.add(&e->F0, px * e->nf * 2, {{2, 4}});
    for (auto &d : e->D) lt.add(&d, px * e->DS * 2, {{2, 4}});
    lt.add(&e->T, px * e->nf * 2, {{4, 5}});
    for (int k = 0; k < e->nup; k++) lt.add(&e->U[k], px * ((size_t)1 << (2 * (k + 1))) * e->nf * 2, {{5 + k, 6 + k}});
    lt.add(&e->Q, px * S * S * e->nf * 2, {{5 + e->nup, 6 + e->nup}});
    lt.add(&e->O16, px * S * S * 16 * 2, {{6 + e->nup, 7 + e->nup}});
    if (!lt.assign([&](size_t n) { return e->alloc((n + 1) / 2); })) return refuse(why, "out of GPU memory"), nullptr;
    // split-K workspace: the most any op needs at the largest and at a small tile
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        const int tw = sz ? std::min(bw, 32) : bw, th = sz ? std::min(bh, 32) : bh;
        e->plan(tw, th, &ops);
        for (auto &o : ops) e->ws_floats = std::max(e->ws_floats, gworkspace(*o.first->w, o.second));
    }
    if (e->ws_floats) {
        e->ws = (float *)e->alloc(e->ws_floats * 2);
        if (!e->ws) return refuse(why, "out of GPU memory"), nullptr;
    }
    // check the ops and pick tile variants at the largest tile, once per distinct shape
    e->plan(bw, bh, &ops);
    std::map<std::tuple<int, int, int, int, int, bool, bool>, int> picked;
    for (auto &o : ops) {
        std::string w2;
        if (!gsupported(*o.first->w, o.second, &w2)) return refuse(why, "kernel: " + w2), nullptr;
        const auto key = std::make_tuple(o.first->w->cin_pad, o.first->w->cout_pad, o.second.H, o.second.W, o.second.src_stride,
                                         o.second.r1 != nullptr, o.second.up2);
        auto it = picked.find(key);
        if (it == picked.end()) {
            const int v = gpick(*o.first->w, o.second, s);
            if (v < 0) return refuse(why, "kernel timing failed"), nullptr;
            it = picked.emplace(key, v).first;
        }
        o.first->variant = it->second;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    char buf[160];
    snprintf(buf, sizeof(buf), "ESRGAN %dx, %d RRDB, %d features, growth %d%s%s", e->scale_, e->nb, e->nf, e->gc,
             e->unshuffle > 1 ? (e->unshuffle == 2 ? ", pixel-unshuffle 2" : ", pixel-unshuffle 4") : "", e->clamp ? ", clamped" : "");
    e->what = buf;
    return std::unique_ptr<ImgNet>(e.release());
}

cudaError_t Esrgan::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    const int bw = w / unshuffle, bh = h / unshuffle, S = 1 << nup;
    cudaError_t e = img_to_nhwc_by(in, in_f32, w, h, unshuffle, in16, in_stride, s);
    if (e != cudaSuccess) return e;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    plan(bw, bh, &ops);
    for (size_t i = 0; i < ops.size(); i++) {
        e = glaunch(*ops[i].first->w, ops[i].second, ops[i].first->variant, s);
        if (e != cudaSuccess) return e;
        if (i == 0) {
            // the trunk input into the first dense buffer's prefix
            e = cudaMemcpy2DAsync(D[0], (size_t)DS * 2, F0, (size_t)nf * 2, (size_t)nf * 2, (size_t)bw * bh, cudaMemcpyDeviceToDevice, s);
            if (e != cudaSuccess) return e;
        }
    }
    return img_from_nhwc(O16, 16, S * bw, S * bh, out, out_f32, s, clamp);
}

}  // namespace

std::unique_ptr<ImgNet> build_fdat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
std::unique_ptr<ImgNet> build_hat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
std::unique_ptr<ImgNet> build_dat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
std::unique_ptr<ImgNet> build_lowered(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);

std::unique_ptr<ImgNet> ImgNet::create(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    if (g.inputs.size() != 1) return refuse(why, "one input expected"), nullptr;
    std::string w;
    if (auto n = Esrgan::build(g, max_w, max_h, s, &w)) return n;
    std::string w2, w3, w4;
    if (auto n = build_fdat(g, max_w, max_h, s, &w2)) return n;
    if (auto n = build_hat(g, max_w, max_h, s, &w3)) return n;
    if (auto n = build_dat(g, max_w, max_h, s, &w4)) return n;
    // the conv nets the video path lowers (SPAN, SPANF3, Compact), on the executor
    std::string w5;
    if (auto n = build_lowered(g, max_w, max_h, s, &w5)) return n;
    // AJI_NATIVE_WHY=1: every family's reason on stderr (which one a model nearly is)
    if (const char *v = getenv("AJI_NATIVE_WHY"); v && *v == '1')
        fprintf(stderr, "native: esrgan: %s\nnative: fdat: %s\nnative: hat: %s\nnative: dat: %s\nnative: lowered: %s\n", w.c_str(), w2.c_str(),
                w3.c_str(), w4.c_str(), w5.c_str());
    // the reason of the family the graph belongs to (the others say "not a ... graph"); the
    // lowering's when it is none of them
    if (why)
        *why = w4.rfind("not a", 0) ? w4 : w3.rfind("not a", 0) ? w3 : w2.rfind("not a", 0) ? w2 : w.rfind("not a", 0) ? w : w5;
    return nullptr;
}

}  // namespace ajinative
