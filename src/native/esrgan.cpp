// native/esrgan.cpp — ESRGAN (RRDBNet, as exported by traiNNer/chaiNNer: /model/model.N node names)
// on gconv, and ImgNet::create.
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
        return w > 0 && h > 0 && w <= maxw && h <= maxh && (!unshuffle || (w % 2 == 0 && h % 2 == 0));
    }

private:
    int scale_ = 4, nf = 64, gc = 32, nb = 0, DS = 0, maxw = 0, maxh = 0;
    bool unshuffle = false;
    float rrdb_scale = 0.2f;
    int acc_mode = 0;
    GOp first, body, up1, up2, hr, last;
    std::vector<GOp> dense;  // nb * 3 RDBs * 5 convs
    std::vector<GWeights *> weights;
    std::vector<void *> bufs;
    size_t bytes = 0, wbytes = 0;  // all device memory; of which the weights
    void *in16 = nullptr, *F0 = nullptr, *D[3] = {}, *T = nullptr, *U1 = nullptr, *P = nullptr, *Q = nullptr, *O16 = nullptr;
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
    ops->push_back({&first, args(in16, 16, bh, bw, F0, nf, bh, bw, N)});
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
                // out = s * (conv + x3) + x: weights x s (at upload), x3 x s, x (in place, D[0])
                a.c1 = rrdb_scale;
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
    GConvArgs u = args(T, nf, 2 * bh, 2 * bw, U1, nf, 2 * bh, 2 * bw, L);
    u.up2 = true;
    ops->push_back({&up1, u});
    GConvArgs u2 = args(U1, nf, 4 * bh, 4 * bw, P, nf, 4 * bh, 4 * bw, L);
    u2.up2 = true;
    ops->push_back({&up2, u2});
    ops->push_back({&hr, args(P, nf, 4 * bh, 4 * bw, Q, nf, 4 * bh, 4 * bw, L)});
    ops->push_back({&last, args(Q, nf, 4 * bh, 4 * bw, O16, 16, 4 * bh, 4 * bw, N)});
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
    if (!rd.node("/model/model.0/Conv")) return refuse(why, "not an ESRGAN graph"), nullptr;
    std::unique_ptr<Esrgan> e(new Esrgan);
    Conv c0;
    if (!rd.conv("/model/model.0/Conv", &c0, why)) return nullptr;
    if (c0.cin != 3 && c0.cin != 12) return refuse(why, "ESRGAN input channels"), nullptr;
    e->unshuffle = c0.cin == 12;
    e->nf = c0.cout;
    if (e->nf % 16 || e->nf > 128) return refuse(why, "ESRGAN feature count"), nullptr;
    // RRDBs
    int nb = 0;
    // the RDB's last conv: node conv5/conv5.0/Conv (its output is named RDBn/Mul: the export folded the 0.2 in)
    const auto c5name = [&](const std::string &pre) {
        return rd.node(pre + "conv5/conv5.0/Conv") ? pre + "conv5/conv5.0/Conv" : pre + "Mul";
    };
    while (rd.node(c5name("/model/model.1/sub/sub." + std::to_string(nb) + "/RDB1/"))) nb++;
    if (nb < 1) return refuse(why, "no RRDB blocks"), nullptr;
    e->nb = nb;
    Conv probe;
    if (!rd.conv("/model/model.1/sub/sub.0/RDB1/conv1/conv1.0/Conv", &probe, why)) return nullptr;
    e->gc = probe.cout;
    if (e->gc % 16) return refuse(why, "ESRGAN growth channels"), nullptr;
    e->DS = e->nf + 4 * e->gc;
    {
        const OnnxNode *m = rd.node("/model/model.1/sub/sub.0/Mul");
        std::vector<float> v;
        if (!m || m->in.size() != 2 || !rd.floats(m->in[1], &v) || v.size() != 1) return refuse(why, "RRDB residual scale"), nullptr;
        e->rrdb_scale = v[0];
    }
    if (!rd.node("/model/model.10/Conv") || !rd.node("/model/model.2/Resize_output_cast_0") && !rd.node("/model/model.2/Resize"))
        return refuse(why, "not the two-upsample ESRGAN layout"), nullptr;
    e->scale_ = e->unshuffle ? 2 : 4;
    // fp16 accumulation by default (what TensorRT does too); AJI_NATIVE_ACC32=1 accumulates in fp32,
    // which this network's last blocks are sensitive to (closer to the fp32 model, ~TensorRT's speed)
    if (const char *am = getenv("AJI_NATIVE_ACC32")) e->acc_mode = atoi(am) ? 1 : 0;
    // weights
    e->first.w = e->upload(c0, 1.f, 16);
    for (int i = 0; i < nb; i++)
        for (int r = 1; r <= 3; r++) {
            const std::string pre = "/model/model.1/sub/sub." + std::to_string(i) + "/RDB" + std::to_string(r) + "/";
            for (int k = 1; k <= 4; k++) {
                Conv c;
                if (!rd.conv(pre + "conv" + std::to_string(k) + "/conv" + std::to_string(k) + ".0/Conv", &c, why)) return nullptr;
                if (c.cin != e->nf + (k - 1) * e->gc || c.cout != e->gc || c.stride != 1 || !c.lrelu)
                    return refuse(why, pre + ": dense conv shape"), nullptr;
                GOp op;
                op.w = e->upload(c, 1.f, c.cin);
                e->dense.push_back(op);
            }
            Conv c5;
            if (!rd.conv(c5name(pre), &c5, why)) return nullptr;
            if (c5.cin != e->DS || c5.cout != e->nf || c5.lrelu) return refuse(why, pre + ": last conv shape"), nullptr;
            GOp op;
            op.w = e->upload(c5, r == 3 ? e->rrdb_scale : 1.f, c5.cin);
            e->dense.push_back(op);
        }
    Conv cb, cu1, cu2, chr, cl;
    if (!rd.conv("/model/model.1/sub/sub." + std::to_string(nb) + "/Conv", &cb, why) || !rd.conv("/model/model.3/Conv", &cu1, why) ||
        !rd.conv("/model/model.6/Conv", &cu2, why) || !rd.conv("/model/model.8/Conv", &chr, why) || !rd.conv("/model/model.10/Conv", &cl, why))
        return nullptr;
    if (cb.cin != e->nf || cb.cout != e->nf || cb.lrelu || !cu1.lrelu || !cu2.lrelu || !chr.lrelu || cl.lrelu || cl.cout != 3 ||
        cu1.cin != e->nf || cu2.cin != e->nf || chr.cin != e->nf || cl.cin != e->nf)
        return refuse(why, "ESRGAN upsampler shape"), nullptr;
    e->body.w = e->upload(cb, 1.f, e->nf);
    e->up1.w = e->upload(cu1, 1.f, e->nf);
    e->up2.w = e->upload(cu2, 1.f, e->nf);
    e->hr.w = e->upload(chr, 1.f, e->nf);
    e->last.w = e->upload(cl, 1.f, e->nf);
    for (GWeights *w : e->weights)
        if (!w) return refuse(why, "weight upload failed"), nullptr;

    // buffers for the largest tile
    e->maxw = max_w;
    e->maxh = max_h;
    const int bw = e->unshuffle ? max_w / 2 : max_w, bh = e->unshuffle ? max_h / 2 : max_h;
    if (bw < 1 || bh < 1) return refuse(why, "tile too small"), nullptr;
    const size_t px = (size_t)bw * bh;
    e->in16 = e->alloc(px * 16);
    e->F0 = e->alloc(px * e->nf);
    for (auto &d : e->D) d = e->alloc(px * e->DS);
    e->T = e->alloc(px * e->nf);
    e->U1 = e->alloc(px * 4 * e->nf);
    e->P = e->alloc(px * 16 * e->nf);
    e->Q = e->alloc(px * 16 * e->nf);
    e->O16 = e->alloc(px * 16 * 16);
    for (void *b : e->bufs)
        if (!b) return refuse(why, "out of GPU memory"), nullptr;
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
    char buf[128];
    snprintf(buf, sizeof(buf), "ESRGAN %dx, %d RRDB, %d features, growth %d", e->scale_, nb, e->nf, e->gc);
    e->what = buf;
    return std::unique_ptr<ImgNet>(e.release());
}

cudaError_t Esrgan::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    const int bw = unshuffle ? w / 2 : w, bh = unshuffle ? h / 2 : h;
    cudaError_t e = img_to_nhwc(in, in_f32, w, h, unshuffle, in16, 16, s);
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
    return img_from_nhwc(O16, 16, 4 * bw, 4 * bh, out, out_f32, s);
}

}  // namespace

std::unique_ptr<ImgNet> build_fdat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
std::unique_ptr<ImgNet> build_hat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
std::unique_ptr<ImgNet> build_dat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);

std::unique_ptr<ImgNet> ImgNet::create(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    if (g.inputs.size() != 1) return refuse(why, "one input expected"), nullptr;
    std::string w;
    if (auto n = Esrgan::build(g, max_w, max_h, s, &w)) return n;
    std::string w2, w3, w4;
    if (auto n = build_fdat(g, max_w, max_h, s, &w2)) return n;
    if (auto n = build_hat(g, max_w, max_h, s, &w3)) return n;
    if (auto n = build_dat(g, max_w, max_h, s, &w4)) return n;
    // the reason of the family the graph belongs to (the others say "not a ... graph")
    if (why) *why = w4.rfind("not a", 0) ? w4 : w3.rfind("not a", 0) ? w3 : w2.rfind("not a", 0) ? w2 : w;
    return nullptr;
}

}  // namespace ajinative
