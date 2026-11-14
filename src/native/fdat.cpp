// native/fdat.cpp — FDAT (traiNNer-redux fdat_arch: IllustrationJaNai's FDAT_M/XL) on native kernels.
//
// Tokens are NHWC fp16 maps at the body resolution with the channels padded to 16. Per block:
//   n1 = LN(x); qkv = n1 Wqkv; conv = GELU(dw3x3(n1))
//   spatial: a = window attention (8x8, learned bias); cm = cg(mean(conv)); x += (a Wp + bp) * cm + conv
//   channel: stats of q, k over all tokens -> per-head softmax -> folded with Wp into one matrix on the
//            GPU; a = v M + bp; x += a + conv * sigmoid(a . wsg)
//   x += dw3x3(GELU(LN(x) W1)) W2
// then the group conv (+ group input), conv_after (+ shallow features), and the transpose+conv
// upsampler (two stride-2 transposed convs as phase convs, a 3x3 conv).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "native/fdat_kernels.h"
#include "native/gconv.h"
#include "native/img_kernels.h"
#include "native/imgnet.h"
#include "native/onnx_read.h"
#include "native/tfm.h"

namespace ajinative {
namespace {

using namespace onnxread;

struct GOp {
    GWeights *w = nullptr;
    int variant = 0;
};

class Fdat : public ImgNet {
public:
    ~Fdat() override
    {
        for (auto &b : blocks) {
            tmat_release(b.qkv);
            tmat_release(b.proj);
            tmat_release(b.fc1);
            tmat_release(b.fc2);
        }
        tmat_release(foldm);
        for (GWeights *w : gw) {
            grelease(*w);
            delete w;
        }
        for (void *p : bufs) cudaFree(p);
    }
    static std::unique_ptr<ImgNet> build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override;
    int scale() const override { return unshuffle ? 2 : 4; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return bytes; }
    bool accepts(int w, int h) const override
    {
        return w > 0 && h > 0 && w <= maxw && h <= maxh && (!unshuffle || (w % 2 == 0 && h % 2 == 0));
    }

private:
    struct Block {
        bool spatial = true;
        TMat qkv, proj, fc1, fc2;
        float *ln1g = nullptr, *ln1b = nullptr, *ln2g = nullptr, *ln2b = nullptr, *dw = nullptr, *smix = nullptr;
        float *bias = nullptr, *temp = nullptr, *cg1 = nullptr, *cg2 = nullptr, *sg = nullptr, *wp = nullptr;
        float scale = 1.f, eps1 = 1e-5f, eps2 = 1e-5f;
    };
    int C = 0, CP = 0, QS = 0, HS = 0, heads = 0, hd = 0, ws = 8, cgr = 0, maxw = 0, maxh = 0, ngroups = 0, nblk = 0;
    bool unshuffle = false;
    std::vector<Block> blocks;
    std::vector<GOp> gconvs;  // group convs
    GOp first, after, up0[2], up2[2], up3;
    std::vector<GWeights *> gw;
    std::vector<void *> bufs;
    size_t bytes = 0;
    TMat foldm;
    void *IN16 = nullptr, *XS = nullptr, *G = nullptr, *X = nullptr, *N1 = nullptr, *QKV = nullptr, *CF = nullptr, *AT = nullptr;
    void *H1 = nullptr, *H2 = nullptr, *T = nullptr, *U1 = nullptr, *U2 = nullptr, *O16 = nullptr;
    float *sums = nullptr, *cm = nullptr, *stats = nullptr, *attnb = nullptr, *fold = nullptr, *ws_ = nullptr;
    size_t ws_floats = 0;
    std::string what;

    void *alloc(size_t bytes_)
    {
        void *p = nullptr;
        if (cudaMalloc(&p, bytes_) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, bytes_);
        bufs.push_back(p);
        bytes += bytes_;
        return p;
    }
    float *dfl(const std::vector<float> &v)
    {
        float *p = (float *)alloc(v.size() * 4);
        if (p) cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
        return p;
    }
    GWeights *gup(const std::vector<uint16_t> &w, const std::vector<float> &b, int cin, int cin_pad, int cout, int cout_pad)
    {
        GWeights *g = new GWeights;
        if (gupload(*g, w.data(), b.data(), cin, cin_pad, cout, cout_pad) != cudaSuccess) {
            delete g;
            return nullptr;
        }
        gw.push_back(g);
        bytes += (size_t)9 * cin_pad * cout_pad * 2;
        return g;
    }
    void gplan(int bw, int bh, std::vector<std::pair<GOp *, GConvArgs>> *ops);
};

GConvArgs ga(const void *src, int ss, int IH, int IW, void *dst, int ds, int H, int W)
{
    GConvArgs a;
    a.src = src;
    a.src_stride = ss;
    a.IH = IH;
    a.IW = IW;
    a.dst = dst;
    a.dst_stride = ds;
    a.H = H;
    a.W = W;
    return a;
}

// The gconv ops of a run, in order (the transformer steps sit between first and the group convs).
void Fdat::gplan(int bw, int bh, std::vector<std::pair<GOp *, GConvArgs>> *ops)
{
    ops->clear();
    ops->push_back({&first, ga(IN16, 16, bh, bw, XS, CP, bh, bw)});
    for (int g = 0; g < ngroups; g++) {
        GConvArgs a = ga(X, CP, bh, bw, G, CP, bh, bw);
        a.r1 = G;
        a.r1_stride = CP;
        ops->push_back({&gconvs[g], a});
    }
    GConvArgs a = ga(G, CP, bh, bw, T, CP, bh, bw);
    a.r1 = XS;
    a.r1_stride = CP;
    ops->push_back({&after, a});
    for (int py = 0; py < 2; py++) {
        GConvArgs u = ga(T, CP, bh, bw, U1, CP, bh, bw);
        u.R = 2;
        u.sy = py;
        u.act = GAct::Gelu;
        ops->push_back({&up0[py], u});
    }
    for (int py = 0; py < 2; py++) {
        GConvArgs u = ga(U1, CP, 2 * bh, 2 * bw, U2, 16, 2 * bh, 2 * bw);
        u.R = 2;
        u.sy = py;
        ops->push_back({&up2[py], u});
    }
    ops->push_back({&up3, ga(U2, 16, 4 * bh, 4 * bw, O16, 16, 4 * bh, 4 * bw)});
    for (auto &o : *ops) {
        o.second.ws = ws_;
        o.second.ws_floats = ws_floats;
    }
}

bool tensor_floats(const Reader &rd, const OnnxNode *n, int idx, std::vector<float> *v, std::vector<int64_t> *dims = nullptr)
{
    if (!n || (int)n->in.size() <= idx) return false;
    const OnnxTensor *t = rd.tensor(n->in[idx]);
    if (!t || !t->floats(v)) return false;
    if (dims) *dims = t->dims;
    return true;
}
// The initializer input of a binary op (e.g. the bias of an Add).
bool init_input(const Reader &rd, const OnnxNode *n, std::vector<float> *v, std::vector<int64_t> *dims = nullptr)
{
    if (!n) return false;
    for (size_t i = 0; i < n->in.size(); i++)
        if (rd.g.inits.count(n->in[i])) return tensor_floats(rd, n, (int)i, v, dims);
    return false;
}

std::unique_ptr<ImgNet> Fdat::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    if (!rd.node("/groups/groups.0/blocks.0/n1/LayerNormalization")) return refuse(why, "not an FDAT graph"), nullptr;
    std::unique_ptr<Fdat> f(new Fdat);
    Conv cfirst, cafter, cu0, cu2, cu3;
    if (rd.node("/conv_first/conv_first.1/Conv")) {
        if (!rd.conv("/conv_first/conv_first.1/Conv", &cfirst, why)) return nullptr;
        f->unshuffle = true;
        if (cfirst.cin != 12) return refuse(why, "FDAT unshuffle input"), nullptr;
    } else {
        if (!rd.conv("/conv_first/Conv", &cfirst, why)) return nullptr;
        if (cfirst.cin != 3) return refuse(why, "FDAT input channels"), nullptr;
    }
    f->C = cfirst.cout;
    f->CP = pad16(f->C);
    f->QS = pad16(3 * f->C);
    f->HS = pad16(2 * f->C);
    if (!rd.conv("/conv_after/Conv", &cafter, why) || !rd.conv("/upsampler/upsampler.0/ConvTranspose", &cu0, why) ||
        !rd.conv("/upsampler/upsampler.2/ConvTranspose", &cu2, why) || !rd.conv("/upsampler/upsampler.3/Conv", &cu3, why))
        return nullptr;
    if (cu0.cin != f->C || cu0.cout != f->C || cu2.cin != f->C || cu2.cout != 3 || cu3.cin != 3 || cu3.cout != 3)
        return refuse(why, "FDAT upsampler shape (expected transpose+conv)"), nullptr;
    // groups and blocks
    while (rd.node("/groups/groups." + std::to_string(f->ngroups) + "/conv/Conv")) f->ngroups++;
    while (rd.node("/groups/groups.0/blocks." + std::to_string(f->nblk) + "/n1/LayerNormalization")) f->nblk++;
    if (!f->ngroups || !f->nblk) return refuse(why, "FDAT groups"), nullptr;
    const int C = f->C;
    for (int gi = 0; gi < f->ngroups; gi++) {
        const std::string G = "/groups/groups." + std::to_string(gi) + "/";
        Conv cg;
        if (!rd.conv(G + "conv/Conv", &cg, why)) return nullptr;
        if (cg.cin != C || cg.cout != C) return refuse(why, "FDAT group conv"), nullptr;
        GOp op;
        op.w = f->gup(conv_taps(cg, nullptr), cg.b, C, f->CP, C, f->CP);
        f->gconvs.push_back(op);
        for (int bi = 0; bi < f->nblk; bi++) {
            const std::string B = G + "blocks." + std::to_string(bi) + "/";
            Block b;
            std::vector<float> v, v2;
            std::vector<int64_t> d;
            const OnnxNode *ln1 = rd.node(B + "n1/LayerNormalization"), *ln2 = rd.node(B + "n2/LayerNormalization");
            if (!tensor_floats(rd, ln1, 1, &v) || !tensor_floats(rd, ln1, 2, &v2) || (int)v.size() != C) return refuse(why, B + "n1"), nullptr;
            b.ln1g = f->dfl(v);
            b.ln1b = f->dfl(v2);
            b.eps1 = ln1->attr("epsilon") ? ln1->attr("epsilon")->f : 1e-5f;
            if (!tensor_floats(rd, ln2, 1, &v) || !tensor_floats(rd, ln2, 2, &v2)) return refuse(why, B + "n2"), nullptr;
            b.ln2g = f->dfl(v);
            b.ln2b = f->dfl(v2);
            b.eps2 = ln2->attr("epsilon") ? ln2->attr("epsilon")->f : 1e-5f;
            if (!tensor_floats(rd, rd.node(B + "attn/qkv/MatMul"), 1, &v, &d) || d != std::vector<int64_t>{C, 3 * C})
                return refuse(why, B + "qkv"), nullptr;
            tmat_upload(b.qkv, v.data(), nullptr, C, 3 * C);
            std::vector<float> wp, bp;
            if (!tensor_floats(rd, rd.node(B + "attn/proj/MatMul"), 1, &wp, &d) || d != std::vector<int64_t>{C, C} ||
                !init_input(rd, rd.node(B + "attn/proj/Add"), &bp) || (int)bp.size() != C)
                return refuse(why, B + "proj"), nullptr;
            if (!tensor_floats(rd, rd.node(B + "conv/conv.0/Conv"), 1, &v, &d) || d != std::vector<int64_t>{C, 1, 3, 3})
                return refuse(why, B + "conv"), nullptr;
            b.dw = f->dfl(v);
            // The attention type from the graph: Softmax <- Add(bias [heads,T,T]) <- MatMul <- Mul(scale) (spatial),
            // or Softmax <- Mul(temperature [heads,1,1]) (channel).
            const OnnxNode *sm = rd.node(B + "attn/Softmax");
            const OnnxNode *pre = sm && !sm->in.empty() && rd.prod.count(sm->in[0]) ? rd.prod.at(sm->in[0]) : nullptr;
            if (!pre) return refuse(why, B + "attention"), nullptr;
            if (pre->op == "Add") {
                b.spatial = true;
                if (!init_input(rd, pre, &v, &d) || d.size() != 3 || d[1] != d[2]) return refuse(why, B + "bias"), nullptr;
                const int heads = (int)d[0], T = (int)d[1], wsz = (int)std::lround(std::sqrt((double)T));
                if (wsz * wsz != T || C % heads) return refuse(why, B + "window"), nullptr;
                if ((f->heads && f->heads != heads) || (f->ws != wsz && bi + gi > 0)) return refuse(why, "FDAT mixed heads/windows"), nullptr;
                f->heads = heads;
                f->ws = wsz;
                b.bias = f->dfl(v);
                std::vector<float> sc;
                const OnnxNode *mm = nullptr, *ml = nullptr;
                for (const std::string &i : pre->in)
                    if (rd.prod.count(i) && rd.prod.at(i)->op == "MatMul") mm = rd.prod.at(i);
                if (mm)
                    for (const std::string &i : mm->in)
                        if (rd.prod.count(i) && rd.prod.at(i)->op == "Mul") ml = rd.prod.at(i);
                if (!init_input(rd, ml, &sc) || sc.size() != 1) return refuse(why, B + "scale"), nullptr;
                b.scale = sc[0];
                if (!tensor_floats(rd, rd.node(B + "inter/cg/cg.1/Conv"), 1, &v, &d) || d.size() != 4 || d[1] != C)
                    return refuse(why, B + "cg"), nullptr;
                f->cgr = (int)d[0];
                b.cg1 = f->dfl(v);
                if (!tensor_floats(rd, rd.node(B + "inter/cg/cg.3/Conv"), 1, &v, &d) || d[0] != C || d[1] != f->cgr) return refuse(why, B + "cg"), nullptr;
                b.cg2 = f->dfl(v);
                tmat_upload(b.proj, wp.data(), bp.data(), C, C);
            } else if (pre->op == "Mul" && rd.node(B + "attn/ReduceL2")) {
                b.spatial = false;
                if (!init_input(rd, pre, &v, &d) || d.size() != 3 || d[1] != 1 || d[2] != 1) return refuse(why, B + "temperature"), nullptr;
                if (f->heads && f->heads != (int)d[0]) return refuse(why, "FDAT mixed heads"), nullptr;
                f->heads = (int)d[0];
                b.temp = f->dfl(v);
                if (!tensor_floats(rd, rd.node(B + "inter/sg/sg.0/Conv"), 1, &v, &d) || d[0] != 1 || d[1] != C) return refuse(why, B + "sg"), nullptr;
                b.sg = f->dfl(v);
                b.wp = f->dfl(wp);
                if (!f->foldm.frag) {
                    float *bpd = f->dfl(bp);
                    tmat_alloc(f->foldm, C, C, bpd);
                }
                // per block bias: kept as a device copy; the fold matrix shares one buffer (blocks run in order)
                b.bias = f->dfl(bp);
            } else {
                return refuse(why, B + "attention type"), nullptr;
            }
            if (!tensor_floats(rd, rd.node(B + "ffn/fc1/MatMul"), 1, &v, &d) || d != std::vector<int64_t>{C, 2 * C}) return refuse(why, B + "fc1"), nullptr;
            tmat_upload(b.fc1, v.data(), nullptr, C, 2 * C);
            if (!tensor_floats(rd, rd.node(B + "ffn/fc2/MatMul"), 1, &v, &d) || d != std::vector<int64_t>{2 * C, C}) return refuse(why, B + "fc2"), nullptr;
            tmat_upload(b.fc2, v.data(), nullptr, 2 * C, C);
            if (!tensor_floats(rd, rd.node(B + "ffn/smix/Conv"), 1, &v, &d) || d != std::vector<int64_t>{2 * C, 1, 3, 3}) return refuse(why, B + "smix"), nullptr;
            b.smix = f->dfl(v);
            f->blocks.push_back(b);
        }
    }
    if (!f->heads || C % f->heads) return refuse(why, "FDAT heads"), nullptr;
    f->hd = C / f->heads;
    if (f->hd > 32 || f->hd % 2 || f->ws * f->ws % 16) return refuse(why, "FDAT head size / window"), nullptr;
    // gconv weights
    f->first.w = f->gup(conv_taps(cfirst, nullptr), cfirst.b, cfirst.cin, 16, C, f->CP);
    f->after.w = f->gup(conv_taps(cafter, nullptr), cafter.b, C, f->CP, C, f->CP);
    for (int py = 0; py < 2; py++) {
        std::vector<uint16_t> w;
        std::vector<float> b;
        phase_weights(cu0, 2, py, f->CP, &w, &b);
        f->up0[py].w = f->gup(w, b, C, f->CP, 2 * f->CP, 2 * f->CP);
        phase_weights(cu2, 2, py, 16, &w, &b);
        f->up2[py].w = f->gup(w, b, C, f->CP, 32, 32);
    }
    f->up3.w = f->gup(conv_taps(cu3, nullptr), cu3.b, 3, 16, 3, 16);
    for (GWeights *w : f->gw)
        if (!w) return refuse(why, "weight upload failed"), nullptr;
    for (auto &b : f->blocks)
        if (!b.qkv.frag || !b.fc1.frag || !b.fc2.frag || (b.spatial && !b.proj.frag)) return refuse(why, "weight upload failed"), nullptr;

    // buffers for the largest tile
    f->maxw = max_w;
    f->maxh = max_h;
    const int bw = f->unshuffle ? max_w / 2 : max_w, bh = f->unshuffle ? max_h / 2 : max_h;
    const size_t M = (size_t)bw * bh, CPb = (size_t)f->CP * 2;
    f->IN16 = f->alloc(M * 32);
    f->XS = f->alloc(M * CPb);
    f->G = f->alloc(M * CPb);
    f->X = f->alloc(M * CPb);
    f->N1 = f->alloc(M * CPb);
    f->CF = f->alloc(M * CPb);
    f->AT = f->alloc(M * CPb);
    f->T = f->alloc(M * CPb);
    f->QKV = f->alloc(M * f->QS * 2);
    f->H1 = f->alloc(M * f->HS * 2);
    f->H2 = f->alloc(M * f->HS * 2);
    f->U1 = f->alloc(M * 4 * CPb);
    f->U2 = f->alloc(M * 16 * 32);
    f->O16 = f->alloc(M * 16 * 32);
    f->sums = (float *)f->alloc(f->CP * 4);
    f->cm = (float *)f->alloc(f->CP * 4);
    f->stats = (float *)f->alloc((size_t)(f->heads * f->hd * f->hd + 2 * C) * 4);
    f->attnb = (float *)f->alloc((size_t)f->heads * f->hd * f->hd * 4);
    f->fold = (float *)f->alloc((size_t)C * C * 4);
    for (void *p : f->bufs)
        if (!p) return refuse(why, "out of GPU memory"), nullptr;
    // gconv split-K workspace and tile variants
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        f->gplan(sz ? std::min(bw, 32) : bw, sz ? std::min(bh, 32) : bh, &ops);
        for (auto &o : ops) f->ws_floats = std::max(f->ws_floats, gworkspace(*o.first->w, o.second));
    }
    if (f->ws_floats) {
        f->ws_ = (float *)f->alloc(f->ws_floats * 4);
        if (!f->ws_) return refuse(why, "out of GPU memory"), nullptr;
    }
    f->gplan(bw, bh, &ops);
    std::map<std::tuple<int, int, int, int, int, int>, int> picked;
    for (auto &o : ops) {
        std::string w2;
        if (!gsupported(*o.first->w, o.second, &w2)) return refuse(why, "kernel: " + w2), nullptr;
        const auto key = std::make_tuple(o.first->w->cin_pad, o.first->w->cout_pad, o.second.H, o.second.W, o.second.R, o.second.r1 != nullptr);
        auto it = picked.find(key);
        if (it == picked.end()) it = picked.emplace(key, gpick(*o.first->w, o.second, s)).first;
        if (it->second < 0) return refuse(why, "kernel timing failed"), nullptr;
        o.first->variant = it->second;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    char buf[160];
    snprintf(buf, sizeof(buf), "FDAT %dx, %d groups x %d blocks, %d channels, %d heads, %dx%d windows", f->scale(), f->ngroups, f->nblk, C,
             f->heads, f->ws, f->ws);
    f->what = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

cudaError_t Fdat::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    const int bw = unshuffle ? w / 2 : w, bh = unshuffle ? h / 2 : h, M = bw * bh;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    gplan(bw, bh, &ops);
    size_t oi = 0;
    auto gnext = [&]() { auto &o = ops[oi++]; return glaunch(*o.first->w, o.second, o.first->variant, s); };
    // AJI_NATIVE_PROFILE: time per step kind (events between steps; the first run of a size only).
    static const bool prof = std::getenv("AJI_NATIVE_PROFILE") != nullptr;
    static const char *const kinds[] = {"io", "conv3x3", "layernorm", "qkv", "dwconv", "winattn", "gate", "proj",
                                        "chanstats", "chanfold", "chanproj", "fc1", "smix", "fc2"};
    std::vector<std::pair<int, cudaEvent_t>> evs;
    auto mark = [&](int k) {
        if (!prof) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        evs.push_back({k, e});
    };
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return e_; } while (0)
    mark(0);
    CK(img_to_nhwc(in, in_f32, w, h, unshuffle, IN16, 16, s));
    CK(gnext());  // first -> XS mark(1);
    CK(cudaMemcpyAsync(G, XS, (size_t)M * CP * 2, cudaMemcpyDeviceToDevice, s));
    for (int gi = 0; gi < ngroups; gi++) {
        CK(cudaMemcpyAsync(X, G, (size_t)M * CP * 2, cudaMemcpyDeviceToDevice, s));
        for (int bi = 0; bi < nblk; bi++) {
            const Block &b = blocks[(size_t)gi * nblk + bi];
            CK(tlayernorm(X, CP, N1, CP, M, C, CP, b.ln1g, b.ln1b, b.eps1, s)); mark(2);
            TGemm q;
            q.a = N1; q.sa = CP; q.M = M; q.out = QKV; q.so = QS;
            CK(tgemm(b.qkv, q, s)); mark(3);
            if (b.spatial) CK(cudaMemsetAsync(sums, 0, CP * 4, s));
            CK(tdwconv(N1, CP, CF, CP, bh, bw, C, b.dw, true, b.spatial ? sums : nullptr, s)); mark(4);
            if (b.spatial) {
                TWinAttn a;
                a.qkv = QKV; a.sq = QS; a.qo = 0; a.ko = C; a.vo = 2 * C; a.H = bh; a.W = bw; a.heads = heads; a.hd = hd;
                a.wh = a.ww = ws; a.scale = b.scale; a.bias = b.bias; a.out = AT; a.so = CP;
                CK(twinattn(a, s)); mark(5);
                CK(fdat_aim_cg(sums, M, C, cgr, b.cg1, b.cg2, cm, CP, s)); mark(6);
                TGemm pj;
                pj.a = AT; pj.sa = CP; pj.M = M; pj.out = X; pj.so = CP; pj.colscale = cm;
                pj.add1 = CF; pj.s1 = CP; pj.add2 = X; pj.s2 = CP;
                CK(tgemm(b.proj, pj, s)); mark(7);
            } else {
                CK(cudaMemsetAsync(stats, 0, (size_t)(heads * hd * hd + 2 * C) * 4, s));
                CK(tchanstats(QKV, QS, 0, C, M, heads, hd, stats, s)); mark(8);
                CK(fdat_chan_soft(stats, b.temp, heads, hd, 1.1920928955078125e-07f, attnb, s));
                CK(fdat_chan_fold(attnb, b.wp, heads, hd, fold, s));
                CK(tmat_pack(foldm, fold, s));
                CK(cudaMemcpyAsync(foldm.bias, b.bias, (size_t)C * 4, cudaMemcpyDeviceToDevice, s)); mark(9);
                TGemm pj;
                pj.a = (const uint16_t *)QKV + 2 * C; pj.sa = QS; pj.M = M; pj.out = AT; pj.so = CP;
                CK(tgemm(foldm, pj, s)); mark(10);
                CK(fdat_aim_sg(AT, CF, b.sg, X, CP, M, C, s)); mark(6);
            }
            CK(tlayernorm(X, CP, N1, CP, M, C, CP, b.ln2g, b.ln2b, b.eps2, s)); mark(2);
            TGemm f1;
            f1.a = N1; f1.sa = CP; f1.M = M; f1.out = H1; f1.so = HS; f1.act = TAct::Gelu;
            CK(tgemm(b.fc1, f1, s)); mark(11);
            CK(tdwconv(H1, HS, H2, HS, bh, bw, 2 * C, b.smix, false, s)); mark(12);
            TGemm f2;
            f2.a = H2; f2.sa = HS; f2.M = M; f2.out = X; f2.so = CP; f2.add1 = X; f2.s1 = CP;
            CK(tgemm(b.fc2, f2, s)); mark(13);
        }
        CK(gnext());  // group conv: G = conv(X) + G mark(1);
    }
    while (oi < ops.size()) { CK(gnext()); mark(1); }  // conv_after, upsampler
    CK(img_from_nhwc(O16, 16, 4 * bw, 4 * bh, out, out_f32, s));
    mark(0);
    if (prof) {
        cudaStreamSynchronize(s);
        double t[16] = {}, total = 0;
        for (size_t i = 1; i < evs.size(); i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, evs[i - 1].second, evs[i].second);
            t[evs[i].first] += ms;
            total += ms;
        }
        for (auto &e : evs) cudaEventDestroy(e.second);
        fprintf(stderr, "fdat %dx%d: %.3f ms:", w, h, total);
        for (int k = 0; k < 14; k++)
            if (t[k] > 0) fprintf(stderr, " %s %.3f", kinds[k], t[k]);
        fprintf(stderr, "\n");
    }
#undef CK
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<ImgNet> build_fdat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    return Fdat::build(g, max_w, max_h, s, why);
}

}  // namespace ajinative
