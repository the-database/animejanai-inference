// native/sst.cpp — SST (Scalable SR Transformer, "Rank-Factorized Implicit Neural Bias", arXiv
// 2603.06738; the export of experimental/native-sst/sst_ref.py) on the tfm, gconv and sst kernels.
//
// proj (3x3 3 -> C), then n blocks of L layers: x += to_out(attn(LN x) * sigmoid(gate(LN x)));
// x += agg(h + gelu(dw(h))), h = gelu(proj(LN x)); the block conv plus the block input. The
// attention runs in windows of each layer's size (16..96, reflect-padded to the window multiple)
// with the RIB position factors as extra q / k dims (tables evaluated here from the graph's small
// MLP). Then LN, a 3x3 conv plus proj's output, and the upsampler: 3x3 C -> F + LeakyReLU, 3x3
// F -> 16F + depth-to-space 4 (4 row-phase convs), 3x3 F -> 3, plus the input upsampled nearest.
// The residual stream is fp32 (as HAT's: bf16-trained models); the rest fp16 with fp32 sums.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <vector>

#include "native/img_kernels.h"
#include "native/imgnet.h"
#include "native/netres.h"
#include "native/sst_kernels.h"
#include "native/static_eval.h"

namespace ajinative {
namespace {

using namespace onnxread;

constexpr float kLog2e = 1.4426950408889634f;

struct Ln {
    float *g = nullptr, *b = nullptr, eps = 1e-6f;
};
struct Layer {
    Ln n1, n2;
    TMat *qkv = nullptr, *gate = nullptr, *out = nullptr, *fc1 = nullptr, *fc2 = nullptr;
    float *gdw = nullptr, *gdwb = nullptr, *fdw = nullptr, *fdwb = nullptr;
    void *qpos = nullptr, *kpos = nullptr;
    int ws = 0, rp = 0;
};
struct Block {
    std::vector<Layer> layers;
    GOp conv;
};

class Sst : public ImgNet {
public:
    static std::unique_ptr<ImgNet> build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override;
    int scale() const override { return 4; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes; }
    size_t weight_bytes() const override { return res.weight_bytes; }
    bool accepts(int w, int h) const override
    {
        if (w < 1 || h < 1 || w > maxw || h > maxh) return false;
        for (int ws : wins)  // one reflection: the pad to the window multiple must stay inside the tile
            if ((ws - w % ws) % ws > w - 1 || (ws - h % ws) % ws > h - 1) return false;
        return true;
    }

private:
    DevRes res;
    int C = 0, CP = 0, heads = 0, hd = 0, HS = 0, HSP = 0, QS = 0, FP = 0, FS = 0, maxw = 0, maxh = 0;
    float slope = 0.1f;
    std::set<int> wins;
    Ln fin;
    std::vector<Block> blocks;
    GOp first, after, feat, up[4], last;
    void *IN16 = nullptr, *XF = nullptr, *N1 = nullptr, *GD = nullptr, *GL = nullptr, *QKV = nullptr, *AT = nullptr, *H1 = nullptr,
         *H2 = nullptr, *S16 = nullptr, *TT = nullptr, *U0 = nullptr, *V = nullptr, *O16 = nullptr;
    float *S = nullptr, *GIN = nullptr, *wsp = nullptr;
    size_t ws_floats = 0;
    std::string what;

    cudaError_t exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
                     std::vector<std::pair<GOp *, GConvArgs>> *collect);
};

const OnnxNode *consumer(const OnnxGraph &g, const std::string &t, const char *op)
{
    for (const OnnxNode &n : g.nodes)
        if (n.op == op)
            for (const std::string &i : n.in)
                if (i == t) return &n;
    return nullptr;
}

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

// A 1x1 Conv K -> N as GEMM weights w[k][n] and its bias.
bool c1x1(const Reader &rd, const std::string &name, int K, int N, std::vector<float> *w, std::vector<float> *b, std::string *why)
{
    const OnnxNode *n = rd.node(name);
    std::vector<float> v;
    std::vector<int64_t> d;
    if (!n || n->op != "Conv" || n->attr_i("group", 1) != 1 || !tensor_floats(rd, n, 1, &v, &d) || d != std::vector<int64_t>{N, K, 1, 1} ||
        !tensor_floats(rd, n, 2, b) || (int)b->size() != N)
        return refuse(why, "missing 1x1 conv " + name);
    w->assign((size_t)K * N, 0.f);
    for (int o = 0; o < N; o++)
        for (int i = 0; i < K; i++) (*w)[(size_t)i * N + o] = v[(size_t)o * K + i];
    return true;
}

TMat *mat1x1(const Reader &rd, DevRes &r, const std::string &name, int K, int N, std::string *why)
{
    std::vector<float> w, b;
    if (!c1x1(rd, name, K, N, &w, &b, why)) return nullptr;
    return r.mat(w, &b, K, N);
}

// A depthwise 3x3 Conv over C channels (bias), padded with zeros to CP: w [CP][9], b [CP].
bool dw3(const Reader &rd, DevRes &r, const std::string &name, int C, int CP, float **w, float **b, std::string *why)
{
    const OnnxNode *n = rd.node(name);
    std::vector<float> v, bb;
    std::vector<int64_t> d;
    if (!n || n->op != "Conv" || n->attr_i("group", 1) != C || !tensor_floats(rd, n, 1, &v, &d) || d != std::vector<int64_t>{C, 1, 3, 3} ||
        !tensor_floats(rd, n, 2, &bb) || (int)bb.size() != C)
        return refuse(why, "missing depthwise conv " + name);
    v.resize((size_t)CP * 9, 0.f);
    bb.resize(CP, 0.f);
    *w = r.dfl(v, (size_t)CP * 9);
    *b = r.dfl(bb, CP);
    return true;
}

// The RIB tables of one layer from the graph's MLP: hid = relu(coords @ W1 + b1); qpos[h] = hid @ Wq[h]
// times the graph's q scale (rank^-0.5), kpos[h] = hid @ Wk[h]; both times log2(e) on the q side, padded
// to rp dims, fp16 [heads][T][rp].
bool rib(const Reader &rd, DevRes &r, const std::string &A, int heads, Layer *L, std::string *why)
{
    const OnnxNode *mm = rd.node(A + "MatMul");
    std::vector<float> coords, w1, b1, wq, wk, sc;
    std::vector<int64_t> dc, d1, dq, dk;
    if (!tensor_floats(rd, mm, 0, &coords, &dc) || dc.size() != 2 || !tensor_floats(rd, mm, 1, &w1, &d1) || d1.size() != 2 || d1[0] != dc[1])
        return refuse(why, A + ": RIB coordinates");
    const int T = (int)dc[0], F = (int)dc[1], Hd = (int)d1[1];
    const OnnxNode *add = consumer(rd.g, mm->out[0], "Add");
    const OnnxNode *relu = add ? consumer(rd.g, add->out[0], "Relu") : nullptr;
    if (!relu || !init_input(rd, add, &b1) || (int)b1.size() != Hd) return refuse(why, A + ": RIB hidden layer");
    const OnnxNode *eq = nullptr, *ek = nullptr;
    float qscale = 0.f;
    for (const OnnxNode &n : rd.g.nodes) {
        if (n.op != "Einsum" || n.in.empty() || n.in[0] != relu->out[0]) continue;
        const OnnxNode *mul = consumer(rd.g, n.out[0], "Mul");
        if (mul && init_input(rd, mul, &sc) && sc.size() == 1) {
            eq = &n;
            qscale = sc[0];
        } else {
            ek = &n;
        }
    }
    if (!eq || !ek || !tensor_floats(rd, eq, 1, &wq, &dq) || !tensor_floats(rd, ek, 1, &wk, &dk) || dq.size() != 3 || dq != dk ||
        dq[0] != heads || dq[1] != Hd)
        return refuse(why, A + ": RIB projections");
    const int R = (int)dq[2], ws = (int)std::lround(std::sqrt((double)T));
    if (ws * ws != T || ws < 16 || ws % 16 || R < 1 || R > 48) return refuse(why, A + ": RIB window / rank");
    L->ws = ws;
    L->rp = R <= 16 ? 16 : R <= 32 ? 32 : 48;
    std::vector<float> hid((size_t)T * Hd);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < Hd; j++) {
            double a = b1[j];
            for (int f = 0; f < F; f++) a += (double)coords[(size_t)t * F + f] * w1[(size_t)f * Hd + j];
            hid[(size_t)t * Hd + j] = a > 0 ? (float)a : 0.f;
        }
    std::vector<uint16_t> tq((size_t)heads * T * L->rp, 0), tk(tq.size(), 0);
    for (int h = 0; h < heads; h++)
        for (int t = 0; t < T; t++)
            for (int k = 0; k < R; k++) {
                double aq = 0, ak = 0;
                for (int j = 0; j < Hd; j++) {
                    aq += (double)hid[(size_t)t * Hd + j] * wq[((size_t)h * Hd + j) * R + k];
                    ak += (double)hid[(size_t)t * Hd + j] * wk[((size_t)h * Hd + j) * R + k];
                }
                tq[((size_t)h * T + t) * L->rp + k] = hb((float)(aq * qscale * kLog2e));
                tk[((size_t)h * T + t) * L->rp + k] = hb((float)ak);
            }
    L->qpos = r.alloc(tq.size() * 2);
    L->kpos = r.alloc(tk.size() * 2);
    if (!L->qpos || !L->kpos) return refuse(why, "out of GPU memory");
    r.weight_bytes += tq.size() * 4;
    cudaMemcpy(L->qpos, tq.data(), tq.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(L->kpos, tk.data(), tk.size() * 2, cudaMemcpyHostToDevice);
    return true;
}

std::unique_ptr<ImgNet> Sst::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    if (!rd.node("/body/body.0/body.0.0/attn/to_qkv/Conv") || !rd.node("/upsampler/up_conv/Conv")) return refuse(why, "not an SST graph"), nullptr;
    std::unique_ptr<Sst> f(new Sst);
    DevRes &r = f->res;
    Conv c;
    if (!rd.conv("/proj/Conv", &c, why)) return nullptr;
    if (c.cin != 3 || c.lrelu) return refuse(why, "SST proj"), nullptr;
    const int C = f->C = c.cout, CP = f->CP = pad16(C);
    f->first.w = r.gup(conv_taps(c, nullptr), c.b, 3, 16, C, CP);
    int nb = 0, nl = 0;
    while (rd.node("/body/body." + std::to_string(nb) + "/body." + std::to_string(nb) + ".0/attn/to_qkv/Conv")) nb++;
    while (rd.node("/body/body.0/body.0." + std::to_string(nl) + "/attn/to_qkv/Conv")) nl++;
    if (!nb || !nl) return refuse(why, "SST blocks"), nullptr;
    {
        std::vector<float> v;
        std::vector<int64_t> d;
        if (!tensor_floats(rd, rd.node("/body/body.0/body.0.0/ffn/proj/Conv"), 1, &v, &d) || d.size() != 4 || d[1] != C) return refuse(why, "SST ffn"), nullptr;
        f->HS = (int)d[0];
        f->HSP = pad16(f->HS);
        if (!tensor_floats(rd, rd.node("/body/body.0/body.0.0/attn/MatMul"), 1, &v, &d) || d.size() != 2) return refuse(why, "SST RIB"), nullptr;
        const OnnxNode *e = nullptr;
        for (const OnnxNode &n : g.nodes)
            if (n.op == "Einsum" && n.name.rfind("/body/body.0/body.0.0/attn/", 0) == 0) e = &n;
        std::vector<int64_t> de;
        if (!e || !tensor_floats(rd, e, 1, &v, &de) || de.size() != 3) return refuse(why, "SST heads"), nullptr;
        f->heads = (int)de[0];
    }
    const int heads = f->heads;
    if (heads < 1 || C % heads || C / heads > 32 || (C / heads) % 2) return refuse(why, "SST head size"), nullptr;
    f->hd = C / heads;
    f->QS = 3 * heads * 32;
    const int hd = f->hd, HS = f->HS, HSP = f->HSP, QS = f->QS;
    const float qs = 1.f / std::sqrt((float)hd) * kLog2e;
    for (int b = 0; b < nb; b++) {
        Block bl;
        const std::string B = "/body/body." + std::to_string(b) + "/body." + std::to_string(b) + ".";
        for (int l = 0; l < nl; l++) {
            const std::string P = B + std::to_string(l) + "/", A = P + "attn/";
            Layer L;
            if (!ln(rd, r, P + "norm_attn/LayerNormalization", C, &L.n1, why) || !ln(rd, r, P + "norm_ffn/LayerNormalization", C, &L.n2, why) ||
                !rib(rd, r, A, heads, &L, why) || !dw3(rd, r, A + "gate/gate.0/Conv", C, CP, &L.gdw, &L.gdwb, why) ||
                !dw3(rd, r, P + "ffn/dwc/Conv", HS, HSP, &L.fdw, &L.fdwb, why))
                return nullptr;
            if (!rd.node(A + "gate/gate.2/Sigmoid") || !rd.node(A + "Softmax")) return refuse(why, A + ": gate / softmax"), nullptr;
            // the attention input is reflect-padded to the window multiple (no Pad when the export's size divided it)
            for (const OnnxNode &n : g.nodes)
                if (n.op == "Pad" && n.name.rfind(A, 0) == 0) {
                    const OnnxAttr *m = n.attr("mode");
                    if (!m || m->s != "reflect") return refuse(why, A + ": padding mode"), nullptr;
                }
            std::vector<float> w, bias;
            if (!c1x1(rd, A + "to_qkv/Conv", C, 3 * C, &w, &bias, why)) return nullptr;
            std::vector<float> w2((size_t)C * QS, 0.f), b2(QS, 0.f);  // each head's q / k / v at 32-dim slots, q scaled
            for (int part = 0; part < 3; part++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < hd; d++) {
                        const int src = part * C + h * hd + d, dst = part * heads * 32 + h * 32 + d;
                        const float sc = part == 0 ? qs : 1.f;
                        for (int k = 0; k < C; k++) w2[(size_t)k * QS + dst] = w[(size_t)k * 3 * C + src] * sc;
                        b2[dst] = bias[src] * sc;
                    }
            L.qkv = r.mat(w2, &b2, C, QS);
            if (!(L.gate = mat1x1(rd, r, A + "gate/gate.1/Conv", C, C, why)) || !(L.out = mat1x1(rd, r, A + "to_out/Conv", C, C, why)) ||
                !(L.fc1 = mat1x1(rd, r, P + "ffn/proj/Conv", C, HS, why)) || !(L.fc2 = mat1x1(rd, r, P + "ffn/agg/Conv", HS, C, why)))
                return nullptr;
            f->wins.insert(L.ws);
            bl.layers.push_back(L);
        }
        if (!rd.conv(B + std::to_string(nl) + "/Conv", &c, why)) return nullptr;
        if (c.cin != C || c.cout != C || c.lrelu) return refuse(why, "SST block conv"), nullptr;
        bl.conv.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, C, CP);
        f->blocks.push_back(std::move(bl));
    }
    const std::string E = "/body/body." + std::to_string(nb);
    if (!ln(rd, r, E + "/LayerNormalization", C, &f->fin, why)) return nullptr;
    if (!rd.conv("/body/body." + std::to_string(nb + 1) + "/Conv", &c, why)) return nullptr;
    if (c.cin != C || c.cout != C || c.lrelu) return refuse(why, "SST body conv"), nullptr;
    f->after.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, C, CP);
    if (!rd.conv("/upsampler/feature_conv/Conv", &c, why)) return nullptr;
    if (!c.lrelu || c.cin != C) return refuse(why, "SST feature conv"), nullptr;
    const int F = c.cout, FP = f->FP = pad16(F), FS = f->FS = (FP % 64 == 0 || 64 % FP == 0) ? FP : (FP + 63) / 64 * 64;  // FS: a phase group fills whole 64-channel launches
    f->slope = c.slope;
    f->feat.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, F, FP);
    if (!rd.conv("/upsampler/up_conv/Conv", &c, why)) return nullptr;
    const OnnxNode *d2s = rd.node("/upsampler/DepthToSpace");
    const OnnxAttr *mode = d2s ? d2s->attr("mode") : nullptr;
    if (c.cin != F || c.cout != 16 * F || !d2s || d2s->attr_i("blocksize", 0) != 4 || (mode && mode->s != "CRD"))
        return refuse(why, "SST upsampler (4x pixel shuffle)"), nullptr;
    for (int sy = 0; sy < 4; sy++) {  // DepthToSpace(4, CRD): output channel ch*16 + sy*4 + sx -> launch sy, channel sx*FS + ch
        std::vector<uint16_t> wt((size_t)9 * F * 4 * FS, 0);
        std::vector<float> bt(4 * FS, 0.f);
        for (int sx = 0; sx < 4; sx++)
            for (int ch = 0; ch < F; ch++) {
                const int o = ch * 16 + sy * 4 + sx;
                bt[sx * FS + ch] = c.b[o];
                for (int t = 0; t < 9; t++)
                    for (int i = 0; i < F; i++) wt[((size_t)t * F + i) * 4 * FS + sx * FS + ch] = hb(c.w[((size_t)o * F + i) * 9 + t]);
            }
        f->up[sy].w = r.gup(wt, bt, F, FP, 4 * FS, 4 * FS);
    }
    if (!rd.conv("/upsampler/final_conv/Conv", &c, why)) return nullptr;
    const OnnxNode *rz = rd.node("/upsampler/Resize");
    if (c.cin != F || c.cout != 3 || !rz) return refuse(why, "SST final conv"), nullptr;
    if (!nearest_replicates(*rz, 4) || rz->in.empty() || rz->in[0] != g.inputs[0] ||
        !static_eval::resize_by(rd, *rz, 4, 3, [&](const std::string &t) { return t == g.inputs[0]; }))
        return refuse(why, "SST input skip is not a nearest 4x replication of the input"), nullptr;
    f->last.w = r.gup(conv_taps(c, nullptr), c.b, F, FP, 3, 16);
    if (r.failed) return refuse(why, "weight upload failed"), nullptr;

    f->maxw = max_w;
    f->maxh = max_h;
    const size_t M = (size_t)max_w * max_h;
    // Steps (kept in step with exec()): 1 IN16, 2 XF, 3 S; per block 9 GIN; per layer 10 N1, 11 GD,
    // 12 GL, 13 QKV, 14 AT, 15 S, 16 N1, 17 H1, 18 H2, 19 S; 27 S16, 28 S; then 30 N1, 31 TT, 32 U0,
    // 33 V, 34 O16, 35 out.
    Lifetimes lt;
    lt.add(&f->IN16, M * 32, {{1, 35}});
    lt.add(&f->XF, M * CP * 2, {{2, 31}});
    lt.add((void **)&f->S, M * CP * 4, {{3, 30}});
    lt.add((void **)&f->GIN, M * CP * 4, {{9, 28}});
    lt.add(&f->N1, M * CP * 2, {{10, 13}, {16, 17}, {30, 31}});
    lt.add(&f->GD, M * CP * 2, {{11, 12}});
    lt.add(&f->GL, M * CP * 2, {{12, 14}});
    lt.add(&f->QKV, M * QS * 2, {{13, 14}});
    lt.add(&f->AT, M * CP * 2, {{14, 15}});
    lt.add(&f->H1, M * HSP * 2, {{17, 18}});
    lt.add(&f->H2, M * HSP * 2, {{18, 19}});
    lt.add(&f->S16, M * CP * 2, {{27, 28}});
    lt.add(&f->TT, M * CP * 2, {{31, 32}});
    lt.add(&f->U0, M * FP * 2, {{32, 33}});
    lt.add(&f->V, 16 * M * FS * 2, {{33, 34}});
    lt.add(&f->O16, 16 * M * 16 * 2, {{34, 35}});
    if (!lt.assign(r)) return refuse(why, "out of GPU memory"), nullptr;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        ops.clear();
        const int w = sz ? std::min(max_w, 64) : max_w, h = sz ? std::min(max_h, 64) : max_h;
        f->exec(nullptr, false, nullptr, false, w, h, s, &ops);
        f->ws_floats = std::max(f->ws_floats, gworkspace_max(ops));
    }
    if (f->ws_floats && !(f->wsp = (float *)r.alloc(f->ws_floats * 4))) return refuse(why, "out of GPU memory"), nullptr;
    ops.clear();
    f->exec(nullptr, false, nullptr, false, max_w, max_h, s, &ops);
    if (!gprepare(ops, s, why)) return nullptr;
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    std::string wl;
    for (const Layer &L : f->blocks[0].layers) wl += (wl.empty() ? "" : "/") + std::to_string(L.ws);
    char buf[240];
    snprintf(buf, sizeof(buf), "SST 4x, %d blocks x %d layers, %d channels, %d heads, windows %s, RIB, fp32 stream", nb, nl, C, heads,
             wl.c_str());
    f->what = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

cudaError_t Sst::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    return exec(in, in_f32, out, out_f32, w, h, s, nullptr);
}

cudaError_t Sst::exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
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
    auto gemm = [&](const TMat &m, const void *a, int sa, void *o, int so, TAct act, bool f32_stream) -> cudaError_t {
        if (collect) return cudaSuccess;
        TGemm gm;
        gm.a = a; gm.sa = sa; gm.M = M; gm.out = o; gm.so = so; gm.act = act;
        if (f32_stream) { gm.out_f32 = true; gm.add1 = S; gm.s1 = CP; gm.add1_f32 = true; }
        return tgemm(m, gm, s);
    };
    auto lnorm = [&](const Ln &l, const void *x, bool xf32, void *y, bool yf32) -> cudaError_t {
        if (collect) return cudaSuccess;
        return tlayernorm2(x, xf32, CP, y, yf32, CP, M, C, CP, l.g, l.b, l.eps, s);
    };

    K(img_to_nhwc(in, in_f32, w, h, false, IN16, 16, s));
    pr.mark(P_IO);
    CK(Gc(first, ga(IN16, 16, h, w, XF, CP, h, w)));
    K(tconvert(XF, false, CP, S, true, CP, M, CP, s));
    pr.mark(P_CONV);
    for (Block &bl : blocks) {
        K(cudaMemcpyAsync(GIN, S, (size_t)M * CP * 4, cudaMemcpyDeviceToDevice, s));
        for (Layer &L : bl.layers) {
            CK(lnorm(L.n1, S, true, N1, false));
            pr.mark(P_LN);
            K(tdwconv2(N1, CP, GD, CP, h, w, CP, L.gdw, L.gdwb, false, nullptr, 0, nullptr, s));
            pr.mark(P_DW);
            CK(gemm(*L.gate, GD, CP, GL, CP, TAct::None, false));
            pr.mark(P_GATE);
            CK(gemm(*L.qkv, N1, CP, QKV, QS, TAct::None, false));
            pr.mark(P_QKV);
            if (!collect) {
                TRibAttn t;
                t.qkv = QKV; t.sq = QS; t.kq = heads * 32; t.vq = 2 * heads * 32; t.H = h; t.W = w; t.heads = heads; t.hd = hd;
                t.ws = L.ws; t.qpos = L.qpos; t.kpos = L.kpos; t.rp = L.rp; t.gate = GL; t.sg = CP; t.out = AT; t.so = CP;
                CK(tribattn(t, s));
            }
            pr.mark(P_WIN);
            CK(gemm(*L.out, AT, CP, S, CP, TAct::None, true));
            pr.mark(P_PROJ);
            CK(lnorm(L.n2, S, true, N1, false));
            pr.mark(P_LN);
            CK(gemm(*L.fc1, N1, CP, H1, HSP, TAct::Gelu, false));
            pr.mark(P_FC1);
            K(tdwres(H1, HSP, H2, HSP, h, w, HSP, L.fdw, L.fdwb, s));
            pr.mark(P_DW);
            CK(gemm(*L.fc2, H2, HSP, S, CP, TAct::None, true));
            pr.mark(P_FC2);
        }
        // the block conv plus the block input, in fp32
        K(tconvert(S, true, CP, S16, false, CP, M, CP, s));
        GConvArgs a = ga(S16, CP, h, w, S, CP, h, w);
        a.acc32 = true;
        a.out_f32 = true;
        a.r1f = GIN;
        a.r1f_stride = CP;
        CK(Gc(bl.conv, a));
        pr.mark(P_CONV);
    }
    CK(lnorm(fin, S, true, N1, false));
    GConvArgs a = ga(N1, CP, h, w, TT, CP, h, w);
    a.r1 = XF;
    a.r1_stride = CP;
    CK(Gc(after, a));
    GConvArgs b = ga(TT, CP, h, w, U0, FP, h, w);
    b.act = GAct::LRelu;
    b.slope = slope;
    CK(Gc(feat, b));
    for (int sy = 0; sy < 4; sy++) {
        GConvArgs u = ga(U0, FP, h, w, V, FS, h, w);
        u.R = 4;
        u.sy = sy;
        CK(Gc(up[sy], u));
    }
    CK(Gc(last, ga(V, FS, 4 * h, 4 * w, O16, 16, 4 * h, 4 * w)));
    pr.mark(P_UP);
    K(sst_out(O16, 16, IN16, 16, w, h, out, out_f32, s));
    pr.mark(P_IO);
#undef K
#undef CK
    if (!collect) {
        char label[48];
        snprintf(label, sizeof(label), "sst %dx%d", w, h);
        pr.report(label);
    }
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<ImgNet> build_sst(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    return Sst::build(g, max_w, max_h, s, why);
}

}  // namespace ajinative
