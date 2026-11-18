// native/dat.cpp — DAT (Dual Aggregation Transformer; IllustrationJaNai's DAT-2 4x, bf16 exports) on the
// tfm and gconv kernels.
//
// Spatial blocks: window attention in two channel halves (8x32 and 32x8 windows, shifted by half a
// window with Swin's mask in every second spatial block of a group pair, zero-padded to the windows'
// multiple), the bias from the dynamic position bias MLP (evaluated at load from the graph's constant
// subgraph); channel blocks: channel attention over all tokens. Both add a depthwise conv on v and the
// adaptive interaction (a channel gate from one branch's average, a per-token gate from the other)
// before the projection, then the SGFN (fc1, GELU, x1 * dwconv(LN(x2)), fc2). Residual groups end in a
// 3x3 conv plus the group input; the stream is fp32 (it outgrows fp16 in the last group). The dynamo
// exports carry no module names: the parser walks the data-path ops in order.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "native/dat_kernels.h"
#include "native/fdat_kernels.h"
#include "native/img_kernels.h"
#include "native/export_forms.h"
#include "native/imgnet.h"
#include "native/netres.h"
#include "native/static_eval.h"

namespace ajinative {
namespace {

using namespace onnxread;

// ---- the model
struct Ln {
    float *g = nullptr, *b = nullptr, eps = 1e-5f;
};
struct Branch {  // one half of a spatial block's attention
    int wh = 0, ww = 0, shift_y = 0, shift_x = 0, heads = 0, rt_w = 0, rt_offy = 0, rt_offx = 0;
    bool mask = false;
    float scale = 1.f, *rtab = nullptr;
};
struct Block {
    bool spatial = true;
    Ln n1, n2, n3;  // n3: the SGFN's norm of x2
    TMat *qkv = nullptr, *proj = nullptr, *fc1 = nullptr, *fc2 = nullptr;
    Branch br[2];
    float *temp = nullptr, clip = 1e-12f;            // channel attention
    float *dw = nullptr, *dwb = nullptr;             // the depthwise conv on v (C rounded up to 8)
    float *cw1 = nullptr, *cb1 = nullptr, *cw2 = nullptr, *cb2 = nullptr;  // channel gate: GAP -> C/8 -> C
    float *sw1 = nullptr, *sb1 = nullptr, *sw2 = nullptr, sb2 = 0.f;       // token gate: C -> C/16 -> 1
    int cr = 0, sr = 0;
    float *fdw = nullptr, *fdwb = nullptr;           // the SGFN's depthwise conv
};

class Dat : public ImgNet {
public:
    static std::unique_ptr<ImgNet> build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override;
    int scale() const override { return 4; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes; }
    size_t weight_bytes() const override { return res.weight_bytes; }
    bool accepts(int w, int h) const override { return w > 0 && h > 0 && w <= maxw && h <= maxh; }

private:
    DevRes res;
    int C = 0, CP = 0, C8 = 0, QS = 0, FS = 0, heads = 0, hd = 0, pad_to = 32, maxw = 0, maxh = 0;
    float mean[3] = {}, cbu_slope = 0.01f;
    bool clamp = false;
    Ln before, fin;
    std::vector<std::vector<Block>> groups;
    std::vector<GOp> gconvs;
    GOp first, after, cbu, up0[2], up2[2], last;
    TMat *ident = nullptr, foldm;
    float *identf = nullptr;
    void *IN16 = nullptr, *XF = nullptr, *N1 = nullptr, *QKV = nullptr, *AT = nullptr, *CX = nullptr, *Z = nullptr, *H1 = nullptr;
    void *N2 = nullptr, *PR = nullptr, *S16 = nullptr, *TT = nullptr, *U0 = nullptr, *V1 = nullptr, *V2 = nullptr, *O16 = nullptr;
    float *S = nullptr, *GIN = nullptr, *sums = nullptr, *cg = nullptr, *tg = nullptr, *stats = nullptr, *attnb = nullptr, *fold = nullptr;
    float *wsp = nullptr;
    size_t ws_floats = 0;
    int maxm = 0;
    std::string what;

    cudaError_t exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
                     std::vector<std::pair<GOp *, GConvArgs>> *collect);
};

// The data-path ops of interest, in order.
struct Ev {
    enum K { LN, MM, CV, SOFTMAX, REDUCEL2, SPLIT, D2S, MULC, ADDS } k;
    const OnnxNode *n;
};

bool consts_of(const Reader &rd, const OnnxNode *n, std::vector<float> *v, std::vector<int64_t> *d)
{
    return init_input(rd, n, v, d);
}

std::unique_ptr<ImgNet> Dat::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    if (g.inputs.empty()) return refuse(why, "not a DAT graph"), nullptr;
    // the data path
    const Lns lns(rd);
    std::set<std::string> data{g.inputs[0]};
    std::vector<Ev> ev;
    const OnnxNode *sub = nullptr;
    for (const OnnxNode &n : g.nodes) {
        if (n.op == "Constant" || n.op == "Shape" || n.op == "Size") continue;
        bool d = false;
        for (const std::string &i : n.in) d |= data.count(i) > 0;
        if (!d) continue;
        for (const std::string &o : n.out) data.insert(o);
        bool allc = true, anyconst = false, allin = true;
        for (const std::string &i : n.in) {
            if (i.empty()) continue;
            const bool c = rd.tensor(i) != nullptr;
            anyconst |= c;
            allin &= data.count(i) > 0;
            allc &= c;
        }
        (void)allc;
        if (lns.is_ln(n)) ev.push_back({Ev::LN, &n});
        else if (lns.span.count(&n)) continue;  // inside a spelled-out LayerNorm
        else if (n.op == "MatMul" && anyconst) ev.push_back({Ev::MM, &n});
        else if (n.op == "Conv") ev.push_back({Ev::CV, &n});
        else if (n.op == "Softmax") ev.push_back({Ev::SOFTMAX, &n});
        else if (n.op == "ReduceL2" || l2_root(rd, n)) ev.push_back({Ev::REDUCEL2, &n});
        else if (n.op == "Split") ev.push_back({Ev::SPLIT, &n});
        else if (n.op == "DepthToSpace") ev.push_back({Ev::D2S, &n});
        else if (n.op == "Mul" && anyconst && sub && n.in[0] == sub->out[0]) continue;  // the image range
        else if (n.op == "Mul" && anyconst && gelu_const_mul(rd, n)) continue;           // a spelled-out GELU
        else if (n.op == "Mul" && anyconst) ev.push_back({Ev::MULC, &n});
        else if (n.op == "Add" && !allin && (!anyconst || attn_const(rd, n))) ev.push_back({Ev::ADDS, &n});
        else if (n.op == "Sub" && anyconst && !sub) sub = &n;
    }
    size_t at = 0;
    auto peek = [&](Ev::K k) { return at < ev.size() && ev[at].k == k; };
    auto take = [&](Ev::K k) -> const OnnxNode * { return peek(k) ? ev[at++].n : nullptr; };
    // the weights of a MatMul and the bias Add consuming it
    auto linear = [&](const OnnxNode *mm, int K, int N, std::vector<float> *w, std::vector<float> *b) -> bool {
        std::vector<int64_t> d;
        if (!mm || !tensor_floats(rd, mm, 1, w, &d) || d != std::vector<int64_t>{K, N}) return false;
        for (const OnnxNode &n : g.nodes)
            if (n.op == "Add" && std::find(n.in.begin(), n.in.end(), mm->out[0]) != n.in.end() && init_input(rd, &n, b) && (int)b->size() == N)
                return true;
        return false;
    };
    auto conv_w = [&](const OnnxNode *n, std::vector<float> *w, std::vector<float> *b, std::vector<int64_t> *d) -> bool {
        if (!n || !tensor_floats(rd, n, 1, w, d) || d->empty() || (*d)[0] < 1) return false;
        if (n->in.size() > 2 && !n->in[2].empty()) {
            if (!rd.floats(n->in[2], b)) return false;
        } else {
            b->assign((size_t)(*d)[0], 0.f);
        }
        fold_bn_after(rd, n, w, b);  // a BatchNorm the export left unfolded
        return true;
    };
    auto lnw = [&](const OnnxNode *n, DevRes &r, Ln *l, int C_) -> bool {
        LnW w;
        if (!lns.weights(rd, n, &w) || (int)w.g.size() != C_) return false;
        l->g = r.dfl(w.g, C_);
        l->b = r.dfl(w.b, C_);
        l->eps = w.eps;
        return true;
    };

    std::unique_ptr<Dat> f(new Dat);
    DevRes &r = f->res;
    std::vector<float> v, b;
    std::vector<int64_t> d;
    // head: mean, conv_first, the LayerNorm before the groups
    ImgRange ir;
    if (!img_range_head(rd, sub, &ir)) return refuse(why, "not a DAT graph"), nullptr;
    std::copy(ir.mean, ir.mean + 3, f->mean);
    const OnnxNode *cf = take(Ev::CV);
    Conv c;
    if (!cf || !rd.conv(cf->name, &c, why) || c.cin != 3 || c.lrelu) return refuse(why, "not a DAT graph"), nullptr;
    img_range_first(ir, &c);
    const int C = f->C = c.cout, CP = f->CP = pad16(C), C8 = f->C8 = (C + 7) / 8 * 8;
    f->QS = pad16(3 * C);
    f->FS = 2 * CP;
    if (f->QS - 2 * C < C8) return refuse(why, "DAT channels"), nullptr;  // v's dwconv reads C8 channels (into the padding)
    f->first.w = r.gup(conv_taps(c, nullptr), c.b, 3, 16, C, CP);
    if (!lnw(take(Ev::LN), r, &f->before, C)) return refuse(why, "not a DAT graph"), nullptr;
    // groups of blocks
    while (peek(Ev::LN) && at + 1 < ev.size() && ev[at + 1].k == Ev::MM) {
        std::vector<Block> blocks;
        while (peek(Ev::LN) && at + 1 < ev.size() && ev[at + 1].k == Ev::MM) {
            Block bk;
            const std::string where = "DAT block " + std::to_string(blocks.size()) + " of group " + std::to_string(f->groups.size());
            if (!lnw(take(Ev::LN), r, &bk.n1, C) || !linear(take(Ev::MM), C, 3 * C, &v, &b)) return refuse(why, where + ": qkv"), nullptr;
            bk.qkv = r.mat(v, &b, C, 3 * C);
            if (peek(Ev::REDUCEL2)) {
                bk.spatial = false;
                take(Ev::REDUCEL2);
                const OnnxNode *rl = take(Ev::REDUCEL2);
                const OnnxNode *tm = take(Ev::MULC);
                if (!rl || !tm || !take(Ev::SOFTMAX) || !consts_of(rd, tm, &v, &d) || d.size() != 3 || d[0] < 1 || d[1] != 1 || d[2] != 1)
                    return refuse(why, where + ": channel attention"), nullptr;
                if (f->heads && f->heads != (int)d[0]) return refuse(why, "DAT mixed heads"), nullptr;
                f->heads = (int)d[0];
                bk.temp = r.dfl(v, f->heads);
                // the norm's clamp: Clip after the ReduceL2
                for (const OnnxNode &n : g.nodes)
                    if (n.op == "Clip" && !n.in.empty() && n.in[0] == rl->out[0] && n.in.size() > 1 && rd.floats(n.in[1], &v) && v.size() == 1)
                        bk.clip = std::max(v[0], 1e-12f);
            } else {
                for (int br = 0; br < 2; br++) {
                    Branch &B = bk.br[br];
                    const OnnxNode *ms = take(Ev::MULC), *ab = take(Ev::ADDS);
                    if (!ms || !ab || !consts_of(rd, ms, &v, &d) || v.size() != 1) return refuse(why, where + ": spatial attention"), nullptr;
                    B.scale = v[0];
                    if (take(Ev::ADDS)) B.mask = true;
                    if (!take(Ev::SOFTMAX)) return refuse(why, where + ": spatial attention"), nullptr;
                    // the bias: the static input of the Add, evaluated
                    StaticEval se(rd);
                    HT bias;
                    std::string bin;
                    for (const std::string &i : ab->in)
                        if (!data.count(i)) bin = i;
                    if (bin.empty() || !se.eval(bin, &bias)) return refuse(why, where + ": position bias (not evaluable)"), nullptr;
                    while (bias.dims.size() > 3 && bias.dims[0] == 1) bias.dims.erase(bias.dims.begin());
                    if (bias.dims.size() != 3 || bias.dims[1] != bias.dims[2]) return refuse(why, where + ": position bias shape"), nullptr;
                    B.heads = (int)bias.dims[0];
                    const int T = (int)bias.dims[1];
                    // the attention kernel's windows hold at most 256 tokens
                    if (B.heads < 1 || T < 1 || T > 256 || bias.v.size() != (size_t)B.heads * T * T)
                        return refuse(why, where + ": position bias shape"), nullptr;
                    std::vector<float> bf, tab;
                    for (double x : bias.v) bf.push_back((float)x);
                    // the window: the factor pair of T whose table is consistent (DAT: branch 0 short and wide)
                    bool found = false;
                    std::vector<std::pair<int, int>> cand;
                    for (int a = 1; a <= T; a++)
                        if (T % a == 0) cand.push_back({a, T / a});
                    if (br == 1) std::reverse(cand.begin(), cand.end());
                    for (auto &ab2 : cand) {
                        if (tbias_table(bf, B.heads, ab2.first, ab2.second, 0, 0, 0, &tab, &B.rt_w, &B.rt_offy, &B.rt_offx)) {
                            B.wh = ab2.first;
                            B.ww = ab2.second;
                            found = true;
                            break;
                        }
                    }
                    if (!found) return refuse(why, where + ": position bias is not a window table"), nullptr;
                    B.rtab = r.dfl(tab, tab.size());
                    if (B.mask) {
                        B.shift_y = B.wh / 2;
                        B.shift_x = B.ww / 2;
                    }
                    f->pad_to = std::max({f->pad_to, B.wh, B.ww});
                }
                if (f->heads && f->heads != bk.br[0].heads + bk.br[1].heads) return refuse(why, "DAT mixed heads"), nullptr;
                f->heads = bk.br[0].heads + bk.br[1].heads;
                if (bk.br[0].heads != bk.br[1].heads) return refuse(why, where + ": branch heads"), nullptr;
            }
            // the depthwise conv on v, the gates, the projection
            const OnnxNode *dwn = take(Ev::CV), *c1 = take(Ev::CV), *c2 = take(Ev::CV), *s1 = take(Ev::CV), *s2 = take(Ev::CV);
            std::vector<float> w0, b0;
            if (!conv_w(dwn, &w0, &b0, &d) || d != std::vector<int64_t>{C, 1, 3, 3} || (int)b0.size() != C) return refuse(why, where + ": v dwconv"), nullptr;
            w0.resize((size_t)C8 * 9, 0.f);
            b0.resize(C8, 0.f);
            bk.dw = r.dfl(w0, (size_t)C8 * 9);
            bk.dwb = r.dfl(b0, C8);
            if (!conv_w(c1, &v, &b, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 || (int64_t)b.size() != d[0])
                return refuse(why, where + ": channel gate"), nullptr;
            bk.cr = (int)d[0];
            bk.cw1 = r.dfl(v, (size_t)bk.cr * C);
            bk.cb1 = r.dfl(b, bk.cr);
            if (!conv_w(c2, &v, &b, &d) || d.size() != 4 || d[0] != C || d[1] != bk.cr || d[2] != 1 || d[3] != 1 || (int)b.size() != C)
                return refuse(why, where + ": channel gate"), nullptr;
            bk.cw2 = r.dfl(v, (size_t)C * bk.cr);
            bk.cb2 = r.dfl(b, C);
            if (!conv_w(s1, &v, &b, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 || d[0] > 16 ||
                (int64_t)b.size() != d[0])
                return refuse(why, where + ": token gate"), nullptr;
            bk.sr = (int)d[0];
            bk.sw1 = r.dfl(v, (size_t)bk.sr * C);
            bk.sb1 = r.dfl(b, bk.sr);
            if (!conv_w(s2, &v, &b, &d) || d.size() != 4 || d[0] != 1 || d[1] != bk.sr || d[2] != 1 || d[3] != 1 || b.size() != 1)
                return refuse(why, where + ": token gate"), nullptr;
            bk.sw2 = r.dfl(v, bk.sr);
            bk.sb2 = b[0];
            if (!linear(take(Ev::MM), C, C, &v, &b)) return refuse(why, where + ": proj"), nullptr;
            bk.proj = r.mat(v, &b, C, C);
            // SGFN: fc1 to [x1 | pad | x2 | pad] (x2 at an aligned offset), GELU, LN(x2), dwconv, * x1, fc2
            if (!lnw(take(Ev::LN), r, &bk.n2, C) || !linear(take(Ev::MM), C, 2 * C, &v, &b)) return refuse(why, where + ": fc1"), nullptr;
            {
                std::vector<float> w2((size_t)C * f->FS, 0.f), b2(f->FS, 0.f);
                for (int k = 0; k < C; k++)
                    for (int o = 0; o < 2 * C; o++) w2[(size_t)k * f->FS + (o < C ? o : CP + o - C)] = v[(size_t)k * 2 * C + o];
                for (int o = 0; o < 2 * C; o++) b2[o < C ? o : CP + o - C] = b[o];
                bk.fc1 = r.mat(w2, &b2, C, f->FS);
            }
            // x1, x2 = chunk(2): a Split, or two Slices (legacy exports); the norm reads x2
            const OnnxNode *sp = take(Ev::SPLIT), *n3 = take(Ev::LN);
            if (!n3 || n3->in.empty() || !(sp ? sp->out.size() == 2 && lns.input(rd, n3) == sp->out[1] : chunk_slice(rd, lns.input(rd, n3))) || !lnw(n3, r, &bk.n3, C))
                return refuse(why, where + ": SGFN"), nullptr;
            const OnnxNode *fdw = take(Ev::CV);
            if (!conv_w(fdw, &w0, &b0, &d) || d != std::vector<int64_t>{C, 1, 3, 3} || (int)b0.size() != C)
                return refuse(why, where + ": SGFN dwconv"), nullptr;
            w0.resize((size_t)C8 * 9, 0.f);
            b0.resize(C8, 0.f);
            bk.fdw = r.dfl(w0, (size_t)C8 * 9);
            bk.fdwb = r.dfl(b0, C8);
            if (!linear(take(Ev::MM), C, C, &v, &b)) return refuse(why, where + ": fc2"), nullptr;
            bk.fc2 = r.mat(v, &b, C, C);
            blocks.push_back(bk);
        }
        // the group conv
        const OnnxNode *gc = take(Ev::CV);
        if (!gc || !rd.conv(gc->name, &c, why) || c.cin != C || c.cout != C || c.lrelu) return refuse(why, "DAT group conv"), nullptr;
        GOp op;
        const std::vector<float> inv(C, 1.f / kStreamScale);  // its input is the stream's scaled fp16 copy
        op.w = r.gup(conv_taps(c, &inv), c.b, C, CP, C, CP);
        f->gconvs.push_back(op);
        f->groups.push_back(std::move(blocks));
    }
    if (f->groups.empty() || !f->heads || C % f->heads) return refuse(why, "DAT groups"), nullptr;
    f->hd = C / f->heads;
    if (f->hd > 32 || f->hd % 2) return refuse(why, "DAT head size"), nullptr;
    // tail
    if (!lnw(take(Ev::LN), r, &f->fin, C)) return refuse(why, "DAT final norm"), nullptr;
    const OnnxNode *ca = take(Ev::CV), *cb = take(Ev::CV), *u0 = take(Ev::CV), *d0 = take(Ev::D2S), *u2 = take(Ev::CV), *d2 = take(Ev::D2S),
                   *cl = take(Ev::CV);
    if (!ca || !cb || !u0 || !d0 || !u2 || !d2 || !cl || at != ev.size()) return refuse(why, "DAT tail"), nullptr;
    if (!rd.conv(ca->name, &c, why)) return nullptr;
    f->after.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, C, CP);
    if (!rd.conv(cb->name, &c, why) || !c.lrelu || c.cout != 64) return refuse(why, "DAT conv_before_upsample"), nullptr;
    f->cbu_slope = c.slope;
    f->cbu.w = r.gup(conv_taps(c, nullptr), c.b, C, CP, 64, 64);
    for (int k = 0; k < 2; k++) {
        if (!rd.conv((k ? u2 : u0)->name, &c, why) || c.cin != 64 || c.cout != 256) return refuse(why, "DAT upsample"), nullptr;
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(c, sy, 64, &wt, &bt);
            (k ? f->up2 : f->up0)[sy].w = r.gup(wt, bt, 64, 64, 128, 128);
        }
    }
    if (!rd.conv(cl->name, &c, why) || c.cin != 64 || c.cout != 3) return refuse(why, "DAT conv_last"), nullptr;
    if (!img_range_tail(rd, cl->out[0], &c, &ir)) return refuse(why, "DAT output range"), nullptr;  // into conv_last
    f->clamp = ir.clamp;
    f->last.w = r.gup(conv_taps(c, nullptr), c.b, 64, 64, 3, 16);
    // the channel attention's per-head mixing, as a GEMM matrix built on the GPU (folded with the identity)
    {
        std::vector<float> id((size_t)C * C, 0.f);
        for (int i = 0; i < C; i++) id[(size_t)i * C + i] = 1.f;
        f->identf = r.dfl(id, (size_t)C * C);
        float *zb = (float *)r.alloc((size_t)C * 4);
        if (!zb || tmat_alloc(f->foldm, C, C, zb) != cudaSuccess) return refuse(why, "weight upload failed"), nullptr;
    }
    if (r.failed) return refuse(why, "weight upload failed"), nullptr;

    // buffers for the largest tile (padded to the windows' multiple for the attention's layout only)
    f->maxw = max_w;
    f->maxh = max_h;
    const size_t M = (size_t)max_w * max_h;
    f->maxm = (int)M;
    // The per-pixel buffers share one arena by lifetime: the steps of exec() each is live in (1-3 the
    // input conv, 9-22 a group (10-20 a block), 30-36 the tail). Kept in step with exec().
    Lifetimes lt;
    lt.add(&f->IN16, M * 32, {{1, 2}});
    lt.add(&f->XF, M * CP * 2, {{2, 31}});
    lt.add((void **)&f->S, M * CP * 4, {{3, 30}});
    lt.add((void **)&f->GIN, M * CP * 4, {{9, 22}});
    lt.add(&f->N1, M * CP * 2, {{10, 11}, {16, 17}, {30, 31}});
    lt.add(&f->QKV, M * f->QS * 2, {{11, 14}});
    lt.add(&f->CX, M * CP * 2, {{12, 14}});
    lt.add(&f->AT, M * CP * 2, {{13, 14}});
    lt.add(&f->Z, M * CP * 2, {{14, 15}});
    lt.add(&f->H1, M * f->FS * 2, {{17, 19}});
    lt.add(&f->N2, M * CP * 2, {{18, 19}});
    lt.add(&f->PR, M * CP * 2, {{19, 20}});
    lt.add(&f->S16, M * CP * 2, {{21, 22}});
    lt.add(&f->TT, M * CP * 2, {{31, 32}});
    lt.add(&f->U0, M * 64 * 2, {{32, 33}});
    lt.add(&f->V1, 4 * M * 64 * 2, {{33, 34}});
    lt.add(&f->V2, 16 * M * 64 * 2, {{34, 35}});
    lt.add(&f->O16, 16 * M * 16 * 2, {{35, 36}});
    if (!lt.assign(r)) return refuse(why, "out of GPU memory"), nullptr;
    f->sums = (float *)r.alloc(CP * 4);
    f->cg = (float *)r.alloc(CP * 4);
    f->tg = (float *)r.alloc(M * 4);
    f->stats = (float *)r.alloc((size_t)(f->heads * f->hd * f->hd + 2 * C) * 4);
    f->attnb = (float *)r.alloc((size_t)f->heads * f->hd * f->hd * 4);
    f->fold = (float *)r.alloc((size_t)C * C * 4);
    if (r.failed) return refuse(why, "out of GPU memory"), nullptr;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        ops.clear();
        f->exec(nullptr, false, nullptr, false, sz ? std::min(max_w, 64) : max_w, sz ? std::min(max_h, 64) : max_h, s, &ops);
        f->ws_floats = std::max(f->ws_floats, gworkspace_max(ops));
    }
    if (f->ws_floats && !(f->wsp = (float *)r.alloc(f->ws_floats * 4))) return refuse(why, "out of GPU memory"), nullptr;
    ops.clear();
    f->exec(nullptr, false, nullptr, false, max_w, max_h, s, &ops);
    if (!gprepare(ops, s, why)) return nullptr;
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    int nb = 0, sh = 0;
    for (auto &gr : f->groups)
        for (auto &bk : gr) {
            nb++;
            sh += bk.spatial && bk.br[0].mask;
        }
    char buf[200];
    snprintf(buf, sizeof(buf), "DAT 4x, %d groups, %d blocks (%d shifted), %d channels, %d heads, %dx%d / %dx%d windows, fp32 stream%s",
             (int)f->groups.size(), nb, sh, C, f->heads, f->groups[0][0].br[0].wh, f->groups[0][0].br[0].ww, f->groups[0][0].br[1].wh,
             f->groups[0][0].br[1].ww, f->clamp ? ", clamped" : "");
    f->what = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

cudaError_t Dat::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    return exec(in, in_f32, out, out_f32, w, h, s, nullptr);
}

cudaError_t Dat::exec(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s,
                      std::vector<std::pair<GOp *, GConvArgs>> *collect)
{
    const int M = w * h, Hp = (h + pad_to - 1) / pad_to * pad_to, Wp = (w + pad_to - 1) / pad_to * pad_to;
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
    auto lnorm = [&](const Ln &l, const void *x, bool xf32, int sx, void *y, bool yf32) -> cudaError_t {
        if (collect) return cudaSuccess;
        return tlayernorm2(x, xf32, sx, y, yf32, CP, M, C, CP, l.g, l.b, l.eps, s);
    };
    const uint16_t *q16 = (const uint16_t *)QKV;

    // Steps (the buffers' lifetimes in build()): 1 IN16, 2 XF, 3 S; per group 9 GIN; per block 10 N1,
    // 11 QKV, 12 CX, 13 AT, 14 Z, 15 S, 16 N1, 17 H1, 18 N2, 19 PR, 20 S; 21 S16, 22 S; then 30 N1,
    // 31 TT, 32 U0, 33 V1, 34 V2, 35 O16, 36 out.
    K(img_to_nhwc(in, in_f32, w, h, false, IN16, 16, s));
    K(tok_sub3(IN16, 16, M, mean[0], mean[1], mean[2], s));
    pr.mark(P_IO);
    CK(Gc(first, ga(IN16, 16, h, w, XF, CP, h, w)));
    CK(lnorm(before, XF, false, CP, S, true));
    pr.mark(P_CONV);
    for (size_t gi = 0; gi < groups.size(); gi++) {
        K(cudaMemcpyAsync(GIN, S, (size_t)M * CP * 4, cudaMemcpyDeviceToDevice, s));
        for (Block &b : groups[gi]) {
            CK(lnorm(b.n1, S, true, CP, N1, false));
            pr.mark(P_LN);
            CK(gemm(*b.qkv, N1, CP, QKV, QS, TAct::None, false));
            pr.mark(P_QKV);
            // conv branch: GELU(dwconv(v) + b); its channel sums for the spatial blocks' channel gate
            K(cudaMemsetAsync(sums, 0, CP * 4, s));
            K(tdwconv2(q16 + 2 * C, QS, CX, CP, h, w, C8, b.dw, b.dwb, true, nullptr, 0, b.spatial ? sums : nullptr, s));
            pr.mark(P_DW);
            if (b.spatial) {
                for (int br = 0; br < 2; br++) {
                    const Branch &B = b.br[br];
                    if (collect) break;
                    TWinAttn a;
                    a.qkv = QKV; a.sq = QS; a.qo = br * C / 2; a.ko = C + br * C / 2; a.vo = 2 * C + br * C / 2;
                    a.H = h; a.W = w; a.Hp = Hp; a.Wp = Wp; a.heads = B.heads; a.hd = hd; a.wh = B.wh; a.ww = B.ww;
                    a.shift_y = B.shift_y; a.shift_x = B.shift_x; a.mask = B.mask; a.scale = B.scale;
                    a.rtab = B.rtab; a.rt_w = B.rt_w; a.rt_offy = B.rt_offy; a.rt_offx = B.rt_offx;
                    a.out = (uint16_t *)AT + br * C / 2; a.so = CP;
                    CK(twinattn(a, s));
                }
                pr.mark(P_WIN);
                // channel gate from the conv branch's average, token gate from the attention
                K(chan_gate(sums, M, C, b.cr, b.cw1, b.cb1, b.cw2, b.cb2, 0, 1.f, cg, CP, s));
                if (!collect && dat_gate_mix(AT, CX, CP, cg, M, C, b.sr, b.sw1, b.sb1, b.sw2, b.sb2, 0, Z, s) == cudaErrorNotSupported) {
                    CK(dat_token_gate(AT, CP, M, C, b.sr, b.sw1, b.sb1, b.sw2, b.sb2, tg, s));
                    CK(dat_mix(AT, CX, CP, cg, tg, 0, Z, M, C, s));
                }
            } else {
                if (!collect) {
                    CK(cudaMemsetAsync(stats, 0, (size_t)(heads * hd * hd + 2 * C) * 4, s));
                    CK(tchanstats(QKV, QS, 0, C, M, heads, hd, stats, s));
                    CK(fdat_chan_soft(stats, b.temp, heads, hd, b.clip, attnb, s));
                    CK(fdat_chan_fold(attnb, identf, heads, hd, fold, s));
                    CK(tmat_pack(foldm, fold, s));
                    TGemm pj;
                    pj.a = q16 + 2 * C; pj.sa = QS; pj.M = M; pj.out = AT; pj.so = CP;
                    CK(tgemm(foldm, pj, s));
                }
                pr.mark(P_CPROJ);
                // channel gate from the attention's average, token gate from the conv branch
                K(tcolsum(AT, CP, M, C, sums, s));
                K(chan_gate(sums, M, C, b.cr, b.cw1, b.cb1, b.cw2, b.cb2, 0, 1.f, cg, CP, s));
                if (!collect && dat_gate_mix(AT, CX, CP, cg, M, C, b.sr, b.sw1, b.sb1, b.sw2, b.sb2, 1, Z, s) == cudaErrorNotSupported) {
                    CK(dat_token_gate(CX, CP, M, C, b.sr, b.sw1, b.sb1, b.sw2, b.sb2, tg, s));
                    CK(dat_mix(AT, CX, CP, cg, tg, 1, Z, M, C, s));
                }
            }
            pr.mark(P_GATE);
            CK(gemm(*b.proj, Z, CP, S, CP, TAct::None, true));
            pr.mark(P_PROJ);
            // SGFN
            CK(lnorm(b.n2, S, true, CP, N1, false));
            CK(gemm(*b.fc1, N1, CP, H1, FS, TAct::Gelu, false));
            pr.mark(P_FC1);
            CK(lnorm(b.n3, (const uint16_t *)H1 + CP, false, FS, N2, false));
            // S += ((dw(N2) + b) * x1) fc2 in one kernel where there is one (else the gated depthwise conv, then the GEMM)
            cudaError_t e2 = cudaErrorNotSupported;
            if (!collect) {
                TDwGemm dg;
                dg.in = N2; dg.sin = CP; dg.H = h; dg.W = w; dg.dw = b.fdw; dg.dwb = b.fdwb; dg.dwc = C8; dg.mul = H1; dg.smul = FS;
                dg.out = S; dg.so = CP; dg.out_f32 = true; dg.add1 = S; dg.s1 = CP; dg.add1_f32 = true;
                e2 = tdwgemm(*b.fc2, dg, s);
            }
            if (e2 != cudaErrorNotSupported) {
                CK(e2);
                pr.mark(P_FC2);
            } else {
                K(tdwconv2(N2, CP, PR, CP, h, w, C8, b.fdw, b.fdwb, false, H1, FS, nullptr, s));
                pr.mark(P_SMIX);
                CK(gemm(*b.fc2, PR, CP, S, CP, TAct::None, true));
                pr.mark(P_FC2);
            }
        }
        K(tconvert(S, true, CP, S16, false, CP, M, CP, s, kStreamScale));
        GConvArgs a = ga(S16, CP, h, w, S, CP, h, w);
        a.acc32 = true;
        a.out_f32 = true;
        a.r1f = GIN;
        a.r1f_stride = CP;
        CK(Gc(gconvs[gi], a));
        pr.mark(P_CONV);
    }
    CK(lnorm(fin, S, true, CP, N1, false));
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
    K(img_from_nhwc(O16, 16, 4 * w, 4 * h, out, out_f32, s, clamp));
    pr.mark(P_IO);
#undef K
#undef CK
    if (!collect) {
        char label[48];
        snprintf(label, sizeof(label), "dat %dx%d", w, h);
        pr.report(label);
    }
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<ImgNet> build_dat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    return Dat::build(g, max_w, max_h, s, why);
}

}  // namespace ajinative
