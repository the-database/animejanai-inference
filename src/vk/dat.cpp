// vk/dat.cpp — DAT (Dual Aggregation Transformer: IllustrationJaNai's DAT-2 4x, bf16 exports) on
// Vulkan for one tile size, as native/dat.cpp runs it: the data path read op by op (the dynamo
// exports carry no module names), the position biases evaluated on the host (native/static_eval.h)
// into relative-position tables. Spatial blocks: window attention in two channel halves (8x32 and
// 32x8 windows, shifted with Swin's mask in every second spatial block, the image zero-padded to the
// windows' multiple inside the attention); channel blocks: channel attention over all tokens
// (folded into a GEMM on the GPU). Both add a depthwise conv on v and the adaptive interaction
// (a channel gate and a token gate) before the projection, then the SGFN. The residual stream is
// fp32; the group conv is the im2col GEMM, as HAT's.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "native/static_eval.h"
#include "vk/imgnet.h"
#include "vk/tnet.h"

namespace ajivk {

using namespace ajinative::onnxread;
using ajinative::HT;
using ajinative::OnnxNode;
using ajinative::StaticEval;

namespace {

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

struct Ln {
    std::vector<float> g, b;
    float eps = 1e-5f;
};
struct Lin {
    std::vector<float> w, b;
    int K = 0, N = 0;
};
struct Branch {  // one half of a spatial block's attention
    int wh = 0, ww = 0, shift_y = 0, shift_x = 0, heads = 0, rt_w = 0, rt_offy = 0, rt_offx = 0, tn = 0;
    bool mask = false;
    float scale = 1.f;
    std::vector<float> rtab;
};
struct Block {
    bool spatial = true;
    Ln n1, n2, n3;  // n3: the SGFN's norm of x2
    Lin qkv, proj, fc1, fc2;
    Branch br[2];
    std::vector<float> temp;  // channel attention
    float clip = 1e-12f;
    std::vector<float> dw, dwb;                         // the depthwise conv on v
    std::vector<float> cw1, cb1, cw2, cb2;              // channel gate: GAP -> cr -> C
    std::vector<float> sw1, sb1, sw2;                   // token gate: C -> sr -> 1
    float sb2 = 0.f;
    int cr = 0, sr = 0;
    std::vector<float> fdw, fdwb;  // the SGFN's depthwise conv
};

// The data-path ops of interest, in order.
struct Ev {
    enum K { LN, MM, CV, SOFTMAX, REDUCEL2, SPLIT, D2S, MULC, ADDS } k;
    const OnnxNode *n;
};

class VkDat : public ImgNet {
public:
    explicit VkDat(Device &d) : net_(d) {}
    static std::unique_ptr<ImgNet> build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
    const Buffer &input() const override { return net_.input(); }
    const Buffer &output() const override { return net_.output(); }
    void bind_io(VkBuffer in, VkBuffer out) override { net_.bind_io(in, out); }
    bool record(VkCommandBuffer cb, int w, int h) const override
    {
        if (w != w_ || h != h_) return false;
        net_.record(cb);
        return true;
    }
    int out_w() const override { return w_ * 4; }
    int out_h() const override { return h_ * 4; }
    size_t device_bytes() const override { return net_.device_bytes(); }
    size_t weight_bytes() const override { return net_.weight_bytes(); }
    std::string describe() const override { return what_; }
    std::string profile(Device &dev) const override { return net_.profile(dev); }
    void suffix(const std::string &s) override { what_ += s; }

private:
    TNet net_;
    int w_ = 0, h_ = 0;
    std::string what_;
};

std::unique_ptr<ImgNet> VkDat::build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    Reader rd(g);
    rd.any_slope = true;
    if (g.inputs.empty()) return refuse_(why, "not a DAT graph"), nullptr;
    // the data path
    std::set<std::string> data{g.inputs[0]};
    std::vector<Ev> ev;
    const OnnxNode *sub = nullptr;
    for (const OnnxNode &n : g.nodes) {
        if (n.op == "Constant" || n.op == "Shape" || n.op == "Size") continue;
        bool d = false;
        for (const std::string &i : n.in) d |= data.count(i) > 0;
        if (!d) continue;
        for (const std::string &o : n.out) data.insert(o);
        bool anyconst = false, allin = true;
        for (const std::string &i : n.in) {
            if (i.empty()) continue;
            anyconst |= rd.tensor(i) != nullptr;
            allin &= data.count(i) > 0;
        }
        if (n.op == "LayerNormalization") ev.push_back({Ev::LN, &n});
        else if (n.op == "MatMul" && anyconst) ev.push_back({Ev::MM, &n});
        else if (n.op == "Conv") ev.push_back({Ev::CV, &n});
        else if (n.op == "Softmax") ev.push_back({Ev::SOFTMAX, &n});
        else if (n.op == "ReduceL2") ev.push_back({Ev::REDUCEL2, &n});
        else if (n.op == "Split") ev.push_back({Ev::SPLIT, &n});
        else if (n.op == "DepthToSpace") ev.push_back({Ev::D2S, &n});
        else if (n.op == "Mul" && anyconst) ev.push_back({Ev::MULC, &n});
        else if (n.op == "Add" && !allin && !anyconst) ev.push_back({Ev::ADDS, &n});
        else if (n.op == "Sub" && anyconst && !sub) sub = &n;
    }
    size_t at = 0;
    auto peek = [&](Ev::K k) { return at < ev.size() && ev[at].k == k; };
    auto take = [&](Ev::K k) -> const OnnxNode * { return peek(k) ? ev[at++].n : nullptr; };
    // the weights of a MatMul and the bias Add consuming it
    auto linear = [&](const OnnxNode *mm, int K, int N, Lin *l) -> bool {
        std::vector<int64_t> d;
        if (!mm || !tensor_floats(rd, mm, 1, &l->w, &d) || d != std::vector<int64_t>{K, N}) return false;
        l->K = K;
        l->N = N;
        for (const OnnxNode &n : g.nodes)
            if (n.op == "Add" && std::find(n.in.begin(), n.in.end(), mm->out[0]) != n.in.end() && init_input(rd, &n, &l->b) && (int)l->b.size() == N)
                return true;
        return false;
    };
    auto conv_w = [&](const OnnxNode *n, std::vector<float> *w_, std::vector<float> *b_, std::vector<int64_t> *d) -> bool {
        return n && tensor_floats(rd, n, 1, w_, d) && n->in.size() > 2 && rd.floats(n->in[2], b_);
    };
    auto lnw = [&](const OnnxNode *n, Ln *l, int C_) -> bool {
        if (!n || n->in.size() < 3 || !tensor_floats(rd, n, 1, &l->g) || !tensor_floats(rd, n, 2, &l->b) || (int)l->g.size() != C_ ||
            (int)l->b.size() != C_)
            return false;
        l->eps = n->attr("epsilon") ? n->attr("epsilon")->f : 1e-5f;
        return true;
    };

    std::vector<float> v, b;
    std::vector<int64_t> d;
    float mean[3];
    // head: mean, conv_first, the LayerNorm before the groups
    if (!sub || !init_input(rd, sub, &v, &d) || v.size() != 3) return refuse_(why, "not a DAT graph"), nullptr;
    std::copy(v.begin(), v.end(), mean);
    const OnnxNode *cf = take(Ev::CV);
    Conv cfirst;
    std::string we;
    if (!cf || !rd.conv(cf->name, &cfirst, &we) || cfirst.cin != 3 || cfirst.lrelu) return refuse_(why, "not a DAT graph"), nullptr;
    const int C = cfirst.cout, CP = pad16(C), QS = pad16(3 * C), FS = 2 * CP;
    Ln before, fin;
    if (!lnw(take(Ev::LN), &before, C)) return refuse_(why, "not a DAT graph"), nullptr;
    int heads = 0, pad_to = 32;
    std::vector<std::vector<Block>> groups;
    std::vector<Conv> gconvs;
    // groups of blocks
    while (peek(Ev::LN) && at + 1 < ev.size() && ev[at + 1].k == Ev::MM) {
        std::vector<Block> blocks;
        while (peek(Ev::LN) && at + 1 < ev.size() && ev[at + 1].k == Ev::MM) {
            Block bk;
            const std::string where = "DAT block " + std::to_string(blocks.size()) + " of group " + std::to_string(groups.size());
            if (!lnw(take(Ev::LN), &bk.n1, C) || !linear(take(Ev::MM), C, 3 * C, &bk.qkv)) return refuse_(why, where + ": qkv"), nullptr;
            if (peek(Ev::REDUCEL2)) {
                bk.spatial = false;
                take(Ev::REDUCEL2);
                const OnnxNode *rl = take(Ev::REDUCEL2);
                const OnnxNode *tm = take(Ev::MULC);
                if (!rl || !tm || !take(Ev::SOFTMAX) || !init_input(rd, tm, &bk.temp, &d) || d.size() != 3 || d[0] < 1 || d[1] != 1 || d[2] != 1)
                    return refuse_(why, where + ": channel attention"), nullptr;
                if (heads && heads != (int)d[0]) return refuse_(why, "DAT mixed heads"), nullptr;
                heads = (int)d[0];
                for (const OnnxNode &n : g.nodes)  // the norm's clamp: Clip after the ReduceL2
                    if (n.op == "Clip" && !n.in.empty() && n.in[0] == rl->out[0] && n.in.size() > 1 && rd.floats(n.in[1], &v) && v.size() == 1)
                        bk.clip = std::max(v[0], 1e-12f);
            } else {
                for (int br = 0; br < 2; br++) {
                    Branch &B = bk.br[br];
                    const OnnxNode *ms = take(Ev::MULC), *ab = take(Ev::ADDS);
                    if (!ms || !ab || !init_input(rd, ms, &v, &d) || v.size() != 1) return refuse_(why, where + ": spatial attention"), nullptr;
                    B.scale = v[0];
                    if (take(Ev::ADDS)) B.mask = true;
                    if (!take(Ev::SOFTMAX)) return refuse_(why, where + ": spatial attention"), nullptr;
                    // the bias: the static input of the Add, evaluated
                    StaticEval se(rd);
                    HT bias;
                    std::string bin;
                    for (const std::string &i : ab->in)
                        if (!data.count(i)) bin = i;
                    if (bin.empty() || !se.eval(bin, &bias)) return refuse_(why, where + ": position bias (not evaluable)"), nullptr;
                    while (bias.dims.size() > 3 && bias.dims[0] == 1) bias.dims.erase(bias.dims.begin());
                    if (bias.dims.size() != 3 || bias.dims[1] != bias.dims[2]) return refuse_(why, where + ": position bias shape"), nullptr;
                    B.heads = (int)bias.dims[0];
                    const int T = (int)bias.dims[1];
                    if (B.heads < 1 || T < 1 || T > 256 || bias.v.size() != (size_t)B.heads * T * T)
                        return refuse_(why, where + ": position bias shape"), nullptr;
                    std::vector<float> bf;
                    for (double x : bias.v) bf.push_back((float)x);
                    // the window: the factor pair of T whose table is consistent (DAT: branch 0 short and wide)
                    bool found = false;
                    std::vector<std::pair<int, int>> cand;
                    for (int a = 1; a <= T; a++)
                        if (T % a == 0) cand.push_back({a, T / a});
                    if (br == 1) std::reverse(cand.begin(), cand.end());
                    for (auto &ab2 : cand)
                        if (bias_table(bf, B.heads, ab2.first, ab2.second, 0, 0, 0, &B.rtab, &B.rt_w, &B.rt_offy, &B.rt_offx, &B.tn)) {
                            B.wh = ab2.first;
                            B.ww = ab2.second;
                            found = true;
                            break;
                        }
                    if (!found) return refuse_(why, where + ": position bias is not a window table"), nullptr;
                    if (B.mask) {
                        B.shift_y = B.wh / 2;
                        B.shift_x = B.ww / 2;
                    }
                    pad_to = std::max({pad_to, B.wh, B.ww});
                }
                if (heads && heads != bk.br[0].heads + bk.br[1].heads) return refuse_(why, "DAT mixed heads"), nullptr;
                heads = bk.br[0].heads + bk.br[1].heads;
                if (bk.br[0].heads != bk.br[1].heads) return refuse_(why, where + ": branch heads"), nullptr;
            }
            // the depthwise conv on v, the gates, the projection
            const OnnxNode *dwn = take(Ev::CV), *c1 = take(Ev::CV), *c2 = take(Ev::CV), *s1 = take(Ev::CV), *s2 = take(Ev::CV);
            if (!conv_w(dwn, &bk.dw, &bk.dwb, &d) || d != std::vector<int64_t>{C, 1, 3, 3} || (int)bk.dwb.size() != C)
                return refuse_(why, where + ": v dwconv"), nullptr;
            if (!conv_w(c1, &bk.cw1, &bk.cb1, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 || (int64_t)bk.cb1.size() != d[0])
                return refuse_(why, where + ": channel gate"), nullptr;
            bk.cr = (int)d[0];
            if (!conv_w(c2, &bk.cw2, &bk.cb2, &d) || d.size() != 4 || d[0] != C || d[1] != bk.cr || d[2] != 1 || d[3] != 1 || (int)bk.cb2.size() != C)
                return refuse_(why, where + ": channel gate"), nullptr;
            if (!conv_w(s1, &bk.sw1, &bk.sb1, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 || d[0] > 16 ||
                (int64_t)bk.sb1.size() != d[0])
                return refuse_(why, where + ": token gate"), nullptr;
            bk.sr = (int)d[0];
            if (!conv_w(s2, &bk.sw2, &b, &d) || d.size() != 4 || d[0] != 1 || d[1] != bk.sr || d[2] != 1 || d[3] != 1 || b.size() != 1)
                return refuse_(why, where + ": token gate"), nullptr;
            bk.sb2 = b[0];
            if (!linear(take(Ev::MM), C, C, &bk.proj)) return refuse_(why, where + ": proj"), nullptr;
            // SGFN: fc1 to [x1 | pad | x2 | pad] (x2 at CP), GELU, LN(x2), dwconv, * x1, fc2
            Lin f1;
            if (!lnw(take(Ev::LN), &bk.n2, C) || !linear(take(Ev::MM), C, 2 * C, &f1)) return refuse_(why, where + ": fc1"), nullptr;
            bk.fc1.K = C;
            bk.fc1.N = FS;
            bk.fc1.w.assign((size_t)C * FS, 0.f);
            bk.fc1.b.assign(FS, 0.f);
            for (int k = 0; k < C; k++)
                for (int o = 0; o < 2 * C; o++) bk.fc1.w[(size_t)k * FS + (o < C ? o : CP + o - C)] = f1.w[(size_t)k * 2 * C + o];
            for (int o = 0; o < 2 * C; o++) bk.fc1.b[o < C ? o : CP + o - C] = f1.b[o];
            const OnnxNode *sp = take(Ev::SPLIT), *n3 = take(Ev::LN);
            if (!sp || sp->out.size() != 2 || !n3 || n3->in.empty() || n3->in[0] != sp->out[1] || !lnw(n3, &bk.n3, C))
                return refuse_(why, where + ": SGFN"), nullptr;
            const OnnxNode *fdw = take(Ev::CV);
            if (!conv_w(fdw, &bk.fdw, &bk.fdwb, &d) || d != std::vector<int64_t>{C, 1, 3, 3} || (int)bk.fdwb.size() != C)
                return refuse_(why, where + ": SGFN dwconv"), nullptr;
            if (!linear(take(Ev::MM), C, C, &bk.fc2)) return refuse_(why, where + ": fc2"), nullptr;
            blocks.push_back(std::move(bk));
        }
        // the group conv
        const OnnxNode *gc = take(Ev::CV);
        Conv c;
        if (!gc || !rd.conv(gc->name, &c, why) || c.cin != C || c.cout != C || c.lrelu) return refuse_(why, "DAT group conv"), nullptr;
        gconvs.push_back(std::move(c));
        groups.push_back(std::move(blocks));
    }
    if (groups.empty() || !heads || C % heads) return refuse_(why, "DAT groups"), nullptr;
    const int hd = C / heads;
    if (hd > 32 || hd % 2) return refuse_(why, "DAT head size"), nullptr;
    if (C > 256) return refuse_(why, "DAT channels"), nullptr;
    // tail
    if (!lnw(take(Ev::LN), &fin, C)) return refuse_(why, "DAT final norm"), nullptr;
    const OnnxNode *ca = take(Ev::CV), *cb = take(Ev::CV), *u0 = take(Ev::CV), *d0 = take(Ev::D2S), *u2 = take(Ev::CV), *d2 = take(Ev::D2S),
                   *cl = take(Ev::CV);
    if (!ca || !cb || !u0 || !d0 || !u2 || !d2 || !cl || at != ev.size()) return refuse_(why, "DAT tail"), nullptr;
    Conv cafter, cbu, cup[2], clast;
    if (!rd.conv(ca->name, &cafter, why)) return nullptr;
    if (!rd.conv(cb->name, &cbu, why) || !cbu.lrelu || cbu.cout != 64) return refuse_(why, "DAT conv_before_upsample"), nullptr;
    for (int k = 0; k < 2; k++)
        if (!rd.conv((k ? u2 : u0)->name, &cup[k], why) || cup[k].cin != 64 || cup[k].cout != 256) return refuse_(why, "DAT upsample"), nullptr;
    if (!rd.conv(cl->name, &clast, why) || clast.cin != 64 || clast.cout != 3) return refuse_(why, "DAT conv_last"), nullptr;
    for (const OnnxNode &n : g.nodes)  // the mean added back after conv_last: into its bias
        if (n.op == "Add" && std::find(n.in.begin(), n.in.end(), cl->out[0]) != n.in.end() && init_input(rd, &n, &v) && v.size() == 3)
            for (int k = 0; k < 3; k++) clast.b[k] += v[k];
    if (w < 1 || h < 1) return refuse_(why, "bad tile size"), nullptr;

    std::unique_ptr<VkDat> f(new VkDat(dev));
    f->w_ = w;
    f->h_ = h;
    const int M = w * h, Hp = (h + pad_to - 1) / pad_to * pad_to, Wp = (w + pad_to - 1) / pad_to * pad_to;
    TNet &n = f->net_;

    // ---- the arena. The body: S, GIN (fp32 stream, group input), N1, QKV, CX (the conv branch), AT,
    // Z (the mix), H1 (the SGFN's hidden layer), N2, PR; S16 is N1, TG the token gate. The tail
    // reuses them: TT in AT, U0 in Z, V1 and the output map in H1 (or fresh), V2 from S on.
    const VkDeviceSize Mb = (VkDeviceSize)M;
    const VkDeviceSize IN16 = n.place(Mb * 16 * 2), XF = n.place(Mb * CP * 2);
    const VkDeviceSize S = n.place(Mb * CP * 4), GIN = n.place(Mb * CP * 4), N1 = n.place(Mb * CP * 2), QKV = n.place(Mb * QS * 2),
                       CX = n.place(Mb * CP * 2), AT = n.place(Mb * CP * 2), Z = n.place(Mb * CP * 2), N2 = n.place(Mb * CP * 2),
                       PR = n.place(Mb * CP * 2);
    const VkDeviceSize H1 = n.place(Mb * std::max(FS, 256) * 2), S16 = N1;
    const VkDeviceSize SUMS = n.place(CP * 4), CG = n.place(CP * 4), TG = n.place(Mb * 4);
    const VkDeviceSize TT = AT, U0 = CP >= 64 ? Z : n.place(Mb * 64 * 2), V1 = H1, O16 = H1;
    const VkDeviceSize V2 = H1 - S >= Mb * 16 * 64 * 2 ? S : n.place(Mb * 16 * 64 * 2);
    // channel attention: per chunk partials, the reduced statistics, the softmax
    const int SN = heads * hd * hd + 2 * C, tpc = 256, nch = (M + tpc - 1) / tpc;
    const int PART = n.add_part((size_t)nch * SN), STATS = n.add_part(SN), ATTN = n.add_part((size_t)heads * hd * hd);
    int ident = -1;
    {
        std::vector<float> id((size_t)C * C, 0.f);
        for (int i = 0; i < C; i++) id[(size_t)i * C + i] = 1.f;
        ident = n.add_wf(id);
    }
    const auto hm = [](VkDeviceSize o, int s, int off = 0) { return hmap(o, s, off); };
    const auto fm = [](VkDeviceSize o, int s) { return fmap(o, s); };
    std::vector<std::vector<uint16_t>> keep;  // the taps outlive the build (their address keys the weights)
    keep.reserve(16);
    const auto conv = [&](const Conv &c, std::vector<uint16_t> taps, const std::vector<float> &bb, int cout, VkDeviceSize src, int sstride,
                          VkDeviceSize dst, int dstride, int H, int W, int act, float slope, VkDeviceSize r1, int out, int sy,
                          std::initializer_list<int> widths) {
        keep.push_back(std::move(taps));
        TNet::ConvIO io;
        io.act = act;
        io.slope = slope;
        io.out = out;
        io.src = src;
        io.src_bytes = (VkDeviceSize)H * W * sstride * 2;
        io.src_stride = sstride;
        io.cinp = sstride;
        io.IH = io.H = H;
        io.IW = io.W = W;
        io.sy = sy;
        if (out == 3) {
            io.dst_out = true;
        } else {
            io.dst = dst;
            io.dst_bytes = (VkDeviceSize)H * W * dstride * 2 * (out == 4 ? 4 : 1);
            io.dst_stride = dstride;
        }
        if (r1 != ~0ull) {
            io.nres = 1;
            io.r1 = r1;
            io.r1_bytes = (VkDeviceSize)H * W * dstride * 2;
            io.r1_stride = dstride;
        }
        n.conv(keep.back(), bb, c.cin, cout, io, widths, &keep.back());
    };
    const auto mat = [&](const Lin &l) { return n.add_mat(l.w, l.b, l.K, l.N); };
    const auto lnorm = [&](const Ln &l, const TMap &in, const TMap &out) { n.layernorm(in, out, M, C, CP, n.add_wf(l.g), n.add_wf(l.b), l.eps); };

    // ---- conv_first on the input minus the mean, the norm before the groups
    n.imgin(hm(IN16, 16), h, w, h, w, mean);
    conv(cfirst, conv_taps(cfirst, nullptr), cfirst.b, C, IN16, 16, XF, CP, h, w, 0, 0.2f, ~0ull, 0, 0, {64, 48, 32, 16});
    lnorm(before, hm(XF, CP), fm(S, CP));
    for (size_t gi = 0; gi < groups.size(); gi++) {
        n.copy(S, GIN, Mb * CP * 4);
        for (const Block &bk : groups[gi]) {
            lnorm(bk.n1, fm(S, CP), hm(N1, CP));
            n.gemm(hm(N1, CP), M, mat(bk.qkv), hm(QKV, QS));
            // the conv branch: GELU(dwconv(v) + b); its channel sums for the spatial blocks' channel gate
            n.dwconv(hm(QKV, QS, 2 * C), hm(CX, CP), h, w, C, n.add_wf(bk.dw), n.add_wf(bk.dwb), true, bk.spatial ? SUMS : ~0ull);
            const int cw1 = n.add_wf(bk.cw1), cb1 = n.add_wf(bk.cb1), cw2 = n.add_wf(bk.cw2), cb2 = n.add_wf(bk.cb2);
            const int sw1 = n.add_wf(bk.sw1), sb1 = n.add_wf(bk.sb1), sw2 = n.add_wf(bk.sw2);
            if (bk.spatial) {
                for (int br = 0; br < 2; br++) {
                    const Branch &B = bk.br[br];
                    TWin a;
                    a.H = h;
                    a.W = w;
                    a.Hp = Hp;
                    a.Wp = Wp;
                    a.heads = B.heads;
                    a.hd = hd;
                    a.wh = B.wh;
                    a.ww = B.ww;
                    a.sy = B.shift_y;
                    a.sx = B.shift_x;
                    a.mask = B.mask;
                    a.scale = B.scale;
                    a.qo = br * C / 2;
                    a.ko = C + br * C / 2;
                    a.vo = 2 * C + br * C / 2;
                    a.rtab = n.add_wf(B.rtab);
                    a.rt_w = B.rt_w;
                    a.rt_offy = B.rt_offy;
                    a.rt_offx = B.rt_offx;
                    a.tn = B.tn;
                    n.winattn(a, hm(QKV, QS), hm(AT, CP, br * C / 2));
                }
                // channel gate from the conv branch's average, token gate from the attention
                n.chan_gate(SUMS, M, C, bk.cr, cw1, cb1, cw2, cb2, false, 1.f, CG, CP);
                n.token_gate(hm(AT, CP), M, C, bk.sr, sw1, sb1, sw2, bk.sb2, TG);
                n.mix(hm(AT, CP), hm(CX, CP), hm(Z, CP), M, C, CP, CG, TG, 0);
            } else {
                TArgs cs;
                cs.map(0, hm(QKV, QS));
                cs.v[16] = M;
                cs.v[17] = heads;
                cs.v[18] = hd;
                cs.v[19] = 0;
                cs.v[20] = C;
                cs.v[21] = tpc;
                cs.v[22] = PART;
                n.pass(6, cs, (uint32_t)nch, (uint32_t)heads, "chanstats");
                TArgs cr;
                cr.v[16] = SN;
                cr.v[17] = nch;
                cr.v[18] = PART;
                cr.v[19] = STATS;
                n.pass_threads(7, cr, SN, "chanreduce");
                TArgs sm;
                sm.v[16] = heads;
                sm.v[17] = hd;
                sm.v[18] = STATS;
                sm.v[19] = n.add_wf(bk.temp);
                sm.v[20] = ATTN;
                sm.f(21, bk.clip);
                n.pass_threads(8, sm, heads * hd, "chansoft");
                // the per-head mixing folded with the identity into a GEMM matrix
                TNet::Mat fold;
                fold.K = fold.N = C;
                fold.Kp = fold.Np = CP;
                fold.w = n.add_wh_zero((size_t)CP * CP);
                fold.bias = -1;
                TArgs fd;
                fd.v[16] = heads;
                fd.v[17] = hd;
                fd.v[18] = ATTN;
                fd.v[19] = ident;
                fd.v[20] = fold.w;
                fd.v[21] = CP;
                n.pass_threads(9, fd, (long long)C * C, "chanfold");
                n.gemm(hm(QKV, QS, 2 * C), M, fold, hm(AT, CP));
                // channel gate from the attention's average, token gate from the conv branch
                n.colsums(hm(AT, CP), M, C, SUMS);
                n.chan_gate(SUMS, M, C, bk.cr, cw1, cb1, cw2, cb2, false, 1.f, CG, CP);
                n.token_gate(hm(CX, CP), M, C, bk.sr, sw1, sb1, sw2, bk.sb2, TG);
                n.mix(hm(AT, CP), hm(CX, CP), hm(Z, CP), M, C, CP, CG, TG, 1);
            }
            n.gemm(hm(Z, CP), M, mat(bk.proj), fm(S, CP), 0, false, -1, fm(S, CP));
            // SGFN
            lnorm(bk.n2, fm(S, CP), hm(N1, CP));
            n.gemm(hm(N1, CP), M, mat(bk.fc1), hm(H1, FS), 1);
            lnorm(bk.n3, hm(H1, FS, CP), hm(N2, CP));
            n.dwconv(hm(N2, CP), hm(PR, CP), h, w, C, n.add_wf(bk.fdw), n.add_wf(bk.fdwb), false, ~0ull, hm(H1, FS));
            n.gemm(hm(PR, CP), M, mat(bk.fc2), fm(S, CP), 0, false, -1, fm(S, CP));
        }
        // the group conv plus the group input, fp32 sums and output
        n.convert(fm(S, CP), hm(S16, CP), M, CP);
        n.conv_gemm(hm(S16, CP), h, w, CP, n.add_conv_mat(conv_taps(gconvs[gi], nullptr), gconvs[gi].b, C, CP, C), fm(S, CP), fm(GIN, CP));
    }
    // ---- the tail
    lnorm(fin, fm(S, CP), hm(N1, CP));
    conv(cafter, conv_taps(cafter, nullptr), cafter.b, C, N1, CP, TT, CP, h, w, 0, 0.2f, XF, 0, 0, {64, 48, 32, 16});
    conv(cbu, conv_taps(cbu, nullptr), cbu.b, 64, TT, CP, U0, 64, h, w, 2, cbu.slope, ~0ull, 0, 0, {64, 48, 32, 16});
    for (int k = 0; k < 2; k++)
        for (int sy = 0; sy < 2; sy++) {
            std::vector<uint16_t> wt;
            std::vector<float> bt;
            ps_weights(cup[k], sy, 64, &wt, &bt);
            const int s = k ? 2 : 1;
            conv(cup[k], std::move(wt), bt, 128, k ? V1 : U0, 64, k ? V2 : V1, 64, s * h, s * w, 0, 0.2f, ~0ull, 4, sy, {64, 48, 32, 16});
        }
    conv(clast, conv_taps(clast, nullptr), clast.b, 3, V2, 64, 0, 0, 4 * h, 4 * w, 0, 0.2f, ~0ull, 3, 0, {16});
    (void)O16;
    if (!n.finish((VkDeviceSize)3 * w * h * 2, (VkDeviceSize)3 * f->out_w() * f->out_h() * 2, why)) return nullptr;
    int nb = 0, sh = 0;
    for (auto &gr : groups)
        for (auto &bk : gr) {
            nb++;
            sh += bk.spatial && bk.br[0].mask;
        }
    char buf[240];
    std::snprintf(buf, sizeof buf, "DAT 4x, %d groups, %d blocks (%d shifted), %d channels, %d heads, %dx%d / %dx%d windows, fp32 stream; Vulkan %s",
                  (int)groups.size(), nb, sh, C, heads, groups[0][0].br[0].wh, groups[0][0].br[0].ww, groups[0][0].br[1].wh, groups[0][0].br[1].ww,
                  family_name(dev.family()));
    f->what_ = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

}  // namespace

std::unique_ptr<ImgNet> build_dat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    return VkDat::build(dev, g, w, h, why);
}

}  // namespace ajivk
