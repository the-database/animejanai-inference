// vk/fdat.cpp — FDAT (traiNNer-redux fdat_arch: IllustrationJaNai's FDAT_M / XL) on Vulkan for one
// tile size, as native/fdat.cpp and native/fdat_body.cpp run it (see fdat_body.h for the block):
// conv_first (on the frame or its 2x pixel-unshuffle), the residual groups on tfm.comp and the
// general conv, conv_after plus the shallow features, and the transposed-conv upsampler as phase
// convs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "vk/imgnet.h"
#include "vk/tnet.h"

namespace ajivk {

using namespace ajinative::onnxread;

namespace {

bool refuse_(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

// One block's weights as the CUDA parse reads them (fdat_body.cpp), on the host.
struct Block {
    bool spatial = true;
    std::vector<float> ln1g, ln1b, ln2g, ln2b, qkv, wp, bp, dw, bias, cg1, cg2, temp, sg, fc1, fc2, smix;
    float scale = 1.f, eps1 = 1e-5f, eps2 = 1e-5f, clip = 1.1920928955078125e-07f, pre = 0.f;
    int shift = 0;
};

struct Body {
    int C = 0, CP = 0, QS = 0, HS = 0, heads = 0, hd = 0, ws = 8, cgr = 0, ngroups = 0, nblk = 0;
    std::vector<Block> blocks;
    std::vector<Conv> gconvs;
};

bool parse_body(const Reader &rd, const std::string &prefix, int C, Body *b, std::string *why)
{
    b->C = C;
    b->CP = pad16(C);
    b->QS = pad16(3 * C);
    b->HS = pad16(2 * C);
    const std::string P = prefix + "groups/groups.";
    while (rd.node(P + std::to_string(b->ngroups) + "/conv/Conv")) b->ngroups++;
    while (rd.node(P + "0/blocks." + std::to_string(b->nblk) + "/n1/LayerNormalization")) b->nblk++;
    if (!b->ngroups || !b->nblk) return refuse_(why, "FDAT groups");
    for (int gi = 0; gi < b->ngroups; gi++) {
        const std::string Gp = P + std::to_string(gi) + "/";
        Conv cg;
        if (!rd.conv(Gp + "conv/Conv", &cg, why)) return false;
        if (cg.cin != C || cg.cout != C || cg.lrelu) return refuse_(why, "FDAT group conv");
        b->gconvs.push_back(cg);
        for (int bi = 0; bi < b->nblk; bi++) {
            const std::string B = Gp + "blocks." + std::to_string(bi) + "/";
            Block k;
            std::vector<float> v, v2;
            std::vector<int64_t> d;
            const ajinative::OnnxNode *ln1 = rd.node(B + "n1/LayerNormalization"), *ln2 = rd.node(B + "n2/LayerNormalization");
            if (!tensor_floats(rd, ln1, 1, &k.ln1g) || !tensor_floats(rd, ln1, 2, &k.ln1b) || (int)k.ln1g.size() != C) return refuse_(why, B + "n1");
            k.eps1 = ln1->attr("epsilon") ? ln1->attr("epsilon")->f : 1e-5f;
            if (!tensor_floats(rd, ln2, 1, &k.ln2g) || !tensor_floats(rd, ln2, 2, &k.ln2b)) return refuse_(why, B + "n2");
            k.eps2 = ln2->attr("epsilon") ? ln2->attr("epsilon")->f : 1e-5f;
            if (rd.node(B + "attn/qkv/Add") || rd.node(B + "ffn/fc1/Add") || rd.node(B + "ffn/fc2/Add")) return refuse_(why, B + "qkv/ffn bias");
            if (!tensor_floats(rd, rd.node(B + "attn/qkv/MatMul"), 1, &k.qkv, &d) || d != std::vector<int64_t>{C, 3 * C}) return refuse_(why, B + "qkv");
            if (!tensor_floats(rd, rd.node(B + "attn/proj/MatMul"), 1, &k.wp, &d) || d != std::vector<int64_t>{C, C} ||
                !init_input(rd, rd.node(B + "attn/proj/Add"), &k.bp) || (int)k.bp.size() != C)
                return refuse_(why, B + "proj");
            if (!tensor_floats(rd, rd.node(B + "conv/conv.0/Conv"), 1, &k.dw, &d) || d != std::vector<int64_t>{C, 1, 3, 3})
                return refuse_(why, B + "conv");
            const ajinative::OnnxNode *sm = rd.node(B + "attn/Softmax");
            const ajinative::OnnxNode *pre = sm && !sm->in.empty() && rd.prod.count(sm->in[0]) ? rd.prod.at(sm->in[0]) : nullptr;
            if (!pre) return refuse_(why, B + "attention");
            if (pre->op == "Add") {
                k.spatial = true;
                if (!init_input(rd, pre, &k.bias, &d) || d.size() != 3 || d[1] != d[2]) return refuse_(why, B + "bias");
                const int nh = (int)d[0], T = (int)d[1], wsz = (int)std::lround(std::sqrt((double)T));
                if (nh < 1 || T < 1 || wsz * wsz != T || C % nh) return refuse_(why, B + "window");
                if ((b->heads && b->heads != nh) || (b->ws != wsz && bi + gi > 0)) return refuse_(why, "FDAT mixed heads/windows");
                b->heads = nh;
                b->ws = wsz;
                std::vector<float> sc;
                const ajinative::OnnxNode *mm = nullptr, *ml = nullptr;
                for (const std::string &i : pre->in)
                    if (rd.prod.count(i) && rd.prod.at(i)->op == "MatMul") mm = rd.prod.at(i);
                if (mm)
                    for (const std::string &i : mm->in)
                        if (rd.prod.count(i) && rd.prod.at(i)->op == "Mul") ml = rd.prod.at(i);
                if (!init_input(rd, ml, &sc) || sc.size() != 1) return refuse_(why, B + "scale");
                k.scale = sc[0];
                if (const ajinative::OnnxNode *sl = rd.node(B + "attn/Slice")) {
                    std::vector<int64_t> st;
                    const ajinative::OnnxTensor *t = sl->in.size() > 1 ? rd.tensor(sl->in[1]) : nullptr;
                    if (!t || !t->ints(&st) || st.size() != 1 || st[0] <= 0 || st[0] >= b->ws) return refuse_(why, B + "window shift");
                    k.shift = (int)st[0];
                }
                if (!tensor_floats(rd, rd.node(B + "inter/cg/cg.1/Conv"), 1, &k.cg1, &d) || d.size() != 4 || d[1] != C || d[0] < 1 || d[2] != 1 ||
                    d[3] != 1)
                    return refuse_(why, B + "cg");
                b->cgr = (int)d[0];
                if (!tensor_floats(rd, rd.node(B + "inter/cg/cg.3/Conv"), 1, &k.cg2, &d) || d.size() != 4 || d[0] != C || d[1] != b->cgr)
                    return refuse_(why, B + "cg");
            } else if (pre->op == "Mul" && rd.node(B + "attn/ReduceL2")) {
                k.spatial = false;
                if (!init_input(rd, pre, &k.temp, &d) || d.size() != 3 || d[0] < 1 || d[1] != 1 || d[2] != 1) return refuse_(why, B + "temperature");
                if (b->heads && b->heads != (int)d[0]) return refuse_(why, "FDAT mixed heads");
                b->heads = (int)d[0];
                std::vector<float> cl, pw;
                const ajinative::OnnxNode *clip = rd.node(B + "attn/Clip");
                if (!clip || clip->in.size() < 2 || !rd.floats(clip->in[1], &cl) || cl.size() != 1) return refuse_(why, B + "norm clip");
                k.clip = cl[0];
                if (const ajinative::OnnxNode *pn = rd.node(B + "attn/Pow")) {
                    if (pn->in.size() < 2 || !rd.floats(pn->in[1], &pw) || pw.size() != 1) return refuse_(why, B + "norm pre-scale");
                    k.pre = pw[0];
                }
                if (!tensor_floats(rd, rd.node(B + "inter/sg/sg.0/Conv"), 1, &k.sg, &d) || d.size() != 4 || d[0] != 1 || d[1] != C)
                    return refuse_(why, B + "sg");
            } else {
                return refuse_(why, B + "attention type");
            }
            if (!tensor_floats(rd, rd.node(B + "ffn/fc1/MatMul"), 1, &k.fc1, &d) || d != std::vector<int64_t>{C, 2 * C}) return refuse_(why, B + "fc1");
            if (!tensor_floats(rd, rd.node(B + "ffn/fc2/MatMul"), 1, &k.fc2, &d) || d != std::vector<int64_t>{2 * C, C}) return refuse_(why, B + "fc2");
            if (!tensor_floats(rd, rd.node(B + "ffn/smix/Conv"), 1, &k.smix, &d) || d != std::vector<int64_t>{2 * C, 1, 3, 3})
                return refuse_(why, B + "smix");
            b->blocks.push_back(std::move(k));
        }
    }
    if (!b->heads || C % b->heads) return refuse_(why, "FDAT heads");
    b->hd = C / b->heads;
    if (b->hd > 32 || b->hd % 2 || b->ws * b->ws % 16 || b->ws * b->ws > 256) return refuse_(why, "FDAT head size / window");
    return true;
}

class VkFdat : public ImgNet {
public:
    explicit VkFdat(Device &d) : net_(d) {}
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
    int out_w() const override { return w_ * scale_; }
    int out_h() const override { return h_ * scale_; }
    size_t device_bytes() const override { return net_.device_bytes(); }
    size_t weight_bytes() const override { return net_.weight_bytes(); }
    std::string describe() const override { return what_; }
    std::string profile(Device &dev) const override { return net_.profile(dev); }
    void suffix(const std::string &s) override { what_ += s; }

private:
    TNet net_;
    int w_ = 0, h_ = 0, scale_ = 4;
    std::string what_;
};

std::unique_ptr<ImgNet> VkFdat::build(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    Reader rd(g);
    if (!rd.node("/groups/groups.0/blocks.0/n1/LayerNormalization")) return refuse_(why, "not an FDAT graph"), nullptr;
    std::unique_ptr<VkFdat> f(new VkFdat(dev));
    Conv cfirst, cafter, cu0, cu2, cu3;
    bool unshuffle = false;
    if (rd.node("/conv_first/conv_first.1/Conv")) {
        if (!rd.conv("/conv_first/conv_first.1/Conv", &cfirst, why)) return nullptr;
        unshuffle = true;
        if (cfirst.cin != 12) return refuse_(why, "FDAT unshuffle input"), nullptr;
    } else {
        if (!rd.conv("/conv_first/Conv", &cfirst, why)) return nullptr;
        if (cfirst.cin != 3) return refuse_(why, "FDAT input channels"), nullptr;
    }
    const int C = cfirst.cout;
    if (!rd.conv("/conv_after/Conv", &cafter, why) || !rd.conv("/upsampler/upsampler.0/ConvTranspose", &cu0, why) ||
        !rd.conv("/upsampler/upsampler.2/ConvTranspose", &cu2, why) || !rd.conv("/upsampler/upsampler.3/Conv", &cu3, why))
        return nullptr;
    if (cu0.cin != C || cu0.cout != C || cu2.cin != C || cu2.cout != 3 || cu3.cin != 3 || cu3.cout != 3)
        return refuse_(why, "FDAT upsampler shape (expected transpose+conv)"), nullptr;
    Body B;
    if (!parse_body(rd, "/", C, &B, why)) return nullptr;
    if (w < 1 || h < 1 || (unshuffle && (w % 2 || h % 2))) return refuse_(why, "this model needs even tile dimensions"), nullptr;
    f->w_ = w;
    f->h_ = h;
    f->scale_ = unshuffle ? 2 : 4;
    const int bw = unshuffle ? w / 2 : w, bh = unshuffle ? h / 2 : h;
    const int CP = B.CP, QS = B.QS, HS = B.HS, heads = B.heads, hd = B.hd, M = bw * bh;
    TNet &n = f->net_;

    // ---- the arena
    const VkDeviceSize Mb = (VkDeviceSize)M;
    const VkDeviceSize XS = n.place(Mb * CP * 2), G = n.place(Mb * CP * 2), X = n.place(Mb * CP * 2), N1 = n.place(Mb * CP * 2),
                       CF = n.place(Mb * CP * 2), AT = n.place(Mb * CP * 2), QKV = n.place(Mb * QS * 2), H2 = n.place(Mb * HS * 2);
    const VkDeviceSize H1 = QKV;  // the MLP's hidden layer: the attention is done with QKV before fc1
    const VkDeviceSize SUMS = n.place(CP * 4), CM = n.place(CP * 4);
    const VkDeviceSize T = n.place(Mb * CP * 2), U1 = n.place(Mb * 4 * CP * 2), U2 = n.place(Mb * 16 * 16 * 2);
    // channel attention: per chunk partials, the reduced statistics, the softmax
    const int S = heads * hd * hd + 2 * C, tpc = 256, nch = (M + tpc - 1) / tpc;  // chunks enough to fill the GPU
    const int PART = n.add_part((size_t)nch * S), STATS = n.add_part(S), ATTN = n.add_part((size_t)heads * hd * hd);

    // ---- conv_first
    {
        TNet::ConvIO io;
        io.in = unshuffle ? 2 : 1;
        io.src_in = true;
        io.cinp = 16;
        io.IH = bh;
        io.IW = bw;
        io.H = bh;
        io.W = bw;
        io.dst = XS;
        io.dst_bytes = Mb * CP * 2;
        io.dst_stride = CP;
        n.conv(conv_taps(cfirst, nullptr), cfirst.b, cfirst.cin, C, io, {32, 16}, &cfirst);
    }
    n.copy(XS, G, Mb * CP * 2);
    const auto hm = [](VkDeviceSize o, int s, int off = 0) { return hmap(o, s, off); };
    size_t bi = 0;
    for (int gi = 0; gi < B.ngroups; gi++) {
        n.copy(G, X, Mb * CP * 2);
        for (int k = 0; k < B.nblk; k++, bi++) {
            const Block &b = B.blocks[bi];
            n.layernorm(hm(X, CP), hm(N1, CP), M, C, CP, n.add_wf(b.ln1g), n.add_wf(b.ln1b), b.eps1);
            n.gemm(hm(N1, CP), M, n.add_mat(b.qkv, {}, C, 3 * C), hm(QKV, QS));
            n.dwconv(hm(N1, CP), hm(CF, CP), bh, bw, C, n.add_wf(b.dw), -1, true, b.spatial ? SUMS : ~0ull);
            if (b.spatial) {
                TWin a;
                a.H = bh;
                a.W = bw;
                a.heads = heads;
                a.hd = hd;
                a.wh = a.ww = B.ws;
                a.sy = a.sx = b.shift;
                a.scale = b.scale;
                a.bias = n.add_wf(b.bias);
                a.ko = C;
                a.vo = 2 * C;
                n.winattn(a, hm(QKV, QS), hm(AT, CP));
                n.chan_gate(SUMS, M, C, B.cgr, n.add_wf(b.cg1), -1, n.add_wf(b.cg2), -1, false, 1.f, CM, CP);
                n.gemm(hm(AT, CP), M, n.add_mat(b.wp, b.bp, C, C), hm(X, CP), 0, false, (int)(CM / 4), hm(CF, CP), hm(X, CP));
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
                cr.v[16] = S;
                cr.v[17] = nch;
                cr.v[18] = PART;
                cr.v[19] = STATS;
                n.pass_threads(7, cr, S, "chanreduce");
                TArgs sm;
                sm.v[16] = heads;
                sm.v[17] = hd;
                sm.v[18] = STATS;
                sm.v[19] = n.add_wf(b.temp);
                sm.v[20] = ATTN;
                sm.f(21, b.clip * std::pow((float)M, -b.pre));
                n.pass_threads(8, sm, heads * hd, "chansoft");
                TNet::Mat fold;
                fold.K = fold.N = C;
                fold.Kp = fold.Np = CP;
                fold.w = n.add_wh_zero((size_t)CP * CP);
                {
                    std::vector<float> bb(CP, 0.f);
                    std::copy(b.bp.begin(), b.bp.end(), bb.begin());
                    fold.bias = n.add_wf(bb);
                }
                TArgs fd;
                fd.v[16] = heads;
                fd.v[17] = hd;
                fd.v[18] = ATTN;
                fd.v[19] = n.add_wf(b.wp);
                fd.v[20] = fold.w;
                fd.v[21] = CP;
                n.pass_threads(9, fd, (long long)C * C, "chanfold");
                n.gemm(hm(QKV, QS, 2 * C), M, fold, hm(AT, CP));
                TArgs sg;
                sg.map(0, hm(AT, CP));
                sg.map(1, hm(CF, CP));
                sg.map(2, hm(X, CP));
                sg.v[16] = M;
                sg.v[17] = C;
                sg.v[18] = n.add_wf(b.sg);
                const long long sgg = (M + 7) / 8;  // 8 tokens a workgroup
                const uint32_t sgx = (uint32_t)std::min<long long>(sgg, 4096);
                n.pass(11, sg, sgx, (uint32_t)((sgg + sgx - 1) / sgx), "aimsg");
            }
            n.layernorm(hm(X, CP), hm(N1, CP), M, C, CP, n.add_wf(b.ln2g), n.add_wf(b.ln2b), b.eps2);
            n.gemm(hm(N1, CP), M, n.add_mat(b.fc1, {}, C, 2 * C), hm(H1, HS), 1);
            n.dwconv(hm(H1, HS), hm(H2, HS), bh, bw, 2 * C, n.add_wf(b.smix), -1, false);
            n.gemm(hm(H2, HS), M, n.add_mat(b.fc2, {}, 2 * C, C), hm(X, CP), 0, false, -1, hm(X, CP));
        }
        // G = conv(X) + G
        TNet::ConvIO io;
        io.src = X;
        io.src_bytes = Mb * CP * 2;
        io.src_stride = CP;
        io.cinp = CP;
        io.IH = io.H = bh;
        io.IW = io.W = bw;
        io.dst = G;
        io.dst_bytes = Mb * CP * 2;
        io.dst_stride = CP;
        io.nres = 1;
        io.r1 = G;
        io.r1_bytes = Mb * CP * 2;
        io.r1_stride = CP;
        n.conv(conv_taps(B.gconvs[gi], nullptr), B.gconvs[gi].b, C, C, io, {64, 48, 32, 16}, &B.gconvs[gi]);
    }
    // conv_after + the shallow features
    {
        TNet::ConvIO io;
        io.src = G;
        io.src_bytes = Mb * CP * 2;
        io.src_stride = CP;
        io.cinp = CP;
        io.IH = io.H = bh;
        io.IW = io.W = bw;
        io.dst = T;
        io.dst_bytes = Mb * CP * 2;
        io.dst_stride = CP;
        io.nres = 1;
        io.r1 = XS;
        io.r1_bytes = Mb * CP * 2;
        io.r1_stride = CP;
        n.conv(conv_taps(cafter, nullptr), cafter.b, C, C, io, {64, 48, 32, 16}, &cafter);
    }
    // the upsampler: transposed conv (GELU), transposed conv, 3x3 conv into the output planes
    std::vector<uint16_t> pw[2][2];
    std::vector<float> pb[2][2];
    for (int py = 0; py < 2; py++) {
        phase_weights(cu0, 2, py, CP, &pw[0][py], &pb[0][py]);
        phase_weights(cu2, 2, py, 16, &pw[1][py], &pb[1][py]);
    }
    for (int py = 0; py < 2; py++) {
        TNet::ConvIO io;
        io.out = 4;
        io.act = 1;
        io.src = T;
        io.src_bytes = Mb * CP * 2;
        io.src_stride = CP;
        io.cinp = CP;
        io.IH = io.H = bh;
        io.IW = io.W = bw;
        io.dst = U1;
        io.dst_bytes = Mb * 4 * CP * 2;
        io.dst_stride = CP;
        io.sy = py;
        n.conv(pw[0][py], pb[0][py], C, 2 * CP, io, {64, 48, 32, 16}, &pw[0][py]);
    }
    for (int py = 0; py < 2; py++) {
        TNet::ConvIO io;
        io.out = 4;
        io.src = U1;
        io.src_bytes = Mb * 4 * CP * 2;
        io.src_stride = CP;
        io.cinp = CP;
        io.IH = io.H = 2 * bh;
        io.IW = io.W = 2 * bw;
        io.dst = U2;
        io.dst_bytes = Mb * 16 * 16 * 2;
        io.dst_stride = 16;
        io.sy = py;
        n.conv(pw[1][py], pb[1][py], C, 32, io, {32}, &pw[1][py]);
    }
    {
        TNet::ConvIO io;
        io.out = 3;
        io.dst_out = true;
        io.src = U2;
        io.src_bytes = Mb * 16 * 16 * 2;
        io.src_stride = 16;
        io.cinp = 16;
        io.IH = io.H = 4 * bh;
        io.IW = io.W = 4 * bw;
        n.conv(conv_taps(cu3, nullptr), cu3.b, 3, 3, io, {16}, &cu3);
    }
    if (!n.finish((VkDeviceSize)3 * w * h * 2, (VkDeviceSize)3 * f->out_w() * f->out_h() * 2, why)) return nullptr;
    char buf[200];
    bool shifted = false;
    for (const Block &b : B.blocks) shifted |= b.shift != 0;
    std::snprintf(buf, sizeof buf, "FDAT %dx, %d groups x %d blocks, %d channels, %d heads, %dx%d windows%s; Vulkan %s", f->scale_, B.ngroups,
                  B.nblk, C, heads, B.ws, B.ws, shifted ? " (shifted)" : "", family_name(dev.family()));
    f->what_ = buf;
    return std::unique_ptr<ImgNet>(f.release());
}

}  // namespace

std::unique_ptr<ImgNet> build_fdat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why)
{
    return VkFdat::build(dev, g, w, h, why);
}

}  // namespace ajivk
