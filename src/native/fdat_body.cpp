// native/fdat_body.cpp — see fdat_body.h.
#include "native/fdat_body.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "native/fdat_kernels.h"

namespace ajinative {

using namespace onnxread;

bool FdatBody::present(const Reader &rd, const std::string &prefix)
{
    return rd.node(prefix + "groups/groups.0/blocks.0/n1/LayerNormalization") != nullptr;
}

bool FdatBody::parse(const Reader &rd, const std::string &prefix, int C_, std::string *why)
{
    C = C_;
    CP = pad16(C);
    QS = pad16(3 * C);
    HS = pad16(2 * C);
    const std::string P = prefix + "groups/groups.";
    while (rd.node(P + std::to_string(ngroups) + "/conv/Conv")) ngroups++;
    while (rd.node(P + "0/blocks." + std::to_string(nblk) + "/n1/LayerNormalization")) nblk++;
    if (!ngroups || !nblk) return refuse(why, "FDAT groups");
    for (int gi = 0; gi < ngroups; gi++) {
        const std::string Gp = P + std::to_string(gi) + "/";
        Conv cg;
        if (!rd.conv(Gp + "conv/Conv", &cg, why)) return false;
        if (cg.cin != C || cg.cout != C || cg.lrelu) return refuse(why, "FDAT group conv");
        GOp op;
        op.w = res.gup(conv_taps(cg, nullptr), cg.b, C, CP, C, CP);
        gconvs.push_back(op);
        for (int bi = 0; bi < nblk; bi++) {
            const std::string B = Gp + "blocks." + std::to_string(bi) + "/";
            Block b;
            std::vector<float> v, v2;
            std::vector<int64_t> d;
            const OnnxNode *ln1 = rd.node(B + "n1/LayerNormalization"), *ln2 = rd.node(B + "n2/LayerNormalization");
            if (!tensor_floats(rd, ln1, 1, &v) || !tensor_floats(rd, ln1, 2, &v2) || (int)v.size() != C) return refuse(why, B + "n1");
            b.ln1g = res.dfl(v);
            b.ln1b = res.dfl(v2);
            b.eps1 = ln1->attr("epsilon") ? ln1->attr("epsilon")->f : 1e-5f;
            if (!tensor_floats(rd, ln2, 1, &v) || !tensor_floats(rd, ln2, 2, &v2)) return refuse(why, B + "n2");
            b.ln2g = res.dfl(v);
            b.ln2b = res.dfl(v2);
            b.eps2 = ln2->attr("epsilon") ? ln2->attr("epsilon")->f : 1e-5f;
            // no qkv / fc biases in these exports (qkv_bias=False); refuse rather than drop one
            if (rd.node(B + "attn/qkv/Add") || rd.node(B + "ffn/fc1/Add") || rd.node(B + "ffn/fc2/Add"))
                return refuse(why, B + "qkv/ffn bias");
            if (!tensor_floats(rd, rd.node(B + "attn/qkv/MatMul"), 1, &v, &d) || d != std::vector<int64_t>{C, 3 * C})
                return refuse(why, B + "qkv");
            b.qkv = res.mat(v, nullptr, C, 3 * C);
            std::vector<float> wp, bp;
            if (!tensor_floats(rd, rd.node(B + "attn/proj/MatMul"), 1, &wp, &d) || d != std::vector<int64_t>{C, C} ||
                !init_input(rd, rd.node(B + "attn/proj/Add"), &bp) || (int)bp.size() != C)
                return refuse(why, B + "proj");
            if (!tensor_floats(rd, rd.node(B + "conv/conv.0/Conv"), 1, &v, &d) || d != std::vector<int64_t>{C, 1, 3, 3})
                return refuse(why, B + "conv");
            b.dw = res.dfl(v);
            // The attention type from the graph: Softmax <- Add(bias [heads,T,T]) <- MatMul <- Mul(scale) (spatial),
            // or Softmax <- Mul(temperature [heads,1,1]) (channel).
            const OnnxNode *sm = rd.node(B + "attn/Softmax");
            const OnnxNode *pre = sm && !sm->in.empty() && rd.prod.count(sm->in[0]) ? rd.prod.at(sm->in[0]) : nullptr;
            if (!pre) return refuse(why, B + "attention");
            if (pre->op == "Add") {
                b.spatial = true;
                if (!init_input(rd, pre, &v, &d) || d.size() != 3 || d[1] != d[2]) return refuse(why, B + "bias");
                const int nh = (int)d[0], T = (int)d[1], wsz = (int)std::lround(std::sqrt((double)T));
                if (wsz * wsz != T || C % nh) return refuse(why, B + "window");
                if ((heads && heads != nh) || (ws != wsz && bi + gi > 0)) return refuse(why, "FDAT mixed heads/windows");
                heads = nh;
                ws = wsz;
                b.bias = res.dfl(v);
                std::vector<float> sc;
                const OnnxNode *mm = nullptr, *ml = nullptr;
                for (const std::string &i : pre->in)
                    if (rd.prod.count(i) && rd.prod.at(i)->op == "MatMul") mm = rd.prod.at(i);
                if (mm)
                    for (const std::string &i : mm->in)
                        if (rd.prod.count(i) && rd.prod.at(i)->op == "Mul") ml = rd.prod.at(i);
                if (!init_input(rd, ml, &sc) || sc.size() != 1) return refuse(why, B + "scale");
                b.scale = sc[0];
                // shifted windows: the roll before the partition is Slice(start = shift) + Slice(0, shift) + Concat
                if (const OnnxNode *sl = rd.node(B + "attn/Slice")) {
                    std::vector<int64_t> st;
                    const OnnxTensor *t = sl->in.size() > 1 ? rd.tensor(sl->in[1]) : nullptr;
                    if (!t || !t->ints(&st) || st.size() != 1 || st[0] <= 0 || st[0] >= ws) return refuse(why, B + "window shift");
                    b.shift = (int)st[0];
                }
                if (!tensor_floats(rd, rd.node(B + "inter/cg/cg.1/Conv"), 1, &v, &d) || d.size() != 4 || d[1] != C)
                    return refuse(why, B + "cg");
                cgr = (int)d[0];
                b.cg1 = res.dfl(v);
                if (!tensor_floats(rd, rd.node(B + "inter/cg/cg.3/Conv"), 1, &v, &d) || d[0] != C || d[1] != cgr) return refuse(why, B + "cg");
                b.cg2 = res.dfl(v);
                b.proj = res.mat(wp, &bp, C, C);
            } else if (pre->op == "Mul" && rd.node(B + "attn/ReduceL2")) {
                b.spatial = false;
                if (!init_input(rd, pre, &v, &d) || d.size() != 3 || d[1] != 1 || d[2] != 1) return refuse(why, B + "temperature");
                if (heads && heads != (int)d[0]) return refuse(why, "FDAT mixed heads");
                heads = (int)d[0];
                b.temp = res.dfl(v);
                std::vector<float> cl, pw;
                const OnnxNode *clip = rd.node(B + "attn/Clip");
                if (!clip || clip->in.size() < 2 || !rd.floats(clip->in[1], &cl) || cl.size() != 1) return refuse(why, B + "norm clip");
                b.clip = cl[0];
                if (const OnnxNode *pn = rd.node(B + "attn/Pow")) {
                    if (pn->in.size() < 2 || !rd.floats(pn->in[1], &pw) || pw.size() != 1) return refuse(why, B + "norm pre-scale");
                    b.pre = pw[0];
                }
                if (!tensor_floats(rd, rd.node(B + "inter/sg/sg.0/Conv"), 1, &v, &d) || d[0] != 1 || d[1] != C) return refuse(why, B + "sg");
                b.sg = res.dfl(v);
                b.wp = res.dfl(wp);
                b.bias = res.dfl(bp);  // copied into the shared fold matrix's bias per block
            } else {
                return refuse(why, B + "attention type");
            }
            if (!tensor_floats(rd, rd.node(B + "ffn/fc1/MatMul"), 1, &v, &d) || d != std::vector<int64_t>{C, 2 * C}) return refuse(why, B + "fc1");
            b.fc1 = res.mat(v, nullptr, C, 2 * C);
            if (!tensor_floats(rd, rd.node(B + "ffn/fc2/MatMul"), 1, &v, &d) || d != std::vector<int64_t>{2 * C, C}) return refuse(why, B + "fc2");
            b.fc2 = res.mat(v, nullptr, 2 * C, C);
            if (!tensor_floats(rd, rd.node(B + "ffn/smix/Conv"), 1, &v, &d) || d != std::vector<int64_t>{2 * C, 1, 3, 3}) return refuse(why, B + "smix");
            b.smix = res.dfl(v);
            blocks.push_back(b);
        }
    }
    if (!heads || C % heads) return refuse(why, "FDAT heads");
    hd = C / heads;
    if (hd > 32 || hd % 2 || ws * ws % 16 || ws * ws > 256) return refuse(why, "FDAT head size / window");
    if (res.failed) return refuse(why, "weight upload failed");
    for (const Block &b : blocks)
        if (!b.spatial) {
            float *zb = (float *)res.alloc((size_t)C * 4);  // a bias buffer: each block copies its own in
            if (!zb || tmat_alloc(foldm, C, C, zb) != cudaSuccess) return refuse(why, "weight upload failed");
            break;
        }
    return true;
}

void FdatBody::carve(Bump &b, size_t M)
{
    const size_t CPb = (size_t)CP * 2;
    void **bufs[8] = {&G, &X, &N1, &CF, &AT, &QKV, &H1, &H2};  // G first: it stays valid while a caller reuses the rest
    const size_t sizes[8] = {M * CPb, M * CPb, M * CPb, M * CPb, M * CPb, M * QS * 2, M * HS * 2, M * HS * 2};
    for (int i = 0; i < 8; i++) {
        void *p = b.take(sizes[i]);
        if (b.base) *bufs[i] = p;
    }
}

size_t FdatBody::scratch_bytes(int bw, int bh) const
{
    Bump b;
    const_cast<FdatBody *>(this)->carve(b, (size_t)bw * bh);  // counting only (no base): nothing assigned
    return b.off;
}

bool FdatBody::setup(int bw, int bh, cudaStream_t s, std::string *why, void *scratch)
{
    const size_t M = (size_t)bw * bh;
    maxm = (int)M;
    Bump b;
    shared = scratch != nullptr;
    b.base = shared ? (char *)scratch : (char *)res.alloc(scratch_bytes(bw, bh));
    if (!b.base) return refuse(why, "out of GPU memory");
    carve(b, M);
    if (shared) cudaMemset(scratch, 0, b.off);
    sums = (float *)res.alloc(CP * 4);
    cm = (float *)res.alloc(CP * 4);
    stats = (float *)res.alloc((size_t)(heads * hd * hd + 2 * C) * 4);
    attnb = (float *)res.alloc((size_t)heads * hd * hd * 4);
    fold = (float *)res.alloc((size_t)C * C * 4);
    if (res.failed) return refuse(why, "out of GPU memory");
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        const int w = sz ? std::min(bw, 32) : bw, h = sz ? std::min(bh, 32) : bh;
        for (int g = 0; g < ngroups; g++) {
            GConvArgs a = ga(X, CP, h, w, G, CP, h, w);
            a.r1 = G;
            a.r1_stride = CP;
            ops.push_back({&gconvs[g], a});
        }
    }
    ws_floats = gworkspace_max(ops);
    if (ws_floats && !(wsp = (float *)res.alloc(ws_floats * 4))) return refuse(why, "out of GPU memory");
    ops.resize(ngroups);  // the full size's
    for (auto &o : ops) {
        o.second.ws = wsp;
        o.second.ws_floats = ws_floats;
    }
    return gprepare(ops, s, why);
}

std::string FdatBody::describe() const
{
    char buf[128];
    bool shifted = false;
    for (const Block &b : blocks) shifted |= b.shift != 0;
    snprintf(buf, sizeof(buf), "%d groups x %d blocks, %d channels, %d heads, %dx%d windows%s", ngroups, nblk, C, heads, ws, ws,
             shifted ? " (shifted)" : "");
    return buf;
}

cudaError_t FdatBody::run(const void *x0, int bw, int bh, cudaStream_t s, StepProf *pr)
{
    const int M = bw * bh;
    if (M > maxm) return cudaErrorInvalidValue;
    auto mark = [&](int k) {
        if (pr) pr->mark(k);
    };
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return e_; } while (0)
    CK(cudaMemcpyAsync(G, x0, (size_t)M * CP * 2, cudaMemcpyDeviceToDevice, s));
    if (shared) {  // padding channels the kernels don't write
        CK(cudaMemsetAsync(CF, 0, (size_t)M * CP * 2, s));
        CK(cudaMemsetAsync(AT, 0, (size_t)M * CP * 2, s));
    }
    for (int gi = 0; gi < ngroups; gi++) {
        CK(cudaMemcpyAsync(X, G, (size_t)M * CP * 2, cudaMemcpyDeviceToDevice, s));
        mark(P_IO);
        for (int bi = 0; bi < nblk; bi++) {
            const Block &b = blocks[(size_t)gi * nblk + bi];
            CK(tlayernorm(X, CP, N1, CP, M, C, CP, b.ln1g, b.ln1b, b.eps1, s));
            mark(P_LN);
            TGemm q;
            q.a = N1; q.sa = CP; q.M = M; q.out = QKV; q.so = QS;
            CK(tgemm(*b.qkv, q, s));
            mark(P_QKV);
            if (b.spatial) CK(cudaMemsetAsync(sums, 0, CP * 4, s));
            CK(tdwconv(N1, CP, CF, CP, bh, bw, C, b.dw, true, b.spatial ? sums : nullptr, s));
            mark(P_DW);
            if (b.spatial) {
                TWinAttn a;
                a.qkv = QKV; a.sq = QS; a.qo = 0; a.ko = C; a.vo = 2 * C; a.H = bh; a.W = bw; a.heads = heads; a.hd = hd;
                a.wh = a.ww = ws; a.scale = b.scale; a.bias = b.bias; a.out = AT; a.so = CP;
                a.shift_y = a.shift_x = b.shift;
                CK(twinattn(a, s));
                mark(P_WIN);
                CK(fdat_aim_cg(sums, M, C, cgr, b.cg1, b.cg2, cm, CP, s));
                mark(P_GATE);
                TGemm pj;
                pj.a = AT; pj.sa = CP; pj.M = M; pj.out = X; pj.so = CP; pj.colscale = cm;
                pj.add1 = CF; pj.s1 = CP; pj.add2 = X; pj.s2 = CP;
                CK(tgemm(*b.proj, pj, s));
                mark(P_PROJ);
            } else {
                CK(cudaMemsetAsync(stats, 0, (size_t)(heads * hd * hd + 2 * C) * 4, s));
                CK(tchanstats(QKV, QS, 0, C, M, heads, hd, stats, s));
                mark(P_CSTATS);
                CK(fdat_chan_soft(stats, b.temp, heads, hd, b.clip * std::pow((float)M, -b.pre), attnb, s));
                CK(fdat_chan_fold(attnb, b.wp, heads, hd, fold, s));
                CK(tmat_pack(foldm, fold, s));
                CK(cudaMemcpyAsync(foldm.bias, b.bias, (size_t)C * 4, cudaMemcpyDeviceToDevice, s));
                mark(P_CFOLD);
                TGemm pj;
                pj.a = (const uint16_t *)QKV + 2 * C; pj.sa = QS; pj.M = M; pj.out = AT; pj.so = CP;
                CK(tgemm(foldm, pj, s));
                mark(P_CPROJ);
                CK(fdat_aim_sg(AT, CF, b.sg, X, CP, M, C, s));
                mark(P_GATE);
            }
            CK(tlayernorm(X, CP, N1, CP, M, C, CP, b.ln2g, b.ln2b, b.eps2, s));
            mark(P_LN);
            TGemm f1;
            f1.a = N1; f1.sa = CP; f1.M = M; f1.out = H1; f1.so = HS; f1.act = TAct::Gelu;
            CK(tgemm(*b.fc1, f1, s));
            mark(P_FC1);
            CK(tdwconv(H1, HS, H2, HS, bh, bw, 2 * C, b.smix, false, s));
            mark(P_SMIX);
            TGemm f2;
            f2.a = H2; f2.sa = HS; f2.M = M; f2.out = X; f2.so = CP; f2.add1 = X; f2.s1 = CP;
            CK(tgemm(*b.fc2, f2, s));
            mark(P_FC2);
        }
        GConvArgs a = ga(X, CP, bh, bw, G, CP, bh, bw);  // G = conv(X) + G
        a.r1 = G;
        a.r1_stride = CP;
        a.ws = wsp;
        a.ws_floats = ws_floats;
        CK(glaunch(*gconvs[gi].w, a, gconvs[gi].variant, s));
        mark(P_CONV);
    }
#undef CK
    return cudaSuccess;
}

}  // namespace ajinative
