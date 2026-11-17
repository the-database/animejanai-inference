// native/hat_graph.h — HAT (Hybrid Attention Transformer) found by its structure, for the CUDA
// (native/hat.cpp) and Vulkan (vk/hat.cpp) builds.
//
// Exports differ in everything but the ops' order: module names (or none, "/MatMul_12"), LayerNorm as
// one op or spelled out (opset < 17), GELU as one op or Erf arithmetic, the position bias and the
// shifted windows' mask as constants (fixed-size exports) or computed (a Gather of the bias table, the
// mask from the input's shape), weights transposed by a node. So the matcher walks the data path's ops
// in order, as the forward pass runs them:
//
//   conv_first, patch_embed's norm; per residual group: HABs (norm1, the CAB's four convs, qkv, the
//   attention scale, the bias [, the mask], Softmax, proj, the conv scale, norm2, fc1, fc2), the OCAB
//   (norm1, qkv, scale, bias, Softmax, proj, norm2, fc1, fc2), the group conv; the final norm,
//   conv_after_body, conv_before_upsample, (conv, DepthToSpace 2) per 2x of the scale, conv_last;
//
// with the image range around it (native/export_forms.h). The position biases are evaluated on the host.
#pragma once
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "native/export_forms.h"
#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "native/static_eval.h"

namespace ajinative {

using HatLn = onnxread::LnW;
struct HatLin {  // y = x w + b, w [K][N]
    std::vector<float> w, b;
    int K = 0, N = 0;
};
struct HatAttn {
    HatLin qkv, proj;
    float scale = 1.f;
    std::vector<float> bias;  // [heads][Tq][Tk]
    int shift = 0;            // shifted windows (with Swin's mask): by half a window
};
struct HatHab {
    HatLn n1, n2;
    HatAttn at;
    HatLin fc1, fc2;
    onnxread::Conv cab0, cab2;                  // 3x3 C -> cr, GELU, 3x3 cr -> C
    std::vector<float> ca1, cab1, ca2, cab2b;  // the channel attention: 1x1 C -> cs, ReLU, 1x1 cs -> C, sigmoid
    float conv_scale = 1.f;
};
struct HatOcab {
    HatLn n1, n2;
    HatAttn at;
    HatLin fc1, fc2;
};
struct HatGroup {
    std::vector<HatHab> habs;
    HatOcab oc;
    onnxread::Conv conv;
};
struct HatGraph {
    int C = 0, heads = 0, ws = 0, kws = 0, kpad = 0, hid = 0, cr = 0, cs = 0, nup = 0;
    onnxread::ImgRange ir;  // conv_first's weights carry the range; conv_last's the inverse and the mean
    onnxread::Conv first, after, cbu, last;
    std::vector<onnxread::Conv> ups;  // 64 -> 256, then DepthToSpace 2 (CRD)
    HatLn pe, fin;
    std::vector<HatGroup> groups;
    int scale() const { return 1 << nup; }
    int blocks() const { return groups.empty() ? 0 : (int)groups[0].habs.size(); }
};

namespace hat_graph {

using onnxread::Reader;

inline bool no(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

struct Ev {
    enum K { LN, MM, CV, SOFTMAX, D2S, MULC, ADD } k;
    const OnnxNode *n;
    HatLn ln;  // LN
};

inline bool find(const Reader &rd, HatGraph *hg, std::string *why)
{
    using namespace onnxread;
    const OnnxGraph &g = rd.g;
    if (g.inputs.empty()) return no(why, "not a HAT graph");
    const Lns lns(rd);
    // the data path
    std::set<std::string> data{g.inputs[0]}, mm_out;
    std::vector<Ev> ev;
    const OnnxNode *sub = nullptr;
    for (const OnnxNode &n : g.nodes) {
        if (n.op == "Constant" || n.op == "Shape" || n.op == "Size") continue;
        bool d = false;
        for (const std::string &i : n.in) d |= data.count(i) > 0;
        if (!d) continue;
        for (const std::string &o : n.out) data.insert(o);
        if (lns.is_ln(n)) {
            Ev e{Ev::LN, &n, {}};
            if (!lns.weights(rd, &n, &e.ln)) return no(why, "HAT LayerNorm weights");
            ev.push_back(e);
            continue;
        }
        if (lns.span.count(&n)) continue;
        int nd = 0, nc = 0;
        for (const std::string &i : n.in) {
            if (i.empty()) continue;
            nd += data.count(i) > 0;
            nc += rd.tensor(i) != nullptr;
        }
        if (n.op == "MatMul" && nc) {
            ev.push_back({Ev::MM, &n, {}});
            mm_out.insert(n.out[0]);
        } else if (n.op == "Conv") {
            ev.push_back({Ev::CV, &n, {}});
        } else if (n.op == "Softmax") {
            ev.push_back({Ev::SOFTMAX, &n, {}});
        } else if (n.op == "DepthToSpace") {
            ev.push_back({Ev::D2S, &n, {}});
        } else if (n.op == "Sub" && nc && !sub) {
            sub = &n;
        } else if (n.op == "Mul" && nc) {
            if (sub && n.in[0] == sub->out[0]) continue;  // the image range
            if (gelu_const_mul(rd, n)) continue;          // a spelled-out GELU
            ev.push_back({Ev::MULC, &n, {}});
        } else if (n.op == "Add" && nd == 1 && n.in.size() == 2) {
            bool skip_add = false;
            for (const std::string &i : n.in) {
                skip_add |= mm_out.count(i) > 0;  // a linear's bias
                auto p = rd.prod.find(i);
                skip_add |= p != rd.prod.end() && p->second->op == "Erf";  // a spelled-out GELU's 1 +
            }
            if (!skip_add) ev.push_back({Ev::ADD, &n, {}});
        }
    }
    size_t at = 0;
    auto kind = [&](size_t i) { return i < ev.size() ? (int)ev[i].k : -1; };
    auto take = [&](Ev::K k) -> const Ev * { return kind(at) == k ? &ev[at++] : nullptr; };
    auto take_n = [&](Ev::K k) -> const OnnxNode * {
        const Ev *e = take(k);
        return e ? e->n : nullptr;
    };
    // head
    if (!img_range_head(rd, sub, &hg->ir)) return no(why, "not a HAT graph");
    const OnnxNode *cf = take_n(Ev::CV);
    if (!cf || !rd.conv(cf, &hg->first, why) || hg->first.cin != 3 || hg->first.k != 3 || hg->first.lrelu) return no(why, "not a HAT graph");
    img_range_first(hg->ir, &hg->first);
    const int C = hg->C = hg->first.cout;
    auto lnorm = [&](HatLn *l) {
        const Ev *e = take(Ev::LN);
        if (!e || (int)e->ln.g.size() != C || (int)e->ln.b.size() != C) return false;
        *l = e->ln;
        return true;
    };
    auto linear = [&](int K, int N, HatLin *l) {
        const OnnxNode *mm = take_n(Ev::MM);
        std::vector<int64_t> d;
        if (!mm || !tensor_floats(rd, mm, 1, &l->w, &d) || d != std::vector<int64_t>{K, N}) return false;
        const OnnxNode *a = consumer(rd, mm->out[0], "Add");
        if (!a || !init_input(rd, a, &l->b) || (int)l->b.size() != N) return false;
        l->K = K;
        l->N = N;
        return true;
    };
    auto scalar = [&](float *v) {
        const OnnxNode *m = take_n(Ev::MULC);
        std::vector<float> x;
        if (!m || !init_input(rd, m, &x) || x.size() != 1) return false;
        *v = x[0];
        return true;
    };
    // the attention after qkv: the scale, the bias [, the mask], Softmax, proj
    StaticEval se(rd);
    auto attention = [&](HatAttn *a, bool overlap, const std::string &where) {
        if (!scalar(&a->scale)) return no(why, where + ": attention scale");
        const OnnxNode *ab = take_n(Ev::ADD);
        if (!ab) return no(why, where + ": position bias");
        a->shift = take_n(Ev::ADD) ? 1 : 0;  // the mask: set to half the window below
        if (overlap && a->shift) return no(why, where + ": a mask on the OCAB");
        if (!take(Ev::SOFTMAX)) return no(why, where + ": attention");
        std::string bin;
        for (const std::string &i : ab->in)
            if (!data.count(i)) bin = i;
        HT bias;
        if (bin.empty() || !se.eval(bin, &bias)) return no(why, where + ": position bias (not evaluable)");
        while (bias.dims.size() > 3 && bias.dims[0] == 1) bias.dims.erase(bias.dims.begin());
        if (bias.dims.size() != 3) return no(why, where + ": position bias shape");
        const int heads = (int)bias.dims[0], Tq = (int)bias.dims[1], Tk = (int)bias.dims[2];
        const int ws = (int)std::lround(std::sqrt((double)Tq)), kws = (int)std::lround(std::sqrt((double)Tk));
        if (heads < 1 || ws < 1 || ws * ws != Tq || kws * kws != Tk || (overlap ? kws <= ws || (kws - ws) % 2 : kws != ws))
            return no(why, where + ": position bias shape");
        if (hg->heads && (hg->heads != heads || hg->ws != ws)) return no(why, where + ": heads or window differ between blocks");
        hg->heads = heads;
        hg->ws = ws;
        if (overlap) {
            if (hg->kws && hg->kws != kws) return no(why, where + ": overlap windows differ between groups");
            hg->kws = kws;
            hg->kpad = (kws - ws) / 2;
        }
        a->bias.resize(bias.v.size());
        for (size_t i = 0; i < bias.v.size(); i++) a->bias[i] = (float)bias.v[i];
        if (a->shift) a->shift = ws / 2;
        if (!linear(C, C, &a->proj)) return no(why, where + ": proj");
        return true;
    };
    auto mlp = [&](HatLn *n2, HatLin *fc1, HatLin *fc2, const std::string &where) {
        if (!lnorm(n2)) return no(why, where + ": norm2");
        if (kind(at) != Ev::MM) return no(why, where + ": mlp");
        std::vector<int64_t> d;
        std::vector<float> w;
        if (!tensor_floats(rd, ev[at].n, 1, &w, &d) || d.size() != 2 || d[0] != C || d[1] < 1) return no(why, where + ": mlp");
        if (hg->hid && hg->hid != (int)d[1]) return no(why, where + ": mlp widths differ");
        hg->hid = (int)d[1];
        if (!linear(C, hg->hid, fc1) || !linear(hg->hid, C, fc2)) return no(why, where + ": mlp");
        return true;
    };
    auto conv_scale = [&](float *v) {  // a scalar in (0, 1)
        std::vector<float> x;
        if (kind(at) != Ev::MULC || !init_input(rd, ev[at].n, &x) || x.size() != 1 || !(x[0] > 0.f && x[0] < 1.f)) return false;
        at++;
        *v = x[0];
        return true;
    };

    if (!lnorm(&hg->pe)) return no(why, "not a HAT graph (patch_embed norm)");
    // a HAB opens with norm1 and the CAB's four convs, then qkv; the OCAB with norm1 and qkv
    auto hab_next = [&] {
        return kind(at) == Ev::LN && kind(at + 1) == Ev::CV && kind(at + 2) == Ev::CV && kind(at + 3) == Ev::CV && kind(at + 4) == Ev::CV &&
               (kind(at + 5) == Ev::MM || (kind(at + 5) == Ev::MULC && kind(at + 6) == Ev::MM));
    };
    while (hab_next()) {
        HatGroup gr;
        while (hab_next()) {
            HatHab hb;
            const std::string where = "HAT block " + std::to_string(gr.habs.size()) + " of group " + std::to_string(hg->groups.size());
            lnorm(&hb.n1);
            const OnnxNode *c0 = take_n(Ev::CV), *c2 = take_n(Ev::CV), *a1 = take_n(Ev::CV), *a3 = take_n(Ev::CV);
            if (!rd.conv(c0, &hb.cab0, why) || !rd.conv(c2, &hb.cab2, why) || hb.cab0.cin != C || hb.cab0.k != 3 || hb.cab2.k != 3 ||
                hb.cab2.cout != C || hb.cab2.cin != hb.cab0.cout || hb.cab0.lrelu || hb.cab2.lrelu)
                return no(why, where + ": CAB");
            if (hg->cr && hg->cr != hb.cab0.cout) return no(why, where + ": CAB widths differ");
            hg->cr = hb.cab0.cout;
            std::vector<int64_t> d, d3;
            if (!tensor_floats(rd, a1, 1, &hb.ca1, &d) || d.size() != 4 || d[0] < 1 || d[1] != C || d[2] != 1 || d[3] != 1 ||
                !tensor_floats(rd, a3, 1, &hb.ca2, &d3) || d3 != std::vector<int64_t>{C, d[0], 1, 1} || a1->in.size() < 3 ||
                !rd.floats(a1->in[2], &hb.cab1) || (int64_t)hb.cab1.size() != d[0] || a3->in.size() < 3 || !rd.floats(a3->in[2], &hb.cab2b) ||
                (int)hb.cab2b.size() != C)
                return no(why, where + ": CAB channel attention");
            if (hg->cs && hg->cs != (int)hb.cab1.size()) return no(why, where + ": CAB channel attention widths differ");
            hg->cs = (int)hb.cab1.size();
            const bool early = conv_scale(&hb.conv_scale);  // the conv scale, before the attention in some exports
            if (!linear(C, 3 * C, &hb.at.qkv)) return no(why, where + ": qkv");
            if (!attention(&hb.at, false, where)) return false;
            if (!early && !conv_scale(&hb.conv_scale)) return no(why, where + ": conv scale");
            if (!mlp(&hb.n2, &hb.fc1, &hb.fc2, where)) return false;
            gr.habs.push_back(std::move(hb));
        }
        if (!hg->groups.empty() && gr.habs.size() != hg->groups[0].habs.size()) return no(why, "HAT groups of different depths");
        const std::string where = "HAT OCAB of group " + std::to_string(hg->groups.size());
        if (!lnorm(&gr.oc.n1) || !linear(C, 3 * C, &gr.oc.at.qkv)) return no(why, where + ": qkv");
        if (!attention(&gr.oc.at, true, where) || !mlp(&gr.oc.n2, &gr.oc.fc1, &gr.oc.fc2, where)) return false;
        const OnnxNode *gc = take_n(Ev::CV);
        if (!rd.conv(gc, &gr.conv, why) || gr.conv.cin != C || gr.conv.cout != C || gr.conv.k != 3 || gr.conv.lrelu)
            return no(why, "HAT group conv");
        hg->groups.push_back(std::move(gr));
    }
    if (hg->groups.empty()) return no(why, "not a HAT graph (no residual groups)");
    if (C % hg->heads) return no(why, "HAT heads");
    // the tail
    if (!lnorm(&hg->fin)) return no(why, "HAT final norm");
    const OnnxNode *ca = take_n(Ev::CV), *cb = take_n(Ev::CV);
    if (!rd.conv(ca, &hg->after, why) || hg->after.cin != C || hg->after.cout != C || hg->after.k != 3 || hg->after.lrelu)
        return no(why, "HAT conv_after_body");
    if (!rd.conv(cb, &hg->cbu, why) || !hg->cbu.lrelu || hg->cbu.cin != C || hg->cbu.cout != 64 || hg->cbu.k != 3)
        return no(why, "HAT conv_before_upsample");
    while (kind(at) == Ev::CV && kind(at + 1) == Ev::D2S) {
        onnxread::Conv u;
        const OnnxNode *un = take_n(Ev::CV), *ds = take_n(Ev::D2S);
        if (!rd.conv(un, &u, why) || u.cin != 64 || u.cout != 256 || u.k != 3 || u.lrelu || ds->attr_i("blocksize", 0) != 2 ||
            ds->attr_s("mode", "DCR") != "CRD")
            return no(why, "HAT upsample (pixel shuffle 2 steps only)");
        hg->ups.push_back(std::move(u));
    }
    hg->nup = (int)hg->ups.size();
    if (hg->nup > 2) return no(why, "HAT scale above 4x");
    const OnnxNode *cl = take_n(Ev::CV);
    if (!rd.conv(cl, &hg->last, why) || hg->last.cin != 64 || hg->last.cout != 3 || hg->last.k != 3 || hg->last.lrelu) return no(why, "HAT conv_last");
    while (kind(at) == Ev::ADD || kind(at) == Ev::MULC) at++;  // the image range
    if (at != ev.size()) return no(why, "HAT: ops after conv_last");
    if (!img_range_tail(rd, cl->out[0], &hg->last, &hg->ir)) return no(why, "HAT output range");
    return true;
}

}  // namespace hat_graph

inline bool find_hat(const onnxread::Reader &rd, HatGraph *hg, std::string *why) { return hat_graph::find(rd, hg, why); }

}  // namespace ajinative
