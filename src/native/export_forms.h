// native/export_forms.h — the forms exporters give the same model, for the structural parsers (DAT,
// HAT; CUDA and Vulkan): the image range the SwinIR family wraps its network in (conv_first sees
// (x - mean) * range, conv_last's output y leaves as y / range + mean, at times clipped to [0, 1];
// exports keep the Mul and the Div even at range 1), and ops the legacy exporter spells out in
// primitives (LayerNorm before opset 17, GELU, F.normalize's L2 norm, chunk as Slices).
#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "native/onnx_read.h"

namespace ajinative {
namespace onnxread {

struct ImgRange {
    float mean[3] = {};
    float range = 1.f;
    bool clamp = false;                             // the tail clips to [0, 1]
    const OnnxNode *sub = nullptr, *mul = nullptr;  // the head's ops (mul: none at range 1 without one)
};

// The first node with op `op` reading `name`.
inline const OnnxNode *consumer(const Reader &rd, const std::string &name, const char *op)
{
    for (const OnnxNode &n : rd.g.nodes)
        if (n.op == op && std::find(n.in.begin(), n.in.end(), name) != n.in.end()) return &n;
    return nullptr;
}

// A Clip of `x` to [0, 1] (bounds as inputs, or attributes before opset 11).
inline bool clip01(const Reader &rd, const OnnxNode *n)
{
    if (!n || n->op != "Clip") return false;
    float lo = n->attr_f("min", -1e30f), hi = n->attr_f("max", 1e30f);
    std::vector<float> v;
    if (n->in.size() > 1 && !n->in[1].empty()) {
        if (!rd.floats(n->in[1], &v) || v.size() != 1) return false;
        lo = v[0];
    }
    if (n->in.size() > 2 && !n->in[2].empty()) {
        if (!rd.floats(n->in[2], &v) || v.size() != 1) return false;
        hi = v[0];
    }
    return lo == 0.f && hi == 1.f;
}

// A Mul of a constant that belongs to a GELU the export spelled out: 0.5 * x * (1 + erf(x / sqrt 2)),
// the 0.5 either last or first.
inline bool gelu_const_mul(const Reader &rd, const OnnxNode &n)
{
    if (n.op != "Mul" || n.in.size() != 2) return false;
    auto erf_add = [&](const std::string &x) {  // x = 1 + erf(...)
        auto a = rd.prod.find(x);
        if (a == rd.prod.end() || a->second->op != "Add") return false;
        for (const std::string &i : a->second->in) {
            auto e = rd.prod.find(i);
            if (e != rd.prod.end() && e->second->op == "Erf") return true;
        }
        return false;
    };
    for (const std::string &x : n.in) {
        if (rd.tensor(x)) continue;
        auto p = rd.prod.find(x);  // 0.5 last: (x * (1 + erf)) * 0.5
        if (p != rd.prod.end() && p->second->op == "Mul")
            for (const std::string &i : p->second->in)
                if (erf_add(i)) return true;
        if (const OnnxNode *m = consumer(rd, n.out[0], "Mul"))  // 0.5 first: (x * 0.5) * (1 + erf)
            for (const std::string &i : m->in)
                if (erf_add(i)) return true;
    }
    return false;
}

// x is the second of two Slices of one tensor (torch's chunk(2) in a legacy export).
inline bool chunk_slice(const Reader &rd, const std::string &x)
{
    auto p = rd.prod.find(x);
    if (p == rd.prod.end() || p->second->op != "Slice" || p->second->in.empty()) return false;
    const std::string &src = p->second->in[0];
    for (const OnnxNode &n : rd.g.nodes)
        if (&n != p->second && n.op == "Slice" && !n.in.empty() && n.in[0] == src) return &n < p->second;
    return false;
}

// The square root of a ReduceSum of squares: an L2 norm spelled out (F.normalize in a legacy export:
// Abs, Pow 2, ReduceSum, Pow 0.5).
inline bool l2_root(const Reader &rd, const OnnxNode &n)
{
    std::vector<float> e;
    if (n.op != "Pow" || n.in.size() != 2 || !rd.floats(n.in[1], &e) || e.size() != 1 || e[0] != 0.5f) return false;
    auto rs = rd.prod.find(n.in[0]);
    if (rs == rd.prod.end() || rs->second->op != "ReduceSum" || rs->second->in.empty()) return false;
    auto sq = rd.prod.find(rs->second->in[0]);
    return sq != rd.prod.end() && sq->second->op == "Pow" && sq->second->in.size() == 2 && rd.floats(sq->second->in[1], &e) && e.size() == 1 &&
           e[0] == 2.f;
}

// An Add of a constant shaped like attention scores ([..., T, T], T > 1): a window attention's bias or
// shifted-window mask that a fixed-size export baked in (the mask then holds only for the export's
// size; the native kernels build it for the input's).
inline bool attn_const(const Reader &rd, const OnnxNode &n)
{
    if (n.op != "Add" || n.in.size() != 2) return false;
    for (const std::string &i : n.in)
        if (const OnnxTensor *t = rd.tensor(i)) {
            const size_t r = t->dims.size();
            return r >= 3 && t->dims[r - 1] == t->dims[r - 2] && t->dims[r - 1] > 1;
        }
    return false;
}

// The head from its Sub (input minus a 3-vector constant), with the scalar Mul of its output if any.
inline bool img_range_head(const Reader &rd, const OnnxNode *sub, ImgRange *r)
{
    std::vector<float> v;
    if (!sub || sub->op != "Sub" || sub->in.size() != 2 || rd.tensor(sub->in[0]) || !tensor_floats(rd, sub, 1, &v) || v.size() != 3)
        return false;
    std::copy(v.begin(), v.end(), r->mean);
    r->sub = sub;
    const OnnxNode *m = consumer(rd, sub->out[0], "Mul");
    if (m && init_input(rd, m, &v) && v.size() == 1) {
        r->range = v[0];
        r->mul = m;
    }
    return r->range != 0.f;
}

// conv_first's weights scaled by the range (its input is then the input minus the mean).
inline void img_range_first(const ImgRange &r, Conv *c)
{
    if (r.range != 1.f)
        for (float &w : c->w) w *= r.range;
}

// The tail after conv_last (c, its output y): a Div by the range (or a Mul by its inverse), the Add of
// the mean, a Clip(0, 1). Folds the scale into c's weights and bias, the mean into its bias. Each is
// optional (an optimizer may have fused the mean into conv_last's bias already); false for a Div or
// Mul that isn't by a scalar.
inline bool img_range_tail(const Reader &rd, const std::string &y, Conv *c, ImgRange *r)
{
    std::string x = y;
    float k = 1.f;
    std::vector<float> v;
    if (const OnnxNode *d = consumer(rd, x, "Div")) {
        if (d->in.size() != 2 || d->in[0] != x || !tensor_floats(rd, d, 1, &v) || v.size() != 1 || v[0] == 0.f) return false;
        k = 1.f / v[0];
        x = d->out[0];
    } else if (const OnnxNode *m = consumer(rd, x, "Mul")) {
        if (!init_input(rd, m, &v) || v.size() != 1) return false;
        k = v[0];
        x = m->out[0];
    }
    if (k != 1.f) {
        for (float &w : c->w) w *= k;
        for (float &b : c->b) b *= k;
    }
    const OnnxNode *a = consumer(rd, x, "Add");
    if (a && init_input(rd, a, &v) && v.size() == 3 && c->b.size() == 3) {
        for (int o = 0; o < 3; o++) c->b[o] += v[o];
        x = a->out[0];
    }
    r->clamp = clip01(rd, consumer(rd, x, "Clip"));
    return true;
}

// LayerNorm weights over the last axis.
struct LnW {
    std::vector<float> g, b;
    float eps = 1e-5f;
};

// A LayerNorm spelled out (opset < 17): d = x - mean(x); d / sqrt(mean(d^2) + eps) * g + b, matched at
// its Div. The nodes it spans go into *span; *out is the last one (the Add of b).
inline bool spelled_ln(const Reader &rd, const OnnxNode &div, LnW *l, std::set<const OnnxNode *> *span, const OnnxNode **out)
{
    auto prod = [&](const std::string &x, const char *op) -> const OnnxNode * {
        auto p = rd.prod.find(x);
        return p != rd.prod.end() && p->second->op == op ? p->second : nullptr;
    };
    if (div.op != "Div" || div.in.size() != 2) return false;
    const OnnxNode *sub = prod(div.in[0], "Sub"), *sq = prod(div.in[1], "Sqrt");
    if (!sub || !sq || sub->in.size() != 2) return false;
    const OnnxNode *m0 = prod(sub->in[1], "ReduceMean"), *ae = prod(sq->in[0], "Add");
    if (!m0 || m0->in.empty() || m0->in[0] != sub->in[0] || !ae) return false;
    const OnnxNode *m1 = nullptr;
    std::vector<float> e;
    for (const std::string &i : ae->in) {
        if (const OnnxNode *p = prod(i, "ReduceMean")) m1 = p;
        else if (!rd.floats(i, &e) || e.size() != 1) return false;
    }
    if (!m1 || e.size() != 1 || m1->in.empty()) return false;
    const OnnxNode *pw = prod(m1->in[0], "Pow"), *mul2 = prod(m1->in[0], "Mul");
    std::vector<float> two;
    if (pw) {
        if (pw->in.size() != 2 || pw->in[0] != sub->out[0] || !rd.floats(pw->in[1], &two) || two.size() != 1 || two[0] != 2.f) return false;
    } else if (!mul2 || mul2->in.size() != 2 || mul2->in[0] != sub->out[0] || mul2->in[1] != sub->out[0]) {
        return false;
    }
    const OnnxNode *mg = consumer(rd, div.out[0], "Mul");
    if (!mg || !init_input(rd, mg, &l->g)) return false;
    const OnnxNode *ab = consumer(rd, mg->out[0], "Add");
    if (!ab || !init_input(rd, ab, &l->b) || l->b.size() != l->g.size()) return false;
    l->eps = e[0];
    *span = {sub, sq, m0, ae, m1, pw ? pw : mul2, &div, mg, ab};
    *out = ab;
    return true;
}


// A graph's LayerNorms in either form: the op, or spelled out (keyed by its Div; span: every node of
// the spelled ones, for a parser to skip).
struct Lns {
    std::map<const OnnxNode *, LnW> spelled;
    std::set<const OnnxNode *> span;
    explicit Lns(const Reader &rd)
    {
        for (const OnnxNode &n : rd.g.nodes) {
            LnW l;
            std::set<const OnnxNode *> sp;
            const OnnxNode *out = nullptr;
            if (n.op == "Div" && spelled_ln(rd, n, &l, &sp, &out)) {
                spelled[&n] = l;
                span.insert(sp.begin(), sp.end());
            }
        }
    }
    bool is_ln(const OnnxNode &n) const { return n.op == "LayerNormalization" || spelled.count(&n); }
    // the weights of a LayerNorm at n (the op or a spelled one's Div)
    bool weights(const Reader &rd, const OnnxNode *n, LnW *l) const
    {
        if (!n) return false;
        auto s = spelled.find(n);
        if (s != spelled.end()) {
            *l = s->second;
            return true;
        }
        if (n->op != "LayerNormalization" || n->in.size() < 3 || !tensor_floats(rd, n, 1, &l->g) || !tensor_floats(rd, n, 2, &l->b) ||
            l->g.size() != l->b.size())
            return false;
        l->eps = n->attr_f("epsilon", 1e-5f);
        return true;
    }
    // its input
    std::string input(const Reader &rd, const OnnxNode *n) const
    {
        if (!n || n->in.empty()) return {};
        if (!spelled.count(n)) return n->in[0];
        auto p = rd.prod.find(n->in[0]);  // the Sub of the mean
        return p != rd.prod.end() && !p->second->in.empty() ? p->second->in[0] : std::string();
    }
};

// A BatchNormalization (inference) that is the only reader of a conv's output, folded into the conv's
// weights w ([co][...], co = b's size) and bias b (newer exporters leave it unfolded). False when
// there is none; the parsers then take the conv as it is.
inline bool fold_bn_after(const Reader &rd, const OnnxNode *conv, std::vector<float> *w, std::vector<float> *b)
{
    if (!conv || conv->out.empty()) return false;
    const OnnxNode *bn = nullptr;
    for (const OnnxNode &n : rd.g.nodes)
        if (std::find(n.in.begin(), n.in.end(), conv->out[0]) != n.in.end()) {
            if (bn || n.op != "BatchNormalization" || n.in[0] != conv->out[0]) return false;
            bn = &n;
        }
    std::vector<float> g, be, m, v;
    const size_t co = b->size();
    if (!bn || bn->in.size() < 5 || !rd.floats(bn->in[1], &g) || !rd.floats(bn->in[2], &be) || !rd.floats(bn->in[3], &m) ||
        !rd.floats(bn->in[4], &v) || g.size() != co || be.size() != co || m.size() != co || v.size() != co || !co || w->size() % co)
        return false;
    const float eps = bn->attr_f("epsilon", 1e-5f);
    const size_t per = w->size() / co;
    for (size_t o = 0; o < co; o++) {
        const float sc = g[o] / std::sqrt(v[o] + eps);
        for (size_t k = 0; k < per; k++) (*w)[o * per + k] *= sc;
        (*b)[o] = ((*b)[o] - m[o]) * sc + be[o];
    }
    return true;
}

}  // namespace onnxread
}  // namespace ajinative
