// native/onnx_read.h — reading convolutions out of an ONNX graph by node name, for the
// hand-mapped native networks (rife.cpp, esrgan.cpp). Internal, host only.
#ifndef AJI_NATIVE_ONNX_READ_H
#define AJI_NATIVE_ONNX_READ_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "native/onnx_graph.h"
#include "native/pack.h"

namespace ajinative {
namespace onnxread {

inline bool refuse(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

inline int pad16(int c) { return (c + 15) / 16 * 16; }
inline int pad8(int c) { return (c + 7) / 8 * 8; }

struct Conv {
    int cout = 0, cin = 0, k = 0, stride = 1;
    bool transposed = false;
    std::vector<float> w;  // ONNX layout: Conv [co][ci][k][k], ConvTranspose [ci][co][k][k]
    std::vector<float> b;
    bool lrelu = false;
    float slope = 0.2f;  // of the LeakyReLU (only other than 0.2 when Reader::any_slope)
};

struct Reader {
    const OnnxGraph &g;
    std::map<std::string, const OnnxNode *> by_name, prod;
    bool any_slope = false;  // accept LeakyReLU slopes other than 0.2 (Conv::slope)
    explicit Reader(const OnnxGraph &gr) : g(gr)
    {
        for (const OnnxNode &n : g.nodes) {
            by_name[n.name] = &n;
            for (const std::string &o : n.out) prod[o] = &n;
        }
    }
    const OnnxNode *node(const std::string &name) const
    {
        auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : it->second;
    }
    const OnnxTensor *tensor(const std::string &name) const { return tensor_at(name, 0); }

private:
    // Constants an exporter left unfolded: Identity / Cast / Unsqueeze / Squeeze / Reshape /
    // Flatten / Transpose of a constant (weights passed through Identity, transposed MatMul
    // weights), folded on the host once (fp32).
    mutable std::map<std::string, OnnxTensor> folded_;
    const OnnxTensor *tensor_at(const std::string &name, int depth) const
    {
        auto it = g.inits.find(name);
        if (it != g.inits.end()) return &it->second;
        auto p = prod.find(name);
        if (p == prod.end()) return nullptr;
        const OnnxNode &n = *p->second;
        if (n.op == "Constant") {
            const OnnxAttr *v = n.attr("value");
            return v && v->type == 4 ? &v->t : nullptr;
        }
        static const char *const kFold[] = {"Identity", "Cast", "Unsqueeze", "Squeeze", "Reshape", "Flatten", "Transpose"};
        if (depth > 16 || n.in.empty() || n.out.empty() || std::none_of(std::begin(kFold), std::end(kFold), [&](const char *o) { return n.op == o; }))
            return nullptr;
        auto f = folded_.find(name);
        if (f != folded_.end()) return f->second.raw.empty() ? nullptr : &f->second;
        OnnxTensor &t = folded_[name];
        const OnnxTensor *src = tensor_at(n.in[0], depth + 1);
        std::vector<float> v;
        if (!src || src->count() == OnnxTensor::kBadCount || !src->floats(&v)) return nullptr;
        std::vector<int64_t> dims = src->dims;
        if (n.op == "Transpose") {
            const OnnxAttr *pa = n.attr("perm");
            const size_t r = dims.size();
            std::vector<int64_t> perm(r);
            for (size_t k = 0; k < r; k++) perm[k] = (int64_t)(r - 1 - k);
            if (pa && !pa->ints.empty()) perm = pa->ints;
            if (perm.size() != r || r > 6) return nullptr;
            std::vector<int64_t> od(r), sstr(r, 1), ostr(r, 1);
            for (size_t k = 0; k < r; k++) {
                if (perm[k] < 0 || perm[k] >= (int64_t)r) return nullptr;
                od[k] = dims[(size_t)perm[k]];
            }
            for (int k = (int)r - 2; k >= 0; k--) sstr[k] = sstr[k + 1] * dims[k + 1];
            for (int k = (int)r - 2; k >= 0; k--) ostr[k] = ostr[k + 1] * od[k + 1];
            std::vector<float> o(v.size());
            for (size_t i = 0; i < v.size(); i++) {
                size_t rem = i, si = 0;
                for (size_t k = 0; k < r; k++) {
                    const size_t c = rem / (size_t)ostr[k];
                    rem %= (size_t)ostr[k];
                    si += c * (size_t)sstr[(size_t)perm[k]];
                }
                o[i] = v[si];
            }
            v.swap(o);
            dims = od;
        } else if (n.op == "Unsqueeze" || n.op == "Squeeze" || n.op == "Reshape" || n.op == "Flatten") {
            // the data stays; the dims follow the op (Reshape: its shape input, 0 = keep, -1 = rest)
            std::vector<int64_t> axes;
            const OnnxAttr *aa = n.attr("axes");
            if (aa) axes = aa->ints;
            else if (n.in.size() > 1 && !n.in[1].empty()) {
                const OnnxTensor *at = tensor_at(n.in[1], depth + 1);
                if (!at || !at->ints(&axes)) return nullptr;
            }
            if (n.op == "Reshape") {
                std::vector<int64_t> sh = axes, out;
                if (aa) return nullptr;  // Reshape takes its shape as an input
                int64_t known = 1, neg = -1;
                for (size_t k = 0; k < sh.size(); k++) {
                    int64_t d = sh[k] == 0 && k < dims.size() ? dims[k] : sh[k];
                    if (d == -1) {
                        if (neg >= 0) return nullptr;
                        neg = (int64_t)k;
                    } else if (d < 0) {
                        return nullptr;
                    } else {
                        known *= d;
                    }
                    out.push_back(d);
                }
                if (neg >= 0) {
                    if (known == 0 || (int64_t)v.size() % known) return nullptr;
                    out[(size_t)neg] = (int64_t)v.size() / known;
                }
                dims = out;
            } else if (n.op == "Flatten") {
                const int64_t ax = n.attr_i("axis", 1);
                int64_t a = 1;
                for (int64_t k = 0; k < ax && k < (int64_t)dims.size(); k++) a *= dims[(size_t)k];
                dims = {a, a ? (int64_t)v.size() / a : 0};
            } else if (n.op == "Squeeze") {
                std::vector<int64_t> out;
                for (size_t k = 0; k < dims.size(); k++) {
                    const bool drop = axes.empty() ? dims[k] == 1
                                                   : std::any_of(axes.begin(), axes.end(), [&](int64_t x) {
                                                         return (x < 0 ? x + (int64_t)dims.size() : x) == (int64_t)k;
                                                     });
                    if (!drop) out.push_back(dims[k]);
                }
                dims = out;
            } else {
                const int64_t r = (int64_t)(dims.size() + axes.size());
                std::vector<int64_t> out;
                size_t src_k = 0;
                for (int64_t k = 0; k < r; k++) {
                    const bool ins = std::any_of(axes.begin(), axes.end(), [&](int64_t x) { return (x < 0 ? x + r : x) == k; });
                    out.push_back(ins ? 1 : (src_k < dims.size() ? dims[src_k++] : 1));
                }
                dims = out;
            }
        }
        size_t cnt = 1;
        for (int64_t d : dims) cnt *= (size_t)(d < 0 ? 0 : d);
        if (cnt != v.size()) return nullptr;
        t.name = name;
        t.dtype = 1;
        t.dims = dims;
        t.raw.resize(v.size() * 4);
        if (!v.empty()) std::memcpy(t.raw.data(), v.data(), v.size() * 4);
        return &t;
    }

public:
    bool floats(const std::string &name, std::vector<float> *out) const
    {
        const OnnxTensor *t = tensor(name);
        return t && t->floats(out);
    }
    // A Conv / ConvTranspose node with constant weights; lrelu if a LeakyRelu(0.2) consumes it.
    bool conv(const std::string &name, Conv *c, std::string *why) const
    {
        *c = Conv();
        const OnnxNode *n = node(name);
        if (!n || (n->op != "Conv" && n->op != "ConvTranspose")) return refuse(why, "missing " + name);
        return conv(n, c, why);
    }
    // The same for a node found by structure (names may be empty or repeated).
    bool conv(const OnnxNode *n, Conv *c, std::string *why) const
    {
        *c = Conv();
        if (!n || (n->op != "Conv" && n->op != "ConvTranspose")) return refuse(why, "missing conv");
        const std::string name = n->name.empty() ? n->out[0] : n->name;
        c->transposed = n->op == "ConvTranspose";
        const OnnxTensor *w = n->in.size() > 1 ? tensor(n->in[1]) : nullptr;
        if (!w || w->dims.size() != 4 || !w->floats(&c->w)) return refuse(why, name + ": weights");
        // dims are bounded by the element count floats() checked against the data (<= 2^28 each)
        c->k = (int)w->dims[2];
        if (w->dims[3] != c->k) return refuse(why, name + ": non-square kernel");
        c->cout = (int)(c->transposed ? w->dims[1] : w->dims[0]);
        c->cin = (int)(c->transposed ? w->dims[0] : w->dims[1]);
        if (c->cout < 1 || c->cin < 1 || c->k < 1) return refuse(why, name + ": empty weights");
        const int64_t ng64 = n->attr_i("group", 1);
        if (ng64 < 1 || ng64 > c->cout) return refuse(why, name + ": groups");
        const int ng = (int)ng64;
        if (ng > 1 && (size_t)c->cout * c->cin * ng * c->k * c->k > OnnxTensor::kMaxElems) return refuse(why, name + ": grouped conv too large");
        if (ng > 1 && !c->transposed && c->cout % ng == 0) {
            // grouped conv: expand to block-diagonal dense weights [co][cin*groups][k][k]
            const int cig = c->cin, cog = c->cout / ng, kk = c->k * c->k;
            std::vector<float> dense((size_t)c->cout * cig * ng * kk, 0.f);
            for (int o = 0; o < c->cout; o++)
                for (int i = 0; i < cig; i++)
                    for (int t = 0; t < kk; t++)
                        dense[((size_t)o * cig * ng + (o / cog) * cig + i) * kk + t] = c->w[((size_t)o * cig + i) * kk + t];
            c->w.swap(dense);
            c->cin = cig * ng;
        }
        c->b.assign(c->cout, 0.f);
        if (n->in.size() > 2 && !n->in[2].empty() && (!floats(n->in[2], &c->b) || (int)c->b.size() != c->cout))
            return refuse(why, name + ": bias");
        const OnnxAttr *st = n->attr("strides"), *pd = n->attr("pads"), *dl = n->attr("dilations");
        if (st && !st->ints.empty() && (st->ints[0] < 1 || st->ints[0] > 2)) return refuse(why, name + ": strides");
        c->stride = st && !st->ints.empty() ? (int)st->ints[0] : 1;
        if (st && st->ints.size() == 2 && st->ints[1] != st->ints[0]) return refuse(why, name + ": strides");
        const int groups = ng;
        if (groups < 1 || (groups > 1 && (c->transposed || c->cout % groups))) return refuse(why, name + ": groups");
        if (dl)
            for (int64_t d : dl->ints)
                if (d != 1) return refuse(why, name + ": dilated");
        const int want_pad = c->transposed ? 1 : c->k / 2;
        if (pd)
            for (int64_t p : pd->ints)
                if (p != want_pad) return refuse(why, name + ": padding");
        if (c->transposed ? (c->k != 4 || c->stride != 2) : (c->k != 3 || (c->stride != 1 && c->stride != 2)))
            return refuse(why, name + ": kernel shape");
        // activation
        for (const OnnxNode &m : g.nodes)
            if (m.op == "LeakyRelu" && !m.in.empty() && m.in[0] == n->out[0]) {
                const OnnxAttr *a = m.attr("alpha");
                const float al = a ? a->f : 0.01f;
                if (!any_slope && std::fabs(al - 0.2f) > 1e-6f) return refuse(why, name + ": LeakyReLU slope");
                c->lrelu = true;
                c->slope = al;
            }
        return true;
    }
};

inline uint16_t hb(float f) { return float_to_half_bits(f); }

// Conv [co][ci][3][3] -> [9][ci][co] fp16 bits, scaled per output channel by sc (or 1).
inline std::vector<uint16_t> conv_taps(const Conv &c, const std::vector<float> *sc)
{
    std::vector<uint16_t> w((size_t)9 * c.cin * c.cout);
    for (int o = 0; o < c.cout; o++)
        for (int i = 0; i < c.cin; i++)
            for (int t = 0; t < 9; t++)
                w[((size_t)t * c.cin + i) * c.cout + o] = hb(c.w[((size_t)o * c.cin + i) * 9 + t] * (sc ? (*sc)[o] : 1.f));
    return w;
}

// ConvTranspose 4x4/2/1 output phase p (0/1) from a 3x3 window tap d (-1/0/1): the kernel index,
// or -1 when that tap does not contribute.
inline int tk(int p, int d) { return p == 0 ? (d == 0 ? 1 : d == -1 ? 3 : -1) : (d == 0 ? 2 : d == 1 ? 0 : -1); }

// A transposed conv as R phase convs: launch for row phase sy writes channels sx*CHP + ch of
// pixel (R*y + sy, R*x + sx). R = 2: the plain transposed conv (ch = output channel). R = 4: the
// transposed conv followed by DepthToSpace(2, CRD): output channel ch*4 + dy2*2 + dx2 of the
// transposed conv lands at sub-position (dy2, dx2) of its 2x2 block.
inline void phase_weights(const Conv &c, int R, int sy, int CHP, std::vector<uint16_t> *w, std::vector<float> *b)
{
    const int CH = R == 2 ? c.cout : c.cout / 4, cout = R * CHP;
    w->assign((size_t)9 * c.cin * cout, 0);
    b->assign(cout, 0.f);
    const int py = R == 2 ? sy : sy / 2, dy2 = R == 2 ? 0 : sy % 2;
    for (int sx = 0; sx < R; sx++) {
        const int px = R == 2 ? sx : sx / 2, dx2 = R == 2 ? 0 : sx % 2;
        for (int ch = 0; ch < CH; ch++) {
            const int o = R == 2 ? ch : ch * 4 + dy2 * 2 + dx2;  // transposed-conv output channel
            (*b)[sx * CHP + ch] = c.b[o];
            for (int ty = -1; ty <= 1; ty++)
                for (int tx = -1; tx <= 1; tx++) {
                    const int ky = tk(py, ty), kx = tk(px, tx);
                    if (ky < 0 || kx < 0) continue;
                    const int t = (ty + 1) * 3 + (tx + 1);
                    for (int i = 0; i < c.cin; i++)
                        (*w)[((size_t)t * c.cin + i) * cout + sx * CHP + ch] = hb(c.w[(((size_t)i * c.cout + o) * 4 + ky) * 4 + kx]);
                }
        }
    }
}

// Input idx of node n as floats (an initializer or Constant).
inline bool tensor_floats(const Reader &rd, const OnnxNode *n, int idx, std::vector<float> *v, std::vector<int64_t> *dims = nullptr)
{
    if (!n || (int)n->in.size() <= idx) return false;
    const OnnxTensor *t = rd.tensor(n->in[idx]);
    if (!t || !t->floats(v)) return false;
    if (dims) *dims = t->dims;
    return true;
}
// The constant input of a binary op (e.g. the bias of an Add).
inline bool init_input(const Reader &rd, const OnnxNode *n, std::vector<float> *v, std::vector<int64_t> *dims = nullptr)
{
    if (!n) return false;
    for (size_t i = 0; i < n->in.size(); i++)
        if (rd.tensor(n->in[i])) return tensor_floats(rd, n, (int)i, v, dims);
    return false;
}

// A 3x3 conv followed by DepthToSpace(2, CRD) as 2 row-phase convs: launch sy writes channels
// sx*CHP + ch of pixel (2y + sy, 2x + sx) from conv output channel ch*4 + sy*2 + sx.
inline void ps_weights(const Conv &c, int sy, int CHP, std::vector<uint16_t> *w, std::vector<float> *b)
{
    const int CH = c.cout / 4, cout = 2 * CHP;
    w->assign((size_t)9 * c.cin * cout, 0);
    b->assign(cout, 0.f);
    for (int sx = 0; sx < 2; sx++)
        for (int ch = 0; ch < CH; ch++) {
            const int o = ch * 4 + sy * 2 + sx;
            (*b)[sx * CHP + ch] = c.b[o];
            for (int t = 0; t < 9; t++)
                for (int i = 0; i < c.cin; i++) (*w)[((size_t)t * c.cin + i) * cout + sx * CHP + ch] = hb(c.w[((size_t)o * c.cin + i) * 9 + t]);
        }
}

// A 1x1 conv's weights [co][ci][1][1] as a GEMM matrix [K][co]: input channel i at row map[i]
// (rows not mapped are zero), K rows in all.
inline bool conv1x1_matrix(const Reader &rd, const std::string &name, int K, const std::vector<int> *map, std::vector<float> *w,
                           std::vector<float> *bias, int *cout, std::string *why)
{
    const OnnxNode *n = rd.node(name);
    std::vector<float> v;
    std::vector<int64_t> d;
    if (!n || n->op != "Conv" || !tensor_floats(rd, n, 1, &v, &d) || d.size() != 4 || d[2] != 1 || d[3] != 1 ||
        n->attr_i("group", 1) != 1)
        return refuse(why, "missing 1x1 conv " + name);
    const int co = (int)d[0], ci = (int)d[1];
    if (co < 1 || ci < 1 || (map && (int)map->size() != ci) || (!map && ci > K)) return refuse(why, name + ": input channels");
    if (map)
        for (int r : *map)
            if (r < 0 || r >= K) return refuse(why, name + ": input channel map");
    w->assign((size_t)K * co, 0.f);
    for (int o = 0; o < co; o++)
        for (int i = 0; i < ci; i++) (*w)[(size_t)(map ? (*map)[i] : i) * co + o] = v[(size_t)o * ci + i];
    bias->clear();
    if (n->in.size() > 2 && !n->in[2].empty() && (!rd.floats(n->in[2], bias) || (int)bias->size() != co)) return refuse(why, name + ": bias");
    *cout = co;
    return true;
}

}  // namespace onnxread
}  // namespace ajinative

#endif
