// native/onnx_read.h — reading convolutions out of an ONNX graph by node name, for the
// hand-mapped native networks (rife.cpp, esrgan.cpp). Internal, host only.
#ifndef AJI_NATIVE_ONNX_READ_H
#define AJI_NATIVE_ONNX_READ_H

#include <cmath>
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
};

struct Reader {
    const OnnxGraph &g;
    std::map<std::string, const OnnxNode *> by_name, prod;
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
    const OnnxTensor *tensor(const std::string &name) const
    {
        auto it = g.inits.find(name);
        if (it != g.inits.end()) return &it->second;
        auto p = prod.find(name);
        if (p != prod.end() && p->second->op == "Constant") {
            const OnnxAttr *v = p->second->attr("value");
            if (v && v->type == 4) return &v->t;
        }
        return nullptr;
    }
    bool floats(const std::string &name, std::vector<float> *out) const
    {
        const OnnxTensor *t = tensor(name);
        return t && t->floats(out);
    }
    // A Conv / ConvTranspose node with constant weights; lrelu if a LeakyRelu(0.2) consumes it.
    bool conv(const std::string &name, Conv *c, std::string *why) const
    {
        const OnnxNode *n = node(name);
        if (!n || (n->op != "Conv" && n->op != "ConvTranspose")) return refuse(why, "missing " + name);
        c->transposed = n->op == "ConvTranspose";
        const OnnxTensor *w = n->in.size() > 1 ? tensor(n->in[1]) : nullptr;
        if (!w || w->dims.size() != 4 || !w->floats(&c->w)) return refuse(why, name + ": weights");
        c->k = (int)w->dims[2];
        if (w->dims[3] != c->k) return refuse(why, name + ": non-square kernel");
        c->cout = (int)(c->transposed ? w->dims[1] : w->dims[0]);
        c->cin = (int)(c->transposed ? w->dims[0] : w->dims[1]);
        const int ng = (int)n->attr_i("group", 1);
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
        c->stride = st && !st->ints.empty() ? (int)st->ints[0] : 1;
        if (st && st->ints.size() == 2 && st->ints[1] != st->ints[0]) return refuse(why, name + ": strides");
        const int groups = (int)n->attr_i("group", 1);
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
                if (!a || std::fabs(a->f - 0.2f) > 1e-6f) return refuse(why, name + ": LeakyReLU slope");
                c->lrelu = true;
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

}  // namespace onnxread
}  // namespace ajinative

#endif
