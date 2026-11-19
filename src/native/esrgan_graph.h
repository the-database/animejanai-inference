// native/esrgan_graph.h — ESRGAN (RRDBNet) found in an ONNX graph by its structure, whatever the
// exporter named the nodes (chaiNNer / spandrel, traiNNer, neosr, the old-arch /model/model.N
// layout), for the CUDA and Vulkan builders. Internal, host only.
//
// The pattern, by data flow from the model input:
//   input [-> pixel-unshuffle by 2 or 4 (Reshape/Transpose/..., or SpaceToDepth); Pad aligns]
//   -> first conv (nf) = F
//   -> nb RRDBs, each 3 RDBs of 5 convs: conv k (k = 1..4) reads Concat(x, l1..l(k-1)) and is
//      LeakyReLU'd (gc channels), conv 5 reads Concat(x, l1..l4) (nf channels), RDB = s_rdb * conv5
//      + x (the Mul may be folded into the weights: s_rdb 1); RRDB = s_rrdb * RDB3 + x
//   -> body conv (nf), + F
//   -> nup times: nearest 2x Resize -> conv -> LeakyReLU
//   -> hr conv -> LeakyReLU -> last conv (3) [-> Slice crop back] [-> Clip(0, 1)] -> output
#ifndef AJI_NATIVE_ESRGAN_GRAPH_H
#define AJI_NATIVE_ESRGAN_GRAPH_H

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "native/onnx_read.h"
#include "native/static_eval.h"

namespace ajinative {
namespace onnxread {

struct EsrganGraph {
    int nf = 0, gc = 0, nb = 0, nup = 0;
    int unshuffle = 1;            // the input's pixel-unshuffle factor (1, 2, 4)
    std::vector<int> cin_from;    // the first conv's input channel order when the unshuffle is DCR (SpaceToDepth)
    bool clamp = false;           // a final Clip(0, 1)
    const OnnxNode *first = nullptr, *body = nullptr, *hr = nullptr, *last = nullptr;
    std::vector<const OnnxNode *> ups;    // nup
    std::vector<const OnnxNode *> dense;  // nb * 15, RRDB i, RDB r, conv k at (i * 3 + r) * 5 + k
    std::vector<float> rdb_scale;         // nb * 3
    std::vector<float> rrdb_scale;        // nb
    int scale() const { return (1 << nup) / unshuffle == 0 ? 0 : (1 << nup) / unshuffle; }
};

class EsrganFinder {
public:
    EsrganFinder(const Reader &r, EsrganGraph *out) : rd(r), e(*out)
    {
        for (size_t i = 0; i < rd.g.nodes.size(); i++) {
            const OnnxNode &n = rd.g.nodes[i];
            order[&n] = (int)i;
            for (const std::string &in : n.in)
                if (!in.empty()) cons[in].push_back(&n);
        }
    }

    bool find(std::string *why)
    {
        const OnnxGraph &g = rd.g;
        if (g.inputs.size() != 1 || g.outputs.size() != 1) return refuse(why, "not an ESRGAN graph (inputs / outputs)");
        // the first conv and the input's path to it
        std::vector<const OnnxNode *> convs;
        for (const OnnxNode &n : g.nodes)
            if (n.op == "Conv") convs.push_back(&n);
        if (convs.size() < 1 + 15 + 3) return refuse(why, "not an ESRGAN graph (too few convs)");
        e.first = convs[0];
        if (!input_path(e.first->in[0], why)) return false;
        const int cin0 = cin_of(e.first);
        if (cin0 != 3 * e.unshuffle * e.unshuffle) return refuse(why, "not an ESRGAN graph (first conv input channels)");
        e.nf = cout_of(e.first);
        const std::string F = e.first->out[0];
        // the global residual: an Add of F and the body conv
        for (const OnnxNode *a : users(F)) {
            if (a->op != "Add" || a->in.size() != 2) continue;
            const OnnxNode *p = producer(a->in[0] == F ? a->in[1] : a->in[0]);
            if (p && p->op == "Conv" && p != e.first) {
                e.body = p;
                trunk = a->out[0];
            }
        }
        if (!e.body) return refuse(why, "not an ESRGAN graph (no global residual)");
        const int ib = (int)(std::find(convs.begin(), convs.end(), e.body) - convs.begin());
        if ((ib - 1) % 15 || ib < 16) return refuse(why, "not an ESRGAN graph (RRDB conv count)");
        e.nb = (ib - 1) / 15;
        e.dense.assign(convs.begin() + 1, convs.begin() + ib);
        e.gc = cout_of(e.dense[0]);
        if (e.gc < 1) return refuse(why, "not an ESRGAN graph (growth channels)");
        // the RRDBs: shapes, the dense concats, the residual scales
        std::string x = F;  // the RRDB input
        for (int i = 0; i < e.nb; i++) {
            std::string xr = x;  // the RDB input
            for (int r = 0; r < 3; r++) {
                std::vector<std::string> feats{xr};
                for (int k = 0; k < 5; k++) {
                    const OnnxNode *c = e.dense[(i * 3 + r) * 5 + k];
                    const int ci = cin_of(c), co = cout_of(c);
                    if (ci != e.nf + k * e.gc || co != (k == 4 ? e.nf : e.gc))
                        return refuse(why, "ESRGAN RRDB " + std::to_string(i) + ": dense conv shape");
                    if (!reads(c, feats)) return refuse(why, "ESRGAN RRDB " + std::to_string(i) + ": dense connections");
                    if (k < 4) {
                        const OnnxNode *l = sole_user(c->out[0], "LeakyRelu");
                        if (!l) return refuse(why, "ESRGAN RRDB " + std::to_string(i) + ": dense conv activation");
                        feats.push_back(l->out[0]);
                    } else {
                        float s = 1.f;
                        std::string rdb;
                        if (!scaled_residual(c->out[0], xr, &s, &rdb)) return refuse(why, "ESRGAN RRDB " + std::to_string(i) + ": RDB residual");
                        e.rdb_scale.push_back(s);
                        xr = rdb;
                    }
                }
            }
            float s = 1.f;
            std::string rrdb;
            if (!scaled_residual(xr, x, &s, &rrdb)) return refuse(why, "ESRGAN RRDB " + std::to_string(i) + ": residual");
            e.rrdb_scale.push_back(s);
            x = rrdb;
        }
        if (!reads(e.body, {x}) || cin_of(e.body) != e.nf || cout_of(e.body) != e.nf) return refuse(why, "ESRGAN body conv");
        // the upsampler: (Resize nearest 2x -> conv -> LeakyReLU) x nup, hr conv -> LeakyReLU, last conv
        std::string t = trunk;
        for (;;) {
            const OnnxNode *rz = sole_user(t, "Resize");
            if (!rz) break;
            if (!nearest2x(rz)) return refuse(why, "ESRGAN upsample is not a nearest 2x resize");
            const OnnxNode *c = sole_user(rz->out[0], "Conv");
            const OnnxNode *l = c ? sole_user(c->out[0], "LeakyRelu") : nullptr;
            if (!l || cin_of(c) != e.nf || cout_of(c) != e.nf) return refuse(why, "ESRGAN upsample conv");
            e.ups.push_back(c);
            t = l->out[0];
        }
        e.nup = (int)e.ups.size();
        e.hr = sole_user(t, "Conv");
        const OnnxNode *lh = e.hr ? sole_user(e.hr->out[0], "LeakyRelu") : nullptr;
        e.last = lh ? sole_user(lh->out[0], "Conv") : nullptr;
        if (!e.last || cin_of(e.hr) != e.nf || cout_of(e.hr) != e.nf || cin_of(e.last) != e.nf || cout_of(e.last) != 3)
            return refuse(why, "ESRGAN output convs");
        // the output: [Slice crops back (alignment)] [Clip(0, 1)]
        std::string o = e.last->out[0];
        for (int guard = 0; guard < 8 && o != g.outputs[0]; guard++) {
            const std::vector<const OnnxNode *> &u = users(o);
            if (u.size() != 1) return refuse(why, "ESRGAN output is used twice");
            const OnnxNode *n = u[0];
            if (n->op == "Slice" && crop_from_zero(n)) {
                o = n->out[0];
            } else if (n->op == "Clip" && clip01(n)) {
                e.clamp = true;
                o = n->out[0];
            } else if (pass(n)) {
                o = n->out[0];
            } else {
                return refuse(why, "ESRGAN output: unsupported " + n->op);
            }
        }
        if (o != g.outputs[0]) return refuse(why, "ESRGAN output");
        if (e.scale() < 1) return refuse(why, "ESRGAN scale below 1");
        return true;
    }

private:
    const Reader &rd;
    EsrganGraph &e;
    std::map<const OnnxNode *, int> order;
    std::map<std::string, std::vector<const OnnxNode *>> cons;
    std::string trunk;
    const std::vector<const OnnxNode *> &users(const std::string &t)
    {
        static const std::vector<const OnnxNode *> none;
        auto it = cons.find(t);
        return it == cons.end() ? none : it->second;
    }
    const OnnxNode *producer(const std::string &t) const
    {
        auto it = rd.prod.find(t);
        return it == rd.prod.end() ? nullptr : it->second;
    }
    // an op the data passes through unchanged: Identity, a Cast between float types
    static bool pass(const OnnxNode *n) { return n->op == "Identity" || float_cast(*n); }
    // the only data user of t that is an `op` (pass() ops followed through)
    const OnnxNode *sole_user(std::string t, const char *op)
    {
        for (int guard = 0; guard < 4; guard++) {
            const std::vector<const OnnxNode *> &u = users(t);
            if (u.size() != 1) return nullptr;
            if (u[0]->op == op) return u[0];
            if (!pass(u[0])) return nullptr;
            t = u[0]->out[0];
        }
        return nullptr;
    }
    int dims_of(const OnnxNode *c, int i) const
    {
        const OnnxTensor *w = c->in.size() > 1 ? rd.tensor(c->in[1]) : nullptr;
        return w && w->dims.size() == 4 && w->dims[2] == 3 && w->dims[3] == 3 ? (int)w->dims[i] : -1;
    }
    int cin_of(const OnnxNode *c) const { return dims_of(c, 1); }
    int cout_of(const OnnxNode *c) const { return dims_of(c, 0); }
    std::string through(std::string t) const
    {
        for (int guard = 0; guard < 4; guard++) {
            const OnnxNode *p = producer(t);
            if (!p || !pass(p)) break;
            t = p->in[0];
        }
        return t;
    }
    // conv c reads `feats` (one tensor, or a Concat of them on channels, in order)
    bool reads(const OnnxNode *c, const std::vector<std::string> &feats) const
    {
        const std::string in = through(c->in[0]);
        if (feats.size() == 1) return in == feats[0];
        const OnnxNode *cat = producer(in);
        if (!cat || cat->op != "Concat" || cat->in.size() != feats.size()) return false;
        const int64_t ax = cat->attr_i("axis", 0);
        if (ax != 1 && ax != -3) return false;
        for (size_t k = 0; k < feats.size(); k++)
            if (through(cat->in[k]) != feats[k]) return false;
        return true;
    }
    // y = s * v + x (Add of x and Mul(v, s), or of x and v itself): *s, *out the Add's output
    bool scaled_residual(const std::string &v, const std::string &x, float *s, std::string *out)
    {
        std::string t = v;
        *s = 1.f;
        const std::vector<const OnnxNode *> &u = users(v);
        if (u.size() == 1 && u[0]->op == "Mul" && u[0]->in.size() == 2) {
            const std::string k = u[0]->in[0] == v ? u[0]->in[1] : u[0]->in[0];
            std::vector<float> f;
            if (!rd.floats(k, &f) || f.size() != 1) return false;
            *s = f[0];
            t = u[0]->out[0];
        }
        const std::vector<const OnnxNode *> &ua = users(t);
        if (ua.size() != 1 || ua[0]->op != "Add" || ua[0]->in.size() != 2) return false;
        const std::string other = ua[0]->in[0] == t ? ua[0]->in[1] : ua[0]->in[0];
        if (through(other) != x) return false;
        *out = ua[0]->out[0];
        return true;
    }
    // a nearest 2x of the nf-channel features at every input size (static_eval::resize_by)
    bool nearest2x(const OnnxNode *rz) const
    {
        if (!nearest_replicates(*rz, 2)) return false;
        const std::string src = through(rz->in[0]);
        return static_eval::resize_by(rd, *rz, 2, e.nf, [&](const std::string &t) { return through(t) == src; });
    }
    bool crop_from_zero(const OnnxNode *n) const
    {
        if (n->in.size() < 3) return false;
        std::vector<float> st;
        const OnnxTensor *t = rd.tensor(n->in[1]);
        std::vector<int64_t> s;
        return !t || (t->ints(&s) && std::all_of(s.begin(), s.end(), [](int64_t v) { return v == 0; }));
    }
    bool clip01(const OnnxNode *n) const
    {
        float lo = n->attr_f("min", -3.4e38f), hi = n->attr_f("max", 3.4e38f);
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
    // the ops between the model input and the first conv: none, or a pixel-unshuffle (with Pad /
    // Shape plumbing): its factor from the first conv's input channels; SpaceToDepth's DCR order
    // becomes the first conv's input channel order
    bool input_path(std::string t, std::string *why)
    {
        bool dcr = false, any = false;
        for (int guard = 0; guard < 12 && through(t) != rd.g.inputs[0]; guard++) {
            const OnnxNode *p = producer(through(t));
            if (!p) return refuse(why, "not an ESRGAN graph (input path)");
            if (p->op == "SpaceToDepth") dcr = true;
            else if (p->op == "Pad") {
                const OnnxTensor *pt = p->in.size() > 1 && !p->in[1].empty() ? rd.tensor(p->in[1]) : nullptr;
                std::vector<int64_t> v;
                if (pt && pt->ints(&v) && std::any_of(v.begin(), v.end(), [](int64_t a) { return a != 0; }))
                    return refuse(why, "ESRGAN input: Pad by a fixed amount");
            } else if (p->op == "Transpose") {
                const OnnxAttr *perm = p->attr("perm");
                // PyTorch's pixel_unshuffle: (n, c, h, r, w, r) -> (n, c, r, r, h, w), as 6-D or with a
                // leading 1 (7-D): channel c * r * r + ry * r + rx
                if (!perm || (perm->ints != std::vector<int64_t>{0, 1, 3, 5, 2, 4} &&
                              perm->ints != std::vector<int64_t>{0, 1, 2, 4, 6, 3, 5}))
                    return refuse(why, "ESRGAN input: a transpose that is not a pixel-unshuffle");
            } else if (p->op != "Reshape" && p->op != "Unsqueeze" && p->op != "Squeeze" && !pass(p)) {
                return refuse(why, "not an ESRGAN graph (" + p->op + " before the first conv)");
            }
            any = true;
            t = p->in[0];
        }
        if (through(t) != rd.g.inputs[0]) return refuse(why, "not an ESRGAN graph (input path)");
        const int ci = cin_of(e.first);
        e.unshuffle = !any ? 1 : ci == 12 ? 2 : ci == 48 ? 4 : 0;
        if (e.unshuffle == 0) return refuse(why, "ESRGAN input unshuffle other than 2 or 4");
        if (dcr) {  // SpaceToDepth (DCR): ONNX channel (ry * r + rx) * 3 + c for CRD channel c * r * r + ry * r + rx
            const int r = e.unshuffle, rr = r * r;
            for (int u = 0; u < 3 * rr; u++) e.cin_from.push_back((u % rr) * 3 + u / rr);
        }
        return true;
    }
};

inline bool find_esrgan(const Reader &rd, EsrganGraph *e, std::string *why)
{
    *e = EsrganGraph();
    EsrganFinder f(rd, e);
    return f.find(why);
}

}  // namespace onnxread
}  // namespace ajinative

#endif
