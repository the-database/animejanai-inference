// native/lower.cpp — see lower.h.
#include "native/lower.h"
#include "native/pack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

namespace ajinative {
namespace {

// ops whose only data input is input 0 (the rest are shapes, pads, scales, ...)
bool first_input_only(const std::string &op)
{
    static const std::set<std::string> s = {"Reshape", "Pad", "Slice", "Transpose", "Cast", "Identity", "Dropout",
                                            "DepthToSpace", "SpaceToDepth", "Resize", "Expand", "Squeeze",
                                            "Unsqueeze", "Tile", "Upsample"};
    return s.count(op) != 0;
}
bool passes_through(const std::string &op)
{
    return op == "Cast" || op == "Identity" || op == "Dropout" || op == "Pad" || op == "Slice";
}

struct Lowerer {
    const OnnxGraph &g;
    Plan &plan;
    std::string why;

    std::map<std::string, int> prod;          // tensor -> producing node
    std::map<std::string, std::string> alias; // pass-through output -> its input
    std::set<std::string> data;               // canonical data tensors
    std::map<std::string, std::vector<int>> cons;  // canonical tensor -> consuming data nodes
    std::vector<int> data_nodes;
    std::vector<bool> used;
    std::string X, Y;                          // canonical model input / output
    std::map<std::string, int> feat;           // canonical tensor -> feature id
    std::map<std::string, int> unshuffled;     // canonical tensor -> 0 CRD, 1 DCR (factor 2)

    Lowerer(const OnnxGraph &g_, Plan &p) : g(g_), plan(p), used(g_.nodes.size(), false) {}

    bool fail(const std::string &m)
    {
        if (why.empty()) why = m;
        return false;
    }
    std::string canon(const std::string &n) const
    {
        std::string c = n;
        for (auto it = alias.find(c); it != alias.end(); it = alias.find(c)) c = it->second;
        return c;
    }
    const OnnxNode &node(int i) const { return g.nodes[i]; }
    const std::vector<int> &consumers(const std::string &t)
    {
        static const std::vector<int> none;
        auto it = cons.find(canon(t));
        return it == cons.end() ? none : it->second;
    }
    // the single consumer of t with op `op`, or -1
    int sole(const std::string &t, const char *op)
    {
        const std::vector<int> &c = consumers(t);
        return c.size() == 1 && node(c[0]).op == op && !used[c[0]] ? c[0] : -1;
    }
    // the consumer of t with op `op` among exactly two consumers, or -1
    int one_of_two(const std::string &t, const char *op)
    {
        const std::vector<int> &c = consumers(t);
        if (c.size() != 2) return -1;
        for (int i : c)
            if (node(i).op == op && !used[i]) return i;
        return -1;
    }
    // the input of binary node n that is not `t` (canonical), or ""
    std::string other(int n, const std::string &t)
    {
        const OnnxNode &b = node(n);
        if (b.in.size() != 2) return "";
        const std::string a = canon(b.in[0]), c = canon(b.in[1]);
        if (a == canon(t)) return c;
        if (c == canon(t)) return a;
        return "";
    }

    const OnnxTensor *constant(const std::string &name)
    {
        auto it = g.inits.find(name);
        if (it != g.inits.end()) return &it->second;
        auto p = prod.find(name);
        if (p != prod.end() && node(p->second).op == "Constant") {
            const OnnxAttr *v = node(p->second).attr("value");
            if (v && v->type == 4) return &v->t;
        }
        return nullptr;
    }
    bool scalar(const std::string &name, float *v)
    {
        auto p = prod.find(name);
        if (p != prod.end() && node(p->second).op == "Constant") {
            const OnnxAttr *f = node(p->second).attr("value_float");
            if (f) { *v = f->f; return true; }
        }
        const OnnxTensor *t = constant(name);
        std::vector<float> f;
        if (!t || !t->floats(&f) || f.size() != 1) return false;
        *v = f[0];
        return true;
    }

    int new_feat(const std::string &t, int c)
    {
        const int id = (int)plan.channels.size();
        plan.channels.push_back(c);
        feat[canon(t)] = id;
        return id;
    }

    bool analyse()
    {
        if (g.inputs.size() != 1 || g.outputs.size() != 1) return fail("model needs one input and one output");
        for (size_t i = 0; i < g.nodes.size(); i++)
            for (const std::string &o : g.nodes[i].out) prod[o] = (int)i;
        X = g.inputs[0];
        data.insert(X);
        for (size_t i = 0; i < g.nodes.size(); i++) {
            const OnnxNode &n = g.nodes[i];
            if (n.op == "Constant" || n.op == "Shape" || n.op == "Size" || n.in.empty()) continue;
            const size_t ndata = first_input_only(n.op) ? 1 : n.in.size();
            bool is_data = false;
            for (size_t k = 0; k < ndata; k++)
                if (!n.in[k].empty() && data.count(canon(n.in[k]))) is_data = true;
            if (!is_data) continue;
            if (!n.domain.empty() && n.domain != "ai.onnx")
                return fail("custom op " + n.domain + "::" + n.op + " on the data path");
            data_nodes.push_back((int)i);
            if (passes_through(n.op) && !n.out.empty()) {
                if (!pass_through_ok(n)) return false;
                alias[n.out[0]] = canon(n.in[0]);
                used[i] = true;
                continue;
            }
            for (const std::string &o : n.out) data.insert(o);
            for (size_t k = 0; k < ndata; k++)
                if (!n.in[k].empty() && data.count(canon(n.in[k]))) cons[canon(n.in[k])].push_back((int)i);
        }
        Y = canon(g.outputs[0]);
        return true;
    }

    // Pad / Slice on the data path: accepted as alignment to the unshuffle factor when their
    // amounts are computed (from Shape) or are zero / open-ended; anything else is a real crop.
    bool pass_through_ok(const OnnxNode &n)
    {
        if (n.op == "Pad") {
            if (n.in.size() < 2 || n.in[1].empty()) {
                const OnnxAttr *p = n.attr("pads");  // opset < 11
                if (p && std::any_of(p->ints.begin(), p->ints.end(), [](int64_t v) { return v != 0; }))
                    return fail("Pad by a fixed amount");
                return true;
            }
            const OnnxTensor *t = constant(n.in[1]);
            std::vector<int64_t> v;
            if (!t) { plan.aligned_only = true; return true; }
            if (!t->ints(&v) || std::any_of(v.begin(), v.end(), [](int64_t x) { return x != 0; }))
                return fail("Pad by a fixed amount");
            return true;
        }
        if (n.op == "Slice") {
            if (n.in.size() < 3) return fail("Slice without starts/ends");
            const OnnxTensor *st = constant(n.in[1]), *en = constant(n.in[2]);
            std::vector<int64_t> s, e;
            if (st && (!st->ints(&s) || std::any_of(s.begin(), s.end(), [](int64_t x) { return x != 0; })))
                return fail("Slice that crops the start");
            if (n.in.size() > 4 && !n.in[4].empty()) {
                const OnnxTensor *sp = constant(n.in[4]);
                std::vector<int64_t> v;
                if (!sp || !sp->ints(&v) || std::any_of(v.begin(), v.end(), [](int64_t x) { return x != 1; }))
                    return fail("strided Slice");
            }
            if (en) {
                if (!en->ints(&e) || std::any_of(e.begin(), e.end(), [](int64_t x) { return x < (int64_t(1) << 30); }))
                    return fail("Slice by a fixed size");
            } else {
                plan.aligned_only = true;
            }
            return true;
        }
        return true;
    }

    // Reshape -> Transpose(0,1,3,5,2,4) -> Reshape: PyTorch's pixel-unshuffle (CRD order).
    bool unshuffle_reshape(int i)
    {
        const OnnxNode &r1 = node(i);
        if (canon(r1.in[0]) != X) return fail("Reshape on the data path");
        const int t = sole(r1.out[0], "Transpose");
        if (t < 0) return fail("Reshape on the data path");
        const OnnxAttr *perm = node(t).attr("perm");
        if (!perm || perm->ints != std::vector<int64_t>{0, 1, 3, 5, 2, 4}) return fail("Reshape/Transpose that is not a pixel-unshuffle");
        const int r2 = sole(node(t).out[0], "Reshape");
        if (r2 < 0) return fail("Reshape/Transpose that is not a pixel-unshuffle");
        used[i] = used[t] = used[r2] = true;
        unshuffled[canon(node(r2).out[0])] = 0;
        return true;
    }

    bool space_to_depth(int i)
    {
        const OnnxNode &n = node(i);
        if (canon(n.in[0]) != X || n.attr_i("blocksize", 0) != 2) return fail("SpaceToDepth other than 2x on the input");
        used[i] = true;
        unshuffled[canon(n.out[0])] = 1;
        return true;
    }

    // Conv weights OIHW -> [taps][cin][cout]; cin_from[k] / cout_from[n] pick the ONNX channel.
    bool weights(const OnnxNode &n, int k, PlanWeights *w, const std::vector<int> *cin_from, const std::vector<int> *cout_from)
    {
        const OnnxTensor *wt = n.in.size() > 1 ? constant(n.in[1]) : nullptr;
        std::vector<float> f, b;
        if (!wt || wt->dims.size() != 4 || wt->dims[2] != k || wt->dims[3] != k || !wt->floats(&f))
            return fail("Conv weights are not a constant " + std::to_string(k) + "x" + std::to_string(k) + " kernel");
        const int co = (int)wt->dims[0], ci = (int)wt->dims[1];
        if (wt->dtype != ONNX_FLOAT16) plan.fp32_weights = true;
        if (n.in.size() > 2 && !n.in[2].empty()) {
            const OnnxTensor *bt = constant(n.in[2]);
            if (!bt || !bt->floats(&b) || (int)b.size() != co) return fail("Conv bias is not a constant");
        } else {
            b.assign(co, 0.f);
        }
        w->taps = k * k;
        w->cin = ci;
        w->cout = co;
        w->w.resize((size_t)k * k * ci * co);
        w->bias.resize(co);
        for (int o = 0; o < co; o++) {
            const int so = cout_from ? (*cout_from)[o] : o;
            w->bias[o] = b[so];
            for (int c = 0; c < ci; c++) {
                const int sc = cin_from ? (*cin_from)[c] : c;
                for (int t = 0; t < k * k; t++)
                    w->w[((size_t)t * ci + c) * co + o] = float_to_half_bits(f[(((size_t)so * ci + sc) * k * k) + t]);
            }
        }
        return true;
    }

    bool conv_attrs_ok(const OnnxNode &n, int k)
    {
        if (n.attr_i("group", 1) != 1) return fail("grouped Conv");
        const std::string ap = n.attr_s("auto_pad", "NOTSET");
        if (ap != "NOTSET" && !(ap == "SAME_UPPER" || ap == "SAME_LOWER")) return fail("Conv auto_pad " + ap);
        for (const char *a : {"strides", "dilations"}) {
            const OnnxAttr *v = n.attr(a);
            if (v && std::any_of(v->ints.begin(), v->ints.end(), [](int64_t x) { return x != 1; })) return fail(std::string("Conv ") + a + " != 1");
        }
        const OnnxAttr *p = n.attr("pads");
        const int64_t want = k / 2;
        if (ap == "NOTSET" && want && !p) return fail("3x3 Conv without padding");
        if (p && std::any_of(p->ints.begin(), p->ints.end(), [&](int64_t x) { return x != want; }))
            return fail("Conv padding other than 'same'");
        return true;
    }

    // The 3x3 conv kernel size of node n, from its weights.
    int kernel_of(const OnnxNode &n)
    {
        const OnnxTensor *w = n.in.size() > 1 ? constant(n.in[1]) : nullptr;
        return w && w->dims.size() == 4 && w->dims[2] == w->dims[3] ? (int)w->dims[2] : 0;
    }

    // Depth-to-space after the conv producing `o`: DepthToSpace, or PyTorch's pixel-shuffle as
    // Reshape -> Transpose(0,1,4,2,5,3) -> Reshape. Returns the nodes in *ns, the factor in *r,
    // and whether the channel order is DCR.
    bool depth_to_space_after(const std::string &o, std::vector<int> *ns, int *r, bool *dcr, std::string *out)
    {
        int d = sole(o, "DepthToSpace");
        if (d >= 0) {
            *r = (int)node(d).attr_i("blocksize", 0);
            const std::string mode = node(d).attr_s("mode", "DCR");
            if (mode != "DCR" && mode != "CRD") return false;
            *dcr = mode == "DCR";
            ns->push_back(d);
            *out = node(d).out[0];
            return true;
        }
        const int r1 = sole(o, "Reshape");
        if (r1 < 0) return false;
        const int t = sole(node(r1).out[0], "Transpose");
        if (t < 0) return false;
        const OnnxAttr *perm = node(t).attr("perm");
        if (!perm || perm->ints != std::vector<int64_t>{0, 1, 4, 2, 5, 3}) return false;
        const int r2 = sole(node(t).out[0], "Reshape");
        if (r2 < 0) return false;
        *r = 0;  // from the channel count
        *dcr = false;
        *ns = {r1, t, r2};
        *out = node(r2).out[0];
        return true;
    }

    bool conv(int i)
    {
        const OnnxNode &n = node(i);
        const int k = kernel_of(n);
        if (k != 3) return fail(k == 1 ? "1x1 Conv outside a concat or attention" : "Conv kernel other than 3x3");
        if (!conv_attrs_ok(n, 3)) return false;
        used[i] = true;

        PlanStep st;
        st.kind = PlanStep::CONV;
        const std::string s = canon(n.in[0]);
        std::vector<int> cin_from;
        const std::vector<int> *cin_map = nullptr;
        if (s == X) {
            st.in = 1;  // NCHW3
            st.src = kModelInput;
        } else if (unshuffled.count(s)) {
            st.in = 2;  // NCHW3Unshuffle2
            st.src = kModelInput;
            plan.unshuffle = 2;
            if (unshuffled[s] == 1) {  // SpaceToDepth (DCR): ONNX channel (ry*2+rx)*3 + c
                for (int u = 0; u < 12; u++) cin_from.push_back(((u % 4) / 2 * 2 + u % 2) * 3 + u / 4);
                cin_map = &cin_from;
            }
        } else if (feat.count(s)) {
            st.in = 0;
            st.src = feat[s];
        } else {
            return fail("Conv input is not the model input or a supported tensor");
        }

        const std::string o = canon(n.out[0]);
        int r = 0;
        bool dcr = false;
        std::vector<int> d2s;
        std::string d2s_out;
        if (depth_to_space_after(o, &d2s, &r, &dcr, &d2s_out)) {
            // last conv -> depth-to-space -> output (+ the nearest-upsampled input)
            const OnnxTensor *wt = constant(n.in[1]);
            const int co = (int)wt->dims[0];
            if (r == 0) r = co == 12 ? 2 : co == 48 ? 4 : 0;
            if ((r != 2 && r != 4) || co != 3 * r * r) return fail("depth-to-space other than 2x/4x to RGB");
            std::vector<int> cout_from;
            for (int u = 0; u < co; u++) {
                const int c = u / (r * r), ry = (u / r) % r, rx = u % r;
                cout_from.push_back(dcr ? (ry * r + rx) * 3 + c : u);
            }
            if (!weights(n, 3, &st.w, cin_map, &cout_from)) return false;
            for (int d : d2s) used[d] = true;
            st.out = r == 2 ? 1 : 2;
            st.dst = kModelOutput;
            const std::string dout = canon(d2s_out);
            if (dout != Y) {
                const int a = sole(dout, "Add");
                if (a < 0 || canon(node(a).out[0]) != Y) return fail("depth-to-space output is not the model output");
                const std::string z = other(a, dout);
                auto pz = prod.find(z);
                if (z.empty() || pz == prod.end() || node(pz->second).op != "Resize")
                    return fail("the output adds something other than the upsampled input");
                const OnnxNode &rz = node(pz->second);
                const std::string ct = rz.attr_s("coordinate_transformation_mode", "half_pixel");
                if (canon(rz.in[0]) != X || rz.attr_s("mode", "nearest") != "nearest" ||
                    (ct != "asymmetric" && ct != "half_pixel"))
                    return fail("the output adds an input upsampled other than nearest");
                used[a] = used[pz->second] = true;
                st.resid = kModelInput;
            }
            plan.depth_to_space = r;
            st.cin = st.w.cin;
            st.cout = st.w.cout;
            plan.steps.push_back(st);
            return true;
        }

        if (!weights(n, 3, &st.w, cin_map, nullptr)) return false;
        st.cin = st.w.cin;
        st.cout = st.w.cout;
        if (st.src >= 0 && plan.channels[st.src] != st.cin) return fail("Conv input channels do not match");

        // SPAN gates: the conv output o and the block input x
        int a = one_of_two(o, "Add");
        if (a >= 0) {
            const std::string x = other(a, o);
            const std::string ao = canon(node(a).out[0]);
            const int m = sole(ao, "Mul");
            const int k1 = one_of_two(o, "Conv"), sg = one_of_two(o, "Sigmoid");
            if (m >= 0 && feat.count(x) && plan.channels[feat[x]] == st.cout) {
                // tanh(1x1 conv(o))
                if (k1 >= 0 && kernel_of(node(k1)) == 1 && conv_attrs_ok(node(k1), 1)) {
                    const int th = sole(node(k1).out[0], "Tanh");
                    if (th >= 0 && other(m, ao) == canon(node(th).out[0])) {
                        if (!weights(node(k1), 1, &st.att, nullptr, nullptr)) return false;
                        if (st.att.cin != st.cout || st.att.cout != st.cout) return fail("attention conv is not cout -> cout");
                        used[a] = used[k1] = used[th] = used[m] = true;
                        st.gate = 2;
                        st.resid = feat[x];
                        st.dst = new_feat(node(m).out[0], st.cout);
                        plan.steps.push_back(st);
                        return true;
                    }
                }
                // sigmoid(o) - 0.5
                if (sg >= 0) {
                    const int sub = sole(node(sg).out[0], "Sub");
                    float half = 0.f;
                    if (sub >= 0 && canon(node(sub).in[0]) == canon(node(sg).out[0]) && scalar(node(sub).in[1], &half) &&
                        half == 0.5f && other(m, ao) == canon(node(sub).out[0])) {
                        used[a] = used[sg] = used[sub] = used[m] = true;
                        st.gate = 3;
                        st.resid = feat[x];
                        st.dst = new_feat(node(m).out[0], st.cout);
                        plan.steps.push_back(st);
                        return true;
                    }
                }
            }
        }
        a = sole(o, "Add");
        if (a >= 0) {
            const std::string x = other(a, o);
            if (feat.count(x) && plan.channels[feat[x]] == st.cout) {
                used[a] = true;
                st.gate = 1;
                st.resid = feat[x];
                st.dst = new_feat(node(a).out[0], st.cout);
                plan.steps.push_back(st);
                return true;
            }
        }

        // activation; if the pre-activation tensor is used elsewhere too, it is computed by a
        // second, plain copy of the conv
        const std::vector<int> &oc = consumers(o);
        int act_node = -1, mul = -1;
        for (int c : oc) {
            if (used[c]) continue;
            const std::string &op = node(c).op;
            if (op == "Relu" || op == "PRelu" || op == "LeakyRelu") { act_node = c; break; }
            if (op == "Sigmoid") {
                const int mm = sole(node(c).out[0], "Mul");
                if (mm >= 0 && other(mm, node(c).out[0]) == o) { act_node = c; mul = mm; break; }
            }
        }
        if (act_node >= 0) {
            const OnnxNode &an = node(act_node);
            std::string aout = an.out[0];
            if (an.op == "Relu") st.act = 1;
            if (an.op == "Sigmoid") { st.act = 3; aout = node(mul).out[0]; }
            if (an.op == "PRelu" || an.op == "LeakyRelu") {
                st.act = 2;
                if (an.op == "LeakyRelu") {
                    const OnnxAttr *al = an.attr("alpha");
                    st.w.slope.assign(st.cout, al ? al->f : 0.01f);
                } else {
                    const OnnxTensor *sl = an.in.size() > 1 ? constant(an.in[1]) : nullptr;
                    std::vector<float> v;
                    if (!sl || !sl->floats(&v) || (v.size() != 1 && (int)v.size() != st.cout)) return fail("PRelu slope is not per channel");
                    st.w.slope.resize(st.cout);
                    for (int c = 0; c < st.cout; c++) st.w.slope[c] = v.size() == 1 ? v[0] : v[c];
                }
            }
            const size_t others = oc.size() - (mul >= 0 ? 2 : 1);
            if (others > 0) {
                PlanStep pre = st;
                pre.act = 0;
                pre.w.slope.clear();
                pre.dst = new_feat(o, st.cout);
                plan.steps.push_back(pre);
            }
            used[act_node] = true;
            if (mul >= 0) used[mul] = true;
            st.dst = new_feat(aout, st.cout);
            plan.steps.push_back(st);
            return true;
        }

        st.dst = new_feat(o, st.cout);
        plan.steps.push_back(st);
        return true;
    }

    bool concat(int i)
    {
        const OnnxNode &n = node(i);
        const int64_t axis = n.attr_i("axis", 0);
        if (axis != 1 && axis != -3) return fail("Concat on an axis other than channels");
        const int ns = (int)n.in.size();
        if (ns < 2 || ns > 4) return fail("Concat of " + std::to_string(ns) + " tensors");
        PlanStep st;
        st.kind = PlanStep::CONCAT;
        st.nsrc = ns;
        for (int k = 0; k < ns; k++) {
            const std::string s = canon(n.in[k]);
            if (!feat.count(s)) return fail("Concat input is not a supported tensor");
            st.srcs[k] = feat[s];
            if (k && plan.channels[st.srcs[k]] != plan.channels[st.srcs[0]]) return fail("Concat of different channel counts");
        }
        st.c = plan.channels[st.srcs[0]];
        const int k1 = sole(n.out[0], "Conv");
        if (k1 < 0 || kernel_of(node(k1)) != 1 || !conv_attrs_ok(node(k1), 1)) return fail("Concat not followed by a 1x1 Conv");
        if (!weights(node(k1), 1, &st.w, nullptr, nullptr)) return false;
        if (st.w.cin != ns * st.c || st.w.cout != st.c) return fail("concat 1x1 Conv is not n*c -> c");
        st.cin = st.w.cin;
        st.cout = st.w.cout;
        used[i] = used[k1] = true;
        st.dst = new_feat(node(k1).out[0], st.c);
        plan.steps.push_back(st);
        return true;
    }

    bool run()
    {
        if (!analyse()) return false;
        for (int i : data_nodes) {
            if (used[i]) continue;
            const std::string &op = node(i).op;
            bool ok;
            if (op == "Conv") ok = conv(i);
            else if (op == "Concat") ok = concat(i);
            else if (op == "Reshape") ok = unshuffle_reshape(i);
            else if (op == "SpaceToDepth") ok = space_to_depth(i);
            else ok = fail("unsupported op " + op + " on the data path");
            if (!ok) return fail("unsupported pattern at " + op + " " + node(i).name);
        }
        for (int i : data_nodes)
            if (!used[i]) return fail("op " + node(i).op + " " + node(i).name + " is not part of a supported pattern");
        bool out = false;
        for (const PlanStep &s : plan.steps) out = out || s.dst == kModelOutput;
        if (!out) return fail("the model does not end in a depth-to-space to RGB");
        if (plan.steps.back().dst != kModelOutput) return fail("ops after the output conv");
        return true;
    }
};

}  // namespace

bool lower(const OnnxGraph &g, Plan *plan, std::string *why)
{
    *plan = Plan{};
    Lowerer l(g, *plan);
    if (l.run()) return true;
    if (why) *why = l.why;
    *plan = Plan{};
    return false;
}

std::string describe(const Plan &p)
{
    std::set<int> cs(p.channels.begin(), p.channels.end());
    std::string ch;
    for (int c : cs) ch += (ch.empty() ? "" : "/") + std::to_string(c);
    char buf[200];
    std::snprintf(buf, sizeof(buf), "%zu kernels, %s ch%s, depth-to-space %d%s", p.steps.size(), ch.c_str(),
                  p.unshuffle > 1 ? ", pixel-unshuffle 2" : "", p.depth_to_space,
                  p.fp32_weights ? ", fp32 model computed in fp16" : "");
    return buf;
}

}  // namespace ajinative
