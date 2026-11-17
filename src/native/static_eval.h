// native/static_eval.h — a host evaluator for an ONNX graph's constant subgraphs (DAT's dynamic
// position bias: the MLP over relative offsets the export leaves in the graph), shared by the CUDA
// and Vulkan builders. Host-only.
#ifndef AJI_NATIVE_STATIC_EVAL_H
#define AJI_NATIVE_STATIC_EVAL_H

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"

namespace ajinative {
namespace static_eval {

using onnxread::Reader;

// The graph is untrusted: every value's dims must match its data (shape_ok), each op checks its
// inputs' shapes and indices, and a value or a Gemm's work is capped.
struct HT {
    std::vector<int64_t> dims;
    std::vector<double> v;
};
inline constexpr size_t kMaxEval = OnnxTensor::kMaxElems;
inline size_t numel(const std::vector<int64_t> &d)  // kMaxEval + 1 when negative or too large
{
    if (d.size() > 8) return kMaxEval + 1;
    size_t n = 1;
    for (int64_t x : d) {
        if (x < 0 || (uint64_t)x > kMaxEval) return kMaxEval + 1;
        n *= (size_t)x;
        if (n > kMaxEval) return kMaxEval + 1;
    }
    return n;
}
inline bool shape_ok(const HT &t) { return numel(t.dims) <= kMaxEval && numel(t.dims) == t.v.size(); }

class StaticEval {
public:
    explicit StaticEval(const Reader &r) : rd(r) {}
    bool eval(const std::string &name, HT *out, int depth = 0)
    {
        auto it = memo.find(name);
        if (it != memo.end()) {
            *out = it->second;
            return true;
        }
        HT t;
        if (const OnnxTensor *c = rd.tensor(name)) {
            t.dims = c->dims;
            std::vector<float> f;
            std::vector<int64_t> i;
            if (c->floats(&f))
                for (float x : f) t.v.push_back((double)x);
            else if (c->ints(&i))
                for (int64_t x : i) t.v.push_back((double)x);
            else
                return false;
        } else {
            auto p = rd.prod.find(name);
            if (p == rd.prod.end() || depth > 64 || !op(*p->second, &t, depth)) return false;
        }
        if (!shape_ok(t)) return false;
        memo[name] = t;
        *out = t;
        return true;
    }

private:
    const Reader &rd;
    std::map<std::string, HT> memo;

    static bool to_ints(const HT &t, std::vector<int64_t> *o)
    {
        o->clear();
        for (double x : t.v) {
            if (!(std::fabs(x) < 9.0e15) || x != std::floor(x)) return false;
            o->push_back((int64_t)x);
        }
        return true;
    }

    bool op(const OnnxNode &n, HT *o, int depth)
    {
        std::vector<HT> in(n.in.size());
        std::vector<bool> have(n.in.size(), false);
        for (size_t i = 0; i < n.in.size(); i++) {
            if (n.in[i].empty()) continue;
            if (!eval(n.in[i], &in[i], depth + 1)) return false;
            have[i] = true;
        }
        auto need = [&](size_t k) {
            for (size_t i = 0; i < k; i++)
                if (i >= have.size() || !have[i]) return false;
            return true;
        };
        const bool third = in.size() > 2 && have[2];
        if (n.op == "Relu") {
            if (!need(1)) return false;
            *o = in[0];
            for (double &x : o->v) x = std::max(x, 0.0);
        } else if (n.op == "Cast" || n.op == "Identity") {
            if (!need(1)) return false;
            *o = in[0];
        } else if (n.op == "LayerNormalization") {  // over the last axis
            if (!need(2) || in[0].dims.empty()) return false;
            const int64_t axis = n.attr_i("axis", -1);
            if (axis != -1 && axis != (int64_t)in[0].dims.size() - 1) return false;
            const size_t C = (size_t)in[0].dims.back();
            if (!C || in[1].v.size() != C || (third && in[2].v.size() != C)) return false;
            const size_t rows = in[0].v.size() / C;
            *o = in[0];
            const double eps = n.attr("epsilon") ? n.attr("epsilon")->f : 1e-5;
            for (size_t r = 0; r < rows; r++) {
                double m = 0, q = 0;
                for (size_t c = 0; c < C; c++) m += in[0].v[r * C + c];
                m /= C;
                for (size_t c = 0; c < C; c++) q += (in[0].v[r * C + c] - m) * (in[0].v[r * C + c] - m);
                const double rs = 1.0 / std::sqrt(q / C + eps);
                for (size_t c = 0; c < C; c++) o->v[r * C + c] = (in[0].v[r * C + c] - m) * rs * in[1].v[c] + (third ? in[2].v[c] : 0.0);
            }
        } else if (n.op == "Gemm") {
            if (!need(2)) return false;
            const bool ta = n.attr_i("transA", 0), tb = n.attr_i("transB", 0);
            const double al = n.attr("alpha") ? n.attr("alpha")->f : 1.0, be = n.attr("beta") ? n.attr("beta")->f : 1.0;
            if (ta || in[0].dims.size() != 2 || in[1].dims.size() != 2) return false;
            const size_t M = in[0].dims[0], K = in[0].dims[1], N = tb ? in[1].dims[0] : in[1].dims[1];
            if ((size_t)(tb ? in[1].dims[1] : in[1].dims[0]) != K) return false;
            if (M * N > kMaxEval || (double)M * N * K > 1e9) return false;
            if (third && in[2].v.size() != N && in[2].v.size() != M * N && in[2].v.size() != 1) return false;
            o->dims = {(int64_t)M, (int64_t)N};
            o->v.assign(M * N, 0.0);
            for (size_t i = 0; i < M; i++)
                for (size_t j = 0; j < N; j++) {
                    double s = 0;
                    for (size_t k = 0; k < K; k++) s += in[0].v[i * K + k] * (tb ? in[1].v[j * K + k] : in[1].v[k * N + j]);
                    double c = 0;
                    if (third) c = in[2].v.size() == 1 ? in[2].v[0] : in[2].v.size() == N ? in[2].v[j] : in[2].v[i * N + j];
                    o->v[i * N + j] = al * s + be * c;
                }
        } else if (n.op == "Reshape") {
            std::vector<int64_t> sh;
            if (!need(2) || !to_ints(in[1], &sh) || sh.size() > 8) return false;
            const size_t total = in[0].v.size();
            int neg = -1;
            size_t known = 1;
            for (size_t i = 0; i < sh.size(); i++) {
                if (sh[i] == 0 && !n.attr_i("allowzero", 0)) {
                    if (i >= in[0].dims.size()) return false;
                    sh[i] = in[0].dims[i];
                }
                if (sh[i] == -1) {
                    if (neg >= 0) return false;
                    neg = (int)i;
                } else if (sh[i] < 0 || (uint64_t)sh[i] > kMaxEval) {
                    return false;
                } else {
                    known *= (size_t)sh[i];
                    if (known > kMaxEval) return false;
                }
            }
            if (neg >= 0) {
                if (!known || total % known) return false;
                sh[neg] = (int64_t)(total / known);
            }
            if (numel(sh) != total) return false;
            *o = in[0];
            o->dims = sh;
        } else if (n.op == "Unsqueeze") {
            if (!need(1)) return false;
            std::vector<int64_t> ax;
            if (in.size() > 1 && have[1]) {
                if (!to_ints(in[1], &ax)) return false;
            } else if (const OnnxAttr *a = n.attr("axes")) {
                ax = a->ints;
            } else {
                return false;
            }
            *o = in[0];
            const int64_t rank = (int64_t)o->dims.size() + (int64_t)ax.size();
            if (rank > 8) return false;
            for (int64_t &a : ax) {
                if (a < 0) a += rank;
                if (a < 0 || a >= rank) return false;
            }
            std::sort(ax.begin(), ax.end());
            if (std::adjacent_find(ax.begin(), ax.end()) != ax.end()) return false;
            for (int64_t a : ax) o->dims.insert(o->dims.begin() + a, 1);
        } else if (n.op == "Transpose") {
            if (!need(1)) return false;
            const OnnxAttr *pa = n.attr("perm");
            const size_t r = in[0].dims.size();
            if (pa && pa->ints.size() != r) return false;
            std::vector<int64_t> perm(r);
            std::vector<bool> seen(r, false);
            for (size_t i = 0; i < r; i++) {
                perm[i] = pa ? pa->ints[i] : (int64_t)(r - 1 - i);
                if (perm[i] < 0 || perm[i] >= (int64_t)r || seen[perm[i]]) return false;
                seen[perm[i]] = true;
            }
            o->dims.resize(r);
            for (size_t i = 0; i < r; i++) o->dims[i] = in[0].dims[perm[i]];
            o->v.assign(in[0].v.size(), 0.0);
            std::vector<size_t> sin(r, 1), idx(r, 0);
            for (int i = (int)r - 2; i >= 0; i--) sin[i] = sin[i + 1] * in[0].dims[i + 1];
            for (size_t k = 0; k < o->v.size(); k++) {
                size_t rem = k, src = 0;
                for (int i = (int)r - 1; i >= 0; i--) {
                    idx[i] = rem % o->dims[i];
                    rem /= o->dims[i];
                }
                for (size_t i = 0; i < r; i++) src += idx[i] * sin[perm[i]];
                o->v[k] = in[0].v[src];
            }
        } else if (n.op == "GatherND") {  // batch_dims 0
            if (!need(2) || n.attr_i("batch_dims", 0) || in[1].dims.empty()) return false;
            const int64_t kk64 = in[1].dims.back();
            if (kk64 < 1 || kk64 > (int64_t)in[0].dims.size()) return false;
            const size_t kk = (size_t)kk64, nidx = in[1].v.size() / kk;
            std::vector<int64_t> ix;
            if (!to_ints(in[1], &ix)) return false;
            size_t inner = 1;
            for (size_t i = kk; i < in[0].dims.size(); i++) inner *= (size_t)in[0].dims[i];
            o->dims.assign(in[1].dims.begin(), in[1].dims.end() - 1);
            for (size_t i = kk; i < in[0].dims.size(); i++) o->dims.push_back(in[0].dims[i]);
            if (numel(o->dims) > kMaxEval) return false;
            o->v.assign(nidx * inner, 0.0);
            for (size_t q = 0; q < nidx; q++) {
                size_t off = 0;
                for (size_t a = 0; a < kk; a++) {
                    int64_t x = ix[q * kk + a];
                    const int64_t dim = in[0].dims[a];
                    if (x < 0) x += dim;
                    if (x < 0 || x >= dim) return false;
                    off = off * (size_t)dim + (size_t)x;
                }
                for (size_t e = 0; e < inner; e++) o->v[q * inner + e] = in[0].v[off * inner + e];
            }
        } else if (n.op == "Gather") {  // the table lookup of the legacy exports' position bias
            if (!need(2) || in[0].dims.empty()) return false;
            const int64_t r = (int64_t)in[0].dims.size();
            int64_t axis = n.attr_i("axis", 0);
            if (axis < 0) axis += r;
            if (axis < 0 || axis >= r) return false;
            std::vector<int64_t> ix;
            if (!to_ints(in[1], &ix)) return false;
            size_t outer = 1, inner = 1;
            for (int64_t i = 0; i < axis; i++) outer *= (size_t)in[0].dims[i];
            for (int64_t i = axis + 1; i < r; i++) inner *= (size_t)in[0].dims[i];
            const int64_t dim = in[0].dims[axis];
            o->dims.assign(in[0].dims.begin(), in[0].dims.begin() + axis);
            o->dims.insert(o->dims.end(), in[1].dims.begin(), in[1].dims.end());
            o->dims.insert(o->dims.end(), in[0].dims.begin() + axis + 1, in[0].dims.end());
            if (numel(o->dims) > kMaxEval) return false;
            o->v.assign(outer * ix.size() * inner, 0.0);
            for (size_t a = 0; a < outer; a++)
                for (size_t q = 0; q < ix.size(); q++) {
                    int64_t x = ix[q];
                    if (x < 0) x += dim;
                    if (x < 0 || x >= dim) return false;
                    for (size_t e = 0; e < inner; e++)
                        o->v[(a * ix.size() + q) * inner + e] = in[0].v[(a * (size_t)dim + (size_t)x) * inner + e];
                }
        } else if (n.op == "Squeeze") {
            if (!need(1)) return false;
            std::vector<int64_t> ax;
            if (in.size() > 1 && have[1]) {
                if (!to_ints(in[1], &ax)) return false;
            } else if (const OnnxAttr *a = n.attr("axes")) {
                ax = a->ints;
            }
            const int64_t r = (int64_t)in[0].dims.size();
            std::vector<bool> drop(r, false);
            for (int64_t a : ax) {
                if (a < 0) a += r;
                if (a < 0 || a >= r || in[0].dims[a] != 1) return false;
                drop[a] = true;
            }
            *o = in[0];
            o->dims.clear();
            for (int64_t i = 0; i < r; i++)
                if (!(ax.empty() ? in[0].dims[i] == 1 : drop[i])) o->dims.push_back(in[0].dims[i]);
        } else if (n.op == "Flatten") {
            if (!need(1)) return false;
            const int64_t r = (int64_t)in[0].dims.size();
            int64_t axis = n.attr_i("axis", 1);
            if (axis < 0) axis += r;
            if (axis < 0 || axis > r) return false;
            int64_t a = 1;
            for (int64_t i = 0; i < axis; i++) a *= in[0].dims[i];
            *o = in[0];
            o->dims = {a, a ? (int64_t)(in[0].v.size() / (size_t)a) : 0};
        } else if (n.op == "Add" || n.op == "Sub" || n.op == "Mul" || n.op == "Div") {  // numpy broadcasting
            if (!need(2)) return false;
            const size_t r = std::max(in[0].dims.size(), in[1].dims.size());
            std::vector<int64_t> da(r, 1), db(r, 1);
            std::copy(in[0].dims.begin(), in[0].dims.end(), da.begin() + (r - in[0].dims.size()));
            std::copy(in[1].dims.begin(), in[1].dims.end(), db.begin() + (r - in[1].dims.size()));
            o->dims.resize(r);
            for (size_t i = 0; i < r; i++) {
                if (da[i] != db[i] && da[i] != 1 && db[i] != 1) return false;
                o->dims[i] = std::max(da[i], db[i]);
            }
            if (numel(o->dims) > kMaxEval) return false;
            o->v.assign(numel(o->dims), 0.0);
            const char k = n.op[0];
            std::vector<size_t> idx(r);
            for (size_t e = 0; e < o->v.size(); e++) {
                size_t rem = e, ia = 0, ib = 0, sa = 1, sb = 1;
                for (int i = (int)r - 1; i >= 0; i--) {
                    const size_t x = rem % (size_t)o->dims[i];
                    rem /= (size_t)o->dims[i];
                    ia += (da[i] == 1 ? 0 : x) * sa;
                    ib += (db[i] == 1 ? 0 : x) * sb;
                    sa *= (size_t)da[i];
                    sb *= (size_t)db[i];
                }
                const double a = in[0].v[ia], b = in[1].v[ib];
                o->v[e] = k == 'A' ? a + b : k == 'S' ? a - b : k == 'M' ? a * b : a / b;
            }
        } else if (n.op == "Sqrt" || n.op == "Neg" || n.op == "Abs" || n.op == "Exp" || n.op == "Erf" || n.op == "Sigmoid" || n.op == "Tanh") {
            if (!need(1)) return false;
            *o = in[0];
            for (double &x : o->v)
                x = n.op == "Sqrt" ? std::sqrt(x) : n.op == "Neg" ? -x : n.op == "Abs" ? std::fabs(x) : n.op == "Exp" ? std::exp(x)
                    : n.op == "Erf" ? std::erf(x) : n.op == "Sigmoid" ? 1.0 / (1.0 + std::exp(-x)) : std::tanh(x);
        } else if (n.op == "Pow") {
            if (!need(2) || in[1].v.size() != 1) return false;  // a scalar exponent
            *o = in[0];
            const double e = in[1].v[0];
            for (double &x : o->v) x = e == 2.0 ? x * x : e == 0.5 ? std::sqrt(x) : std::pow(x, e);
        } else if (n.op == "ReduceMean" || n.op == "ReduceSum") {  // the spelled-out LayerNorm's means
            if (!need(1) || in[0].dims.empty()) return false;
            std::vector<int64_t> ax;
            if (in.size() > 1 && have[1]) {
                if (!to_ints(in[1], &ax)) return false;
            } else if (const OnnxAttr *a = n.attr("axes")) {
                ax = a->ints;
            }
            const int64_t r = (int64_t)in[0].dims.size();
            std::vector<bool> red(r, ax.empty() && !n.attr_i("noop_with_empty_axes", 0));
            for (int64_t a : ax) {
                if (a < 0) a += r;
                if (a < 0 || a >= r) return false;
                red[a] = true;
            }
            const bool keep = n.attr_i("keepdims", 1) != 0;
            std::vector<int64_t> od;
            for (int64_t i = 0; i < r; i++)
                if (!red[i]) od.push_back(in[0].dims[i]);
                else if (keep) od.push_back(1);
            o->dims = od;
            o->v.assign(numel(od), 0.0);
            std::vector<int64_t> kd(r);  // the kept dims (1 where reduced), for the output index
            for (int64_t i = 0; i < r; i++) kd[i] = red[i] ? 1 : in[0].dims[i];
            size_t cnt = 1;
            for (int64_t i = 0; i < r; i++)
                if (red[i]) cnt *= (size_t)in[0].dims[i];
            for (size_t e = 0; e < in[0].v.size(); e++) {
                size_t rem = e, oi = 0, st = 1;
                for (int64_t i = r - 1; i >= 0; i--) {
                    const size_t x = rem % (size_t)in[0].dims[i];
                    rem /= (size_t)in[0].dims[i];
                    oi += (red[i] ? 0 : x) * st;
                    st *= (size_t)kd[i];
                }
                o->v[oi] += in[0].v[e];
            }
            if (n.op == "ReduceMean" && cnt)
                for (double &x : o->v) x /= (double)cnt;
        } else if (n.op == "Concat") {
            if (in.empty() || !need(in.size()) || in[0].dims.empty()) return false;
            const int64_t r = (int64_t)in[0].dims.size();
            int64_t axis = n.attr_i("axis", 0);
            if (axis < 0) axis += r;
            if (axis < 0 || axis >= r) return false;
            o->dims = in[0].dims;
            o->dims[axis] = 0;
            for (const HT &t : in) {
                if ((int64_t)t.dims.size() != r) return false;
                for (int64_t i = 0; i < r; i++)
                    if (i != axis && t.dims[i] != in[0].dims[i]) return false;
                o->dims[axis] += t.dims[axis];
            }
            size_t outer = 1;
            for (int64_t i = 0; i < axis; i++) outer *= (size_t)in[0].dims[i];
            o->v.clear();
            for (size_t a = 0; a < outer; a++)
                for (const HT &t : in) {
                    const size_t blk = t.v.size() / outer;
                    o->v.insert(o->v.end(), t.v.begin() + a * blk, t.v.begin() + (a + 1) * blk);
                }
        } else {
            return false;
        }
        return shape_ok(*o);
    }
};

}  // namespace static_eval
using static_eval::HT;
using static_eval::StaticEval;
}  // namespace ajinative

#endif
