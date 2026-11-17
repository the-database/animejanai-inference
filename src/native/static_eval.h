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
