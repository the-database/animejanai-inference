// The forms exporters give the same model (native/export_forms.h), the host evaluator's ops
// (native/static_eval.h) and the Reader's folding of unfolded constants (native/onnx_read.h), on
// small graphs written here. Numbers are checked against values worked out by hand. CPU only.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "native/export_forms.h"
#include "native/onnx_graph.h"
#include "native/onnx_read.h"
#include "native/static_eval.h"
#include "onnx_pb.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

using namespace ajinative;
using namespace ajinative::onnxread;
using namespace onnx_pb;

namespace {

bool near(double a, double b) { return std::fabs(a - b) < 1e-5 * (1 + std::fabs(b)); }

bool parse(G &g, const char *in, const char *out, OnnxGraph *og)
{
    const std::string b = g.bytes(in, out);
    std::string why;
    return parse_onnx(b.data(), b.size(), og, &why);
}

const OnnxNode *find_op(const OnnxGraph &g, const char *op, int nth = 0)
{
    for (const OnnxNode &n : g.nodes)
        if (n.op == op && nth-- == 0) return &n;
    return nullptr;
}

}  // namespace

int main()
{
    // ---- Reader: a MatMul weight through Identity and Transpose, read as the folded tensor
    {
        G g;
        g.init(tensor_v("w", {2, 3}, {1, 2, 3, 4, 5, 6}));
        g.node("Identity", {"w"}, {"wi"});
        g.node("Transpose", {"wi"}, {"wt"}, {attr_ints("perm", {1, 0})});
        g.node("MatMul", {"input", "wt"}, {"output"});
        OnnxGraph og;
        CHECK(parse(g, "input", "output", &og));
        Reader rd(og);
        const OnnxTensor *t = rd.tensor("wt");
        std::vector<float> v;
        CHECK(t && t->dims == (std::vector<int64_t>{3, 2}) && t->floats(&v));
        CHECK(v == (std::vector<float>{1, 4, 2, 5, 3, 6}));
        CHECK(!rd.tensor("input") && !rd.tensor("output"));
    }

    // ---- StaticEval: a position-bias-like subgraph (Gather of a table by index, a spelled-out
    // LayerNorm's ReduceMean / Pow / Sqrt, broadcast arithmetic, Concat)
    {
        G g;
        g.init(tensor_v("tab", {3, 2}, {1, 2, 3, 4, 5, 6}));
        g.init(tensor_i("idx", {2, 2}, {2, 0, 1, -1}));
        g.node("Gather", {"tab", "idx"}, {"gt"}, {attr_i("axis", 0)});  // [2,2,2]
        g.node("ReduceMean", {"gt"}, {"m"}, {attr_ints("axes", {-1})});  // [2,2,1]
        g.init(tensor_v("two", {}, {2}));
        g.node("Pow", {"m", "two"}, {"p"});
        g.node("Sqrt", {"p"}, {"q"});
        g.init(tensor_v("row", {2}, {10, 20}));
        g.node("Add", {"q", "row"}, {"a"});  // [2,2,1] + [2] -> [2,2,2]
        g.node("Concat", {"a", "gt"}, {"c"}, {attr_i("axis", 2)});  // [2,2,4]
        g.node("Add", {"input", "c"}, {"output"});
        OnnxGraph og;
        CHECK(parse(g, "input", "output", &og));
        Reader rd(og);
        StaticEval se(rd);
        HT gt, c;
        CHECK(se.eval("gt", &gt) && gt.dims == (std::vector<int64_t>{2, 2, 2}));
        CHECK(gt.v == (std::vector<double>{5, 6, 1, 2, 3, 4, 5, 6}));
        CHECK(se.eval("c", &c) && c.dims == (std::vector<int64_t>{2, 2, 4}));
        // row 0: mean(5, 6) = 5.5 -> 5.5 + (10, 20), then the gathered (5, 6)
        CHECK(near(c.v[0], 15.5) && near(c.v[1], 25.5) && c.v[2] == 5 && c.v[3] == 6);
        CHECK(near(c.v[12], 15.5) && c.v[14] == 5);  // index -1 -> the last row
        HT bad;
        CHECK(!se.eval("output", &bad));  // depends on the input
    }

    // ---- a spelled-out LayerNorm (opset < 17), the GELU's constant Mul, F.normalize's L2 root,
    // chunk as two Slices
    {
        G g;
        g.init(tensor_v("eps", {}, {1e-6f}));
        g.init(tensor_v("two", {}, {2}));
        g.init(tensor_v("half", {}, {0.5f}));
        g.init(tensor_v("one", {}, {1}));
        g.init(tensor_v("rt2", {}, {1.41421356f}));
        g.init(tensor_v("gam", {4}, {1, 2, 3, 4}));
        g.init(tensor_v("bet", {4}, {0, 0, 1, 1}));
        g.node("ReduceMean", {"input"}, {"mu"}, {attr_ints("axes", {-1})});
        g.node("Sub", {"input", "mu"}, {"d"});
        g.node("Pow", {"d", "two"}, {"d2"});
        g.node("ReduceMean", {"d2"}, {"var"}, {attr_ints("axes", {-1})});
        g.node("Add", {"var", "eps"}, {"ve"});
        g.node("Sqrt", {"ve"}, {"sd"});
        g.node("Div", {"d", "sd"}, {"nrm"});
        g.node("Mul", {"nrm", "gam"}, {"ng"});
        g.node("Add", {"ng", "bet"}, {"ln"});
        // GELU: x * 0.5 * (1 + erf(x / sqrt 2)), the 0.5 last
        g.node("Div", {"ln", "rt2"}, {"xs"});
        g.node("Erf", {"xs"}, {"er"});
        g.node("Add", {"er", "one"}, {"e1"});
        g.node("Mul", {"ln", "e1"}, {"xe"});
        g.node("Mul", {"xe", "half"}, {"gelu"});
        // L2: (sum |x|^2)^0.5
        g.node("Abs", {"gelu"}, {"ab"});
        g.node("Pow", {"ab", "two"}, {"sq"});
        g.node("ReduceSum", {"sq"}, {"ss"}, {attr_ints("axes", {-1})});
        g.node("Pow", {"ss", "half"}, {"l2"});
        // chunk(2)
        g.init(tensor_i("s0", {1}, {0}));
        g.init(tensor_i("s2", {1}, {2}));
        g.init(tensor_i("s4", {1}, {4}));
        g.init(tensor_i("ax", {1}, {-1}));
        g.node("Slice", {"gelu", "s0", "s2", "ax"}, {"x1"});
        g.node("Slice", {"gelu", "s2", "s4", "ax"}, {"x2"});
        g.node("Add", {"x1", "x2"}, {"output"});
        OnnxGraph og;
        CHECK(parse(g, "input", "output", &og));
        Reader rd(og);
        const Lns lns(rd);
        CHECK(lns.spelled.size() == 1 && lns.span.size() == 9);
        const OnnxNode *div = find_op(og, "Div");
        CHECK(div && lns.is_ln(*div) && lns.input(rd, div) == "input");
        LnW w;
        CHECK(lns.weights(rd, div, &w) && w.g == (std::vector<float>{1, 2, 3, 4}) && w.b == (std::vector<float>{0, 0, 1, 1}) && near(w.eps, 1e-6));
        const OnnxNode *mul_half = find_op(og, "Mul", 2), *mul_gam = find_op(og, "Mul", 0);
        CHECK(mul_half && mul_half->in[1] == "half" && gelu_const_mul(rd, *mul_half));
        CHECK(mul_gam && !gelu_const_mul(rd, *mul_gam));
        const OnnxNode *p2 = find_op(og, "Pow", 2), *p0 = find_op(og, "Pow", 0);
        CHECK(p2 && l2_root(rd, *p2) && p0 && !l2_root(rd, *p0));
        CHECK(chunk_slice(rd, "x2") && !chunk_slice(rd, "x1"));
    }

    // ---- the image range around a network, and a BatchNorm left after a conv
    {
        G g;
        g.init(tensor_v("mean", {1, 3, 1, 1}, {0.4f, 0.5f, 0.6f}));
        g.init(tensor_v("range", {}, {255}));
        g.node("Sub", {"input", "mean"}, {"xm"});
        g.node("Mul", {"xm", "range"}, {"xr"});
        g.init(tensor_v("w0", {3, 3, 1, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}));
        g.init(tensor_v("b0", {3}, {1, 2, 3}));
        g.node("Conv", {"xr", "w0", "b0"}, {"c0"}, {attr_ints("kernel_shape", {1, 1})});
        g.init(tensor_v("bg", {3}, {2, 2, 2}));
        g.init(tensor_v("bb", {3}, {0, 1, 0}));
        g.init(tensor_v("bm", {3}, {1, 1, 1}));
        g.init(tensor_v("bv", {3}, {3, 3, 3}));
        g.node("BatchNormalization", {"c0", "bg", "bb", "bm", "bv"}, {"bn"}, {attr_f("epsilon", 1.f)});
        g.node("Div", {"bn", "range"}, {"yd"});
        g.node("Add", {"yd", "mean"}, {"ya"});
        g.init(tensor_v("lo", {}, {0}));
        g.init(tensor_v("hi", {}, {1}));
        g.node("Clip", {"ya", "lo", "hi"}, {"output"});
        OnnxGraph og;
        CHECK(parse(g, "input", "output", &og));
        Reader rd(og);
        ImgRange ir;
        CHECK(img_range_head(rd, find_op(og, "Sub"), &ir));
        CHECK(near(ir.mean[1], 0.5) && ir.range == 255.f && ir.mul);
        // the BatchNorm: scale 2 / sqrt(3 + 1) = 1, so w stays, b = (b - 1) * 1 + beta
        std::vector<float> w = {1, 0, 0, 0, 1, 0, 0, 0, 1}, b = {1, 2, 3};
        CHECK(fold_bn_after(rd, find_op(og, "Conv"), &w, &b));
        CHECK(w[0] == 1 && w[4] == 1 && b == (std::vector<float>{0, 2, 2}));
        // the tail after the BatchNorm (as a parser sees it, BN folded): / 255, + mean, clipped
        Conv c;
        c.cout = 3;
        c.cin = 3;
        c.k = 1;
        c.w = w;
        c.b = b;
        CHECK(img_range_tail(rd, "bn", &c, &ir));
        CHECK(ir.clamp && near(c.w[0], 1.0 / 255) && near(c.b[1], 2.0 / 255 + 0.5));
        img_range_first(ir, &c);
        CHECK(near(c.w[0], 1.0));
    }

    // ---- a shifted-window mask a fixed-size export baked in, against a linear's bias
    {
        G g;
        g.init(tensor_v("mask", {1, 2, 1, 4, 4}, std::vector<float>(32, -100.f)));
        g.init(tensor_v("bias", {4}, {0, 0, 0, 0}));
        g.node("Add", {"input", "mask"}, {"m"});
        g.node("Add", {"m", "bias"}, {"output"});
        OnnxGraph og;
        CHECK(parse(g, "input", "output", &og));
        Reader rd(og);
        CHECK(attn_const(rd, *find_op(og, "Add", 0)) && !attn_const(rd, *find_op(og, "Add", 1)));
    }

    std::printf("native forms: ok\n");
    return 0;
}
