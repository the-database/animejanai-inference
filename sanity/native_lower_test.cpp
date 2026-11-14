// native/lower: small ONNX graphs written here with a minimal protobuf encoder, in the shapes
// the supported families export (SPANF3, SRVGG Compact, span_s), lowered to native steps;
// plus graphs that must be refused. Checks the plan's structure, not numbers. CPU only.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "native/lower.h"
#include "native/onnx_graph.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

using namespace ajinative;

namespace {

// ---- protobuf encoding
struct Pb {
    std::string b;
    Pb &varint(uint64_t v)
    {
        while (v >= 0x80) { b.push_back((char)(v | 0x80)); v >>= 7; }
        b.push_back((char)v);
        return *this;
    }
    Pb &tag(int f, int wire) { return varint((uint64_t)f << 3 | wire); }
    Pb &bytes(int f, const std::string &s) { tag(f, 2).varint(s.size()); b += s; return *this; }
    Pb &msg(int f, const Pb &m) { return bytes(f, m.b); }
    Pb &i(int f, int64_t v) { tag(f, 0).varint((uint64_t)v); return *this; }
    Pb &f32(int f, float v) { tag(f, 5); char c[4]; std::memcpy(c, &v, 4); b.append(c, 4); return *this; }
};

Pb tensor(const std::string &name, std::vector<int64_t> dims, int dtype = 10, float fill = 0.f)
{
    Pb t;
    size_t n = 1;
    for (int64_t d : dims) { t.i(1, d); n *= (size_t)d; }
    t.i(2, dtype).bytes(8, name);
    std::string raw;
    if (dtype == 10) raw.assign(n * 2, '\0');
    if (dtype == 1) for (size_t k = 0; k < n; k++) { char c[4]; std::memcpy(c, &fill, 4); raw.append(c, 4); }
    if (dtype == 7) raw.assign(n * 8, '\0');
    return t.bytes(9, raw);
}
Pb attr_i(const char *name, int64_t v) { return Pb().bytes(1, name).i(3, v).i(20, 2); }
Pb attr_s(const char *name, const char *v) { return Pb().bytes(1, name).bytes(4, v).i(20, 3); }
Pb attr_ints(const char *name, std::vector<int64_t> v)
{
    Pb a;
    a.bytes(1, name);
    for (int64_t x : v) a.i(8, x);
    return a.i(20, 7);
}
Pb attr_f(const char *name, float v) { return Pb().bytes(1, name).f32(2, v).i(20, 1); }

struct G {
    Pb graph;
    int n = 0;
    std::string fresh(const char *p) { return std::string(p) + std::to_string(n++); }
    void node(const char *op, std::vector<std::string> in, std::vector<std::string> out, std::vector<Pb> attrs = {})
    {
        Pb nd;
        for (auto &s : in) nd.bytes(1, s);
        for (auto &s : out) nd.bytes(2, s);
        nd.bytes(3, fresh(op)).bytes(4, op);
        for (auto &a : attrs) nd.msg(5, a);
        graph.msg(1, nd);
    }
    void init(const Pb &t) { graph.msg(5, t); }
    // conv with fresh weights; k = 3 or 1
    std::string conv(const std::string &x, int ci, int co, int k)
    {
        const std::string w = fresh("w"), b = fresh("b"), y = fresh("conv");
        init(tensor(w, {co, ci, k, k}));
        init(tensor(b, {co}));
        node("Conv", {x, w, b}, {y}, {attr_ints("kernel_shape", {k, k}), attr_ints("pads", {k / 2, k / 2, k / 2, k / 2}),
                                      attr_ints("strides", {1, 1})});
        return y;
    }
    std::string op1(const char *op, const std::string &x, std::vector<Pb> attrs = {})
    {
        const std::string y = fresh(op);
        node(op, {x}, {y}, attrs);
        return y;
    }
    std::string op2(const char *op, const std::string &a, const std::string &b)
    {
        const std::string y = fresh(op);
        node(op, {a, b}, {y});
        return y;
    }
    std::string bytes(const std::string &in, const std::string &out)
    {
        graph.msg(11, Pb().bytes(1, in));
        graph.msg(12, Pb().bytes(1, out));
        return Pb().msg(7, graph).b;
    }
};

bool lower_bytes(const std::string &model, Plan *p, std::string *why)
{
    OnnxGraph g;
    if (!parse_onnx(model.data(), model.size(), &g, why)) return false;
    return lower(g, p, why);
}

// SPANF3 2x with unshuffle, `blocks` blocks of c channels, exported the way PyTorch does it
std::string spanf3(int c, int blocks, bool space_to_depth)
{
    G g;
    // the shape plumbing for the alignment pad and output crop: Shape -> Gather -> ... (not data)
    g.node("Shape", {"input"}, {"shp"});
    g.init(tensor("idx", {1}, 7));
    g.node("Gather", {"shp", "idx"}, {"hdim"});
    g.node("Pad", {"input", "hdim"}, {"padded"}, {attr_s("mode", "reflect")});
    std::string u;
    if (space_to_depth) {
        u = g.op1("SpaceToDepth", "padded", {attr_i("blocksize", 2)});
    } else {
        g.node("Reshape", {"padded", "shp"}, {"r1"});
        g.node("Transpose", {"r1"}, {"tr"}, {attr_ints("perm", {0, 1, 3, 5, 2, 4})});
        g.node("Reshape", {"tr", "shp"}, {"unsh"});
        u = "unsh";
    }
    const std::string feat = g.conv(u, 12, c, 3);
    std::string x = feat, out_b1, mid;
    for (int i = 0; i < blocks; i++) {
        const std::string o1 = g.conv(x, c, c, 3), a1 = g.op1("Relu", o1);
        const std::string a2 = g.op1("Relu", g.conv(a1, c, c, 3));
        const std::string o3 = g.conv(a2, c, c, 3);
        const std::string add = g.op2("Add", x, o3);
        const std::string att = g.op1("Tanh", g.conv(o3, c, c, 1));
        x = g.op2("Mul", add, att);
        if (i == 0) out_b1 = x;
        mid = a1;
    }
    const std::string cat = g.fresh("cat");
    g.node("Concat", {feat, x, out_b1, mid}, {cat}, {attr_i("axis", 1)});
    const std::string y = g.conv(g.conv(cat, 4 * c, c, 1), c, 48, 3);
    const std::string d = g.op1("DepthToSpace", y, {attr_i("blocksize", 4), attr_s("mode", "CRD")});
    g.node("Slice", {d, "zero", "hdim"}, {"output"});
    g.init(tensor("zero", {1}, 7));
    return g.bytes("input", "output");
}

// SRVGG Compact 2x: conv+PReLU stack, depth-to-space 2, + nearest-upsampled input
std::string compact(int c, int body, bool leaky)
{
    G g;
    auto act = [&](const std::string &x) {
        if (leaky) return g.op1("LeakyRelu", x, {attr_f("alpha", 0.1f)});
        const std::string s = g.fresh("slope");
        g.init(tensor(s, {c, 1, 1}));
        return g.op2("PRelu", x, s);
    };
    std::string x = act(g.conv("input", 3, c, 3));
    for (int i = 0; i < body; i++) x = act(g.conv(x, c, c, 3));
    const std::string d = g.op1("DepthToSpace", g.conv(x, c, 12, 3), {attr_i("blocksize", 2), attr_s("mode", "CRD")});
    g.init(tensor("scales", {4}, 1, 2.f));
    g.node("Cast", {"input"}, {"in32"}, {attr_i("to", 1)});
    g.node("Resize", {"in32", "", "scales"}, {"up32"}, {attr_s("mode", "nearest"), attr_s("coordinate_transformation_mode", "asymmetric")});
    g.node("Cast", {"up32"}, {"up"}, {attr_i("to", 10)});
    g.node("Add", {d, "up"}, {"output"});
    return g.bytes("input", "output");
}

// span_s 4x: SiLU (x * sigmoid(x)), sigmoid(o) - 0.5 gates, conv_2, DCR depth-to-space
std::string span_s(int c, int blocks)
{
    G g;
    auto silu = [&](const std::string &x) { return g.op2("Mul", x, g.op1("Sigmoid", x)); };
    g.init(tensor("half", {}, 1, 0.5f));
    const std::string feat = g.conv("input", 3, c, 3);
    std::string x = feat, out_b1, mid;
    for (int i = 0; i < blocks; i++) {
        const std::string o1 = g.conv(x, c, c, 3), a1 = silu(o1);
        const std::string o3 = g.conv(silu(g.conv(a1, c, c, 3)), c, c, 3);
        const std::string m = g.op2("Sub", g.op1("Sigmoid", o3), "half");
        x = g.op2("Mul", g.op2("Add", o3, x), m);
        if (i == 0) out_b1 = x;
        mid = o1;  // pre-activation: lowered as a second, plain conv
    }
    const std::string o2 = g.conv(x, c, c, 3);
    const std::string cat = g.fresh("cat");
    g.node("Concat", {feat, o2, out_b1, mid}, {cat}, {attr_i("axis", 1)});
    const std::string y = g.conv(g.conv(cat, 4 * c, c, 1), c, 48, 3);
    g.node("DepthToSpace", {y}, {"output"}, {attr_i("blocksize", 4)});  // default mode DCR
    return g.bytes("input", "output");
}

int count(const Plan &p, int gate, int act)
{
    int n = 0;
    for (const PlanStep &s : p.steps) n += s.kind == PlanStep::CONV && s.gate == gate && s.act == act;
    return n;
}

}  // namespace

int main()
{
    Plan p;
    std::string why;

    for (bool s2d : {false, true}) {
        CHECK(lower_bytes(spanf3(48, 5, s2d), &p, &why));
        CHECK(p.steps.size() == 18);                 // head + 5 x 3 + concat + tail
        CHECK(p.unshuffle == 2 && p.depth_to_space == 4 && p.aligned_only);
        CHECK(p.steps[0].in == 2 && p.steps[0].src == kModelInput && p.steps[0].cin == 12);
        CHECK(count(p, 2, 0) == 5 && count(p, 0, 1) == 10);   // tanh gates, ReLU convs
        CHECK(p.steps[16].kind == PlanStep::CONCAT && p.steps[16].nsrc == 4 && p.steps[16].c == 48);
        CHECK(p.steps[17].out == 2 && p.steps[17].dst == kModelOutput && p.steps[17].resid == -1);
        // concat sources: head output, last block, first block, relu(c1) of the last block
        CHECK(p.steps[16].srcs[0] == p.steps[0].dst && p.steps[16].srcs[1] == p.steps[15].dst &&
              p.steps[16].srcs[2] == p.steps[3].dst && p.steps[16].srcs[3] == p.steps[13].dst);
        CHECK(p.steps[3].resid == p.steps[0].dst);   // block 1's residual is the head output
    }
    CHECK(lower_bytes(spanf3(64, 8, false), &p, &why) && p.steps.size() == 27);

    for (bool leaky : {false, true}) {
        CHECK(lower_bytes(compact(64, 16, leaky), &p, &why));
        CHECK(p.steps.size() == 18 && p.unshuffle == 1 && p.depth_to_space == 2 && !p.aligned_only);
        CHECK(p.steps[0].in == 1 && p.steps[0].act == 2 && !p.steps[0].w.slope.empty());
        CHECK(p.steps.back().out == 1 && p.steps.back().resid == kModelInput);
        if (leaky) CHECK(p.steps[5].w.slope[3] == 0.1f);
    }

    CHECK(lower_bytes(span_s(48, 6), &p, &why));
    // head, 6 x (c1 plain copy for the last block only, c1+silu, c2+silu, c3 gate), conv_2, concat, tail
    CHECK(p.steps.size() == 1 + 6 * 3 + 1 + 1 + 1 + 1);
    CHECK(count(p, 3, 0) == 6 && count(p, 0, 3) == 12);
    CHECK(p.depth_to_space == 4 && p.steps.back().out == 2);
    const std::string span_desc = describe(p);

    // refusals, with a reason
    {
        G g;
        const std::string y = g.op1("Softmax", g.conv("input", 3, 48, 3));
        CHECK(!lower_bytes(g.bytes("input", y), &p, &why) && !why.empty());
    }
    {
        G g;  // 5x5 conv
        g.init(tensor("w", {48, 3, 5, 5}));
        g.node("Conv", {"input", "w"}, {"y"}, {attr_ints("pads", {2, 2, 2, 2})});
        CHECK(!lower_bytes(g.bytes("input", "y"), &p, &why) && why.find("3x3") != std::string::npos);
    }
    {
        G g;  // ends without depth-to-space
        const std::string y = g.conv(g.op1("Relu", g.conv("input", 3, 48, 3)), 48, 3, 3);
        CHECK(!lower_bytes(g.bytes("input", y), &p, &why) && !why.empty());
    }
    {
        std::string bad = "not a model";
        OnnxGraph g;
        CHECK(!parse_onnx(bad.data(), bad.size(), &g, &why));
    }
    std::printf("native lower: OK (span_s: %s)\n", span_desc.c_str());
    return 0;
}
