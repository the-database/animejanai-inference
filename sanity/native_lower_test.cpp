// native/lower: small ONNX graphs written here with a minimal protobuf encoder, in the shapes
// the supported families export (SPANF3, SRVGG Compact, span_s), lowered to native steps;
// plus graphs that must be refused. Checks the plan's structure, not numbers. CPU only.
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "native/lower.h"
#include "native/onnx_graph.h"
#include "onnx_pb.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

using namespace ajinative;
using namespace onnx_pb;

namespace {

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

// The Compact skip's Resize: its factor, sampling attributes (nullptr: the ONNX default), scales,
// the Cast types around it, and a Cast pair on the body's data path.
struct Skip {
    int r = 2;
    const char *ct = "asymmetric", *nm = nullptr;
    float scale = 0.f;  // 0: r
    int64_t cast_to = 1;
    int64_t body_cast = 0;  // a Cast to this type and back to fp16 after the first conv (0: none)
    enum Sizes { SCALES, SHAPE, SHAPE_ALL, CONST_BROADCAST, CONST_SIZES, SCALES_COMPUTED, SCALES_FOLDED, AXES, SIZES_BOOL, SIZES_FP16, SCALES_BOOL } sizes = SCALES;
};

// SRVGG Compact: conv+PReLU stack, depth-to-space r, + nearest-upsampled input
std::string compact(int c, int body, bool leaky, const Skip &sk = Skip())
{
    G g;
    auto act = [&](const std::string &x) {
        if (leaky) return g.op1("LeakyRelu", x, {attr_f("alpha", 0.1f)});
        const std::string s = g.fresh("slope");
        g.init(tensor(s, {c, 1, 1}));
        return g.op2("PRelu", x, s);
    };
    std::string x = act(g.conv("input", 3, c, 3));
    if (sk.body_cast) {
        const std::string t = g.op1("Cast", x, {attr_i("to", sk.body_cast)});
        x = g.op1("Cast", t, {attr_i("to", 10)});
    }
    for (int i = 0; i < body; i++) x = act(g.conv(x, c, c, 3));
    const std::string d = g.op1("DepthToSpace", g.conv(x, c, 3 * sk.r * sk.r, 3), {attr_i("blocksize", sk.r), attr_s("mode", "CRD")});
    const float s = sk.scale > 0.f ? sk.scale : (float)sk.r;
    g.node("Cast", {"input"}, {"in32"}, {attr_i("to", sk.cast_to)});
    std::vector<Pb> ra = {attr_s("mode", "nearest"), attr_s("coordinate_transformation_mode", sk.ct)};
    if (sk.nm) ra.push_back(attr_s("nearest_mode", sk.nm));
    std::vector<std::string> rin = {"in32", "", "scales"};
    switch (sk.sizes) {
    case Skip::SCALES:
        g.init(tensor_v("scales", {4}, {1.f, 1.f, s, s}));
        break;
    case Skip::SHAPE:
    case Skip::SHAPE_ALL: {
        // the legacy exporters' sizes: Concat(Shape[:2], Cast(Floor(Cast(Shape[2:]) * s)))
        g.node("Shape", {"input"}, {"shp"});
        g.init(tensor_i("i0", {1}, {0}));
        g.init(tensor_i("i2", {1}, {2}));
        g.init(tensor_i("i4", {1}, {4}));
        g.init(tensor_v("sf", {sk.sizes == Skip::SHAPE ? 1 : 4}, sk.sizes == Skip::SHAPE ? std::vector<float>{s} : std::vector<float>{s, s, s, s}));
        g.node("Slice", {"shp", "i0", "i2"}, {"nc"});
        g.node("Slice", {"shp", "i2", "i4"}, {"hw"});
        const std::string hwf = g.op1("Cast", sk.sizes == Skip::SHAPE ? "hw" : "shp", {attr_i("to", 1)});
        const std::string hws = g.op1("Cast", g.op1("Floor", g.op2("Mul", hwf, "sf")), {attr_i("to", 7)});
        if (sk.sizes == Skip::SHAPE) g.node("Concat", {"nc", hws}, {"sizes"}, {attr_i("axis", 0)});
        else g.node("Identity", {hws}, {"sizes"});  // N and C scaled too
        rin = {"in32", "", "", "sizes"};
        break;
    }
    case Skip::CONST_BROADCAST:  // [1, 3, 1, 1]: the Add broadcasts it (the review's reproducer)
    case Skip::CONST_SIZES:      // one fixed output size
        g.init(tensor_i("sizes", {4}, sk.sizes == Skip::CONST_BROADCAST ? std::vector<int64_t>{1, 3, 1, 1} : std::vector<int64_t>{1, 3, 32, 32}));
        rin = {"in32", "", "", "sizes"};
        break;
    case Skip::SCALES_COMPUTED:  // from the input's shape: not a constant
        g.node("Shape", {"input"}, {"shp"});
        g.node("Cast", {"shp"}, {"scales"}, {attr_i("to", 1)});
        break;
    case Skip::SCALES_FOLDED:  // a Concat of constants the exporter left unfolded
        g.init(tensor_v("ones", {2}, {1.f, 1.f}));
        g.init(tensor_v("rr", {2}, {s, s}));
        g.node("Concat", {"ones", "rr"}, {"scales"}, {attr_i("axis", 0)});
        break;
    case Skip::AXES:  // opset 18: scales for H and W alone
        g.init(tensor_v("scales", {2}, {s, s}));
        ra.push_back(attr_ints("axes", {2, 3}));
        break;
    case Skip::SIZES_BOOL:     // Cast<INT64>(Cast<BOOL>(Shape * [1, 1, r, r])): [1, 1, 1, 1] (the review's)
    case Skip::SIZES_FP16: {   // the same through fp16: not shape arithmetic the check evaluates
        g.node("Shape", {"input"}, {"shp"});
        g.init(tensor_i("factors", {4}, {1, 1, (int64_t)s, (int64_t)s}));
        const std::string m = g.op2("Mul", "shp", "factors");
        const std::string t = g.op1("Cast", m, {attr_i("to", sk.sizes == Skip::SIZES_BOOL ? 9 : 10)});
        g.node("Cast", {t}, {"sizes"}, {attr_i("to", 7)});
        rin = {"in32", "", "", "sizes"};
        break;
    }
    case Skip::SCALES_BOOL:  // Cast<FLOAT>(Cast<BOOL>([1, 1, r, r])) of constants: [1, 1, 1, 1]
        g.init(tensor_v("sc0", {4}, {1.f, 1.f, s, s}));
        g.node("Cast", {"sc0"}, {"scb"}, {attr_i("to", 9)});
        g.node("Cast", {"scb"}, {"scales"}, {attr_i("to", 1)});
        break;
    }
    g.node("Resize", rin, {"up32"}, ra);
    g.node("Cast", {"up32"}, {"up"}, {attr_i("to", 10)});
    g.node("Add", {d, "up"}, {"output"});
    return g.bytes("input", "output");
}

// span_s 4x: SiLU (x * sigmoid(x)), sigmoid(o) - 0.5 gates, conv_2, DCR depth-to-space; norm:
// SPAN's input normalization in front, (input - mean) * 255
std::string span_s(int c, int blocks, bool norm = false)
{
    G g;
    auto silu = [&](const std::string &x) { return g.op2("Mul", x, g.op1("Sigmoid", x)); };
    g.init(tensor("half", {}, 1, 0.5f));
    std::string in = "input";
    if (norm) {
        g.init(tensor("mean", {1, 3, 1, 1}, 1, 0.45f));
        g.init(tensor("range", {}, 1, 255.f));
        in = g.op2("Mul", g.op2("Sub", in, "mean"), "range");
    }
    const std::string feat = g.conv(in, 3, c, 3);
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
    // the skip's Resize: only the forms that sample by pixel replication (out x reads in x / r)
    {
        struct Form {
            int r;
            const char *ct, *nm;
            bool ok;
        };
        const Form forms[] = {
            {2, "asymmetric", nullptr, true},  // the default nearest_mode, round_prefer_floor: replication at 2x
            {2, "asymmetric", "floor", true},  // the torch exporters'
            {4, "asymmetric", "floor", true},
            {4, "asymmetric", nullptr, false},  // round_prefer_floor at 4x: x = 4i + 3 reads i + 1
            {2, "asymmetric", "ceil", false},
            {2, "asymmetric", "round_prefer_ceil", false},
            {2, "half_pixel", nullptr, true},
            {4, "half_pixel", "round_prefer_ceil", true},
            {3, "pytorch_half_pixel", nullptr, true},
            {2, "half_pixel", "floor", false},  // x = 2i reads i - 1
            {2, "half_pixel", "ceil", false},
            {2, "align_corners", nullptr, false},
            {2, "tf_half_pixel_for_nn", nullptr, false},
        };
        for (const Form &f : forms) {
            Skip sk;
            sk.r = f.r;
            sk.ct = f.ct;
            sk.nm = f.nm;
            const bool ok = lower_bytes(compact(16, 2, true, sk), &p, &why);
            if (ok != f.ok) std::fprintf(stderr, "Resize %dx %s / %s: %s\n", f.r, f.ct, f.nm ? f.nm : "default", ok ? "lowered" : why.c_str());
            CHECK(ok == f.ok);
            if (ok) CHECK(p.depth_to_space == f.r && p.steps.back().resid == kModelInput);
        }
        Skip sk;
        sk.scale = 2.5f;  // scales other than the depth-to-space factor
        CHECK(!lower_bytes(compact(16, 2, true, sk), &p, &why) && why.find("scale") != std::string::npos);
        // scales or sizes: only those that are r on H and W at every input size
        struct Sz {
            Skip::Sizes s;
            int r;
            bool ok;
        };
        const Sz szs[] = {
            {Skip::SHAPE, 2, true}, {Skip::SHAPE, 4, true}, {Skip::SHAPE_ALL, 2, false}, {Skip::CONST_BROADCAST, 2, false},
            {Skip::CONST_SIZES, 2, false}, {Skip::SCALES_COMPUTED, 2, false}, {Skip::SCALES_FOLDED, 2, true}, {Skip::AXES, 2, true},
            {Skip::SIZES_BOOL, 2, false}, {Skip::SIZES_FP16, 2, false}, {Skip::SCALES_BOOL, 2, false},
        };
        for (const Sz &z : szs) {
            sk = Skip();
            sk.sizes = z.s;
            sk.r = z.r;
            sk.nm = "floor";
            const bool ok = lower_bytes(compact(16, 2, true, sk), &p, &why);
            if (ok != z.ok) std::fprintf(stderr, "Resize sizes form %d at %dx: %s\n", (int)z.s, z.r, ok ? "lowered" : why.c_str());
            CHECK(ok == z.ok);
            if (!ok) CHECK(why.find("scale") != std::string::npos);
            if (ok) CHECK(p.depth_to_space == z.r && p.steps.back().resid == kModelInput);
        }
        sk = Skip();
        sk.sizes = Skip::SHAPE;
        sk.scale = 3.f;  // sizes 3x the input under a 2x depth-to-space
        sk.nm = "floor";
        CHECK(!lower_bytes(compact(16, 2, true, sk), &p, &why));
    }
    // Casts on the data path: float types pass through, others change the values (refused)
    for (int64_t to : {1, 10, 11, 16, 6, 7, 9, 2}) {
        Skip sk;
        sk.body_cast = to;
        const bool fl = to == 1 || to == 10 || to == 11 || to == 16;
        CHECK(lower_bytes(compact(16, 2, true, sk), &p, &why) == fl);
        if (!fl) CHECK(why.find("Cast") != std::string::npos);
        sk = Skip();
        sk.cast_to = to;  // around the skip's Resize
        CHECK(lower_bytes(compact(16, 2, true, sk), &p, &why) == fl);
    }

    CHECK(lower_bytes(span_s(48, 6), &p, &why));
    // head, 6 x (c1 plain copy for the last block only, c1+silu, c2+silu, c3 gate), conv_2, concat, tail
    CHECK(p.steps.size() == 1 + 6 * 3 + 1 + 1 + 1 + 1);
    CHECK(count(p, 3, 0) == 6 && count(p, 0, 3) == 12);
    CHECK(p.depth_to_space == 4 && p.steps.back().out == 2);
    CHECK(!p.in_norm && !p.steps[0].in_norm);
    // the input normalization: off the graph, onto the step reading the input
    CHECK(lower_bytes(span_s(48, 6, true), &p, &why));
    CHECK(p.in_norm && p.steps[0].in_norm && p.steps[0].src == kModelInput);
    CHECK(std::fabs(p.steps[0].in_mean[1] - 0.45f) < 1e-6f && p.steps[0].in_scale[2] == 255.f);
    CHECK(!p.steps[1].in_norm);
    const std::string span_desc = describe(p);

    // SuperUltraCompact's 24 channels: widened to the 32-channel kernels with zero weights
    {
        CHECK(lower_bytes(compact(24, 8, false), &p, &why));
        CHECK(p.model_channels.size() == p.channels.size());
        for (size_t i = 0; i < p.channels.size(); i++) CHECK(p.model_channels[i] == 24 && p.channels[i] == 32);
        const PlanStep &h = p.steps[0], &b = p.steps[1], &t = p.steps.back();
        CHECK(h.cin == 3 && h.cout == 32 && h.w.cout == 32 && h.w.bias.size() == 32 && h.w.slope.size() == 32);
        CHECK(b.cin == 32 && b.cout == 32 && b.w.w.size() == (size_t)9 * 32 * 32);
        CHECK(t.cin == 32 && t.cout == 12 && t.dst == kModelOutput);
        bool zero = true;  // the added inputs and outputs carry zero weights, bias and slope
        for (int tap = 0; tap < 9; tap++)
            for (int c = 0; c < 32; c++)
                for (int o = 0; o < 32; o++)
                    if ((c >= 24 || o >= 24) && b.w.w[((size_t)tap * 32 + c) * 32 + o] != 0) zero = false;
        for (int o = 24; o < 32; o++) zero = zero && b.w.bias[o] == 0.f && b.w.slope[o] == 0.f;
        for (int tap = 0; tap < 9; tap++)
            for (int c = 24; c < 32; c++)
                for (int o = 0; o < 12; o++) zero = zero && t.w.w[((size_t)tap * 32 + c) * 12 + o] == 0;
        CHECK(zero);
        CHECK(describe(p).find("24 ch padded to 32") != std::string::npos);
        CHECK(lower_bytes(compact(64, 2, false), &p, &why) && p.model_channels.empty());  // a kernel width: as it is
        // CUDA has 24-channel kernels: kept as it is
        OnnxGraph g;
        const std::string m = compact(24, 8, false);
        CHECK(parse_onnx(m.data(), m.size(), &g, &why) && lower(g, &p, &why, kernel_widths_cuda()));
        CHECK(p.model_channels.empty() && p.channels[0] == 24 && p.steps[1].cin == 24 && p.steps[1].w.w.size() == (size_t)9 * 24 * 24);
    }

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
        G g;  // ends without depth-to-space, not in RGB
        const std::string y = g.conv(g.op1("Relu", g.conv("input", 3, 48, 3)), 48, 6, 3);
        CHECK(!lower_bytes(g.bytes("input", y), &p, &why) && !why.empty());
    }

    // output forms: 1x (the last conv's 3 channels are the output), 3x, a final Clip(0, 1)
    {
        G g;
        const std::string y = g.conv(g.op1("Relu", g.conv("input", 3, 48, 3)), 48, 3, 3);
        CHECK(lower_bytes(g.bytes("input", y), &p, &why));
        CHECK(p.depth_to_space == 1 && p.steps.back().out == 3 && p.steps.back().dst == kModelOutput && !p.out_clamp);
    }
    for (bool clip : {false, true}) {
        G g;
        const std::string t = g.conv(g.op1("Relu", g.conv("input", 3, 48, 3)), 48, 27, 3);
        std::string y = g.fresh("d2s");
        g.node("DepthToSpace", {t}, {y}, {attr_i("blocksize", 3)});
        if (clip) {
            g.init(tensor("lo", {}, 1, 0.f));
            g.init(tensor("hi", {}, 1, 1.f));
            const std::string c = g.fresh("clip");
            g.node("Clip", {y, "lo", "hi"}, {c});
            y = c;
        }
        CHECK(lower_bytes(g.bytes("input", y), &p, &why));
        CHECK(p.depth_to_space == 3 && p.steps.back().out == 4 && p.steps.back().out_clamp == clip && p.out_clamp == clip);
    }
    {
        G g;  // Clip to another range: refused
        const std::string t = g.conv(g.conv("input", 3, 48, 3), 48, 12, 3);
        const std::string d = g.fresh("d2s");
        g.node("DepthToSpace", {t}, {d}, {attr_i("blocksize", 2)});
        g.init(tensor("lo2", {}, 1, -1.f));
        g.init(tensor("hi2", {}, 1, 1.f));
        g.node("Clip", {d, "lo2", "hi2"}, {"y"});
        CHECK(!lower_bytes(g.bytes("input", "y"), &p, &why) && !why.empty());
    }
    {
        std::string bad = "not a model";
        OnnxGraph g;
        CHECK(!parse_onnx(bad.data(), bad.size(), &g, &why));
    }
    std::printf("native lower: OK (span_s: %s)\n", span_desc.c_str());
    return 0;
}
