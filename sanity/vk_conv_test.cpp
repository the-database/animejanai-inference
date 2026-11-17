// aji_vk_conv_test: every entry of the native kernel table (the CUDA kConv / kConcat shapes,
// native/conv.cu) on aji_vk's kernels, against the fp32 CPU reference (sanity/native_ref.h).
// For each entry it builds a small plan around it with random weights:
//   - a head (NCHW or pixel-unshuffled input, each activation) feeding a depth-to-space 2 tail;
//   - a body conv (each activation and gate, the tanh gate with its 1x1 attention conv) between
//     a head and a tail;
//   - a depth-to-space 4 tail, with and without the model input added;
//   - a concat of 2..4 feature maps;
// at 32, 48 and 64 channels, runs it at a ragged size with each tile variant, and compares.
// Fails above --max-levels (default 2, in 8-bit levels) or below --min-db (default 50).
//   aji_vk_conv_test [--device N] [--size WxH]   (AJI_VK_FAMILY picks the kernel family)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "native/lower.h"
#include "native/pack.h"
#include "native_ref.h"
#include "vk/device.h"
#include "vk/executor.h"

using namespace ajinative;

namespace {

uint32_t g_seed = 0x1234567u;
float urand(float lo, float hi)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((g_seed >> 8) & 0xffffff) / 16777215.f;
}

PlanWeights weights(int taps, int cin, int cout, bool slope)
{
    PlanWeights w;
    w.taps = taps;
    w.cin = cin;
    w.cout = cout;
    const float a = 1.2f / std::sqrt((float)taps * cin);
    w.w.resize((size_t)taps * cin * cout);
    for (auto &v : w.w) v = float_to_half_bits(urand(-a, a));
    w.bias.resize(cout);
    for (auto &b : w.bias) b = urand(-0.05f, 0.05f);
    if (slope) {
        w.slope.resize(cout);
        for (auto &s : w.slope) s = urand(0.05f, 0.3f);
    }
    return w;
}

PlanStep conv(int in, int act, int gate, int out, int cin, int cout, int src, int resid, int dst)
{
    PlanStep s;
    s.kind = PlanStep::CONV;
    s.in = in;
    s.act = act;
    s.gate = gate;
    s.out = out;
    s.cin = cin;
    s.cout = cout;
    s.w = weights(9, cin, cout, act == (int)Act::PRelu);
    if (gate == (int)Gate::Tanh1x1) s.att = weights(1, cout, cout, false);
    s.src = src;
    s.resid = resid;
    s.dst = dst;
    return s;
}

// head: model input -> feature 0 (C channels)
void add_head(Plan &p, int C, int in, int act)
{
    p.unshuffle = in == (int)Input::NCHW3Unshuffle2 ? 2 : 1;
    p.channels.push_back(C);
    p.steps.push_back(conv(in, act, 0, 0, in == (int)Input::NCHW3Unshuffle2 ? 12 : 3, C, kModelInput, -1, 0));
}

// tail: feature src -> model output, depth-to-space r
void add_tail(Plan &p, int src, int C, int r, bool add_input)
{
    p.depth_to_space = r;
    p.steps.push_back(conv(0, 0, 0, r == 2 ? (int)Output::DepthToSpace2 : (int)Output::DepthToSpace4, C, 3 * r * r, src,
                           add_input ? kModelInput : -1, -1));
    p.steps.back().dst = kModelOutput;
}

int feature(Plan &p, int C)
{
    p.channels.push_back(C);
    return (int)p.channels.size() - 1;
}

std::vector<float> synth(int w, int h)
{
    std::vector<float> v((size_t)3 * w * h);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                float f = 0.5f + 0.35f * std::sin(0.11f * x + 1.3f * c) * std::cos(0.07f * y - 0.7f * c);
                if (((x / 7) + (y / 5)) % 2) f = 0.8f * f + 0.1f;
                v[((size_t)c * h + y) * w + x] = f + urand(-0.03f, 0.03f);
            }
    return v;
}

void set_variant(int v)
{
    const std::string s = v < 0 ? "" : std::to_string(v);
#ifdef _WIN32
    _putenv_s("AJI_VK_VARIANT", s.c_str());
#else
    if (v < 0) unsetenv("AJI_VK_VARIANT");
    else setenv("AJI_VK_VARIANT", s.c_str(), 1);
#endif
}

}  // namespace

int main(int argc, char **argv)
{
    int device = -1, W = 61, H = 37;
    double max_levels = 2.0, min_db = 50.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--device")) device = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--size")) std::sscanf(argv[i + 1], "%dx%d", &W, &H);
        else if (!std::strcmp(argv[i], "--max-levels")) max_levels = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--min-db")) min_db = std::atof(argv[i + 1]);
    }
    std::string why;
    auto dev = ajivk::Device::create(device, nullptr, &why);
    if (!dev) {
        // no Vulkan GPU with a kernel family here: nothing to test
        std::printf("aji_vk_conv_test: skipped (%s)\n", why.c_str());
        return 0;
    }
    std::printf("device %s, family %s, %dx%d\n", dev->info().name.c_str(), ajivk::family_name(dev->family()), W, H);

    struct Case {
        std::string name;
        Plan plan;
    };
    std::vector<Case> cases;
    for (int C : {32, 48, 64}) {
        const std::string c = std::to_string(C);
        for (int in : {1, 2})
            for (int act : {0, 1, 2}) {
                Plan p;
                add_head(p, C, in, act);
                add_tail(p, 0, C, 2, false);
                cases.push_back({"head c" + c + " in" + std::to_string(in) + " act" + std::to_string(act), p});
            }
        const int body[7][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {0, 2}, {0, 3}};
        for (const auto &b : body) {
            Plan p;
            add_head(p, C, 1, 0);
            const int f1 = feature(p, C);
            p.steps.push_back(conv(0, b[0], b[1], 0, C, C, 0, b[1] ? 0 : -1, f1));
            add_tail(p, f1, C, 2, false);
            cases.push_back({"body c" + c + " act" + std::to_string(b[0]) + " gate" + std::to_string(b[1]), p});
        }
        for (bool add : {false, true}) {
            Plan p;
            add_head(p, C, 1, 1);
            add_tail(p, 0, C, 4, add);
            cases.push_back({"tail4 c" + c + (add ? " +input" : ""), p});
        }
        {
            Plan p;
            add_head(p, C, 1, 1);
            add_tail(p, 0, C, 2, true);
            cases.push_back({"tail2 c" + c + " +input", p});
        }
        for (int n = 2; n <= 4; n++) {
            Plan p;
            add_head(p, C, 1, 1);
            std::vector<int> srcs = {0};
            for (int k = 1; k < n; k++) {
                const int f = feature(p, C);
                p.steps.push_back(conv(0, 1, 0, 0, C, C, srcs.back(), -1, f));
                srcs.push_back(f);
            }
            const int dst = feature(p, C);
            PlanStep s;
            s.kind = PlanStep::CONCAT;
            s.c = C;
            s.nsrc = n;
            for (int k = 0; k < n; k++) s.srcs[k] = srcs[k];
            s.w = weights(1, n * C, C, false);
            s.dst = dst;
            p.steps.push_back(s);
            add_tail(p, dst, C, 2, false);
            cases.push_back({"concat c" + c + " n" + std::to_string(n), p});
        }
    }

    int bad = 0, ran = 0;
    for (const Case &cs : cases) {
        const int u = cs.plan.unshuffle, w = W / u * u, h = H / u * u;
        const std::vector<float> in = synth(w, h);
        const std::vector<float> ref = ajiref::run_plan(cs.plan, in, w, h);
        std::vector<uint16_t> hin(in.size());
        for (size_t i = 0; i < in.size(); i++) hin[i] = float_to_half_bits(in[i]);
        for (int v = 0; v < ajivk::kConvVariants; v++) {
            set_variant(v);
            auto ex = ajivk::Executor::create(*dev, cs.plan, w, h, &why);
            if (!ex) {
                std::printf("%-26s v%d: FAIL (%s)\n", cs.name.c_str(), v, why.c_str());
                bad++;
                continue;
            }
            std::vector<uint16_t> out((size_t)3 * ex->out_w() * ex->out_h());
            if (!dev->upload(ex->input(), hin.data(), hin.size() * 2, &why) || !ex->run(w, h, &why) ||
                !dev->download(ex->output(), out.data(), out.size() * 2, &why) || out.size() != ref.size()) {
                std::printf("%-26s v%d: FAIL (%s)\n", cs.name.c_str(), v, why.c_str());
                bad++;
                continue;
            }
            double se = 0, maxd = 0;
            size_t nan = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                const float x = half_bits_to_float(out[i]);
                if (!std::isfinite(x)) {
                    nan++;
                    continue;
                }
                const double d = std::fabs((double)x - ref[i]);
                se += d * d;
                maxd = std::max(maxd, d);
            }
            const double mse = se / ref.size(), db = mse > 0 ? 10.0 * std::log10(1.0 / mse) : 200.0;
            const bool ok = nan == 0 && db >= min_db && maxd * 255 <= max_levels;
            std::printf("%-26s v%d: %6.1f dB, max %.2f levels%s\n", cs.name.c_str(), v, db, maxd * 255, ok ? "" : "  FAIL");
            bad += !ok;
            ran++;
        }
    }
    set_variant(-1);
    std::printf("vk conv: %d runs, %d failed\n", ran, bad);
    return bad ? 1 : 0;
}
