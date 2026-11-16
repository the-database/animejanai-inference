/*
 * sanity/native_ref.h — a CPU reference for a lowered model (native/lower.h): every step in fp32
 * with the kernels' semantics (native/conv.h), each feature map rounded to fp16 as the kernels
 * store it. Slow (a few GFLOP/s across threads): for small sizes in tests. Header-only.
 */
#ifndef AJI_SANITY_NATIVE_REF_H
#define AJI_SANITY_NATIVE_REF_H

#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "native/lower.h"
#include "native/pack.h"

namespace ajiref {

using namespace ajinative;

inline float f16r(float x) { return half_bits_to_float(float_to_half_bits(x)); }

// in: NCHW [3][h][w] floats; returns NCHW [3][h*R/u][w*R/u].
inline std::vector<float> run_plan(const Plan &plan, const std::vector<float> &in, int w, int h)
{
    const int u = plan.unshuffle, W = w / u, H = h / u;
    std::vector<std::vector<float>> feat(plan.channels.size());
    const int R = plan.depth_to_space, OW = W * R, OH = H * R;
    std::vector<float> out((size_t)3 * OW * OH, 0.f);
    const int nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
    auto par_rows = [&](int rows, auto fn) {
        std::vector<std::thread> th;
        for (int t = 0; t < nthreads; t++)
            th.emplace_back([&, t] {
                for (int y = t; y < rows; y += nthreads) fn(y);
            });
        for (auto &x : th) x.join();
    };
    for (const PlanStep &s : plan.steps) {
        if (s.kind == PlanStep::CONCAT) {
            std::vector<float> wf(s.w.w.size());
            for (size_t i = 0; i < wf.size(); i++) wf[i] = half_bits_to_float(s.w.w[i]);
            std::vector<float> &dst = feat[s.dst];
            dst.assign((size_t)W * H * s.c, 0.f);
            par_rows(H, [&](int y) {
                for (int x = 0; x < W; x++) {
                    const size_t p = (size_t)y * W + x;
                    for (int j = 0; j < s.c; j++) {
                        double acc = s.w.bias[j];
                        for (int q = 0; q < s.nsrc; q++) {
                            const float *f = &feat[s.srcs[q]][p * s.c];
                            for (int k = 0; k < s.c; k++) acc += (double)f[k] * wf[(size_t)(q * s.c + k) * s.c + j];
                        }
                        dst[p * s.c + j] = f16r((float)acc);
                    }
                }
            });
            continue;
        }
        const int cin = s.cin, cout = s.cout;
        std::vector<float> wf(s.w.w.size()), af(s.att.w.size());
        for (size_t i = 0; i < wf.size(); i++) wf[i] = half_bits_to_float(s.w.w[i]);
        for (size_t i = 0; i < af.size(); i++) af[i] = half_bits_to_float(s.att.w[i]);
        auto input = [&](int y, int x, int c) -> float {
            if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;
            if (s.in == (int)Input::NHWC) return feat[s.src][((size_t)y * W + x) * cin + c];
            if (s.in == (int)Input::NCHW3) return in[((size_t)c * h + y) * w + x];
            const int cc = c / 4, ry = (c / 2) % 2, rx = c % 2;  // unshuffle: c*4 + ry*2 + rx
            return in[((size_t)cc * h + 2 * y + ry) * w + 2 * x + rx];
        };
        std::vector<float> nhwc;
        if (s.out == (int)Output::NHWC) nhwc.assign((size_t)W * H * cout, 0.f);
        const int r = s.out == (int)Output::DepthToSpace2 ? 2 : 4;
        par_rows(H, [&](int y) {
            std::vector<double> acc(cout);
            std::vector<float> o(cout), ga(cout);
            for (int x = 0; x < W; x++) {
                for (int j = 0; j < cout; j++) acc[j] = s.w.bias[j];
                for (int t = 0; t < 9; t++)
                    for (int c = 0; c < cin; c++) {
                        const float v = input(y + t / 3 - 1, x + t % 3 - 1, c);
                        if (v == 0.f) continue;
                        const float *wr = &wf[((size_t)t * cin + c) * cout];
                        for (int j = 0; j < cout; j++) acc[j] += (double)v * wr[j];
                    }
                for (int j = 0; j < cout; j++) {
                    float v = (float)acc[j];
                    if (s.act == (int)Act::Relu) v = std::fmax(v, 0.f);
                    if (s.act == (int)Act::PRelu) v = v >= 0 ? v : v * s.w.slope[j];
                    if (s.act == (int)Act::SiLU) v = v / (1.f + std::exp(-v));
                    o[j] = v;
                }
                if (s.gate == (int)Gate::Tanh1x1)
                    for (int j = 0; j < cout; j++) {
                        double a = s.att.bias[j];
                        for (int k = 0; k < cout; k++) a += (double)f16r(o[k]) * af[(size_t)k * cout + j];
                        ga[j] = (float)a;
                    }
                const size_t p = (size_t)y * W + x;
                for (int j = 0; j < cout; j++) {
                    float v = o[j];
                    if (s.gate != (int)Gate::None) {
                        const float xr = feat[s.resid][p * cout + j];
                        if (s.gate == (int)Gate::Residual) v = xr + o[j];
                        if (s.gate == (int)Gate::Tanh1x1) v = (xr + f16r(o[j])) * std::tanh(ga[j]);
                        if (s.gate == (int)Gate::SigmoidHalf) v = (xr + o[j]) * (1.f / (1.f + std::exp(-o[j])) - 0.5f);
                    }
                    if (s.out == (int)Output::NHWC) {
                        nhwc[p * cout + j] = f16r(v);
                    } else if (j < 3 * r * r) {
                        const int c = j / (r * r), ry = (j / r) % r, rx = j % r;
                        float add = 0.f;
                        if (s.resid == kModelInput) add = in[((size_t)c * h + y) * w + x];
                        out[((size_t)c * OH + r * y + ry) * OW + r * x + rx] = f16r(v + add);
                    }
                }
            }
        });
        if (s.out == (int)Output::NHWC) feat[s.dst] = std::move(nhwc);
    }
    return out;
}

}  // namespace ajiref

#endif
