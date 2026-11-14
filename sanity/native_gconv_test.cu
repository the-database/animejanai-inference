// aji_native_gconv_test: every k_gconv mode against a CPU reference (fp32 math on the same fp16
// inputs and weights). Modes: stride 1 with ReLU / LeakyReLU / none, a strided input (a channel
// prefix of a wider buffer) and an output slice, one and two scaled residuals, stride 2 from NHWC
// and from NCHW planes, and the phase scatter with R = 2 and 4; ragged and aligned sizes, both
// tile variants. Skips (exit 0) without an sm_80+ GPU.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "native/gconv.h"
#include "native/pack.h"

using namespace ajinative;

static std::mt19937 rng(1234);
static float urand(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }
static uint16_t h(float f) { return float_to_half_bits(f); }
static float f(uint16_t b) { return half_bits_to_float(b); }

struct Case {
    const char *name;
    int cin, cout, stride, IH, IW;
    bool nchw;
    int src_extra;   // extra channels in the input pixel (prefix read)
    int dst_extra, dst_off;
    int nres;
    GAct act;
    int R;
    bool up2 = false;
    bool acc32 = false;
};

static int fails = 0, checks = 0;
static double worst = 0;

static void run(const Case &c, int variant, bool split)
{
    const int cin_pad = (c.cin + 15) / 16 * 16;
    // up2: the source buffer is IH/2 x IW/2, read nearest-upsampled to IH x IW
    const int SH = c.up2 ? c.IH / 2 : c.IH, SW = c.up2 ? c.IW / 2 : c.IW;
    const int OH = c.stride == 2 ? (c.IH + 1) / 2 : c.IH, OW = c.stride == 2 ? (c.IW + 1) / 2 : c.IW;
    const int cout_pad = (c.cout + 15) / 16 * 16;
    const int CH = c.R ? cout_pad / c.R : 0;  // phase: channels per output pixel
    // weights [9][cin][cout]
    std::vector<uint16_t> w((size_t)9 * c.cin * c.cout);
    for (auto &x : w) x = h(urand(0.08f));
    std::vector<float> b(c.cout);
    for (auto &x : b) x = urand(0.2f);
    GWeights gw;
    if (gupload(gw, w.data(), b.data(), c.cin, cin_pad, c.cout, cout_pad) != cudaSuccess) {
        printf("FAIL %s: upload\n", c.name);
        fails++;
        return;
    }
    // input
    const int sstride = c.nchw ? c.IH * c.IW : cin_pad + c.src_extra;
    std::vector<uint16_t> in(c.nchw ? (size_t)3 * c.IH * c.IW : (size_t)SH * SW * sstride);
    for (auto &x : in) x = h(urand(1.f));
    if (!c.nchw)
        for (int p = 0; p < SH * SW; p++)
            for (int k = c.cin; k < cin_pad; k++) in[(size_t)p * sstride + k] = 0;  // padded channels are zero
    auto inv = [&](int y, int x, int k) -> float {
        if (y < 0 || y >= c.IH || x < 0 || x >= c.IW) return 0.f;
        if (c.up2) return f(in[((size_t)(y / 2) * SW + x / 2) * sstride + k]);
        return c.nchw ? f(in[(size_t)k * c.IH * c.IW + (size_t)y * c.IW + x]) : f(in[((size_t)y * c.IW + x) * sstride + k]);
    };
    // residuals
    std::vector<uint16_t> r1((size_t)OH * OW * (cout_pad + 8)), r2((size_t)OH * OW * cout_pad);
    for (auto &x : r1) x = h(urand(1.f));
    for (auto &x : r2) x = h(urand(1.f));
    const float c1 = 0.2f, c2 = 1.0f;
    // output buffer
    const int dstride = c.R ? CH + c.dst_extra : cout_pad + c.dst_extra;
    const int DH = c.R ? c.R * OH : OH, DW = c.R ? c.R * OW : OW;
    std::vector<uint16_t> out((size_t)DH * DW * dstride, h(7.f));

    void *dsrc, *ddst, *dr1, *dr2;
    cudaMalloc(&dsrc, in.size() * 2);
    cudaMalloc(&ddst, out.size() * 2);
    cudaMalloc(&dr1, r1.size() * 2);
    cudaMalloc(&dr2, r2.size() * 2);
    cudaMemcpy(dsrc, in.data(), in.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ddst, out.data(), out.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dr1, r1.data(), r1.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dr2, r2.data(), r2.size() * 2, cudaMemcpyHostToDevice);
    GConvArgs a;
    a.src = dsrc;
    a.src_stride = sstride;
    a.IH = c.IH;
    a.IW = c.IW;
    a.dst = (uint16_t *)ddst + c.dst_off;
    a.dst_stride = dstride;
    a.H = OH;
    a.W = OW;
    if (c.nres >= 1) { a.r1 = dr1; a.r1_stride = cout_pad + 8; a.c1 = c1; }
    if (c.nres >= 2) { a.r2 = dr2; a.r2_stride = cout_pad; a.c2 = c2; }
    a.act = c.act;
    a.slope = 0.2f;
    a.stride = c.stride;
    a.nchw = c.nchw;
    a.R = c.R;
    a.up2 = c.up2;
    a.acc32 = c.acc32;
    float *dws = nullptr;
    if (split) {
        a.ws_floats = gworkspace(gw, a);
        if (a.ws_floats) cudaMalloc(&dws, a.ws_floats * 4);
        a.ws = dws;
    }
    std::string why;
    if (!gsupported(gw, a, &why)) {
        printf("FAIL %s: unsupported: %s\n", c.name, why.c_str());
        fails++;
        return;
    }
    const int sys = c.R ? c.R : 1;
    for (int sy = 0; sy < sys; sy++) {
        a.sy = sy;
        if (glaunch(gw, a, variant, nullptr) != cudaSuccess) {
            printf("FAIL %s: launch\n", c.name);
            fails++;
            return;
        }
    }
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(out.data(), ddst, out.size() * 2, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", c.name, cudaGetErrorString(e));
        fails++;
        return;
    }
    double maxe = 0, se = 0;
    long n = 0;
    bool clobber = false;
    for (int y = 0; y < OH; y++)
        for (int x = 0; x < OW; x++)
            for (int o = 0; o < cout_pad; o++) {
                double v = o < c.cout ? b[o] : 0;
                if (o < c.cout)
                    for (int t = 0; t < 9; t++)
                        for (int k = 0; k < c.cin; k++)
                            v += inv(c.stride * y + t / 3 - 1, c.stride * x + t % 3 - 1, k) * f(w[((size_t)t * c.cin + k) * c.cout + o]);
                if (c.nres >= 1) v += c1 * f(r1[((size_t)y * OW + x) * (cout_pad + 8) + o]);
                if (c.nres >= 2) v += c2 * f(r2[((size_t)y * OW + x) * cout_pad + o]);
                if (c.act == GAct::Relu) v = v > 0 ? v : 0;
                if (c.act == GAct::LRelu) v = v > 0 ? v : 0.2 * v;
                if (c.act == GAct::Gelu) v = 0.5 * v * (1.0 + std::erf(v / 1.4140625));
                size_t idx;
                if (c.R) {
                    const int sy = o / CH / c.R, sx = (o / CH) % c.R, ch = o % CH;  // layout per launch: sx*CH + ch
                    (void)sy;
                    // each launch sy writes channels [sx*CH + ch] of its own weights; here the test
                    // uses the same weights for every sy, so row phase sy repeats the conv
                    for (int s2 = 0; s2 < c.R; s2++) {
                        idx = ((size_t)(c.R * y + s2) * DW + c.R * x + sx) * dstride + c.dst_off + ch;
                        const double d = std::fabs(f(out[idx]) - v);
                        maxe = std::max(maxe, d);
                        se += d * d;
                        n++;
                    }
                    continue;
                }
                idx = ((size_t)y * OW + x) * dstride + c.dst_off + o;
                const double d = std::fabs(f(out[idx]) - v);
                maxe = std::max(maxe, d);
                se += d * d;
                n++;
            }
    // the slice must not touch channels outside it
    if (!c.R && (c.dst_extra || c.dst_off))
        for (int p = 0; p < OH * OW && !clobber; p++)
            for (int k = 0; k < dstride; k++)
                if ((k < c.dst_off || k >= c.dst_off + cout_pad) && out[(size_t)p * dstride + k] != h(7.f)) clobber = true;
    const double rms = std::sqrt(se / std::max(n, 1L));
    const bool ok = maxe < 0.03 && rms < 0.004 && !clobber;
    checks++;
    worst = std::max(worst, maxe);
    if (!ok) {
        fails++;
        printf("FAIL %s v%d %dx%d: max %.4f rms %.5f%s\n", c.name, variant, c.IW, c.IH, maxe, rms, clobber ? " (wrote outside its slice)" : "");
    }
    cudaFree(dsrc);
    cudaFree(ddst);
    cudaFree(dr1);
    cudaFree(dr2);
    cudaFree(dws);
    grelease(gw);
}

int main()
{
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess || p.major < 8) {
        printf("native gconv: SKIP (no sm_80+ GPU)\n");
        return 0;
    }
    const Case cases[] = {
        {"s1 32->32 relu", 32, 32, 1, 0, 0, false, 0, 0, 0, 0, GAct::Relu, 0},
        {"s1 64->64 lrelu", 64, 64, 1, 0, 0, false, 0, 0, 0, 0, GAct::LRelu, 0},
        {"s1 192->192 lrelu", 192, 192, 1, 0, 0, false, 0, 0, 0, 0, GAct::LRelu, 0},
        {"s1 96->32 prefix + slice", 96, 32, 1, 0, 0, false, 96, 64, 32, 0, GAct::LRelu, 0},
        {"s1 160->32 none", 160, 32, 1, 0, 0, false, 32, 0, 0, 0, GAct::None, 0},
        {"s1 192->64 two residuals", 192, 64, 1, 0, 0, false, 0, 0, 0, 2, GAct::None, 0},
        {"s1 64->64 residual lrelu", 64, 64, 1, 0, 0, false, 0, 0, 0, 1, GAct::LRelu, 0},
        {"s1 13->48 cout 48", 13, 48, 1, 0, 0, false, 0, 0, 0, 0, GAct::None, 0},
        {"s2 28->16 lrelu", 28, 16, 2, 0, 0, false, 0, 0, 0, 0, GAct::LRelu, 0},
        {"s2 64->128 lrelu", 64, 128, 2, 0, 0, false, 0, 0, 0, 0, GAct::LRelu, 0},
        {"s2 nchw 3->16", 3, 16, 2, 0, 0, true, 0, 0, 0, 0, GAct::LRelu, 0},
        {"s2 nchw 3->32", 3, 32, 2, 0, 0, true, 0, 0, 0, 0, GAct::LRelu, 0},
        {"phase R2 16->16", 16, 16, 1, 0, 0, false, 0, 0, 0, 0, GAct::None, 2},
        {"phase R2 32->32", 32, 32, 1, 0, 0, false, 0, 0, 0, 0, GAct::None, 2},
        {"phase R4 192->64", 192, 64, 1, 0, 0, false, 0, 0, 0, 0, GAct::None, 4},
        {"phase R4 64->32", 64, 32, 1, 0, 0, false, 0, 0, 0, 0, GAct::None, 4},
        {"up2 64->64 lrelu", 64, 64, 1, 0, 0, false, 0, 0, 0, 0, GAct::LRelu, 0, true},
        {"up2 64->64 prefix", 64, 64, 1, 0, 0, false, 64, 0, 0, 0, GAct::None, 0, true},
        {"acc32 192->64 two residuals", 192, 64, 1, 0, 0, false, 0, 0, 0, 2, GAct::None, 0, false, true},
        {"acc32 96->32 prefix + slice", 96, 32, 1, 0, 0, false, 96, 64, 32, 0, GAct::LRelu, 0, false, true},
        {"acc32 up2 64->64", 64, 64, 1, 0, 0, false, 0, 0, 0, 0, GAct::LRelu, 0, true, true},
        {"phase R2 128->256 gelu (4 chunks)", 128, 256, 1, 0, 0, false, 0, 0, 0, 0, GAct::Gelu, 2},
        {"s1 120->120 gelu", 120, 120, 1, 0, 0, false, 8, 0, 0, 1, GAct::Gelu, 0},
    };
    const int sizes[][2] = {{37, 23}, {38, 24}, {64, 32}};
    for (const Case &c0 : cases)
        for (const auto &sz : sizes)
            for (int v = 0; v < gvariant_count(); v++)
                for (int split = 0; split < 2; split++) {
                    Case c = c0;
                    c.IW = sz[0];
                    c.IH = sz[1];
                    if (c.up2 && (c.IW % 2 || c.IH % 2)) continue;
                    run(c, v, split != 0);
                }
    printf("native gconv: %s (%d checks; worst max %.4f)\n", fails ? "FAIL" : "OK", checks, worst);
    return fails ? 1 : 0;
}
