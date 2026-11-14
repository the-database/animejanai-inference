// native/conv: every op in the kernel table, every tile variant, at sizes that are and are not
// multiples of the tile, against a float reference on the CPU computed from the same fp16 data.
// The outputs are written into buffers with guard bands that must stay untouched. Needs a GPU of
// compute capability 8.0+ (skips otherwise).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "native/conv.h"

using namespace ajinative;

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "line %d: %s: %s\n", __LINE__, #x, cudaGetErrorString(e_)); return 1; \
} } while (0)

namespace {

double worst_max = 0, worst_rms = 0;
uint32_t rng = 12345u;
float urand() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xffffff) / 16777216.0f; }
float rh(float f) { return half_bits_to_float(float_to_half_bits(f)); }  // round to fp16

struct HostConv {   // weights as the kernels see them (fp16-rounded)
    int taps, cin, cout;
    std::vector<uint16_t> w;   // [taps][cin][cout]
    std::vector<float> wf, b, slope;
};
HostConv make_conv(int taps, int cin, int cout, bool prelu)
{
    HostConv c{taps, cin, cout, {}, {}, {}, {}};
    const float s = std::sqrt(3.0f / (taps * cin));
    for (int i = 0; i < taps * cin * cout; i++) {
        const uint16_t h = float_to_half_bits((urand() * 2 - 1) * s);
        c.w.push_back(h);
        c.wf.push_back(half_bits_to_float(h));
    }
    for (int i = 0; i < cout; i++) c.b.push_back(rh((urand() * 2 - 1) * 0.2f));
    if (prelu) for (int i = 0; i < cout; i++) c.slope.push_back(rh(urand() * 0.5f));
    return c;
}

struct DevBuf {
    void *p = nullptr;
    size_t n = 0;  // halves, excluding the guard bands
    static constexpr size_t kGuard = 4096;
    int alloc(size_t halves)
    {
        n = halves;
        if (cudaMalloc(&p, (n + 2 * kGuard) * 2) != cudaSuccess) return 1;
        std::vector<uint16_t> s(n + 2 * kGuard, 0x7bab);
        return cudaMemcpy(p, s.data(), s.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess;
    }
    void *data() const { return static_cast<uint16_t *>(p) + kGuard; }
    int put(const std::vector<uint16_t> &v) const
    {
        return cudaMemcpy(data(), v.data(), v.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess;
    }
    // copies the payload out; returns 1 if a guard half changed
    int get(std::vector<float> &out) const
    {
        std::vector<uint16_t> s(n + 2 * kGuard);
        if (cudaMemcpy(s.data(), p, s.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        for (size_t i = 0; i < kGuard; i++)
            if (s[i] != 0x7bab || s[kGuard + n + i] != 0x7bab) return 1;
        out.resize(n);
        for (size_t i = 0; i < n; i++) out[i] = half_bits_to_float(s[kGuard + i]);
        return 0;
    }
    ~DevBuf() { cudaFree(p); }
};

std::vector<uint16_t> random_tensor(size_t n, std::vector<float> &f)
{
    std::vector<uint16_t> h(n);
    f.resize(n);
    for (size_t i = 0; i < n; i++) { h[i] = float_to_half_bits(urand() * 2 - 1); f[i] = half_bits_to_float(h[i]); }
    return h;
}

float act_ref(Act a, float v, float slope)
{
    switch (a) {
    case Act::Relu: return v > 0 ? v : 0;
    case Act::PRelu: return v > 0 ? v : v * slope;
    case Act::SiLU: return v / (1 + std::exp(-v));
    default: return v;
    }
}

// one op at one size; returns 0 on success
int run_case(Input in, Act act, Gate gate, Output out, int C, int H, int W, int *checked)
{
    const int cin = in == Input::NCHW3 ? 3 : in == Input::NCHW3Unshuffle2 ? 12 : C;
    const int cout = out == Output::DepthToSpace2 ? 12 : out == Output::DepthToSpace4 ? 48 : C;
    const int cin_pad = in == Input::NHWC ? cin : 16, cout_pad = out == Output::DepthToSpace2 ? 16 : cout;
    const int R = out == Output::DepthToSpace2 ? 2 : out == Output::DepthToSpace4 ? 4 : 1;
    HostConv hc = make_conv(9, cin, cout, act == Act::PRelu), ha{};
    Weights w, att;
    CUDA(upload(w, hc.w.data(), hc.b.data(), act == Act::PRelu ? hc.slope.data() : nullptr, 9, cin, cin_pad, cout, cout_pad));
    if (gate == Gate::Tanh1x1) {
        ha = make_conv(1, cout, cout, false);
        CUDA(upload(att, ha.w.data(), ha.b.data(), nullptr, 1, cout, cout, cout, cout));
    }
    Conv3x3 op;
    op.in = in; op.act = act; op.gate = gate; op.out = out; op.cin = cin; op.cout = cout; op.w = &w;
    op.att = gate == Gate::Tanh1x1 ? &att : nullptr;
    std::string why;
    if (!supported(op, &why)) { std::fprintf(stderr, "unsupported: %s\n", why.c_str()); return 1; }

    // input
    std::vector<float> xf;
    const int iH = in == Input::NCHW3Unshuffle2 ? 2 * H : H, iW = in == Input::NCHW3Unshuffle2 ? 2 * W : W;
    const size_t nin = in == Input::NHWC ? (size_t)H * W * cin : (size_t)3 * iH * iW;
    std::vector<uint16_t> xh = random_tensor(nin, xf);
    auto X = [&](int y, int x, int c) -> float {  // the conv's input channel c at (y, x), zero outside
        if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;
        if (in == Input::NHWC) return xf[((size_t)y * W + x) * cin + c];
        if (in == Input::NCHW3) return xf[((size_t)c * H + y) * W + x];
        const int ch = c / 4, ry = (c / 2) % 2, rx = c % 2;
        return xf[((size_t)ch * iH + 2 * y + ry) * iW + 2 * x + rx];
    };
    // residual: NHWC block input for gates, NCHW model input for depth-to-space
    std::vector<float> rf;
    std::vector<uint16_t> rh_;
    const bool d2s_add = out != Output::NHWC && C != 48;  // exercise both with and without
    if (gate != Gate::None) rh_ = random_tensor((size_t)H * W * cout, rf);
    if (d2s_add) rh_ = random_tensor((size_t)3 * H * W, rf);

    DevBuf din, dres, dout;
    CHECK(din.alloc(nin) == 0 && din.put(xh) == 0);
    if (!rh_.empty()) CHECK(dres.alloc(rh_.size()) == 0 && dres.put(rh_) == 0);
    const size_t nout = out == Output::NHWC ? (size_t)H * W * cout : (size_t)3 * R * H * R * W;
    CHECK(dout.alloc(nout) == 0);

    // reference
    std::vector<float> ref(nout);
    std::vector<float> o(cout);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            for (int co = 0; co < cout; co++) {
                double s = hc.b[co];
                for (int t = 0; t < 9; t++)
                    for (int ci = 0; ci < cin; ci++) {
                        const float v = X(y + t / 3 - 1, x + t % 3 - 1, ci);
                        if (v != 0.f) s += (double)v * hc.wf[((size_t)t * cin + ci) * cout + co];
                    }
                o[co] = act_ref(act, (float)s, act == Act::PRelu ? hc.slope[co] : 0.f);
            }
            for (int co = 0; co < cout; co++) {
                float v = o[co];
                if (gate != Gate::None) {
                    const float r = rf[((size_t)y * W + x) * cout + co];
                    if (gate == Gate::Residual) v = r + v;
                    if (gate == Gate::SigmoidHalf) v = (r + v) * (1 / (1 + std::exp(-v)) - 0.5f);
                    if (gate == Gate::Tanh1x1) {
                        double a = ha.b[co];
                        for (int ci = 0; ci < cout; ci++) a += (double)rh(o[ci]) * ha.wf[(size_t)ci * cout + co];
                        v = (r + v) * std::tanh((float)a);
                    }
                }
                if (out == Output::NHWC) {
                    ref[((size_t)y * W + x) * cout + co] = v;
                } else {
                    const int c = co / (R * R), ry = (co / R) % R, rx = co % R;
                    if (d2s_add) v += rf[((size_t)c * H + y) * W + x];
                    ref[((size_t)c * R * H + R * y + ry) * (R * W) + R * x + rx] = v;
                }
            }
        }

    Tensors t;
    t.src = din.data();
    t.resid = rh_.empty() ? nullptr : dres.data();
    t.dst = dout.data();
    for (int v = 0; v < variant_count(op); v++) {
        CUDA(launch(op, t, H, W, v, 0));
        CUDA(cudaDeviceSynchronize());
        std::vector<float> got;
        if (dout.get(got)) { std::fprintf(stderr, "guard band overwritten\n"); return 1; }
        double se = 0, mx = 0;
        for (size_t i = 0; i < nout; i++) {
            const double e = std::fabs((double)got[i] - ref[i]);
            if (!(e == e)) mx = 1e9;
            se += e * e;
            if (e > mx) mx = e;
        }
        const double rms = std::sqrt(se / nout);
        if (mx > 0.02 || rms > 0.003) {
            std::fprintf(stderr, "in %d act %d gate %d out %d C %d %dx%d variant %d: max %.4f rms %.5f\n",
                         (int)in, (int)act, (int)gate, (int)out, C, W, H, v, mx, rms);
            return 1;
        }
        if (mx > worst_max) worst_max = mx;
        if (rms > worst_rms) worst_rms = rms;
        (*checked)++;
    }
    int pv = pick_variant(op, t, H, W, 0);
    CHECK(pv >= 0 && pv < variant_count(op));
    release(w);
    release(att);
    return 0;
}

int run_concat(int C, int nsrc, int npx, int *checked)
{
    HostConv hc = make_conv(1, nsrc * C, C, false);
    Weights w;
    CUDA(upload(w, hc.w.data(), hc.b.data(), nullptr, 1, nsrc * C, nsrc * C, C, C));
    ConcatConv1x1 op{C, nsrc, &w};
    CHECK(supported(op));
    std::vector<std::vector<float>> sf(nsrc);
    DevBuf src[4], dst;
    const void *ptr[4] = {};
    for (int s = 0; s < nsrc; s++) {
        std::vector<uint16_t> h = random_tensor((size_t)npx * C, sf[s]);
        CHECK(src[s].alloc(h.size()) == 0 && src[s].put(h) == 0);
        ptr[s] = src[s].data();
    }
    CHECK(dst.alloc((size_t)npx * C) == 0);
    CUDA(launch(op, ptr, dst.data(), npx, 0, 0));
    CUDA(cudaDeviceSynchronize());
    std::vector<float> got;
    CHECK(dst.get(got) == 0);
    double mx = 0;
    for (int p = 0; p < npx; p++)
        for (int co = 0; co < C; co++) {
            double s = hc.b[co];
            for (int k = 0; k < nsrc * C; k++) s += (double)sf[k / C][(size_t)p * C + k % C] * hc.wf[(size_t)k * C + co];
            mx = std::fmax(mx, std::fabs(s - got[(size_t)p * C + co]));
        }
    if (mx > 0.02) { std::fprintf(stderr, "concat C %d nsrc %d: max %.4f\n", C, nsrc, mx); return 1; }
    (*checked)++;
    release(w);
    return 0;
}

}  // namespace

int main()
{
    int dev = 0;
    std::string why;
    if (cudaGetDevice(&dev) != cudaSuccess || !device_supported(dev, &why)) {
        std::printf("native conv: skipped (%s)\n", why.empty() ? "no CUDA device" : why.c_str());
        return 0;
    }
    int checked = 0;
    const int sizes[][2] = {{19, 37}, {32, 48}};  // H, W: ragged and tile-aligned
    for (const auto &sz : sizes)
        for (int C : {32, 48, 64}) {
            for (Act a : {Act::None, Act::Relu, Act::PRelu, Act::SiLU})
                CHECK(run_case(Input::NHWC, a, Gate::None, Output::NHWC, C, sz[0], sz[1], &checked) == 0);
            for (Gate g : {Gate::Residual, Gate::Tanh1x1, Gate::SigmoidHalf})
                CHECK(run_case(Input::NHWC, Act::None, g, Output::NHWC, C, sz[0], sz[1], &checked) == 0);
            for (Input in : {Input::NCHW3, Input::NCHW3Unshuffle2})
                for (Act a : {Act::None, Act::Relu, Act::PRelu})
                    CHECK(run_case(in, a, Gate::None, Output::NHWC, C, sz[0], sz[1], &checked) == 0);
            for (Output o : {Output::DepthToSpace2, Output::DepthToSpace4})
                CHECK(run_case(Input::NHWC, Act::None, Gate::None, o, C, sz[0], sz[1], &checked) == 0);
            for (int ns : {2, 3, 4}) CHECK(run_concat(C, ns, 1237, &checked) == 0);
        }

    // refusals
    Weights w;
    HostConv hc = make_conv(9, 48, 48, false);
    CUDA(upload(w, hc.w.data(), hc.b.data(), nullptr, 9, 48, 48, 48, 48));
    Conv3x3 bad;
    bad.cin = bad.cout = 48; bad.w = &w; bad.gate = Gate::Residual; bad.act = Act::Relu;
    CHECK(!supported(bad, &why));                 // gate with an activation
    bad.gate = Gate::None; bad.act = Act::PRelu;
    CHECK(!supported(bad, &why));                 // PReLU without slopes
    bad.act = Act::None; bad.cin = bad.cout = 40;
    CHECK(!supported(bad, &why));                 // weights do not match
    release(w);

    std::printf("native conv: OK (%d op/variant/size checks; worst max %.4f, rms %.5f; limits 0.02, 0.003)\n",
                checked, worst_max, worst_rms);
    return 0;
}
