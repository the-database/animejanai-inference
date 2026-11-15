// Whole SPANF3 (unshuffle 2x) or SRVGG Compact 2x models assembled from src/native calls only, on
// the data written by experimental/native-span/v2/prep_full.py / prep_compact.py: picks each op's
// tile variant, times the model as one CUDA graph and compares with the ONNX Runtime output.
//   aji_native_bench spanf3|compact DATADIR W H [iters]
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "native/conv.h"

using namespace ajinative;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

static std::vector<char> slurp(const std::string &p)
{
    FILE *f = std::fopen(p.c_str(), "rb");
    if (!f) { std::printf("cannot open %s\n", p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> v(n);
    if (std::fread(v.data(), 1, n, f) != (size_t)n) std::exit(1);
    std::fclose(f);
    return v;
}

struct Reader {
    std::vector<char> b;
    size_t o = 0;
    Weights conv(int taps, int cin, int cin_pad, int cout, int cout_pad, bool slope)
    {
        const uint16_t *w = reinterpret_cast<const uint16_t *>(b.data() + o); o += (size_t)taps * cin * cout * 2;
        const float *bias = reinterpret_cast<const float *>(b.data() + o); o += (size_t)cout * 4;
        const float *sl = nullptr;
        if (slope) { sl = reinterpret_cast<const float *>(b.data() + o); o += (size_t)cout * 4; }
        Weights out;
        CK(upload(out, w, bias, sl, taps, cin, cin_pad, cout, cout_pad));
        return out;
    }
};

// one step of the model: a 3x3 conv or the concat 1x1
struct Step {
    bool concat = false;
    Conv3x3 op;
    ConcatConv1x1 cop;
    Tensors t;
    const void *srcs[4] = {};
    int H = 0, W = 0, variant = 0;
};

static void run(const std::vector<Step> &steps, cudaStream_t s)
{
    for (const Step &st : steps) {
        if (st.concat) CK(launch(st.cop, st.srcs, st.t.dst, st.H * st.W, 0, s));
        else CK(launch(st.op, st.t, st.H, st.W, st.variant, s));
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) { std::printf("usage: aji_native_bench spanf3|compact DATADIR W H [iters]\n"); return 2; }
    const std::string kind = argv[1], d = argv[2];
    const int FW = std::atoi(argv[3]), FH = std::atoi(argv[4]), iters = argc > 5 ? std::atoi(argv[5]) : 500;
    std::string why;
    if (!device_supported(0, &why)) { std::printf("unsupported device: %s\n", why.c_str()); return 2; }
    int C = 0, n = 0;
    {
        FILE *f = std::fopen((d + "/model.txt").c_str(), "r");
        if (!f || std::fscanf(f, "%d %d", &C, &n) != 2) return 1;
        std::fclose(f);
    }
    Reader r{slurp(d + "/weights.bin")};
    const std::vector<char> inb = slurp(d + "/input.bin");
    void *in;
    CK(cudaMalloc(&in, inb.size()));
    CK(cudaMemcpy(in, inb.data(), inb.size(), cudaMemcpyHostToDevice));

    std::vector<Weights> ws;   // keeps the weights alive
    ws.reserve(64);
    std::vector<Step> steps;
    std::vector<void *> bufs;
    auto buf = [&](size_t halves) { void *p; CK(cudaMalloc(&p, halves * 2)); bufs.push_back(p); return p; };
    auto conv = [&](Input in_, Act a, Gate g, Output o, int cin, int cout, const Weights *w, const Weights *att,
                    const void *src, const void *resid, void *dst, int H, int W) {
        Step st;
        st.op.in = in_; st.op.act = a; st.op.gate = g; st.op.out = o; st.op.cin = cin; st.op.cout = cout;
        st.op.w = w; st.op.att = att;
        st.t.src = src; st.t.resid = resid; st.t.dst = dst; st.H = H; st.W = W;
        std::string why;
        if (!supported(st.op, &why)) { std::printf("unsupported step: %s\n", why.c_str()); std::exit(1); }
        steps.push_back(st);
    };
    size_t out_halves = 0;
    void *out = nullptr;
    if (kind == "spanf3") {
        const int H = FH / 2, W = FW / 2, nb = n;
        ws.push_back(r.conv(9, 12, 16, C, C, false));
        const Weights *head = &ws.back();
        std::vector<void *> X(nb + 1);
        for (auto &x : X) x = buf((size_t)H * W * C);
        void *A1 = buf((size_t)H * W * C), *A2 = buf((size_t)H * W * C), *Y = buf((size_t)H * W * C);
        conv(Input::NCHW3Unshuffle2, Act::None, Gate::None, Output::NHWC, 12, C, head, nullptr, in, nullptr, X[0], H, W);
        for (int b = 0; b < nb; b++) {
            ws.push_back(r.conv(9, C, C, C, C, false)); const Weights *c1 = &ws.back();
            ws.push_back(r.conv(9, C, C, C, C, false)); const Weights *c2 = &ws.back();
            ws.push_back(r.conv(9, C, C, C, C, false)); const Weights *c3 = &ws.back();
            ws.push_back(r.conv(1, C, C, C, C, false)); const Weights *att = &ws.back();
            conv(Input::NHWC, Act::Relu, Gate::None, Output::NHWC, C, C, c1, nullptr, X[b], nullptr, A1, H, W);
            conv(Input::NHWC, Act::Relu, Gate::None, Output::NHWC, C, C, c2, nullptr, A1, nullptr, A2, H, W);
            conv(Input::NHWC, Act::None, Gate::Tanh1x1, Output::NHWC, C, C, c3, att, A2, X[b], X[b + 1], H, W);
        }
        ws.push_back(r.conv(1, 4 * C, 4 * C, C, C, false));
        Step cat;
        cat.concat = true;
        cat.cop = ConcatConv1x1{C, 4, &ws.back()};
        cat.srcs[0] = X[0]; cat.srcs[1] = X[nb]; cat.srcs[2] = X[1]; cat.srcs[3] = A1;  // relu(c1) of the last block
        cat.t.dst = Y; cat.H = H; cat.W = W;
        if (!supported(cat.cop, &why)) { std::printf("unsupported concat: %s\n", why.c_str()); return 1; }
        steps.push_back(cat);
        ws.push_back(r.conv(9, C, C, 48, 48, false));
        out_halves = (size_t)3 * 4 * H * 4 * W;
        out = buf(out_halves);
        conv(Input::NHWC, Act::None, Gate::None, Output::DepthToSpace4, C, 48, &ws.back(), nullptr, Y, nullptr, out, H, W);
    } else if (kind == "compact") {
        const int H = FH, W = FW, nbody = n - 2;
        void *A = buf((size_t)H * W * C), *B = buf((size_t)H * W * C);
        ws.push_back(r.conv(9, 3, 16, C, C, true));
        conv(Input::NCHW3, Act::PRelu, Gate::None, Output::NHWC, 3, C, &ws.back(), nullptr, in, nullptr, A, H, W);
        void *x = A, *y = B;
        for (int i = 0; i < nbody; i++) {
            ws.push_back(r.conv(9, C, C, C, C, true));
            conv(Input::NHWC, Act::PRelu, Gate::None, Output::NHWC, C, C, &ws.back(), nullptr, x, nullptr, y, H, W);
            std::swap(x, y);
        }
        ws.push_back(r.conv(9, C, C, 12, 16, false));
        out_halves = (size_t)3 * 2 * H * 2 * W;
        out = buf(out_halves);
        conv(Input::NHWC, Act::None, Gate::None, Output::DepthToSpace2, C, 12, &ws.back(), nullptr, x, in, out, H, W);
    } else {
        std::printf("unknown model kind %s\n", kind.c_str());
        return 2;
    }
    if (r.o != r.b.size()) { std::printf("weights size mismatch %zu vs %zu\n", r.o, r.b.size()); return 1; }

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    // variants, in order, so each step sees its real inputs
    int v0 = 0, v1 = 0;
    for (Step &st : steps) {
        if (st.concat) { CK(launch(st.cop, st.srcs, st.t.dst, st.H * st.W, 0, s)); continue; }
        st.variant = pick_variant(st.op, st.t, st.H, st.W, s);
        if (st.variant < 0) { std::printf("variant selection failed\n"); return 1; }
        (st.variant ? v1 : v0)++;
    }
    run(steps, s);
    CK(cudaStreamSynchronize(s));

    // output vs ONNX Runtime CUDA fp16, in 8-bit levels
    std::vector<uint16_t> got(out_halves);
    CK(cudaMemcpy(got.data(), out, out_halves * 2, cudaMemcpyDeviceToHost));
    const std::vector<char> refb = slurp(d + "/ref_out.bin");
    const uint16_t *ref = reinterpret_cast<const uint16_t *>(refb.data());
    if (refb.size() != out_halves * 2) { std::printf("reference size mismatch\n"); return 1; }
    auto q = [](float v) { return std::fmin(255.f, std::fmax(0.f, std::nearbyint(v * 255))); };
    double se = 0; int mx = 0;
    for (size_t i = 0; i < out_halves; i++) {
        const int e = (int)std::fabs(q(half_bits_to_float(got[i])) - q(half_bits_to_float(ref[i])));
        se += (double)e * e;
        mx = std::max(mx, e);
    }
    const double mse = se / out_halves;

    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    run(steps, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 200; i++) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int rep = 0; rep < 5; rep++) {
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < iters; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float t;
        CK(cudaEventElapsedTime(&t, e0, e1));
        ms.push_back(t / iters);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%s %dx%d: %zu steps (variants: %d x [%s], %d x [%s]) whole model %.3f ms median [min %.3f max %.3f]; "
                "vs ORT fp16: PSNR %.2f dB, max %d levels\n",
                kind.c_str(), FW, FH, steps.size(), v0, variant_name(steps[0].op, 0), v1, variant_name(steps[0].op, 1),
                ms[2], ms[0], ms[4], mse > 0 ? 10 * std::log10(255.0 * 255 / mse) : 99.0, mx);
    return 0;
}
