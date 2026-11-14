// SPANF3 2x (pixel-unshuffle head) on hand-written kernels: NCHW fp16 RGB in -> NCHW fp16 RGB out.
//   head:   pixel-unshuffle(2) + conv_1 3x3 (12->C)                 -> X0
//   block:  k1 relu(conv3x3) -> k2 relu(conv3x3) -> k3 conv3x3 + tanh(1x1) gate + residual
//   tail:   conv_cat 1x1 over [X0, Xn, X1, relu(c1 of block n)] (no concat buffer) -> to_pixel 3x3 + depth-to-space(4)
// Tensor cores via mma.sync m16n8k16 with fp16 accumulation; A from padded shared memory via ldmatrix;
// B (weights) pre-packed in fragment order, one 16-byte __ldg per lane per fragment pair.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "native_kernels.cuh"

// ---------------------------------------------------------------- host side
struct Packed { uint4* w; half2* b; };

static std::vector<char> slurp(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return v;
}
static uint16_t hbits(const half& h) { uint16_t u; memcpy(&u, &h, 2); return u; }

// W: [taps][cin][cout] fp16 -> fragment order [taps][cin_pad/16][cout/16][lane] uint4
static Packed pack(const half* W, const float* bias, int taps, int cin, int cin_pad, int cout) {
    const int KC = cin_pad / 16, NTP = cout / 16;
    std::vector<uint32_t> v((size_t)taps * KC * NTP * 32 * 4);
    auto w = [&](int t, int k, int n) -> uint32_t { return k < cin ? hbits(W[((size_t)t * cin + k) * cout + n]) : 0u; };
    for (int t = 0; t < taps; t++) for (int kc = 0; kc < KC; kc++) for (int j = 0; j < NTP; j++) for (int lane = 0; lane < 32; lane++) {
        const int g = lane / 4, tq = lane % 4; uint32_t* o = &v[((((size_t)t * KC + kc) * NTP + j) * 32 + lane) * 4];
        for (int h = 0; h < 2; h++) {
            const int n = (2 * j + h) * 8 + g, k0 = kc * 16 + tq * 2;
            o[h * 2 + 0] = w(t, k0, n) | (w(t, k0 + 1, n) << 16);
            o[h * 2 + 1] = w(t, k0 + 8, n) | (w(t, k0 + 9, n) << 16);
        }
    }
    std::vector<half> hb(cout); for (int i = 0; i < cout; i++) hb[i] = __float2half(bias[i]);
    Packed p; CK(cudaMalloc(&p.w, v.size() * 4)); CK(cudaMemcpy(p.w, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&p.b, cout * 2)); CK(cudaMemcpy(p.b, hb.data(), cout * 2, cudaMemcpyHostToDevice));
    return p;
}

constexpr int C = 48;
struct Model { Packed head, c1[8], c2[8], c3[8], att[8], cat, topix; int nb; };

template <int WARPS, int WR>
static void launch(const Model& m, const half* in, int H, int W, half* const* X, half* A1, half* A2, half* Y, half* out, cudaStream_t s) {
    constexpr int TH = WARPS * WR;
    const dim3 grid(W / TW, (H + TH - 1) / TH), blk(WARPS * 32);
    conv3x3<16, C, WARPS, WR, IN_NCHW_UNSHUFFLE2, E_BIAS><<<grid, blk, conv_smem<16, C, WARPS, WR>(), s>>>(in, H, W, m.head.w, m.head.b, nullptr, nullptr, nullptr, X[0]);
    for (int b = 0; b < m.nb; b++) {
        conv3x3<C, C, WARPS, WR, IN_NHWC, E_RELU><<<grid, blk, conv_smem<C, C, WARPS, WR>(), s>>>(X[b], H, W, m.c1[b].w, m.c1[b].b, nullptr, nullptr, nullptr, A1);
        conv3x3<C, C, WARPS, WR, IN_NHWC, E_RELU><<<grid, blk, conv_smem<C, C, WARPS, WR>(), s>>>(A1, H, W, m.c2[b].w, m.c2[b].b, nullptr, nullptr, nullptr, A2);
        conv3x3<C, C, WARPS, WR, IN_NHWC, E_GATE><<<grid, blk, conv_smem<C, C, WARPS, WR>(), s>>>(A2, H, W, m.c3[b].w, m.c3[b].b, m.att[b].w, m.att[b].b, X[b], X[b + 1]);
    }
    constexpr int CW = 4;
    convcat<C, CW><<<(H * W + CW * 16 - 1) / (CW * 16), CW * 32, 4 * CW * 16 * (C + 8) * 2, s>>>(X[0], X[m.nb], X[1], A1, H * W, m.cat.w, m.cat.b, Y);
    conv3x3<C, C, WARPS, WR, IN_NHWC, E_D2S4><<<grid, blk, conv_smem<C, C, WARPS, WR>(), s>>>(Y, H, W, m.topix.w, m.topix.b, nullptr, nullptr, nullptr, out);
}

template <int WARPS, int WR>
static float run_config(const char* label, const Model& m, const half* in, int H, int W, half* const* X, half* A1, half* A2, half* Y, half* out,
                        int iters, const std::string& dump) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    launch<WARPS, WR>(m, in, H, W, X, A1, A2, Y, out, s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
    if (!dump.empty()) {
        std::vector<half> h((size_t)3 * 4 * H * 4 * W); CK(cudaMemcpy(h.data(), out, h.size() * 2, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump.c_str(), "wb"); fwrite(h.data(), 2, h.size(), f); fclose(f);
    }
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    launch<WARPS, WR>(m, in, H, W, X, A1, A2, Y, out, s);
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 200; i++) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int rep = 0; rep < 5; rep++) {
        CK(cudaEventRecord(e0, s)); for (int i = 0; i < iters; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t / iters);
    }
    std::sort(ms.begin(), ms.end());
    printf("%-14s whole model %.3f ms median [min %.3f max %.3f]\n", label, ms[2], ms[0], ms[4]);
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s));
    return ms[2];
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: spanf3_native DATADIR W H [iters]\n"); return 2; }
    const std::string d = argv[1]; const int FW = atoi(argv[2]), FH = atoi(argv[3]), iters = argc > 4 ? atoi(argv[4]) : 500;
    const int W = FW / 2, H = FH / 2;
    if (W % TW) { printf("feature width must be a multiple of %d\n", TW); return 2; }
    int c = 0, nb = 0; { FILE* f = fopen((d + "/model.txt").c_str(), "r"); if (!f || fscanf(f, "%d %d", &c, &nb) != 2) return 1; fclose(f); }
    if (c != C || nb > 8) { printf("model C=%d blocks=%d not supported by this build\n", c, nb); return 2; }
    auto wb = slurp(d + "/weights.bin"); size_t o = 0;
    auto take = [&](int taps, int cin, int cin_pad, int cout) {
        const half* w = reinterpret_cast<const half*>(wb.data() + o); o += (size_t)taps * cin * cout * 2;
        const float* b = reinterpret_cast<const float*>(wb.data() + o); o += (size_t)cout * 4;
        return pack(w, b, taps, cin, cin_pad, cout);
    };
    Model m; m.nb = nb;
    m.head = take(9, 12, 16, C);
    for (int b = 0; b < nb; b++) { m.c1[b] = take(9, C, C, C); m.c2[b] = take(9, C, C, C); m.c3[b] = take(9, C, C, C); m.att[b] = take(1, C, C, C); }
    m.cat = take(1, 4 * C, 4 * C, C); m.topix = take(9, C, C, C);
    if (o != wb.size()) { printf("weights size mismatch %zu vs %zu\n", o, wb.size()); return 1; }

    auto inb = slurp(d + "/input.bin");
    half* in; CK(cudaMalloc(&in, inb.size())); CK(cudaMemcpy(in, inb.data(), inb.size(), cudaMemcpyHostToDevice));
    const size_t N = (size_t)H * W * C;
    std::vector<half*> X(nb + 1); for (auto& p : X) CK(cudaMalloc(&p, N * 2));
    half *A1, *A2, *Y, *out; CK(cudaMalloc(&A1, N * 2)); CK(cudaMalloc(&A2, N * 2)); CK(cudaMalloc(&Y, N * 2));
    CK(cudaMalloc(&out, (size_t)3 * 4 * H * 4 * W * 2));

    run_config<4, 2>("warps4 rows2", m, in, H, W, X.data(), A1, A2, Y, out, iters, d + "/native_out.bin");
    run_config<4, 4>("warps4 rows4", m, in, H, W, X.data(), A1, A2, Y, out, iters, "");
    run_config<8, 2>("warps8 rows2", m, in, H, W, X.data(), A1, A2, Y, out, iters, "");
    run_config<2, 4>("warps2 rows4", m, in, H, W, X.data(), A1, A2, Y, out, iters, "");
    return 0;
}
