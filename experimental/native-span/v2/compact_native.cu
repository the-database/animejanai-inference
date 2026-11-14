// SRVGG Compact 2x on the native kernels: NCHW fp16 RGB in -> NCHW fp16 RGB out.
//   conv_first 3x3 (3->C) + PReLU, N x [conv 3x3 (C->C) + PReLU], conv_last 3x3 (C->12) + depth-to-space(2)
//   + nearest-2x input, all at the input resolution.
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

struct Packed { uint4* w; half2* b; half2* slope; };
static std::vector<char> slurp(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return v;
}
static uint16_t hbits(const half& h) { uint16_t u; memcpy(&u, &h, 2); return u; }
static half2* upload_h2(const float* v, int n, int npad) {
    std::vector<half> h(npad, __float2half(0.f)); for (int i = 0; i < n; i++) h[i] = __float2half(v[i]);
    half2* d; CK(cudaMalloc(&d, npad * 2)); CK(cudaMemcpy(d, h.data(), npad * 2, cudaMemcpyHostToDevice)); return d;
}
static Packed pack(const half* W, const float* bias, int taps, int cin, int cin_pad, int cout, int cout_pad) {
    const int KC = cin_pad / 16, NTP = cout_pad / 16;
    std::vector<uint32_t> v((size_t)taps * KC * NTP * 32 * 4);
    auto w = [&](int t, int k, int n) -> uint32_t { return (k < cin && n < cout) ? hbits(W[((size_t)t * cin + k) * cout + n]) : 0u; };
    for (int t = 0; t < taps; t++) for (int kc = 0; kc < KC; kc++) for (int j = 0; j < NTP; j++) for (int lane = 0; lane < 32; lane++) {
        const int g = lane / 4, tq = lane % 4; uint32_t* o = &v[((((size_t)t * KC + kc) * NTP + j) * 32 + lane) * 4];
        for (int h = 0; h < 2; h++) {
            const int n = (2 * j + h) * 8 + g, k0 = kc * 16 + tq * 2;
            o[h * 2 + 0] = w(t, k0, n) | (w(t, k0 + 1, n) << 16);
            o[h * 2 + 1] = w(t, k0 + 8, n) | (w(t, k0 + 9, n) << 16);
        }
    }
    Packed p; CK(cudaMalloc(&p.w, v.size() * 4)); CK(cudaMemcpy(p.w, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    p.b = upload_h2(bias, cout, cout_pad); p.slope = nullptr; return p;
}

constexpr int C = 64;
struct Model { Packed first, body[32], last; int nbody; };

template <int WARPS, int WR>
static void launch(const Model& m, const half* in, int H, int W, half* A, half* B, half* out, cudaStream_t s) {
    constexpr int TH = WARPS * WR;
    const dim3 grid(W / TW, (H + TH - 1) / TH), blk(WARPS * 32);
    conv3x3<16, C, WARPS, WR, IN_NCHW3, E_PRELU><<<grid, blk, conv_smem<16, C, WARPS, WR>(), s>>>(in, H, W, m.first.w, m.first.b, nullptr, m.first.slope, nullptr, A);
    half *x = A, *y = B;
    for (int i = 0; i < m.nbody; i++) {
        conv3x3<C, C, WARPS, WR, IN_NHWC, E_PRELU><<<grid, blk, conv_smem<C, C, WARPS, WR>(), s>>>(x, H, W, m.body[i].w, m.body[i].b, nullptr, m.body[i].slope, nullptr, y);
        std::swap(x, y);
    }
    conv3x3<C, 16, WARPS, WR, IN_NHWC, E_D2S2_ADDIN><<<grid, blk, conv_smem<C, 16, WARPS, WR>(), s>>>(x, H, W, m.last.w, m.last.b, nullptr, nullptr, in, out);
}

template <int WARPS, int WR>
static void run_config(const char* label, const Model& m, const half* in, int H, int W, half* A, half* B, half* out, int iters, const std::string& dump) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    launch<WARPS, WR>(m, in, H, W, A, B, out, s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
    if (!dump.empty()) {
        std::vector<half> h((size_t)3 * 2 * H * 2 * W); CK(cudaMemcpy(h.data(), out, h.size() * 2, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump.c_str(), "wb"); fwrite(h.data(), 2, h.size(), f); fclose(f);
    }
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); launch<WARPS, WR>(m, in, H, W, A, B, out, s);
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
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: compact_native DATADIR W H [iters]\n"); return 2; }
    const std::string d = argv[1]; const int W = atoi(argv[2]), H = atoi(argv[3]), iters = argc > 4 ? atoi(argv[4]) : 500;
    if (W % TW) { printf("width must be a multiple of %d\n", TW); return 2; }
    int c = 0, nconv = 0; { FILE* f = fopen((d + "/model.txt").c_str(), "r"); if (!f || fscanf(f, "%d %d", &c, &nconv) != 2) return 1; fclose(f); }
    if (c != C || nconv < 3 || nconv - 2 > 32) { printf("model C=%d convs=%d not supported by this build\n", c, nconv); return 2; }
    auto wb = slurp(d + "/weights.bin"); size_t o = 0;
    auto take = [&](int cin, int cin_pad, int cout, int cout_pad, bool prelu) {
        const half* w = reinterpret_cast<const half*>(wb.data() + o); o += (size_t)9 * cin * cout * 2;
        const float* b = reinterpret_cast<const float*>(wb.data() + o); o += (size_t)cout * 4;
        Packed p = pack(w, b, 9, cin, cin_pad, cout, cout_pad);
        if (prelu) { p.slope = upload_h2(reinterpret_cast<const float*>(wb.data() + o), cout, cout_pad); o += (size_t)cout * 4; }
        return p;
    };
    Model m; m.nbody = nconv - 2;
    m.first = take(3, 16, C, C, true);
    for (int i = 0; i < m.nbody; i++) m.body[i] = take(C, C, C, C, true);
    m.last = take(C, C, 12, 16, false);
    if (o != wb.size()) { printf("weights size mismatch %zu vs %zu\n", o, wb.size()); return 1; }
    auto inb = slurp(d + "/input.bin");
    half* in; CK(cudaMalloc(&in, inb.size())); CK(cudaMemcpy(in, inb.data(), inb.size(), cudaMemcpyHostToDevice));
    half *A, *B, *out; const size_t N = (size_t)H * W * C;
    CK(cudaMalloc(&A, N * 2)); CK(cudaMalloc(&B, N * 2)); CK(cudaMalloc(&out, (size_t)3 * 2 * H * 2 * W * 2));
    run_config<4, 2>("warps4 rows2", m, in, H, W, A, B, out, iters, d + "/native_out.bin");
    run_config<4, 4>("warps4 rows4", m, in, H, W, A, B, out, iters, "");
    run_config<8, 2>("warps8 rows2", m, in, H, W, A, B, out, iters, "");
    run_config<2, 4>("warps2 rows4", m, in, H, W, A, B, out, iters, "");
    return 0;
}
