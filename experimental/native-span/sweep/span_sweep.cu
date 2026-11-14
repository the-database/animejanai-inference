// Generalised fused SPAN-family body: N blocks of conv3x3 -> act -> conv3x3 -> act -> conv3x3 -> attention gate,
// 3 kernels per block, NHWC fp16, WMMA tensor cores with fp16 accumulation.
// Template: C (channels, multiple of 16), ACT (relu/silu), ATT (tanh(1x1 conv) / sigmoid-0.5 / none).
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

using namespace nvcuda;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TW = 16, TH = 16, WARPS = 8, HALO_PX = (TH + 2) * (TW + 2);
enum Act { RELU = 0, SILU = 1 };
enum Att { ATT_TANH1X1 = 0, ATT_SIGMOID = 1, ATT_NONE = 2 };
enum Stage { S_ACT = 0, S_ACT_MID = 1, S_GATE = 2 };   // k1/k2, k1 of the last block, k3

__device__ __forceinline__ float fast_tanh(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float fast_sigmoid(float x) { return 0.5f * fast_tanh(0.5f * x) + 0.5f; }
template <int ACT> __device__ __forceinline__ float act(float x) {
    if constexpr (ACT == RELU) return fmaxf(x, 0.f); else return x * fast_sigmoid(x);
}

template <int C, int STAGE, int ATT>
__host__ __device__ constexpr int stage_bytes() { return 32 * C * 2 + ((STAGE == S_GATE && ATT == ATT_TANH1X1) ? 32 * C * 2 : 0); }
template <int C, int STAGE, int ATT>
__host__ __device__ constexpr int smem_bytes() {
    return HALO_PX * C * 2 > WARPS * stage_bytes<C, STAGE, ATT>() ? HALO_PX * C * 2 : WARPS * stage_bytes<C, STAGE, ATT>();
}

template <int C, int STAGE, int ACT, int ATT>
__global__ void __launch_bounds__(WARPS * 32)
conv3x3_fused(const half* __restrict__ in, const half* __restrict__ resid, half* __restrict__ out, half* __restrict__ mid,
              const half* __restrict__ w, const float* __restrict__ b, const half* __restrict__ watt,
              const float* __restrict__ batt, int H, int W)
{
    constexpr int NB = C / 16, KC = C / 16, VPP = C / 8, CHUNKS = 32 * VPP;
    extern __shared__ __align__(128) unsigned char smem[];
    half* tile = reinterpret_cast<half*>(smem);
    const int tx0 = blockIdx.x * TW, ty0 = blockIdx.y * TH;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, row0 = warp * 2;

    for (int i = tid; i < HALO_PX * VPP; i += WARPS * 32) {
        int p = i / VPP, v = i % VPP;
        int yy = ty0 - 1 + p / (TW + 2), xx = tx0 - 1 + p % (TW + 2);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
            val = __ldg(reinterpret_cast<const uint4*>(in + ((size_t)yy * W + xx) * C) + v);
        reinterpret_cast<uint4*>(tile + p * C)[v] = val;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[2][NB];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int n = 0; n < NB; n++) wmma::fill_fragment(acc[r][n], __float2half(0.f));
#pragma unroll 1
    for (int t = 0; t < 9; t++) {
        const int dy = t / 3, dx = t % 3;
#pragma unroll
        for (int kc = 0; kc < KC; kc++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[NB];
#pragma unroll
            for (int n = 0; n < NB; n++) wmma::load_matrix_sync(bf[n], w + (t * C + kc * 16) * C + n * 16, C);
#pragma unroll
            for (int r = 0; r < 2; r++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::load_matrix_sync(af, tile + ((row0 + r + dy) * (TW + 2) + dx) * C + kc * 16, C);
#pragma unroll
                for (int n = 0; n < NB; n++) wmma::mma_sync(acc[r][n], af, bf[n], acc[r][n]);
            }
        }
    }
    __syncthreads();

    half* st = reinterpret_cast<half*>(smem + warp * stage_bytes<C, STAGE, ATT>());
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int n = 0; n < NB; n++) wmma::store_matrix_sync(st + (r * 16) * C + n * 16, acc[r][n], C, wmma::mem_row_major);
    __syncwarp();

    if constexpr (STAGE != S_GATE) {
#pragma unroll
        for (int q = lane; q < CHUNKS; q += 32) {
            const int px = q / VPP, v = q % VPP, oy = ty0 + row0 + (px >> 4), ox = tx0 + (px & 15);
            if (oy >= H || ox >= W) continue;
            const half* sp = st + px * C + v * 8;
            __align__(16) half o[8], m[8];
#pragma unroll
            for (int j = 0; j < 8; j++) { float x = __half2float(sp[j]) + __ldg(b + v * 8 + j); m[j] = __float2half(x); o[j] = __float2half(act<ACT>(x)); }
            const size_t g = ((size_t)oy * W + ox) * C;
            reinterpret_cast<uint4*>(out + g)[v] = *reinterpret_cast<uint4*>(o);
            if constexpr (STAGE == S_ACT_MID) reinterpret_cast<uint4*>(mid + g)[v] = *reinterpret_cast<uint4*>(m);
        }
    } else {
        // o3 = acc + b (in place, half)
#pragma unroll
        for (int q = lane; q < CHUNKS; q += 32) {
            const int px = q / VPP, v = q % VPP;
            half* sp = st + px * C + v * 8;
#pragma unroll
            for (int j = 0; j < 8; j++) sp[j] = __float2half(__half2float(sp[j]) + __ldg(b + v * 8 + j));
        }
        __syncwarp();
        const half* gate_src = st;  // per-element pre-activation of the gate
        if constexpr (ATT == ATT_TANH1X1) {
            half* ga = st + 32 * C;
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> aacc[2][NB];
#pragma unroll
            for (int r = 0; r < 2; r++)
#pragma unroll
                for (int n = 0; n < NB; n++) wmma::fill_fragment(aacc[r][n], __float2half(0.f));
#pragma unroll
            for (int kc = 0; kc < KC; kc++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[NB];
#pragma unroll
                for (int n = 0; n < NB; n++) wmma::load_matrix_sync(bf[n], watt + (kc * 16) * C + n * 16, C);
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                    wmma::load_matrix_sync(af, st + (r * 16) * C + kc * 16, C);
#pragma unroll
                    for (int n = 0; n < NB; n++) wmma::mma_sync(aacc[r][n], af, bf[n], aacc[r][n]);
                }
            }
#pragma unroll
            for (int r = 0; r < 2; r++)
#pragma unroll
                for (int n = 0; n < NB; n++) wmma::store_matrix_sync(ga + (r * 16) * C + n * 16, aacc[r][n], C, wmma::mem_row_major);
            __syncwarp();
            gate_src = ga;
        }
#pragma unroll
        for (int q = lane; q < CHUNKS; q += 32) {
            const int px = q / VPP, v = q % VPP, oy = ty0 + row0 + (px >> 4), ox = tx0 + (px & 15);
            if (oy >= H || ox >= W) continue;
            const size_t g = ((size_t)oy * W + ox) * C;
            uint4 xr = __ldg(reinterpret_cast<const uint4*>(resid + g) + v);
            uint4 o3r = *reinterpret_cast<const uint4*>(st + px * C + v * 8);
            uint4 gr = *reinterpret_cast<const uint4*>(gate_src + px * C + v * 8);
            const half *xh = reinterpret_cast<const half*>(&xr), *oh = reinterpret_cast<const half*>(&o3r), *gh = reinterpret_cast<const half*>(&gr);
            __align__(16) half o[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float s = __half2float(xh[j]) + __half2float(oh[j]);
                float gate;
                if constexpr (ATT == ATT_TANH1X1) gate = fast_tanh(__half2float(gh[j]) + __ldg(batt + v * 8 + j));
                else if constexpr (ATT == ATT_SIGMOID) gate = fast_sigmoid(__half2float(gh[j])) - 0.5f;
                else gate = 1.f;
                o[j] = __float2half(s * gate);
            }
            reinterpret_cast<uint4*>(out + g)[v] = *reinterpret_cast<uint4*>(o);
        }
    }
}

struct BlockW { const half *w1, *w2, *w3, *watt; const float *b1, *b2, *b3, *batt; };

template <int C, int ACT, int ATT>
static void run_body(const BlockW* bw, int nb, half* const* x, half* a1, half* a2, half* mid, int H, int W, cudaStream_t s) {
    dim3 grid((W + TW - 1) / TW, (H + TH - 1) / TH), blk(WARPS * 32);
    for (int i = 0; i < nb; i++) {
        // ReLU blocks use nn.ReLU(inplace=True), so the exported last_mid is relu(out1) == a1: no extra store.
        if (i == nb - 1 && ACT != RELU) conv3x3_fused<C, S_ACT_MID, ACT, ATT><<<grid, blk, smem_bytes<C, S_ACT_MID, ATT>(), s>>>(x[i], nullptr, a1, mid, bw[i].w1, bw[i].b1, nullptr, nullptr, H, W);
        else             conv3x3_fused<C, S_ACT, ACT, ATT><<<grid, blk, smem_bytes<C, S_ACT, ATT>(), s>>>(x[i], nullptr, a1, nullptr, bw[i].w1, bw[i].b1, nullptr, nullptr, H, W);
        conv3x3_fused<C, S_ACT, ACT, ATT><<<grid, blk, smem_bytes<C, S_ACT, ATT>(), s>>>(a1, nullptr, a2, nullptr, bw[i].w2, bw[i].b2, nullptr, nullptr, H, W);
        conv3x3_fused<C, S_GATE, ACT, ATT><<<grid, blk, smem_bytes<C, S_GATE, ATT>(), s>>>(a2, x[i], x[i + 1], nullptr, bw[i].w3, bw[i].b3, bw[i].watt, bw[i].batt, H, W);
    }
}

static std::vector<char> slurp(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); fread(v.data(), 1, n, f); fclose(f); return v;
}

template <int C, int ACT, int ATT>
static int go(const std::string& d, int nb, int H, int W, int iters) {
    for (int st : {S_ACT, S_ACT_MID}) (void)st;
    CK(cudaFuncSetAttribute(conv3x3_fused<C, S_ACT, ACT, ATT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<C, S_ACT, ATT>()));
    CK(cudaFuncSetAttribute(conv3x3_fused<C, S_ACT_MID, ACT, ATT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<C, S_ACT_MID, ATT>()));
    CK(cudaFuncSetAttribute(conv3x3_fused<C, S_GATE, ACT, ATT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<C, S_GATE, ATT>()));
    const size_t N = (size_t)H * W * C;
    auto wbuf = slurp(d + "/weights.bin"); auto feat = slurp(d + "/feat.bin");
    char* dw; CK(cudaMalloc(&dw, wbuf.size())); CK(cudaMemcpy(dw, wbuf.data(), wbuf.size(), cudaMemcpyHostToDevice));
    std::vector<BlockW> bw(nb); size_t o = 0;
    auto take = [&](size_t bytes) { char* p = dw + o; o += bytes; return p; };
    for (int i = 0; i < nb; i++) {
        bw[i].w1 = (half*)take(9 * C * C * 2); bw[i].b1 = (float*)take(C * 4);
        bw[i].w2 = (half*)take(9 * C * C * 2); bw[i].b2 = (float*)take(C * 4);
        bw[i].w3 = (half*)take(9 * C * C * 2); bw[i].b3 = (float*)take(C * 4);
        if (ATT == ATT_TANH1X1) { bw[i].watt = (half*)take(C * C * 2); bw[i].batt = (float*)take(C * 4); }
        else { bw[i].watt = nullptr; bw[i].batt = nullptr; }
    }
    if (o != wbuf.size() || feat.size() != N * 2) { printf("size mismatch: weights %zu/%zu feat %zu/%zu\n", o, wbuf.size(), feat.size(), N * 2); return 1; }
    std::vector<half*> x(nb + 1); half *a1, *a2, *mid;
    for (auto& p : x) CK(cudaMalloc(&p, N * 2));
    CK(cudaMalloc(&a1, N * 2)); CK(cudaMalloc(&a2, N * 2)); CK(cudaMalloc(&mid, N * 2));
    CK(cudaMemcpy(x[0], feat.data(), N * 2, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    run_body<C, ACT, ATT>(bw.data(), nb, x.data(), a1, a2, mid, H, W, s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
    std::vector<half> h(N);
    auto dump = [&](half* p, const char* name) {
        CK(cudaMemcpy(h.data(), p, N * 2, cudaMemcpyDeviceToHost));
        FILE* f = fopen((d + "/proto_" + name + ".bin").c_str(), "wb"); fwrite(h.data(), 2, N, f); fclose(f);
    };
    dump(x[1], "outb1"); dump(x[nb], "outlast"); dump(ACT == RELU ? a1 : mid, "midlast");
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    run_body<C, ACT, ATT>(bw.data(), nb, x.data(), a1, a2, mid, H, W, s);
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 200; i++) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int rep = 0; rep < 5; rep++) {
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < iters; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t / iters);
    }
    std::sort(ms.begin(), ms.end());
    printf("fused body: %.3f ms median [min %.3f max %.3f]\n", ms[2], ms[0], ms[4]);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 8) { printf("usage: span_sweep DIR C BLOCKS ACT(relu|silu) ATT(tanh1x1|sigmoid|none) H W [iters]\n"); return 2; }
    std::string d = argv[1]; int C = atoi(argv[2]), nb = atoi(argv[3]), H = atoi(argv[6]), W = atoi(argv[7]);
    int iters = argc > 8 ? atoi(argv[8]) : 500;
    int a = strcmp(argv[4], "silu") == 0 ? SILU : RELU;
    int t = strcmp(argv[5], "sigmoid") == 0 ? ATT_SIGMOID : strcmp(argv[5], "none") == 0 ? ATT_NONE : ATT_TANH1X1;
#define CASE(c, aa, tt) if (C == c && a == aa && t == tt) return go<c, aa, tt>(d, nb, H, W, iters);
    CASE(32, RELU, ATT_TANH1X1) CASE(48, RELU, ATT_TANH1X1) CASE(64, RELU, ATT_TANH1X1) CASE(96, RELU, ATT_TANH1X1)
    CASE(48, SILU, ATT_TANH1X1) CASE(48, RELU, ATT_SIGMOID) CASE(48, RELU, ATT_NONE)
    CASE(64, SILU, ATT_TANH1X1) CASE(64, RELU, ATT_SIGMOID) CASE(64, RELU, ATT_NONE)
    printf("unsupported combination\n"); return 2;
}
