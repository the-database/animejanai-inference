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

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2*>(&u); }
__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* s, const void* g, int bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(a), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ half2 tanh_h2(half2 x) { uint32_t y; asm("tanh.approx.f16x2 %0, %1;" : "=r"(y) : "r"(h2u(x))); return u2h(y); }

enum InMode { IN_NHWC = 0, IN_NCHW_UNSHUFFLE2 = 1 };
enum EpiMode { E_BIAS = 0, E_RELU = 1, E_GATE = 2, E_D2S4 = 3 };
constexpr int TW = 16;

template <int CIN, int COUT, int WARPS, int WR>
__host__ __device__ constexpr int conv_smem() {
    constexpr int TH = WARPS * WR, halo = (TH + 2) * (TW + 2) * (CIN + 8) * 2, stage = TH * TW * (COUT + 8) * 2;
    return halo > stage ? halo : stage;
}

// 3x3 conv, stride 1, pad 1, at resolution H x W (NHWC in/out unless noted)
template <int CIN, int COUT, int WARPS, int WR, int INM, int EPI>
__global__ void __launch_bounds__(WARPS * 32)
conv3x3(const half* __restrict__ in, int H, int W, const uint4* __restrict__ wp, const half2* __restrict__ bias,
        const uint4* __restrict__ wpatt, const half2* __restrict__ batt, const half* __restrict__ resid,
        half* __restrict__ out)
{
    constexpr int TH = WARPS * WR, CP = CIN + 8, KC = CIN / 16, NT = COUT / 8, NTP = NT / 2, HP = (TH + 2) * (TW + 2);
    constexpr int SP = COUT + 8;
    extern __shared__ __align__(16) unsigned char smraw[];
    half* sm = reinterpret_cast<half*>(smraw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tx0 = blockIdx.x * TW, ty0 = blockIdx.y * TH;

    // ---- halo tile -> shared [(TH+2)*(TW+2)][CP]
    if constexpr (INM == IN_NHWC) {
        constexpr int VPP = CIN / 8;
        for (int i = tid; i < HP * VPP; i += WARPS * 32) {
            const int p = i / VPP, v = i % VPP;
            const int yy = ty0 - 1 + p / (TW + 2), xx = tx0 - 1 + p % (TW + 2);
            const bool ok = yy >= 0 && yy < H && xx >= 0 && xx < W;
            cp_async16(sm + p * CP + v * 8, ok ? in + ((size_t)yy * W + xx) * CIN + v * 8 : in, ok ? 16 : 0);
        }
        cp_async_wait_all();
    } else {
        // full-res NCHW RGB (3 x 2H x 2W) -> unshuffled channel c*4 + ry*2 + rx, channels 12..15 = 0
        for (int i = tid; i < HP * 6; i += WARPS * 32) {
            const int p = i / 6, k = i % 6, c = k >> 1, ry = k & 1;
            const int yy = ty0 - 1 + p / (TW + 2), xx = tx0 - 1 + p % (TW + 2);
            half2 v = __float2half2_rn(0.f);
            if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                v = __ldg(reinterpret_cast<const half2*>(in + ((size_t)c * (2 * H) + 2 * yy + ry) * (2 * W) + 2 * xx));
            reinterpret_cast<half2*>(sm + p * CP)[c * 2 + ry] = v;
        }
        for (int p = tid; p < HP; p += WARPS * 32) *reinterpret_cast<uint2*>(sm + p * CP + 12) = make_uint2(0, 0);
    }
    __syncthreads();

    // ---- implicit GEMM: warp = WR output rows x 16 px, all COUT channels
    uint32_t acc[WR][NT][2];
#pragma unroll
    for (int i = 0; i < WR; i++)
#pragma unroll
        for (int n = 0; n < NT; n++) acc[i][n][0] = acc[i][n][1] = 0u;
    const int row0 = warp * WR;
#pragma unroll 1
    for (int t = 0; t < 9; t++) {
        const int dy = t / 3, dx = t % 3;
#pragma unroll
        for (int kc = 0; kc < KC; kc++) {
            uint4 b[NTP];
#pragma unroll
            for (int j = 0; j < NTP; j++) b[j] = __ldg(wp + ((t * KC + kc) * NTP + j) * 32 + lane);
#pragma unroll
            for (int i = 0; i < WR; i++) {
                uint32_t a[4];
                ldsm_x4(a, sm + ((row0 + i + dy) * (TW + 2) + (lane & 15) + dx) * CP + kc * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < NTP; j++) { mma16816(acc[i][2 * j], a, b[j].x, b[j].y); mma16816(acc[i][2 * j + 1], a, b[j].z, b[j].w); }
            }
        }
    }
    __syncthreads();  // halo no longer read: reuse shared memory as the output staging tile [TH*16][SP]

    const int g = lane >> 2, tq = lane & 3;
    if constexpr (EPI == E_GATE) {
        uint32_t o3[WR][NT][2], ga[WR][NT][2];
#pragma unroll
        for (int i = 0; i < WR; i++)
#pragma unroll
            for (int n = 0; n < NT; n++) {
                const half2 bb = __ldg(bias + n * 4 + tq);
                o3[i][n][0] = h2u(__hadd2(u2h(acc[i][n][0]), bb)); o3[i][n][1] = h2u(__hadd2(u2h(acc[i][n][1]), bb));
                ga[i][n][0] = ga[i][n][1] = 0u;
            }
        // 1x1 attention GEMM straight from the accumulator registers (m16n8 C layout == m16k16 A layout)
#pragma unroll
        for (int kc = 0; kc < COUT / 16; kc++) {
            uint4 b[NTP];
#pragma unroll
            for (int j = 0; j < NTP; j++) b[j] = __ldg(wpatt + (kc * NTP + j) * 32 + lane);
#pragma unroll
            for (int i = 0; i < WR; i++) {
                const uint32_t a[4] = {o3[i][2 * kc][0], o3[i][2 * kc][1], o3[i][2 * kc + 1][0], o3[i][2 * kc + 1][1]};
#pragma unroll
                for (int j = 0; j < NTP; j++) { mma16816(ga[i][2 * j], a, b[j].x, b[j].y); mma16816(ga[i][2 * j + 1], a, b[j].z, b[j].w); }
            }
        }
#pragma unroll
        for (int i = 0; i < WR; i++) {
            const int r = row0 + i, oy = ty0 + r;
#pragma unroll
            for (int n = 0; n < NT; n++) {
                const half2 bb = __ldg(batt + n * 4 + tq);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int px = g + h * 8;
                    half2 x = __float2half2_rn(0.f);
                    if (oy < H) x = __ldg(reinterpret_cast<const half2*>(resid + ((size_t)oy * W + tx0 + px) * COUT + n * 8 + tq * 2));
                    const half2 gate = tanh_h2(__hadd2(u2h(ga[i][n][h]), bb));
                    *reinterpret_cast<half2*>(sm + (r * TW + px) * SP + n * 8 + tq * 2) = __hmul2(__hadd2(x, u2h(o3[i][n][h])), gate);
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < WR; i++) {
            const int r = row0 + i;
#pragma unroll
            for (int n = 0; n < NT; n++) {
                const half2 bb = __ldg(bias + n * 4 + tq);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    half2 v = __hadd2(u2h(acc[i][n][h]), bb);
                    if constexpr (EPI == E_RELU) v = __hmax2(v, __float2half2_rn(0.f));
                    *reinterpret_cast<half2*>(sm + (r * TW + g + h * 8) * SP + n * 8 + tq * 2) = v;
                }
            }
        }
    }
    __syncthreads();

    if constexpr (EPI != E_D2S4) {
        // coalesced NHWC store: each tile row is TW*COUT contiguous halfs
        constexpr int VPR = TW * COUT / 8;
        for (int q = tid; q < TH * VPR; q += WARPS * 32) {
            const int r = q / VPR, w = q % VPR, px = w / (COUT / 8), v = w % (COUT / 8), oy = ty0 + r;
            if (oy >= H) continue;
            *reinterpret_cast<uint4*>(out + ((size_t)oy * W + tx0 + px) * COUT + v * 8) =
                *reinterpret_cast<const uint4*>(sm + (r * TW + px) * SP + v * 8);
        }
    } else {
        // depth-to-space(4): channel c*16 + ry*4 + rx -> NCHW out[c][4y+ry][4x+rx]; 64 contiguous halfs per output row
        static_assert(COUT == 48, "D2S4 epilogue expects 3 output channels x 16");
        const int OH = 4 * H, OW = 4 * W;
        for (int q = tid; q < 3 * TH * 4 * 8; q += WARPS * 32) {
            const int k = q & 7, ry = (q >> 3) & 3, r = (q >> 5) % TH, c = (q >> 5) / TH, oy = ty0 + r;
            if (oy >= H) continue;
            __align__(16) half o[8];
#pragma unroll
            for (int e = 0; e < 8; e++) o[e] = sm[(r * TW + 2 * k + (e >> 2)) * SP + c * 16 + ry * 4 + (e & 3)];
            *reinterpret_cast<uint4*>(out + ((size_t)c * OH + 4 * oy + ry) * OW + 4 * tx0 + 8 * k) = *reinterpret_cast<uint4*>(o);
        }
    }
}

// conv_cat: 1x1 conv over the virtual concat [s0, s1, s2, s3] (each C channels, NHWC) -> C channels
template <int C, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
convcat(const half* __restrict__ s0, const half* __restrict__ s1, const half* __restrict__ s2, const half* __restrict__ s3,
        int npx, const uint4* __restrict__ wp, const half2* __restrict__ bias, half* __restrict__ out)
{
    constexpr int M = WARPS * 16, CP = C + 8, NT = C / 8, NTP = NT / 2, KCS = C / 16, VPP = C / 8;
    extern __shared__ __align__(16) unsigned char smraw[];
    half* sm = reinterpret_cast<half*>(smraw);   // [4][M][CP]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, p0 = blockIdx.x * M;
    const half* src[4] = {s0, s1, s2, s3};
#pragma unroll
    for (int s = 0; s < 4; s++)
        for (int i = tid; i < M * VPP; i += WARPS * 32) {
            const int p = i / VPP, v = i % VPP; const bool ok = p0 + p < npx;
            cp_async16(sm + (s * M + p) * CP + v * 8, ok ? src[s] + (size_t)(p0 + p) * C + v * 8 : s0, ok ? 16 : 0);
        }
    cp_async_wait_all();
    __syncthreads();
    uint32_t acc[NT][2];
#pragma unroll
    for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = 0u;
#pragma unroll
    for (int s = 0; s < 4; s++)
#pragma unroll
        for (int kc = 0; kc < KCS; kc++) {
            uint32_t a[4];
            ldsm_x4(a, sm + (s * M + warp * 16 + (lane & 15)) * CP + kc * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int j = 0; j < NTP; j++) {
                const uint4 b = __ldg(wp + ((s * KCS + kc) * NTP + j) * 32 + lane);
                mma16816(acc[2 * j], a, b.x, b.y); mma16816(acc[2 * j + 1], a, b.z, b.w);
            }
        }
    __syncthreads();
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        const half2 bb = __ldg(bias + n * 4 + tq);
#pragma unroll
        for (int h = 0; h < 2; h++)
            *reinterpret_cast<half2*>(sm + (warp * 16 + g + h * 8) * CP + n * 8 + tq * 2) = __hadd2(u2h(acc[n][h]), bb);
    }
    __syncthreads();
    for (int i = tid; i < M * VPP; i += WARPS * 32) {
        const int p = i / VPP, v = i % VPP;
        if (p0 + p < npx) *reinterpret_cast<uint4*>(out + (size_t)(p0 + p) * C + v * 8) = *reinterpret_cast<const uint4*>(sm + p * CP + v * 8);
    }
}

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
