// Shared native conv kernels (mma.sync m16n8k16, fp16 accumulation, fragment-packed weights).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
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

enum InMode { IN_NHWC = 0, IN_NCHW_UNSHUFFLE2 = 1, IN_NCHW3 = 2 };
// E_PRELU: per-channel slope passed in `batt`. E_D2S2_ADDIN: COUT=16 (3x4 used), depth-to-space(2) + nearest-2x
// copy of the NCHW input (`resid`) added, written as NCHW 3 x 2H x 2W.
enum EpiMode { E_BIAS = 0, E_RELU = 1, E_GATE = 2, E_D2S4 = 3, E_PRELU = 4, E_D2S2_ADDIN = 5 };
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
    } else if constexpr (INM == IN_NCHW3) {
        static_assert(CIN == 16, "IN_NCHW3 pads 3 channels to 16");
        for (int p = tid; p < HP; p += WARPS * 32) {
            const int yy = ty0 - 1 + p / (TW + 2), xx = tx0 - 1 + p % (TW + 2);
            __align__(16) half v[16];
#pragma unroll
            for (int c = 0; c < 16; c++) v[c] = __float2half(0.f);
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
#pragma unroll
                for (int c = 0; c < 3; c++) v[c] = in[((size_t)c * H + yy) * W + xx];
            }
            reinterpret_cast<uint4*>(sm + p * CP)[0] = reinterpret_cast<uint4*>(v)[0];
            reinterpret_cast<uint4*>(sm + p * CP)[1] = reinterpret_cast<uint4*>(v)[1];
        }
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
                half2 slope = __float2half2_rn(0.f);
                if constexpr (EPI == E_PRELU) slope = __ldg(batt + n * 4 + tq);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    half2 v = __hadd2(u2h(acc[i][n][h]), bb);
                    if constexpr (EPI == E_RELU) v = __hmax2(v, __float2half2_rn(0.f));
                    if constexpr (EPI == E_PRELU) {
                        const half2 z = __float2half2_rn(0.f);
                        v = __hfma2(__hmin2(v, z), slope, __hmax2(v, z));
                    }
                    *reinterpret_cast<half2*>(sm + (r * TW + g + h * 8) * SP + n * 8 + tq * 2) = v;
                }
            }
        }
    }
    __syncthreads();

    if constexpr (EPI == E_D2S2_ADDIN) {
        // depth-to-space(2): channel c*4 + ry*2 + rx -> out[c][2y+ry][2x+rx] (+ nearest-2x input); 32 halfs per output row
        static_assert(COUT == 16, "D2S2 epilogue expects 3 output channels x 4 padded to 16");
        const int OH = 2 * H, OW = 2 * W;
        for (int q = tid; q < 3 * TH * 2 * 4; q += WARPS * 32) {
            const int k = q & 3, ry = (q >> 2) & 1, r = (q >> 3) % TH, c = (q >> 3) / TH, oy = ty0 + r;
            if (oy >= H) continue;
            const uint2 inr = __ldg(reinterpret_cast<const uint2*>(resid + ((size_t)c * H + oy) * W + tx0 + 4 * k));
            const half* ih = reinterpret_cast<const half*>(&inr);
            __align__(16) half o[8];
#pragma unroll
            for (int e = 0; e < 8; e++)
                o[e] = __hadd(sm[(r * TW + 4 * k + (e >> 1)) * SP + c * 4 + ry * 2 + (e & 1)], ih[e >> 1]);
            *reinterpret_cast<uint4*>(out + ((size_t)c * OH + 2 * oy + ry) * OW + 2 * tx0 + 8 * k) = *reinterpret_cast<uint4*>(o);
        }
    } else if constexpr (EPI != E_D2S4) {
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

