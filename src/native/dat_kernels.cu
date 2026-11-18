// native/dat_kernels.cu — see dat_kernels.h.
#include "native/dat_kernels.h"

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v / 1.4140625f)); }
__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

constexpr int TG_R = 16;

// One warp per token; W1, b1, w2 in shared memory.
__global__ void __launch_bounds__(256) k_token_gate(const half *x, int st, int M, int C, int r, const float *w1, const float *b1,
                                                    const float *w2, float b2, float *g)
{
    extern __shared__ float sw[];
    float *sb1 = sw + r * C, *sw2 = sb1 + r;
    for (int i = threadIdx.x; i < r * C; i += blockDim.x) sw[i] = w1[i];
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
        sb1[i] = b1 ? b1[i] : 0.f;
        sw2[i] = w2[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; t < M; t += gridDim.x * (blockDim.x / 32)) {
        float acc[TG_R];
#pragma unroll
        for (int j = 0; j < TG_R; j++) acc[j] = 0.f;
        const half *xt = x + (size_t)t * st;
        for (int c = lane; c < C; c += 32) {
            const float v = __half2float(xt[c]);
#pragma unroll
            for (int j = 0; j < TG_R; j++)
                if (j < r) acc[j] += v * sw[j * C + c];
        }
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < TG_R; j++) {
            if (j >= r) break;
            float a = acc[j];
#pragma unroll
            for (int k = 16; k; k >>= 1) a += __shfl_xor_sync(0xffffffffu, a, k);
            o += sw2[j] * gelu(a + sb1[j]);
        }
        if (lane == 0) g[t] = sigm(o + b2);
    }
}

__global__ void k_mix(const half *a, const half *b, int st, const float *cg, const float *tg, int mode, half *z, int M, int C)
{
    const int pairs = (C + 15) / 16 * 8;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * pairs) return;
    const int t = (int)(i / pairs), c = (int)(i % pairs) * 2;
    float2 o = make_float2(0.f, 0.f);
    if (c < C) {
        const float2 va = __half22float2(*reinterpret_cast<const half2 *>(a + (size_t)t * st + c));
        const float2 vb = __half22float2(*reinterpret_cast<const half2 *>(b + (size_t)t * st + c));
        const float k = tg[t], c0 = cg[c], c1 = c + 1 < C ? cg[c + 1] : 0.f;
        if (mode == 0) {
            o.x = va.x * c0 + vb.x * k;
            o.y = c + 1 < C ? va.y * c1 + vb.y * k : 0.f;
        } else {
            o.x = va.x * k + vb.x * c0;
            o.y = c + 1 < C ? va.y * k + vb.y * c1 : 0.f;
        }
    }
    *reinterpret_cast<half2 *>(z + (size_t)t * st + c) = __floats2half2_rn(o.x, o.y);
}

// The token gate and the mix in one pass, a thread per token: g = sigm(w2 . gelu(W1 x_src + b1) + b2)
// over the token's gate source row (channel order, the weights broadcast from shared memory), then
// z = a * cg + b * g (mode 0, the source a) or a * g + b * cg (mode 1, the source b), zeros to pad16(C).
__global__ void __launch_bounds__(128) k_gate_mix(const half *a, const half *b, int st, const float *cg, int M, int C, int r,
                                                  const float *w1, const float *b1, const float *w2, float b2, int mode, half *z)
{
    extern __shared__ float sw[];
    float *sb1 = sw + r * C, *sw2 = sb1 + r;
    for (int i = threadIdx.x; i < r * C; i += blockDim.x) sw[i] = w1[i];
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
        sb1[i] = b1 ? b1[i] : 0.f;
        sw2[i] = w2[i];
    }
    __syncthreads();
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M) return;
    const half *xa = a + (size_t)t * st, *xb = b + (size_t)t * st, *src = mode ? xb : xa;
    float acc[TG_R];
#pragma unroll
    for (int j = 0; j < TG_R; j++) acc[j] = 0.f;
    for (int c0 = 0; c0 < C; c0 += 8) {
        float v[8];
        const uint4 u = *reinterpret_cast<const uint4 *>(src + c0);
        const half2 *hp = reinterpret_cast<const half2 *>(&u);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f = __half22float2(hp[i]);
            v[2 * i] = f.x;
            v[2 * i + 1] = f.y;
        }
        const int n = min(8, C - c0);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i >= n) break;
#pragma unroll
            for (int j = 0; j < TG_R; j++)
                if (j < r) acc[j] += v[i] * sw[j * C + c0 + i];
        }
    }
    float o = 0.f;
#pragma unroll
    for (int j = 0; j < TG_R; j++) {
        if (j >= r) break;
        o += sw2[j] * gelu(acc[j] + sb1[j]);
    }
    const float k = sigm(o + b2);
    const int cp = (C + 15) / 16 * 16;
    for (int c0 = 0; c0 < cp; c0 += 8) {
        const uint4 ua = *reinterpret_cast<const uint4 *>(xa + c0), ub = *reinterpret_cast<const uint4 *>(xb + c0);
        const half2 *ha = reinterpret_cast<const half2 *>(&ua), *hb = reinterpret_cast<const half2 *>(&ub);
        uint4 uo;
        half2 *ho = reinterpret_cast<half2 *>(&uo);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int c = c0 + 2 * i;
            const float2 fa = __half22float2(ha[i]), fb = __half22float2(hb[i]);
            float2 q = make_float2(0.f, 0.f);
            if (c < C) {
                const float g0 = cg[c], g1 = c + 1 < C ? cg[c + 1] : 0.f;
                if (mode == 0) {
                    q.x = fa.x * g0 + fb.x * k;
                    q.y = c + 1 < C ? fa.y * g1 + fb.y * k : 0.f;
                } else {
                    q.x = fa.x * k + fb.x * g0;
                    q.y = c + 1 < C ? fa.y * k + fb.y * g1 : 0.f;
                }
            }
            ho[i] = __floats2half2_rn(q.x, q.y);
        }
        *reinterpret_cast<uint4 *>(z + (size_t)t * st + c0) = uo;
    }
}

}  // namespace

cudaError_t dat_token_gate(const void *x, int st, int M, int C, int r, const float *w1, const float *b1, const float *w2, float b2,
                           float *g, cudaStream_t s)
{
    if (r > TG_R) return cudaErrorInvalidValue;
    const int smem = (r * C + 2 * r) * 4;
    const unsigned grid = (unsigned)std::min<long long>((M + 7) / 8, 4096);
    k_token_gate<<<grid, 256, smem, s>>>((const half *)x, st, M, C, r, w1, b1, w2, b2, g);
    return cudaGetLastError();
}

cudaError_t dat_gate_mix(const void *a, const void *b, int st, const float *cg, int M, int C, int r, const float *w1, const float *b1,
                         const float *w2, float b2, int mode, void *z, cudaStream_t s)
{
    if (r > TG_R || st % 8 || (C + 15) / 16 * 16 > st || ((uintptr_t)a | (uintptr_t)b | (uintptr_t)z) % 16) return cudaErrorNotSupported;
    const int smem = (r * C + 2 * r) * 4;
    k_gate_mix<<<(unsigned)((M + 127) / 128), 128, smem, s>>>((const half *)a, (const half *)b, st, cg, M, C, r, w1, b1, w2, b2, mode,
                                                             (half *)z);
    return cudaGetLastError();
}

cudaError_t dat_mix(const void *a, const void *b, int st, const float *cg, const float *tg, int mode, void *z, int M, int C, cudaStream_t s)
{
    if ((C + 15) / 16 * 16 > st) return cudaErrorInvalidValue;
    const size_t n = (size_t)M * ((C + 15) / 16 * 8);
    k_mix<<<(unsigned)((n + 255) / 256), 256, 0, s>>>((const half *)a, (const half *)b, st, cg, tg, mode, (half *)z, M, C);
    return cudaGetLastError();
}

}  // namespace ajinative
