// native/dat_kernels.cu — see dat_kernels.h.
#include "native/dat_kernels.h"

#include <algorithm>

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

cudaError_t dat_mix(const void *a, const void *b, int st, const float *cg, const float *tg, int mode, void *z, int M, int C, cudaStream_t s)
{
    if ((C + 15) / 16 * 16 > st) return cudaErrorInvalidValue;
    const size_t n = (size_t)M * ((C + 15) / 16 * 8);
    k_mix<<<(unsigned)((n + 255) / 256), 256, 0, s>>>((const half *)a, (const half *)b, st, cg, tg, mode, (half *)z, M, C);
    return cudaGetLastError();
}

}  // namespace ajinative
