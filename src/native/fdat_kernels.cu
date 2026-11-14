// native/fdat_kernels.cu — FDAT's small kernels (see fdat_kernels.h).
#include "native/fdat_kernels.h"

#include <cstdint>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v / 1.4140625f)); }
__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

// attn[h][d][e] = softmax_e(gram[h][d][e] / (max(|q_d|, eps) max(|k_e|, eps)) * temp[h]); one thread per row (h, d).
__global__ void k_chan_soft(const float *stats, const float *temp, int heads, int hd, float eps, float *attn)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * hd) return;
    const int h = i / hd, d = i % hd, C = heads * hd;
    const float *gram = stats + (size_t)i * hd, *nq = stats + heads * hd * hd, *nk = nq + C;
    const float qn = fmaxf(sqrtf(nq[h * hd + d]), eps);
    float v[64], mx = -1e30f;
    for (int e = 0; e < hd; e++) {
        v[e] = gram[e] / (qn * fmaxf(sqrtf(nk[h * hd + e]), eps)) * temp[h];
        mx = fmaxf(mx, v[e]);
    }
    float s = 0.f;
    for (int e = 0; e < hd; e++) s += (v[e] = __expf(v[e] - mx));
    for (int e = 0; e < hd; e++) attn[(size_t)i * hd + e] = v[e] / s;
}

// fold[h*hd + e][n] = sum_d attn[h][d][e] * wp[h*hd + d][n]: the per-head mixing followed by the projection.
__global__ void k_chan_fold(const float *attn, const float *wp, int heads, int hd, float *fold)
{
    const int C = heads * hd;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)C * C) return;
    const int k = (int)(i / C), n = (int)(i % C), h = k / hd, e = k % hd;
    float s = 0.f;
    for (int d = 0; d < hd; d++) s += attn[((size_t)h * hd + d) * hd + e] * wp[(size_t)(h * hd + d) * C + n];
    fold[i] = s;
}

// cm = sigmoid(W2 gelu(W1 mean)), mean = sums / M; W1 [r][C], W2 [C][r]. One CTA.
__global__ void k_aim_cg(const float *sums, int M, int C, int r, const float *w1, const float *w2, float *cm, int cpad)
{
    extern __shared__ float sh[];
    float *mean = sh, *hid = sh + C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) mean[c] = sums[c] / M;
    __syncthreads();
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        float s = 0.f;
        for (int c = 0; c < C; c++) s += w1[j * C + c] * mean[c];
        hid[j] = gelu(s);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < cpad; c += blockDim.x) {
        float s = 0.f;
        if (c < C)
            for (int j = 0; j < r; j++) s += w2[c * r + j] * hid[j];
        cm[c] = c < C ? sigm(s) : 0.f;
    }
}

// x[t] += attn[t] + conv[t] * sigmoid(attn[t] . wsg); one warp per token.
__global__ void k_aim_sg(const half *attn, const half *conv, const float *wsg, half *x, int st, int M, int C)
{
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= M) return;
    const half *a = attn + (size_t)t * st, *cv = conv + (size_t)t * st;
    half *xo = x + (size_t)t * st;
    float s = 0.f;
    for (int c = lane; c < C; c += 32) s += __half2float(a[c]) * wsg[c];
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float g = sigm(s);
    for (int c = lane; c < C; c += 32) xo[c] = __float2half(__half2float(xo[c]) + __half2float(a[c]) + __half2float(cv[c]) * g);
}

// The same with L lanes per token, each one uint4 of channels (C <= 8 L, strides multiples of 8).
template <int L>
__global__ void __launch_bounds__(256) k_aim_sg8(const half *attn, const half *conv, const float *wsg, half *x, int st, int M, int C)
{
    const int sub = threadIdx.x % L, t = (blockIdx.x * blockDim.x + threadIdx.x) / L, c = sub * 8;
    const bool live = t < M && c < C;
    float a[8] = {}, cv[8] = {}, xv[8] = {};
    const size_t o = (size_t)t * st + c;
    if (live) {
        const uint4 ua = *reinterpret_cast<const uint4 *>(attn + o), uc = *reinterpret_cast<const uint4 *>(conv + o),
                    ux = *reinterpret_cast<const uint4 *>(x + o);
        const half2 *ha = reinterpret_cast<const half2 *>(&ua), *hc = reinterpret_cast<const half2 *>(&uc),
                    *hx = reinterpret_cast<const half2 *>(&ux);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 fa = __half22float2(ha[j]), fc = __half22float2(hc[j]), fx = __half22float2(hx[j]);
            a[2 * j] = fa.x; a[2 * j + 1] = fa.y;
            cv[2 * j] = fc.x; cv[2 * j + 1] = fc.y;
            xv[2 * j] = fx.x; xv[2 * j + 1] = fx.y;
        }
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++)
        if (live && c + j < C) s += a[j] * wsg[c + j];
#pragma unroll
    for (int k = L / 2; k; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    if (!live) return;
    const float g = sigm(s);
    uint4 u;
    uint32_t *pu = reinterpret_cast<uint32_t *>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        half2 h = __floats2half2_rn(xv[2 * j] + a[2 * j] + cv[2 * j] * g, xv[2 * j + 1] + a[2 * j + 1] + cv[2 * j + 1] * g);
        pu[j] = *reinterpret_cast<uint32_t *>(&h);
    }
    *reinterpret_cast<uint4 *>(x + o) = u;
}

}  // namespace

cudaError_t fdat_chan_soft(const float *stats, const float *temp, int heads, int hd, float eps, float *attn, cudaStream_t s)
{
    if (hd > 64) return cudaErrorInvalidValue;
    k_chan_soft<<<(heads * hd + 63) / 64, 64, 0, s>>>(stats, temp, heads, hd, eps, attn);
    return cudaGetLastError();
}

cudaError_t fdat_chan_fold(const float *attn, const float *wp, int heads, int hd, float *fold, cudaStream_t s)
{
    const size_t n = (size_t)heads * hd * heads * hd;
    k_chan_fold<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(attn, wp, heads, hd, fold);
    return cudaGetLastError();
}

cudaError_t fdat_aim_cg(const float *sums, int M, int C, int r, const float *w1, const float *w2, float *cm, int cpad, cudaStream_t s)
{
    k_aim_cg<<<1, 128, (C + r) * sizeof(float), s>>>(sums, M, C, r, w1, w2, cm, cpad);
    return cudaGetLastError();
}

cudaError_t fdat_aim_sg(const void *attn, const void *conv, const float *wsg, void *x, int st, int M, int C, cudaStream_t s)
{
    const bool vec = C % 8 == 0 && st % 8 == 0 && C <= 256 && ((uintptr_t)attn | (uintptr_t)conv | (uintptr_t)x) % 16 == 0;
    if (vec && C <= 128)
        k_aim_sg8<16><<<(M + 15) / 16, 256, 0, s>>>((const half *)attn, (const half *)conv, wsg, (half *)x, st, M, C);
    else if (vec)
        k_aim_sg8<32><<<(M + 7) / 8, 256, 0, s>>>((const half *)attn, (const half *)conv, wsg, (half *)x, st, M, C);
    else
        k_aim_sg<<<(M + 7) / 8, 256, 0, s>>>((const half *)attn, (const half *)conv, wsg, (half *)x, st, M, C);
    return cudaGetLastError();
}

}  // namespace ajinative
