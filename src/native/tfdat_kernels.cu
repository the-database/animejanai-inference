// native/tfdat_kernels.cu — see tfdat_kernels.h.
#include "native/tfdat_kernels.h"

#include <cstdint>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

// Resize linear half_pixel x2 source coordinate for output index o along an axis of n inputs.
__device__ __forceinline__ void src2(int o, int n, int *i0, int *i1, float *f)
{
    float x = (o + 0.5f) * 0.5f - 0.5f;
    x = fminf(fmaxf(x, 0.f), (float)(n - 1));
    const int a = (int)x;
    *i0 = a;
    *i1 = min(a + 1, n - 1);
    *f = x - a;
}

// 2 * up(F) at (y, x) of the 2h x 2w grid.
__device__ __forceinline__ float2 upflow(const float *F, int h, int w, int y, int x)
{
    int y0, y1, x0, x1;
    float fy, fx;
    src2(y, h, &y0, &y1, &fy);
    src2(x, w, &x0, &x1, &fx);
    const float2 *f = reinterpret_cast<const float2 *>(F);
    const float2 a = f[y0 * w + x0], b = f[y0 * w + x1], c = f[y1 * w + x0], d = f[y1 * w + x1];
    float2 r;
    r.x = 2.f * ((a.x * (1 - fx) + b.x * fx) * (1 - fy) + (c.x * (1 - fx) + d.x * fx) * fy);
    r.y = 2.f * ((a.y * (1 - fx) + b.y * fx) * (1 - fy) + (c.y * (1 - fx) + d.y * fx) * fy);
    return r;
}

// Corner indices and weights of the clamped bilinear warp at (x + u.x, y + u.y).
struct Taps {
    int i[4];
    float w[4];
};
__device__ __forceinline__ Taps warp_taps(int y, int x, float2 u, int H, int W)
{
    const float fx0 = floorf(u.x), fy0 = floorf(u.y), ax = u.x - fx0, ay = u.y - fy0;
    const int xa = x + (int)fx0, ya = y + (int)fy0;
    const int x0 = min(max(xa, 0), W - 1), x1 = min(max(xa + 1, 0), W - 1);
    const int y0 = min(max(ya, 0), H - 1), y1 = min(max(ya + 1, 0), H - 1);
    Taps t;
    t.i[0] = y0 * W + x0;
    t.i[1] = y0 * W + x1;
    t.i[2] = y1 * W + x0;
    t.i[3] = y1 * W + x1;
    t.w[0] = (1 - ax) * (1 - ay);
    t.w[1] = ax * (1 - ay);
    t.w[2] = (1 - ax) * ay;
    t.w[3] = ax * ay;
    return t;
}

__global__ void k_flow_add(const float *U, const half *D, int ds, float *F, int M)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M) return;
    const float2 d = __half22float2(*reinterpret_cast<const half2 *>(D + (size_t)t * ds));
    float2 u = U ? reinterpret_cast<const float2 *>(U)[t] : make_float2(0.f, 0.f);
    u.x += d.x;
    u.y += d.y;
    reinterpret_cast<float2 *>(F)[t] = u;
}

// One thread per (pixel, 8 channels).
__global__ void k_up_warp(const float *F, int h, int w, const half *feat, int fs, int C, half *dst, int ds, int uoff, float *U)
{
    const int G = C / 8, H = 2 * h, W = 2 * w;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W * G) return;
    const int g = i % G, p = i / G, y = p / W, x = p % W, c = g * 8;
    const float2 u = upflow(F, h, w, y, x);
    const Taps t = warp_taps(y, x, u, H, W);
    float a[8] = {};
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint4 v = *reinterpret_cast<const uint4 *>(feat + (size_t)t.i[k] * fs + c);
        const half2 *hv = reinterpret_cast<const half2 *>(&v);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 f = __half22float2(hv[j]);
            a[2 * j] += t.w[k] * f.x;
            a[2 * j + 1] += t.w[k] * f.y;
        }
    }
    uint4 o;
    uint32_t *po = reinterpret_cast<uint32_t *>(&o);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        half2 hh = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
        po[j] = *reinterpret_cast<uint32_t *>(&hh);
    }
    *reinterpret_cast<uint4 *>(dst + (size_t)p * ds + c) = o;
    if (g == 0) {
        *reinterpret_cast<half2 *>(dst + (size_t)p * ds + uoff) = __floats2half2_rn(u.x, u.y);
        for (int k = uoff + 2; k < ds; k += 2) *reinterpret_cast<half2 *>(dst + (size_t)p * ds + k) = __floats2half2_rn(0.f, 0.f);
        reinterpret_cast<float2 *>(U)[p] = u;
    }
}

// One thread per output (half-resolution) pixel: its 2x2 full-resolution block.
__global__ void k_frame_warp(const float *F, const half *nb, const half *ctr, int H, int W, half *wus, half *conf)
{
    const int h = H / 2, w = W / 2;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= h * w) return;
    const int qy = q / w, qx = q % w;
    const size_t plane = (size_t)H * W;
    half o[16];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j] = __float2half(0.f);
    float occ = 0.f;
#pragma unroll
    for (int dy = 0; dy < 2; dy++)
#pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            const int y = 2 * qy + dy, x = 2 * qx + dx;
            const float2 u = upflow(F, h, w, y, x);
            const Taps t = warp_taps(y, x, u, H, W);
            float d = 0.f;
#pragma unroll
            for (int c = 0; c < 3; c++) {
                const half *pl = nb + c * plane;
                const float v = t.w[0] * __half2float(pl[t.i[0]]) + t.w[1] * __half2float(pl[t.i[1]]) + t.w[2] * __half2float(pl[t.i[2]]) +
                                t.w[3] * __half2float(pl[t.i[3]]);
                const half vh = __float2half(v);
                o[c * 4 + dy * 2 + dx] = vh;
                d += fabsf(__half2float(vh) - __half2float(ctr[c * plane + (size_t)y * W + x]));
            }
            occ += d / 3.f;
        }
    uint4 *pw = reinterpret_cast<uint4 *>(wus + (size_t)q * 16);
    pw[0] = *reinterpret_cast<uint4 *>(&o[0]);
    pw[1] = *reinterpret_cast<uint4 *>(&o[8]);
    half cr[16];
#pragma unroll
    for (int j = 0; j < 16; j++) cr[j] = __float2half(0.f);
    cr[1] = __float2half(occ * 0.25f);  // channel 0: the similarity, written later
    uint4 *pc = reinterpret_cast<uint4 *>(conf + (size_t)q * 16);
    pc[0] = *reinterpret_cast<uint4 *>(&cr[0]);
    pc[1] = *reinterpret_cast<uint4 *>(&cr[8]);
}

// One warp per token.
__global__ void k_sim(const half *a, const half *b, int st, int C, float scale, half *conf, int M)
{
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= M) return;
    float s = 0.f;
    for (int c = 2 * lane; c < C; c += 64) {
        const float2 x = __half22float2(*reinterpret_cast<const half2 *>(a + (size_t)t * st + c));
        const float2 y = __half22float2(*reinterpret_cast<const half2 *>(b + (size_t)t * st + c));
        s += x.x * y.x + (c + 1 < C ? x.y * y.y : 0.f);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) conf[(size_t)t * 16] = __float2half(s * scale);
}

__global__ void k_gate(const half *x, const half *g, int gs, half *out, int st, int C, int M)
{
    const int pairs = (C + 15) / 16 * 8;  // through the padding to 16
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * pairs) return;
    const int t = i / pairs, c = (i % pairs) * 2;
    const float k = sigm(__half2float(g[(size_t)t * gs]));
    float2 v = c < C ? __half22float2(*reinterpret_cast<const half2 *>(x + (size_t)t * st + c)) : make_float2(0.f, 0.f);
    v.x *= k;
    v.y = c + 1 < C ? v.y * k : 0.f;
    *reinterpret_cast<half2 *>(out + (size_t)t * st + c) = __floats2half2_rn(v.x, v.y);
}

__global__ void k_colscale(const half *x, const float *cm, half *out, int st, int C, int M)
{
    const int pairs = (C + 15) / 16 * 8;  // through the padding to 16
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * pairs) return;
    const int t = i / pairs, c = (i % pairs) * 2;
    float2 v = c < C ? __half22float2(*reinterpret_cast<const half2 *>(x + (size_t)t * st + c)) : make_float2(0.f, 0.f);
    v.x = c < C ? v.x * cm[c] : 0.f;
    v.y = c + 1 < C ? v.y * cm[c + 1] : 0.f;
    *reinterpret_cast<half2 *>(out + (size_t)t * st + c) = __floats2half2_rn(v.x, v.y);
}

__global__ void k_resize2x(const half *pl, int H, int W, half *dst)
{
    const int OW = 2 * W;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= 4 * H * W) return;
    const int y = p / OW, x = p % OW;
    int y0, y1, x0, x1;
    float fy, fx;
    src2(y, H, &y0, &y1, &fy);
    src2(x, W, &x0, &x1, &fx);
    half o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = __float2half(0.f);
#pragma unroll
    for (int c = 0; c < 3; c++) {
        const half *q = pl + (size_t)c * H * W;
        const float a = __half2float(q[y0 * W + x0]), b = __half2float(q[y0 * W + x1]), cc = __half2float(q[y1 * W + x0]),
                    d = __half2float(q[y1 * W + x1]);
        o[c] = __float2half((a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy);
    }
    uint4 *pd = reinterpret_cast<uint4 *>(dst + (size_t)p * 16);
    pd[0] = *reinterpret_cast<uint4 *>(o);
    pd[1] = make_uint4(0, 0, 0, 0);
}

unsigned nb(long long n, int t) { return (unsigned)((n + t - 1) / t); }

}  // namespace

cudaError_t tf_flow_add(const float *U, const void *D, int ds, float *F, int M, cudaStream_t s)
{
    k_flow_add<<<nb(M, 256), 256, 0, s>>>(U, (const half *)D, ds, F, M);
    return cudaGetLastError();
}

cudaError_t tf_up_warp(const float *F, int h, int w, const void *feat, int fs, int C, void *dst, int ds, int uoff, float *U, cudaStream_t s)
{
    if (C % 8 || fs % 8 || ds % 8 || uoff % 2) return cudaErrorInvalidValue;
    k_up_warp<<<nb(4ll * h * w * (C / 8), 256), 256, 0, s>>>(F, h, w, (const half *)feat, fs, C, (half *)dst, ds, uoff, U);
    return cudaGetLastError();
}

cudaError_t tf_frame_warp(const float *F, const void *nbp, const void *ctr, int H, int W, void *wus, void *conf, cudaStream_t s)
{
    if (H % 2 || W % 2) return cudaErrorInvalidValue;
    k_frame_warp<<<nb((long long)H * W / 4, 128), 128, 0, s>>>(F, (const half *)nbp, (const half *)ctr, H, W, (half *)wus, (half *)conf);
    return cudaGetLastError();
}

cudaError_t tf_sim(const void *a, const void *b, int st, int C, float scale, void *conf, int M, cudaStream_t s)
{
    k_sim<<<nb(M, 8), 256, 0, s>>>((const half *)a, (const half *)b, st, C, scale, (half *)conf, M);
    return cudaGetLastError();
}

cudaError_t tf_gate(const void *x, const void *g, int gs, void *out, int st, int C, int M, cudaStream_t s)
{
    if ((C + 15) / 16 * 16 > st) return cudaErrorInvalidValue;
    k_gate<<<nb((long long)M * ((C + 15) / 16 * 8), 256), 256, 0, s>>>((const half *)x, (const half *)g, gs, (half *)out, st, C, M);
    return cudaGetLastError();
}

cudaError_t tf_colscale(const void *x, const float *cm, void *out, int st, int C, int M, cudaStream_t s)
{
    if ((C + 15) / 16 * 16 > st) return cudaErrorInvalidValue;
    k_colscale<<<nb((long long)M * ((C + 15) / 16 * 8), 256), 256, 0, s>>>((const half *)x, cm, (half *)out, st, C, M);
    return cudaGetLastError();
}

cudaError_t tf_resize2x(const void *planes, int H, int W, void *dst, cudaStream_t s)
{
    k_resize2x<<<nb(4ll * H * W, 256), 256, 0, s>>>((const half *)planes, H, W, (half *)dst);
    return cudaGetLastError();
}

}  // namespace ajinative
