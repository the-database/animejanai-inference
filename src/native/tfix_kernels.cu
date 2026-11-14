// native/tfix_kernels.cu — see tfix_kernels.h.
#include "native/tfix_kernels.h"

#include <cstdint>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

__device__ __forceinline__ float ld(const TxMap &m, size_t pix, int c)
{
    const size_t i = pix * m.stride + m.off + c;
    return m.f32 ? reinterpret_cast<const float *>(m.p)[i] : __half2float(reinterpret_cast<const half *>(m.p)[i]);
}
__device__ __forceinline__ void st(const TxMap &m, size_t pix, int c, float v)
{
    const size_t i = pix * m.stride + m.off + c;
    if (m.f32) reinterpret_cast<float *>(m.p)[i] = v;
    else reinterpret_cast<half *>(m.p)[i] = __float2half(v);
}

// source coordinate for output index o (half-pixel, clamped like ONNX Resize linear)
__device__ __forceinline__ void srcc(int o, int n_in, float scale, int *i0, int *i1, float *f)
{
    float x = (o + 0.5f) / scale - 0.5f;
    x = fminf(fmaxf(x, 0.f), (float)(n_in - 1));
    const int a = (int)x;
    *i0 = a;
    *i1 = min(a + 1, n_in - 1);
    *f = x - a;
}

__device__ __forceinline__ void unpack8h(const uint4 &u, float (&v)[8])
{
    const half2 *h = reinterpret_cast<const half2 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 f = __half22float2(h[j]);
        v[2 * j] = f.x;
        v[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ uint4 pack8h(const float (&v)[8])
{
    uint4 u;
    half2 *h = reinterpret_cast<half2 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) h[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
    return u;
}

__global__ void k_resize(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, int add)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= oh * ow) return;
    const int y = p / ow, x = p % ow;
    int y0, y1, x0, x1;
    float fy, fx;
    srcc(y, ih, (float)oh / ih, &y0, &y1, &fy);
    srcc(x, iw, (float)ow / iw, &x0, &x1, &fx);
    const size_t a = (size_t)y0 * iw + x0, b = (size_t)y0 * iw + x1, c = (size_t)y1 * iw + x0, d = (size_t)y1 * iw + x1;
    for (int ch = 0; ch < C; ch++) {
        float v = (ld(src, a, ch) * (1 - fx) + ld(src, b, ch) * fx) * (1 - fy) + (ld(src, c, ch) * (1 - fx) + ld(src, d, ch) * fx) * fy;
        if (ch == 0) v *= mx;
        else if (ch == 1) v *= my;
        if (add) v += ld(dst, p, ch);
        st(dst, p, ch, v);
    }
}

// One output pixel: rows/columns i0 - 2 .. i0 + 3 of the source with weights (1 - f) k[r - i0 + 2] + f k[r - i0 + 1].
__device__ __forceinline__ void taps6(int o, int n_in, float scale, int *base, float (&wt)[6])
{
    int i0, i1;
    float f;
    srcc(o, n_in, scale, &i0, &i1, &f);
    const float k[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
    *base = i0 - 2;
    const float f1 = i1 == i0 ? 0.f : f, f0 = 1.f - f1;
#pragma unroll
    for (int t = 0; t < 6; t++) {
        const int a = t, b = t - 1;  // offsets into k for the i0 and i1 windows
        wt[t] = (a <= 4 ? f0 * k[a] : 0.f) + (b >= 0 && b <= 4 ? f1 * k[b] : 0.f);
    }
}

// VEC 1: fp16 sources with 8-channel uint4 pixels; 2: fp32 2-channel float2 pixels (flows); 0: any.
template <int VEC>
__global__ void k_down(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= oh * ow) return;
    const int y = p / ow, x = p % ow;
    int by, bx;
    float wy[6], wx[6];
    taps6(y, ih, (float)oh / ih, &by, wy);
    taps6(x, iw, (float)ow / iw, &bx, wx);
    float acc[8] = {};
#pragma unroll
    for (int r = 0; r < 6; r++) {
        const int yy = by + r;
        if (yy < 0 || yy >= ih || wy[r] == 0.f) continue;
#pragma unroll
        for (int t = 0; t < 6; t++) {
            const int xx = bx + t;
            if (xx < 0 || xx >= iw || wx[t] == 0.f) continue;
            const float wgt = wy[r] * wx[t];
            const size_t pix = (size_t)yy * iw + xx;
            if constexpr (VEC == 2) {
                const float2 v = *reinterpret_cast<const float2 *>(reinterpret_cast<const float *>(src.p) + pix * src.stride + src.off);
                acc[0] += wgt * v.x;
                acc[1] += wgt * v.y;
            } else if constexpr (VEC == 1) {
                const uint4 u = *reinterpret_cast<const uint4 *>(reinterpret_cast<const half *>(src.p) + pix * src.stride + src.off);
                const half2 *hv = reinterpret_cast<const half2 *>(&u);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 fv = __half22float2(hv[j]);
                    acc[2 * j] += wgt * fv.x;
                    acc[2 * j + 1] += wgt * fv.y;
                }
            } else {
                for (int c = 0; c < C; c++) acc[c] += wgt * ld(src, pix, c);
            }
        }
    }
    for (int c = 0; c < C; c++) st(dst, p, c, acc[c] * (c == 0 ? mx : c == 1 ? my : 1.f));
}

__global__ void k_lowpass(TxMap src, int h, int w, TxMap dst, int C)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= h * w) return;
    const int y = p / w, x = p % w;
    const float k[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
    for (int ch = 0; ch < C; ch++) {
        float s = 0.f;
#pragma unroll
        for (int dy = -2; dy <= 2; dy++) {
            const int yy = y + dy;
            if (yy < 0 || yy >= h) continue;
            float r = 0.f;
#pragma unroll
            for (int dx = -2; dx <= 2; dx++) {
                const int xx = x + dx;
                if (xx >= 0 && xx < w) r += k[dx + 2] * ld(src, (size_t)yy * w + xx, ch);
            }
            s += k[dy + 2] * r;
        }
        st(dst, p, ch, s);
    }
}

__global__ void k_warp(TxMap src, int C, const float *flow, int h, int w, TxMap dst, TxMap valid)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= h * w) return;
    const int y = p / w, x = p % w;
    const float2 f = reinterpret_cast<const float2 *>(flow)[p];
    const float sx = x + f.x, sy = y + f.y;
    const float fx0 = floorf(sx), fy0 = floorf(sy), ax = sx - fx0, ay = sy - fy0;
    const int x0 = (int)fx0, y0 = (int)fy0;
    float wt[4] = {(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay};
    int ix[4] = {x0, x0 + 1, x0, x0 + 1}, iy[4] = {y0, y0, y0 + 1, y0 + 1};
    float vs = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if (ix[k] < 0 || ix[k] >= w || iy[k] < 0 || iy[k] >= h) wt[k] = 0.f;
        vs += wt[k];
    }
    for (int ch = 0; ch < C; ch++) {
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++)
            if (wt[k] != 0.f) v += wt[k] * ld(src, (size_t)iy[k] * w + ix[k], ch);
        st(dst, p, ch, v);
    }
    if (valid.p) st(valid, p, 0, vs);
}

__global__ void k_planes(const half *pl, int n, TxMap dst)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    for (int c = 0; c < 3; c++) st(dst, p, c, __half2float(pl[(size_t)c * n + p]));
}

__global__ void k_frame(const half *pl, const half *feat, int fs, int h, int w, half *dst, int H, int W)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;
    const int y = p / W, x = p % W;
    const size_t n = (size_t)H * W;
    float o[8];
    o[0] = __half2float(pl[p]);
    o[1] = __half2float(pl[n + p]);
    o[2] = __half2float(pl[2 * n + p]);
    int y0, y1, x0, x1;
    float fy, fx;
    srcc(y, h, (float)H / h, &y0, &y1, &fy);
    srcc(x, w, (float)W / w, &x0, &x1, &fx);
    const uint2 ua = *reinterpret_cast<const uint2 *>(feat + ((size_t)y0 * w + x0) * fs), ub = *reinterpret_cast<const uint2 *>(feat + ((size_t)y0 * w + x1) * fs),
                uc = *reinterpret_cast<const uint2 *>(feat + ((size_t)y1 * w + x0) * fs), ud = *reinterpret_cast<const uint2 *>(feat + ((size_t)y1 * w + x1) * fs);
    const half *ha = reinterpret_cast<const half *>(&ua), *hb = reinterpret_cast<const half *>(&ub), *hc = reinterpret_cast<const half *>(&uc),
               *hd = reinterpret_cast<const half *>(&ud);
#pragma unroll
    for (int j = 0; j < 4; j++)
        o[3 + j] = (__half2float(ha[j]) * (1 - fx) + __half2float(hb[j]) * fx) * (1 - fy) + (__half2float(hc[j]) * (1 - fx) + __half2float(hd[j]) * fx) * fy;
    o[7] = 0.f;
    *reinterpret_cast<uint4 *>(dst + (size_t)p * 8) = pack8h(o);
}

__global__ void k_full(const float *F, const half *logit, int ls, int h, int w, const half *fr, float *ff, half *full, int H, int W)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;
    const int y = p / W, x = p % W;
    int y0, y1, x0, x1;
    float fy, fx;
    srcc(y, h, (float)H / h, &y0, &y1, &fy);
    srcc(x, w, (float)W / w, &x0, &x1, &fx);
    const float2 *f2 = reinterpret_cast<const float2 *>(F);
    const float2 a = f2[y0 * w + x0], b = f2[y0 * w + x1], c = f2[y1 * w + x0], d = f2[y1 * w + x1];
    const float ux = ((a.x * (1 - fx) + b.x * fx) * (1 - fy) + (c.x * (1 - fx) + d.x * fx) * fy) * ((float)W / w);
    const float uy = ((a.y * (1 - fx) + b.y * fx) * (1 - fy) + (c.y * (1 - fx) + d.y * fx) * fy) * ((float)H / h);
    reinterpret_cast<float2 *>(ff)[p] = make_float2(ux, uy);
    const float lg = (__half2float(logit[((size_t)y0 * w + x0) * ls]) * (1 - fx) + __half2float(logit[((size_t)y0 * w + x1) * ls]) * fx) * (1 - fy) +
                     (__half2float(logit[((size_t)y1 * w + x0) * ls]) * (1 - fx) + __half2float(logit[((size_t)y1 * w + x1) * ls]) * fx) * fy;
    const float px = x + ux, py = y + uy, gx0 = floorf(px), gy0 = floorf(py), ax = px - gx0, ay = py - gy0;
    const int ix = (int)gx0, iy = (int)gy0;
    float o[8] = {};
    float vs = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int xx = ix + (k & 1), yy = iy + (k >> 1);
        const float wt = ((k & 1) ? ax : 1 - ax) * ((k >> 1) ? ay : 1 - ay);
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
        vs += wt;
        float v[8];
        unpack8h(*reinterpret_cast<const uint4 *>(fr + ((size_t)yy * W + xx) * 8), v);
        o[0] += wt * v[0];
        o[1] += wt * v[1];
        o[2] += wt * v[2];
    }
    o[3] = lg;
    o[4] = vs;
    *reinterpret_cast<uint4 *>(full + (size_t)p * 8) = pack8h(o);
}

__global__ void k_fill(TxMap dst, int n, float v)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) st(dst, p, 0, v);
}

__global__ void k_conf_in(TxMap cen, TxMap wh, TxMap wf, const float *flow, float dt, half *out, int n)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float o[32] = {};
    for (int c = 0; c < 3; c++) {
        const float cc = ld(cen, p, c), ww = ld(wh, p, c);
        o[c] = cc;
        o[3 + c] = ww;
        o[6 + c] = fabsf(ww - cc);
    }
    for (int c = 0; c < 4; c++) {
        const float fc = ld(cen, p, 3 + c), fw = ld(wf, p, c);
        o[9 + c] = fc;
        o[13 + c] = fw;
        o[17 + c] = fabsf(fw - fc);
    }
    const float2 f = reinterpret_cast<const float2 *>(flow)[p];
    o[21] = f.x;
    o[22] = f.y;
    o[23] = sqrtf(f.x * f.x + f.y * f.y);
    o[24] = dt;
    half *q = out + (size_t)p * 32;
#pragma unroll
    for (int c = 0; c < 32; c += 2) *reinterpret_cast<half2 *>(q + c) = __floats2half2_rn(o[c], o[c + 1]);
}

__global__ void k_accum(TxMap logit, int h2, int w2, TxMap full, int vch, float *acc, int H, int W, float thresh, float gslope)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;
    const int y = p / W, x = p % W;
    int y0, y1, x0, x1;
    float fy, fx;
    srcc(y, h2, (float)H / h2, &y0, &y1, &fy);
    srcc(x, w2, (float)W / w2, &x0, &x1, &fx);
    const float l = (ld(logit, (size_t)y0 * w2 + x0, 0) * (1 - fx) + ld(logit, (size_t)y0 * w2 + x1, 0) * fx) * (1 - fy) +
                    (ld(logit, (size_t)y1 * w2 + x0, 0) * (1 - fx) + ld(logit, (size_t)y1 * w2 + x1, 0) * fx) * fy;
    float fv[8];
    unpack8h(*reinterpret_cast<const uint4 *>(reinterpret_cast<const half *>(full.p) + (size_t)p * 8), fv);
    const float conf = sigm(l) * fv[vch], gate = sigm((conf - thresh) * gslope), wgt = conf * gate;
    float4 *a = reinterpret_cast<float4 *>(acc + (size_t)p * 8);
    float4 a0 = a[0], a1 = a[1];
    a0.x += wgt * fv[0];
    a0.y += wgt * fv[1];
    a0.z += wgt * fv[2];
    a0.w += wgt;
    a1.x += gate;
    a[0] = a0;
    a[1] = a1;
}

__global__ void k_final(TxMap center, const float *acc, int H, int W, float minsup, float cslope, float eps, half *out)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;
    const float4 a0 = reinterpret_cast<const float4 *>(acc + (size_t)p * 8)[0], a1 = reinterpret_cast<const float4 *>(acc + (size_t)p * 8)[1];
    const float r = sigm((a1.x - minsup + 0.5f) * cslope), den = fmaxf(1.f + a0.w, eps);
    float cv[8];
    unpack8h(*reinterpret_cast<const uint4 *>(reinterpret_cast<const half *>(center.p) + (size_t)p * 8), cv);
    const float aa[3] = {a0.x, a0.y, a0.z};
    for (int c = 0; c < 3; c++) {
        const float cc = cv[c], bl = (cc + aa[c]) / den;
        out[(size_t)c * H * W + p] = __float2half(cc + r * (bl - cc));
    }
}

unsigned nb(long long n) { return (unsigned)((n + 255) / 256); }

}  // namespace

cudaError_t tx_resize(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, bool add, cudaStream_t s)
{
    k_resize<<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my, add ? 1 : 0);
    return cudaGetLastError();
}
cudaError_t tx_down(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, cudaStream_t s)
{
    if (C > 8) return cudaErrorInvalidValue;
    const bool vec = !src.f32 && src.stride % 8 == 0 && src.off % 8 == 0 && ((uintptr_t)src.p % 16) == 0;
    const bool f2 = src.f32 && C == 2 && src.stride == 2 && src.off == 0 && ((uintptr_t)src.p % 8) == 0;
    if (vec)
        k_down<1><<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    else if (f2)
        k_down<2><<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    else
        k_down<0><<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    return cudaGetLastError();
}
cudaError_t tx_lowpass(TxMap src, int h, int w, TxMap dst, int C, cudaStream_t s)
{
    k_lowpass<<<nb((long long)h * w), 256, 0, s>>>(src, h, w, dst, C);
    return cudaGetLastError();
}
cudaError_t tx_warp(TxMap src, int C, const float *flow, int h, int w, TxMap dst, TxMap valid, cudaStream_t s)
{
    k_warp<<<nb((long long)h * w), 256, 0, s>>>(src, C, flow, h, w, dst, valid);
    return cudaGetLastError();
}
cudaError_t tx_planes(const void *planes, int h, int w, TxMap dst, cudaStream_t s)
{
    k_planes<<<nb((long long)h * w), 256, 0, s>>>((const half *)planes, h * w, dst);
    return cudaGetLastError();
}
cudaError_t tx_frame(const void *planes, const void *feat, int fs, int h, int w, void *dst, int H, int W, cudaStream_t s)
{
    if (fs % 4) return cudaErrorInvalidValue;
    k_frame<<<nb((long long)H * W), 256, 0, s>>>((const half *)planes, (const half *)feat, fs, h, w, (half *)dst, H, W);
    return cudaGetLastError();
}
cudaError_t tx_full(const float *F, const void *logit, int ls, int h, int w, const void *fr, float *ff, void *full, int H, int W, cudaStream_t s)
{
    k_full<<<nb((long long)H * W), 256, 0, s>>>(F, (const half *)logit, ls, h, w, (const half *)fr, ff, (half *)full, H, W);
    return cudaGetLastError();
}
cudaError_t tx_fill(TxMap dst, int h, int w, float v, cudaStream_t s)
{
    k_fill<<<nb((long long)h * w), 256, 0, s>>>(dst, h * w, v);
    return cudaGetLastError();
}
cudaError_t tx_conf_in(TxMap cen, TxMap wh, TxMap wf, const float *flow, float dt, void *out, int h, int w, cudaStream_t s)
{
    k_conf_in<<<nb((long long)h * w), 256, 0, s>>>(cen, wh, wf, flow, dt, (half *)out, h * w);
    return cudaGetLastError();
}
cudaError_t tx_accum(TxMap logit, int h2, int w2, TxMap full, int vch, float *acc, int H, int W, float thresh, float gslope, cudaStream_t s)
{
    k_accum<<<nb((long long)H * W), 256, 0, s>>>(logit, h2, w2, full, vch, acc, H, W, thresh, gslope);
    return cudaGetLastError();
}
cudaError_t tx_final(TxMap center, const float *acc, int H, int W, float minsup, float cslope, float eps, void *out, cudaStream_t s)
{
    k_final<<<nb((long long)H * W), 256, 0, s>>>(center, acc, H, W, minsup, cslope, eps, (half *)out);
    return cudaGetLastError();
}

}  // namespace ajinative
