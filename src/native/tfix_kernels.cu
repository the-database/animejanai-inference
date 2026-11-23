// native/tfix_kernels.cu — see tfix_kernels.h.
#include "native/tfix_kernels.h"

#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }

// The map's buffer, its entry chosen on the device when it has one.
__device__ __forceinline__ void *txr(const TxMap &m)
{
    return m.sel ? (void *)((char *)m.p + (long long)m.sel[m.pos] * m.fbytes) : m.p;
}

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
    src.p = txr(src);
    dst.p = txr(dst);
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

// k_down for downscales of at most 2x (the half-resolution maps, nearly all of k_down's time): a
// 32 x 8 output tile reads its source block into shared memory once (about 5 loads an output pixel
// instead of 36), filters it along rows, then along columns. Same taps and fp32 sums as k_down (the
// row pass first, so a sum can round differently in its last bit).
constexpr int kDtW = 32, kDtH = 8, kDtSX = 72, kDtSY = 22;  // (kDtW-1)*2+1+6 <= 72, (kDtH-1)*2+1+6 <= 22
template <int VEC>
__global__ void __launch_bounds__(256) k_down_t(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my)
{
    constexpr int NC = VEC == 1 ? 8 : 2;
    using V = typename std::conditional<VEC == 1, uint4, float2>::type;
    __shared__ V ts[kDtSY * kDtSX];
    __shared__ float hs[NC * kDtSY * kDtW];  // [channel][source row][output column]
    src.p = txr(src);
    dst.p = txr(dst);
    const int ox0 = blockIdx.x * kDtW, oy0 = blockIdx.y * kDtH;
    const float sx = (float)ow / iw, sy = (float)oh / ih;
    float w6[6];
    int bx0, by0, bxl, byl;
    taps6(ox0, iw, sx, &bx0, w6);
    taps6(oy0, ih, sy, &by0, w6);
    taps6(min(ox0 + kDtW, ow) - 1, iw, sx, &bxl, w6);
    taps6(min(oy0 + kDtH, oh) - 1, ih, sy, &byl, w6);
    const int SX = bxl + 6 - bx0, SY = byl + 6 - by0;
    for (int i = threadIdx.x; i < SX * SY; i += blockDim.x) {
        const int r = i / SX, c = i % SX, yy = by0 + r, xx = bx0 + c;
        V v{};
        if (yy >= 0 && yy < ih && xx >= 0 && xx < iw) {
            const size_t pix = (size_t)yy * iw + xx;
            if constexpr (VEC == 1)
                v = *reinterpret_cast<const uint4 *>(reinterpret_cast<const half *>(src.p) + pix * src.stride + src.off);
            else
                v = *reinterpret_cast<const float2 *>(reinterpret_cast<const float *>(src.p) + pix * src.stride + src.off);
        }
        ts[r * kDtSX + c] = v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < SY * kDtW; i += blockDim.x) {
        const int r = i / kDtW, x = i % kDtW, ox = ox0 + x;
        float acc[NC] = {};
        if (ox < ow) {
            int bx;
            float wx[6];
            taps6(ox, iw, sx, &bx, wx);
            const V *row = ts + r * kDtSX + (bx - bx0);
#pragma unroll
            for (int t = 0; t < 6; t++) {
                if (wx[t] == 0.f) continue;
                if constexpr (VEC == 1) {
                    float v[8];
                    unpack8h(row[t], v);
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[j] += wx[t] * v[j];
                } else {
                    acc[0] += wx[t] * row[t].x;
                    acc[1] += wx[t] * row[t].y;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; j++) hs[(j * kDtSY + r) * kDtW + x] = acc[j];
    }
    __syncthreads();
    const int x = threadIdx.x % kDtW, ox = ox0 + x, oy = oy0 + threadIdx.x / kDtW;
    if (ox >= ow || oy >= oh) return;
    int by;
    float wy[6];
    taps6(oy, ih, sy, &by, wy);
    const int r0 = by - by0;
    float acc[NC] = {};
#pragma unroll
    for (int r = 0; r < 6; r++) {
        if (wy[r] == 0.f) continue;
#pragma unroll
        for (int j = 0; j < NC; j++) acc[j] += wy[r] * hs[(j * kDtSY + r0 + r) * kDtW + x];
    }
    const size_t p = (size_t)oy * ow + ox;
    for (int c = 0; c < C; c++) st(dst, p, c, acc[c] * (c == 0 ? mx : c == 1 ? my : 1.f));
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

__global__ void k_full(const float *F, const half *logit, int ls, int h, int w, TxMap frm, float *ff, half *full, int H, int W)
{
    const half *fr = (const half *)txr(frm);
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


// tx_assemble: one thread a pixel, the pixel written whole (16-byte stores; writing a few channels of a
// 32- or 64-byte pixel per pass touched a memory segment per thread).
template <bool UP>
__global__ void k_assemble(TxMap cen, TxMap sup, const float *pf, const half *pst, int ph, int pw, float *F, half *X, int xs, int h,
                           int w, float dt)
{
    const half *cp = (const half *)txr(cen), *sp = (const half *)txr(sup);
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= h * w) return;
    float o[32] = {};
    float c8[8];
    unpack8h(reinterpret_cast<const uint4 *>(cp)[p], c8);
    o[0] = c8[0];
    o[1] = c8[1];
    o[2] = c8[2];
#pragma unroll
    for (int c = 0; c < 4; c++) o[6 + c] = c8[3 + c];
    o[14] = dt;
    if constexpr (!UP) {
        float s8[8];
        unpack8h(reinterpret_cast<const uint4 *>(sp)[p], s8);
        o[3] = s8[0];
        o[4] = s8[1];
        o[5] = s8[2];
#pragma unroll
        for (int c = 0; c < 4; c++) o[10 + c] = s8[3 + c];
    } else {
        const int y = p / w, x = p % w;
        int y0, y1, x0, x1;
        float fy, fx;
        srcc(y, ph, (float)h / ph, &y0, &y1, &fy);
        srcc(x, pw, (float)w / pw, &x0, &x1, &fx);
        const size_t a = (size_t)y0 * pw + x0, b = (size_t)y0 * pw + x1, c = (size_t)y1 * pw + x0, d = (size_t)y1 * pw + x1;
        const float2 *f2 = reinterpret_cast<const float2 *>(pf);
        const float2 fa = f2[a], fb = f2[b], fc = f2[c], fd = f2[d];
        float ux = (fa.x * (1 - fx) + fb.x * fx) * (1 - fy) + (fc.x * (1 - fx) + fd.x * fx) * fy;
        float uy = (fa.y * (1 - fx) + fb.y * fx) * (1 - fy) + (fc.y * (1 - fx) + fd.y * fx) * fy;
        ux *= (float)w / pw;
        uy *= (float)h / ph;
        reinterpret_cast<float2 *>(F)[p] = make_float2(ux, uy);
        // the state, X 15..23 (pst channels 0..8)
        float sa[16], sb[16], sc[16], sd[16];
        const uint4 *qa = reinterpret_cast<const uint4 *>(pst + a * 16), *qb = reinterpret_cast<const uint4 *>(pst + b * 16),
                    *qc = reinterpret_cast<const uint4 *>(pst + c * 16), *qd = reinterpret_cast<const uint4 *>(pst + d * 16);
        float t8[8];
#pragma unroll
        for (int hlf = 0; hlf < 2; hlf++) {
            unpack8h(qa[hlf], t8);
#pragma unroll
            for (int j = 0; j < 8; j++) sa[8 * hlf + j] = t8[j];
            unpack8h(qb[hlf], t8);
#pragma unroll
            for (int j = 0; j < 8; j++) sb[8 * hlf + j] = t8[j];
            unpack8h(qc[hlf], t8);
#pragma unroll
            for (int j = 0; j < 8; j++) sc[8 * hlf + j] = t8[j];
            unpack8h(qd[hlf], t8);
#pragma unroll
            for (int j = 0; j < 8; j++) sd[8 * hlf + j] = t8[j];
        }
#pragma unroll
        for (int j = 0; j < 9; j++) o[15 + j] = (sa[j] * (1 - fx) + sb[j] * fx) * (1 - fy) + (sc[j] * (1 - fx) + sd[j] * fx) * fy;
        o[24] = __half2float(__float2half(ux));  // X takes the fp16 of F, as the copy did
        o[25] = __half2float(__float2half(uy));
        // the support warped by F (grid_sample, zero padding)
        const float sx = x + ux, sy = y + uy;
        const float gx0 = floorf(sx), gy0 = floorf(sy), ax = sx - gx0, ay = sy - gy0;
        const int ix0 = (int)gx0, iy0 = (int)gy0;
        float wt[4] = {(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay};
        const int ix[4] = {ix0, ix0 + 1, ix0, ix0 + 1}, iy[4] = {iy0, iy0, iy0 + 1, iy0 + 1};
        float v[7] = {};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (ix[k] < 0 || ix[k] >= w || iy[k] < 0 || iy[k] >= h) wt[k] = 0.f;
            if (wt[k] == 0.f) continue;
            float s8[8];
            unpack8h(reinterpret_cast<const uint4 *>(sp)[(size_t)iy[k] * w + ix[k]], s8);
#pragma unroll
            for (int j = 0; j < 7; j++) v[j] += wt[k] * s8[j];
        }
        o[3] = v[0];
        o[4] = v[1];
        o[5] = v[2];
#pragma unroll
        for (int c = 0; c < 4; c++) o[10 + c] = v[3 + c];
    }
    uint4 *q = reinterpret_cast<uint4 *>(X + (size_t)p * xs);
    float t8[8];
#pragma unroll
    for (int g = 0; g < 4; g++) {
        if (8 * g >= xs) break;
#pragma unroll
        for (int j = 0; j < 8; j++) t8[j] = o[8 * g + j];
        q[g] = pack8h(t8);
    }
}

// tx_head_up: the head's flow and state resized in one pass, the state pixel written whole.
__global__ void k_head_up(const half *O, int ih, int iw, float *F, half *ST, int oh, int ow, int add)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= oh * ow) return;
    const int y = p / ow, x = p % ow;
    int y0, y1, x0, x1;
    float fy, fx;
    srcc(y, ih, (float)oh / ih, &y0, &y1, &fy);
    srcc(x, iw, (float)ow / iw, &x0, &x1, &fx);
    const uint4 *qa = reinterpret_cast<const uint4 *>(O + ((size_t)y0 * iw + x0) * 16), *qb = reinterpret_cast<const uint4 *>(O + ((size_t)y0 * iw + x1) * 16),
                *qc = reinterpret_cast<const uint4 *>(O + ((size_t)y1 * iw + x0) * 16), *qd = reinterpret_cast<const uint4 *>(O + ((size_t)y1 * iw + x1) * 16);
    float v[16];
#pragma unroll
    for (int hlf = 0; hlf < 2; hlf++) {
        float a[8], b[8], c[8], d[8];
        unpack8h(qa[hlf], a);
        unpack8h(qb[hlf], b);
        unpack8h(qc[hlf], c);
        unpack8h(qd[hlf], d);
#pragma unroll
        for (int j = 0; j < 8; j++) v[8 * hlf + j] = (a[j] * (1 - fx) + b[j] * fx) * (1 - fy) + (c[j] * (1 - fx) + d[j] * fx) * fy;
    }
    float2 f = make_float2(v[0], v[1]);
    if (add) {
        const float2 g = reinterpret_cast<const float2 *>(F)[p];
        f.x += g.x;
        f.y += g.y;
    }
    reinterpret_cast<float2 *>(F)[p] = f;
    float s8[8];
#pragma unroll
    for (int j = 0; j < 8; j++) s8[j] = v[2 + j];
    uint4 *q = reinterpret_cast<uint4 *>(ST + (size_t)p * 16);
    q[0] = pack8h(s8);
#pragma unroll
    for (int j = 0; j < 8; j++) s8[j] = j == 0 ? v[10] : 0.f;
    q[1] = pack8h(s8);
}

// tx_conf_in_warp: the warped features are rounded to fp16, as the model's fp16 tensor holds them.
__global__ void k_conf_in_warp(TxMap cen, TxMap wh, TxMap sup, const float *flow, float dt, half *out, half *rch, int h, int w)
{
    const half *cp = (const half *)txr(cen), *hp = (const half *)txr(wh), *sp = (const half *)txr(sup);
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= h * w) return;
    const int y = p / w, x = p % w;
    const float2 f = reinterpret_cast<const float2 *>(flow)[p];
    const float sx = x + f.x, sy = y + f.y;
    const float gx0 = floorf(sx), gy0 = floorf(sy), ax = sx - gx0, ay = sy - gy0;
    const int ix0 = (int)gx0, iy0 = (int)gy0;
    float wt[4] = {(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay};
    const int ix[4] = {ix0, ix0 + 1, ix0, ix0 + 1}, iy[4] = {iy0, iy0, iy0 + 1, iy0 + 1};
    float fw[4] = {};
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if (ix[k] < 0 || ix[k] >= w || iy[k] < 0 || iy[k] >= h) wt[k] = 0.f;
        if (wt[k] == 0.f) continue;
        float s8[8];
        unpack8h(reinterpret_cast<const uint4 *>(sp)[(size_t)iy[k] * w + ix[k]], s8);
#pragma unroll
        for (int j = 0; j < 4; j++) fw[j] += wt[k] * s8[3 + j];
    }
    float c8[8], h8[8];
    unpack8h(reinterpret_cast<const uint4 *>(cp)[p], c8);
    unpack8h(reinterpret_cast<const uint4 *>(hp)[p], h8);
    float o[32] = {};
    for (int c = 0; c < 3; c++) {
        o[c] = c8[c];
        o[3 + c] = h8[c];
        o[6 + c] = fabsf(h8[c] - c8[c]);
    }
    for (int c = 0; c < 4; c++) {
        const float fc = c8[3 + c], wv = __half2float(__float2half(fw[c]));
        o[9 + c] = fc;
        o[13 + c] = wv;
        o[17 + c] = fabsf(wv - fc);
    }
    o[21] = f.x;
    o[22] = f.y;
    o[23] = sqrtf(f.x * f.x + f.y * f.y);
    o[24] = dt;
    uint4 *q = reinterpret_cast<uint4 *>(out + (size_t)p * 32);
    float t8[8];
#pragma unroll
    for (int g = 0; g < 4; g++) {
#pragma unroll
        for (int j = 0; j < 8; j++) t8[j] = o[8 * g + j];
        q[g] = pack8h(t8);
    }
    uint4 *r = reinterpret_cast<uint4 *>(rch + (size_t)p * 16);
#pragma unroll
    for (int j = 0; j < 8; j++) t8[j] = j == 0 ? h8[3] : 0.f;
    r[0] = pack8h(t8);
#pragma unroll
    for (int j = 0; j < 8; j++) t8[j] = 0.f;
    r[1] = pack8h(t8);
}


// acc: float4 [H][W] (weighted rgb sum, weight sum), then float [H][W] (gate sum); first: the first
// support writes it (no clear, no read).
__global__ void k_accum(TxMap logit, int h2, int w2, TxMap full, int vch, float *acc, int H, int W, float thresh, float gslope, int first)
{
    logit.p = txr(logit);
    full.p = txr(full);
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
    float4 *a = reinterpret_cast<float4 *>(acc) + p;
    float *g = acc + (size_t)4 * H * W + p;
    float4 a0 = first ? make_float4(0.f, 0.f, 0.f, 0.f) : *a;
    float g0 = first ? 0.f : *g;
    a0.x += wgt * fv[0];
    a0.y += wgt * fv[1];
    a0.z += wgt * fv[2];
    a0.w += wgt;
    g0 += gate;
    *a = a0;
    *g = g0;
}

__global__ void k_final(TxMap center, const float *acc, int H, int W, float minsup, float cslope, float eps, half *out)
{
    center.p = txr(center);
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;
    const float4 a0 = reinterpret_cast<const float4 *>(acc)[p];
    const float gs = acc[(size_t)4 * H * W + p];
    const float r = sigm((gs - minsup + 0.5f) * cslope), den = fmaxf(1.f + a0.w, eps);
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

cudaError_t tx_down(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, cudaStream_t s)
{
    if (C > 8) return cudaErrorInvalidValue;
    const bool vec = !src.f32 && src.stride % 8 == 0 && src.off % 8 == 0 && ((uintptr_t)src.p % 16) == 0 && src.fbytes % 16 == 0;
    const bool f2 = src.f32 && C == 2 && src.stride == 2 && src.off == 0 && ((uintptr_t)src.p % 8) == 0 && src.fbytes % 8 == 0;
    // the tiled kernel's source block holds downscales of at most 2x (the half-resolution maps)
    const bool tiled = ow > 0 && oh > 0 && iw <= 2 * ow && ih <= 2 * oh;
    const dim3 tg((ow + kDtW - 1) / kDtW, (oh + kDtH - 1) / kDtH);
    if (tiled && vec)
        k_down_t<1><<<tg, kDtW * kDtH, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    else if (tiled && f2)
        k_down_t<2><<<tg, kDtW * kDtH, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    else if (vec)
        k_down<1><<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    else if (f2)
        k_down<2><<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    else
        k_down<0><<<nb((long long)oh * ow), 256, 0, s>>>(src, ih, iw, dst, oh, ow, C, mx, my);
    return cudaGetLastError();
}
cudaError_t tx_assemble(TxMap cen, TxMap sup, const float *pf, const void *pst, int ph, int pw, float *F, void *X, int xs, int h,
                        int w, float dt, cudaStream_t s)
{
    if ((xs != 16 && xs != 32) || cen.f32 || sup.f32 || cen.stride != 8 || sup.stride != 8 || cen.off || sup.off)
        return cudaErrorInvalidValue;
    if (pf)
        k_assemble<true><<<nb((long long)h * w), 256, 0, s>>>(cen, sup, pf, (const half *)pst, ph, pw, F, (half *)X, xs, h, w, dt);
    else
        k_assemble<false><<<nb((long long)h * w), 256, 0, s>>>(cen, sup, nullptr, nullptr, 0, 0, nullptr, (half *)X, xs, h, w, dt);
    return cudaGetLastError();
}
cudaError_t tx_head_up(const void *O, int ih, int iw, float *F, void *ST, int oh, int ow, bool add, cudaStream_t s)
{
    k_head_up<<<nb((long long)oh * ow), 256, 0, s>>>((const half *)O, ih, iw, F, (half *)ST, oh, ow, add ? 1 : 0);
    return cudaGetLastError();
}
cudaError_t tx_conf_in_warp(TxMap cen, TxMap wh, TxMap sup, const float *flow, float dt, void *out, void *rch, int h, int w,
                            cudaStream_t s)
{
    if (cen.f32 || wh.f32 || sup.f32 || cen.stride != 8 || wh.stride != 8 || sup.stride != 8 || cen.off || wh.off || sup.off)
        return cudaErrorInvalidValue;
    k_conf_in_warp<<<nb((long long)h * w), 256, 0, s>>>(cen, wh, sup, flow, dt, (half *)out, (half *)rch, h, w);
    return cudaGetLastError();
}
cudaError_t tx_frame(const void *planes, const void *feat, int fs, int h, int w, void *dst, int H, int W, cudaStream_t s)
{
    if (fs % 4) return cudaErrorInvalidValue;
    k_frame<<<nb((long long)H * W), 256, 0, s>>>((const half *)planes, (const half *)feat, fs, h, w, (half *)dst, H, W);
    return cudaGetLastError();
}
cudaError_t tx_full(const float *F, const void *logit, int ls, int h, int w, TxMap fr, float *ff, void *full, int H, int W, cudaStream_t s)
{
    k_full<<<nb((long long)H * W), 256, 0, s>>>(F, (const half *)logit, ls, h, w, fr, ff, (half *)full, H, W);
    return cudaGetLastError();
}
cudaError_t tx_accum(TxMap logit, int h2, int w2, TxMap full, int vch, float *acc, int H, int W, float thresh, float gslope, bool first,
                     cudaStream_t s)
{
    k_accum<<<nb((long long)H * W), 256, 0, s>>>(logit, h2, w2, full, vch, acc, H, W, thresh, gslope, first ? 1 : 0);
    return cudaGetLastError();
}
cudaError_t tx_final(TxMap center, const float *acc, int H, int W, float minsup, float cslope, float eps, void *out, cudaStream_t s)
{
    k_final<<<nb((long long)H * W), 256, 0, s>>>(center, acc, H, W, minsup, cslope, eps, (half *)out);
    return cudaGetLastError();
}

}  // namespace ajinative
