// native/rife_kernels.cu — see rife_kernels.h and rife.h.
//
// The graph's resampling, evaluated pointwise:
//  - Resize (linear, half_pixel) by 1/s with even s samples exactly between two pixels, so a
//    downsampled value is the mean of a 2x2 block of full-resolution pixels (one pixel at s = 1);
//  - Resize by s (upsampling a block's output): u = (X + 0.5) / s - 0.5, clamped at 0, linear;
//  - GridSample (bilinear, border, align_corners) on grid = mesh + flow * 2/(n-1), with the mesh
//    and multiplier rounded to fp16 as libaji's constant input planes are, and the flow rounded to
//    fp16 as the graph's flow tensor is, so positions match the TensorRT model's.
#include "native/rife_kernels.h"

#include <cuda_fp16.h>

namespace ajinative {
namespace {

constexpr int kMaxCin = 64;

struct Dev {
    RifeFrame f;
    RifeGather g;
};

__device__ __forceinline__ float h2f(half v) { return __half2float(v); }
__device__ __forceinline__ float rh(float v) { return __half2float(__float2half(v)); }

// Bilinear taps of a linear upsample by s (half_pixel, clamped) of a bw x bh map at full-res (X, Y).
struct Taps {
    size_t i00, i01, i10, i11;
    float wx, wy;
};
__device__ __forceinline__ Taps up_taps(int s, int bw, int bh, int X, int Y)
{
    float u = (X + 0.5f) / s - 0.5f, v = (Y + 0.5f) / s - 0.5f;
    u = u < 0.f ? 0.f : u;
    v = v < 0.f ? 0.f : v;
    int x0 = (int)u, y0 = (int)v;
    x0 = x0 > bw - 1 ? bw - 1 : x0;
    y0 = y0 > bh - 1 ? bh - 1 : y0;
    const int x1 = x0 + 1 < bw ? x0 + 1 : bw - 1, y1 = y0 + 1 < bh ? y0 + 1 : bh - 1;
    return {(size_t)y0 * bw + x0, (size_t)y0 * bw + x1, (size_t)y1 * bw + x0, (size_t)y1 * bw + x1, u - x0, v - y0};
}
__device__ __forceinline__ void h8(const half *p, float *o)
{
    const uint4 v = __ldg(reinterpret_cast<const uint4 *>(p));
    const half2 *h = reinterpret_cast<const half2 *>(&v);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const float2 f = __half22float2(h[k]);
        o[2 * k] = f.x;
        o[2 * k + 1] = f.y;
    }
}
__device__ __forceinline__ void h4(const half *p, float *o)
{
    const uint2 v = __ldg(reinterpret_cast<const uint2 *>(p));
    const half2 *h = reinterpret_cast<const half2 *>(&v);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
    o[0] = a.x;
    o[1] = a.y;
    o[2] = b.x;
    o[3] = b.y;
}
// Linear interpolation along x, then y, rounded to fp16 like the graph's tensors.
__device__ __forceinline__ float lerp4(float a, float b, float c, float d, const Taps &t)
{
    const float top = a + (b - a) * t.wx, bot = c + (d - c) * t.wx;
    return rh(top + (bot - top) * t.wy);
}

// Upsampled channels [0, 8*NV) of a block output at (X, Y).
template <int NV>
__device__ __forceinline__ void up_vec(const RifeBlockOut &b, int W, int H, int X, int Y, float *out)
{
    const Taps t = up_taps(b.s, W / b.s, H / b.s, X, Y);
    const half *p = (const half *)b.b;
#pragma unroll
    for (int v = 0; v < NV; v++) {
        float a[8], bb[8], c[8], d[8];
        h8(p + t.i00 * b.kstride + v * 8, a);
        h8(p + t.i01 * b.kstride + v * 8, bb);
        h8(p + t.i10 * b.kstride + v * 8, c);
        h8(p + t.i11 * b.kstride + v * 8, d);
#pragma unroll
        for (int k = 0; k < 8; k++) out[v * 8 + k] = lerp4(a[k], bb[k], c[k], d[k], t);
    }
}
// Upsampled flow (channels 0-3) x s of a block output at (X, Y).
__device__ void up_flow(const RifeBlockOut &b, int W, int H, int X, int Y, float *fl)
{
    const Taps t = up_taps(b.s, W / b.s, H / b.s, X, Y);
    const half *p = (const half *)b.b;
    float a[4], bb[4], c[4], d[4];
    h4(p + t.i00 * b.kstride, a);
    h4(p + t.i01 * b.kstride, bb);
    h4(p + t.i10 * b.kstride, c);
    h4(p + t.i11 * b.kstride, d);
#pragma unroll
    for (int k = 0; k < 4; k++) fl[k] = rh(lerp4(a[k], bb[k], c[k], d[k], t) * b.s);
}

// The full-resolution flow at (X, Y): the sum of each block's upsampled flow x s, in block order.
// last: the last block's channels already upsampled (or null).
__device__ void flow_sum(const RifeBlockOut *bl, int nb, int W, int H, int X, int Y, const float *last, float *fl)
{
    fl[0] = fl[1] = fl[2] = fl[3] = 0.f;
    for (int j = 0; j < nb; j++) {
        float t[4];
        if (j == nb - 1 && last) {
#pragma unroll
            for (int k = 0; k < 4; k++) t[k] = rh(last[k] * bl[j].s);
        } else {
            up_flow(bl[j], W, H, X, Y, t);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) fl[k] = rh(fl[k] + t[k]);
    }
}

// GridSample of frame k (image channels, then NF features) at pixel (X, Y) moved by (fx, fy).
template <int NF>
__device__ __forceinline__ void warp(const RifeFrame &f, int k, int X, int Y, float fx, float fy, float *out)
{
    const int W = f.W, H = f.H;
    const float mx = rh(2.0f * X / (W - 1) - 1.0f), my = rh(2.0f * Y / (H - 1) - 1.0f);
    const float mw = rh(2.0f / (W - 1)), mh = rh(2.0f / (H - 1));
    const float gx = mx + rh(fx) * mw, gy = my + rh(fy) * mh;
    float ix = (gx + 1.f) * 0.5f * (W - 1), iy = (gy + 1.f) * 0.5f * (H - 1);
    ix = fminf(fmaxf(ix, 0.f), (float)(W - 1));
    iy = fminf(fmaxf(iy, 0.f), (float)(H - 1));
    const int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
    const int x1 = x0 + 1 < W ? x0 + 1 : W - 1, y1 = y0 + 1 < H ? y0 + 1 : H - 1;
    const float wx = ix - x0, wy = iy - y0;
    const float w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy), w10 = (1 - wx) * wy, w11 = wx * wy;
    const half *img = (const half *)f.in11 + (size_t)k * 3 * W * H;
    const size_t plane = (size_t)W * H;
    const size_t i00 = (size_t)y0 * W + x0, i01 = (size_t)y0 * W + x1, i10 = (size_t)y1 * W + x0, i11 = (size_t)y1 * W + x1;
#pragma unroll
    for (int c = 0; c < 3; c++) {
        const half *pl = img + c * plane;
        out[c] = rh(w00 * h2f(__ldg(pl + i00)) + w01 * h2f(__ldg(pl + i01)) + w10 * h2f(__ldg(pl + i10)) + w11 * h2f(__ldg(pl + i11)));
    }
    if constexpr (NF > 0) {
        const half *ft = (const half *)f.feat[k];
        const int st = f.fstride;
#pragma unroll
        for (int v = 0; v * 8 < NF; v++) {
            float a[8], b[8], c[8], d[8];
            h8(ft + i00 * st + v * 8, a);
            h8(ft + i01 * st + v * 8, b);
            h8(ft + i10 * st + v * 8, c);
            h8(ft + i11 * st + v * 8, d);
#pragma unroll
            for (int e = 0; e < 8; e++)
                if (v * 8 + e < NF) out[3 + v * 8 + e] = rh(w00 * a[e] + w01 * b[e] + w10 * c[e] + w11 * d[e]);
        }
    }
}

// F: features per frame; NM: mask (+ fed features) channels; KV: 8-channel vectors of the last
// block's output; NT: taps per axis (1 at scale 1, else 2); B0: block 0's input.
template <int F, int NM, int KV, int NT, int B0, int REV>
__global__ void k_gather(Dev p)
{
    constexpr int CIN = B0 ? 6 + 2 * F + 1 : 6 + 2 * F + 1 + NM + 4, CP8 = (CIN + 7) / 8 * 8;
    const RifeFrame &f = p.f;
    const RifeGather &g = p.g;
    const int bw = f.W / g.s, bh = f.H / g.s;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;
    float acc[CP8];
#pragma unroll
    for (int c = 0; c < CP8; c++) acc[c] = 0.f;
    constexpr float wt = NT == 1 ? 1.f : 0.25f;
    const float tt = REV ? rh(1.f - rh(*g.t)) : rh(*g.t);
    constexpr int A0 = REV ? 1 : 0;  // the frame that comes first
#pragma unroll
    for (int ty = 0; ty < NT; ty++)
#pragma unroll
        for (int tx = 0; tx < NT; tx++) {
            const int X = NT == 1 ? x : g.s * x + g.s / 2 - 1 + tx, Y = NT == 1 ? y : g.s * y + g.s / 2 - 1 + ty;
            if constexpr (B0) {
                // [frame A, frame B, features A, features B, timestep]
                const size_t plane = (size_t)f.W * f.H, pix = (size_t)Y * f.W + X;
                const half *in = (const half *)f.in11;
#pragma unroll
                for (int c = 0; c < 6; c++) acc[c] += wt * h2f(__ldg(in + ((c + 3 * A0) % 6) * plane + pix));
#pragma unroll
                for (int k = 0; k < 2; k++)
                    if constexpr (F > 0) {
                        float fv[16];
#pragma unroll
                        for (int v = 0; v * 8 < F; v++) h8((const half *)f.feat[k ^ A0] + pix * f.fstride + v * 8, fv + v * 8);
#pragma unroll
                        for (int c = 0; c < F; c++) acc[6 + k * F + c] += wt * fv[c];
                    }
                acc[6 + 2 * F] += wt * tt;
            } else {
                // [warped A, warped B, warped features A, B, timestep, mask, (features), flow / s]
                const RifeBlockOut &last = g.prev[g.nprev - 1];
                float lv[8 * KV];
                up_vec<KV>(last, f.W, f.H, X, Y, lv);
                float fl[4];
                flow_sum(g.prev, g.nprev, f.W, f.H, X, Y, lv, fl);
                float wa[3 + F + 1], wb[3 + F + 1];
                if constexpr (REV) {
                    warp<F>(f, 1, X, Y, fl[2], fl[3], wa);
                    warp<F>(f, 0, X, Y, fl[0], fl[1], wb);
                } else {
                    warp<F>(f, 0, X, Y, fl[0], fl[1], wa);
                    warp<F>(f, 1, X, Y, fl[2], fl[3], wb);
                }
#pragma unroll
                for (int c = 0; c < 3; c++) {
                    acc[c] += wt * wa[c];
                    acc[3 + c] += wt * wb[c];
                }
#pragma unroll
                for (int c = 0; c < F; c++) {
                    acc[6 + c] += wt * wa[3 + c];
                    acc[6 + F + c] += wt * wb[3 + c];
                }
                acc[6 + 2 * F] += wt * tt;
#pragma unroll
                for (int c = 0; c < NM; c++) acc[7 + 2 * F + c] += wt * (REV && c == 0 ? -lv[4] : lv[4 + c]);
#pragma unroll
                for (int c = 0; c < 4; c++) acc[7 + 2 * F + NM + c] += wt * rh(fl[REV ? (c + 2) % 4 : c] * (1.0f / g.s));
            }
        }
    half *d = (half *)g.dst + ((size_t)y * bw + x) * g.dstride;
#pragma unroll
    for (int c = 0; c < CP8; c += 8) {
        __align__(16) half o[8];
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = __float2half(acc[c + e]);
        *reinterpret_cast<uint4 *>(d + c) = *reinterpret_cast<uint4 *>(o);
    }
    for (int c = CP8; c < g.dstride; c += 8) *reinterpret_cast<uint4 *>(d + c) = make_uint4(0, 0, 0, 0);
}

using GatherFn = void (*)(Dev);
template <int F, int NM, int KV, int REV>
GatherFn pick_nt(int nt, bool b0)
{
    if (b0) return nt == 1 ? &k_gather<F, NM, KV, 1, 1, REV> : &k_gather<F, NM, KV, 2, 1, REV>;
    return nt == 1 ? &k_gather<F, NM, KV, 1, 0, REV> : &k_gather<F, NM, KV, 2, 0, REV>;
}
template <int F>
GatherFn pick_f(int nm, int kv, int nt, bool b0, bool rev)
{
    if (rev) {
        // ensembles: six-channel block outputs, no fed features
        if (nm == 1 && kv == 1) return pick_nt<F, 1, 1, 1>(nt, b0);
        return nullptr;
    }
    if (nm == 1 && kv == 1) return pick_nt<F, 1, 1, 0>(nt, b0);
    if (nm == 1 && kv == 2) return pick_nt<F, 1, 2, 0>(nt, b0);
    if (nm == 9 && kv == 2) return pick_nt<F, 9, 2, 0>(nt, b0);
    return nullptr;
}
GatherFn gather_fn(int F, int nm, int kv, int nt, bool b0, bool rev)
{
    switch (F) {
    case 0: return pick_f<0>(nm, kv, nt, b0, rev);
    case 4: return pick_f<4>(nm, kv, nt, b0, rev);
    case 8: return pick_f<8>(nm, kv, nt, b0, rev);
    case 16: return pick_f<16>(nm, kv, nt, b0, rev);
    }
    return nullptr;
}

__global__ void k_merge(RifeBlockOut o, const half *r, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    half *a = (half *)o.b + (size_t)i * o.kstride;
    const half *b = r + (size_t)i * o.kstride;
    float fl[4];
#pragma unroll
    for (int k = 0; k < 4; k++) fl[k] = rh((h2f(a[k]) + h2f(b[(k + 2) % 4])) * 0.5f);
#pragma unroll
    for (int k = 0; k < 4; k++) a[k] = __float2half(fl[k]);
    a[4] = __float2half((h2f(a[4]) - h2f(b[4])) * 0.5f);
}

int gather_nm(const RifeGather &g) { return g.nprev == 0 ? 1 : (g.feeds_feat ? g.K - 4 : 1); }
int gather_kv(const RifeGather &g) { return g.nprev == 0 ? 1 : g.prev[g.nprev - 1].kstride / 8; }

struct FinalDev {
    RifeFrame f;
    int nb;
    RifeBlockOut b[kRifeMaxBlocks];
    half *out;
};

__global__ void k_final(FinalDev p)
{
    const RifeFrame &f = p.f;
    const int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= f.W || Y >= f.H) return;
    float lv[8];
    up_vec<1>(p.b[p.nb - 1], f.W, f.H, X, Y, lv);  // flow and mask are channels 0-4
    float fl[4];
    flow_sum(p.b, p.nb, f.W, f.H, X, Y, lv, fl);
    float wa[3], wb[3];
    warp<0>(f, 0, X, Y, fl[0], fl[1], wa);
    warp<0>(f, 1, X, Y, fl[2], fl[3], wb);
    const float m = lv[4];
    const float sg = rh(1.f / (1.f + __expf(-m)));
    const float isg = rh(1.f - sg);
    const size_t plane = (size_t)f.W * f.H, pix = (size_t)Y * f.W + X;
    for (int c = 0; c < 3; c++) p.out[c * plane + pix] = __float2half(rh(wa[c] * sg) + rh(wb[c] * isg));
}

}  // namespace

bool rife_gather_supported(const RifeFrame &f, const RifeGather &g)
{
    return g.dstride % 8 == 0 && g.dstride >= (g.cin + 7) / 8 * 8 &&
           gather_fn(f.F, gather_nm(g), gather_kv(g), g.s == 1 ? 1 : 2, g.nprev == 0, g.rev) != nullptr;
}

cudaError_t rife_gather(const RifeFrame &f, const RifeGather &g, cudaStream_t s)
{
    GatherFn fn = gather_fn(f.F, gather_nm(g), gather_kv(g), g.s == 1 ? 1 : 2, g.nprev == 0, g.rev);
    if (!fn || g.dstride % 8) return cudaErrorInvalidValue;
    Dev p{f, g};
    const dim3 blk(16, 8), grid((f.W / g.s + 15) / 16, (f.H / g.s + 7) / 8);
    fn<<<grid, blk, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t rife_merge(const RifeFrame &f, const RifeBlockOut &o, const void *r, cudaStream_t s)
{
    const int n = (f.W / o.s) * (f.H / o.s);
    k_merge<<<(n + 255) / 256, 256, 0, s>>>(o, (const half *)r, n);
    return cudaGetLastError();
}

cudaError_t rife_final(const RifeFrame &f, int nblocks, const RifeBlockOut *blocks, void *out, cudaStream_t s)
{
    if (nblocks < 1 || nblocks > kRifeMaxBlocks) return cudaErrorInvalidValue;
    FinalDev p{};
    p.f = f;
    p.nb = nblocks;
    for (int i = 0; i < nblocks; i++) p.b[i] = blocks[i];
    p.out = (half *)out;
    const dim3 blk(16, 8), grid((f.W + 15) / 16, (f.H + 7) / 8);
    k_final<<<grid, blk, 0, s>>>(p);
    return cudaGetLastError();
}

}  // namespace ajinative
