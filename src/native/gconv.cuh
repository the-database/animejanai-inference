/*
 * native/gconv.cuh — device code for native/gconv.h. Included by gconv.cu only.
 *
 * k_gconv: a CTA computes WARPS rows x 16 columns of output pixels for COUTP output channels;
 * each warp one row of 16 pixels (one m16 tile). blockIdx.z = output chunk * ksplit + K split.
 * The input tile with its halo is copied to shared memory one K chunk (KCH channels) at a time,
 * pixel stride KCH + 8 halfs (conflict-free ldmatrix). Stride-2 tiles are stored de-interleaved by
 * column parity (even columns, then odd), so the 16 rows of an A fragment are still consecutive
 * shared-memory pixels for every tap. Weights are B fragments packed on the host (pack.h), read
 * straight from global memory.
 *
 * Split K (small images, where a few CTAs would leave most SMs idle): the 9 taps x K chunks are
 * divided among ksplit CTAs, each writes fp32 partial sums to a workspace, and k_gconv_reduce sums
 * them and runs the epilogue.
 */
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace ajinative {
namespace gdev {

constexpr int TW = 16;

struct GParams {
    const half *src;
    int sstride, IH, IW, nchunks, kct;  // kct: total 16-channel K chunks (cin_pad / 16)
    half *dst;
    int dstride, H, W;
    const uint4 *w;
    size_t wchunk;                      // uint4s per output chunk
    const half2 *bias;
    const half *r1, *r2;
    int r1s, r2s;
    half2 c1, c2, slope;
    float c1f, c2f, slopef;
    int act, R, sy;
    int coutp_all, ksplit;              // all output channels; K splits (1: no workspace)
    int up2;                            // input is nearest-upsampled by 2 (IH, IW: the upsampled size)
    float *ws;                          // [ksplit][H*W][coutp_all]
    int outf;                           // fp32 output through the workspace and k_gconv_reduce
    float *dstf;
    const float *r1f;
    int r1fs;
    float *sums;                        // or null: the output's channel sums added here (NHWC, no split)
};

__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2 *>(&u); }
__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t *>(&h); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define AJI_GCONV_ARCH_OK 1
__device__ __forceinline__ void g_ldsm_x4(uint32_t (&r)[4], const void *p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void g_mma(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void g_mma32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void g_cp16(void *s, const void *g, int bytes)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void g_cp_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }
#else
#define AJI_GCONV_ARCH_OK 0
#endif

template <int KCH, int COUTP, int S, int WARPS>
__host__ __device__ constexpr int gconv_smem_bytes()
{
    constexpr int TR = S == 1 ? WARPS + 2 : 2 * WARPS + 1, TC = S == 1 ? TW + 2 : 2 * TW + 1;
    constexpr int halo = TR * TC * (KCH + 8) * 2, stage = WARPS * TW * (COUTP + 8) * 2;
    return halo > stage ? halo : stage;
}

// The epilogue on one value pair (channels ch, ch+1 of pixel (y, x)): residuals and activation.
__device__ __forceinline__ half2 g_epi(const GParams &p, half2 v, int y, int x, int ch)
{
    const half2 z2 = __float2half2_rn(0.f);
    const size_t pix = (size_t)y * p.W + x;
    if (p.r1) v = __hfma2(p.c1, __ldg(reinterpret_cast<const half2 *>(p.r1 + pix * p.r1s + ch)), v);
    if (p.r2) v = __hfma2(p.c2, __ldg(reinterpret_cast<const half2 *>(p.r2 + pix * p.r2s + ch)), v);
    if (p.act == 1) v = __hmax2(v, z2);
    else if (p.act == 2) v = __hfma2(__hmin2(v, z2), p.slope, __hmax2(v, z2));
    else if (p.act == 3) {
        const float2 f = __half22float2(v);
        v = __floats2half2_rn(0.5f * f.x * (1.f + erff(f.x / 1.4140625f)), 0.5f * f.y * (1.f + erff(f.y / 1.4140625f)));
    }
    return v;
}

// IN: 0 NHWC, 1 NCHW planes (3 channels, KCH 16). OUT: 0 NHWC, 1 phase scatter (R from params).
// ACC: 0 fp16 accumulation (full rate), 1 fp32 accumulation (half rate on GeForce, for deep sums).
// SUMS: the output's channel sums into p.sums (an instantiation of its own: the code would change how
// every other one compiles, which cost ESRGAN's convs ~20% on sm_89).
template <int KCH, int COUTP, int S, int IN, int OUT, int WARPS, int ACC = 0, int SUMS = 0>
__global__ void __launch_bounds__(WARPS * 32) k_gconv(GParams p)
{
#if AJI_GCONV_ARCH_OK
    constexpr int TH = WARPS, CP = KCH + 8, SP = COUTP + 8, NT = COUTP / 8, NJ = NT / 2, KCK = KCH / 16;
    [[maybe_unused]] constexpr int VPP = KCH / 8;
    constexpr int TR = S == 1 ? TH + 2 : 2 * TH + 1, TC = S == 1 ? TW + 2 : 2 * TW + 1;
    static_assert(KCH % 16 == 0 && COUTP % 16 == 0, "channel counts are multiples of 16");
    static_assert(IN == 0 || KCH == 16, "NCHW inputs are one 16-channel chunk");
    extern __shared__ __align__(16) unsigned char smraw[];
    half *sm = reinterpret_cast<half *>(smraw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tx0 = blockIdx.x * TW, ty0 = blockIdx.y * TH;
    const int iy0 = S * ty0 - 1, ix0 = S * tx0 - 1;
    const int och = blockIdx.z / p.ksplit, kz = blockIdx.z % p.ksplit;
    const uint4 *wt = p.w + och * p.wchunk;
    const int units = p.nchunks * 9, u0 = kz * units / p.ksplit, u1 = (kz + 1) * units / p.ksplit;

    uint32_t acc[NT][2];
    [[maybe_unused]] float accf[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++) {
        acc[n][0] = acc[n][1] = 0u;
#pragma unroll
        for (int k = 0; k < 4; k++) accf[n][k] = 0.f;
    }

#pragma unroll 1
    for (int c = u0 / 9; c < p.nchunks && c * 9 < u1; c++) {
        if constexpr (IN == 0) {
            for (int i = tid; i < TR * TC * VPP; i += WARPS * 32) {
                const int px = i / VPP, v = i % VPP, r = px / TC, col = px % TC;
                const int gc = S == 1 ? col : (col < TW + 1 ? 2 * col : 2 * (col - (TW + 1)) + 1);
                const int yy = iy0 + r, xx = ix0 + gc;
                const bool ok = yy >= 0 && yy < p.IH && xx >= 0 && xx < p.IW;
                const size_t sp = p.up2 ? (size_t)(yy >> 1) * (p.IW >> 1) + (xx >> 1) : (size_t)yy * p.IW + xx;
                g_cp16(sm + px * CP + v * 8, ok ? p.src + sp * p.sstride + c * KCH + v * 8 : p.src, ok ? 16 : 0);
            }
            g_cp_wait();
        } else {
            for (int px = tid; px < TR * TC; px += WARPS * 32) {
                const int r = px / TC, col = px % TC;
                const int gc = S == 1 ? col : (col < TW + 1 ? 2 * col : 2 * (col - (TW + 1)) + 1);
                const int yy = iy0 + r, xx = ix0 + gc;
                __align__(16) half v[16];
#pragma unroll
                for (int k = 0; k < 16; k++) v[k] = __float2half(0.f);
                if (yy >= 0 && yy < p.IH && xx >= 0 && xx < p.IW) {
#pragma unroll
                    for (int k = 0; k < 3; k++) v[k] = p.src[(size_t)k * p.sstride + (size_t)yy * p.IW + xx];
                }
                reinterpret_cast<uint4 *>(sm + px * CP)[0] = reinterpret_cast<uint4 *>(v)[0];
                reinterpret_cast<uint4 *>(sm + px * CP)[1] = reinterpret_cast<uint4 *>(v)[1];
            }
        }
        __syncthreads();
        const int t0 = max(u0 - c * 9, 0), t1 = min(u1 - c * 9, 9);
#pragma unroll 1
        for (int t = t0; t < t1; t++) {
            const int dy = t / 3, dx = t % 3, px = lane & 15;
            const int row = S == 1 ? warp + dy : 2 * warp + dy;
            const int col = S == 1 ? px + dx : ((dx & 1) ? TW + 1 + px : px + (dx >> 1));
            const half *ab = sm + (row * TC + col) * CP + (lane >> 4) * 8;
#pragma unroll
            for (int kc = 0; kc < KCK; kc++) {
                const int kk = c * KCK + kc;
                uint4 b[NJ];
#pragma unroll
                for (int j = 0; j < NJ; j++) b[j] = __ldg(wt + ((t * p.kct + kk) * NJ + j) * 32 + lane);
                uint32_t a[4];
                g_ldsm_x4(a, ab + kc * 16);
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    if constexpr (ACC) {
                        g_mma32(accf[2 * j], a, b[j].x, b[j].y);
                        g_mma32(accf[2 * j + 1], a, b[j].z, b[j].w);
                    } else {
                        g_mma(acc[2 * j], a, b[j].x, b[j].y);
                        g_mma(acc[2 * j + 1], a, b[j].z, b[j].w);
                    }
                }
            }
        }
        __syncthreads();
    }

    // epilogue: lane holds pixels g, g+8 of row `warp`, channels n*8 + 2tq, +1
    const int g = lane >> 2, tq = lane & 3, oy = ty0 + warp;
    if (p.ksplit > 1 || p.outf) {
        // fp32 partial sums to the workspace; k_gconv_reduce adds the bias and runs the epilogue
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int ox = tx0 + g + h * 8;
                if (oy >= p.H || ox >= p.W) continue;
                const float2 f = ACC ? make_float2(accf[n][2 * h], accf[n][2 * h + 1]) : __half22float2(u2h(acc[n][h]));
                *reinterpret_cast<float2 *>(p.ws + ((size_t)kz * p.H * p.W + (size_t)oy * p.W + ox) * p.coutp_all + och * COUTP + n * 8 +
                                            tq * 2) = f;
            }
        return;
    }
    const half2 *bias = p.bias + och * COUTP / 2;
    GParams q = p;
    if (q.r1) q.r1 += och * COUTP;
    if (q.r2) q.r2 += och * COUTP;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        const half2 bb = __ldg(bias + n * 4 + tq);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int ox = tx0 + g + h * 8;
            half2 v;
            if constexpr (ACC) {
                const float2 bf = __half22float2(bb);
                v = __floats2half2_rn(accf[n][2 * h] + bf.x, accf[n][2 * h + 1] + bf.y);
            } else {
                v = __hadd2(u2h(acc[n][h]), bb);
            }
            if (oy < p.H && ox < p.W) v = g_epi(q, v, oy, ox, n * 8 + tq * 2);
            *reinterpret_cast<half2 *>(sm + (warp * TW + g + h * 8) * SP + n * 8 + tq * 2) = v;
        }
    }
    __syncthreads();

    if constexpr (OUT == 0) {
        constexpr int VO = COUTP / 8, VPR = TW * VO;
        half *dst = p.dst + och * COUTP;
        for (int q2 = tid; q2 < TH * VPR; q2 += WARPS * 32) {
            const int r = q2 / VPR, w = q2 % VPR, px = w / VO, v = w % VO, yy = ty0 + r, xx = tx0 + px;
            if (yy >= p.H || xx >= p.W) continue;
            *reinterpret_cast<uint4 *>(dst + ((size_t)yy * p.W + xx) * p.dstride + v * 8) =
                *reinterpret_cast<const uint4 *>(sm + (r * TW + px) * SP + v * 8);
        }
        if constexpr (SUMS) {
            // the tile's channel sums of the stored (fp16) values, one atomic per channel per CTA
            for (int c = tid; c < COUTP; c += WARPS * 32) {
                float sum = 0.f;
                for (int r = 0; r < TH; r++)
                    if (ty0 + r < p.H)
                        for (int px = 0; px < TW && tx0 + px < p.W; px++) sum += __half2float(sm[(r * TW + px) * SP + c]);
                atomicAdd(p.sums + och * COUTP + c, sum);
            }
        }
    } else {
        // phase scatter: channel sx*CHP + c of all output channels -> dst[R*y + sy][R*x + sx][c]; this CTA
        // holds channels [och*COUTP, och*COUTP + COUTP) (CHP a multiple of 8, so no vector straddles sx)
        const int R = p.R, CHP = p.coutp_all / R, OW = R * p.W;
        constexpr int VO = COUTP / 8;
        for (int q2 = tid; q2 < TH * TW * VO; q2 += WARPS * 32) {
            const int v = q2 % VO, px = (q2 / VO) % TW, r = q2 / (VO * TW);
            const int yy = ty0 + r, xx = tx0 + px;
            if (yy >= p.H || xx >= p.W) continue;
            const int gc = och * COUTP + v * 8, sx = gc / CHP, ch = gc % CHP;
            *reinterpret_cast<uint4 *>(p.dst + ((size_t)(R * yy + p.sy) * OW + R * xx + sx) * p.dstride + ch) =
                *reinterpret_cast<const uint4 *>(sm + (r * TW + px) * SP + v * 8);
        }
    }
#else
    __trap();
#endif
}

// Sums the split-K partials, adds the bias, runs the epilogue and writes the output (NHWC or the
// phase scatter). One thread per (pixel, channel pair).
__global__ void k_gconv_reduce(GParams p)
{
    const int pairs = p.coutp_all / 2;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, npix = (size_t)p.H * p.W;
    if (i >= npix * pairs) return;
    const int ch = (int)(i % pairs) * 2;
    const size_t pix = i / pairs;
    const int y = (int)(pix / p.W), x = (int)(pix % p.W);
    float2 s = make_float2(0.f, 0.f);
    for (int k = 0; k < p.ksplit; k++) {
        const float2 v = *reinterpret_cast<const float2 *>(p.ws + ((size_t)k * npix + pix) * p.coutp_all + ch);
        s.x += v.x;
        s.y += v.y;
    }
    const float2 b = __half22float2(p.bias[ch / 2]);
    if (p.outf) {
        s.x += b.x;
        s.y += b.y;
        if (p.r1f) {
            const float2 r = *reinterpret_cast<const float2 *>(p.r1f + pix * p.r1fs + ch);
            s.x += r.x;
            s.y += r.y;
        }
        if (p.act == 1) {
            s.x = fmaxf(s.x, 0.f);
            s.y = fmaxf(s.y, 0.f);
        } else if (p.act == 2) {
            s.x = s.x < 0.f ? s.x * p.slopef : s.x;
            s.y = s.y < 0.f ? s.y * p.slopef : s.y;
        }
        *reinterpret_cast<float2 *>(p.dstf + pix * p.dstride + ch) = s;
        return;
    }
    half2 v = __floats2half2_rn(s.x + b.x, s.y + b.y);
    v = g_epi(p, v, y, x, ch);
    if (p.R == 0) {
        *reinterpret_cast<half2 *>(p.dst + pix * p.dstride + ch) = v;
    } else {
        const int CHP = p.coutp_all / p.R, sx = ch / CHP, c = ch % CHP;
        *reinterpret_cast<half2 *>(p.dst + ((size_t)(p.R * y + p.sy) * (p.R * p.W) + p.R * x + sx) * p.dstride + c) = v;
    }
}

}  // namespace gdev
}  // namespace ajinative
