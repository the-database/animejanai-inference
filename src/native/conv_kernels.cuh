/*
 * native/conv_kernels.cuh — device code for native/conv.h. Included by conv.cu only.
 *
 * k_conv3x3: a CTA computes TH x TW output pixels (TW = 16, TH = WARPS * WR) for all output
 * channels. Its input tile with a one-pixel halo is copied to shared memory (pixel stride
 * cin + 8 halfs, which makes ldmatrix conflict-free); each warp computes WR rows of 16 pixels as
 * an implicit GEMM over 9 taps x cin, A from shared memory via ldmatrix, B as pre-packed
 * fragments straight from global memory (L1-resident). The epilogue runs on the accumulator
 * registers, stages the tile in the (now free) shared memory and writes it out coalesced.
 */
#pragma once

#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>

namespace ajinative {
namespace dev {

constexpr int TW = 16;

struct ConvParams {
    const half *src;
    const half *resid;
    half *dst;
    const uint4 *w;
    const half2 *bias;
    const half2 *slope;
    const uint4 *watt;
    const half2 *batt;
    int H, W;
    // the model input normalized as it is loaded: (x - in_mean[c]) * in_scale[c] (NCHW inputs)
    int in_norm;
    float in_mean[3], in_scale[3];
    int out_clamp;  // depth-to-space outputs: clamp to [0, 1] (after the input is added)
};

struct SpanConvParams : ConvParams {
    bool reverse_tile_rows;
};

struct ConcatParams {
    const half *src[4];
    half *dst;
    const uint4 *w;
    const half2 *bias;
    int npx;
};

// Inputs to the four-map projection following the last SPANF3-48 gate.
struct GateProjectParams : SpanConvParams {
    const half *maps[4];
    const uint4 *project_w;
    const half2 *project_bias;
};

__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2 *>(&u); }
__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t *>(&h); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define AJI_NATIVE_ARCH_OK 1
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void *p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void *p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma1688(uint32_t (&d)[2], const uint32_t (&a)[2], uint32_t b0)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(b0));
}
__device__ __forceinline__ void mma16816(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void *s, const void *g, int bytes)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ half2 tanh_h2(half2 x)
{
    uint32_t y;
    asm("tanh.approx.f16x2 %0, %1;" : "=r"(y) : "r"(h2u(x)));
    return u2h(y);
}
#else
#define AJI_NATIVE_ARCH_OK 0
#endif

// The shared-memory pixel stride for c channels (halfs): an odd number of 16-byte units, which
// makes ldmatrix's eight row reads hit distinct banks (c + 8 for multiples of 16, c itself for 24).
__host__ __device__ constexpr int smem_stride(int c) { return (c / 8) % 2 ? c : c + 8; }

// Shared memory of a k_conv3x3 instance: max(halo tile, output staging tile).
template <int CINP, int COUTP, int WARPS, int WR>
__host__ __device__ constexpr int conv_smem_bytes()
{
    return (WARPS * WR + 2) * (TW + 2) * smem_stride(CINP) * 2 > WARPS * WR * TW * smem_stride(COUTP) * 2
               ? (WARPS * WR + 2) * (TW + 2) * smem_stride(CINP) * 2
               : WARPS * WR * TW * smem_stride(COUTP) * 2;
}

#include "native/span_fp8_kernels.inl"
#include "native/span48_project.inl"

// IN/ACT/GATE/OUT take the integer values of Input/Act/Gate/Output.
template <int CINP, int COUTP, int WARPS, int WR, int IN, int ACT, int GATE, int OUT, bool FP8 = false,
          bool PACKED_IN = false, bool PACKED_OUT = false, class Params = ConvParams>
__global__ void __launch_bounds__(WARPS * 32) k_conv3x3(Params p)
{
#if AJI_NATIVE_ARCH_OK
    constexpr int TH = WARPS * WR, KC = CINP / 16, NT = COUTP / 8, NJ = NT / 2;
    [[maybe_unused]] constexpr int CP = smem_stride(CINP), SP = smem_stride(COUTP);
    // 24-channel shapes: an 8-row K tail (m16n8k8) after the 16-row chunks, an odd last n8 tile
    constexpr int K8 = (CINP % 16) / 8, NO = NT % 2, WK = NJ * 4 + NO * 2;
    [[maybe_unused]] constexpr int WT = KC * WK + K8 * (NJ * 2 + NO);
    constexpr int HP = (TH + 2) * (TW + 2);
    static_assert(CINP % 8 == 0 && COUTP % 8 == 0 && NJ > 0, "channel counts are multiples of 8, at least 16");
    static_assert(GATE != 2 || COUTP % 16 == 0, "the tanh gate's 1x1 conv takes 16-channel K chunks");
    static_assert(IN == 0 || CINP == 16, "NCHW inputs are padded to 16 channels");
    extern __shared__ __align__(16) unsigned char smraw[];
    [[maybe_unused]] half *sm = reinterpret_cast<half *>(smraw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned tile_y = blockIdx.y;
    if constexpr (std::is_base_of_v<SpanConvParams, Params>) {
        if (p.reverse_tile_rows) tile_y = gridDim.y - 1 - tile_y;
    }
    const int tx0 = blockIdx.x * TW, ty0 = tile_y * TH, H = p.H, W = p.W;

    // ---- input tile with halo -> shared [HP][CP]
    static_assert(!(PACKED_IN || PACKED_OUT) || (FP8 && IN == 0 && OUT == 0));
    if constexpr (PACKED_IN) {
        load_packed_fp8_halo<CINP, WARPS, HP>(smraw, p, tx0, ty0);
    } else if constexpr (FP8) {
        load_fp8_halo<CINP,WARPS,HP>(smraw, p, tx0, ty0);
    } else if constexpr (IN == 0) {
        constexpr int VPP = CINP / 8;
        for (int i = tid; i < HP * VPP; i += WARPS * 32) {
            const int px = i / VPP, v = i % VPP;
            const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
            const bool ok = yy >= 0 && yy < H && xx >= 0 && xx < W;
            cp_async16(sm + px * CP + v * 8, ok ? p.src + ((size_t)yy * W + xx) * CINP + v * 8 : p.src, ok ? 16 : 0);
        }
        cp_async_wait_all();
    } else if constexpr (IN == 1) {
        // NCHW RGB [3][H][W] -> channels 0..2, rest zero
        for (int px = tid; px < HP; px += WARPS * 32) {
            const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
            __align__(16) half v[16];
#pragma unroll
            for (int c = 0; c < 16; c++) v[c] = __float2half(0.f);
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
#pragma unroll
                for (int c = 0; c < 3; c++) v[c] = p.src[((size_t)c * H + yy) * W + xx];
                if (p.in_norm)
#pragma unroll
                    for (int c = 0; c < 3; c++) v[c] = __float2half((__half2float(v[c]) - p.in_mean[c]) * p.in_scale[c]);
            }
            reinterpret_cast<uint4 *>(sm + px * CP)[0] = reinterpret_cast<uint4 *>(v)[0];
            reinterpret_cast<uint4 *>(sm + px * CP)[1] = reinterpret_cast<uint4 *>(v)[1];
        }
    } else {
        // NCHW RGB [3][2H][2W], pixel-unshuffle by 2: channel c*4 + ry*2 + rx (PyTorch order)
        for (int i = tid; i < HP * 6; i += WARPS * 32) {
            const int px = i / 6, k = i % 6, c = k >> 1, ry = k & 1;
            const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
            half2 v = __float2half2_rn(0.f);
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                v = __ldg(reinterpret_cast<const half2 *>(p.src + ((size_t)c * (2 * H) + 2 * yy + ry) * (2 * W) + 2 * xx));
                if (p.in_norm) {
                    const float2 f = __half22float2(v);
                    v = __floats2half2_rn((f.x - p.in_mean[c]) * p.in_scale[c], (f.y - p.in_mean[c]) * p.in_scale[c]);
                }
            }
            reinterpret_cast<half2 *>(sm + px * CP)[c * 2 + ry] = v;
        }
        for (int px = tid; px < HP; px += WARPS * 32) *reinterpret_cast<uint2 *>(sm + px * CP + 12) = make_uint2(0, 0);
    }
    __syncthreads();

    // ---- implicit GEMM
    uint32_t acc[WR][NT][2];
#pragma unroll
    for (int i = 0; i < WR; i++)
#pragma unroll
        for (int n = 0; n < NT; n++) acc[i][n][0] = acc[i][n][1] = 0u;
    const int row0 = warp * WR;
    if constexpr (FP8) {
        fp8_product<CINP,COUTP>(acc, smraw, p, row0, lane);
    } else {
        [[maybe_unused]] const uint32_t *wv = reinterpret_cast<const uint32_t *>(p.w);  // pack_fragments8's layout
#pragma unroll 1
        for (int t = 0; t < 9; t++) {
            const int dy = t / 3, dx = t % 3;
#pragma unroll
            for (int kc = 0; kc < KC; kc++) {
                uint4 b[NJ];
                [[maybe_unused]] uint2 bo;
                if constexpr (K8 || NO) {
                    const uint32_t *seg = wv + (t * WT + kc * WK) * 32;
#pragma unroll
                    for (int j = 0; j < NJ; j++) b[j] = __ldg(reinterpret_cast<const uint4 *>(seg) + j * 32 + lane);
                    if constexpr (NO) bo = __ldg(reinterpret_cast<const uint2 *>(seg + NJ * 4 * 32) + lane);
                } else {  // multiples of 16: pack_fragments' layout, addressed as it always was
#pragma unroll
                    for (int j = 0; j < NJ; j++) b[j] = __ldg(p.w + ((t * KC + kc) * NJ + j) * 32 + lane);
                }
#pragma unroll
                for (int i = 0; i < WR; i++) {
                    uint32_t a[4];
                    ldsm_x4(a, sm + ((row0 + i + dy) * (TW + 2) + (lane & 15) + dx) * CP + kc * 16 + (lane >> 4) * 8);
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        mma16816(acc[i][2 * j], a, b[j].x, b[j].y);
                        mma16816(acc[i][2 * j + 1], a, b[j].z, b[j].w);
                    }
                    if constexpr (NO) mma16816(acc[i][NT - 1], a, bo.x, bo.y);
                }
            }
            if constexpr (K8) {
                const uint32_t *seg = wv + (t * WT + KC * WK) * 32;
                uint2 b[NJ];
#pragma unroll
                for (int j = 0; j < NJ; j++) b[j] = __ldg(reinterpret_cast<const uint2 *>(seg) + j * 32 + lane);
                [[maybe_unused]] uint32_t bo;
                if constexpr (NO) bo = __ldg(seg + NJ * 2 * 32 + lane);
#pragma unroll
                for (int i = 0; i < WR; i++) {
                    uint32_t a[2];  // m16k8: rows 0-7 and 8-15 of the K tail
                    ldsm_x2(a, sm + ((row0 + i + dy) * (TW + 2) + (lane & 15) + dx) * CP + KC * 16);
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        mma1688(acc[i][2 * j], a, b[j].x);
                        mma1688(acc[i][2 * j + 1], a, b[j].y);
                    }
                    if constexpr (NO) mma1688(acc[i][NT - 1], a, bo);
                }
            }
        }
    }
    __syncthreads();  // the halo is no longer read: shared memory becomes the staging tile [TH*TW][SP]

    // accumulator layout: m-tile i (output row row0+i), n-tile n: lane holds pixels g and g+8,
    // channels n*8 + 2t and +1
    const int g = lane >> 2, tq = lane & 3;
    const half2 z2 = __float2half2_rn(0.f);
    if constexpr (GATE == 0) {
#pragma unroll
        for (int i = 0; i < WR; i++)
#pragma unroll
            for (int n = 0; n < NT; n++) {
                const half2 bb = __ldg(p.bias + n * 4 + tq);
                half2 sl = z2;
                if constexpr (ACT == 2) sl = __ldg(p.slope + n * 4 + tq);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    half2 v = __hadd2(u2h(acc[i][n][h]), bb);
                    if constexpr (ACT == 1) v = __hmax2(v, z2);
                    if constexpr (ACT == 2) v = __hfma2(__hmin2(v, z2), sl, __hmax2(v, z2));
                    if constexpr (ACT == 3) {  // x * sigmoid(x), sigmoid(x) = 0.5 tanh(x/2) + 0.5
                        const half2 hf = __float2half2_rn(0.5f);
                        v = __hmul2(v, __hfma2(tanh_h2(__hmul2(v, hf)), hf, hf));
                    }
                    const int pixel = (row0 + i) * TW + g + h * 8;
                    if constexpr (PACKED_OUT) {
                        static_assert(ACT == 1 && GATE == 0);
                        *reinterpret_cast<uint16_t *>(smraw + pixel * fp8_stride(COUTP) + n * 8 + tq * 2) = fp8_pair(h2u(v));
                    } else {
                        *reinterpret_cast<half2 *>(sm + pixel * SP + n * 8 + tq * 2) = v;
                    }
                }
            }
    } else {
        uint32_t o3[WR][NT][2];
#pragma unroll
        for (int i = 0; i < WR; i++)
#pragma unroll
            for (int n = 0; n < NT; n++) {
                const half2 bb = __ldg(p.bias + n * 4 + tq);
                o3[i][n][0] = h2u(__hadd2(u2h(acc[i][n][0]), bb));
                o3[i][n][1] = h2u(__hadd2(u2h(acc[i][n][1]), bb));
            }
        [[maybe_unused]] uint32_t ga[WR][NT][2];
        if constexpr (GATE == 2) {
            // 1x1 attention conv straight from the registers: the m16n8 accumulators of n-tiles
            // 2kc and 2kc+1 are exactly the m16k16 A fragment of K chunk kc
#pragma unroll
            for (int i = 0; i < WR; i++)
#pragma unroll
                for (int n = 0; n < NT; n++) ga[i][n][0] = ga[i][n][1] = 0u;
#pragma unroll
            for (int kc = 0; kc < COUTP / 16; kc++) {
                uint4 b[NJ];
#pragma unroll
                for (int j = 0; j < NJ; j++) b[j] = __ldg(p.watt + (kc * NJ + j) * 32 + lane);
#pragma unroll
                for (int i = 0; i < WR; i++) {
                    const uint32_t a[4] = {o3[i][2 * kc][0], o3[i][2 * kc][1], o3[i][2 * kc + 1][0], o3[i][2 * kc + 1][1]};
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        mma16816(ga[i][2 * j], a, b[j].x, b[j].y);
                        mma16816(ga[i][2 * j + 1], a, b[j].z, b[j].w);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < WR; i++) {
            const int oy = ty0 + row0 + i;
#pragma unroll
            for (int n = 0; n < NT; n++) {
                half2 ab = z2;
                if constexpr (GATE == 2) ab = __ldg(p.batt + n * 4 + tq);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int ox = tx0 + g + h * 8;
                    half2 x = z2;
                    if (oy < H && ox < W)
                        x = *reinterpret_cast<const half2 *>(p.resid + ((size_t)oy * W + ox) * COUTP + n * 8 + tq * 2);
                    const half2 o = u2h(o3[i][n][h]);
                    half2 v = __hadd2(x, o);
                    if constexpr (GATE == 2) v = __hmul2(v, tanh_h2(__hadd2(u2h(ga[i][n][h]), ab)));
                    if constexpr (GATE == 3) {  // sigmoid(o) - 0.5 = 0.5 tanh(o/2)
                        const half2 hf = __float2half2_rn(0.5f);
                        v = __hmul2(v, __hmul2(tanh_h2(__hmul2(o, hf)), hf));
                    }
                    if constexpr (std::is_same_v<Params, GateProjectParams>)
                        acc[i][n][h] = h2u(v);
                    else
                        *reinterpret_cast<half2 *>(sm + ((row0 + i) * TW + g + h * 8) * SP + n * 8 + tq * 2) = v;
                }
            }
        }
        if constexpr (std::is_same_v<Params, GateProjectParams>) {
            static_assert(GATE == 2 && CINP == 48 && COUTP == 48 && OUT == 0 && !PACKED_OUT);
            span48_project(p, acc, sm + row0 * TW * SP, tx0, ty0 + row0, lane);
        }
    }
    __syncthreads();

    if constexpr (OUT == 0) {
        // NHWC: each tile row is up to TW * COUTP contiguous halfs
        constexpr int elements = PACKED_OUT ? 16 : 8, VPP = COUTP / elements, VPR = TW * VPP;
        for (int q = tid; q < TH * VPR; q += WARPS * 32) {
            const int r = q / VPR, w = q % VPR, px = w / VPP, v = w % VPP, oy = ty0 + r, ox = tx0 + px;
            if (oy >= H || ox >= W) continue;
            const size_t offset = ((size_t)oy * W + ox) * COUTP + v * elements;
            if constexpr (PACKED_OUT) {
                *reinterpret_cast<uint4 *>(reinterpret_cast<unsigned char *>(p.dst) + offset) =
                    *reinterpret_cast<const uint4 *>(smraw + (r * TW + px) * fp8_stride(COUTP) + v * elements);
            } else {
                *reinterpret_cast<uint4 *>(p.dst + offset) =
                    *reinterpret_cast<const uint4 *>(sm + (r * TW + px) * SP + v * elements);
            }
        }
    } else {
        // depth-to-space by R (PyTorch PixelShuffle / ONNX DepthToSpace CRD): channel
        // c*R*R + ry*R + rx -> dst[c][R*y + ry][R*x + rx], plus the model input when given
        constexpr int R = OUT == 1 ? 2 : OUT == 2 ? 4 : OUT == 3 ? 1 : 3, CPO = (TW * R + 7) / 8;  // 8-column chunks per row
        static_assert(COUTP >= 3 * R * R, "depth-to-space needs 3*R*R channels");
        const int OH = R * H, OW = R * W;
        const bool add = p.resid != nullptr;
        for (int q = tid; q < 3 * TH * R * CPO; q += WARPS * 32) {
            const int k = q % CPO, ry = (q / CPO) % R, r = (q / (CPO * R)) % TH, c = q / (CPO * R * TH);
            const int oy = ty0 + r;
            if (oy >= H) continue;
            const int col0 = R * tx0 + 8 * k;  // first output column of the chunk
            __align__(16) half o[8];
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int lx = min((8 * k + e) / R, TW - 1), rx = (8 * k + e) % R, ox = tx0 + lx;
                half v = sm[(r * TW + lx) * SP + c * R * R + ry * R + rx];
                if (add && ox < W) v = __hadd(v, __ldg(p.resid + ((size_t)c * H + oy) * W + ox));
                if (p.out_clamp) v = __hmin(__hmax(v, __float2half(0.f)), __float2half(1.f));
                o[e] = v;
            }
            half *d = p.dst + ((size_t)c * OH + R * oy + ry) * OW + col0;
            const int tile_end = R * (tx0 + TW);  // columns of this tile only
            if (col0 + 8 <= OW && col0 + 8 <= tile_end && (OW % 8) == 0) {
                *reinterpret_cast<uint4 *>(d) = *reinterpret_cast<uint4 *>(o);
            } else {
#pragma unroll
                for (int e = 0; e < 8; e++)
                    if (col0 + e < OW && col0 + e < tile_end) d[e] = o[e];
            }
        }
    }
#else
    __trap();
#endif
}

// 1x1 conv over the virtual concat of NSRC NHWC tensors of C channels each, on flat pixel tiles
template <int C, int NSRC, int WARPS>
__host__ __device__ constexpr int concat_smem_bytes() { return NSRC * WARPS * 16 * (C + 8) * 2; }

template <int C, int NSRC, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) k_concat1x1(ConcatParams p)
{
#if AJI_NATIVE_ARCH_OK
    constexpr int M = WARPS * 16, CP = C + 8, NT = C / 8, NJ = NT / 2, KCS = C / 16, VPP = C / 8;
    extern __shared__ __align__(16) unsigned char smraw[];
    half *sm = reinterpret_cast<half *>(smraw);  // [NSRC][M][CP]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, p0 = blockIdx.x * M;
#pragma unroll
    for (int s = 0; s < NSRC; s++)
        for (int i = tid; i < M * VPP; i += WARPS * 32) {
            const int px = i / VPP, v = i % VPP;
            const bool ok = p0 + px < p.npx;
            cp_async16(sm + (s * M + px) * CP + v * 8, ok ? p.src[s] + (size_t)(p0 + px) * C + v * 8 : p.src[0], ok ? 16 : 0);
        }
    cp_async_wait_all();
    __syncthreads();
    uint32_t acc[NT][2];
#pragma unroll
    for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = 0u;
#pragma unroll
    for (int s = 0; s < NSRC; s++)
#pragma unroll
        for (int kc = 0; kc < KCS; kc++) {
            uint32_t a[4];
            ldsm_x4(a, sm + (s * M + warp * 16 + (lane & 15)) * CP + kc * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                const uint4 b = __ldg(p.w + ((s * KCS + kc) * NJ + j) * 32 + lane);
                mma16816(acc[2 * j], a, b.x, b.y);
                mma16816(acc[2 * j + 1], a, b.z, b.w);
            }
        }
    __syncthreads();
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        const half2 bb = __ldg(p.bias + n * 4 + tq);
#pragma unroll
        for (int h = 0; h < 2; h++)
            *reinterpret_cast<half2 *>(sm + (warp * 16 + g + h * 8) * CP + n * 8 + tq * 2) = __hadd2(u2h(acc[n][h]), bb);
    }
    __syncthreads();
    for (int i = tid; i < M * VPP; i += WARPS * 32) {
        const int px = i / VPP, v = i % VPP;
        if (p0 + px < p.npx)
            *reinterpret_cast<uint4 *>(p.dst + (size_t)(p0 + px) * C + v * 8) = *reinterpret_cast<const uint4 *>(sm + px * CP + v * 8);
    }
#else
    __trap();
#endif
}

}  // namespace dev
}  // namespace ajinative
