// native/sst_kernels.cu — see sst_kernels.h.
#include "native/sst_kernels.h"

#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t *>(&h); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define SST_OK 1
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void *p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void *p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void *s, const void *g)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
#else
#define SST_OK 0
#endif

// ---- RIB window attention (see sst_kernels.h)
constexpr int BQ = 128, BK = 64, VP = 40;  // queries per CTA (4 warps x 32 rows), keys per block, V row pitch

struct RibP {
    const half *qkv;
    int sq, kq, vq, H, W, heads, hd, ws, nwx, nqb;
    const half *qpos, *kpos;
    const half *gate;
    int sg;
    half *out;
    int so;
};

__device__ __forceinline__ int refl(int i, int n) { return i < n ? i : 2 * (n - 1) - i; }

template <int DK>  // q.k dims: 32 content + DK - 32 position (16 or 32)
__global__ void __launch_bounds__(128) k_ribattn(RibP p)
{
#if SST_OK
    constexpr int KP = DK + 8, KC = DK / 16, RP = DK - 32, QCH = 4 + RP / 8;  // QCH: 16-byte chunks per q / k row
    extern __shared__ __align__(16) unsigned char smraw[];
    // the queries pass through the K/V buffers' space (read once into registers, before the first key
    // block lands): 29-41 KB per CTA, 3 CTAs per SM
    half *sk = reinterpret_cast<half *>(smraw);  // [2][BK][KP]
    half *sv = sk + 2 * BK * KP;                 // [2][BK][VP]
    half *sq = sk;                               // [BQ][KP], before the first key block
    static_assert(BQ * KP <= 2 * BK * (KP + VP), "the queries fit in the K/V buffers");
    int *spx = reinterpret_cast<int *>(sv + 2 * BK * VP);  // [2][BK]: pixel of each key row of a block
    const int T = p.ws * p.ws;
    const int qb = (int)(blockIdx.x % p.nqb), wh = (int)(blockIdx.x / p.nqb), h = wh % p.heads, win = wh / p.heads;
    const int wy0 = (win / p.nwx) * p.ws, wx0 = (win % p.nwx) * p.ws;
    // queries in row order, or column by column in a window whose only padding is on the right, so that
    // whole blocks of queries in the padding can be skipped (their outputs are discarded)
    const bool cm = wx0 + p.ws > p.W && wy0 + p.ws <= p.H;
    auto qtok = [&](int i) { return cm ? (i % p.ws) * p.ws + i / p.ws : i; };  // window token of query i
    if (cm ? wx0 + qb * BQ / p.ws >= p.W : wy0 + qb * BQ / p.ws >= p.H) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const half *qtab = p.qpos + (size_t)h * T * RP, *ktab = p.kpos + (size_t)h * T * RP;
    auto pix = [&](int t) -> size_t { return (size_t)refl(wy0 + t / p.ws, p.H) * p.W + refl(wx0 + t % p.ws, p.W); };

    for (int i = tid; i < BQ * QCH; i += 128) {
        const int r = i / QCH, c = i % QCH, t = qtok(qb * BQ + r);
        cp16(sq + r * KP + c * 8, c < 4 ? p.qkv + pix(t) * p.sq + h * 32 + c * 8 : qtab + (size_t)t * RP + (c - 4) * 8);
    }
    // A key block's rows: their pixels come from spx[buf] (computed a block ahead, once per row, instead
    // of a division and modulo by the window size per 16-byte chunk)
    auto load_kv = [&](int kb, int buf) {
        half *dk = sk + buf * BK * KP, *dv = sv + buf * BK * VP;
        const int *px = spx + buf * BK;
        for (int i = tid; i < BK * (QCH + 4); i += 128) {
            const int r = i / (QCH + 4), c = i % (QCH + 4), t = kb + r;
            const half *row = p.qkv + (size_t)px[r] * p.sq + h * 32;
            if (c < 4) cp16(dk + r * KP + c * 8, row + p.kq + c * 8);
            else if (c < QCH) cp16(dk + r * KP + c * 8, ktab + (size_t)t * RP + (c - 4) * 8);
            else cp16(dv + r * VP + (c - QCH) * 8, row + p.vq + (c - QCH) * 8);
        }
    };
    const int nkb = T / BK;
    if (tid < BK) {
        spx[tid] = (int)pix(tid);
        if (nkb > 1) spx[BK + tid] = (int)pix(BK + tid);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t qa[2][KC][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int kc = 0; kc < KC; kc++) ldsm_x4(qa[mt][kc], sq + (warp * 32 + mt * 16 + (lane & 15)) * KP + kc * 16 + (lane >> 4) * 8);
    __syncthreads();  // every warp has its queries: the buffers take keys now
    load_kv(0, 0);
    cp_commit();
    float o[2][4][4], mrow[2][2], lrow[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        mrow[mt][0] = mrow[mt][1] = -1e30f;
        lrow[mt][0] = lrow[mt][1] = 0.f;
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) o[mt][n][e] = 0.f;
    }
    for (int kbi = 0; kbi < nkb; kbi++) {
        const int buf = kbi & 1;
        if (kbi + 1 < nkb) load_kv((kbi + 1) * BK, buf ^ 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        const half *ck = sk + buf * BK * KP, *cv = sv + buf * BK * VP;
        // block kbi + 2's pixels into spx[buf] (block kbi's, whose copies were issued last iteration)
        if (tid < BK && kbi + 2 < nkb) spx[buf * BK + tid] = (int)pix((kbi + 2) * BK + tid);
        float s[2][8][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) s[mt][n][e] = 0.f;
#pragma unroll
        for (int np = 0; np < 4; np++)
#pragma unroll
            for (int kc = 0; kc < KC; kc++) {
                uint32_t b[4];
                ldsm_x4(b, ck + (np * 16 + (lane & 7) + (lane >> 4) * 8) * KP + kc * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    mma32(s[mt][2 * np], qa[mt][kc], b[0], b[1]);
                    mma32(s[mt][2 * np + 1], qa[mt][kc], b[2], b[3]);
                }
            }
        // online softmax in the log2 domain (rows g: s[..][0..1], g + 8: s[..][2..3])
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            float mx[2] = {mrow[mt][0], mrow[mt][1]};
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) mx[e >> 1] = fmaxf(mx[e >> 1], s[mt][n][e]);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            }
            const float al0 = exp2f(mrow[mt][0] - mx[0]), al1 = exp2f(mrow[mt][1] - mx[1]);
            mrow[mt][0] = mx[0];
            mrow[mt][1] = mx[1];
            float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
            for (int n = 0; n < 8; n++) {
                s[mt][n][0] = exp2f(s[mt][n][0] - mx[0]);
                s[mt][n][1] = exp2f(s[mt][n][1] - mx[0]);
                s[mt][n][2] = exp2f(s[mt][n][2] - mx[1]);
                s[mt][n][3] = exp2f(s[mt][n][3] - mx[1]);
                ls0 += s[mt][n][0] + s[mt][n][1];
                ls1 += s[mt][n][2] + s[mt][n][3];
            }
            lrow[mt][0] = lrow[mt][0] * al0 + ls0;
            lrow[mt][1] = lrow[mt][1] * al1 + ls1;
#pragma unroll
            for (int n = 0; n < 4; n++) {
                o[mt][n][0] *= al0;
                o[mt][n][1] *= al0;
                o[mt][n][2] *= al1;
                o[mt][n][3] *= al1;
            }
        }
        // o += P V (P rounded to fp16)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t pa[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                pa[mt][0] = h2u(__floats2half2_rn(s[mt][2 * j][0], s[mt][2 * j][1]));
                pa[mt][1] = h2u(__floats2half2_rn(s[mt][2 * j][2], s[mt][2 * j][3]));
                pa[mt][2] = h2u(__floats2half2_rn(s[mt][2 * j + 1][0], s[mt][2 * j + 1][1]));
                pa[mt][3] = h2u(__floats2half2_rn(s[mt][2 * j + 1][2], s[mt][2 * j + 1][3]));
            }
#pragma unroll
            for (int dp = 0; dp < 2; dp++) {
                uint32_t b[4];
                ldsm_x4_t(b, cv + (j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * VP + dp * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    mma32(o[mt][2 * dp], pa[mt], b[0], b[1]);
                    mma32(o[mt][2 * dp + 1], pa[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();  // every warp is done with buf before the next iteration loads into it
    }
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            float l = lrow[mt][hh];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            const float inv = 1.f / l;
            const int t = qtok(qb * BQ + warp * 32 + mt * 16 + g + hh * 8), y = wy0 + t / p.ws, x = wx0 + t % p.ws;
            if (y >= p.H || x >= p.W) continue;
            const size_t px = (size_t)y * p.W + x;
#pragma unroll
            for (int n = 0; n < 4; n++) {
                const int d = n * 8 + tq * 2;
                if (d >= p.hd) continue;
                float v0 = o[mt][n][2 * hh] * inv, v1 = o[mt][n][2 * hh + 1] * inv;
                if (p.gate) {
                    const float2 gl = __half22float2(*reinterpret_cast<const half2 *>(p.gate + px * p.sg + h * p.hd + d));
                    v0 *= 1.f / (1.f + __expf(-gl.x));
                    v1 *= 1.f / (1.f + __expf(-gl.y));
                }
                *reinterpret_cast<half2 *>(p.out + px * p.so + h * p.hd + d) = __floats2half2_rn(v0, v1);
            }
        }
#else
    __trap();
#endif
}

// ---- depthwise 3x3 + bias, GELU, plus the input: one thread per (8 channels, column, strip of rows)
constexpr int DW_R = 8;
__device__ __forceinline__ void unpack8(const uint4 &u, float (&v)[8])
{
    const half2 *h = reinterpret_cast<const half2 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 f = __half22float2(h[j]);
        v[2 * j] = f.x;
        v[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ uint4 pack8(const float (&v)[8])
{
    uint4 u;
    uint32_t *q = reinterpret_cast<uint32_t *>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) q[j] = h2u(__floats2half2_rn(v[2 * j], v[2 * j + 1]));
    return u;
}

__global__ void __launch_bounds__(128) k_dwres(const half *in, int sin, half *out, int sout, int H, int W, int C, const float *w, const float *bias)
{
    const int G = C / 8, strips = (H + DW_R - 1) / DW_R;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= G * W * strips) return;
    const int c = (i % G) * 8, rest = i / G, x = rest % W, y0 = (rest / W) * DW_R;
    float wr[9][8], bs[8];
#pragma unroll
    for (int k = 0; k < 9; k++)
#pragma unroll
        for (int j = 0; j < 8; j++) wr[k][j] = __ldg(w + (c + j) * 9 + k);
#pragma unroll
    for (int j = 0; j < 8; j++) bs[j] = bias ? __ldg(bias + c + j) : 0.f;
    auto row = [&](int y, uint4 (&r)[3]) {
#pragma unroll
        for (int dx = 0; dx < 3; dx++) {
            const int xx = x + dx - 1;
            r[dx] = (y >= 0 && y < H && xx >= 0 && xx < W) ? *reinterpret_cast<const uint4 *>(in + ((size_t)y * W + xx) * sin + c) : make_uint4(0, 0, 0, 0);
        }
    };
    uint4 r0[3], r1[3], r2[3];
    row(y0 - 1, r0);
    row(y0, r1);
    for (int y = y0; y < min(H, y0 + DW_R); y++) {
        row(y + 1, r2);
        float a[8], v[8], ctr[8];
#pragma unroll
        for (int j = 0; j < 8; j++) a[j] = bs[j];
#pragma unroll
        for (int dx = 0; dx < 3; dx++) {
            unpack8(r0[dx], v);
#pragma unroll
            for (int j = 0; j < 8; j++) a[j] += v[j] * wr[dx][j];
            unpack8(r1[dx], v);
#pragma unroll
            for (int j = 0; j < 8; j++) a[j] += v[j] * wr[3 + dx][j];
            unpack8(r2[dx], v);
#pragma unroll
            for (int j = 0; j < 8; j++) a[j] += v[j] * wr[6 + dx][j];
        }
        unpack8(r1[1], ctr);
#pragma unroll
        for (int j = 0; j < 8; j++) a[j] = ctr[j] + 0.5f * a[j] * (1.f + erff(a[j] * 0.70710678f));
        *reinterpret_cast<uint4 *>(out + ((size_t)y * W + x) * sout + c) = pack8(a);
#pragma unroll
        for (int dx = 0; dx < 3; dx++) {
            r0[dx] = r1[dx];
            r1[dx] = r2[dx];
        }
    }
}

__global__ void k_sst_out(const half *o16, int so, const half *in16, int si, int W, int H, void *out, bool f32)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y, W4 = 4 * W, H4 = 4 * H;
    if (X >= W4 || Y >= H4) return;
    const size_t plane = (size_t)W4 * H4, pix = (size_t)Y * W4 + X;
    const half *a = o16 + pix * so, *b = in16 + ((size_t)(Y >> 2) * W + (X >> 2)) * si;
#pragma unroll
    for (int c = 0; c < 3; c++) {
        const float v = __half2float(a[c]) + __half2float(b[c]);
        if (f32) ((float *)out)[c * plane + pix] = v;
        else ((half *)out)[c * plane + pix] = __float2half(v);
    }
}

}  // namespace

cudaError_t tribattn(const TRibAttn &a, cudaStream_t s)
{
    const int T = a.ws * a.ws;
    if (a.hd < 2 || a.hd > 32 || a.hd % 2 || a.ws < 16 || T % BQ || (a.rp != 16 && a.rp != 32 && a.rp != 48) || a.sq % 8 || a.kq % 8 || a.vq % 8 ||
        a.so % 2 || (a.gate && a.sg % 2) || a.heads < 1 || a.H < 1 || a.W < 1 || (uintptr_t)a.qkv % 16 || (uintptr_t)a.qpos % 16 ||
        (uintptr_t)a.kpos % 16)
        return cudaErrorInvalidValue;
    const int nwy = (a.H + a.ws - 1) / a.ws, nwx = (a.W + a.ws - 1) / a.ws;
    if (nwy * a.ws - a.H > a.H - 1 || nwx * a.ws - a.W > a.W - 1) return cudaErrorInvalidValue;  // one reflection
    RibP p{(const half *)a.qkv, a.sq, a.kq, a.vq, a.H, a.W, a.heads, a.hd, a.ws, nwx, T / BQ, (const half *)a.qpos, (const half *)a.kpos,
           (const half *)a.gate, a.sg, (half *)a.out, a.so};
    const unsigned grid = (unsigned)(nwy * nwx) * (unsigned)a.heads * (unsigned)(T / BQ);
    auto launch = [&](auto dk) -> cudaError_t {
        constexpr int DK = decltype(dk)::value, KP = DK + 8;
        const int smem = (2 * BK * KP + 2 * BK * VP) * 2 + 2 * BK * 4;
        if (smem > 48 * 1024) {
            static bool set = false;
            if (!set && cudaFuncSetAttribute(k_ribattn<DK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return cudaErrorInvalidValue;
            set = true;
        }
        k_ribattn<DK><<<grid, 128, smem, s>>>(p);
        return cudaGetLastError();
    };
    if (a.rp == 16) return launch(std::integral_constant<int, 48>());
    if (a.rp == 32) return launch(std::integral_constant<int, 64>());
    return launch(std::integral_constant<int, 80>());
}

cudaError_t tdwres(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, const float *bias, cudaStream_t s)
{
    if (C % 8 || sin % 8 || sout % 8 || ((uintptr_t)in | (uintptr_t)out) % 16) return cudaErrorInvalidValue;
    const long long n = (long long)(C / 8) * W * ((H + DW_R - 1) / DW_R);
    k_dwres<<<(unsigned)((n + 127) / 128), 128, 0, s>>>((const half *)in, sin, (half *)out, sout, H, W, C, w, bias);
    return cudaGetLastError();
}

cudaError_t sst_out(const void *o16, int so, const void *in16, int si, int W, int H, void *out, bool out_f32, cudaStream_t s)
{
    const dim3 blk(32, 8), grid((unsigned)((4 * W + 31) / 32), (unsigned)((4 * H + 7) / 8));
    k_sst_out<<<grid, blk, 0, s>>>((const half *)o16, so, (const half *)in16, si, W, H, out, out_f32);
    return cudaGetLastError();
}

}  // namespace ajinative
