// native/tfm.cu — see tfm.h.
#include "native/tfm.h"

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>

#include "native/pack.h"

namespace ajinative {
namespace {

__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t *>(&h); }
__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v / 1.4140625f)); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define TFM_OK 1
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
__device__ __forceinline__ void cp16(void *s, const void *g, int bytes)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }
#else
#define TFM_OK 0
#endif

// ---- GEMM: CTA = 64 rows x 64 columns, 4 warps of 16 rows, K in chunks of 64 through shared memory.
struct GemmP {
    const half *a;
    int sa, M, kpad;
    const uint4 *w;
    int njt;  // Npad / 16
    const float *bias, *colscale;
    half *out;
    int so;
    const half *add1, *add2;
    int s1, s2, act;
};

__global__ void __launch_bounds__(128) k_gemm(GemmP p)
{
#if TFM_OK
    constexpr int BM = 64, KC = 64, AP = KC + 8;
    __shared__ __align__(16) half sa[BM * AP];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // linear grid with the column tiles of a row block adjacent: they run together and share its A rows in L2
    const int ncol = (p.njt + 3) / 4, m0 = (int)(blockIdx.x / ncol) * BM, j0 = (int)(blockIdx.x % ncol) * 4;  // 4 n16 chunks = 64 columns
    float acc[8][4];
#pragma unroll
    for (int n = 0; n < 8; n++)
#pragma unroll
        for (int k = 0; k < 4; k++) acc[n][k] = 0.f;
    for (int k0 = 0; k0 < p.kpad; k0 += KC) {
        const int kn = min(KC, p.kpad - k0);
        for (int i = tid; i < BM * (KC / 8); i += 128) {
            const int r = i / (KC / 8), v = i % (KC / 8);
            const bool ok = m0 + r < p.M && v * 8 < kn;
            cp16(sa + r * AP + v * 8, ok ? p.a + (size_t)(m0 + r) * p.sa + k0 + v * 8 : p.a, ok ? 16 : 0);
        }
        cp_wait();
        __syncthreads();
        for (int kc = 0; kc < kn / 16; kc++) {
            uint32_t a[4];
            ldsm_x4(a, sa + (warp * 16 + (lane & 15)) * AP + kc * 16 + (lane >> 4) * 8);
            const int kk = k0 / 16 + kc;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (j0 + j >= p.njt) break;
                const uint4 b = __ldg(p.w + ((size_t)kk * p.njt + j0 + j) * 32 + lane);
                mma32(acc[2 * j], a, b.x, b.y);
                mma32(acc[2 * j + 1], a, b.z, b.w);
            }
        }
        __syncthreads();
    }
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int n = 0; n < 8; n++) {
        const int c = (j0 * 2 + n) * 8 + tq * 2;
        if (c >= p.njt * 16) break;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = m0 + warp * 16 + g + h * 8;
            if (r >= p.M) continue;
            float v0 = acc[n][2 * h], v1 = acc[n][2 * h + 1];
            if (p.bias) {
                v0 += p.bias[c];
                v1 += p.bias[c + 1];
            }
            if (p.act == 1) {
                v0 = gelu(v0);
                v1 = gelu(v1);
            }
            if (p.colscale) {
                v0 *= p.colscale[c];
                v1 *= p.colscale[c + 1];
            }
            if (p.add1) {
                const float2 f = __half22float2(*reinterpret_cast<const half2 *>(p.add1 + (size_t)r * p.s1 + c));
                v0 += f.x;
                v1 += f.y;
            }
            if (p.add2) {
                const float2 f = __half22float2(*reinterpret_cast<const half2 *>(p.add2 + (size_t)r * p.s2 + c));
                v0 += f.x;
                v1 += f.y;
            }
            *reinterpret_cast<half2 *>(p.out + (size_t)r * p.so + c) = __floats2half2_rn(v0, v1);
        }
    }
#else
    __trap();
#endif
}

// ---- LayerNorm: one warp per token
__global__ void k_ln(const half *in, int sin, half *out, int sout, int M, int C, int cpad, const float *gm, const float *bt, float eps)
{
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= M) return;
    const half *x = in + (size_t)t * sin;
    float v[8];
    int n = 0;
    float s = 0.f;
    for (int c = lane; c < C; c += 32) {
        v[n] = __half2float(x[c]);
        s += v[n++];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / C;
    float q = 0.f;
    for (int i = 0; i < n; i++) q += (v[i] - mean) * (v[i] - mean);
#pragma unroll
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rs = rsqrtf(q / C + eps);
    half *y = out + (size_t)t * sout;
    n = 0;
    for (int c = lane; c < C; c += 32) y[c] = __float2half((v[n++] - mean) * rs * gm[c] + bt[c]);
    for (int c = C + lane; c < cpad; c += 32) y[c] = __float2half(0.f);
}

// ---- window attention: one CTA per (window, head), 4 warps, query blocks of 16 rows per warp
constexpr int HP = 40;  // smem row pitch for 32-dim heads (conflict-free ldmatrix)

struct WinP {
    const half *qkv;
    int sq, qo, ko, vo, H, W, heads, hd, wh, ww, sy, sx, nwx;
    float scale;
    const float *bias;
    half *out;
    int so;
};

__device__ __forceinline__ bool win_tok(const WinP &p, int win, int t, int *pix)
{
    const int wy = win / p.nwx, wx = win % p.nwx;
    int y = wy * p.wh + t / p.ww, x = wx * p.ww + t % p.ww;
    if (y >= p.H || x >= p.W) return false;  // padding: zero q/k/v, no output
    if (p.sy || p.sx) {
        y = (y + p.sy) % p.H;
        x = (x + p.sx) % p.W;
    }
    *pix = y * p.W + x;
    return true;
}

__global__ void __launch_bounds__(128) k_winattn(WinP p)
{
#if TFM_OK
    extern __shared__ __align__(16) unsigned char smraw[];
    half *sq = reinterpret_cast<half *>(smraw);
    const int T = p.wh * p.ww;
    half *sk = sq + T * HP, *sv = sk + T * HP;
    const int win = (int)(blockIdx.x / p.heads), h = (int)(blockIdx.x % p.heads), tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // gather q, k, v of this head (dims padded to 32 with zeros)
    for (int i = tid; i < T * 16; i += 128) {
        const int t = i / 16, d2 = (i % 16) * 2;
        int pix;
        half2 q = __float2half2_rn(0.f), k = q, v = q;
        if (d2 < p.hd && win_tok(p, win, t, &pix)) {
            const half *src = p.qkv + (size_t)pix * p.sq + h * p.hd + d2;
            q = *reinterpret_cast<const half2 *>(src + p.qo);
            k = *reinterpret_cast<const half2 *>(src + p.ko);
            v = *reinterpret_cast<const half2 *>(src + p.vo);
        }
        *reinterpret_cast<half2 *>(sq + t * HP + d2) = q;
        *reinterpret_cast<half2 *>(sk + t * HP + d2) = k;
        *reinterpret_cast<half2 *>(sv + t * HP + d2) = v;
    }
    __syncthreads();
    const int g = lane >> 2, tq = lane & 3;
    const float *bias = p.bias ? p.bias + (size_t)h * T * T : nullptr;
    for (int qb = warp; qb * 16 < T; qb += 4) {
        uint32_t qa[2][4];
        ldsm_x4(qa[0], sq + (qb * 16 + (lane & 15)) * HP + (lane >> 4) * 8);
        ldsm_x4(qa[1], sq + (qb * 16 + (lane & 15)) * HP + 16 + (lane >> 4) * 8);
        float o[4][4], mrow[2] = {-1e30f, -1e30f}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int k = 0; k < 4; k++) o[n][k] = 0.f;
        const int r0 = qb * 16 + g, r1 = r0 + 8;
        for (int kb = 0; kb < T; kb += 64) {
            const int nk = min(64, T - kb);  // a multiple of 16
            float s[8][4];
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int k = 0; k < 4; k++) s[n][k] = 0.f;
#pragma unroll
            for (int np = 0; np < 4; np++) {
                if (np * 16 >= nk) break;
#pragma unroll
                for (int kc = 0; kc < 2; kc++) {
                    uint32_t b[4];
                    ldsm_x4(b, sk + (kb + np * 16 + (lane & 7) + (lane >> 4) * 8) * HP + kc * 16 + ((lane >> 3) & 1) * 8);
                    mma32(s[2 * np], qa[kc], b[0], b[1]);
                    mma32(s[2 * np + 1], qa[kc], b[2], b[3]);
                }
            }
            // scale, bias, online softmax (rows r0: s[..][0..1], r1: s[..][2..3])
            float mx[2] = {mrow[0], mrow[1]};
#pragma unroll
            for (int n = 0; n < 8; n++) {
                if (n * 8 >= nk) break;
                const int c = kb + n * 8 + tq * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = e < 2 ? r0 : r1, cc = c + (e & 1);
                    float v = s[n][e] * p.scale;
                    if (bias) v += bias[(size_t)r * T + cc];
                    s[n][e] = v;
                    mx[e >> 1] = fmaxf(mx[e >> 1], v);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            }
            const float al0 = __expf(mrow[0] - mx[0]), al1 = __expf(mrow[1] - mx[1]);
            mrow[0] = mx[0];
            mrow[1] = mx[1];
            float ls[2] = {0.f, 0.f};
#pragma unroll
            for (int n = 0; n < 8; n++) {
                if (n * 8 >= nk) break;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    s[n][e] = __expf(s[n][e] - mx[e >> 1]);
                    ls[e >> 1] += s[n][e];
                }
            }
            lrow[0] = lrow[0] * al0 + ls[0];
            lrow[1] = lrow[1] * al1 + ls[1];
#pragma unroll
            for (int n = 0; n < 4; n++) {
                o[n][0] *= al0;
                o[n][1] *= al0;
                o[n][2] *= al1;
                o[n][3] *= al1;
            }
            // o += P V (P from the score accumulators, fp16)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (j * 16 >= nk) break;
                uint32_t pa[4];
                pa[0] = h2u(__floats2half2_rn(s[2 * j][0], s[2 * j][1]));
                pa[1] = h2u(__floats2half2_rn(s[2 * j][2], s[2 * j][3]));
                pa[2] = h2u(__floats2half2_rn(s[2 * j + 1][0], s[2 * j + 1][1]));
                pa[3] = h2u(__floats2half2_rn(s[2 * j + 1][2], s[2 * j + 1][3]));
#pragma unroll
                for (int dp = 0; dp < 2; dp++) {
                    uint32_t b[4];
                    ldsm_x4_t(b, sv + (kb + j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * HP + dp * 16 + (lane >> 4) * 8);
                    mma32(o[2 * dp], pa, b[0], b[1]);
                    mma32(o[2 * dp + 1], pa, b[2], b[3]);
                }
            }
        }
        lrow[0] += __shfl_xor_sync(0xffffffffu, lrow[0], 1);
        lrow[0] += __shfl_xor_sync(0xffffffffu, lrow[0], 2);
        lrow[1] += __shfl_xor_sync(0xffffffffu, lrow[1], 1);
        lrow[1] += __shfl_xor_sync(0xffffffffu, lrow[1], 2);
        const float i0 = 1.f / lrow[0], i1 = 1.f / lrow[1];
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = hh ? r1 : r0;
            int pix;
            if (r >= T || !win_tok(p, win, r, &pix)) continue;
            const float inv = hh ? i1 : i0;
#pragma unroll
            for (int n = 0; n < 4; n++) {
                const int d = n * 8 + tq * 2;
                if (d >= p.hd) continue;
                *reinterpret_cast<half2 *>(p.out + (size_t)pix * p.so + h * p.hd + d) =
                    __floats2half2_rn(o[n][2 * hh] * inv, o[n][2 * hh + 1] * inv);
            }
        }
    }
#else
    __trap();
#endif
}

// ---- depthwise 3x3: one thread per (token, channel pair)
__global__ void k_dw(const half *in, int sin, half *out, int sout, int H, int W, int C, const float *w, int gelu_)
{
    const int pairs = (C + 1) / 2;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)H * W * pairs) return;
    const int c = (int)(i % pairs) * 2, t = (int)(i / pairs), y = t / W, x = t % W;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++)
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            const float2 v = __half22float2(*reinterpret_cast<const half2 *>(in + ((size_t)yy * W + xx) * sin + c));
            const int k = (dy + 1) * 3 + dx + 1;
            a0 += v.x * w[c * 9 + k];
            if (c + 1 < C) a1 += v.y * w[(c + 1) * 9 + k];
        }
    if (gelu_) {
        a0 = gelu(a0);
        a1 = gelu(a1);
    }
    if (c + 1 >= C) a1 = 0.f;
    *reinterpret_cast<half2 *>(out + (size_t)t * sout + c) = __floats2half2_rn(a0, a1);
}

// ---- channel sums: CTA of 256 threads over 256-token chunks
__global__ void k_colsum(const half *in, int sin, int M, int C, float *sums)
{
    const int t0 = blockIdx.x * 256;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float s = 0.f;
        for (int t = t0; t < min(M, t0 + 256); t++) s += __half2float(in[(size_t)t * sin + c]);
        atomicAdd(sums + c, s);
    }
}

// ---- channel attention statistics: CTA over 256 tokens in sub-chunks of 64 (q, k slices in smem)
__global__ void __launch_bounds__(256) k_chanstats(const half *qkv, int sq, int qo, int ko, int M, int heads, int hd, float *stats)
{
    extern __shared__ __align__(16) unsigned char smraw[];
    const int C = heads * hd;
    half *s_q = reinterpret_cast<half *>(smraw), *s_k = s_q + 64 * C;
    const int pairs = heads * hd * hd, total = pairs + 2 * C;
    float acc[24];
    for (int i = 0; i < 24; i++) acc[i] = 0.f;
    const int t0 = blockIdx.x * 256;
    for (int sub = 0; sub < 4; sub++) {
        const int tb = t0 + sub * 64;
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * C; i += 256) {
            const int t = i / C, c = i % C;
            const bool ok = tb + t < M;
            s_q[i] = ok ? qkv[(size_t)(tb + t) * sq + qo + c] : __float2half(0.f);
            s_k[i] = ok ? qkv[(size_t)(tb + t) * sq + ko + c] : __float2half(0.f);
        }
        __syncthreads();
        int n = 0;
        for (int id = threadIdx.x; id < total; id += 256, n++) {
            float s = 0.f;
            if (id < pairs) {
                const int h = id / (hd * hd), d = (id / hd) % hd, e = id % hd;
                for (int t = 0; t < 64; t++) s += __half2float(s_q[t * C + h * hd + d]) * __half2float(s_k[t * C + h * hd + e]);
            } else if (id < pairs + C) {
                const int c = id - pairs;
                for (int t = 0; t < 64; t++) {
                    const float v = __half2float(s_q[t * C + c]);
                    s += v * v;
                }
            } else {
                const int c = id - pairs - C;
                for (int t = 0; t < 64; t++) {
                    const float v = __half2float(s_k[t * C + c]);
                    s += v * v;
                }
            }
            acc[n] += s;
        }
    }
    int n = 0;
    for (int id = threadIdx.x; id < total; id += 256, n++) atomicAdd(stats + id, acc[n]);
}

// ---- vector paths (8 channels per thread, uint4 access): strides and channel offsets multiples of 8

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
    uint32_t *p = reinterpret_cast<uint32_t *>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) p[j] = h2u(__floats2half2_rn(v[2 * j], v[2 * j + 1]));
    return u;
}

// LayerNorm: L lanes per token (cpad <= 8 L), each one uint4 of channels.
template <int L>
__global__ void __launch_bounds__(256) k_ln8(const half *in, int sin, half *out, int sout, int M, int C, int cpad, const float *gm,
                                             const float *bt, float eps)
{
    const int sub = threadIdx.x % L, t = (blockIdx.x * blockDim.x + threadIdx.x) / L, c = sub * 8;
    const bool ok = t < M, live = ok && c < cpad;
    float v[8] = {};
    if (live) unpack8(*reinterpret_cast<const uint4 *>(in + (size_t)t * sin + c), v);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (c + j >= C) v[j] = 0.f;
        s += v[j];
    }
#pragma unroll
    for (int o = L / 2; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / C;
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++)
        if (c + j < C) q += (v[j] - mean) * (v[j] - mean);
#pragma unroll
    for (int o = L / 2; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rs = rsqrtf(q / C + eps);
    if (!live) return;
    float y[8];
#pragma unroll
    for (int j = 0; j < 8; j++) y[j] = c + j < C ? (v[j] - mean) * rs * gm[c + j] + bt[c + j] : 0.f;
    *reinterpret_cast<uint4 *>(out + (size_t)t * sout + c) = pack8(y);
}

// Depthwise 3x3, one thread per (8 channels, column, strip of DW_R rows): the 3x3 window slides down
// the strip so each output row loads one new input row (3 pixels), weights held in registers.
constexpr int DW_R = 8;
__global__ void __launch_bounds__(128) k_dw8s(const half *in, int sin, half *out, int sout, int H, int W, int C, const float *w, int gelu_,
                                              float *sums)
{
    __shared__ float ss[1024];
    if (sums) {
        for (int i = threadIdx.x; i < C; i += blockDim.x) ss[i] = 0.f;
        __syncthreads();
    }
    const int G = C / 8, strips = (H + DW_R - 1) / DW_R;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = i < G * W * strips;
    const int g = i % G, rest = i / G, x = rest % W, y0 = (rest / W) * DW_R, c = g * 8;
    float part[8] = {};
    if (live) {
        float wr[9][8];
#pragma unroll
        for (int k = 0; k < 9; k++)
#pragma unroll
            for (int j = 0; j < 8; j++) wr[k][j] = __ldg(w + (c + j) * 9 + k);
        auto row = [&](int y, uint4 (&r)[3]) {
#pragma unroll
            for (int dx = 0; dx < 3; dx++) {
                const int xx = x + dx - 1;
                r[dx] = (y >= 0 && y < H && xx >= 0 && xx < W) ? *reinterpret_cast<const uint4 *>(in + ((size_t)y * W + xx) * sin + c)
                                                               : make_uint4(0, 0, 0, 0);
            }
        };
        uint4 r0[3], r1[3], r2[3];
        row(y0 - 1, r0);
        row(y0, r1);
#pragma unroll 2
        for (int y = y0; y < min(H, y0 + DW_R); y++) {
            row(y + 1, r2);
            float a[8] = {};
#pragma unroll
            for (int dx = 0; dx < 3; dx++) {
                float v[8];
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
            if (gelu_)
#pragma unroll
                for (int j = 0; j < 8; j++) a[j] = gelu(a[j]);
            *reinterpret_cast<uint4 *>(out + ((size_t)y * W + x) * sout + c) = pack8(a);
#pragma unroll
            for (int j = 0; j < 8; j++) part[j] += a[j];
#pragma unroll
            for (int dx = 0; dx < 3; dx++) {
                r0[dx] = r1[dx];
                r1[dx] = r2[dx];
            }
        }
    }
    if (sums) {
        if (live)
#pragma unroll
            for (int j = 0; j < 8; j++) atomicAdd(ss + c + j, part[j]);
        __syncthreads();
        for (int k = threadIdx.x; k < C; k += blockDim.x) atomicAdd(sums + k, ss[k]);
    }
}

// Channel attention statistics on tensor cores: per head the Gram matrix q^T k over the CTA's tokens
// (64-token chunks in shared memory, each head's dims padded to 32), warp w doing heads w and w + 4;
// the squared norms on the CUDA cores while copying.
constexpr int CS_MAXH = 8;
__global__ void __launch_bounds__(128) k_chanstats_mma(const half *qkv, int sq, int qo, int ko, int M, int heads, int hd, int tpc, float *stats)
{
#if TFM_OK
    extern __shared__ __align__(16) unsigned char smraw[];
    const int P = heads * 32 + 8, C = heads * hd, pairs = C / 2;
    half *s_q = reinterpret_cast<half *>(smraw), *s_k = s_q + 64 * P;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 64 * P; i += 128) s_q[i] = s_k[i] = __float2half(0.f);
    float acc[2][2][4][4] = {};
    float nrm[2][2] = {};
    const int t0 = blockIdx.x * tpc, t1 = min(M, t0 + tpc);
    for (int tb = t0; tb < t1; tb += 64) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int sl = tid + 128 * j;
            if (sl >= 2 * pairs) continue;
            const bool isk = sl >= pairs;
            const int c = 2 * (sl - (isk ? pairs : 0)), h = c / hd, d = c % hd;
            const half *src = qkv + (size_t)tb * sq + (isk ? ko : qo) + c;
            half *dst = (isk ? s_k : s_q) + h * 32 + d;
            float n0 = 0.f, n1 = 0.f;
#pragma unroll 8
            for (int r = 0; r < 64; r++) {
                half2 v = __float2half2_rn(0.f);
                if (tb + r < t1) v = *reinterpret_cast<const half2 *>(src + (size_t)r * sq);
                *reinterpret_cast<half2 *>(dst + r * P) = v;
                const float2 f = __half22float2(v);
                n0 += f.x * f.x;
                n1 += f.y * f.y;
            }
            nrm[j][0] += n0;
            nrm[j][1] += n1;
        }
        __syncthreads();
#pragma unroll
        for (int hi = 0; hi < 2; hi++) {
            const int h = warp + 4 * hi;
            if (h >= heads) continue;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const int r = ks * 16 + (lane & 7), mj = lane >> 3;
                uint32_t a[2][4], b[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; mt++)  // A[d][t] = s_q[t][d]: matrix j at k + (j>>1)*8, m + (j&1)*8
                    ldsm_x4_t(a[mt], s_q + (size_t)(r + (mj >> 1) * 8) * P + h * 32 + mt * 16 + (mj & 1) * 8);
#pragma unroll
                for (int np = 0; np < 2; np++)  // B[t][e] = s_k[t][e]: matrix j at k + (j&1)*8, n + (j>>1)*8
                    ldsm_x4_t(b[np], s_k + (size_t)(r + (mj & 1) * 8) * P + h * 32 + np * 16 + (mj >> 1) * 8);
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) mma32(acc[hi][mt][nt], a[mt], b[nt >> 1][(nt & 1) * 2], b[nt >> 1][(nt & 1) * 2 + 1]);
            }
        }
    }
#pragma unroll
    for (int hi = 0; hi < 2; hi++) {
        const int h = warp + 4 * hi;
        if (h >= heads) continue;
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const int d = mt * 16 + lane / 4 + (q >> 1) * 8, e = nt * 8 + (lane % 4) * 2 + (q & 1);
                    if (d < hd && e < hd) atomicAdd(stats + ((size_t)h * hd + d) * hd + e, acc[hi][mt][nt][q]);
                }
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int sl = tid + 128 * j;
        if (sl >= 2 * pairs) continue;
        const bool isk = sl >= pairs;
        const int c = 2 * (sl - (isk ? pairs : 0));
        float *dst = stats + heads * hd * hd + (isk ? C : 0) + c;
        atomicAdd(dst, nrm[j][0]);
        atomicAdd(dst + 1, nrm[j][1]);
    }
#endif
}

// pack.h's layout: per (k16 chunk kc, n16 chunk j, lane) one uint4 = {b0, b1} of n-tile 2j, then of
// 2j + 1; b0 = {B[2t][g], B[2t+1][g]}, b1 = {B[2t+8][g], B[2t+9][g]}, lane = g*4 + t.
__global__ void k_pack(const float *w, int K, int N, int Kpad, int Npad, half *frag)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)Kpad * Npad) return;
    const int k = (int)(i / Npad), n = (int)(i % Npad);
    const float v = k < K && n < N ? w[(size_t)k * N + n] : 0.f;
    const int kc = k / 16, r = k % 16, nt = n / 8, g = n % 8, j = nt / 2, sub = nt % 2;
    const int hi = r >= 8, rr = r % 8, t = rr / 2, lo = rr % 2, lane = g * 4 + t;
    const size_t word = (((size_t)kc * (Npad / 16) + j) * 32 + lane) * 4 + sub * 2 + hi;
    frag[word * 2 + lo] = __float2half(v);
}

}  // namespace

cudaError_t tmat_alloc(TMat &m, int K, int N, const float *bias_dev)
{
    m = TMat{};
    m.K = K;
    m.N = N;
    m.Kpad = (K + 15) / 16 * 16;
    m.Npad = (N + 15) / 16 * 16;
    cudaError_t e = cudaMalloc(&m.frag, (size_t)m.Kpad * m.Npad * 2);
    if (e == cudaSuccess) e = cudaMemset(m.frag, 0, (size_t)m.Kpad * m.Npad * 2);
    if (e == cudaSuccess && bias_dev) {
        e = cudaMalloc(&m.bias, (size_t)m.Npad * 4);
        if (e == cudaSuccess) e = cudaMemset(m.bias, 0, (size_t)m.Npad * 4);
        if (e == cudaSuccess) e = cudaMemcpy(m.bias, bias_dev, (size_t)N * 4, cudaMemcpyDeviceToDevice);
    }
    if (e != cudaSuccess) tmat_release(m);
    return e;
}

cudaError_t tmat_pack(TMat &m, const float *w_dev, cudaStream_t s)
{
    const size_t n = (size_t)m.Kpad * m.Npad;
    k_pack<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(w_dev, m.K, m.N, m.Kpad, m.Npad, (half *)m.frag);
    return cudaGetLastError();
}

cudaError_t tmat_upload(TMat &m, const float *w, const float *bias, int K, int N)
{
    m = TMat{};
    m.K = K;
    m.N = N;
    m.Kpad = (K + 15) / 16 * 16;
    m.Npad = (N + 15) / 16 * 16;
    std::vector<uint16_t> wh((size_t)K * N);
    for (size_t i = 0; i < wh.size(); i++) wh[i] = float_to_half_bits(w[i]);
    std::vector<uint32_t> f = pack_fragments(wh.data(), 1, K, m.Kpad, N, m.Npad);
    std::vector<float> b(m.Npad, 0.f);
    if (bias)
        for (int i = 0; i < N; i++) b[i] = bias[i];
    cudaError_t e = cudaMalloc(&m.frag, f.size() * 4);
    if (e == cudaSuccess) e = cudaMemcpy(m.frag, f.data(), f.size() * 4, cudaMemcpyHostToDevice);
    if (e == cudaSuccess && bias) e = cudaMalloc(&m.bias, b.size() * 4);
    if (e == cudaSuccess && bias) e = cudaMemcpy(m.bias, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) tmat_release(m);
    return e;
}

void tmat_release(TMat &m)
{
    cudaFree(m.frag);
    cudaFree(m.bias);
    m = TMat{};
}

cudaError_t tgemm(const TMat &w, const TGemm &g, cudaStream_t s)
{
    if (g.sa % 8 || g.so % 2 || (g.add1 && g.s1 % 2) || (g.add2 && g.s2 % 2) || g.sa < w.Kpad || g.so < w.Npad)
        return cudaErrorInvalidValue;
    GemmP p{(const half *)g.a, g.sa, g.M, w.Kpad, (const uint4 *)w.frag, w.Npad / 16, (const float *)w.bias, g.colscale,
            (half *)g.out, g.so, (const half *)g.add1, (const half *)g.add2, g.s1, g.s2, (int)g.act};
    const unsigned grid = (unsigned)((g.M + 63) / 64) * (unsigned)((w.Npad / 16 + 3) / 4);
    k_gemm<<<grid, 128, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t tlayernorm(const void *in, int sin, void *out, int sout, int M, int C, int cpad, const float *gamma, const float *beta,
                       float eps, cudaStream_t s)
{
    if (C > 256) return cudaErrorInvalidValue;
    const bool vec = cpad % 8 == 0 && cpad <= 256 && sin % 8 == 0 && sout % 8 == 0 && ((uintptr_t)in | (uintptr_t)out) % 16 == 0;
    if (vec && cpad <= 128)
        k_ln8<16><<<(M + 15) / 16, 256, 0, s>>>((const half *)in, sin, (half *)out, sout, M, C, cpad, gamma, beta, eps);
    else if (vec)
        k_ln8<32><<<(M + 7) / 8, 256, 0, s>>>((const half *)in, sin, (half *)out, sout, M, C, cpad, gamma, beta, eps);
    else
        k_ln<<<(M + 7) / 8, 256, 0, s>>>((const half *)in, sin, (half *)out, sout, M, C, cpad, gamma, beta, eps);
    return cudaGetLastError();
}

cudaError_t twinattn(const TWinAttn &a, cudaStream_t s)
{
    const int T = a.wh * a.ww;
    if (a.hd > 32 || a.hd % 2 || T % 16 || T > 256 || a.sq % 2 || a.so % 2) return cudaErrorInvalidValue;
    const int nwx = (a.W + a.ww - 1) / a.ww, nwy = (a.H + a.wh - 1) / a.wh;
    WinP p{(const half *)a.qkv, a.sq, a.qo, a.ko, a.vo, a.H, a.W, a.heads, a.hd, a.wh, a.ww, a.shift_y, a.shift_x, nwx, a.scale, a.bias,
           (half *)a.out, a.so};
    const int smem = 3 * T * HP * 2;
    if (smem > 48 * 1024) {
        static bool set = false;
        if (!set) {
            cudaFuncSetAttribute(k_winattn, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * 256 * HP * 2);
            set = true;
        }
    }
    k_winattn<<<(unsigned)(nwx * nwy) * (unsigned)a.heads, 128, smem, s>>>(p);
    return cudaGetLastError();
}

cudaError_t tdwconv(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, bool gelu_, float *sums,
                    cudaStream_t s)
{
    const bool vec = C % 8 == 0 && C <= 1024 && sin % 8 == 0 && sout % 8 == 0 && ((uintptr_t)in | (uintptr_t)out) % 16 == 0;
    if (vec) {
        const long long n = (long long)(C / 8) * W * ((H + DW_R - 1) / DW_R);
        if (n < (1ll << 31)) {
            k_dw8s<<<(unsigned)((n + 127) / 128), 128, 0, s>>>((const half *)in, sin, (half *)out, sout, H, W, C, w, gelu_ ? 1 : 0, sums);
            return cudaGetLastError();
        }
    }
    const size_t n = (size_t)H * W * ((C + 1) / 2);
    k_dw<<<(unsigned)((n + 255) / 256), 256, 0, s>>>((const half *)in, sin, (half *)out, sout, H, W, C, w, gelu_ ? 1 : 0);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess && sums) e = tcolsum(out, sout, H * W, C, sums, s);
    return e;
}

cudaError_t tdwconv(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, bool gelu_, cudaStream_t s)
{
    return tdwconv(in, sin, out, sout, H, W, C, w, gelu_, nullptr, s);
}

cudaError_t tcolsum(const void *in, int sin, int M, int C, float *sums, cudaStream_t s)
{
    k_colsum<<<(M + 255) / 256, 128, 0, s>>>((const half *)in, sin, M, C, sums);
    return cudaGetLastError();
}

cudaError_t tchanstats(const void *qkv, int sq, int qo, int ko, int M, int heads, int hd, float *stats, cudaStream_t s)
{
    const int C = heads * hd, total = heads * hd * hd + 2 * C;
    if (hd <= 32 && hd % 2 == 0 && heads <= CS_MAXH && C <= 256 && sq % 2 == 0 && qo % 2 == 0 && ko % 2 == 0 && (uintptr_t)qkv % 4 == 0) {
        const int smem = 2 * 64 * (heads * 32 + 8) * 2;
        static bool set = false;
        if (!set) {
            cudaFuncSetAttribute(k_chanstats_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 64 * (CS_MAXH * 32 + 8) * 2);
            set = true;
        }
        const int tpc = std::max(64, ((M + 255) / 256 + 63) / 64 * 64);
        k_chanstats_mma<<<(M + tpc - 1) / tpc, 128, smem, s>>>((const half *)qkv, sq, qo, ko, M, heads, hd, tpc, stats);
        return cudaGetLastError();
    }
    if ((total + 255) / 256 > 24) return cudaErrorInvalidValue;
    const int smem = 2 * 64 * C * 2;
    if (smem > 48 * 1024) cudaFuncSetAttribute(k_chanstats, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    k_chanstats<<<(M + 255) / 256, 256, smem, s>>>((const half *)qkv, sq, qo, ko, M, heads, hd, stats);
    return cudaGetLastError();
}

}  // namespace ajinative
