// native/tfm.cu — see tfm.h.
#include "native/tfm.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

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
__device__ __forceinline__ void cp4(void *s, const void *g, int bytes)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(a), "l"(g), "r"(bytes));
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
    float slope;
    int act_last, out_f32, add1_f32;
    const float *add2scale;
    int k;  // A's real columns: those from k up to kpad load as zeros (A may be a slice whose row ends sooner)
    const float *lng, *lnb;  // k_gemm_rs only: A LayerNorm'd over its k channels on load, or null
    float lneps;
};

// RES: out is one of the adds, an fp32 stream (prefetched, the epilogue in two passes; see below).
template <bool RES>
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
    // the tile's residual rows into L2 now, so the epilogue's reads don't wait on DRAM
    if (RES) {
        const int cols = min(64, p.njt * 16 - j0 * 16);
        const int b1 = p.add1 ? cols * (p.add1_f32 ? 4 : 2) : 0, b2 = p.add2 ? cols * 2 : 0;
        const int l1 = (b1 + 127) / 128, l2 = (b2 + 127) / 128;
        for (int i = tid; i < BM * (l1 + l2); i += 128) {
            const int r = m0 + i / (l1 + l2), l = i % (l1 + l2);
            if (r >= p.M) break;
            const char *a = l < l1 ? reinterpret_cast<const char *>(p.add1) + ((size_t)r * p.s1 + j0 * 16) * (p.add1_f32 ? 4 : 2) + l * 128
                                   : reinterpret_cast<const char *>(p.add2 + (size_t)r * p.s2 + j0 * 16) + (l - l1) * 128;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
        }
    }
    for (int k0 = 0; k0 < p.kpad; k0 += KC) {
        const int kn = min(KC, p.kpad - k0);
        for (int i = tid; i < BM * (KC / 8); i += 128) {
            const int r = i / (KC / 8), v = i % (KC / 8);
            const int rem = p.k - (k0 + v * 8);  // real columns left from this chunk
            const bool ok = m0 + r < p.M && v * 8 < kn && rem > 0;
            cp16(sa + r * AP + v * 8, ok ? p.a + (size_t)(m0 + r) * p.sa + k0 + v * 8 : p.a, ok ? min(16, 2 * rem) : 0);
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
    // In place (out is one of the adds: a residual stream), the epilogue in two passes, every load then
    // every store: one pass would order each load after the previous store.
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
            if (!p.act_last) {
                if (p.act == 1) {
                    v0 = gelu(v0);
                    v1 = gelu(v1);
                } else if (p.act == 2) {
                    v0 = v0 < 0.f ? v0 * p.slope : v0;
                    v1 = v1 < 0.f ? v1 * p.slope : v1;
                }
            }
            if (p.colscale) {
                v0 *= p.colscale[c];
                v1 *= p.colscale[c + 1];
            }
            if (p.add1) {
                const float2 f = p.add1_f32 ? *reinterpret_cast<const float2 *>(reinterpret_cast<const float *>(p.add1) + (size_t)r * p.s1 + c)
                                            : __half22float2(*reinterpret_cast<const half2 *>(p.add1 + (size_t)r * p.s1 + c));
                v0 += f.x;
                v1 += f.y;
            }
            if (p.add2) {
                float2 f = __half22float2(*reinterpret_cast<const half2 *>(p.add2 + (size_t)r * p.s2 + c));
                if (p.add2scale) {
                    f.x *= p.add2scale[c];
                    f.y *= p.add2scale[c + 1];
                }
                v0 += f.x;
                v1 += f.y;
            }
            if (p.act_last) {
                if (p.act == 1) {
                    v0 = gelu(v0);
                    v1 = gelu(v1);
                } else if (p.act == 2) {
                    v0 = v0 < 0.f ? v0 * p.slope : v0;
                    v1 = v1 < 0.f ? v1 * p.slope : v1;
                }
            }
            if (!RES) {
                if (p.out_f32)
                    *reinterpret_cast<float2 *>(reinterpret_cast<float *>(p.out) + (size_t)r * p.so + c) = make_float2(v0, v1);
                else
                    *reinterpret_cast<half2 *>(p.out + (size_t)r * p.so + c) = __floats2half2_rn(v0, v1);
            }
            acc[n][2 * h] = v0;
            acc[n][2 * h + 1] = v1;
        }
    }
#pragma unroll
    for (int n = 0; n < 8; n++) {
        if (!RES) break;
        const int c = (j0 * 2 + n) * 8 + tq * 2;
        if (c >= p.njt * 16) break;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = m0 + warp * 16 + g + h * 8;
            if (r >= p.M) continue;
            if (p.out_f32)
                *reinterpret_cast<float2 *>(reinterpret_cast<float *>(p.out) + (size_t)r * p.so + c) = make_float2(acc[n][2 * h], acc[n][2 * h + 1]);
            else
                *reinterpret_cast<half2 *>(p.out + (size_t)r * p.so + c) = __floats2half2_rn(acc[n][2 * h], acc[n][2 * h + 1]);
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
    int sq, qo, ko, vo, H, W, heads, hd, wh, ww, sy, sx, nwx, Hp, Wp;
    float scale;
    const float *bias;
    half *out;
    int so;
};

__device__ __forceinline__ bool win_tok(const WinP &p, int win, int t, int *pix)
{
    const int wy = win / p.nwx, wx = win % p.nwx;
    int y = wy * p.wh + t / p.ww, x = wx * p.ww + t % p.ww;
    if (y >= p.Hp || x >= p.Wp) return false;  // beyond the (padded) image: zero q/k/v, no output
    if (p.sy || p.sx) {
        y = (y + p.sy) % p.Hp;
        x = (x + p.sx) % p.Wp;
    }
    if (y >= p.H || x >= p.W) return false;
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

// LayerNorm with fp32 and/or fp16 sides: L lanes per token, 8 channels each.
template <int L, typename TI, typename TO>
__global__ void __launch_bounds__(256) k_ln8x(const TI *in, int sin, TO *out, int sout, int M, int C, int cpad, const float *gm,
                                              const float *bt, float eps)
{
    const int sub = threadIdx.x % L, t = (blockIdx.x * blockDim.x + threadIdx.x) / L, c = sub * 8;
    const bool live = t < M && c < cpad;
    float v[8] = {};
    if (live) {
        if constexpr (sizeof(TI) == 4) {
            const float4 a = *reinterpret_cast<const float4 *>(in + (size_t)t * sin + c), b = *reinterpret_cast<const float4 *>(in + (size_t)t * sin + c + 4);
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
            unpack8(*reinterpret_cast<const uint4 *>(in + (size_t)t * sin + c), v);
        }
    }
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
    if constexpr (sizeof(TO) == 4) {
        *reinterpret_cast<float4 *>(out + (size_t)t * sout + c) = make_float4(y[0], y[1], y[2], y[3]);
        *reinterpret_cast<float4 *>(out + (size_t)t * sout + c + 4) = make_float4(y[4], y[5], y[6], y[7]);
    } else {
        *reinterpret_cast<uint4 *>(out + (size_t)t * sout + c) = pack8(y);
    }
}

// Copies between fp32 and fp16 token maps, 8 channels per thread, times a scale.
template <typename TI, typename TO>
__global__ void k_convert(const TI *in, int sin, TO *out, int sout, int M, int n, float scale)
{
    const int G = n / 8;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * G) return;
    const int t = (int)(i / G), c = (int)(i % G) * 8;
    float v[8];
    if constexpr (sizeof(TI) == 4) {
        const float4 a = *reinterpret_cast<const float4 *>(in + (size_t)t * sin + c), b = *reinterpret_cast<const float4 *>(in + (size_t)t * sin + c + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
        unpack8(*reinterpret_cast<const uint4 *>(in + (size_t)t * sin + c), v);
    }
    if (scale != 1.f)
        for (float &x : v) x *= scale;
    if constexpr (sizeof(TO) == 4) {
        *reinterpret_cast<float4 *>(out + (size_t)t * sout + c) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4 *>(out + (size_t)t * sout + c + 4) = make_float4(v[4], v[5], v[6], v[7]);
    } else {
        *reinterpret_cast<uint4 *>(out + (size_t)t * sout + c) = pack8(v);
    }
}

// ---- window attention, general form: overlapping key windows (HAT's OCAB), Swin's shifted-window
// mask, and the bias as a relative-position table in shared memory. One CTA per (window, head); k and v
// of the key window in shared memory, q fragments straight from global memory.
struct WinG {
    WinP b;
    int kwh, kww, kpad;  // key window: kwh x kww tokens at (wy*wh - kpad, wx*ww - kpad); kwh = 0: the query window
    int mask;            // Swin mask (-100 between the regions the shift joins)
    const float *rtab;   // [heads][th*tw]: bias(q, k) = rtab[(ky - kpad - qy + offy) * tw + (kx - kpad - qx + offx)]
    int tw, offy, offx, tn;  // tn: table entries per head
};

// The Swin region (0..2 per axis) of rolled coordinate v on an axis of n with window w and shift s.
__device__ __forceinline__ int swin_rid(int v, int n, int w, int s) { return v < n - w ? 0 : (v < n - s ? 1 : 2); }

__global__ void __launch_bounds__(512) k_winattn_g(WinG P)
{
#if TFM_OK
    const WinP &p = P.b;
    extern __shared__ __align__(16) unsigned char smraw[];
    const int Tq = p.wh * p.ww, kw = P.kwh ? P.kww : p.ww, kh = P.kwh ? P.kwh : p.wh, Tk = kh * kw;
    half *sk = reinterpret_cast<half *>(smraw), *sv = sk + Tk * HP;
    float *stab = reinterpret_cast<float *>(sv + Tk * HP);
    // per key token: its pixel (the query window's own keys only: -1 outside the image), its offset
    // into the bias table, its Swin region; computed once here instead of per element / score
    int *spix = reinterpret_cast<int *>(stab + (P.rtab ? P.tn : 0));
    short *skid = reinterpret_cast<short *>(spix + (P.kwh ? 0 : Tk));
    signed char *sreg = reinterpret_cast<signed char *>(skid + (P.rtab ? Tk : 0));
    const int win = (int)(blockIdx.x / p.heads), h = (int)(blockIdx.x % p.heads), tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wy = win / p.nwx, wx = win % p.nwx;
    // keys and values of this head (dims padded to 32 with zeros; tokens outside the image zero), and
    // the bias table: copies straight to shared memory, all in flight together (cp.async, 4 bytes:
    // a head's dims start 4-byte aligned)
    const int nt = (int)blockDim.x;  // 32 per warp, a warp per 16-query block in turn
    for (int t = tid; t < Tk; t += nt) {
        if (!P.kwh) {
            int px;
            spix[t] = win_tok(p, win, t, &px) ? px : -1;
        }
        if (P.rtab) skid[t] = (short)((t / kw - P.kpad) * P.tw + (t % kw - P.kpad));
        if (P.mask)
            sreg[t] = (signed char)(swin_rid(wy * p.wh + t / p.ww, p.Hp, p.wh, p.sy) * 3 + swin_rid(wx * p.ww + t % p.ww, p.Wp, p.ww, p.sx));
    }
    __syncthreads();
    for (int i = tid; i < Tk * 16; i += nt) {
        const int t = i / 16, d2 = (i % 16) * 2;
        int pix = -1;
        if (P.kwh) {
            const int y = wy * p.wh - P.kpad + t / kw, x = wx * p.ww - P.kpad + t % kw;
            if (y >= 0 && y < p.H && x >= 0 && x < p.W) pix = y * p.W + x;
        } else {
            pix = spix[t];
        }
        const bool ok = d2 < p.hd && pix >= 0;
        const half *src = ok ? p.qkv + (size_t)pix * p.sq + h * p.hd + d2 : p.qkv;
        cp4(sk + t * HP + d2, ok ? src + p.ko : src, ok ? 4 : 0);
        cp4(sv + t * HP + d2, ok ? src + p.vo : src, ok ? 4 : 0);
    }
    if (P.rtab)
        for (int i = tid; i < P.tn; i += nt) cp4(stab + i, P.rtab + (size_t)h * P.tn + i, 4);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    const int g = lane >> 2, tq = lane & 3;
    const float *bias = p.bias ? p.bias + (size_t)h * Tq * Tk : nullptr;
    for (int qb = warp; qb * 16 < Tq; qb += nt / 32) {
        const int r0 = qb * 16 + g, r1 = r0 + 8;
        // q fragments: a0 (r0, 2tq), a1 (r1, 2tq), a2 (r0, 2tq + 8), a3 (r1, 2tq + 8) per 16-dim chunk
        uint32_t qa[2][4];
        int p0 = -1, p1 = -1;
        if (P.kwh) {
            if (!win_tok(p, win, r0, &p0)) p0 = -1;
            if (!win_tok(p, win, r1, &p1)) p1 = -1;
        } else {
            p0 = r0 < Tq ? spix[r0] : -1;
            p1 = r1 < Tq ? spix[r1] : -1;
        }
        const bool v0 = p0 >= 0, v1 = p1 >= 0;
#pragma unroll
        for (int kc = 0; kc < 2; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int d = kc * 16 + 2 * tq + (j >> 1) * 8;
                const bool ok = d < p.hd && ((j & 1) ? v1 : v0);
                half2 q = __float2half2_rn(0.f);
                if (ok) q = *reinterpret_cast<const half2 *>(p.qkv + (size_t)((j & 1) ? p1 : p0) * p.sq + p.qo + h * p.hd + d);
                qa[kc][j] = h2u(q);
            }
        const int qy[2] = {r0 / p.ww, r1 / p.ww}, qx[2] = {r0 % p.ww, r1 % p.ww};
        int qreg[2] = {0, 0};
        const int qoff[2] = {(P.offy - qy[0]) * P.tw + (P.offx - qx[0]), (P.offy - qy[1]) * P.tw + (P.offx - qx[1])};
        if (P.mask)
#pragma unroll
            for (int i = 0; i < 2; i++)
                qreg[i] = swin_rid(wy * p.wh + qy[i], p.Hp, p.wh, p.sy) * 3 + swin_rid(wx * p.ww + qx[i], p.Wp, p.ww, p.sx);
        float o[4][4], mrow[2] = {-1e30f, -1e30f}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int k = 0; k < 4; k++) o[n][k] = 0.f;
        for (int kb = 0; kb < Tk; kb += 64) {
            const int nk = min(64, Tk - kb);  // a multiple of 16
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
            float mx[2] = {mrow[0], mrow[1]};
#pragma unroll
            for (int n = 0; n < 8; n++) {
                if (n * 8 >= nk) break;
                const int c = kb + n * 8 + tq * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int ri = e >> 1, r = ri ? r1 : r0, cc = c + (e & 1);
                    float v = s[n][e] * p.scale;
                    if (P.rtab) {
                        v += stab[skid[cc] + qoff[ri]];
                    } else if (bias) {
                        v += bias[(size_t)r * Tk + cc];
                    }
                    if (P.mask && sreg[cc] != qreg[ri]) v -= 100.f;
                    s[n][e] = v;
                    mx[ri] = fmaxf(mx[ri], v);
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
            const int r = hh ? r1 : r0, pix = hh ? p1 : p0;
            if (r >= Tq || !(hh ? v1 : v0)) continue;
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

// Depthwise 3x3, one thread per (8 channels, column, strip of DW_R rows): the 3x3 window slides down
// the strip so each output row loads one new input row (3 pixels), weights held in registers.
constexpr int DW_R = 8;
// C may stop inside the last group of 8 (C even; the rows are read 8 channels wide, so the strides
// cover the rounded-up width): the channels past C get zero weights and are neither stored nor summed
// (depthwise, so nothing of them reaches a real channel).
__global__ void __launch_bounds__(128) k_dw8s(const half *in, int sin, half *out, int sout, int H, int W, int C, const float *w, int gelu_,
                                              float *sums, const float *bias = nullptr, const half *mul = nullptr, int smul = 0)
{
    __shared__ float ss[1024];
    if (sums) {
        for (int i = threadIdx.x; i < C; i += blockDim.x) ss[i] = 0.f;
        __syncthreads();
    }
    const int G = (C + 7) / 8, strips = (H + DW_R - 1) / DW_R;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = i < G * W * strips;
    const int g = i % G, rest = i / G, x = rest % W, y0 = (rest / W) * DW_R, c = g * 8;
    float part[8] = {};
    const bool full = c + 8 <= C;
    if (live) {
        float wr[9][8];
#pragma unroll
        for (int k = 0; k < 9; k++)
#pragma unroll
            for (int j = 0; j < 8; j++) wr[k][j] = c + j < C ? __ldg(w + (c + j) * 9 + k) : 0.f;
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
            if (bias)
#pragma unroll
                for (int j = 0; j < 8; j++) a[j] += c + j < C ? __ldg(bias + c + j) : 0.f;
            if (gelu_)
#pragma unroll
                for (int j = 0; j < 8; j++) a[j] = gelu(a[j]);
            if (mul) {
                float m[8];
                unpack8(*reinterpret_cast<const uint4 *>(mul + ((size_t)y * W + x) * smul + c), m);
#pragma unroll
                for (int j = 0; j < 8; j++) a[j] *= m[j];
            }
            const uint4 pk = pack8(a);
            if (full) {
                *reinterpret_cast<uint4 *>(out + ((size_t)y * W + x) * sout + c) = pk;
            } else {
                const uint32_t *pw = reinterpret_cast<const uint32_t *>(&pk);
#pragma unroll
                for (int j = 0; j < 4; j++)
                    if (c + 2 * j < C) *reinterpret_cast<uint32_t *>(out + ((size_t)y * W + x) * sout + c + 2 * j) = pw[j];
            }
#pragma unroll
            for (int j = 0; j < 8; j++) part[j] += c + j < C ? a[j] : 0.f;
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
            for (int j = 0; j < 8; j++)
                if (c + j < C) atomicAdd(ss + c + j, part[j]);
        __syncthreads();
        for (int k = threadIdx.x; k < C; k += blockDim.x) atomicAdd(sums + k, ss[k]);
    }
}

// Channel attention statistics on tensor cores: per head the Gram matrix q^T k over the CTA's tokens
// (64-token chunks in shared memory, each head's dims padded to 32), warp w doing heads w and w + 4;
// the squared norms on the CUDA cores while copying.
constexpr int CS_MAXH = 8;
constexpr int CS_ROWS = 32;  // k_chanstats_mma's tokens per step: 2 x 32 x (heads x 32 + 8) halves of shared memory, 3 CTAs per SM
__global__ void __launch_bounds__(128) k_chanstats_mma(const half *qkv, int sq, int qo, int ko, int M, int heads, int hd, int tpc, float *stats)
{
#if TFM_OK
    extern __shared__ __align__(16) unsigned char smraw[];
    const int P = heads * 32 + 8, C = heads * hd, pairs = C / 2;
    constexpr int R = CS_ROWS;  // tokens per step
    half *s_q = reinterpret_cast<half *>(smraw), *s_k = s_q + R * P;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < R * P; i += 128) s_q[i] = s_k[i] = __float2half(0.f);
    float acc[2][2][4][4] = {};
    float nrm[2][2] = {};
    const int t0 = blockIdx.x * tpc, t1 = min(M, t0 + tpc);
    for (int tb = t0; tb < t1; tb += R) {
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
            for (int r = 0; r < R; r++) {
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
            for (int ks = 0; ks < R / 16; ks++) {
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

// ---- GEMM without adds, row-stationary (the fused MLP's fc1 half): a warp owns 16 rows and keeps
// their A fragments in registers for the whole K; the CTA walks every 16-column chunk of W, two
// chunks per cp.async stage through shared memory (two stages), and writes each 16x16 result as it
// is done. No accumulator lives across chunks, so it runs at ~100 registers and 4 CTAs per SM, and
// W is read once per 64 rows instead of once per 64x64 tile. Same sums as k_gemm.
template <int KC1>
__global__ void __launch_bounds__(128) k_gemm_rs(GemmP p)  // with p.lng: LayerNorm of A's rows on load (see below)
{
#if TFM_OK
    constexpr int CH = KC1 * 32, NT = 128;  // uint4 per chunk slice
    extern __shared__ __align__(16) uint4 smw[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int nch = p.njt, npair = (nch + 1) / 2, r0 = (int)blockIdx.x * 64 + warp * 16 + g, r1 = r0 + 8;
    auto load_pair = [&](int c0, int st) {  // slice row q = idx / 32 of the pair: chunk q / KC1, k16 row q % KC1
#pragma unroll
        for (int k = 0; k < (2 * CH + NT - 1) / NT; k++) {
            const int idx = k * NT + tid, q = idx >> 5, c = c0 + q / KC1, sl = q % KC1;
            if (idx < 2 * CH && c < nch) cp16(smw + (size_t)st * 2 * CH + idx, p.w + ((size_t)sl * nch + c) * 32 + lane, 16);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    load_pair(0, 0);
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const int r = hh ? r1 : r0;
        if (r >= p.M) continue;
        const char *ra = reinterpret_cast<const char *>(p.a + (size_t)r * p.sa);
        for (int l = tq; l * 128 < KC1 * 32; l += 4) asm volatile("prefetch.global.L2 [%0];" ::"l"(ra + l * 128));
    }
    uint32_t a[KC1][4];
#pragma unroll
    for (int kc = 0; kc < KC1; kc++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int r = (j & 1) ? r1 : r0, c = kc * 16 + 2 * tq + (j >> 1) * 8;
            half2 v = __float2half2_rn(0.f);
            if (r < p.M && c < p.k) v = *reinterpret_cast<const half2 *>(p.a + (size_t)r * p.sa + c);
            a[kc][j] = h2u(v);
        }
    if (p.lng) {
        // the rows' LayerNorm over their k channels, in place on the raw fp16 fragments (as tlayernorm):
        // a row's channels sit in its lane quad, so the sums take two shuffles
        float s[2] = {0.f, 0.f}, q[2] = {0.f, 0.f}, mean[2], rs[2];
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 v = __half22float2(*reinterpret_cast<const half2 *>(&a[kc][j]));
                s[j & 1] += v.x + v.y;
            }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            s[i] += __shfl_xor_sync(0xffffffffu, s[i], 1);
            s[i] += __shfl_xor_sync(0xffffffffu, s[i], 2);
            mean[i] = s[i] / p.k;
        }
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (kc * 16 + 2 * tq + (j >> 1) * 8 < p.k) {
                    const float2 v = __half22float2(*reinterpret_cast<const half2 *>(&a[kc][j]));
                    const float d0 = v.x - mean[j & 1], d1 = v.y - mean[j & 1];
                    q[j & 1] += d0 * d0 + d1 * d1;
                }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            q[i] += __shfl_xor_sync(0xffffffffu, q[i], 1);
            q[i] += __shfl_xor_sync(0xffffffffu, q[i], 2);
            rs[i] = rsqrtf(q[i] / p.k + p.lneps);
        }
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = kc * 16 + 2 * tq + (j >> 1) * 8, i = j & 1;
                half2 y = __float2half2_rn(0.f);
                if (c < p.k) {
                    const float2 v = __half22float2(*reinterpret_cast<const half2 *>(&a[kc][j]));
                    const float2 gm = __ldg(reinterpret_cast<const float2 *>(p.lng + c)), bt = __ldg(reinterpret_cast<const float2 *>(p.lnb + c));
                    y = __floats2half2_rn((v.x - mean[i]) * rs[i] * gm.x + bt.x, (v.y - mean[i]) * rs[i] * gm.y + bt.y);
                }
                a[kc][j] = h2u(y);
            }
    }
    for (int pp = 0; pp < npair; pp++) {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        if (pp + 1 < npair) load_pair(2 * pp + 2, (pp + 1) & 1);
        const uint4 *sw = smw + (size_t)(pp & 1) * 2 * CH;
        const bool two = 2 * pp + 1 < nch;
        float h[2][2][4];
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) h[c][n][e] = 0.f;
#pragma unroll
        for (int kc = 0; kc < KC1; kc++) {
            const uint4 b0 = sw[kc * 32 + lane];
            mma32(h[0][0], a[kc], b0.x, b0.y);
            mma32(h[0][1], a[kc], b0.z, b0.w);
            if (two) {
                const uint4 b1 = sw[CH + kc * 32 + lane];
                mma32(h[1][0], a[kc], b1.x, b1.y);
                mma32(h[1][1], a[kc], b1.z, b1.w);
            }
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
            if (c && !two) break;
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const int col = (2 * pp + c) * 16 + n * 8 + 2 * tq;
                float2 bb = make_float2(0.f, 0.f), cs = make_float2(1.f, 1.f);
                if (p.bias) bb = __ldg(reinterpret_cast<const float2 *>(p.bias + col));
                if (p.colscale) cs = __ldg(reinterpret_cast<const float2 *>(p.colscale + col));
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const int r = hh ? r1 : r0;
                    float v0 = h[c][n][2 * hh] + bb.x, v1 = h[c][n][2 * hh + 1] + bb.y;
                    if (p.act == 1) {
                        v0 = gelu(v0);
                        v1 = gelu(v1);
                    } else if (p.act == 2) {
                        v0 = v0 < 0.f ? v0 * p.slope : v0;
                        v1 = v1 < 0.f ? v1 * p.slope : v1;
                    }
                    if (p.colscale) {
                        v0 *= cs.x;
                        v1 *= cs.y;
                    }
                    if (r >= p.M) continue;
                    if (p.out_f32)
                        *reinterpret_cast<float2 *>(reinterpret_cast<float *>(p.out) + (size_t)r * p.so + col) = make_float2(v0, v1);
                    else
                        *reinterpret_cast<half2 *>(p.out + (size_t)r * p.so + col) = __floats2half2_rn(v0, v1);
                }
            }
        }
    }
#else
    __trap();
#endif
}

// ---- depthwise 3x3 then GEMM (FDAT's smix then fc2): out = dw(in) W + bias + add1, fp16, for maps whose
// width is a multiple of 64 (a CTA's 64 tokens then sit in one image row). A warp owns 16 tokens and keeps
// their fc2 accumulators (16 x N) in registers. Per 16-channel K chunk, a cp.async stage brings W's chunk
// slice, the chunk's 16 x 9 tap weights and the 3 x 66-token halo of the CTA's tokens (zeros outside the
// image) into shared memory; each thread forms its A fragment's depthwise values from the halo (the order
// and fp16 rounding of k_dw8s, as if the map were stored), then the chunk's MMAs. Three stages, one chunk
// each. The residual tile goes through shared memory at the end, as k_mlp. Same values as tdwconv then
// tgemm.
struct DwmmP {
    const half *in;
    int sin, H, W, M, kc;  // kc: real input channels (dw has kc x 9 floats)
    const float *dw;
    const uint4 *w;
    const float *bias;
    int nh;  // K chunks (W's Kpad / 16)
    half *out;
    int so;
    const half *add1;
    int s1;
};

template <int NJ2>
struct DwmmSmem {
    static constexpr int cw = NJ2 * 32, ct = 36, chalo = 3 * 66 * 2;  // uint4: W slice, taps, halo (3 x 66 tokens x 32 bytes)
    static constexpr int chunk = cw + ct + chalo;
    static constexpr int bytes = 3 * chunk * 16;
};

template <int NJ2>
__global__ void __launch_bounds__(128) k_dwmm(DwmmP p)
{
#if TFM_OK
    using S = DwmmSmem<NJ2>;
    constexpr int CW = S::cw, CT = S::ct, CH = S::chunk, NT = 128, BM = 64;
    extern __shared__ __align__(16) uint4 smw[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int m0 = (int)blockIdx.x * BM, y0 = m0 / p.W, x0 = m0 % p.W;
    const int r0 = m0 + warp * 16 + g, r1 = r0 + 8;
    const int tbytes = p.kc * 36;  // the tap weights' bytes
    auto load = [&](int c, int st) {
        if (c < p.nh) {
            uint4 *base = smw + (size_t)st * CH;
            for (int i = tid; i < CH; i += NT) {
                if (i < CW) {
                    cp16(base + i, p.w + (size_t)c * NJ2 * 32 + i, 16);
                } else if (i < CW + CT) {
                    const int off = c * 576 + (i - CW) * 16, n = min(16, max(0, tbytes - off));
                    cp16(base + i, reinterpret_cast<const char *>(p.dw) + (n ? off : 0), n);
                } else {
                    const int e = i - CW - CT, t = e >> 1, half_ = e & 1, dy = t / 66, xx = x0 - 1 + t % 66, yy = y0 - 1 + dy;
                    const bool ok = yy >= 0 && yy < p.H && xx >= 0 && xx < p.W;
                    cp16(base + i, ok ? p.in + ((size_t)yy * p.W + xx) * p.sin + c * 16 + half_ * 8 : p.in, ok ? 16 : 0);
                }
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    load(0, 0);
    load(1, 1);
#pragma unroll
    for (int hh = 0; hh < 2; hh++)
        if (p.add1)
            for (int l = tq; l * 128 < NJ2 * 32; l += 4)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char *>(p.add1 + (size_t)(hh ? r1 : r0) * p.s1) + l * 128));
    float acc[NJ2 * 2][4];
#pragma unroll
    for (int n = 0; n < NJ2 * 2; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = 0.f;
    for (int c = 0; c < p.nh; c++) {
        asm volatile("cp.async.wait_group 1;" ::: "memory");  // chunk c is in (c + 1 may still be landing)
        __syncthreads();  // ... for every thread; and every warp is done with chunk c - 1's stage
        load(c + 2, (c + 2) % 3);
        const uint4 *sw = smw + (size_t)(c % 3) * CH;
        const float *tw = reinterpret_cast<const float *>(sw + CW);  // [16][9]
        const half *halo = reinterpret_cast<const half *>(sw + CW + CT);  // [3][66][16]
        uint32_t pa[4];  // pa[j]: token (j & 1 ? r1 : r0), channels 2tq + (j >> 1) * 8, +1 of the chunk
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int lt = warp * 16 + g + (j & 1) * 8, cl = 2 * tq + (j >> 1) * 8;
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int dx = 0; dx < 3; dx++)
#pragma unroll
                for (int dy = 0; dy < 3; dy++) {
                    const float2 v = __half22float2(*reinterpret_cast<const half2 *>(halo + (dy * 66 + lt + dx) * 16 + cl));
                    a0 += v.x * tw[cl * 9 + dy * 3 + dx];
                    a1 += v.y * tw[(cl + 1) * 9 + dy * 3 + dx];
                }
            pa[j] = h2u(__floats2half2_rn(a0, a1));
        }
#pragma unroll
        for (int n = 0; n < NJ2; n++) {
            const uint4 b = sw[n * 32 + lane];
            mma32(acc[2 * n], pa, b.x, b.y);
            mma32(acc[2 * n + 1], pa, b.z, b.w);
        }
    }
    // the residual tile (fp16) through shared memory, swizzled by row, then the epilogue
    constexpr int cpr = NJ2 * 2;  // 16-byte chunks per row
    const unsigned char *sres = reinterpret_cast<const unsigned char *>(smw);
    if (p.add1) {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        for (int i = tid; i < BM * cpr; i += NT) {
            const int lr = i / cpr, chn = i % cpr;
            cp16(reinterpret_cast<unsigned char *>(smw) + lr * cpr * 16 + ((chn ^ (lr & 7)) * 16),
                 reinterpret_cast<const char *>(p.add1 + (size_t)(m0 + lr) * p.s1) + chn * 16, 16);
        }
        cp_wait();
        __syncthreads();
    }
#pragma unroll
    for (int n = 0; n < NJ2 * 2; n++) {
        const int c = n * 8 + tq * 2;
        const float2 bb = p.bias ? __ldg(reinterpret_cast<const float2 *>(p.bias + c)) : make_float2(0.f, 0.f);
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int lr = warp * 16 + g + hh * 8;
            float v0 = acc[n][2 * hh], v1 = acc[n][2 * hh + 1];
            if (p.bias) {
                v0 += bb.x;
                v1 += bb.y;
            }
            if (p.add1) {
                const float2 f = __half22float2(*reinterpret_cast<const half2 *>(sres + lr * cpr * 16 + ((n ^ g) * 16) + tq * 4));
                v0 += f.x;
                v1 += f.y;
            }
            acc[n][2 * hh] = v0;
            acc[n][2 * hh + 1] = v1;
        }
    }
#pragma unroll
    for (int n = 0; n < NJ2 * 2; n++) {
        const int c = n * 8 + tq * 2;
#pragma unroll
        for (int hh = 0; hh < 2; hh++)
            *reinterpret_cast<half2 *>(p.out + (size_t)(hh ? r1 : r0) * p.so + c) = __floats2half2_rn(acc[n][2 * hh], acc[n][2 * hh + 1]);
    }
#else
    __trap();
#endif
}

template <int NJ2>
cudaError_t launch_dwmm(const DwmmP &p, cudaStream_t s)
{
    constexpr int sm = DwmmSmem<NJ2>::bytes;
    static_assert(64 * NJ2 * 2 * 16 <= sm, "the residual tile fits in the stages");
    static const cudaError_t attr = cudaFuncSetAttribute(k_dwmm<NJ2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (attr != cudaSuccess) return attr;
    k_dwmm<NJ2><<<(unsigned)(p.M / 64), 128, sm, s>>>(p);
    return cudaGetLastError();
}

// ---- fused MLP: out = epi(act(A W1 + b1) W2 + b2), the hidden kept on chip (flash-attention style).
// A warp owns 16 rows: their A fragments stay in registers for the whole K, and the fc2 accumulators
// (16 x N2) too. Per 16-column hidden chunk: fc1 into a 16x16 fragment, bias and GELU, rounded to fp16
// and reused in registers as fc2's A fragment (as P in attention). The weight slices of each chunk (W1:
// its 16 columns over K, W2: its 16 rows over N2) go through shared memory, two chunks per stage, two
// stages, shared by the CTA's warps. Same accumulation order and roundings as k_gemm fc1 then fc2.
struct MlpP {
    const void *a;  // fp16, or with LN the fp32 stream (normalised over its first k channels on load)
    int sa, M, k;   // k: A's real columns (zeros from k on)
    const uint4 *w1, *w2;
    const float *b1, *b2;
    int nh;  // hidden 16-column chunks (W1's Npad / 16 = W2's Kpad / 16)
    half *out;
    int so, out_f32;
    const half *add1;
    int s1, add1_f32;
    const float *lg, *lb;  // LN: gamma, beta [k]
    float eps;
    half *lo;              // or null: the LayerNorm (over the first n2 channels) of the output rows, fp16
    int slo, n2;
    const float *log_, *lob;
    float loeps;
};

template <int KC1, int NJ2>
struct MlpSmem {
    static constexpr int chunk = (KC1 + NJ2) * 32;  // uint4 per hidden chunk: W1 slice, then W2 slice
    static constexpr int bytes = 4 * chunk * 16;    // two stages of two chunks
};

template <int KC1, int NJ2, int NW, bool LN>
__global__ void __launch_bounds__(NW * 32, 8 / NW) k_mlp(MlpP p)
{
#if TFM_OK
    constexpr int CH = MlpSmem<KC1, NJ2>::chunk, NT = NW * 32, BM = 16 * NW;
    static_assert(CH % NT == 0, "a chunk's weights split evenly over the threads");
    extern __shared__ __align__(16) uint4 smw[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int npair = (p.nh + 1) / 2, r0 = (int)blockIdx.x * BM + warp * 16 + g, r1 = r0 + 8;
    // the weights of hidden chunks c0, c0 + 1 into stage st (slice row sl = 32 uint4: W1 rows, then W2's)
    auto load_pair = [&](int c0, int st) {
#pragma unroll
        for (int k = 0; k < 2 * CH / NT; k++) {
            const int c = c0 + k / (CH / NT), sl = warp + (k % (CH / NT)) * NW;
            if (c < p.nh) {
                const uint4 *src = sl < KC1 ? p.w1 + ((size_t)sl * p.nh + c) * 32 + lane : p.w2 + ((size_t)c * NJ2 + (sl - KC1)) * 32 + lane;
                cp16(smw + (size_t)st * 2 * CH + k * NT + tid, src, 16);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    load_pair(0, 0);
    // this thread's two rows of A and of the residual into L2 first: the A fragment loads below then
    // wait less, and the epilogue's residual reads (after the whole K) don't wait on DRAM
    {
        const int res = p.add1_f32 ? 4 : 2;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = hh ? r1 : r0;
            if (r >= p.M) continue;
            const char *ra = reinterpret_cast<const char *>(p.a) + (size_t)r * p.sa * (LN ? 4 : 2);
            for (int l = tq; l * 128 < KC1 * 16 * (LN ? 4 : 2); l += 4) asm volatile("prefetch.global.L2 [%0];" ::"l"(ra + l * 128));
            if (!p.add1) continue;
            const char *rr = reinterpret_cast<const char *>(p.add1) + (size_t)r * p.s1 * res;
            for (int l = tq; l * 128 < NJ2 * 16 * res; l += 4) asm volatile("prefetch.global.L2 [%0];" ::"l"(rr + l * 128));
        }
    }
    // A fragments: a0 (r0, 2tq), a1 (r1, 2tq), a2 (r0, 2tq + 8), a3 (r1, 2tq + 8) per k16 chunk
    uint32_t a[KC1][4];
    if constexpr (LN) {
        // LayerNorm of the fp32 rows straight into the fragments: a row's channels are spread over
        // the 4 lanes of its quad (lanes 4g..4g+3), so the sums take two shuffles (as tlayernorm2's maths)
        float2 x[KC1][4];
        float s[2] = {0.f, 0.f}, q[2] = {0.f, 0.f};
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r = (j & 1) ? r1 : r0, c = kc * 16 + 2 * tq + (j >> 1) * 8;
                x[kc][j] = r < p.M && c < p.k ? *reinterpret_cast<const float2 *>(reinterpret_cast<const float *>(p.a) + (size_t)r * p.sa + c)
                                              : make_float2(0.f, 0.f);
                s[j & 1] += x[kc][j].x + x[kc][j].y;
            }
        float mean[2], rs[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            s[i] += __shfl_xor_sync(0xffffffffu, s[i], 1);
            s[i] += __shfl_xor_sync(0xffffffffu, s[i], 2);
            mean[i] = s[i] / p.k;
        }
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (kc * 16 + 2 * tq + (j >> 1) * 8 < p.k) {
                    const float dx = x[kc][j].x - mean[j & 1], dy = x[kc][j].y - mean[j & 1];
                    q[j & 1] += dx * dx + dy * dy;
                }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            q[i] += __shfl_xor_sync(0xffffffffu, q[i], 1);
            q[i] += __shfl_xor_sync(0xffffffffu, q[i], 2);
            rs[i] = rsqrtf(q[i] / p.k + p.eps);
        }
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = kc * 16 + 2 * tq + (j >> 1) * 8, i = j & 1;
                half2 v = __float2half2_rn(0.f);
                if (c < p.k) {
                    const float2 gm = __ldg(reinterpret_cast<const float2 *>(p.lg + c)), bt = __ldg(reinterpret_cast<const float2 *>(p.lb + c));
                    v = __floats2half2_rn((x[kc][j].x - mean[i]) * rs[i] * gm.x + bt.x, (x[kc][j].y - mean[i]) * rs[i] * gm.y + bt.y);
                }
                a[kc][j] = h2u(v);
            }
    } else {
#pragma unroll
        for (int kc = 0; kc < KC1; kc++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r = (j & 1) ? r1 : r0, c = kc * 16 + 2 * tq + (j >> 1) * 8;
                half2 v = __float2half2_rn(0.f);
                if (r < p.M && c < p.k) v = *reinterpret_cast<const half2 *>(reinterpret_cast<const half *>(p.a) + (size_t)r * p.sa + c);
                a[kc][j] = h2u(v);
            }
    }
    float acc[NJ2 * 2][4];
#pragma unroll
    for (int n = 0; n < NJ2 * 2; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = 0.f;
    for (int pp = 0; pp < npair; pp++) {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();  // this pair's weights are in; every warp is done with the other stage
        if (pp + 1 < npair) load_pair(2 * pp + 2, (pp + 1) & 1);
        const uint4 *sw = smw + (size_t)(pp & 1) * 2 * CH;
        const bool two = 2 * pp + 1 < p.nh;
        float h[2][2][4];
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) h[c][n][e] = 0.f;
#pragma unroll
        for (int kc = 0; kc < KC1; kc++) {
            const uint4 b0 = sw[kc * 32 + lane];
            mma32(h[0][0], a[kc], b0.x, b0.y);
            mma32(h[0][1], a[kc], b0.z, b0.w);
            if (two) {
                const uint4 b1 = sw[CH + kc * 32 + lane];
                mma32(h[1][0], a[kc], b1.x, b1.y);
                mma32(h[1][1], a[kc], b1.z, b1.w);
            }
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
            if (c && !two) break;
            const int col = (2 * pp + c) * 16 + 2 * tq;
            const float2 b01 = __ldg(reinterpret_cast<const float2 *>(p.b1 + col)), b89 = __ldg(reinterpret_cast<const float2 *>(p.b1 + col + 8));
            const float bb[4] = {b01.x, b01.y, b89.x, b89.y};
            uint32_t pa[4];  // C fragment -> A fragment: pa[2n + hh] = (row g + 8hh, columns 8n + 2tq, +1)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int hh = 0; hh < 2; hh++)
                    pa[n * 2 + hh] = h2u(__floats2half2_rn(gelu(h[c][n][2 * hh] + bb[2 * n]), gelu(h[c][n][2 * hh + 1] + bb[2 * n + 1])));
            const uint4 *s2 = sw + c * CH + KC1 * 32;
#pragma unroll
            for (int n = 0; n < NJ2; n++) {
                const uint4 b = s2[n * 32 + lane];
                mma32(acc[2 * n], pa, b.x, b.y);
                mma32(acc[2 * n + 1], pa, b.z, b.w);
            }
        }
    }
    // The residual tile goes through shared memory (the weight stages are free now): one batch of
    // copies, all in flight at once, where register loads would wait one by one (the registers are
    // all taken, and out is usually add1 itself, so they couldn't pass the stores either). 16-byte
    // chunks XOR-swizzled by row so a warp's reads hit every bank once per wavefront.
    const int es = p.add1_f32 ? 4 : 2, cpr = NJ2 * es;  // chunks per row: 48 (fp32) or 24 (fp16)
    const unsigned char *sres = reinterpret_cast<const unsigned char *>(smw);
    if (p.add1) {
        __syncthreads();  // every warp is done with the weights
        for (int i = tid; i < BM * cpr; i += NT) {
            const int lr = i / cpr, ch = i % cpr, r = (int)blockIdx.x * BM + lr;
            const char *src = reinterpret_cast<const char *>(p.add1) + ((size_t)r * p.s1 * es + ch * 16);
            cp16(reinterpret_cast<unsigned char *>(smw) + lr * cpr * 16 + ((ch ^ (lr & 7)) * 16), r < p.M ? src : reinterpret_cast<const char *>(p.add1),
                 r < p.M ? 16 : 0);
        }
        cp_wait();
        __syncthreads();
    }
#pragma unroll
    for (int n = 0; n < NJ2 * 2; n++) {
        const int c = n * 8 + tq * 2;
        const float2 bb = __ldg(reinterpret_cast<const float2 *>(p.b2 + c));
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int lr = warp * 16 + g + hh * 8;  // lr & 7 = g
            float v0 = acc[n][2 * hh] + bb.x, v1 = acc[n][2 * hh + 1] + bb.y;
            if (p.add1) {
                const unsigned char *row = sres + lr * cpr * 16;
                const float2 f = p.add1_f32 ? *reinterpret_cast<const float2 *>(row + (((2 * n + (tq >> 1)) ^ g) * 16) + (tq & 1) * 8)
                                            : __half22float2(*reinterpret_cast<const half2 *>(row + ((n ^ g) * 16) + tq * 4));
                v0 += f.x;
                v1 += f.y;
            }
            acc[n][2 * hh] = v0;
            acc[n][2 * hh + 1] = v1;
        }
    }
    if (p.lo) {
        // the next LayerNorm of the rows just made (as tlayernorm2 on the output; zeros past n2)
        float s[2] = {0.f, 0.f}, q[2] = {0.f, 0.f}, mean[2], rs[2];
#pragma unroll
        for (int n = 0; n < NJ2 * 2; n++)
            if (n * 8 + tq * 2 < p.n2)
#pragma unroll
                for (int hh = 0; hh < 2; hh++) s[hh] += acc[n][2 * hh] + acc[n][2 * hh + 1];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            s[i] += __shfl_xor_sync(0xffffffffu, s[i], 1);
            s[i] += __shfl_xor_sync(0xffffffffu, s[i], 2);
            mean[i] = s[i] / p.n2;
        }
#pragma unroll
        for (int n = 0; n < NJ2 * 2; n++)
            if (n * 8 + tq * 2 < p.n2)
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const float d0 = acc[n][2 * hh] - mean[hh], d1 = acc[n][2 * hh + 1] - mean[hh];
                    q[hh] += d0 * d0 + d1 * d1;
                }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            q[i] += __shfl_xor_sync(0xffffffffu, q[i], 1);
            q[i] += __shfl_xor_sync(0xffffffffu, q[i], 2);
            rs[i] = rsqrtf(q[i] / p.n2 + p.loeps);
        }
#pragma unroll
        for (int n = 0; n < NJ2 * 2; n++) {
            const int c = n * 8 + tq * 2;
            float2 gm = make_float2(0.f, 0.f), bt = gm;
            if (c < p.n2) {
                gm = __ldg(reinterpret_cast<const float2 *>(p.log_ + c));
                bt = __ldg(reinterpret_cast<const float2 *>(p.lob + c));
            }
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = hh ? r1 : r0;
                if (r >= p.M) continue;
                *reinterpret_cast<half2 *>(p.lo + (size_t)r * p.slo + c) =
                    __floats2half2_rn((acc[n][2 * hh] - mean[hh]) * rs[hh] * gm.x + bt.x, (acc[n][2 * hh + 1] - mean[hh]) * rs[hh] * gm.y + bt.y);
            }
        }
    }
#pragma unroll
    for (int n = 0; n < NJ2 * 2; n++) {
        const int c = n * 8 + tq * 2;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = hh ? r1 : r0;
            if (r >= p.M) continue;
            if (p.out_f32)
                *reinterpret_cast<float2 *>(reinterpret_cast<float *>(p.out) + (size_t)r * p.so + c) = make_float2(acc[n][2 * hh], acc[n][2 * hh + 1]);
            else
                *reinterpret_cast<half2 *>(p.out + (size_t)r * p.so + c) = __floats2half2_rn(acc[n][2 * hh], acc[n][2 * hh + 1]);
        }
    }
#else
    __trap();
#endif
}

template <int KC1, int NJ2, int NW, bool LN>
cudaError_t launch_mlp(const MlpP &p, cudaStream_t s)
{
    constexpr int sm = MlpSmem<KC1, NJ2>::bytes;
    static const cudaError_t attr = cudaFuncSetAttribute(k_mlp<KC1, NJ2, NW, LN>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (attr != cudaSuccess) return attr;
    k_mlp<KC1, NJ2, NW, LN><<<(unsigned)((p.M + 16 * NW - 1) / (16 * NW)), NW * 32, sm, s>>>(p);
    return cudaGetLastError();
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
    if (g.sa % 8 || g.so % 2 || (g.add1 && g.s1 % 2) || (g.add2 && g.s2 % 2) || g.sa < w.Kpad || g.so < w.Npad ||
        (g.add2scale && !g.add2))
        return cudaErrorInvalidValue;
    GemmP p{(const half *)g.a, g.sa, g.M, w.Kpad, (const uint4 *)w.frag, w.Npad / 16, (const float *)w.bias, g.colscale,
            (half *)g.out, g.so, (const half *)g.add1, (const half *)g.add2, g.s1, g.s2, (int)g.act, g.slope, g.act_last ? 1 : 0, g.out_f32 ? 1 : 0, g.add1_f32 ? 1 : 0, g.add2scale};
    p.k = w.K;
    p.lng = g.ln_gamma;
    p.lnb = g.ln_beta;
    p.lneps = g.ln_eps;
    if (g.ln_gamma && (g.add1 || g.add2 || g.act_last || w.K % 2 || !g.ln_beta || !(w.Kpad == 128 || w.Kpad == 192 || w.Kpad == 224 || w.Kpad == 256)))
        return cudaErrorNotSupported;
    // without adds, K = 128 / 192 / 224 / 256 (the transformers' qkv / fc1): the row-stationary kernel
    // (AJI_NATIVE_GEMM_RS=0: k_gemm)
    static const bool rs = [] {
        const char *v = std::getenv("AJI_NATIVE_GEMM_RS");
        return !v || std::atoi(v) != 0;
    }();
    if ((rs || g.ln_gamma) && !g.add1 && !g.add2 && !g.act_last && (w.Kpad == 128 || w.Kpad == 192 || w.Kpad == 224 || w.Kpad == 256)) {
        const unsigned grid = (unsigned)((g.M + 63) / 64);
        switch (w.Kpad) {
        case 128: k_gemm_rs<8><<<grid, 128, 4 * 8 * 32 * 16, s>>>(p); break;
        case 192: k_gemm_rs<12><<<grid, 128, 4 * 12 * 32 * 16, s>>>(p); break;
        case 224: k_gemm_rs<14><<<grid, 128, 4 * 14 * 32 * 16, s>>>(p); break;
        default: k_gemm_rs<16><<<grid, 128, 4 * 16 * 32 * 16, s>>>(p); break;
        }
        return cudaGetLastError();
    }
    const unsigned grid = (unsigned)((g.M + 63) / 64) * (unsigned)((w.Npad / 16 + 3) / 4);
    // an fp32 stream in place: see k_gemm's epilogue (with fp16 streams, as FDAT's, one pass is faster)
    if (g.out_f32 && ((g.add1 && g.add1 == g.out) || (g.add2 && g.add2 == g.out)))
        k_gemm<true><<<grid, 128, 0, s>>>(p);
    else
        k_gemm<false><<<grid, 128, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t tmlp(const TMat &w1, const TMat &w2, const TMlp &g, cudaStream_t s)
{
    // AJI_NATIVE_FMLP: 0 runs the two GEMMs instead, 1 the fused MLP without the LayerNorm (default 2)
    static const int mode = [] {
        const char *v = std::getenv("AJI_NATIVE_FMLP");
        return v && *v ? std::atoi(v) : 2;
    }();
    const bool ln = g.ln_gamma != nullptr;
    if (!mode || ((ln || g.lno) && mode < 2) || (ln && !g.ln_beta) || (g.lno && (!g.lno_gamma || !g.lno_beta || g.slno % 2 || g.slno < w2.Npad)) ||
        !w1.bias || !w2.bias || w2.Kpad != w1.Npad || w1.K % 2 || w2.N % 2 || g.sa % 2 || g.so % 2 ||
        (g.add1 && (g.s1 * (g.add1_f32 ? 4 : 2) % 16 || (uintptr_t)g.add1 % 16)) || g.sa < w1.Kpad || g.so < w2.Npad)
        return cudaErrorNotSupported;
    MlpP p{g.a, g.sa, g.M, w1.K, (const uint4 *)w1.frag, (const uint4 *)w2.frag, (const float *)w1.bias, (const float *)w2.bias,
           w1.Npad / 16, (half *)g.out, g.so, g.out_f32 ? 1 : 0, (const half *)g.add1, g.s1, g.add1_f32 ? 1 : 0, g.ln_gamma, g.ln_beta,
           g.ln_eps, (half *)g.lno, g.slno, w2.N, g.lno_gamma, g.lno_beta, g.lno_eps};
    const int kc1 = w1.Kpad / 16, nj2 = w2.Npad / 16;
    if (kc1 == 12 && nj2 == 12) return ln ? launch_mlp<12, 12, 4, true>(p, s) : launch_mlp<12, 12, 4, false>(p, s);
    return cudaErrorNotSupported;
}

cudaError_t tdwgemm(const TMat &w, const void *in, int sin, int H, int W, const float *dw, void *out, int so, const void *add1, int s1,
                    cudaStream_t s)
{
    static const bool on = [] {
        const char *v = std::getenv("AJI_NATIVE_DWMM");
        return !v || std::atoi(v) != 0;
    }();
    const int nj2 = w.Npad / 16;
    if (!on || (nj2 != 8 && nj2 != 12) || w.K % 2 || sin % 8 || sin < w.Kpad || (uintptr_t)in % 16 || so % 2 || so < w.Npad ||
        (uintptr_t)dw % 16 || (add1 && (s1 % 8 || s1 < w.Npad || (uintptr_t)add1 % 16)) || H < 1 || W < 1 || W % 64)
        return cudaErrorNotSupported;
    DwmmP p{(const half *)in, sin, H, W, H * W, w.K, dw, (const uint4 *)w.frag, (const float *)w.bias, w.Kpad / 16, (half *)out, so,
            (const half *)add1, s1};
    return nj2 == 8 ? launch_dwmm<8>(p, s) : launch_dwmm<12>(p, s);
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

cudaError_t tlayernorm2(const void *in, bool in_f32, int sin, void *out, bool out_f32, int sout, int M, int C, int cpad,
                        const float *gamma, const float *beta, float eps, cudaStream_t s)
{
    if (C > 256 || cpad % 8 || cpad > 256 || sin % 8 || sout % 8 || ((uintptr_t)in | (uintptr_t)out) % 16) return cudaErrorInvalidValue;
    const bool l16 = cpad <= 128;
    const unsigned grid = (unsigned)((M + (l16 ? 15 : 7)) / (l16 ? 16 : 8));
#define LNX(TI, TO)                                                                                                                       \
    do {                                                                                                                                  \
        if (l16)                                                                                                                          \
            k_ln8x<16, TI, TO><<<grid, 256, 0, s>>>((const TI *)in, sin, (TO *)out, sout, M, C, cpad, gamma, beta, eps);                \
        else                                                                                                                              \
            k_ln8x<32, TI, TO><<<grid, 256, 0, s>>>((const TI *)in, sin, (TO *)out, sout, M, C, cpad, gamma, beta, eps);                \
    } while (0)
    if (in_f32 && out_f32) LNX(float, float);
    else if (in_f32) LNX(float, half);
    else if (out_f32) LNX(half, float);
    else LNX(half, half);
#undef LNX
    return cudaGetLastError();
}

cudaError_t tconvert(const void *in, bool in_f32, int sin, void *out, bool out_f32, int sout, int M, int n, cudaStream_t s, float scale)
{
    if (n % 8 || sin % 8 || sout % 8 || ((uintptr_t)in | (uintptr_t)out) % 16) return cudaErrorInvalidValue;
    const unsigned grid = (unsigned)(((size_t)M * (n / 8) + 255) / 256);
    if (in_f32 && !out_f32) k_convert<float, half><<<grid, 256, 0, s>>>((const float *)in, sin, (half *)out, sout, M, n, scale);
    else if (!in_f32 && out_f32) k_convert<half, float><<<grid, 256, 0, s>>>((const half *)in, sin, (float *)out, sout, M, n, scale);
    else if (in_f32) k_convert<float, float><<<grid, 256, 0, s>>>((const float *)in, sin, (float *)out, sout, M, n, scale);
    else k_convert<half, half><<<grid, 256, 0, s>>>((const half *)in, sin, (half *)out, sout, M, n, scale);
    return cudaGetLastError();
}

bool tbias_table(const std::vector<float> &bias, int heads, int wh, int ww, int kwh, int kww, int kpad, std::vector<float> *tab,
                 int *rt_w, int *rt_offy, int *rt_offx)
{
    const int kh = kwh ? kwh : wh, kw = kwh ? kww : ww, kp = kwh ? kpad : 0, Tq = wh * ww, Tk = kh * kw;
    const int th = kh + wh - 1, tw = kw + ww - 1, offy = kp + wh - 1, offx = kp + ww - 1;
    if ((int)bias.size() != heads * Tq * Tk) return false;
    tab->assign((size_t)heads * th * tw, 0.f);
    std::vector<char> seen((size_t)heads * th * tw, 0);
    for (int h = 0; h < heads; h++)
        for (int r = 0; r < Tq; r++)
            for (int c = 0; c < Tk; c++) {
                const int dy = c / kw - kp - r / ww + offy, dx = c % kw - kp - r % ww + offx;
                if (dy < 0 || dy >= th || dx < 0 || dx >= tw) return false;
                const size_t ti = ((size_t)h * th + dy) * tw + dx;
                const float v = bias[((size_t)h * Tq + r) * Tk + c];
                if (seen[ti] && (*tab)[ti] != v) return false;
                seen[ti] = 1;
                (*tab)[ti] = v;
            }
    *rt_w = tw;
    *rt_offy = offy;
    *rt_offx = offx;
    return true;
}

cudaError_t twinattn(const TWinAttn &a, cudaStream_t s)
{
    const int T = a.wh * a.ww;
    if (a.kwh || a.mask || a.rtab || (a.Hp && a.Hp != a.H) || (a.Wp && a.Wp != a.W)) {
        const int Tk = a.kwh ? a.kwh * a.kww : T;
        if (a.hd > 32 || a.hd % 2 || T % 16 || Tk % 16 || T > 1024 || a.sq % 2 || a.so % 2 || (a.kwh && (a.shift_y || a.shift_x)) ||
            (a.mask && a.kwh))
            return cudaErrorInvalidValue;
        const int Hp = a.Hp ? a.Hp : a.H, Wp = a.Wp ? a.Wp : a.W;
        if (Hp < a.H || Wp < a.W || (a.kwh && (Hp != a.H || Wp != a.W))) return cudaErrorInvalidValue;
        const int nwx = (Wp + a.ww - 1) / a.ww, nwy = (Hp + a.wh - 1) / a.wh;
        const int tn = a.rtab ? ((a.kwh ? a.kwh : a.wh) + a.wh - 1) * a.rt_w : 0;
        WinG g{{(const half *)a.qkv, a.sq, a.qo, a.ko, a.vo, a.H, a.W, a.heads, a.hd, a.wh, a.ww, a.shift_y, a.shift_x, nwx, Hp, Wp, a.scale,
                a.bias, (half *)a.out, a.so},
               a.kwh, a.kww, a.kpad, a.mask ? 1 : 0, a.rtab, a.rt_w, a.rt_offy, a.rt_offx, tn};
        const int smem = (Tk * HP * 2 * 2 + tn * 4 + (a.kwh ? 0 : Tk * 4) + (a.rtab ? Tk * 2 : 0) + (a.mask ? Tk : 0) + 15) / 16 * 16;
        static int set = 0;
        if (smem > 48 * 1024 && smem > set) {
            if (cudaFuncSetAttribute(k_winattn_g, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return cudaErrorInvalidValue;
            set = smem;
        }
        // Warps per CTA: as many as the registers allow with the CTAs the shared memory fits per SM
        // (each CTA's warps take 16-query blocks in turn); AJI_NATIVE_WINWARPS overrides.
        static const int smem_sm = [] {
            int dev = 0, v = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
            return v;
        }();
        static const int force = [] {
            const char *v = std::getenv("AJI_NATIVE_WINWARPS");
            return v ? std::atoi(v) : 0;
        }();
        const int per_sm = std::max(1, smem_sm / (smem + 1024));
        int nw = force > 0 ? force : per_sm >= 4 ? 4 : per_sm >= 2 ? 8 : 16;
        nw = std::max(1, std::min({nw, 16, T / 16}));
        k_winattn_g<<<(unsigned)(nwx * nwy) * (unsigned)a.heads, 32 * nw, smem, s>>>(g);
        return cudaGetLastError();
    }
    if (a.hd > 32 || a.hd % 2 || T % 16 || T > 256 || a.sq % 2 || a.so % 2) return cudaErrorInvalidValue;
    const int nwx = (a.W + a.ww - 1) / a.ww, nwy = (a.H + a.wh - 1) / a.wh;
    WinP p{(const half *)a.qkv, a.sq, a.qo, a.ko, a.vo, a.H, a.W, a.heads, a.hd, a.wh, a.ww, a.shift_y, a.shift_x, nwx, a.H, a.W, a.scale,
           a.bias, (half *)a.out, a.so};
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
    const int C8 = (C + 7) / 8 * 8;  // k_dw8s reads (not writes) the rows this wide
    const bool vec = C % 2 == 0 && C <= 1024 && sin % 8 == 0 && sout % 8 == 0 && sin >= C8 && sout >= C8 &&
                     ((uintptr_t)in | (uintptr_t)out) % 16 == 0;
    if (vec) {
        const long long n = (long long)(C8 / 8) * W * ((H + DW_R - 1) / DW_R);
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

cudaError_t tdwconv2(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, const float *bias, bool gelu_,
                     const void *mul, int smul, float *sums, cudaStream_t s)
{
    const int C8 = (C + 7) / 8 * 8;
    if (C % 2 || C > 1024 || sin % 8 || sout % 8 || sin < C8 || sout < C8 || (mul && (smul % 8 || smul < C8)) ||
        ((uintptr_t)in | (uintptr_t)out | (uintptr_t)mul) % 16)
        return cudaErrorInvalidValue;
    const long long n = (long long)(C8 / 8) * W * ((H + DW_R - 1) / DW_R);
    k_dw8s<<<(unsigned)((n + 127) / 128), 128, 0, s>>>((const half *)in, sin, (half *)out, sout, H, W, C, w, gelu_ ? 1 : 0, sums, bias,
                                                      (const half *)mul, smul);
    return cudaGetLastError();
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
        const int smem = 2 * CS_ROWS * (heads * 32 + 8) * 2;
        static const int sms = [] {
            int dev = 0, n = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
            cudaFuncSetAttribute(k_chanstats_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * CS_ROWS * (CS_MAXH * 32 + 8) * 2);
            return std::max(1, n);
        }();
        // ~2 waves of 3 CTAs per SM (each CTA's partial sums go out as atomics: fewer, longer CTAs)
        const int tpc = std::max(CS_ROWS, (M + 6 * sms - 1) / (6 * sms) + CS_ROWS - 1) / CS_ROWS * CS_ROWS;
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
