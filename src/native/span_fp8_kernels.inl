__host__ __device__ constexpr int fp8_stride(int c) { return (c / 16) % 2 ? c : c + 16; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
__device__ __forceinline__ uint32_t fp8x4(uint32_t lo, uint32_t hi)
{
    uint32_t packed;
    asm("{ .reg .b16 a,b; cvt.rn.satfinite.e4m3x2.f16x2 a, %1;"
        " cvt.rn.satfinite.e4m3x2.f16x2 b, %2; mov.b32 %0,{a,b}; }" : "=r"(packed) : "r"(lo), "r"(hi));
    return packed;
}

__device__ __forceinline__ void fp8_mma32(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
        : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void fp8_mma16(uint32_t (&d)[2], const uint32_t (&a)[2], uint32_t b)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3},{%4},{%0,%1};"
        : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(b));
}
#endif

// Included inside ajinative::dev, after the native warp load and half conversion helpers.
__device__ __forceinline__ uint16_t fp8_pair(uint32_t values)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(packed) : "r"(values));
    return packed;
#else
    __trap();
    return 0;
#endif
}

template <int C, int WARPS, int HP>
__device__ __forceinline__ void load_packed_fp8_halo(unsigned char *sm, const ConvParams &p, int tx0, int ty0)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    constexpr int vectors = C / 16;
    const auto *src = reinterpret_cast<const unsigned char *>(p.src);
    for (int i = threadIdx.x; i < HP * vectors; i += WARPS * 32) {
        const int pixel = i / vectors, column = (i % vectors) * 16;
        const int y = ty0 - 1 + pixel / (TW + 2), x = tx0 - 1 + pixel % (TW + 2);
        const bool valid = y >= 0 && y < p.H && x >= 0 && x < p.W;
        cp_async16(sm + pixel * fp8_stride(C) + column, valid ? src + (size_t(y) * p.W + x) * C + column : src,
                   valid ? 16 : 0);
    }
    cp_async_wait_all();
#else
    __trap();
#endif
}

template<int CINP,int WARPS,int HP>
__device__ __forceinline__ void load_fp8_halo(unsigned char *smraw, const ConvParams &p, int tx0, int ty0)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const int tid = threadIdx.x, H = p.H, W = p.W;
    constexpr int stride = fp8_stride(CINP);
    for (int i = tid; i < HP * (CINP/8); i += WARPS*32) {
        const int px = i/(CINP/8), v = i%(CINP/8);
        const int y = ty0-1 + px/(TW+2), x = tx0-1 + px%(TW+2);
        uint2 pair = make_uint2(0,0);
        if (y>=0 && y<H && x>=0 && x<W) {
            const uint4 q = __ldg(reinterpret_cast<const uint4 *>(p.src + (size_t(y)*W+x)*CINP)+v);
            pair = make_uint2(fp8x4(q.x,q.y),fp8x4(q.z,q.w));
        }
        *reinterpret_cast<uint2 *>(smraw + px*stride+8*v) = pair;
    }
#else
    __trap();
#endif
}

template<int CINP,int COUTP,int WR,int NT>
__device__ __forceinline__ void fp8_product(uint32_t (&acc)[WR][NT][2], unsigned char *smraw,
                                           const ConvParams &p, int row0, int lane)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    constexpr int NJ = COUTP / 16;
    constexpr int stride = fp8_stride(CINP), full = CINP / 32;
    const unsigned char *weights = reinterpret_cast<const unsigned char *>(p.w);
#pragma unroll 1
    for (int t = 0; t < 9; ++t) {
        const int dx = t % 3, dy = t / 3;
        const auto *base = weights + t * CINP * COUTP;
#pragma unroll
        for (int kc = 0; kc < full; ++kc) {
            uint4 b[NJ];
#pragma unroll
            for (int j = 0; j < NJ; ++j)
                b[j] = __ldg(reinterpret_cast<const uint4 *>(base) + (kc * NJ + j) * 32 + lane);
#pragma unroll
            for (int i = 0; i < WR; ++i) {
                uint32_t a[4];
                ldsm_x4(a, smraw + ((row0+i+dy)*(TW+2)+(lane&15)+dx)*stride + 32*kc + (lane>>4)*16);
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    fp8_mma32(acc[i][2*j],a,b[j].x,b[j].y);
                    fp8_mma32(acc[i][2*j+1],a,b[j].z,b[j].w);
                }
            }
        }
        if constexpr (CINP % 32) {
            uint2 b[NJ];
#pragma unroll
            for (int j = 0; j < NJ; ++j)
                b[j] = __ldg(reinterpret_cast<const uint2 *>(base + full*32*COUTP)+j*32+lane);
#pragma unroll
            for (int i = 0; i < WR; ++i) {
                uint32_t a[2];
                ldsm_x2(a,smraw+((row0+i+dy)*(TW+2)+(lane&15)+dx)*stride + full*32);
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    fp8_mma16(acc[i][2*j],a,b[j].x);
                    fp8_mma16(acc[i][2*j+1],a,b[j].y);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WR; ++i)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                acc[i][n][h] = h2u(__hmul2(u2h(acc[i][n][h]), __float2half2_rn(1.f/16.f)));
#else
    __trap();
#endif
}
