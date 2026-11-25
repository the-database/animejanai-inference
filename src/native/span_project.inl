// Reuse the final gate's half2 registers as the second A fragment in the
// four-input projection. Accumulation stays in the original source and K order.
#if AJI_NATIVE_ARCH_OK
template<int C, int ROWS>
__device__ __forceinline__ void span_project(const GateProjectParams &p, const uint32_t (&gate)[ROWS][C / 8][2],
                                               half *out, int x0, int y0, int lane)
{
    constexpr int chunks = C / 16;
    const int g = lane >> 2, pair = lane & 3;
    uint32_t result[ROWS][C / 8][2] = {};
#pragma unroll
    for (int src = 0; src < 4; ++src) {
#pragma unroll
        for (int kc = 0; kc < chunks; ++kc) {
            uint4 weights[chunks];
#pragma unroll
            for (int j = 0; j < chunks; ++j) weights[j] = __ldg(p.project_w + ((src * chunks + kc) * chunks + j) * 32 + lane);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                uint32_t a[4] = {};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (src == 1) {
                        a[h] = gate[r][2 * kc][h];
                        a[2 + h] = gate[r][2 * kc + 1][h];
                    } else {
                        const int x = x0 + g + 8 * h, y = y0 + r;
                        if (x < p.W && y < p.H) {
                            const auto *v = reinterpret_cast<const uint32_t *>(p.maps[src] + ((size_t)y * p.W + x) * C + kc * 16);
                            a[h] = __ldg(v + pair);
                            a[2 + h] = __ldg(v + pair + 4);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < chunks; ++j) {
                    mma16816(result[r][2 * j], a, weights[j].x, weights[j].y);
                    mma16816(result[r][2 * j + 1], a, weights[j].z, weights[j].w);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int n = 0; n < C / 8; ++n) {
            const half2 bias = __ldg(p.project_bias + n * 4 + pair);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int px = r * TW + g + 8 * h;
                reinterpret_cast<half2 *>(out + px * smem_stride(C))[n * 4 + pair] = __hadd2(u2h(result[r][n][h]), bias);
            }
        }
}
#endif
