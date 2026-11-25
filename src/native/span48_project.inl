// Reuse the final gate's half2 registers as the second A fragment in the
// four-input projection. Accumulation stays in the original source and K order.
#if AJI_NATIVE_ARCH_OK
template<int ROWS>
__device__ __forceinline__ void span48_project(const GateProjectParams &p, const uint32_t (&gate)[ROWS][6][2],
                                               half *out, int x0, int y0, int lane)
{
    const int g = lane >> 2, pair = lane & 3;
    uint32_t result[ROWS][6][2] = {};
#pragma unroll
    for (int src = 0; src < 4; ++src) {
#pragma unroll
        for (int kc = 0; kc < 3; ++kc) {
            uint4 weights[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) weights[j] = __ldg(p.project_w + ((src * 3 + kc) * 3 + j) * 32 + lane);
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
                            const auto *v = reinterpret_cast<const uint32_t *>(p.maps[src] + ((size_t)y * p.W + x) * 48 + kc * 16);
                            a[h] = __ldg(v + pair);
                            a[2 + h] = __ldg(v + pair + 4);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    mma16816(result[r][2 * j], a, weights[j].x, weights[j].y);
                    mma16816(result[r][2 * j + 1], a, weights[j].z, weights[j].w);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int n = 0; n < 6; ++n) {
            const half2 bias = __ldg(p.project_bias + n * 4 + pair);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int px = r * TW + g + 8 * h;
                reinterpret_cast<half2 *>(out + px * smem_stride(48))[n * 4 + pair] = __hadd2(u2h(result[r][n][h]), bias);
            }
        }
}
#endif
