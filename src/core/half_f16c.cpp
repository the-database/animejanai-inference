// Compiled separately with AVX, called only after CPU/OS capability checks.
#include "half_array.h"
#include "half.h"
#include <immintrin.h>

namespace ajicore {
namespace {
// The scalar helpers use integer operations and leave MXCSR unchanged. F16C
// can raise conversion exceptions even when its rounding mode is immediate.
struct PreserveMxcsr {
    unsigned saved = _mm_getcsr();
    PreserveMxcsr() { _mm_setcsr(saved | _MM_MASK_MASK); }
    ~PreserveMxcsr() { _mm_setcsr(saved); }
};
}

void pack_f16c(const float *src, uint16_t *dst, size_t n)
{
    PreserveMxcsr csr;
    const __m256 signless = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 infinity = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 input = _mm256_loadu_ps(src + i);
        const __m256 finite = _mm256_cmp_ps(_mm256_and_ps(input, signless),
                                           infinity, _CMP_LT_OQ);
        const __m128i half = _mm256_cvtps_ph(_mm256_and_ps(input, finite),
                                           _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), half);
        const unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(finite));
        if (mask != 255) {
            // Canonicalize NaNs exactly like half.h, including their sign.
            for (unsigned lane = 0; lane < 8; ++lane)
                if (!(mask & (1u << lane)))
                    dst[i + lane] = f32_to_f16(src[i + lane]);
        }
    }
    for (; i < n; ++i)
        dst[i] = f32_to_f16(src[i]);
}

void unpack_f16c(const uint16_t *src, float *dst, size_t n)
{
    PreserveMxcsr csr;
    const __m128i exponent = _mm_set1_epi16(0x7c00);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i input = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
        const __m128i nonfinite = _mm_cmpeq_epi16(_mm_and_si128(input, exponent), exponent);
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_andnot_si128(nonfinite, input)));
        const unsigned mask = static_cast<unsigned>(_mm_movemask_epi8(nonfinite));
        if (mask) {
            // Preserve the payload and signaling bit instead of quieting NaNs.
            for (unsigned lane = 0; lane < 8; ++lane)
                if (mask & (1u << (2 * lane)))
                    dst[i + lane] = f16_to_f32(src[i + lane]);
        }
    }
    for (; i < n; ++i)
        dst[i] = f16_to_f32(src[i]);
}
} // namespace ajicore
