#include "half_array.h"
#include "half.h"

#ifndef AJI_HAVE_F16C
#define AJI_HAVE_F16C 0
#endif

#if AJI_HAVE_F16C
#include <intrin.h>
#endif

namespace ajicore {
#if AJI_HAVE_F16C
// These functions live in a separate translation unit compiled with /arch:AVX.
void pack_f16c(const float *, uint16_t *, size_t);
void unpack_f16c(const uint16_t *, float *, size_t);

static bool cpu_has_f16c()
{
    int caps[4];
    __cpuid(caps, 1);
    const unsigned required = (1u << 27) | (1u << 28) | (1u << 29);
    return (static_cast<unsigned>(caps[2]) & required) == required &&
           (_xgetbv(0) & 6) == 6; // OS saves both XMM and YMM state.
}
#endif

bool half_array_uses_f16c()
{
#if AJI_HAVE_F16C
    static const bool supported = cpu_has_f16c();
    return supported;
#else
    return false;
#endif
}

void f32_to_f16_array(const float *src, uint16_t *dst, size_t n)
{
#if AJI_HAVE_F16C
    if (n >= 8 && half_array_uses_f16c()) {
        pack_f16c(src, dst, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i)
        dst[i] = f32_to_f16(src[i]);
}

void f16_to_f32_array(const uint16_t *src, float *dst, size_t n)
{
#if AJI_HAVE_F16C
    if (n >= 8 && half_array_uses_f16c()) {
        unpack_f16c(src, dst, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i)
        dst[i] = f16_to_f32(src[i]);
}
} // namespace ajicore
