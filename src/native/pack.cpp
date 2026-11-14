// native/pack.cpp — see pack.h.
#include "native/pack.h"

#include <cmath>
#include <cstring>

namespace ajinative {

std::vector<uint32_t> pack_fragments(const uint16_t *w, int taps, int cin, int cin_pad, int cout, int cout_pad)
{
    const int KC = cin_pad / 16, NJ = cout_pad / 16;
    std::vector<uint32_t> v((size_t)taps * KC * NJ * 32 * 4, 0u);
    auto at = [&](int t, int k, int n) -> uint32_t {
        return (k < cin && n < cout) ? w[((size_t)t * cin + k) * cout + n] : 0u;
    };
    for (int t = 0; t < taps; t++)
        for (int kc = 0; kc < KC; kc++)
            for (int j = 0; j < NJ; j++)
                for (int lane = 0; lane < 32; lane++) {
                    const int g = lane / 4, tq = lane % 4, k0 = kc * 16 + tq * 2;
                    uint32_t *o = &v[((((size_t)t * KC + kc) * NJ + j) * 32 + lane) * 4];
                    for (int h = 0; h < 2; h++) {
                        const int n = (2 * j + h) * 8 + g;
                        o[h * 2 + 0] = at(t, k0, n) | (at(t, k0 + 1, n) << 16);
                        o[h * 2 + 1] = at(t, k0 + 8, n) | (at(t, k0 + 9, n) << 16);
                    }
                }
    return v;
}

uint16_t float_to_half_bits(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u)  // inf / nan
        return (uint16_t)(sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));
    if (ax >= 0x477ff000u)  // rounds to >= 65520: overflow
        return (uint16_t)(sign | 0x7c00u);
    if (ax < 0x38800000u) {  // subnormal half (or zero)
        if (ax < 0x33000000u) return (uint16_t)sign;
        const uint32_t m = (ax & 0x7fffffu) | 0x800000u;
        const int shift = 126 - (int)(ax >> 23);  // 14..24
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((ax - 0x38000000u) >> 13);
    const uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
    return (uint16_t)(sign | r);
}

float half_bits_to_float(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) x = sign;
        else { const float f = std::ldexp((float)m, -24); std::memcpy(&x, &f, 4); x |= sign; }
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

}  // namespace ajinative
