/*
 * core/half.h — scalar fp32 <-> fp16 / bf16 conversions (round to
 * nearest even) for host-side tensor staging.
 */

#ifndef AJI_CORE_HALF_H
#define AJI_CORE_HALF_H

#include <cstdint>
#include <cstring>

namespace ajicore {

inline uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x7fffffu;
    int32_t exp = (int32_t)((x >> 23) & 0xffu);
    if (exp == 0xff)
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    exp = exp - 127 + 15;
    if (exp >= 0x1f)
        return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return (uint16_t)sign;
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (h & 1u)))
            h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
        h++;  // a carry into the exponent is the correct rounding
    return (uint16_t)h;
}

inline float f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    if ((x & 0x7fffffffu) > 0x7f800000u)
        return (uint16_t)((x >> 16) | 0x40u);  // quiet NaN
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

inline float bf16_to_f32(uint16_t b)
{
    const uint32_t x = (uint32_t)b << 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

} // namespace ajicore

#endif
