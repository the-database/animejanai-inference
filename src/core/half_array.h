// Host tensor staging. Array conversions retain half.h's bit-level semantics.
#ifndef AJI_CORE_HALF_ARRAY_H
#define AJI_CORE_HALF_ARRAY_H

#include <cstddef>
#include <cstdint>

namespace ajicore {
bool half_array_uses_f16c();
void f32_to_f16_array(const float *src, uint16_t *dst, size_t n);
void f16_to_f32_array(const uint16_t *src, float *dst, size_t n);
} // namespace ajicore

#endif
