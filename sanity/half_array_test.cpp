// Exact host conversion parity, including the NaN/MXCSR gap in the prototype.
#include "core/half.h"
#include "core/half_array.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>
#define AJI_TEST_MXCSR 1
#else
#define AJI_TEST_MXCSR 0
#endif

using namespace ajicore;

static void require(bool ok, const char *message)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main()
{
    std::vector<uint16_t> halves(65536), half_expected(65536);
    std::vector<float> expected(65536), actual(65538);
    for (unsigned i = 0; i < halves.size(); ++i) {
        halves[i] = static_cast<uint16_t>(i);
        expected[i] = f16_to_f32(halves[i]);
        half_expected[i] = f32_to_f16(expected[i]);
    }
    std::vector<float> inputs(1); // Offset the source and destination addresses.
    auto add_bits = [&](uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        inputs.push_back(f);
    };
    // Adjacent half midpoints and their FP32 neighbours protect ties-to-even.
    for (unsigned h = 0; h < 0x7bff; ++h) {
        float mid = (f16_to_f32(static_cast<uint16_t>(h)) +
                     f16_to_f32(static_cast<uint16_t>(h + 1))) * .5f;
        uint32_t bits;
        std::memcpy(&bits, &mid, sizeof(bits));
        for (uint32_t b : {bits - 1, bits, bits + 1}) {
            add_bits(b);
            add_bits(b | 0x80000000u);
        }
    }
    // Includes overflow, infinities, signed zeros, subnormals and NaN payloads.
    for (uint32_t bits : {0u, 0x80000000u, 1u, 0x80000001u,
                         0x477fe000u, 0x477ff000u, 0x47800000u,
                         0x7f800000u, 0xff800000u, 0x7f800001u,
                         0xff800001u, 0x7fc12345u, 0xffc12345u})
        add_bits(bits);
    uint32_t rng = 7711;
    for (unsigned i = 0; i < (1u << 20); ++i) {
        rng = rng * 1664525u + 1013904223u;
        add_bits(rng);
    }
    const size_t n = inputs.size() - 1;
    std::vector<uint16_t> packed(n + 2), reference(n);
    for (size_t i = 0; i < n; ++i)
        reference[i] = f32_to_f16(inputs[i + 1]);

#if AJI_TEST_MXCSR
    const unsigned saved = _mm_getcsr();
    // Non-default rounding, DAZ/FTZ, sticky flags and unmasked exceptions.
    const unsigned controls[] = {0x1f80u, 0xbfc0u, 0x4000u, 0x7fbfu};
#else
    const unsigned controls[] = {0};
#endif
    for (unsigned control : controls) {
#if AJI_TEST_MXCSR
        _mm_setcsr(control);
#endif
        f16_to_f32_array(halves.data(), actual.data() + 1, halves.size());
#if AJI_TEST_MXCSR
        require(_mm_getcsr() == control, "unpack changed MXCSR");
#endif
        require(std::memcmp(expected.data(), actual.data() + 1,
                            expected.size() * sizeof(float)) == 0,
                "half-to-float output bits differ");
        f32_to_f16_array(expected.data(), packed.data() + 1, expected.size());
        require(std::memcmp(half_expected.data(), packed.data() + 1,
                            half_expected.size() * sizeof(uint16_t)) == 0,
                "nonfinite repack semantics differ");
        f32_to_f16_array(inputs.data() + 1, packed.data() + 1, n);
#if AJI_TEST_MXCSR
        require(_mm_getcsr() == control, "pack changed MXCSR");
#endif
        require(std::memcmp(reference.data(), packed.data() + 1,
                            n * sizeof(uint16_t)) == 0,
                "float-to-half rounding or payload differs");
        // Scalar-only short arrays and a vector with an odd tail.
        for (size_t count : {size_t(0), size_t(3), size_t(11)}) {
            f32_to_f16_array(inputs.data() + 1, packed.data() + 1, count);
            require(std::memcmp(reference.data(), packed.data() + 1,
                                count * sizeof(uint16_t)) == 0, "pack tail differs");
            f16_to_f32_array(halves.data() + 1, actual.data() + 1, count);
            require(std::memcmp(expected.data() + 1, actual.data() + 1,
                                count * sizeof(float)) == 0, "unpack tail differs");
        }
    }
#if AJI_TEST_MXCSR
    _mm_setcsr(saved);
#endif
    std::printf("PASS: f16c=%d; 65536 half patterns, %zu float patterns, "
                "NaN bits, rounding/tails and MXCSR unchanged\n",
                half_array_uses_f16c(), n);
}
