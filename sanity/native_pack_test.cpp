// native/pack: the weight packing against the mma.sync m16n8k16 B-fragment layout as the PTX ISA
// states it (row = 2t + (i & 1) + 8 * (i >= 2), col = g for element i of lane 4g + t), written
// out independently here; and the host fp16 conversion. CPU only.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "native/pack.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

using namespace ajinative;

static int check_layout(int taps, int cin, int cin_pad, int cout, int cout_pad)
{
    std::vector<uint16_t> w((size_t)taps * cin * cout);
    for (size_t i = 0; i < w.size(); i++) w[i] = (uint16_t)(1 + (i * 2654435761u) % 0x7bff);  // nonzero
    const std::vector<uint32_t> p = pack_fragments(w.data(), taps, cin, cin_pad, cout, cout_pad);
    const int KC = cin_pad / 16, NJ = cout_pad / 16;
    CHECK(p.size() == (size_t)taps * KC * NJ * 32 * 4);
    std::vector<int> seen((size_t)taps * cin_pad * cout_pad, 0);
    for (int t = 0; t < taps; t++)
        for (int kc = 0; kc < KC; kc++)
            for (int j = 0; j < NJ; j++)
                for (int lane = 0; lane < 32; lane++) {
                    const uint32_t *q = &p[((((size_t)t * KC + kc) * NJ + j) * 32 + lane) * 4];
                    const int g = lane / 4, tq = lane % 4;
                    for (int tile = 0; tile < 2; tile++)       // n-tiles 2j, 2j+1
                        for (int i = 0; i < 4; i++) {          // b0.lo, b0.hi, b1.lo, b1.hi
                            const uint32_t word = q[tile * 2 + i / 2];
                            const uint16_t v = (uint16_t)(i % 2 ? word >> 16 : word & 0xffffu);
                            const int k = kc * 16 + 2 * tq + (i & 1) + (i >= 2 ? 8 : 0);
                            const int n = (2 * j + tile) * 8 + g;
                            const uint16_t want = (k < cin && n < cout) ? w[((size_t)t * cin + k) * cout + n] : 0;
                            CHECK(v == want);
                            seen[((size_t)t * cin_pad + k) * cout_pad + n]++;
                        }
                }
    for (int s : seen) CHECK(s == 1);  // every (tap, k, n) exactly once
    return 0;
}

int main()
{
    CHECK(check_layout(9, 48, 48, 48, 48) == 0);
    CHECK(check_layout(9, 12, 16, 64, 64) == 0);   // unshuffled head
    CHECK(check_layout(9, 3, 16, 32, 32) == 0);    // NCHW3 head
    CHECK(check_layout(9, 64, 64, 12, 16) == 0);   // depth-to-space 2 tail
    CHECK(check_layout(1, 192, 192, 48, 48) == 0); // conv_cat
    CHECK(check_layout(2, 20, 32, 24, 32) == 0);   // ragged

    // fp16 conversion: every finite half survives a round trip; rounding is to nearest even
    for (uint32_t h = 0; h < 0x10000u; h++) {
        if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu)) continue;  // nan
        CHECK(float_to_half_bits(half_bits_to_float((uint16_t)h)) == h);
    }
    CHECK(float_to_half_bits(1.0f) == 0x3c00);
    CHECK(float_to_half_bits(-2.5f) == 0xc100);
    CHECK(float_to_half_bits(65504.0f) == 0x7bff);
    CHECK(float_to_half_bits(65520.0f) == 0x7c00);
    CHECK(float_to_half_bits(1.0f + 1.0f / 2048) == 0x3c00);           // tie -> even
    CHECK(float_to_half_bits(1.0f + 3.0f / 2048) == 0x3c02);           // tie -> even (up)
    CHECK(float_to_half_bits(std::ldexp(1.0f, -24)) == 0x0001);        // smallest subnormal
    CHECK(float_to_half_bits(std::ldexp(1.0f, -25)) == 0x0000);        // tie -> even (zero)
    CHECK(float_to_half_bits(std::ldexp(3.0f, -26)) == 0x0001);
    std::printf("native pack: OK\n");
    return 0;
}
