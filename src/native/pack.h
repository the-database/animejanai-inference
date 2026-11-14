/*
 * native/pack.h — weight packing for the native kernels, host only (no CUDA dependency).
 *
 * mma.sync m16n8k16 (.row.col) takes B as a 16x8 K x N tile spread over the warp: lane l holds
 *   b0 = {B[2t][g], B[2t+1][g]}, b1 = {B[2t+8][g], B[2t+9][g]}   with g = l / 4, t = l % 4,
 * the lower 16 bits holding the first element. The kernels consume two adjacent n8 tiles per
 * 16-byte load, so for every (tap, 16-row K chunk kc, 16-column N chunk j, lane) the packed array
 * holds one uint4 = {b0, b1} of n-tile 2j followed by {b0, b1} of n-tile 2j + 1.
 */
#ifndef AJI_NATIVE_PACK_H
#define AJI_NATIVE_PACK_H

#include <cstdint>
#include <vector>

namespace ajinative {

// w: [taps][cin][cout] fp16 bit patterns. Returns taps * (cin_pad/16) * (cout_pad/16) * 32 * 4
// words; entries with k >= cin or n >= cout are zero. cin_pad, cout_pad: multiples of 16.
std::vector<uint32_t> pack_fragments(const uint16_t *w, int taps, int cin, int cin_pad, int cout, int cout_pad);

// fp32 -> fp16 bits, round to nearest even (host side; bias and slope uploads, tests).
uint16_t float_to_half_bits(float f);
float half_bits_to_float(uint16_t h);

}  // namespace ajinative

#endif
