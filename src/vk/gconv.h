/*
 * vk/gconv.h — the general 3x3 conv on Vulkan (src/vk/shaders/gconv_*.comp), the counterpart of
 * libaji's CUDA gconv (native/gconv.h) for the larger image networks: a channel prefix of a wider
 * NHWC buffer in, a channel slice of another out, bias, up to two scaled residuals and LeakyReLU
 * applied per element in fp32, an optional nearest 2x upsample of the input on the fly. Inputs
 * can also be three NCHW planes (or their 2x pixel-unshuffle), and the output three NCHW planes.
 *
 * Shapes are compiled modules (shaders.txt lists the ones built); offsets and strides are per
 * dispatch, in a parameter entry (GConvParams) the caller places in a buffer.
 */
#ifndef AJI_VK_GCONV_H
#define AJI_VK_GCONV_H

#include <cstdint>
#include <string>
#include <vector>

#include "vk/device.h"
#include "vk/kernels.h"

namespace ajivk {

// A conv's compiled shape: the module's defines (gconv_common.glsl).
struct GConvShape {
    int cinp = 16, coutp = 16;
    int in = 0;    // 0 NHWC, 1 NCHW planes (3 channels), 2 pixel-unshuffle of NCHW planes (12 channels)
    int act = 0;   // 0 none, 2 LeakyReLU
    int nres = 0;  // residuals added
    int up2 = 0;   // input nearest-upsampled by 2
    int out = 0;   // 0 NHWC, 3 NCHW planes (channels 0..2)
};

// One dispatch's parameters (16 ints, the Params buffer entry layout). Offsets and strides in
// halfs, multiples of 8.
struct GConvParams {
    int32_t H = 0, W = 0;
    int32_t src_off = 0, src_stride = 0, dst_off = 0, dst_stride = 0;
    int32_t r1_off = 0, r1_stride = 0, r2_off = 0, r2_stride = 0;
    float c1 = 1.f, c2 = 1.f, slope = 0.2f;
    int32_t pad[3] = {};
};
static_assert(sizeof(GConvParams) == 64, "one Params entry");

// The module and geometry of variant v (kConvVariants of them) of shape s on dev's family.
bool gconv_dispatch(const Device &dev, const GConvShape &s, int v, Dispatch *d, std::string *why);
// Its weights for the family: w [9][cin][cout] fp16 bits, padded to the shape.
std::vector<uint16_t> gconv_pack(Family f, const uint16_t *w, int cin, const GConvShape &s, int cout);

}  // namespace ajivk

#endif
