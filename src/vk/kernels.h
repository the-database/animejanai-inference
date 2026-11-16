/*
 * vk/kernels.h — the fused conv kernels of aji_vk (src/vk/shaders): which embedded module
 * implements an op on a kernel family, its tile variants and dispatch geometry, and the weight
 * layouts the families read. The ops are native/lower's (native/ops.h); the semantics are those
 * of the CUDA kernels in native/conv.h.
 *
 * Descriptor bindings (all storage buffers; vk/device.h's shared layout):
 *   conv3x3:   0 src, 1 resid, 2 dst, 3 weights, 4 bias, 5 PReLU slope, 6 attention weights,
 *              7 attention bias; push constants {H, W, has_resid, 0}.
 *   concat1x1: 0-3 sources, 4 dst, 5 weights, 6 bias; push constants {pixels, 0, 0, 0}.
 */
#ifndef AJI_VK_KERNELS_H
#define AJI_VK_KERNELS_H

#include <cstdint>
#include <string>
#include <vector>

#include "native/ops.h"
#include "vk/device.h"

namespace ajivk {

// A fused 3x3 conv as the kernels see it: padded channels (an NCHW input pads to 16; a
// depth-to-space 2 output pads 12 to 16) and the op values of native/ops.h.
struct ConvKey {
    int cinp = 0, coutp = 0, in = 0, act = 0, gate = 0, out = 0;
};

struct Dispatch {
    std::string module;
    uint32_t local_size = 0;  // specialization constant 0, or 0 when the shader fixes it
    uint32_t subgroup = 0;    // required subgroup size
    uint32_t shared_bytes = 0;
    int tile_h = 0;           // conv: output rows per workgroup (16 pixels wide)
    int tile_px = 0;          // concat: pixels per workgroup
};

constexpr int kConvVariants = 2;

// The dispatch for variant v (0 .. kConvVariants-1) of a conv / concat on family f, or false
// (and *why) when no embedded module implements it or the device can't run it.
bool conv_dispatch(const Device &dev, const ConvKey &k, int v, Dispatch *d, std::string *why);
bool concat_dispatch(const Device &dev, int c, int nsrc, Dispatch *d, std::string *why);
const char *conv_variant_name(Family f, int v);

// Weights [taps][cin][cout] (fp16 bits, ONNX OIHW transposed) packed for family f, zero-padded
// to cin_pad x cout_pad: cm16 row-major [taps][cin_pad][cout_pad]; sg16
// [taps][cin_pad/16][cout_pad/16][16 lanes][16 k], lane l of group g holding w[t][16kc+k][16g+l].
std::vector<uint16_t> pack_weights(Family f, const uint16_t *w, int taps, int cin, int cin_pad, int cout, int cout_pad);
// n floats as fp16 bits, zero-padded to npad.
std::vector<uint16_t> pack_vector(const float *v, int n, int npad);

}  // namespace ajivk

#endif
