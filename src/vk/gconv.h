/*
 * vk/gconv.h — the general 3x3 conv on Vulkan (src/vk/shaders/gconv_*.comp), the counterpart of
 * libaji's CUDA gconv (native/gconv.h) for the larger networks (ESRGAN, RIFE): a channel prefix of
 * a wider NHWC buffer in, a channel slice of another out, bias, up to two scaled residuals and
 * LeakyReLU applied per element in fp32; an optional nearest 2x upsample of the input on the fly;
 * stride 1 or 2; inputs that are three NCHW planes (or their 2x pixel-unshuffle); outputs that are
 * three NCHW planes or a phase scatter (a transposed conv and depth-to-space as R plain convs).
 *
 * Shapes are compiled modules (shaders.txt lists the ones built); input channels, offsets and
 * strides are per dispatch, in a parameter entry (GConvParams) the caller places in a buffer.
 */
#ifndef AJI_VK_GCONV_H
#define AJI_VK_GCONV_H

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "vk/device.h"
#include "vk/kernels.h"

namespace ajivk {

// A conv's compiled shape: the module's defines (gconv_common.glsl).
struct GConvShape {
    int coutp = 16;  // output channels computed by one dispatch: 16..64
    int in = 0;      // 0 NHWC, 1 NCHW planes (3 channels), 2 pixel-unshuffle of NCHW planes (12 channels)
    int act = 0;     // 0 none, 2 LeakyReLU
    int nres = 0;    // residuals added
    int up2 = 0;     // input nearest-upsampled by 2
    int stride = 1;  // 1 or 2
    int out = 0;     // 0 NHWC, 3 NCHW planes (channels 0..2), 4 / 5 phase scatter by 2 / 4
};

// One dispatch's parameters (32 ints, the Params buffer entry layout). Offsets and strides in
// halfs, multiples of 8; cinp a multiple of 16, kch from gconv_dispatch.
struct GConvParams {
    int32_t H = 0, W = 0;
    int32_t src_off = 0, src_stride = 0, dst_off = 0, dst_stride = 0;
    int32_t r1_off = 0, r1_stride = 0, r2_off = 0, r2_stride = 0;
    float c1 = 1.f, c2 = 1.f, slope = 0.2f;
    int32_t cinp = 16, IH = 0, IW = 0, sy = 0, kch = 16;
    int32_t clamp = 0;  // OUT 3: clamp to [0, 1]
    int32_t pad[13] = {};
};
static_assert(sizeof(GConvParams) == 128, "one Params entry");

// The module and geometry of variant v (kConvVariants of them) of shape s on dev's family, and
// the halo chunk (*kch) for cinp input channels.
bool gconv_dispatch(const Device &dev, const GConvShape &s, int cinp, int v, Dispatch *d, int *kch, std::string *why);
// The weights of output channels [co0, co0 + coutp) for the family: w [9][cin][cout] fp16 bits,
// zero padded to cinp inputs and coutp outputs.
std::vector<uint16_t> gconv_pack(Family f, const uint16_t *w, int cin, int cinp, int cout, int co0, int coutp);
// nz output slices of coutp channels for one dispatch (GConvOp::nz), weights and bias one slice
// after another.
std::vector<uint16_t> gconv_pack_slices(Family f, const uint16_t *w, int cin, int cinp, int cout, int coutp, int nz);
std::vector<uint16_t> gconv_bias_slices(const float *b, int cout, int coutp, int nz);

// The output slice width for a conv of cout channels at W x H (stride: its tile rows): the widest
// of widths (descending, the shapes built) dividing pad16(cout) whose dispatch still has enough
// workgroups to fill a GPU (AJI_VK_SLICE_TARGET, default below), else the narrowest dividing one.
// Small low-resolution convs (RIFE's first blocks, ESRGAN on small frames) then fill the GPU.
int gconv_slice_width(int cout, int W, int H, int stride, std::initializer_list<int> widths);

// A conv dispatch as a network records it: shape, parameter entry, descriptor set (the caller's)
// and tile variants.
struct GConvOp {
    GConvShape s;
    GConvParams p;
    int entry = 0;  // its entry in the Params buffer the set binds at 7
    int nz = 1;     // output slices (dispatch z): channels z*coutp.. of a conv nz*coutp wide
    VkDescriptorSet set = VK_NULL_HANDLE;
    Dispatch disp[kConvVariants];
    VkPipeline pipe[kConvVariants] = {};
    int nvar = 0, variant = 0;
};
// The op's variants (disp, nvar) and halo chunk (p.kch) for p.cinp; false and *why when variant
// 0 can't run.
bool gconv_prepare(const Device &dev, GConvOp *o, std::string *why);
// Records variant v of the op (pipeline, set, entry, dispatch).
void gconv_record(const Device &dev, VkCommandBuffer cb, const GConvOp &o, int v);
// Each op's variant (AJI_VK_VARIANT, the device's stored pick, or timed once per module and size
// on one-shot submits) and its pipeline. Sets and Params must be ready.
bool gconv_choose(Device &dev, const std::vector<GConvOp *> &ops, std::string *why);

}  // namespace ajivk

#endif
