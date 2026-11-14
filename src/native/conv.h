/*
 * native/conv.h — hand-written tensor-core convolutions for small super-resolution convnets
 * (SPAN, SPANF3, SRVGG Compact and similar), the building blocks of the native executor.
 *
 * One kernel family: a 3x3 convolution (stride 1, pad 1) at the input resolution, on fp16 NHWC
 * activations, computed with mma.sync m16n8k16 and fp16 accumulation. Everything a model does
 * around the conv is folded into it:
 *   - input:  NHWC, or the model's NCHW RGB input (optionally pixel-unshuffled by 2);
 *   - epilogue: bias, then an activation (ReLU, PReLU, SiLU; LeakyReLU is a PReLU with a
 *     constant slope), or a SPAN-style gate with the block residual
 *     ((x + o) * tanh(W o + b), (x + o) * (sigmoid(o) - 0.5), or x + o);
 *   - output: NHWC, or depth-to-space by 2 or 4 into NCHW RGB, optionally adding the
 *     nearest-upsampled model input (SRVGG's global skip).
 * plus a 1x1 convolution over the virtual concatenation of up to four NHWC tensors (SPAN's
 * conv_cat), so the concat is never materialised.
 *
 * Weights are packed once, on the host, into mma B-fragment order (pack.cpp), so each lane
 * fetches a fragment pair with one 16-byte load. Each op has a few tile-shape variants;
 * pick_variant times them on the current GPU and the caller keeps the fastest.
 *
 * Requires compute capability 8.0 or newer (device_supported). Internal to the CUDA backends:
 * no ABI of its own.
 */
#ifndef AJI_NATIVE_CONV_H
#define AJI_NATIVE_CONV_H

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "native/pack.h"

namespace ajinative {

enum class Input { NHWC = 0, NCHW3 = 1, NCHW3Unshuffle2 = 2 };
enum class Act { None = 0, Relu = 1, PRelu = 2, SiLU = 3 };
enum class Gate { None = 0, Residual = 1, Tanh1x1 = 2, SigmoidHalf = 3 };
enum class Output { NHWC = 0, DepthToSpace2 = 1, DepthToSpace4 = 2 };

// Fragment-packed weights in device memory (owned; release() frees).
struct Weights {
    void *frag = nullptr;   // uint4[taps][cin_pad/16][cout_pad/16][32]
    void *bias = nullptr;   // half2[cout_pad/2]
    void *slope = nullptr;  // half2[cout_pad/2], PReLU only
    int taps = 0, cin = 0, cin_pad = 0, cout = 0, cout_pad = 0;
};

// Pack (pack.h: w is [taps][cin][cout] fp16 bit patterns, ONNX OIHW transposed to HWIO) and upload. bias has cout floats; slope (PReLU) has cout floats or is null.
cudaError_t upload(Weights &out, const uint16_t *w, const float *bias, const float *slope,
                   int taps, int cin, int cin_pad, int cout, int cout_pad);
void release(Weights &w);

// One fused 3x3 convolution. Channel counts are the logical ones: an NCHW3 input has cin 3,
// an unshuffled one 12; a DepthToSpace2 output has cout 12, DepthToSpace4 48.
struct Conv3x3 {
    Input in = Input::NHWC;
    Act act = Act::None;
    Gate gate = Gate::None;      // gates take no activation
    Output out = Output::NHWC;
    int cin = 0, cout = 0;
    const Weights *w = nullptr;
    const Weights *att = nullptr;  // Gate::Tanh1x1: the 1x1 attention conv (taps 1)
};

// H, W: the conv's resolution (an unshuffled input is 2H x 2W). src: NHWC [H][W][cin] or NCHW
// [3][H][W] ([3][2H][2W] unshuffled). resid: Gate inputs take the block input x as NHWC
// [H][W][cout]; DepthToSpace outputs take the NCHW model input [3][H][W] to add, or null.
// dst: NHWC [H][W][cout] or NCHW [3][R*H][R*W].
struct Tensors {
    const void *src = nullptr;
    const void *resid = nullptr;
    void *dst = nullptr;
};

// 1x1 convolution over the channel concatenation of nsrc NHWC tensors of c channels each
// (weights cin = nsrc * c), bias only, NHWC output with c channels.
struct ConcatConv1x1 {
    int c = 0, nsrc = 0;
    const Weights *w = nullptr;
};

bool device_supported(int device, std::string *why = nullptr);

// Whether this build has kernels for op; why explains a refusal.
bool supported(const Conv3x3 &op, std::string *why = nullptr);
bool supported(const ConcatConv1x1 &op, std::string *why = nullptr);
int variant_count(const Conv3x3 &op);
int variant_count(const ConcatConv1x1 &op);
const char *variant_name(const Conv3x3 &op, int variant);

cudaError_t launch(const Conv3x3 &op, const Tensors &t, int H, int W, int variant, cudaStream_t s);
cudaError_t launch(const ConcatConv1x1 &op, const void *const *srcs, void *dst, int npx, int variant, cudaStream_t s);

// Time every variant of op on stream s (writes t.dst) and return the fastest, or -1 on error.
// Not inside stream capture.
int pick_variant(const Conv3x3 &op, const Tensors &t, int H, int W, cudaStream_t s, float *best_ms = nullptr);

}  // namespace ajinative

#endif
