/*
 * native/sst_kernels.h — the kernels SST (Scalable SR Transformer with Rank-factorized Implicit
 * neural Bias, arXiv 2603.06738) needs beyond tfm.h:
 *
 *   - tribattn: window attention for large windows (any ws a multiple of 16, up to 96 and beyond),
 *     flash style: a CTA per (window, head, 128 queries) streams 64-key blocks through shared
 *     memory (cp.async, double buffered) with an online softmax in the log2 domain. The scores are
 *     [q | qpos] . [k | kpos]: the RIB factors are per-head tables over the window's positions,
 *     concatenated to q and k as extra dims, so the position bias costs a few more MMA k-steps
 *     and no bias matrix. Windows past the image's bottom / right edge read the reflected pixels
 *     (SST reflect-pads each layer's input to its window multiple); their queries write nothing.
 *     The output is multiplied by sigmoid(gate) (the layer's gate logits) before it is stored.
 *   - tdwres: depthwise 3x3 + bias, GELU, plus its own input (SST's ConvFFN middle).
 *   - sst_out: the 4x output plus the nearest-upsampled input, to NCHW RGB.
 */
#ifndef AJI_NATIVE_SST_KERNELS_H
#define AJI_NATIVE_SST_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

struct TRibAttn {
    // tokens [H][W][sq] fp16: q of head h at h*32, k at kq + h*32, v at vq + h*32 (each head's dims
    // padded to 32 with zeros; q pre-scaled by hd^-0.5 * log2(e))
    const void *qkv = nullptr;
    int sq = 0, kq = 0, vq = 0;
    int H = 0, W = 0, heads = 0, hd = 0;  // hd <= 32, even
    int ws = 0;                           // window side (ws*ws a multiple of 128)
    // RIB tables [heads][ws*ws][rp] fp16, rp = 16, 32 or 48 (the rank padded with zeros); qpos pre-scaled
    // by rank^-0.5 * log2(e)
    const void *qpos = nullptr, *kpos = nullptr;
    int rp = 0;
    const void *gate = nullptr;  // gate logits [H][W][sg] fp16, channel h*hd + d, or null
    int sg = 0;
    void *out = nullptr;  // [H][W][so] fp16, channel h*hd + d
    int so = 0;
};
cudaError_t tribattn(const TRibAttn &a, cudaStream_t s);

// out = in + gelu(dw3x3(in) + bias) over C channels (a multiple of 8, strides too); w: [C][9] floats.
cudaError_t tdwres(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, const float *bias, cudaStream_t s);

// out (NCHW RGB, 4W x 4H, fp16 or fp32) = o16 (NHWC fp16, stride so, at 4W x 4H) + the input in16
// (NHWC fp16, stride si, W x H) upsampled 4x nearest.
cudaError_t sst_out(const void *o16, int so, const void *in16, int si, int W, int H, void *out, bool out_f32, cudaStream_t s);

}  // namespace ajinative

#endif
