// native/tfix_kernels.h — TemporalFix's resampling, warping and blending kernels (internal).
//
// Token maps are NHWC with a stride and a channel offset per buffer, fp16 or fp32 (the *_f32 flags).
// Resizes are bilinear with half-pixel centres and no antialiasing (ONNX Resize linear / PyTorch
// interpolate, align_corners=False); warps are grid_sample bilinear with zero padding.
#ifndef AJI_NATIVE_TFIX_KERNELS_H
#define AJI_NATIVE_TFIX_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

struct TxMap {
    void *p = nullptr;
    bool f32 = false;
    int stride = 0, off = 0;  // elements per pixel; first channel
};

// dst[off + c] (=, or += with add) = resize(src[off + c]) for c < C; channels 0, 1 times mx, my (flows).
cudaError_t tx_resize(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, bool add, cudaStream_t s);
// Downsizing as the model does it: the 5x5 binomial low-pass ([1 4 6 4 1]^2 / 256, zero padding) then
// the bilinear resize, in one pass (a separable 6-tap filter per output pixel); flows times mx, my.
cudaError_t tx_down(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, cudaStream_t s);
// The 5x5 binomial low-pass ([1 4 6 4 1]^2 / 256, zero padding) of C channels, same size.
cudaError_t tx_lowpass(TxMap src, int h, int w, TxMap dst, int C, cudaStream_t s);
// dst[off + c] = src[off + c] sampled at (x + f.x, y + f.y), flow fp32 [h][w][2]; valid (fp16 map, or
// p null): the sum of the in-image corner weights (grid_sample of ones).
cudaError_t tx_warp(TxMap src, int C, const float *flow, int h, int w, TxMap dst, TxMap valid, cudaStream_t s);
// dst[off + c] = plane c (3 fp16 planes of h x w), c < 3.
cudaError_t tx_planes(const void *planes, int h, int w, TxMap dst, cudaStream_t s);
// A frame into its 8-channel map: rgb from the 3 fp16 planes, channels 3..6 the bilinear upsample of feat
// (fp16 [h][w][fs], channels 0..3), 7 zero.
cudaError_t tx_frame(const void *planes, const void *feat, int fs, int h, int w, void *dst, int H, int W, cudaStream_t s);
// One support at full resolution from its coarser flow F (fp32 [h][w][2]) and logit (fp16 [h][w][ls],
// channel 0): ff = the flow resized (scaled by W/w, H/h; fp32 [H][W][2]); full (fp16 [H][W][8]) = the
// frame (fr: rgb 0..2 of an 8-channel map) warped by ff, the resized logit, the warp's validity, 0...
cudaError_t tx_full(const float *F, const void *logit, int ls, int h, int w, const void *fr, float *ff, void *full, int H, int W,
                    cudaStream_t s);
// dst[off] = v for every pixel.
cudaError_t tx_fill(TxMap dst, int h, int w, float v, cudaStream_t s);
// The confidence head's input [h][w][32] from centre (rgb 0..2, feat 3..6 of cen), the half-res warped
// support and raw logit (wh: rgb 0..2), the warped features (wf 0..3) and flow (fp32 [h][w][2]):
// [c, wr, |wr-c|, fc, fw, |fw-fc|, flow, |flow|, dt, 0...].
cudaError_t tx_conf_in(TxMap cen, TxMap wh, TxMap wf, const float *flow, float dt, void *out, int h, int w, cudaStream_t s);
// Per support at full resolution: conf = sigmoid(resize(logit, h2 x w2 -> H x W)) * valid; gate =
// sigmoid((conf - thresh) * gslope); w = conf * gate; acc[0..2] += w * warped, acc[3] += w, acc[4] += gate
// (acc fp32 [H][W][8]; full: warped 0..2, valid at vch).
cudaError_t tx_accum(TxMap logit, int h2, int w2, TxMap full, int vch, float *acc, int H, int W, float thresh, float gslope,
                     cudaStream_t s);
// out (3 fp16 planes) = c + r * ((c + acc[0..2]) / max(1 + acc[3], eps) - c), r = sigmoid((acc[4] - minsup + 0.5) * cslope).
cudaError_t tx_final(TxMap center, const float *acc, int H, int W, float minsup, float cslope, float eps, void *out, cudaStream_t s);

}  // namespace ajinative

#endif
