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
    // With sel: one of several same-size buffers spaced fbytes apart from p, entry sel[pos] (read on
    // the device, so a CUDA graph can replay with a different choice).
    const int *sel = nullptr;
    int pos = 0;
    long long fbytes = 0;
};

// Downsizing as the model does it: the 5x5 binomial low-pass ([1 4 6 4 1]^2 / 256, zero padding) then
// the bilinear resize, in one pass (a separable 6-tap filter per output pixel); flows times mx, my.
cudaError_t tx_down(TxMap src, int ih, int iw, TxMap dst, int oh, int ow, int C, float mx, float my, cudaStream_t s);
// A frame into its 8-channel map: rgb from the 3 fp16 planes, channels 3..6 the bilinear upsample of feat
// (fp16 [h][w][fs], channels 0..3), 7 zero.
cudaError_t tx_frame(const void *planes, const void *feat, int fs, int h, int w, void *dst, int H, int W, cudaStream_t s);
// One support at full resolution from its coarser flow F (fp32 [h][w][2]) and logit (fp16 [h][w][ls],
// channel 0): ff = the flow resized (scaled by W/w, H/h; fp32 [H][W][2]); full (fp16 [H][W][8]) = the
// frame (fr: rgb 0..2 of an 8-channel map) warped by ff, the resized logit, the warp's validity, 0...
cudaError_t tx_full(const float *F, const void *logit, int ls, int h, int w, TxMap fr, float *ff, void *full, int H, int W,
                    cudaStream_t s);
// One MCBlock's input X (fp16 [h][w][xs], xs 16 or 32), every channel in one pass: centre rgb 0..2 and
// features 6..9 (cen: 8-channel map, rgb 0..2, features 3..6), the support's 3..5 and 10..13 (sup, the
// same) warped by the flow, dt at 14; with the previous stage (pf: fp32 [ph][pw][2] flow, pst: fp16
// [ph][pw][16] state 0..8): F (fp32 [h][w][2]) = the flow resized and scaled by w/pw, h/ph, X 15..23 the
// state resized, 24..25 F. Without it (the first stage) the flow is zero and 15.. stay 0. The rest is 0.
cudaError_t tx_assemble(TxMap cen, TxMap sup, const float *pf, const void *pst, int ph, int pw, float *F, void *X, int xs, int h,
                        int w, float dt, cudaStream_t s);
// An MCBlock head's outputs (O: fp16 [ih][iw][16]) resized to oh x ow in one pass: F (fp32 [oh][ow][2]) =
// (or += with add) channels 0..1; ST (fp16 [oh][ow][16]) 0..8 = channels 2..10, the rest 0.
cudaError_t tx_head_up(const void *O, int ih, int iw, float *F, void *ST, int oh, int ow, bool add, cudaStream_t s);
// The confidence head's input out [h][w][32] from the centre (cen: rgb 0..2, features 3..6), the half-res
// warped support and raw logit (wh: rgb 0..2, logit 3), the support's features (sup: 3..6) warped here by
// flow (fp32 [h][w][2]): [c, wr, |wr-c|, fc, fw, |fw-fc|, flow, |flow|, dt, 0...]; and rch (fp16 [h][w][16])
// = the raw logit in channel 0, the rest 0 (the head's residual).
cudaError_t tx_conf_in_warp(TxMap cen, TxMap wh, TxMap sup, const float *flow, float dt, void *out, void *rch, int h, int w,
                            cudaStream_t s);
// Per support at full resolution: conf = sigmoid(resize(logit, h2 x w2 -> H x W)) * valid; gate =
// sigmoid((conf - thresh) * gslope); w = conf * gate; acc[0..2] += w * warped, acc[3] += w (fp32 float4 [H][W]),
// then gate sum += gate (fp32 [H][W], after them); first: = instead of += (no clear needed). full: warped 0..2, valid at vch.
cudaError_t tx_accum(TxMap logit, int h2, int w2, TxMap full, int vch, float *acc, int H, int W, float thresh, float gslope, bool first,
                     cudaStream_t s);
// out (3 fp16 planes) = c + r * ((c + acc[0..2]) / max(1 + acc[3], eps) - c), r = sigmoid((gate sum - minsup + 0.5) * cslope).
cudaError_t tx_final(TxMap center, const float *acc, int H, int W, float minsup, float cslope, float eps, void *out, cudaStream_t s);

}  // namespace ajinative

#endif
