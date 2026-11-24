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
// One support at full resolution from its coarser flow F (fp32 [h][w][2]) and logit (fp16 [h][w][ls],
// channel 0): the flow resized (scaled by W/w, H/h), the logit resized, and the frame (fr: rgb 0..2 of an
// 8-channel map) warped by the flow; downsized in the same pass to the half resolution oh x ow the
// confidence head reads (nothing at full resolution is stored): fh (fp32 [oh][ow][2]) = the flow (scaled by
// ow/W, oh/H), wh (fp16 [oh][ow][8], 0..3) = warped rgb and logit.
cudaError_t tx_full_down(const float *F, const void *logit, int ls, int h, int w, TxMap fr, float *fh, void *wh, int H, int W, int oh, int ow,
                         cudaStream_t s);
// The confidence head's input out [h][w][32] from the centre (cen: rgb 0..2, features 3..6), the half-res
// warped support and raw logit (wh: rgb 0..2, logit 3), the support's features (sup: 3..6) warped here by
// flow (fp32 [h][w][2]): [c, wr, |wr-c|, fc, fw, |fw-fc|, flow, |flow|, dt, 0...]; and rch (fp16 [h][w][16])
// = the raw logit in channel 0, the rest 0 (the head's residual).
cudaError_t tx_conf_in_warp(TxMap cen, TxMap wh, TxMap sup, const float *flow, float dt, void *out, void *rch, int h, int w,
                            cudaStream_t s);
// The last stage's tx_head_up: F += the flow resized; lg (fp16, every other element) = the raw logit resized.
cudaError_t tx_head_up_last(const void *O, int ih, int iw, float *F, void *lg, int oh, int ow, cudaStream_t s);
// dst[2 p + 1] = src[ss p] (fp16), p < n.
cudaError_t tx_pick(const void *src, int ss, void *dst, int n, cudaStream_t s);
struct TxBlend {
    const float *pf = nullptr;  // 6 supports' F, fp32 [6][h][w][2]
    const void *pl = nullptr;   // 6 supports' (raw logit, confidence logit), fp16 [6][h][w][2]
    TxMap fr[6], center;        // the supports' and the centre's 8-channel frame maps
    float thresh = 0, gslope = 0, minsup = 0, cslope = 0, eps = 0;
};
// The blend, per pixel over the 6 supports in order: each support warped as tx_full_down does (from pf and
// the raw logit), conf = sigmoid(resize(confidence logit)) * valid; gate = sigmoid((conf - thresh) * gslope);
// w = conf * gate; sums of w * warped, w and gate. out (3 fp16 planes) = c + r * ((c + sum w*warped) /
// max(1 + sum w, eps) - c), r = sigmoid((sum gate - minsup + 0.5) * cslope). Nothing is accumulated in memory.
cudaError_t tx_blend(const TxBlend &b, int h, int w, int H, int W, void *out, cudaStream_t s);
}  // namespace ajinative

#endif
