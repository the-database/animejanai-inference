// native/tfdat_kernels.h — TFDAT's alignment and fusion kernels (internal).
//
// Flows are fp32 [h][w][2] (x, then y, in pixels). Upsampling a flow is ONNX Resize (linear,
// half_pixel, x2) times 2. Warps sample bilinearly at (x + fx, y + fy) with each corner index
// clamped to the image (the export's GatherElements warp). Feature maps are NHWC fp16.
#ifndef AJI_NATIVE_TFDAT_KERNELS_H
#define AJI_NATIVE_TFDAT_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

// F[t] = (U ? U[t] : 0) + D[t][0..1] (D: fp16, stride ds), M pixels.
cudaError_t tf_flow_add(const float *U, const void *D, int ds, float *F, int M, cudaStream_t s);
// At H x W = 2h x 2w: U = 2 * up(F); dst[0:C] = warp(feat, U) (feat: C channels, stride fs, C % 8 == 0);
// dst[uoff, uoff + 1] = U as fp16, dst[uoff + 2, ds) = 0.
cudaError_t tf_up_warp(const float *F, int h, int w, const void *feat, int fs, int C, void *dst, int ds, int uoff, float *U,
                       cudaStream_t s);
// Full resolution H x W (a multiple of 2U): flow 2 * up(F) (F at H/2 x W/2) warps the neighbour
// frame nb (3 fp16 planes); writes the warped frame pixel-unshuffled by U (1, 2 or 4) at H/U x W/U
// into wus (stride UC >= 3U^2, channel c*U*U + dy*U + dx, the rest zero), and conf (stride 16):
// channel 1 the U x U average of mean_c |warped - ctr|, 2.. zero.
cudaError_t tf_frame_warp(const float *F, const void *nb, const void *ctr, int H, int W, int U, void *wus, int UC, void *conf,
                          cudaStream_t s);
// pixel_unshuffle(U) (U 1, 2 or 4) of the 3 fp16 planes of an H x W frame into NHWC fp16 at H/U x W/U,
// stride UC >= 3U^2 (the rest zero).
cudaError_t tf_unshuffle(const void *planes, int H, int W, int U, void *dst, int UC, cudaStream_t s);
// conf[t][0] = scale * sum_c a[t][c] b[t][c] (a, b: stride st), M tokens. (tf_frame_warp writes the rest.)
cudaError_t tf_sim(const void *a, const void *b, int st, int C, float scale, void *conf, int M, cudaStream_t s);
// out[t][c] = x[t][c] * sigmoid(g[t][0]) for c < C, zero through the padding to 16 (x, out: stride st; g: stride gs).
cudaError_t tf_gate(const void *x, const void *g, int gs, void *out, int st, int C, int M, cudaStream_t s);
// out[t][c] = x[t][c] * cm[c] for c < C, zero through the padding to 16 (stride st).
cudaError_t tf_colscale(const void *x, const float *cm, void *out, int st, int C, int M, cudaStream_t s);
// The 3 fp16 planes of an H x W frame bilinearly resized xS (Resize linear, half_pixel; S = 1 copies)
// into NHWC fp16 with stride 16 at SH x SW (channels 3..15 zero).
cudaError_t tf_resize(const void *planes, int H, int W, int S, void *dst, cudaStream_t s);

}  // namespace ajinative

#endif
