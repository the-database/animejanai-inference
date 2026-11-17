// native/img_kernels.h — layout conversions for native/imgnet (internal).
#ifndef AJI_NATIVE_IMG_KERNELS_H
#define AJI_NATIVE_IMG_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

// NCHW RGB (fp16 or fp32), w x h -> NHWC fp16 with dstride channels (zero padded); with unshuffle,
// the 2x pixel-unshuffle (channel c*4 + ry*2 + rx) at w/2 x h/2.
cudaError_t img_to_nhwc(const void *in, bool in_f32, int w, int h, bool unshuffle, void *dst, int dstride, cudaStream_t s);
// The same with the pixel-unshuffle by `factor` (1, 2, 4: channel c*f*f + ry*f + rx) at w/f x h/f;
// dstride >= 3*f*f.
cudaError_t img_to_nhwc_by(const void *in, bool in_f32, int w, int h, int factor, void *dst, int dstride, cudaStream_t s);
// NHWC fp16 (sstride channels, the first 3 used), w x h -> NCHW RGB (fp16 or fp32), clamped to
// [0, 1] when clamp.
cudaError_t img_from_nhwc(const void *src, int sstride, int w, int h, void *out, bool out_f32, cudaStream_t s, bool clamp = false);
// NCHW fp16 RGB, w x h -> pw x ph (pw >= w, ph >= h), the added right / bottom edge reflected
// (numpy / torch "reflect": the edge pixel is not repeated).
cudaError_t img_pad_reflect(const void *in, int w, int h, void *out, int pw, int ph, cudaStream_t s);
// n values fp16 <-> fp32 (or a copy when both sides are the same type).
cudaError_t img_cast(const void *in, bool in_f32, void *out, bool out_f32, size_t n, cudaStream_t s);

}  // namespace ajinative

#endif
