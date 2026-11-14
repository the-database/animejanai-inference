// native/img_kernels.h — layout conversions for native/imgnet (internal).
#ifndef AJI_NATIVE_IMG_KERNELS_H
#define AJI_NATIVE_IMG_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

// NCHW RGB (fp16 or fp32), w x h -> NHWC fp16 with dstride channels (zero padded); with unshuffle,
// the 2x pixel-unshuffle (channel c*4 + ry*2 + rx) at w/2 x h/2.
cudaError_t img_to_nhwc(const void *in, bool in_f32, int w, int h, bool unshuffle, void *dst, int dstride, cudaStream_t s);
// NHWC fp16 (sstride channels, the first 3 used), w x h -> NCHW RGB (fp16 or fp32).
cudaError_t img_from_nhwc(const void *src, int sstride, int w, int h, void *out, bool out_f32, cudaStream_t s);

}  // namespace ajinative

#endif
