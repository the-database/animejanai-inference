/*
 * native/imgnet.h — image models on native kernels, for the tiled image path (aji_img_trt).
 *
 * create() recognises a model from its ONNX graph (ESRGAN / RRDBNet, FDAT, HAT, DAT, 1x3xHxW, with
 * or without the 2x pixel-unshuffle; then the conv nets the video path lowers, native/lower.h:
 * SPAN, SPANF3, Compact, on native/executor.h) and builds it for tiles up to max_w x max_h; anything
 * else is refused with a reason and the caller builds a TensorRT engine as before. run() takes one tile,
 * NCHW RGB (fp16 or fp32) in [0, 1], and writes NCHW RGB at scale() times its size.
 *
 * Body and tail: create() with tail_rows > 0 asks a model to split its run in two, the body (everything
 * at the input's size) over a whole tile, then the upscale tail (its few convs at the output size) in
 * strips of tail_rows input rows, so the tail's buffers are a strip's, not a tile's. A model that can
 * splits (tail_rows() > 0); the others build as without it.
 */
#ifndef AJI_NATIVE_IMGNET_H
#define AJI_NATIVE_IMGNET_H

#include <memory>
#include <string>

#include <cuda_runtime.h>

namespace ajinative {

struct OnnxGraph;

class ImgNet {
public:
    virtual ~ImgNet() = default;
    // w, h: this tile's size (<= the max given to create; even for unshuffle models).
    virtual cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) = 0;
    virtual int scale() const = 0;
    virtual std::string describe() const = 0;
    virtual size_t device_bytes() const = 0;
    // Of device_bytes: the weights; the rest is the buffers sized for the tile (they scale with its area).
    virtual size_t weight_bytes() const = 0;
    // Whether a w x h tile is accepted (size constraints of the model).
    virtual bool accepts(int w, int h) const = 0;

    // Split models: the input rows of a tail() strip (a multiple of the model's input granularity), else 0.
    virtual int tail_rows() const { return 0; }
    // The body on a w x h tile (as run() takes it), kept for tail().
    virtual cudaError_t body(const void *in, bool in_f32, int w, int h, cudaStream_t s)
    {
        (void)in, (void)in_f32, (void)w, (void)h, (void)s;
        return cudaErrorNotSupported;
    }
    // The tail on input rows [r0, r1) of the last body() (r1 - r0 <= tail_rows(), both on the input
    // granularity): output rows [r0, r1) x scale(), NCHW, 3 planes of w x scale() by (r1 - r0) x scale().
    virtual cudaError_t tail(int r0, int r1, void *out, bool out_f32, cudaStream_t s)
    {
        (void)r0, (void)r1, (void)out, (void)out_f32, (void)s;
        return cudaErrorNotSupported;
    }

    static std::unique_ptr<ImgNet> create(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why, int tail_rows = 0);
};

}  // namespace ajinative

#endif
