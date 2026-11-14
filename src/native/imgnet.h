/*
 * native/imgnet.h — image models on native kernels, for the tiled image path (aji_img_trt).
 *
 * create() recognises a model from its ONNX graph (today: ESRGAN / RRDBNet and FDAT, 1x3xHxW, with
 * or without the 2x pixel-unshuffle) and builds it for tiles up to max_w x max_h; anything else is
 * refused with a reason and the caller builds a TensorRT engine as before. run() takes one tile,
 * NCHW RGB (fp16 or fp32) in [0, 1], and writes NCHW RGB at scale() times its size.
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
    // Whether a w x h tile is accepted (size constraints of the model).
    virtual bool accepts(int w, int h) const = 0;

    static std::unique_ptr<ImgNet> create(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
};

}  // namespace ajinative

#endif
