/*
 * native/vidnet.h — temporal (multi-frame) video models on native kernels, for libaji's temporal
 * stages (aji_trt): today TFDAT (bndl animefilm: per-neighbour flow alignment, temporal fusion,
 * the FDAT body) and TemporalFix (pifroggi's temporal stabiliser, same-size output). create()
 * recognises the model from its ONNX graph and builds it for one padded frame size; anything else
 * is refused with a reason and the caller builds a TensorRT engine.
 * run() takes the stage's model input (fp16 [3 * frames][h][w], frames oldest first) and writes
 * its output (fp16 [3][scale * h][scale * w]).
 */
#ifndef AJI_NATIVE_VIDNET_H
#define AJI_NATIVE_VIDNET_H

#include <memory>
#include <string>

#include <cuda_runtime.h>

namespace ajinative {

struct OnnxGraph;

class VidNet {
public:
    virtual ~VidNet() = default;
    virtual cudaError_t run(const void *in, void *out, cudaStream_t s) = 0;
    virtual int frames() const = 0;
    virtual int scale() const = 0;
    virtual std::string describe() const = 0;
    virtual size_t device_bytes() const = 0;

    static std::unique_ptr<VidNet> create(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why);
};

}  // namespace ajinative

#endif
