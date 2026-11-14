/*
 * native/vidnet.h — temporal (multi-frame) video models on native kernels, for libaji's temporal
 * stages (aji_trt): today TFDAT (bndl animefilm: per-neighbour flow alignment, temporal fusion,
 * the FDAT body) and TemporalFix (pifroggi's temporal stabiliser, same-size output). create()
 * recognises the model from its ONNX graph and builds it for one padded frame size; anything else
 * is refused with a reason and the caller builds a TensorRT engine.
 * run() takes the stage's model input (fp16 [3 * frames][h][w], frames oldest first) and writes
 * its output (fp16 [3][scale * h][scale * w]).
 * Models whose per-frame work does not depend on the frame's place in the window (TemporalFix's
 * encoder) can also keep it per frame: cache_entries() > 0 slots, encode() fills one from a frame
 * (fp16 [3][h][w]), and run_cached() runs a window from the slots that sel (on the device, one per
 * window position, oldest first) names. Which frame is in which slot is the caller's bookkeeping.
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
    virtual int cache_entries() const { return 0; }
    virtual cudaError_t encode(int entry, const void *frame, cudaStream_t s)
    {
        (void)entry, (void)frame, (void)s;
        return cudaErrorNotSupported;
    }
    virtual cudaError_t run_cached(const int *sel, void *out, cudaStream_t s)
    {
        (void)sel, (void)out, (void)s;
        return cudaErrorNotSupported;
    }

    static std::unique_ptr<VidNet> create(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why);
};

}  // namespace ajinative

#endif
