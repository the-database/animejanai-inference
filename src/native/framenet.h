/*
 * native/framenet.h — a single-frame model on native kernels for the video path (aji_trt's
 * Step::NATIVE): built for one input size, NCHW fp16 RGB in and out, like a TensorRT model step.
 * run() only launches kernels, so it can be captured into a CUDA graph.
 *
 * lowered(): a conv net the lowering takes (native/lower.h: SPAN, SPANF3, Compact), on the executor.
 * image(): an image model (native/imgnet.h: ESRGAN, FDAT, HAT, DAT) on whole frames. Those models
 * size their buffers for the largest input (HAT and DAT ~10 KB per input pixel), so image() first
 * builds the model at a small probe size, scales its buffers to the frame, and refuses the frame
 * when that does not fit in the device's free memory with a margin (1 GiB + 10% of the device)
 * kept for the rest of the chain and the player. Frames the model's size rule does not take (HAT:
 * multiples of 16) are reflect-padded at the right and bottom, as the model's own graph pads, and
 * the output cropped back.
 */
#ifndef AJI_NATIVE_FRAMENET_H
#define AJI_NATIVE_FRAMENET_H

#include <memory>
#include <string>

#include <cuda_runtime.h>

namespace ajinative {

struct OnnxGraph;
struct Plan;

class FrameNet {
public:
    virtual ~FrameNet() = default;
    virtual cudaError_t run(const void *in, void *out, cudaStream_t s) = 0;
    virtual int out_w() const = 0;
    virtual int out_h() const = 0;
    virtual std::string describe() const = 0;
    virtual size_t device_bytes() const = 0;

    // Null with *why on refusal; the caller then uses TensorRT. s must not be capturing.
    static std::unique_ptr<FrameNet> lowered(const Plan &plan, int w, int h, cudaStream_t s, std::string *why);
    // *size_only (if given) tells whether the refusal is about this frame size or the free memory
    // (another size may work) rather than the model.
    static std::unique_ptr<FrameNet> image(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why,
                                           bool *size_only = nullptr);
};

}  // namespace ajinative

#endif
