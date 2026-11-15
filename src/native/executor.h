/*
 * native/executor.h — a lowered model (native/lower.h) instantiated for one input size: weights
 * uploaded, feature buffers allocated (reused once a tensor's last reader has run), and each
 * conv's tile variant timed and fixed. run() only launches kernels, so it can be captured into
 * a CUDA graph. Input and output are NCHW fp16 RGB, the same as a TensorRT model step.
 */
#ifndef AJI_NATIVE_EXECUTOR_H
#define AJI_NATIVE_EXECUTOR_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "native/conv.h"
#include "native/lower.h"

namespace ajinative {

class Executor {
public:
    // Fails (null, *why says why) if the device, the size or a step is not supported. Runs on
    // stream s while it times the variants; s must not be capturing.
    static std::unique_ptr<Executor> create(const Plan &plan, int in_w, int in_h, cudaStream_t s, std::string *why);
    ~Executor();
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    int out_w() const { return out_w_; }
    int out_h() const { return out_h_; }
    size_t device_bytes() const { return bytes_; }
    cudaError_t run(const void *in, void *out, cudaStream_t s) const;

private:
    Executor() = default;
    struct Step {
        bool concat = false;
        Conv3x3 op;
        ConcatConv1x1 cop;
        int src = -1, resid = -1, dst = -1, variant = 0;
        int srcs[4] = {-1, -1, -1, -1};
    };
    const void *ptr(int t, const void *in, void *out) const;
    std::vector<std::unique_ptr<Weights>> weights_;
    std::vector<Step> steps_;
    std::vector<void *> buffers_;  // owned
    std::vector<void *> tensor_;   // feature id -> buffer
    int H_ = 0, W_ = 0, out_w_ = 0, out_h_ = 0;
    size_t bytes_ = 0;
};

}  // namespace ajinative

#endif
