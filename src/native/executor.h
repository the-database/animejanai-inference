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
    // Gate reuse can be disabled for a same-precision reference with distinct residuals.
    // use_fp8 is requested only for a full SPANF3 video plan. Device and kernel support are checked at load.
    static std::unique_ptr<Executor> create(const Plan &plan, int in_w, int in_h, cudaStream_t s, std::string *why,
                                            bool reuse_gate_residuals = true, bool use_fp8 = false);
    bool uses_fp8() const { return !fp8_owned_.empty(); }
    int packed_links() const { return packed_links_; }
    bool enable_fp16_cache_order();
    bool enable_gate_projection();
    ~Executor();
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    int in_w() const { return W_ * u_; }
    int in_h() const { return H_ * u_; }
    int out_w() const { return out_w_; }
    int out_h() const { return out_h_; }
    size_t device_bytes() const { return bytes_; }
    // Of device_bytes: the packed weights (the rest is the feature buffers, sized for the input).
    size_t weight_bytes() const { return weight_bytes_; }
    size_t feature_buffer_count() const { return buffers_.size(); }
    cudaError_t run(const void *in, void *out, cudaStream_t s) const { return run(in, out, in_w(), in_h(), s); }
    // An input up to the size created for (w <= in_w(), h <= in_h(), multiples of the model's
    // pixel-unshuffle factor): the buffers and tile variants of the created size serve it.
    cudaError_t run(const void *in, void *out, int w, int h, cudaStream_t s) const;

private:
    Executor() = default;
    struct Step {
        bool concat = false;
        bool packed_input = false, packed_output = false;
        void *fp8_weights = nullptr;
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
    int H_ = 0, W_ = 0, out_w_ = 0, out_h_ = 0, u_ = 1;
    size_t bytes_ = 0, weight_bytes_ = 0;
    std::vector<void *> fp8_owned_;
    int packed_links_ = 0;
    bool fused_projection_ = false;
};

}  // namespace ajinative

#endif
