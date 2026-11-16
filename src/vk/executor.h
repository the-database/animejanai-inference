/*
 * vk/executor.h — a lowered model (native/lower.h) instantiated on a Vulkan device for one input
 * size, the counterpart of native/executor.h: weights packed for the device's kernel family,
 * feature buffers as native/layout assigns them, one descriptor set per step, and each conv's
 * tile variant timed and fixed. The model's input and output are buffers it owns (NCHW fp16
 * RGB); record() appends the steps to a command buffer.
 */
#ifndef AJI_VK_EXECUTOR_H
#define AJI_VK_EXECUTOR_H

#include <memory>
#include <string>
#include <vector>

#include "native/lower.h"
#include "vk/device.h"
#include "vk/kernels.h"

namespace ajivk {

class Executor {
public:
    // Fails (null, *why says why) if the size or a step is not supported here. Times the tile
    // variants on the device (one-shot submits) unless AJI_VK_VARIANT=<n> fixes them.
    static std::unique_ptr<Executor> create(Device &dev, const ajinative::Plan &plan, int in_w, int in_h, std::string *why);
    ~Executor();
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    int in_w() const { return W_ * u_; }
    int in_h() const { return H_ * u_; }
    int out_w() const { return out_w_; }
    int out_h() const { return out_h_; }
    size_t device_bytes() const { return bytes_; }
    size_t weight_bytes() const { return weight_bytes_; }
    // NCHW fp16 [3][in_h][in_w] / [3][out_h][out_w]; a smaller frame packs its planes at its own size.
    const Buffer &input() const { return in_; }
    const Buffer &output() const { return out_; }
    // Read the input from in and write the output to out instead (buffers the caller owns,
    // e.g. chain buffers shared with D3D12; at least the sizes above), and free the executor's
    // own. Not while a recorded command buffer is pending. input()/output() then refer to
    // in/out (handle only).
    void bind_io(VkBuffer in, VkBuffer out);

    // Appends the model for an input of w x h (<= the created size, multiples of the pixel-
    // unshuffle factor) to cb: the caller makes input() visible to compute shaders before, and
    // waits for compute writes to output() after. False for a size it can't take.
    bool record(VkCommandBuffer cb, int w, int h) const;
    // record() on a one-shot command buffer, submitted and waited for.
    bool run(int w, int h, std::string *why) const;
    // One line per step: its kernel module and the variant picked (with its time, when timed).
    const std::string &describe() const { return desc_; }

private:
    explicit Executor(Device &dev) : dev_(dev) {}
    struct Step {
        bool concat = false;
        VkPipeline pipe[kConvVariants] = {};
        Dispatch disp[kConvVariants];
        int nvar = 1, variant = 0;
        bool has_resid = false;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };
    void record_step(VkCommandBuffer cb, const Step &st, int v, int W, int H) const;
    Device &dev_;
    std::vector<Buffer> owned_;     // weights and feature buffers
    std::vector<Step> steps_;
    Buffer in_, out_;
    bool io_external_ = false;
    struct IoRef {
        VkDescriptorSet set;
        uint32_t binding;
        bool output;
    };
    std::vector<IoRef> io_refs_;  // descriptors that point at the model input / output
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    int H_ = 0, W_ = 0, out_w_ = 0, out_h_ = 0, u_ = 1;
    size_t bytes_ = 0, weight_bytes_ = 0;
    std::string desc_;
};

}  // namespace ajivk

#endif
