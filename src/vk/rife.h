/*
 * vk/rife.h — RIFE v4 frame interpolation on Vulkan for one (padded) frame size, the counterpart
 * of native/rife.h: the graph read by native/rife_plan, its convs on the general conv (vk/gconv.h)
 * and its resampling (warps, block inputs, the ensemble merge, the output blend) in
 * src/vk/shaders/rife.comp, ports of native/rife_kernels.cu with the same fp16 roundings.
 *
 * Input: the 11-channel NCHW fp16 tensor libaji feeds RIFE (frame A at planes 0-2, frame B at
 * 3-5, the timestep at plane 6; planes 7-10, the constant mesh, are not read). Output: NCHW fp16
 * RGB. Both are buffers it owns until bind_io points it at the caller's.
 */
#ifndef AJI_VK_RIFE_H
#define AJI_VK_RIFE_H

#include <memory>
#include <string>

#include "vk/device.h"

namespace ajinative {
struct OnnxGraph;
}

namespace ajivk {

class Rife {
public:
    // W, H: multiples of 64. Null and *why when the graph isn't a RIFE this reads, or the device
    // can't run it at that size. Times the conv tile variants (one-shot submits) unless stored.
    static std::unique_ptr<Rife> create(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why);
    virtual ~Rife() = default;

    virtual const Buffer &input() const = 0;   // [11][H][W]
    virtual const Buffer &output() const = 0;  // [3][H][W]
    // Read from in and write to out instead (at least the sizes above), freeing the owned ones.
    // Not while a recorded command buffer is pending.
    virtual void bind_io(VkBuffer in, VkBuffer out) = 0;
    // Appends the network to cb: the caller makes the input visible to compute shaders before
    // and waits for compute writes to the output after.
    virtual void record(VkCommandBuffer cb) const = 0;
    // record() on a one-shot command buffer, submitted and waited for.
    bool run(Device &dev, std::string *why) const;
    // One run with a timestamp after every step: GPU time per kind of step, one line each.
    virtual std::string profile(Device &dev) const = 0;
    virtual std::string describe() const = 0;
    virtual size_t device_bytes() const = 0;
};

}  // namespace ajivk

#endif
