/*
 * vk/video.h — the video ABI on Vulkan alone, for a context from aji_create_vk: the caller's
 * device (vk/device.h's ExternalDevice), frames as VkImages on it (aji_vk_image), the conf's
 * chains on the Vulkan native kernels between vk/frame.h's pre and post. Linux has only this;
 * on Windows aji_vk.dll's exports (aji_dml.cpp) forward a Vulkan-frames context here.
 *
 * Per frame, one submit: wait for both images' semaphores, copy the input image's planes into
 * FrameIO's buffers, pre, each model, post, copy into the output image's planes, signal both
 * images' semaphores (value + 1, AVVkFrame's rule) and the context's own timeline semaphore,
 * whose values are aji_flush's tickets.
 *
 * Not yet here (the chain is passed through, or the step left out, with the reason logged):
 * resizes before upscale, RIFE, temporal models, frame differences, models without Vulkan
 * kernels (there is no DirectML to fall back to on this path).
 */
#ifndef AJI_VK_VIDEO_H
#define AJI_VK_VIDEO_H

#include <memory>
#include <string>
#include <vector>

#include "aji.h"
#include "aji_conf.h"
#include "vk/device.h"
#include "vk/frame.h"
#include "vk/imgnet.h"

namespace ajivk {

class VkVideo {
public:
    static std::unique_ptr<VkVideo> create(const aji_create_params *p, const aji_vk_device *d, std::string *why);
    ~VkVideo();

    int set_slot(int slot);
    int configure(int w, int h, double fps, int *out_w, int *out_h);
    int infer(const aji_frame *in, const aji_frame *out);
    uint64_t flush() const { return ticket_; }
    int done(uint64_t t);
    int wait(uint64_t t);
    const char *current_log() const { return log_.c_str(); }
    const char *last_error() const { return error_.c_str(); }

private:
    VkVideo() = default;
    void logf(int level, const char *fmt, ...);
    void fail(const char *fmt, ...);
    void finalize_log();
    void teardown();

    struct Step {
        std::unique_ptr<ImgNet> net;
        int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    };

    std::unique_ptr<Device> dev_;
    std::string conf_path_, model_dir_;
    aji_log_fn log_fn_ = nullptr;
    void *log_opaque_ = nullptr;
    AjiConf conf_;
    int slot_ = 1;
    // the configured chain
    bool active_ = false;
    int in_w_ = 0, in_h_ = 0, out_w_ = 0, out_h_ = 0;
    std::vector<Step> steps_;
    Buffer chain_[2];  // fp16 NCHW RGB, ping-pong between the steps
    std::unique_ptr<FrameIO> io_;  // for the last frames' formats
    // tickets
    VkSemaphore tl_ = VK_NULL_HANDLE;
    uint64_t ticket_ = 0;
    std::vector<std::string> log_info_, log_steps_;
    std::string log_, error_;
};

}  // namespace ajivk

#endif
