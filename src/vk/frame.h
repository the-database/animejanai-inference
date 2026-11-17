/*
 * vk/frame.h — a 4:2:0 video frame through a model on Vulkan alone, no D3D: the raw NV12 / P010
 * planes in device buffers, pre (chroma upsample, YUV -> RGB, src/vk/shaders/frame_pre.comp), the
 * model (vk/executor.h), post (RGB -> YUV, chroma downsample, frame_post.comp) into the output
 * planes, in one command buffer. The kernels are the DirectML backend's HLSL pre/post (and so
 * the CUDA backend's) ported to GLSL, with the same weight tables (resample.h): Spline36, the
 * input's chroma siting, a left-sited 4:2:0 output; within one level of the DirectML path.
 *
 * For the callers without D3D (Linux, Android): the frames' own path into these buffers (a
 * shared Vulkan device's images, host memory, an AHardwareBuffer) is the caller's; aji_harness_vk
 * uploads raw files.
 */
#ifndef AJI_VK_FRAME_H
#define AJI_VK_FRAME_H

#include <memory>
#include <string>

#include "native/lower.h"
#include "vk/device.h"
#include "vk/executor.h"

namespace ajivk {

struct FrameFormat {
    int format = 1;  // AJI_FMT_NV12 or AJI_FMT_P010
    int matrix = 2, range = 1, siting = 1;  // aji.h's enums (BT.709, limited, left)
};

class FramePipeline {
public:
    // The model (a lowered plan) for w x h frames (even sizes). Null and *why on failure.
    static std::unique_ptr<FramePipeline> create(Device &dev, const ajinative::Plan &plan, int w, int h, const FrameFormat &f,
                                                 std::string *why);
    ~FramePipeline();
    FramePipeline(const FramePipeline &) = delete;
    FramePipeline &operator=(const FramePipeline &) = delete;

    int in_w() const { return w_; }
    int in_h() const { return h_; }
    int out_w() const { return ow_; }
    int out_h() const { return oh_; }
    // Raw planes: Y rows of *_pitch bytes (h rows), then the interleaved CbCr plane (h/2 rows),
    // each its own buffer; pitches are multiples of 4.
    const Buffer &in_y() const { return in_y_; }
    const Buffer &in_uv() const { return in_uv_; }
    const Buffer &out_y() const { return out_y_; }
    const Buffer &out_uv() const { return out_uv_; }
    uint32_t in_pitch() const { return in_pitch_; }
    uint32_t out_pitch() const { return out_pitch_; }
    // Appends pre, the model and post to cb; the caller makes the input planes visible to compute
    // shaders before and waits for compute writes to the output planes after.
    void record(VkCommandBuffer cb) const;
    const Executor &model() const { return *ex_; }

private:
    explicit FramePipeline(Device &d) : dev_(d) {}
    Device &dev_;
    int w_ = 0, h_ = 0, ow_ = 0, oh_ = 0;
    uint32_t in_pitch_ = 0, out_pitch_ = 0;
    std::unique_ptr<Executor> ex_;
    Buffer in_y_, in_uv_, out_y_, out_uv_, tmp_h_, tmp_v_, ctmp_, tables_, params_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipeline pipe_[5] = {};
    VkDescriptorSet set_[5] = {};
    uint32_t groups_[5][3] = {};
};

}  // namespace ajivk

#endif
