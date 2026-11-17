/*
 * vk/frame.h — 4:2:0 video frames into and out of the Vulkan native kernels, no D3D: the raw NV12
 * / P010 planes in device buffers, pre (chroma upsample, YUV -> RGB, src/vk/shaders/frame_pre.comp)
 * into a model's input (fp16 NCHW RGB), post (RGB -> YUV, chroma downsample, frame_post.comp) from
 * its output into the output planes. The kernels are the DirectML backend's HLSL pre/post (and so
 * the CUDA backend's) ported to GLSL, with the same weight tables (resample.h): Spline36, the
 * input's chroma siting, a left-sited 4:2:0 output; within one level of the DirectML path.
 *
 * FrameIO is the two ends around any RGB buffers (vk/video.h runs a chain between them);
 * FramePipeline is FrameIO around one lowered model, for aji_harness_vk.
 *
 * For the callers without D3D (Linux, Android): the frames' own path into these buffers (a
 * shared Vulkan device's images: vk/video.h; host memory, an AHardwareBuffer) is the caller's.
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
    bool operator==(const FrameFormat &o) const
    {
        return format == o.format && matrix == o.matrix && range == o.range && siting == o.siting;
    }
};

class FrameIO {
public:
    // w x h input planes (format in) to rgb_in (fp16 [3][h][w]), rgb_out (fp16 [3][oh][ow]) to
    // ow x oh output planes (format out: its matrix and range, a left-sited 4:2:0). Even sizes.
    static std::unique_ptr<FrameIO> create(Device &dev, int w, int h, int ow, int oh, const FrameFormat &in, const FrameFormat &out,
                                           VkBuffer rgb_in, VkBuffer rgb_out, std::string *why);
    ~FrameIO();
    FrameIO(const FrameIO &) = delete;
    FrameIO &operator=(const FrameIO &) = delete;

    // Raw planes: Y rows of *_pitch bytes (h rows), then the interleaved CbCr plane (h/2 rows),
    // each its own buffer; pitches are multiples of 4.
    const Buffer &in_y() const { return in_y_; }
    const Buffer &in_uv() const { return in_uv_; }
    const Buffer &out_y() const { return out_y_; }
    const Buffer &out_uv() const { return out_uv_; }
    uint32_t in_pitch() const { return in_pitch_; }
    uint32_t out_pitch() const { return out_pitch_; }
    const FrameFormat &in_format() const { return in_; }
    const FrameFormat &out_format() const { return out_; }
    // Pre into rgb_in; post from rgb_out. Each ends with a compute barrier; the caller makes the
    // input planes visible before pre and waits for compute writes to the output planes after post.
    void record_pre(VkCommandBuffer cb) const;
    void record_post(VkCommandBuffer cb) const;

private:
    explicit FrameIO(Device &d) : dev_(d) {}
    void run(VkCommandBuffer cb, int k) const;
    Device &dev_;
    FrameFormat in_, out_;
    uint32_t in_pitch_ = 0, out_pitch_ = 0;
    Buffer in_y_, in_uv_, out_y_, out_uv_, tmp_h_, tmp_v_, ctmp_, tables_, params_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipeline pipe_[5] = {};
    VkDescriptorSet set_[5] = {};
    uint32_t groups_[5][3] = {};
};

class FramePipeline {
public:
    // The model (a lowered plan) for w x h frames (even sizes). Null and *why on failure.
    static std::unique_ptr<FramePipeline> create(Device &dev, const ajinative::Plan &plan, int w, int h, const FrameFormat &f,
                                                 std::string *why);

    int in_w() const { return w_; }
    int in_h() const { return h_; }
    int out_w() const { return ow_; }
    int out_h() const { return oh_; }
    const Buffer &in_y() const { return io_->in_y(); }
    const Buffer &in_uv() const { return io_->in_uv(); }
    const Buffer &out_y() const { return io_->out_y(); }
    const Buffer &out_uv() const { return io_->out_uv(); }
    uint32_t in_pitch() const { return io_->in_pitch(); }
    uint32_t out_pitch() const { return io_->out_pitch(); }
    // Appends pre, the model and post to cb; the caller makes the input planes visible to compute
    // shaders before and waits for compute writes to the output planes after.
    void record(VkCommandBuffer cb) const;
    const Executor &model() const { return *ex_; }

private:
    int w_ = 0, h_ = 0, ow_ = 0, oh_ = 0;
    std::unique_ptr<Executor> ex_;
    std::unique_ptr<FrameIO> io_;
};

}  // namespace ajivk

#endif
