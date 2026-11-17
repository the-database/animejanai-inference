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
 * RIFE as the other backends run it (padded to 64, centred, bilinear BT.709 pre / post, the scene
 * detection's readback). Not yet here (the chain is passed through, or the step left out, with the
 * reason logged): the pre-RIFE resize hoist (RIFE-first interpolates at the source size), frame
 * differences, models without Vulkan kernels (there is no DirectML to fall back to on this path).
 * Temporal models as the other backends run them (aji_ingest + aji_infer_seq).
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
#include "vk/rife.h"
#include "vk/tfix.h"

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
    // RIFE (aji.h's aji_rife_factor ... aji_infer_rife_with_scene): synchronous like the other
    // backends' (the scene readback), the frames a, b, out Vulkan frames at the RIFE size.
    int rife_factor(int *num, int *den) const;
    int rife_before_upscale() const { return rife_.on && rife_.before ? 1 : 0; }
    int rife_enhanced_scene() const { return rife_.on && rife_.enhanced ? 1 : 0; }
    int infer_rife(const aji_frame *a, const aji_frame *b, double t, const aji_frame *out, int scene);
    // Temporal models (aji.h's aji_temporal_radius ... aji_temporal_reset), as aji_dml.cpp's: one
    // ring of frames per temporal model at its input size, the scene-cut metric per frame, the
    // window gathered (padded to 32) into the model's input, its frame cache where it has one.
    int temporal_radius() const { return active_ ? temporal_r_ : 0; }
    int ingest(const aji_frame *in, uint64_t seq);
    int infer_seq(uint64_t seq, const aji_frame *out);
    void temporal_reset();

private:
    VkVideo() = default;
    void logf(int level, const char *fmt, ...);
    void fail(const char *fmt, ...);
    void finalize_log();
    void teardown();

    struct Step {
        std::unique_ptr<ImgNet> net;      // a model, or
        std::unique_ptr<RgbResize> rs;    // a resize before upscale (made once the chain buffers are), or
        std::unique_ptr<TemporalNet> tnet;  // a temporal model (stages_[stage])
        int stage = -1;
        int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    };

    std::unique_ptr<Device> dev_;
    std::string conf_path_, model_dir_, rife_dir_;
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
    struct RifeState {
        bool on = false, before = false, enhanced = false;
        int w = 0, h = 0, pw = 0, ph = 0, pad_l = 0, pad_t = 0, num = 1, den = 1, fmt = 0, range = 0;
        double threshold = 0.15;
        std::unique_ptr<Rife> net;
        std::unique_ptr<FrameIO> io[2];  // frames a (and post) and b, for fmt / range
        Buffer part, sum, read;          // the scene detection's partials, sum and readback
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet sets[3] = {};  // scene partial, scene final, the constant channels
        VkPipeline pipes[3] = {};
        Buffer params;
        uint32_t groups_x = 0, groups_y = 0;
    } rife_;
    // a temporal model's stage: its ring (frames at the step's input size), the cut metrics
    struct Stage {
        int t = 1, r = 0;
        size_t step = 0;
        int w = 0, h = 0, pw = 0, ph = 0, ow = 0, oh = 0, pow = 0, poh = 0;
        float threshold = 0.0f;
        int nslots = 0, part_groups = 0;
        bool cache = false;               // the model's frame cache (gen / cgen as aji_dml.cpp's)
        std::vector<int64_t> seq;         // per slot, -1 = empty
        std::vector<int> format, matrix, range;
        std::vector<uint64_t> gen, cgen;
        Buffer ring, diff, part;
        VkDescriptorSet sets[3] = {};     // diff partial, diff final, gather
    };
    std::vector<Stage> stages_;
    int temporal_r_ = 0;
    uint64_t gen_next_ = 1;
    Buffer tparams_;
    VkDescriptorPool tpool_ = VK_NULL_HANDLE;
    VkPipeline tpipes_[3] = {};
    std::unique_ptr<FrameIO> io_pre_;  // the ingest's pre, for its input format
    bool temporal_setup(std::string *why);
    void temporal_teardown();
    int tslot(size_t lv, int64_t seq) const;
    void set_param(VkCommandBuffer cb, uint32_t entry, const void *ka) const;
    void rec_steps(VkCommandBuffer cb, size_t from, size_t to) const;
    void rec_store(VkCommandBuffer cb, size_t lv, int64_t seq, int format, int matrix, int range);
    void rec_stage(VkCommandBuffer cb, size_t lv, int64_t center, int cslot);
    void rec_ensure(VkCommandBuffer cb, size_t lv, int64_t p);
    bool rife_setup(const AjiChainConf &chain, int w, int h, double fps);
    bool rife_formats(int fmt, int range, std::string *why);
    void rife_teardown();
    // tickets
    VkSemaphore tl_ = VK_NULL_HANDLE;
    uint64_t ticket_ = 0;
    std::vector<std::string> log_info_, log_steps_;
    std::string log_, error_;
};

}  // namespace ajivk

#endif
