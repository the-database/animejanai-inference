/*
 * aji_vk_frames.cpp — libaji_vk's video ABI where there is no D3D (Linux, Android): Vulkan
 * frames only (aji_create_vk), each export forwarding to vk/video.h. On Windows aji_vk.dll's
 * exports are aji_dml.cpp's, which forward a Vulkan-frames context to the same class.
 */
#include <memory>
#include <string>

#include "aji.h"
#include "vk/video.h"

struct aji_ctx {
    std::unique_ptr<ajivk::VkVideo> v;
};

#define AJI_VKV(ret)    \
    if (!c || !c->v)    \
        return ret;

extern "C" AJI_EXPORT aji_ctx *aji_create(const aji_create_params *params)
{
    if (params && params->log)
        params->log(params->log_opaque, 0,
                    "this aji_vk takes Vulkan frames only (aji_create_vk): no D3D11 or CUDA frames here");
    return nullptr;
}

extern "C" AJI_EXPORT aji_ctx *aji_create_vk(const aji_create_params *params, const aji_vk_device *device)
{
    if (!params || params->api_version != AJI_API_VERSION) return nullptr;
    std::string why;
    std::unique_ptr<ajivk::VkVideo> v = ajivk::VkVideo::create(params, device, &why);
    if (!v) {
        if (params->log) params->log(params->log_opaque, 0, ("Vulkan frames: " + why).c_str());
        return nullptr;
    }
    aji_ctx *c = new aji_ctx;
    c->v = std::move(v);
    return c;
}

extern "C" AJI_EXPORT int aji_set_slot(aji_ctx *c, int slot)
{
    AJI_VKV(AJI_ERR)
    return c->v->set_slot(slot);
}

extern "C" AJI_EXPORT int aji_configure(aji_ctx *c, int w, int h, double fps, int *out_w, int *out_h)
{
    AJI_VKV(AJI_ERR)
    return c->v->configure(w, h, fps, out_w, out_h);
}

extern "C" AJI_EXPORT int aji_infer(aji_ctx *c, const aji_frame *in, const aji_frame *out, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->infer(in, out);
}

extern "C" AJI_EXPORT uint64_t aji_flush(aji_ctx *c, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(0)
    return c->v->flush();
}

extern "C" AJI_EXPORT int aji_done(aji_ctx *c, uint64_t ticket)
{
    AJI_VKV(AJI_ERR)
    return c->v->done(ticket);
}

extern "C" AJI_EXPORT int aji_wait(aji_ctx *c, uint64_t ticket)
{
    AJI_VKV(AJI_ERR)
    return c->v->wait(ticket);
}

extern "C" AJI_EXPORT const char *aji_current_log(aji_ctx *c)
{
    AJI_VKV("")
    return c->v->current_log();
}

extern "C" AJI_EXPORT int aji_scale_factor(aji_ctx *c)
{
    (void)c;
    return 0;  // conf mode
}

extern "C" AJI_EXPORT int aji_rife_factor(aji_ctx *c, int *num, int *den)
{
    AJI_VKV(0)
    return c->v->rife_factor(num, den);
}

extern "C" AJI_EXPORT int aji_rife_before_upscale(aji_ctx *c)
{
    AJI_VKV(0)
    return c->v->rife_before_upscale();
}

extern "C" AJI_EXPORT int aji_rife_enhanced_scene(aji_ctx *c)
{
    AJI_VKV(0)
    return c->v->rife_enhanced_scene();
}

extern "C" AJI_EXPORT int aji_rife_scene_supported(aji_ctx *c)
{
    return c && c->v ? 1 : 0;
}

extern "C" AJI_EXPORT int aji_pre_resize(aji_ctx *c, int *work_w, int *work_h)
{
    AJI_VKV(0)
    return c->v->pre_resize(work_w, work_h);
}

extern "C" AJI_EXPORT int aji_resize(aji_ctx *c, const aji_frame *in, const aji_frame *out, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->resize(in, out);
}

extern "C" AJI_EXPORT int aji_poll(aji_ctx *c)
{
    (void)c;
    return 0;  // no background builds
}

extern "C" AJI_EXPORT int aji_infer_rife(aji_ctx *c, const aji_frame *a, const aji_frame *b, double t, const aji_frame *out, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->infer_rife(a, b, t, out, -1);
}

extern "C" AJI_EXPORT int aji_infer_rife_with_scene(aji_ctx *c, const aji_frame *a, const aji_frame *b, double t, const aji_frame *out,
                                                    void *cu_stream, int scene)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->infer_rife(a, b, t, out, scene);
}

extern "C" AJI_EXPORT int aji_temporal_radius(aji_ctx *c)
{
    AJI_VKV(0)
    return c->v->temporal_radius();
}

extern "C" AJI_EXPORT int aji_ingest(aji_ctx *c, const aji_frame *in, uint64_t seq, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->ingest(in, seq);
}

extern "C" AJI_EXPORT int aji_infer_seq(aji_ctx *c, uint64_t seq, const aji_frame *out, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->infer_seq(seq, out);
}

extern "C" AJI_EXPORT void aji_temporal_reset(aji_ctx *c)
{
    if (c && c->v) c->v->temporal_reset();
}

extern "C" AJI_EXPORT int aji_copy_frame(aji_ctx *c, const aji_frame *in, const aji_frame *out)
{
    AJI_VKV(AJI_ERR)
    return c->v->copy_frame(in, out);
}

extern "C" AJI_EXPORT int aji_frame_diff(aji_ctx *c, const aji_frame *a, const aji_frame *b, aji_frame_diff_result *out, void *cu_stream)
{
    (void)cu_stream;
    AJI_VKV(AJI_ERR)
    return c->v->frame_diff(a, b, out);
}

extern "C" AJI_EXPORT const char *aji_last_error(aji_ctx *c)
{
    AJI_VKV("no context")
    return c->v->last_error();
}

extern "C" AJI_EXPORT void aji_destroy(aji_ctx **pc)
{
    if (!pc || !*pc) return;
    delete *pc;
    *pc = nullptr;
}

extern "C" AJI_EXPORT int aji_stream_policy_version(void)
{
    return 0;
}
