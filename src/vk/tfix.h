/*
 * vk/tfix.h — TemporalFix (pifroggi/vs_temporalfix: 7 frames in, the centre one stabilised out,
 * same size) on Vulkan for one frame size, the counterpart of native/tfix.cpp: its 3x3 convs on the
 * general conv (vk/gconv.h), its depthwise / pointwise convs and resampling in
 * src/vk/shaders/tfix.comp, in native/tfix.cpp's order (the six supports one after another).
 *
 * Input: the window, fp16 NCHW [7 * 3][H][W], oldest frame first (the temporal stage's model
 * input). Output: fp16 [3][H][W]. Both are buffers it owns until bind_io points it at the caller's.
 *
 * The encoder's per-frame work can also be kept across windows, as native/vidnet.h's cache:
 * record_encode() encodes one input frame into one of cache_entries() entries, and record_cached()
 * runs a window from the entries ent[] names (one per window position), each position falling back
 * to a nearer one across a scene cut (the temporal gather's rule, from the stage's per-slot frame
 * differences, bind_diff). Which frame is in which entry is the caller's bookkeeping.
 */
#ifndef AJI_VK_TFIX_H
#define AJI_VK_TFIX_H

#include <memory>
#include <string>

#include "vk/device.h"

namespace ajinative {
struct OnnxGraph;
}

namespace ajivk {

class TemporalNet {
public:
    // W, H multiples of 8. Null and *why when the graph isn't TemporalFix ("not a TemporalFix
    // graph") or the device can't run it at that size.
    static std::unique_ptr<TemporalNet> create(Device &dev, const ajinative::OnnxGraph &g, int W, int H, std::string *why);
    virtual ~TemporalNet() = default;
    virtual int frames() const = 0;
    virtual const Buffer &input() const = 0;
    virtual const Buffer &output() const = 0;
    virtual void bind_io(VkBuffer in, VkBuffer out) = 0;
    // Appends one window to cb (the caller orders the input before and the output after).
    virtual void record(VkCommandBuffer cb) const = 0;
    int cache_entries() const { return frames(); }
    // Input frame `frame` of the window buffer into cache entry `entry`.
    virtual void record_encode(VkCommandBuffer cb, int entry, int frame) const = 0;
    // A window from the cache: ent[z] the entry of window position z; cut_slot[z] (< 0: none) the
    // diff slot of the cut between positions z and z + 1, a cut when diff * inv_area > thresh.
    virtual void record_cached(VkCommandBuffer cb, const int *ent, const int *cut_slot, float thresh, float inv_area) const = 0;
    // The stage's per-slot frame-difference sums (floats), for record_cached's cuts.
    virtual void bind_diff(VkBuffer diff) = 0;
    bool run(Device &dev, std::string *why) const;
    virtual std::string profile(Device &dev) const = 0;
    virtual std::string describe() const = 0;
    virtual size_t device_bytes() const = 0;
};

}  // namespace ajivk

#endif
