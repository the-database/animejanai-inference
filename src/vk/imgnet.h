/*
 * vk/imgnet.h — an image network on Vulkan for one input size: a tile on the image path
 * (aji_img_vk.cpp), the counterpart of native/imgnet.h, or a whole frame on the video path
 * (aji_vk's model steps), the counterpart of native/framenet.h. create() recognises ESRGAN
 * (RRDBNet, as traiNNer/chaiNNer export it, with or without the 2x pixel-unshuffle; vk/esrgan.cpp,
 * on the general conv), FDAT (vk/fdat.cpp, on tfm.comp) and otherwise lowers the graph to the conv nets the video path runs (SPAN,
 * SPANF3, Compact; on vk/executor.h). HAT (vk/hat.cpp) and DAT (vk/dat.cpp) are tried after FDAT. Input and output are buffers it owns, NCHW fp16 RGB at the
 * input size and at scale times it, until bind_io points it at the caller's.
 */
#ifndef AJI_VK_IMGNET_H
#define AJI_VK_IMGNET_H

#include <memory>
#include <string>

#include "vk/device.h"

namespace ajinative {
struct OnnxGraph;
struct Plan;
}

namespace ajivk {

class ImgNet {
public:
    virtual ~ImgNet() = default;
    virtual const Buffer &input() const = 0;
    virtual const Buffer &output() const = 0;
    // Read from in and write to out instead (at least the sizes above), freeing the owned ones.
    // Not while a recorded command buffer is pending.
    virtual void bind_io(VkBuffer in, VkBuffer out) = 0;
    // Appends the network for a w x h input (the size it was created for) to cb; the caller makes
    // input() visible to compute shaders before and waits for compute writes to output() after.
    virtual bool record(VkCommandBuffer cb, int w, int h) const = 0;
    virtual int out_w() const = 0;
    virtual int out_h() const = 0;
    virtual size_t device_bytes() const = 0;  // all its device memory
    virtual size_t weight_bytes() const = 0;  // of which the weights
    virtual std::string describe() const = 0;
    // GPU time per kind of step for one run, one line each ("" when the network can't say).
    virtual std::string profile(Device &dev) const
    {
        (void)dev;
        return "";
    }
    // Appends s to describe() (the video path's ", whole frame").
    virtual void suffix(const std::string &s) = 0;
    // record() on a one-shot command buffer, submitted and waited for.
    bool run(Device &dev, int w, int h, std::string *why) const;

    // Null (and *why) when the graph is neither, or the device can't run it at w x h.
    static std::unique_ptr<ImgNet> create(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
    // A lowered plan on the executor, at w x h (record() then takes any size up to it).
    static std::unique_ptr<ImgNet> lowered(Device &dev, const ajinative::Plan &plan, int w, int h, std::string *why);
    // An image model (ESRGAN) on a whole w x h video frame, as native/framenet.h's image(): built
    // first at a 128x128 probe for its buffers per pixel, and refused when weights plus buffers
    // scaled to the frame don't fit in the free device memory less 1 GiB + 10% of the device
    // (kept for the rest of the chain and the player). "not an ESRGAN graph" for other models.
    static std::unique_ptr<ImgNet> frame(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
};

// ESRGAN at w x h, or null with *why ("not an ESRGAN graph" when it is some other network).
std::unique_ptr<ImgNet> build_esrgan(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
// FDAT (vk/fdat.cpp) the same way ("not an FDAT graph").
std::unique_ptr<ImgNet> build_fdat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
// HAT (vk/hat.cpp; "not a HAT graph"). Any size: the input is reflect-padded to its windows.
std::unique_ptr<ImgNet> build_hat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
// DAT (vk/dat.cpp; "not a DAT graph"). Any size: the attention pads to its windows.
std::unique_ptr<ImgNet> build_dat(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);

}  // namespace ajivk

#endif
