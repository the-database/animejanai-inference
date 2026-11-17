/*
 * vk/imgnet.h — an image network on Vulkan for one tile size, for the image path (aji_img_vk.cpp):
 * the counterpart of native/imgnet.h. create() recognises ESRGAN (RRDBNet, as traiNNer/chaiNNer
 * export it, with or without the 2x pixel-unshuffle; vk/esrgan.cpp, on the general conv) and
 * otherwise lowers the graph to the conv nets the video path runs (SPAN, SPANF3, Compact; on
 * vk/executor.h). Input and output are buffers it owns: NCHW fp16 RGB at the tile size and at
 * scale() times it.
 */
#ifndef AJI_VK_IMGNET_H
#define AJI_VK_IMGNET_H

#include <memory>
#include <string>

#include "vk/device.h"

namespace ajinative {
struct OnnxGraph;
}

namespace ajivk {

class ImgNet {
public:
    virtual ~ImgNet() = default;
    virtual const Buffer &input() const = 0;
    virtual const Buffer &output() const = 0;
    // Appends the network for a w x h input (the size it was created for) to cb; the caller makes
    // input() visible to compute shaders before and waits for compute writes to output() after.
    virtual bool record(VkCommandBuffer cb, int w, int h) const = 0;
    virtual int out_w() const = 0;
    virtual int out_h() const = 0;
    virtual size_t device_bytes() const = 0;  // all its device memory
    virtual size_t weight_bytes() const = 0;  // of which the weights
    virtual std::string describe() const = 0;
    // record() on a one-shot command buffer, submitted and waited for.
    bool run(Device &dev, int w, int h, std::string *why) const;

    // Null (and *why) when the graph is neither, or the device can't run it at w x h.
    static std::unique_ptr<ImgNet> create(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);
};

// ESRGAN at w x h, or null with *why ("not an ESRGAN graph" when it is some other network).
std::unique_ptr<ImgNet> build_esrgan(Device &dev, const ajinative::OnnxGraph &g, int w, int h, std::string *why);

}  // namespace ajivk

#endif
