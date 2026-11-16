/*
 * vk/device.h — a Vulkan device for aji_vk's kernels: the physical device (by index, or by the
 * LUID of the D3D11 adapter the caller renders on), its own VkDevice and compute queue, which
 * kernel family it runs, buffers, one-shot submits, and the compute pipelines (one shared
 * descriptor-set layout: eight storage buffers and 16 bytes of push constants).
 *
 * Kernel families (plan section 3): cm16, VK_KHR_cooperative_matrix 16x16x16 with fp16
 * accumulate, for NVIDIA Ampere+ and AMD RDNA3/4; sg16, register-blocked with the subgroup pinned
 * to 16 lanes, for Intel Xe and Mali. AJI_VK_FAMILY=cm16|sg16 overrides the choice.
 */
#ifndef AJI_VK_DEVICE_H
#define AJI_VK_DEVICE_H

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vk/vk.h"

namespace ajivk {

enum class Family { None = 0, CM16 = 1, SG16 = 2 };
const char *family_name(Family f);

struct DeviceInfo {
    uint32_t index = 0;
    std::string name;
    uint32_t vendor = 0, device_id = 0, driver_version = 0;
    uint8_t luid[8] = {};
    bool luid_valid = false;
    uint32_t subgroup = 0, min_subgroup = 0, max_subgroup = 0;
    uint32_t max_shared = 0;             // maxComputeSharedMemorySize
    double timestamp_ns = 0.0;           // timestampPeriod; 0 if the queue can't time
    uint64_t device_local_bytes = 0;     // the largest device-local heap
    bool cm16 = false, sg16 = false;     // the device can run that family
    std::string cm16_why, sg16_why;      // why not, when it can't
    Family family = Family::None;        // the one aji_vk uses (AJI_VK_FAMILY overrides)
};

// Every Vulkan 1.3 physical device. Empty (and *why) without a Vulkan loader.
std::vector<DeviceInfo> list_devices(std::string *why);

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *map = nullptr;  // host-visible buffers stay mapped
};

class Device {
public:
    // index >= 0 picks that device; otherwise luid (8 bytes) picks the matching one; otherwise
    // the first discrete GPU, else device 0.
    static std::unique_ptr<Device> create(int index, const uint8_t *luid, std::string *why);
    ~Device();
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    const DeviceInfo &info() const { return info_; }
    Family family() const { return info_.family; }
    VkDevice device() const { return dev_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return qfam_; }

    // Storage + transfer buffer, device-local, or host-visible (mapped) when host.
    bool make_buffer(Buffer *b, VkDeviceSize size, bool host, std::string *why);
    void destroy(Buffer &b);

    // One-shot work on the queue, serialized: begin() returns a recording command buffer,
    // submit_wait() runs it and waits.
    VkCommandBuffer begin();
    bool submit_wait(VkCommandBuffer cb, std::string *why);
    bool upload(const Buffer &dst, const void *data, size_t bytes, std::string *why)
    {
        return copy_staged(dst, const_cast<void *>(data), bytes, true, why);
    }
    bool download(const Buffer &src, void *data, size_t bytes, std::string *why)
    {
        return copy_staged(src, data, bytes, false, why);
    }

    VkDescriptorSetLayout set_layout() const { return set_layout_; }
    VkPipelineLayout pipeline_layout() const { return pipe_layout_; }
    static constexpr int kBindings = 8;
    static constexpr uint32_t kPushBytes = 16;
    // The compute pipeline for an embedded SPIR-V module (vk/spirv.h). local_size, when not 0,
    // is specialization constant 0; subgroup, when not 0, is required (full subgroups).
    // Cached by (module, local_size, subgroup); the Device owns it. Null and *why on failure.
    VkPipeline pipeline(const char *module, uint32_t local_size, uint32_t subgroup, std::string *why);

    // A buffer of kBindings * 16 zero bytes for descriptor slots a kernel does not read.
    VkBuffer dummy() const { return dummy_.buf; }

private:
    Device() = default;
    VkCommandBuffer begin_locked();
    bool submit_locked(VkCommandBuffer cb, std::string *why);
    bool copy_staged(const Buffer &buf, void *data, size_t bytes, bool up, std::string *why);
    DeviceInfo info_;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkPhysicalDeviceMemoryProperties mem_{};
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::mutex mu_;       // the one-shot command buffer and the staging buffer
    std::mutex pipe_mu_;  // the pipeline map
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    std::map<std::string, VkPipeline> pipelines_;
    Buffer dummy_;
    Buffer stage_;  // upload/download staging, grown on demand
};

}  // namespace ajivk

#endif
