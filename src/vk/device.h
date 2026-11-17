/*
 * vk/device.h — a Vulkan device for aji_vk's kernels: the physical device (by index, or by the
 * LUID of the D3D11 adapter the caller renders on), its own VkDevice and compute queue, which
 * kernel family it runs, buffers, one-shot submits, and the compute pipelines (one shared
 * descriptor-set layout: eight storage buffers and 16 bytes of push constants).
 *
 * Kernel families (plan section 3): cm16, VK_KHR_cooperative_matrix 16x16x16 with fp16
 * accumulate, for NVIDIA Ampere+ and AMD RDNA3/4; sg16, register-blocked with the subgroup pinned
 * to 16 lanes, for Intel Xe and Mali; plain, sg16's blocking with no subgroup operations at all,
 * for the rest (Adreno's 64/128-lane waves). AJI_VK_FAMILY=cm16|sg16|plain overrides the choice.
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

enum class Family { None = 0, CM16 = 1, SG16 = 2, PLAIN = 3 };
const char *family_name(Family f);

struct DeviceInfo {
    uint32_t index = 0;
    std::string name;
    uint32_t vendor = 0, device_id = 0, driver_version = 0;
    uint8_t luid[8] = {};
    bool luid_valid = false;
    uint32_t subgroup = 0, min_subgroup = 0, max_subgroup = 0;
    uint32_t max_shared = 0;             // maxComputeSharedMemorySize
    uint32_t max_invocations = 0;        // maxComputeWorkGroupInvocations
    uint32_t max_groups[2] = {0, 0};     // maxComputeWorkGroupCount x, y
    double timestamp_ns = 0.0;           // timestampPeriod; 0 if the queue can't time
    uint64_t device_local_bytes = 0;     // the largest device-local heap
    bool cm16 = false, sg16 = false, plain = false;  // the device can run that family
    std::string cm16_why, sg16_why, plain_why;       // why not, when it can't
    Family family = Family::None;        // the one aji_vk uses (AJI_VK_FAMILY overrides)
};

// Every Vulkan 1.3 physical device. Empty (and *why) without a Vulkan loader.
std::vector<DeviceInfo> list_devices(std::string *why);

// What a device offers the kernel families, as text: API and driver, subgroup sizes and
// operations, fp16 and 16-bit storage, every cooperative matrix shape, compute limits, memory
// heaps, timestamps, and the extensions that matter here. For first contact with a new GPU
// (a phone over adb) before any family is written for it. Empty (and *why) if index is out of
// range or there is no loader.
std::string capability_report(uint32_t index, std::string *why);

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

    // Caches that make a later configure fast, kept in dir (e.g. the model directory) as
    // aji-vk-<vendor>-<device>-<driver>.vkcache (the driver's VkPipelineCache, which it
    // validates itself) and .variants (the tile variant timed fastest per kernel and size, one
    // "<key> <variant>" line each). set_cache_dir loads them; save_caches writes what changed.
    // AJI_VK_RETUNE=1 ignores the stored variants (and times again).
    void set_cache_dir(const std::string &dir);
    void save_caches();
    // The stored variant for key, or -1.
    int cached_variant(const std::string &key);
    void store_variant(const std::string &key, int variant);

    // Frame work on the queue without a CPU wait: frame_begin returns a recording command
    // buffer from a small ring (waiting, on the CPU, only if that buffer's previous submit is
    // still running; null if it still is after 10 s), frame_submit submits it waiting for
    // timeline value wait_value of sem and signaling signal_value (0: none).
    VkCommandBuffer frame_begin(int *slot);
    bool frame_submit(int slot, VkSemaphore sem, uint64_t wait_value, uint64_t signal_value, std::string *why);

#ifdef _WIN32
    // D3D12 interop (aji_vk on Windows runs inside aji_dml's D3D12 frame pipeline): import a
    // shared D3D12 buffer (an NT handle from CreateSharedHandle; the caller closes it) as a
    // storage buffer, and a shared D3D12 fence as a timeline semaphore. False and *why when
    // the driver can't.
    bool interop() const { return interop_; }
    bool import_d3d12_buffer(HANDLE h, VkDeviceSize size, Buffer *out, std::string *why);
    bool import_d3d12_fence(HANDLE h, VkSemaphore *out, std::string *why);
#endif

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
    std::mutex cache_mu_;  // the variant table and the cache directory
    std::string cache_dir_;
    std::map<std::string, int> variants_;
    bool variants_dirty_ = false, pipelines_dirty_ = false;
    std::string cache_base() const;
    static constexpr int kFrameSlots = 8;
    struct FrameSlot {
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkSemaphore sem = VK_NULL_HANDLE;  // what its last submit signaled, and the value
        uint64_t value = 0;
        VkFence fence = VK_NULL_HANDLE;    // when it signaled nothing
        bool fenced = false;
    };
    VkCommandPool frame_pool_ = VK_NULL_HANDLE;
    FrameSlot frames_[kFrameSlots];
    int next_frame_ = 0;
    std::mutex frame_mu_;
    bool interop_ = false;
    Buffer dummy_;
    Buffer stage_;  // upload/download staging, grown on demand
};

}  // namespace ajivk

#endif
