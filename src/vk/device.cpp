// vk/device.cpp — see device.h.
#include "vk/device.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "core/fileio.h"
#include "vk/spirv.h"

namespace ajivk {
namespace {

constexpr uint32_t kVendorNvidia = 0x10de, kVendorAmd = 0x1002, kVendorIntel = 0x8086;

bool has_ext(VkPhysicalDevice p, const char *name)
{
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(p, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(p, nullptr, &n, e.data());
    for (const auto &x : e)
        if (!std::strcmp(x.extensionName, name)) return true;
    return false;
}

// Features both families need, and the ones each adds.
struct Features {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void chain(bool with_cm)
    {
        f2.pNext = &f11;
        f11.pNext = &f12;
        f12.pNext = &f13;
        f13.pNext = with_cm ? &cm : nullptr;
        cm.pNext = nullptr;
    }
};

DeviceInfo probe(VkPhysicalDevice p, uint32_t index)
{
    DeviceInfo d;
    d.index = index;
    VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties ssp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &idp};
    VkPhysicalDeviceSubgroupProperties sp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &ssp};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sp};
    vkGetPhysicalDeviceProperties2(p, &p2);
    const VkPhysicalDeviceProperties &pr = p2.properties;
    d.name = pr.deviceName;
    d.vendor = pr.vendorID;
    d.device_id = pr.deviceID;
    d.driver_version = pr.driverVersion;
    d.luid_valid = idp.deviceLUIDValid == VK_TRUE;
    std::memcpy(d.luid, idp.deviceLUID, 8);
    d.subgroup = sp.subgroupSize;
    d.min_subgroup = ssp.minSubgroupSize;
    d.max_subgroup = ssp.maxSubgroupSize;
    d.max_shared = pr.limits.maxComputeSharedMemorySize;
    d.max_invocations = pr.limits.maxComputeWorkGroupInvocations;
    d.max_groups[0] = pr.limits.maxComputeWorkGroupCount[0];
    d.max_groups[1] = pr.limits.maxComputeWorkGroupCount[1];
    d.timestamp_ns = pr.limits.timestampComputeAndGraphics ? pr.limits.timestampPeriod : 0.0;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(p, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mp.memoryHeaps[i].size > d.device_local_bytes)
            d.device_local_bytes = mp.memoryHeaps[i].size;

    if (pr.apiVersion < VK_API_VERSION_1_3) {
        d.cm16_why = d.sg16_why = "the driver is older than Vulkan 1.3";
        return d;
    }
    Features f;
    const bool cm_ext = has_ext(p, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
    f.chain(cm_ext);
    vkGetPhysicalDeviceFeatures2(p, &f.f2);
    // core: what every family needs; base: what the subgroup families (cm16, sg16) add
    std::string core;
    if (!f.f12.shaderFloat16) core = "no fp16 shader arithmetic";
    else if (!f.f11.storageBuffer16BitAccess) core = "no 16-bit storage buffers";
    std::string base = core;
    if (base.empty() && (!f.f13.subgroupSizeControl || !f.f13.computeFullSubgroups)) base = "no subgroup size control";
    else if (base.empty() && !(sp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) base = "no subgroup operations in compute shaders";

    // cm16: 16x16x16 fp16 x fp16 + fp16 at subgroup scope
    if (!base.empty()) d.cm16_why = base;
    else if (!cm_ext || !f.cm.cooperativeMatrix) d.cm16_why = "no VK_KHR_cooperative_matrix";
    else {
        uint32_t n = 0;
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(p, &n, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cm(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(p, &n, cm.data());
        bool shape = false;
        for (const auto &c : cm)
            shape |= c.MSize == 16 && c.NSize == 16 && c.KSize == 16 && c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                     c.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                     c.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.scope == VK_SCOPE_SUBGROUP_KHR;
        if (!shape) d.cm16_why = "no 16x16x16 fp16-accumulate cooperative matrix";
        // Intel Arc reports other shapes only; RDNA2 and older emulate the extension (plan
        // section 3: gate on the architecture). Until the device tables land, only NVIDIA and AMD.
        else if (d.vendor != kVendorNvidia && d.vendor != kVendorAmd) d.cm16_why = "cooperative matrix kernels are tuned for NVIDIA and AMD only";
        else if (d.subgroup != 32 && !(d.min_subgroup <= 32 && d.max_subgroup >= 32)) d.cm16_why = "no 32-lane subgroups";
        else if (!(ssp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) d.cm16_why = "compute shaders can't require a subgroup size";
        else d.cm16 = true;
    }
    // sg16: subgroups of exactly 16 lanes with broadcasts and shuffles (of 32-bit values: fp16
    // pairs travel packed, so no fp16 subgroup operations are needed)
    const VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if (!base.empty()) d.sg16_why = base;
    else if (d.min_subgroup > 16 || d.max_subgroup < 16) d.sg16_why = "no 16-lane subgroups";
    else if ((sp.supportedOperations & need) != need) d.sg16_why = "no subgroup ballot and shuffle";
    else if (!(ssp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) d.sg16_why = "compute shaders can't require a subgroup size";
    else d.sg16 = true;
    // plain: fp16 arithmetic and 16-bit storage only
    if (!core.empty()) d.plain_why = core;
    else d.plain = true;

    // cm16 on AMD hasn't run on an AMD GPU yet (stride-0 matrix loads, element indexing, wave32):
    // only on request, AJI_VK_FAMILY=cm16, until it has
    const bool cm16_default = d.cm16 && d.vendor == kVendorNvidia;
    d.family = cm16_default ? Family::CM16 : d.sg16 ? Family::SG16 : d.plain ? Family::PLAIN : Family::None;
    if (d.cm16 && !cm16_default && d.family == Family::None) d.cm16_why = "untested on this vendor (AJI_VK_FAMILY=cm16 enables it)";
    if (const char *e = std::getenv("AJI_VK_FAMILY")) {
        if (!std::strcmp(e, "cm16")) d.family = d.cm16 ? Family::CM16 : Family::None;
        else if (!std::strcmp(e, "sg16")) d.family = d.sg16 ? Family::SG16 : Family::None;
        else if (!std::strcmp(e, "plain")) d.family = d.plain ? Family::PLAIN : Family::None;
    }
    (void)kVendorIntel;
    return d;
}

std::vector<VkPhysicalDevice> physical_devices()
{
    std::string why;
    VkInstance inst = instance(&why);
    if (!inst) return {};
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> v(n);
    vkEnumeratePhysicalDevices(inst, &n, v.data());
    v.resize(n);
    return v;
}

std::string err(const char *what, VkResult r) { return std::string(what) + " failed: " + result_name(r); }

}  // namespace

const char *family_name(Family f)
{
    switch (f) {
    case Family::CM16: return "cm16";
    case Family::SG16: return "sg16";
    case Family::PLAIN: return "plain";
    default: return "none";
    }
}

std::vector<DeviceInfo> list_devices(std::string *why)
{
    if (!instance(why)) return {};
    std::vector<DeviceInfo> out;
    const auto pds = physical_devices();
    for (uint32_t i = 0; i < pds.size(); i++) out.push_back(probe(pds[i], i));
    return out;
}

namespace {

void addf(std::string *s, const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    *s += buf;
}

const char *component_name(VkComponentTypeKHR t)
{
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    default: return "other";
    }
}

std::string subgroup_ops(VkSubgroupFeatureFlags f)
{
    static const std::pair<VkSubgroupFeatureFlagBits, const char *> names[] = {
        {VK_SUBGROUP_FEATURE_BASIC_BIT, "basic"}, {VK_SUBGROUP_FEATURE_VOTE_BIT, "vote"},
        {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "arithmetic"}, {VK_SUBGROUP_FEATURE_BALLOT_BIT, "ballot"},
        {VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "shuffle"}, {VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT, "shuffle_relative"},
        {VK_SUBGROUP_FEATURE_CLUSTERED_BIT, "clustered"}, {VK_SUBGROUP_FEATURE_QUAD_BIT, "quad"}};
    std::string s;
    for (const auto &n : names)
        if (f & n.first) s += std::string(s.empty() ? "" : " ") + n.second;
    return s;
}

}  // namespace

std::string capability_report(uint32_t index, std::string *why)
{
    if (!instance(why)) return {};
    const auto pds = physical_devices();
    if (index >= pds.size()) {
        if (why) *why = "no Vulkan device " + std::to_string(index);
        return {};
    }
    VkPhysicalDevice p = pds[index];
    const DeviceInfo d = probe(p, index);
    std::string s;

    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties ssp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &drv};
    VkPhysicalDeviceSubgroupProperties sp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &ssp};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sp};
    vkGetPhysicalDeviceProperties2(p, &p2);
    const VkPhysicalDeviceProperties &pr = p2.properties;
    const VkPhysicalDeviceLimits &l = pr.limits;
    addf(&s, "device %u: %s, vendor %04x device %04x, type %d\n", index, pr.deviceName, pr.vendorID, pr.deviceID,
         (int)pr.deviceType);
    addf(&s, "  api %u.%u.%u, driver %s (%s), driver version %08x\n", VK_API_VERSION_MAJOR(pr.apiVersion),
         VK_API_VERSION_MINOR(pr.apiVersion), VK_API_VERSION_PATCH(pr.apiVersion), drv.driverName, drv.driverInfo,
         pr.driverVersion);
    addf(&s, "  subgroup %u, size control [%u..%u], required-size stages 0x%x, ops in stages 0x%x: %s\n", sp.subgroupSize,
         ssp.minSubgroupSize, ssp.maxSubgroupSize, ssp.requiredSubgroupSizeStages, sp.supportedStages,
         subgroup_ops(sp.supportedOperations).c_str());

    Features f;
    const bool cm_ext = has_ext(p, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
    if (pr.apiVersion >= VK_API_VERSION_1_3) {
        f.chain(cm_ext);
        vkGetPhysicalDeviceFeatures2(p, &f.f2);
        addf(&s, "  fp16 arithmetic %d, int8 arithmetic %d, 16-bit storage %d, 8-bit storage %d, fp16 subgroup ops %d, "
                 "subgroup size control %d, full subgroups %d, integer dot product %d\n",
             f.f12.shaderFloat16, f.f12.shaderInt8, f.f11.storageBuffer16BitAccess, f.f12.storageBuffer8BitAccess,
             f.f12.shaderSubgroupExtendedTypes, f.f13.subgroupSizeControl, f.f13.computeFullSubgroups,
             f.f13.shaderIntegerDotProduct);
    }
    if (cm_ext) {
        uint32_t n = 0;
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(p, &n, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cm(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(p, &n, cm.data());
        addf(&s, "  cooperative matrix (feature %d): %u shapes\n", f.cm.cooperativeMatrix, n);
        for (const auto &c : cm)
            addf(&s, "    %ux%ux%u  A %s B %s C %s result %s  scope %d%s\n", c.MSize, c.NSize, c.KSize, component_name(c.AType),
                 component_name(c.BType), component_name(c.CType), component_name(c.ResultType), (int)c.scope,
                 c.saturatingAccumulation ? " saturating" : "");
    } else {
        s += "  cooperative matrix: none\n";
    }
    addf(&s, "  compute: shared %u B, invocations %u, group size %ux%ux%u, groups %ux%ux%u, push constants %u B, "
             "storage buffer range %u B\n",
         l.maxComputeSharedMemorySize, l.maxComputeWorkGroupInvocations, l.maxComputeWorkGroupSize[0],
         l.maxComputeWorkGroupSize[1], l.maxComputeWorkGroupSize[2], l.maxComputeWorkGroupCount[0],
         l.maxComputeWorkGroupCount[1], l.maxComputeWorkGroupCount[2], l.maxPushConstantsSize, l.maxStorageBufferRange);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(p, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        addf(&s, "  heap %u: %.0f MiB%s\n", i, mp.memoryHeaps[i].size / 1048576.0,
             (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? ", device-local" : "");
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        addf(&s, "  memory type %u: heap %u, flags 0x%x\n", i, mp.memoryTypes[i].heapIndex, mp.memoryTypes[i].propertyFlags);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, qf.data());
    for (uint32_t i = 0; i < nq; i++)
        addf(&s, "  queue family %u: flags 0x%x, %u queues, timestamp bits %u\n", i, qf[i].queueFlags, qf[i].queueCount,
             qf[i].timestampValidBits);
    addf(&s, "  timestamps: compute and graphics %d, period %.3f ns\n", l.timestampComputeAndGraphics, l.timestampPeriod);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(p, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ext(ne);
    vkEnumerateDeviceExtensionProperties(p, nullptr, &ne, ext.data());
    addf(&s, "  %u device extensions:", ne);
    for (const auto &e : ext) s += std::string(" ") + e.extensionName;
    s += "\n";
    addf(&s, "  aji_vk: family %s (cm16: %s; sg16: %s; plain: %s)\n", family_name(d.family), d.cm16 ? "yes" : d.cm16_why.c_str(),
         d.sg16 ? "yes" : d.sg16_why.c_str(), d.plain ? "yes" : d.plain_why.c_str());
    return s;
}

std::unique_ptr<Device> Device::create(int index, const uint8_t *luid, std::string *why)
{
    auto fail = [&](const std::string &m) -> std::unique_ptr<Device> {
        if (why) *why = m;
        return nullptr;
    };
    std::string reason;
    if (!instance(&reason)) return fail(reason);
    const auto pds = physical_devices();
    if (pds.empty()) return fail("no Vulkan devices");
    int pick = -1;
    if (index >= 0) {
        if (index >= (int)pds.size()) return fail("no Vulkan device " + std::to_string(index));
        pick = index;
    } else if (luid) {
        // two drivers can expose one adapter: prefer the one with a kernel family
        for (uint32_t i = 0; i < pds.size(); i++) {
            const DeviceInfo d = probe(pds[i], i);
            if (!d.luid_valid || std::memcmp(d.luid, luid, 8)) continue;
            if (pick < 0 || d.family != Family::None) pick = (int)i;
            if (d.family != Family::None) break;
        }
        if (pick < 0) return fail("no Vulkan device matches the D3D11 adapter");
    } else {
        for (uint32_t i = 0; i < pds.size() && pick < 0; i++) {
            VkPhysicalDeviceProperties pr;
            vkGetPhysicalDeviceProperties(pds[i], &pr);
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pick = (int)i;
        }
        if (pick < 0) pick = 0;
    }

    std::unique_ptr<Device> d(new Device());
    d->phys_ = pds[pick];
    d->info_ = probe(d->phys_, (uint32_t)pick);
    const DeviceInfo &in = d->info_;
    if (in.family == Family::None)
        return fail(in.name + ": no aji_vk kernel family runs here (cm16: " + in.cm16_why + "; sg16: " + in.sg16_why +
                    "; plain: " + in.plain_why + ")");
    vkGetPhysicalDeviceMemoryProperties(d->phys_, &d->mem_);

    // a compute queue, compute-only when there is one
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys_, &nq, qf.data());
    int qi = -1;
    for (uint32_t i = 0; i < nq && qi < 0; i++)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qf[i].timestampValidBits)
            qi = (int)i;
    for (uint32_t i = 0; i < nq && qi < 0; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qi = (int)i;
    if (qi < 0) return fail("no compute queue");
    d->qfam_ = (uint32_t)qi;
    if (!qf[qi].timestampValidBits) d->info_.timestamp_ns = 0.0;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = d->qfam_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    Features f;
    const bool cm = in.family == Family::CM16;
    f.chain(cm);
    f.f11.storageBuffer16BitAccess = VK_TRUE;
    f.f12.shaderFloat16 = VK_TRUE;
    f.f12.vulkanMemoryModel = cm ? VK_TRUE : VK_FALSE;
    f.f12.vulkanMemoryModelDeviceScope = cm ? VK_TRUE : VK_FALSE;
    // plain pins no subgroup size, and its devices may lack the feature
    f.f13.subgroupSizeControl = in.family != Family::PLAIN ? VK_TRUE : VK_FALSE;
    f.f13.computeFullSubgroups = in.family != Family::PLAIN ? VK_TRUE : VK_FALSE;
    f.f13.synchronization2 = VK_TRUE;
    f.f12.timelineSemaphore = VK_TRUE;  // required by Vulkan 1.3
    f.cm.cooperativeMatrix = cm ? VK_TRUE : VK_FALSE;
    std::vector<const char *> exts;
    if (cm) exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#ifdef _WIN32
    d->interop_ = has_ext(d->phys_, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) &&
                  has_ext(d->phys_, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME) && vkGetMemoryWin32HandlePropertiesKHR &&
                  vkImportSemaphoreWin32HandleKHR;
    if (d->interop_) {
        exts.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#endif
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f.f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.data();
    VkResult r = vkCreateDevice(d->phys_, &dci, nullptr, &d->dev_);
    if (r != VK_SUCCESS) return fail(err("vkCreateDevice", r));
    vkGetDeviceQueue(d->dev_, d->qfam_, 0, &d->queue_);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = d->qfam_;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if ((r = vkCreateCommandPool(d->dev_, &cpi, nullptr, &d->pool_)) != VK_SUCCESS) return fail(err("vkCreateCommandPool", r));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = d->pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if ((r = vkAllocateCommandBuffers(d->dev_, &cai, &d->cb_)) != VK_SUCCESS) return fail(err("vkAllocateCommandBuffers", r));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if ((r = vkCreateFence(d->dev_, &fci, nullptr, &d->fence_)) != VK_SUCCESS) return fail(err("vkCreateFence", r));
    // frame command buffers: their own pool (frame_mu_), so frames and one-shot work can come
    // from different threads
    if ((r = vkCreateCommandPool(d->dev_, &cpi, nullptr, &d->frame_pool_)) != VK_SUCCESS)
        return fail(err("vkCreateCommandPool", r));
    cai.commandPool = d->frame_pool_;
    for (FrameSlot &fs : d->frames_) {
        if ((r = vkAllocateCommandBuffers(d->dev_, &cai, &fs.cb)) != VK_SUCCESS) return fail(err("vkAllocateCommandBuffers", r));
        if ((r = vkCreateFence(d->dev_, &fci, nullptr, &fs.fence)) != VK_SUCCESS) return fail(err("vkCreateFence", r));
    }

    VkDescriptorSetLayoutBinding lb[kBindings];
    for (int i = 0; i < kBindings; i++) lb[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = kBindings;
    dl.pBindings = lb;
    if ((r = vkCreateDescriptorSetLayout(d->dev_, &dl, nullptr, &d->set_layout_)) != VK_SUCCESS)
        return fail(err("vkCreateDescriptorSetLayout", r));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &d->set_layout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if ((r = vkCreatePipelineLayout(d->dev_, &pli, nullptr, &d->pipe_layout_)) != VK_SUCCESS)
        return fail(err("vkCreatePipelineLayout", r));
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if ((r = vkCreatePipelineCache(d->dev_, &pci, nullptr, &d->cache_)) != VK_SUCCESS) return fail(err("vkCreatePipelineCache", r));

    if (!d->make_buffer(&d->dummy_, kBindings * 16, false, &reason)) return fail(reason);
    VkCommandBuffer cb = d->begin();
    vkCmdFillBuffer(cb, d->dummy_.buf, 0, VK_WHOLE_SIZE, 0);
    if (!d->submit_wait(cb, &reason)) return fail(reason);
    return d;
}

Device::~Device()
{
    if (!dev_) return;
    vkDeviceWaitIdle(dev_);
    save_caches();
    for (auto &p : pipelines_) vkDestroyPipeline(dev_, p.second, nullptr);
    destroy(dummy_);
    destroy(stage_);
    if (cache_) vkDestroyPipelineCache(dev_, cache_, nullptr);
    if (pipe_layout_) vkDestroyPipelineLayout(dev_, pipe_layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(dev_, set_layout_, nullptr);
    if (fence_) vkDestroyFence(dev_, fence_, nullptr);
    for (FrameSlot &fs : frames_)
        if (fs.fence) vkDestroyFence(dev_, fs.fence, nullptr);
    if (frame_pool_) vkDestroyCommandPool(dev_, frame_pool_, nullptr);
    if (pool_) vkDestroyCommandPool(dev_, pool_, nullptr);
    vkDestroyDevice(dev_, nullptr);
}

VkCommandBuffer Device::frame_begin(int *slot)
{
    frame_mu_.lock();  // released by frame_submit
    const int s = next_frame_;
    next_frame_ = (next_frame_ + 1) % kFrameSlots;
    FrameSlot &fs = frames_[s];
    // the slot's previous submit must have finished before its buffer is re-recorded; a submit
    // still pending after 10 s (a lost device, a hand-off that never came) fails the frame
    constexpr uint64_t kTimeoutNs = 10ull * 1000 * 1000 * 1000;
    VkResult r = VK_SUCCESS;
    if (fs.fenced) {
        r = vkWaitForFences(dev_, 1, &fs.fence, VK_TRUE, kTimeoutNs);
        if (r == VK_SUCCESS) {
            vkResetFences(dev_, 1, &fs.fence);
            fs.fenced = false;
        }
    } else if (fs.sem) {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &fs.sem;
        wi.pValues = &fs.value;
        r = vkWaitSemaphores(dev_, &wi, kTimeoutNs);
        if (r == VK_SUCCESS) fs.sem = VK_NULL_HANDLE;
    }
    if (r != VK_SUCCESS) {
        frame_mu_.unlock();
        return VK_NULL_HANDLE;
    }
    vkResetCommandBuffer(fs.cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fs.cb, &bi);
    *slot = s;
    return fs.cb;
}

bool Device::frame_submit(int slot, VkSemaphore sem, uint64_t wait_value, uint64_t signal_value, std::string *why)
{
    FrameSlot &fs = frames_[slot];
    VkResult r = vkEndCommandBuffer(fs.cb);
    if (r == VK_SUCCESS) {
        VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ts.waitSemaphoreValueCount = sem && wait_value ? 1 : 0;
        ts.pWaitSemaphoreValues = &wait_value;
        ts.signalSemaphoreValueCount = sem && signal_value ? 1 : 0;
        ts.pSignalSemaphoreValues = &signal_value;
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts};
        si.waitSemaphoreCount = sem && wait_value ? 1 : 0;
        si.pWaitSemaphores = &sem;
        si.pWaitDstStageMask = &stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &fs.cb;
        si.signalSemaphoreCount = sem && signal_value ? 1 : 0;
        si.pSignalSemaphores = &sem;
        const bool fenced = !(sem && signal_value);
        r = vkQueueSubmit(queue_, 1, &si, fenced ? fs.fence : VK_NULL_HANDLE);
        if (r == VK_SUCCESS) {
            fs.fenced = fenced;
            fs.sem = fenced ? VK_NULL_HANDLE : sem;
            fs.value = signal_value;
        }
    }
    frame_mu_.unlock();
    if (r != VK_SUCCESS && why) *why = err("frame submit", r);
    return r == VK_SUCCESS;
}

#ifdef _WIN32
bool Device::import_d3d12_buffer(HANDLE h, VkDeviceSize size, Buffer *out, std::string *why)
{
    *out = Buffer{};
    if (!interop_) {
        if (why) *why = "the Vulkan driver can't import D3D12 memory";
        return false;
    }
    const VkExternalMemoryHandleTypeFlagBits ht = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkExternalMemoryBufferCreateInfo emb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    emb.handleTypes = ht;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &emb};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(dev_, &bi, nullptr, &out->buf);
    if (r != VK_SUCCESS) {
        if (why) *why = err("vkCreateBuffer (D3D12 import)", r);
        return false;
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev_, out->buf, &mr);
    VkMemoryWin32HandlePropertiesKHR hp{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    if ((r = vkGetMemoryWin32HandlePropertiesKHR(dev_, ht, h, &hp)) != VK_SUCCESS) {
        destroy(*out);
        if (why) *why = err("vkGetMemoryWin32HandlePropertiesKHR", r);
        return false;
    }
    const uint32_t bits = mr.memoryTypeBits & hp.memoryTypeBits;
    uint32_t type = 0;
    while (type < 32 && !(bits & (1u << type))) type++;
    if (type == 32) {
        destroy(*out);
        if (why) *why = "no memory type for the imported D3D12 buffer";
        return false;
    }
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.buffer = out->buf;
    VkImportMemoryWin32HandleInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &ded};
    imp.handleType = ht;
    imp.handle = h;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if ((r = vkAllocateMemory(dev_, &ai, nullptr, &out->mem)) != VK_SUCCESS ||
        (r = vkBindBufferMemory(dev_, out->buf, out->mem, 0)) != VK_SUCCESS) {
        destroy(*out);
        if (why) *why = err("importing the D3D12 buffer", r);
        return false;
    }
    out->size = size;
    return true;
}

bool Device::import_d3d12_fence(HANDLE h, VkSemaphore *out, std::string *why)
{
    *out = VK_NULL_HANDLE;
    if (!interop_) {
        if (why) *why = "the Vulkan driver can't import D3D12 fences";
        return false;
    }
    VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st};
    VkResult r = vkCreateSemaphore(dev_, &si, nullptr, out);
    if (r != VK_SUCCESS) {
        if (why) *why = err("vkCreateSemaphore", r);
        return false;
    }
    VkImportSemaphoreWin32HandleInfoKHR ii{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    ii.semaphore = *out;
    ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    ii.handle = h;
    if ((r = vkImportSemaphoreWin32HandleKHR(dev_, &ii)) != VK_SUCCESS) {
        vkDestroySemaphore(dev_, *out, nullptr);
        *out = VK_NULL_HANDLE;
        if (why) *why = err("importing the D3D12 fence", r);
        return false;
    }
    return true;
}
#endif

bool Device::make_buffer(Buffer *b, VkDeviceSize size, bool host, std::string *why)
{
    *b = Buffer{};
    if (size == 0) size = 16;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(dev_, &bi, nullptr, &b->buf);
    if (r != VK_SUCCESS) {
        if (why) *why = err("vkCreateBuffer", r);
        return false;
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev_, b->buf, &mr);
    const VkMemoryPropertyFlags want = host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                            : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type = UINT32_MAX;
    // host: prefer cached memory for readback; device: the first device-local type
    for (int pass = 0; pass < 2 && type == UINT32_MAX; pass++)
        for (uint32_t i = 0; i < mem_.memoryTypeCount && type == UINT32_MAX; i++) {
            const VkMemoryPropertyFlags fl = mem_.memoryTypes[i].propertyFlags;
            if (!(mr.memoryTypeBits & (1u << i)) || (fl & want) != want) continue;
            if (host && pass == 0 && !(fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) continue;
            type = i;
        }
    if (type == UINT32_MAX) {
        destroy(*b);
        if (why) *why = "no suitable memory type";
        return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if ((r = vkAllocateMemory(dev_, &ai, nullptr, &b->mem)) != VK_SUCCESS ||
        (r = vkBindBufferMemory(dev_, b->buf, b->mem, 0)) != VK_SUCCESS ||
        (host && (r = vkMapMemory(dev_, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map)) != VK_SUCCESS)) {
        destroy(*b);
        if (why) *why = err(r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "device memory allocation" : "buffer memory", r);
        return false;
    }
    b->size = size;
    return true;
}

void Device::destroy(Buffer &b)
{
    if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buffer{};
}

VkCommandBuffer Device::begin()
{
    mu_.lock();  // released by submit_wait
    return begin_locked();
}

VkCommandBuffer Device::begin_locked()
{
    vkResetCommandBuffer(cb_, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb_, &bi);
    return cb_;
}

bool Device::submit_wait(VkCommandBuffer cb, std::string *why)
{
    const bool ok = submit_locked(cb, why);
    mu_.unlock();
    return ok;
}

bool Device::submit_locked(VkCommandBuffer cb, std::string *why)
{
    VkResult r = vkEndCommandBuffer(cb);
    if (r == VK_SUCCESS) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        r = vkQueueSubmit(queue_, 1, &si, fence_);
        if (r == VK_SUCCESS) r = vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX);
        vkResetFences(dev_, 1, &fence_);
    }
    if (r != VK_SUCCESS && why) *why = err("queue submit", r);
    return r == VK_SUCCESS;
}

bool Device::copy_staged(const Buffer &buf, void *data, size_t bytes, bool up, std::string *why)
{
    if (bytes == 0) return true;
    if (bytes > buf.size) {
        if (why) *why = "copy larger than the buffer";
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (stage_.size < bytes) {
        destroy(stage_);
        if (!make_buffer(&stage_, bytes, true, why)) return false;
    }
    if (up) std::memcpy(stage_.map, data, bytes);
    VkCommandBuffer cb = begin_locked();
    VkBufferCopy c{0, 0, bytes};
    if (up) vkCmdCopyBuffer(cb, stage_.buf, buf.buf, 1, &c);
    else vkCmdCopyBuffer(cb, buf.buf, stage_.buf, 1, &c);
    if (!submit_locked(cb, why)) return false;
    if (!up) std::memcpy(data, stage_.map, bytes);
    return true;
}

VkPipeline Device::pipeline(const char *module, uint32_t local_size, uint32_t subgroup, std::string *why)
{
    const std::string key = std::string(module) + "/" + std::to_string(local_size) + "/" + std::to_string(subgroup);
    std::lock_guard<std::mutex> lock(pipe_mu_);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    const SpirvModule *m = spirv_find(module);
    if (!m) {
        if (why) *why = std::string("no embedded shader ") + module;
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = m->words * 4;
    smi.pCode = m->code;
    VkShaderModule sm;
    VkResult r = vkCreateShaderModule(dev_, &smi, nullptr, &sm);
    if (r != VK_SUCCESS) {
        if (why) *why = err("vkCreateShaderModule", r);
        return VK_NULL_HANDLE;
    }
    VkSpecializationMapEntry se{0, 0, 4};
    VkSpecializationInfo si{1, &se, 4, &local_size};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = subgroup;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    if (local_size) cpi.stage.pSpecializationInfo = &si;
    if (subgroup) {
        cpi.stage.pNext = &rss;
        cpi.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    cpi.layout = pipe_layout_;
    VkPipeline p = VK_NULL_HANDLE;
    const auto t0 = std::chrono::steady_clock::now();
    r = vkCreateComputePipelines(dev_, cache_, 1, &cpi, nullptr, &p);
    if (std::getenv("AJI_VK_TRACE"))
        std::fprintf(stderr, "aji_vk: pipeline %s: %.1f ms\n", module,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    vkDestroyShaderModule(dev_, sm, nullptr);
    if (r != VK_SUCCESS) {
        if (why) *why = err((std::string("pipeline ") + module).c_str(), r);
        return VK_NULL_HANDLE;
    }
    pipelines_[key] = p;
    pipelines_dirty_ = true;
    return p;
}

std::string Device::cache_base() const
{
    char name[96];
    std::snprintf(name, sizeof name, "aji-vk-%04x-%04x-%08x", info_.vendor, info_.device_id, info_.driver_version);
    std::string dir = cache_dir_;
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';
    return dir + name;
}

void Device::set_cache_dir(const std::string &dir)
{
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (dir.empty() || dir == cache_dir_) return;
    cache_dir_ = dir;
    const std::string base = cache_base();
    std::vector<char> data;
    const char *retune = std::getenv("AJI_VK_RETUNE");
    if (!(retune && std::strcmp(retune, "0")) && ajicore::read_file_bytes(base + ".variants", &data)) {
        std::istringstream in(std::string(data.begin(), data.end()));
        std::string key;
        int v = -1;
        while (in >> key >> v)
            if (v >= 0) variants_.emplace(key, v);
    }
    data.clear();
    if (ajicore::read_file_bytes(base + ".vkcache", &data) && !data.empty()) {
        // the driver checks the header and ignores data it can't use
        VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        pci.initialDataSize = data.size();
        pci.pInitialData = data.data();
        VkPipelineCache c = VK_NULL_HANDLE;
        std::lock_guard<std::mutex> plock(pipe_mu_);
        if (pipelines_.empty() && vkCreatePipelineCache(dev_, &pci, nullptr, &c) == VK_SUCCESS) {
            vkDestroyPipelineCache(dev_, cache_, nullptr);
            cache_ = c;
        }
    }
}

void Device::save_caches()
{
    std::lock_guard<std::mutex> lock(cache_mu_);
    if (cache_dir_.empty()) return;
    const std::string base = cache_base();
    auto replace = [](const std::string &path, const void *p, size_t n) {
        return ajicore::write_file_bytes(path + ".tmp", p, n) && ajicore::move_file(path + ".tmp", path);
    };
    if (variants_dirty_) {
        std::string text;
        for (const auto &kv : variants_) text += kv.first + " " + std::to_string(kv.second) + "\n";
        if (replace(base + ".variants", text.data(), text.size())) variants_dirty_ = false;
    }
    std::lock_guard<std::mutex> plock(pipe_mu_);
    if (pipelines_dirty_) {
        size_t n = 0;
        if (vkGetPipelineCacheData(dev_, cache_, &n, nullptr) == VK_SUCCESS && n) {
            std::vector<char> data(n);
            if (vkGetPipelineCacheData(dev_, cache_, &n, data.data()) == VK_SUCCESS && replace(base + ".vkcache", data.data(), n))
                pipelines_dirty_ = false;
        }
    }
}

int Device::cached_variant(const std::string &key)
{
    std::lock_guard<std::mutex> lock(cache_mu_);
    auto it = variants_.find(key);
    return it == variants_.end() ? -1 : it->second;
}

void Device::store_variant(const std::string &key, int variant)
{
    std::lock_guard<std::mutex> lock(cache_mu_);
    auto it = variants_.find(key);
    if (it != variants_.end() && it->second == variant) return;
    variants_[key] = variant;
    variants_dirty_ = true;
}

}  // namespace ajivk
