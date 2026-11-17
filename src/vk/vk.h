/*
 * vk/vk.h — the Vulkan entry points aji_vk uses, loaded at run time from vulkan-1.dll /
 * libvulkan.so.1 (nothing links against a Vulkan SDK, and a machine without a driver gets an
 * error, not a missing-DLL failure). One process-wide VkInstance; every function, device-level
 * ones included, is resolved through vkGetInstanceProcAddr, so the pointers serve any device.
 */
#ifndef AJI_VK_VK_H
#define AJI_VK_VK_H

#define VK_NO_PROTOTYPES
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <vulkan/vulkan_core.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include <string>

#define AJI_VK_INSTANCE_FUNCS(X)                     \
    X(vkDestroyInstance)                             \
    X(vkEnumeratePhysicalDevices)                    \
    X(vkGetPhysicalDeviceProperties)                 \
    X(vkGetPhysicalDeviceProperties2)                \
    X(vkGetPhysicalDeviceFeatures2)                  \
    X(vkGetPhysicalDeviceMemoryProperties)           \
    X(vkGetPhysicalDeviceMemoryProperties2)          \
    X(vkGetPhysicalDeviceQueueFamilyProperties)      \
    X(vkEnumerateDeviceExtensionProperties)          \
    X(vkCreateDevice)                                \
    X(vkGetDeviceProcAddr)

// Optional (extensions): null when the driver lacks them.
#ifdef _WIN32
#define AJI_VK_INSTANCE_EXT_FUNCS(X)                     \
    X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) \
    X(vkGetMemoryWin32HandlePropertiesKHR)               \
    X(vkImportSemaphoreWin32HandleKHR)
#else
#define AJI_VK_INSTANCE_EXT_FUNCS(X) X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
#endif

#define AJI_VK_DEVICE_FUNCS(X)          \
    X(vkDestroyDevice)                  \
    X(vkGetDeviceQueue)                 \
    X(vkDeviceWaitIdle)                 \
    X(vkCreateBuffer)                   \
    X(vkDestroyBuffer)                  \
    X(vkGetBufferMemoryRequirements)    \
    X(vkAllocateMemory)                 \
    X(vkFreeMemory)                     \
    X(vkBindBufferMemory)               \
    X(vkMapMemory)                      \
    X(vkUnmapMemory)                    \
    X(vkCreateShaderModule)             \
    X(vkDestroyShaderModule)            \
    X(vkCreateDescriptorSetLayout)      \
    X(vkDestroyDescriptorSetLayout)     \
    X(vkCreatePipelineLayout)           \
    X(vkDestroyPipelineLayout)          \
    X(vkCreateComputePipelines)         \
    X(vkDestroyPipeline)                \
    X(vkCreatePipelineCache)            \
    X(vkDestroyPipelineCache)           \
    X(vkGetPipelineCacheData)           \
    X(vkCreateDescriptorPool)           \
    X(vkDestroyDescriptorPool)          \
    X(vkAllocateDescriptorSets)         \
    X(vkUpdateDescriptorSets)           \
    X(vkCreateCommandPool)              \
    X(vkDestroyCommandPool)             \
    X(vkAllocateCommandBuffers)         \
    X(vkFreeCommandBuffers)             \
    X(vkBeginCommandBuffer)             \
    X(vkEndCommandBuffer)               \
    X(vkResetCommandBuffer)             \
    X(vkCmdBindPipeline)                \
    X(vkCmdBindDescriptorSets)          \
    X(vkCmdPushConstants)               \
    X(vkCmdDispatch)                    \
    X(vkCmdPipelineBarrier)             \
    X(vkCmdCopyBuffer)                  \
    X(vkCmdFillBuffer)                  \
    X(vkCmdUpdateBuffer)                \
    X(vkCmdWriteTimestamp)              \
    X(vkCmdResetQueryPool)              \
    X(vkCreateQueryPool)                \
    X(vkDestroyQueryPool)               \
    X(vkGetQueryPoolResults)            \
    X(vkQueueSubmit)                    \
    X(vkQueueWaitIdle)                  \
    X(vkCreateFence)                    \
    X(vkDestroyFence)                   \
    X(vkWaitForFences)                  \
    X(vkResetFences)                    \
    X(vkCreateImage)                    \
    X(vkDestroyImage)                   \
    X(vkGetImageMemoryRequirements)     \
    X(vkBindImageMemory)                \
    X(vkCmdCopyImageToBuffer)           \
    X(vkCmdCopyBufferToImage)           \
    X(vkCmdCopyImage)                   \
    X(vkCreateSemaphore)                \
    X(vkDestroySemaphore)               \
    X(vkWaitSemaphores)                 \
    X(vkGetSemaphoreCounterValue)

namespace ajivk {

#define AJI_VK_DECLARE(name) extern PFN_##name name;
AJI_VK_INSTANCE_FUNCS(AJI_VK_DECLARE)
AJI_VK_INSTANCE_EXT_FUNCS(AJI_VK_DECLARE)
AJI_VK_DEVICE_FUNCS(AJI_VK_DECLARE)
#undef AJI_VK_DECLARE

// Loads the Vulkan loader and creates the process-wide instance (Vulkan 1.3) on first use;
// later calls return the same instance. Null (and *why) without a usable Vulkan 1.3 loader.
VkInstance instance(std::string *why);

const char *result_name(VkResult r);

}  // namespace ajivk

#endif
