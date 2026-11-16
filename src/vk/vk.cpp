// vk/vk.cpp — see vk.h.
#include "vk/vk.h"

#include <mutex>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace ajivk {

#define AJI_VK_DEFINE(name) PFN_##name name = nullptr;
AJI_VK_INSTANCE_FUNCS(AJI_VK_DEFINE)
AJI_VK_INSTANCE_EXT_FUNCS(AJI_VK_DEFINE)
AJI_VK_DEVICE_FUNCS(AJI_VK_DEFINE)
#undef AJI_VK_DEFINE

namespace {

std::mutex g_mu;
VkInstance g_instance = VK_NULL_HANDLE;
std::string g_error;
bool g_tried = false;

PFN_vkGetInstanceProcAddr load_loader()
{
#ifdef _WIN32
    HMODULE h = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h) h = LoadLibraryW(L"vulkan-1.dll");
    return h ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(h, "vkGetInstanceProcAddr")) : nullptr;
#elif defined(__ANDROID__)
    void *h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return h ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(h, "vkGetInstanceProcAddr")) : nullptr;
#else
    void *h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return h ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(h, "vkGetInstanceProcAddr")) : nullptr;
#endif
}

VkInstance create(std::string *why)
{
    PFN_vkGetInstanceProcAddr gipa = load_loader();
    if (!gipa) {
        *why = "no Vulkan loader (vulkan-1.dll / libvulkan.so.1): install a GPU driver with Vulkan";
        return VK_NULL_HANDLE;
    }
    auto enum_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    uint32_t version = VK_API_VERSION_1_0;
    if (enum_version) enum_version(&version);
    if (!create_instance || version < VK_API_VERSION_1_3) {
        *why = "the Vulkan loader is older than 1.3";
        return VK_NULL_HANDLE;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "libaji";
    app.pEngineName = "aji_vk";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    const VkResult r = create_instance(&ici, nullptr, &inst);
    if (r != VK_SUCCESS) {
        *why = std::string("vkCreateInstance failed: ") + result_name(r);
        return VK_NULL_HANDLE;
    }
    bool ok = true;
#define AJI_VK_LOAD(name)                                                       \
    name = reinterpret_cast<PFN_##name>(gipa(inst, #name));                     \
    if (!name) {                                                                \
        ok = false;                                                             \
        *why = "the Vulkan loader lacks " #name;                                \
    }
    AJI_VK_INSTANCE_FUNCS(AJI_VK_LOAD)
    AJI_VK_DEVICE_FUNCS(AJI_VK_LOAD)
#undef AJI_VK_LOAD
#define AJI_VK_LOAD_OPT(name) name = reinterpret_cast<PFN_##name>(gipa(inst, #name));
    AJI_VK_INSTANCE_EXT_FUNCS(AJI_VK_LOAD_OPT)
#undef AJI_VK_LOAD_OPT
    if (!ok) {
        auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(gipa(inst, "vkDestroyInstance"));
        if (destroy) destroy(inst, nullptr);
        return VK_NULL_HANDLE;
    }
    return inst;
}

}  // namespace

VkInstance instance(std::string *why)
{
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_tried) {
        g_tried = true;
        g_instance = create(&g_error);
    }
    if (!g_instance && why) *why = g_error;
    return g_instance;
}

const char *result_name(VkResult r)
{
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    default: return "VkResult error";
    }
}

}  // namespace ajivk
