/*
 * vk/spirv.h — the Vulkan compute shaders embedded in the build (AJI_VULKAN=ON): every module
 * src/vk/shaders/shaders.txt lists, compiled by glslc and turned into a table by
 * cmake/aji_spirv_gen.cmake. Host only; no Vulkan headers.
 */
#ifndef AJI_VK_SPIRV_H
#define AJI_VK_SPIRV_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ajivk {

struct SpirvModule {
    const char *name;      // the manifest's module name, e.g. conv3x3_cm16_c48_w4_r4
    const uint32_t *code;  // SPIR-V words, ready for VkShaderModuleCreateInfo::pCode
    size_t words;          // codeSize is words * 4 bytes
};

extern const SpirvModule kSpirvModules[];
extern const size_t kSpirvModuleCount;

// The module of that name, or null.
inline const SpirvModule *spirv_find(const char *name)
{
    for (size_t i = 0; i < kSpirvModuleCount; i++)
        if (!std::strcmp(kSpirvModules[i].name, name)) return &kSpirvModules[i];
    return nullptr;
}

}  // namespace ajivk

#endif
