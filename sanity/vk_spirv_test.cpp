// vk/spirv: the embedded Vulkan compute modules (AJI_VULKAN=ON) are well-formed SPIR-V and
// declare what their kernel family needs: the cooperative-matrix modules the
// CooperativeMatrixKHR capability, the subgroup-16 ones fp16 and subgroup operations. Walks the
// instruction stream only; no Vulkan device. CPU only.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>

#include "vk/spirv.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

using namespace ajivk;

namespace {

constexpr uint32_t kMagic = 0x07230203u;
constexpr uint32_t kOpCapability = 17, kOpEntryPoint = 15;
constexpr uint32_t kCapFloat16 = 9, kCapGroupNonUniform = 61, kCapCooperativeMatrixKHR = 6022;
constexpr uint32_t kExecutionModelGLCompute = 5;

int check_module(const SpirvModule &m)
{
    CHECK(m.name && m.code && m.words > 5);
    CHECK(m.code[0] == kMagic);
    const uint32_t version = m.code[1];
    CHECK(version >= 0x00010300u);  // SPIR-V 1.3 or newer (Vulkan 1.1+ targets)
    std::set<uint32_t> caps;
    int entry_points = 0;
    size_t i = 5;
    while (i < m.words) {
        const uint32_t wc = m.code[i] >> 16, op = m.code[i] & 0xffffu;
        CHECK(wc >= 1 && i + wc <= m.words);
        if (op == kOpCapability) caps.insert(m.code[i + 1]);
        if (op == kOpEntryPoint) {
            CHECK(m.code[i + 1] == kExecutionModelGLCompute);
            entry_points++;
        }
        i += wc;
    }
    CHECK(i == m.words);
    CHECK(entry_points == 1);
    CHECK(caps.count(kCapFloat16));
    if (std::strstr(m.name, "_cm16_")) CHECK(caps.count(kCapCooperativeMatrixKHR));
    if (std::strstr(m.name, "_sg16_")) {
        CHECK(caps.count(kCapGroupNonUniform));
        CHECK(!caps.count(kCapCooperativeMatrixKHR));
    }
    return 0;
}

}  // namespace

int main()
{
    CHECK(kSpirvModuleCount > 0);
    std::set<const char *> seen;
    for (size_t k = 0; k < kSpirvModuleCount; k++) {
        const SpirvModule &m = kSpirvModules[k];
        if (check_module(m)) {
            std::fprintf(stderr, "module %s\n", m.name ? m.name : "(null)");
            return 1;
        }
        CHECK(spirv_find(m.name) == &m);
    }
    CHECK(spirv_find("conv3x3_cm16_c48_w4_r4") && spirv_find("conv3x3_sg16_c64_t8_r1"));
    CHECK(!spirv_find("no_such_module"));
    std::printf("vk spirv: %zu modules, all checks passed\n", kSpirvModuleCount);
    return 0;
}
