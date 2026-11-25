#pragma once
#include "aji_image.h"
#include "native/allocation_count.h"
#include "native/lifetime_check.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace ajinative::lifetime_test {
inline void require(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}
inline void gpu(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }

struct Device {
    uint16_t *p = nullptr;
    explicit Device(size_t elements) { gpu(cudaMalloc(&p, elements * sizeof(*p))); }
    ~Device() { cudaFree(p); }
};

using Shape = std::pair<int, int>;
using Output = std::vector<uint16_t>;

inline std::array<uint64_t,3> account_native(aji_img_ctx *ctx, const aji_img_model_params &params,
                                             LifetimeCheck &probe_check, LifetimeCheck &load_check)
{
    aji_img_memory estimate{}, live{};
    LifetimeCheck::active = &probe_check;
    const int rc = aji_img_native_probe(ctx, &params, &estimate);
    LifetimeCheck::active = nullptr;
    require(rc == AJI_IMG_OK, aji_img_last_error(ctx));

    AllocationCount actual;
    AllocationCount::active = &actual;
    LifetimeCheck::active = &load_check;
    auto *model = aji_img_load(ctx, &params);
    LifetimeCheck::active = nullptr;
    AllocationCount::active = nullptr;
    require(model, aji_img_last_error(ctx));
    const int status = aji_img_memory_get(ctx, model, &live);
    aji_img_release(&model);
    require(status == AJI_IMG_OK && !live.activations_shared && live.model_activations, "expected a native model");
    const std::array<uint64_t,3> reported{estimate.model_weights,estimate.model_activations,estimate.model_buffers};
    const std::array<uint64_t,3> loaded{live.model_weights,live.model_activations,live.model_buffers};
    require(reported == loaded && std::equal(reported.begin(), reported.end(), actual.bytes),
            "native probe or live model differs from its successful CUDA allocation requests");
    return reported;
}
} // namespace ajinative::lifetime_test
