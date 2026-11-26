#pragma once
#include <algorithm>
#include <array>
#include "native/lifetime_check.h"

namespace ajinative {
inline constexpr int gconv_tile_width = 16;

// The stock split policy targets two CTAs per SM, bounded by the number of tap/chunk units.
inline int gconv_split_count(size_t ctas, int units, int sms)
{
    const size_t target = size_t(2) * sms;
    return ctas >= target ? 1 : std::min(units, int((target + ctas - 1) / ctas));
}

struct RowPhasePlan {
    int batch = 1;
    int splits = 1;
    size_t phase_floats = 0;
};

// Prefer four disjoint partial-sum ranges, then two. One means keep the original launches.
// Compare by division before multiplying. An unsplit convolution writes directly to its output.
inline RowPhasePlan row_phase_plan(int splits, size_t pixels, int channels, size_t available)
{
    if (splits == 1) return {4, 1, 0};
    if (available / 2 / splits / channels < pixels) return {};
    const size_t phase = pixels * channels * splits;
    return {available / phase >= 4 ? 4 : 2, splits, phase};
}

// In floats. For q unsplit CTAs and k > 1, q < 2*SM and k <= ceil(2*SM/q).
// Thus q*k <= 2*SM + q - 1 <= 4*SM - 2. Each CTA covers at most
// tile_width * rows * output_chunk floats. An unsplit FP32 output has its own full-map bound.
inline size_t gconv_workspace_bound(int sms, int rows, int output_chunk, size_t full_fp32, size_t stock_limit)
{
    const size_t split = (size_t(4) * sms - 2) * gconv_tile_width * rows * output_chunk;
    return std::max(full_fp32, std::min(split, stock_limit));
}

struct ConvScratch {
    float *ptr = nullptr;
    size_t floats = 0;
};

// Dispatches on the same stream can reuse a range. Each step keeps the largest requirement
// of any op at that step. Ops without a hole share one external allocation made at build time.
template<size_t Steps>
struct StepWorkspaces {
    std::array<ConvScratch, Steps> slots{};

    void need(int step, size_t bound) { slots[step].floats = std::max(slots[step].floats, bound); }

    template<class Alloc>
    bool place(const LifetimePlan &plan, Alloc allocate, LifetimeCheck *check)
    {
        size_t external = 0;
        const size_t arena_limit = check ? check->workspace_arena_limit : size_t(-1);
        for (size_t i = 0; i < Steps; ++i) {
            auto &slot = slots[i];
            if (!slot.floats) continue;
            slot.ptr = static_cast<float *>(plan.vacant((int)i, slot.floats * sizeof(float), arena_limit));
            if (!slot.ptr) external = std::max(external, slot.floats);
        }
        if (!external) return true;
        const size_t bytes = external * sizeof(float);
        auto *fallback = static_cast<float *>(allocate(bytes));
        if (!fallback) return false;
        if (check) check->region(fallback, bytes);
        for (auto &slot : slots) if (slot.floats && !slot.ptr) slot.ptr = fallback;
        return true;
    }
};
} // namespace ajinative
