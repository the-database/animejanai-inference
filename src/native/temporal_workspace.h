#pragma once

#include <algorithm>
#include <cstddef>

namespace ajinative {

// RTX 3090 benefits from overlapping two large alignment chains. L4 and RTX 5090 can share one.
inline constexpr int tfix_alignment_chains(size_t pixels, int sm)
{
    if (pixels <= 500000) return 6;
    return (sm == 89 || sm == 120) ? 1 : 2;
}

// One phase uses a single span, the other uses two disjoint equal spans.
// Sizes and offsets use the caller's units (bytes or fp16 elements).
struct PingPongLayout {
    size_t size, second_offset;
};

inline PingPongLayout ping_pong_layout(size_t single, size_t lane)
{
    return {std::max(single, 2 * lane), lane};
}

struct TemporalInputLayout {
    int input_frames, cache_entries;
};

inline TemporalInputLayout temporal_input_layout(int window, int entries, bool cache_enabled)
{
    if (cache_enabled && entries >= window) return {1, entries};
    return {window, 0};
}

}  // namespace ajinative
