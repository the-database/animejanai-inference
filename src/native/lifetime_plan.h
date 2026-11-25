// Host part of Lifetimes. The CPU verifier and the device allocator use the same placement.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ajinative {
class LifetimePlan {
public:
    using Range = std::pair<int, int>;
    struct Buffer {
        void **slot;
        size_t bytes;
        std::vector<Range> live;
        size_t offset = 0;
        const char *name;
    };

    void add(void **slot, size_t bytes, std::vector<Range> live, const char *name = nullptr)
    {
        buffers.push_back({slot, (bytes + 255) / 256 * 256, std::move(live), 0, name});
    }

    size_t arrange()
    {
        std::vector<size_t> order(buffers.size()), placed;
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return buffers[a].bytes > buffers[b].bytes; });
        size_t total = 0;
        for (size_t i : order) {
            Buffer &b = buffers[i];
            std::vector<std::pair<size_t, size_t>> taken;
            for (size_t j : placed) {
                const Buffer &a = buffers[j];
                if (meets(a, b)) taken.push_back({a.offset, a.offset + a.bytes});
            }
            std::sort(taken.begin(), taken.end());
            size_t off = 0;
            for (const auto &t : taken) {
                if (off + b.bytes <= t.first) break;
                off = std::max(off, t.second);
            }
            b.offset = off;
            total = std::max(total, off + b.bytes);
            placed.push_back(i);
        }
        capacity = total;
        return total;
    }

    void bind(void *base)
    {
        origin = reinterpret_cast<uintptr_t>(base);
        for (const Buffer &b : buffers) *b.slot = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) + b.offset);
    }

    // A free range at one dispatch step. limit can restrict the arena in a fallback test.
    void *vacant(int step, size_t bytes, size_t limit) const
    {
        const size_t end = std::min(capacity, limit);
        if (!origin || !bytes || bytes > end) return nullptr;
        std::vector<std::pair<size_t, size_t>> occupied;
        for (const auto &b : buffers)
            for (auto r : b.live)
                if (r.first <= step && step <= r.second) {
                    occupied.push_back({b.offset, b.offset + b.bytes});
                    break;
                }
        std::sort(occupied.begin(), occupied.end());
        size_t offset = 0;
        for (auto r : occupied) {
            if (r.first >= offset && r.first - offset >= bytes) break;
            offset = std::max(offset, r.second);
        }
        return offset <= end - bytes ? reinterpret_cast<void *>(origin + offset) : nullptr;
    }

    std::vector<Buffer> buffers;

private:
    uintptr_t origin = 0;
    size_t capacity = 0;
    static bool meets(const Buffer &a, const Buffer &b)
    {
        for (const Range &x : a.live)
            for (const Range &y : b.live)
                if (x.first <= y.second && y.first <= x.second) return true;
        return false;
    }
};
} // namespace ajinative
