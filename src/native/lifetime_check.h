// Optional test observer. Captures the assigned pointers, independently checks their ranges,
// and poisons every byte except values that must survive the next step on the same stream.
#pragma once
#include "native/lifetime_plan.h"
#include <string>

namespace ajinative {
class LifetimeCheck {
public:
    // Test hook, null in production. Models capture it at build time and test the saved pointer at dispatch.
    inline static thread_local LifetimeCheck *active = nullptr;
    void (*edit)(LifetimePlan &) = nullptr; // test reference layout or deliberately incorrect lifetime
    bool poison = false;
    size_t workspace_arena_limit = size_t(-1); // test the build-time separate allocation with a smaller arena
    std::string error;

    void region(const void *base, size_t bytes)
    {
        if (bytes) regions.push_back(span(base, bytes));
    }
    void capture(const LifetimePlan &plan, const void *base, size_t bytes)
    {
        region(base, bytes);
        for (const auto &b : plan.buffers) {
            if (b.offset > bytes || b.bytes > bytes - b.offset ||
                reinterpret_cast<uintptr_t>(*b.slot) != reinterpret_cast<uintptr_t>(base) + b.offset)
                fail("slot does not match its arena offset");
            features.push_back({span(*b.slot, b.bytes), b.live, b.slot});
        }
    }

    bool valid()
    {
        if (!error.empty()) return false;
        for (size_t i = 0; i < regions.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                if (overlaps(regions[i], regions[j])) return fail("overlapping allocations");
        for (size_t i = 0; i < features.size(); ++i) {
            const auto &a = features[i];
            if (!contains(a.mem) || a.live.empty()) return fail("feature outside allocation or missing lifetime");
            for (const auto &x : a.live) {
                if (x.first > x.second) return fail("reversed lifetime");
                for (size_t j = 0; j < i; ++j) {
                    const auto &b = features[j];
                    if (overlaps(a.mem, b.mem))
                        for (const auto &y : b.live)
                            if (std::max(x.first, y.first) <= std::min(x.second, y.second)) return fail("live features alias");
                }
            }
        }
        return true;
    }

    bool workspace(int step, const void *ptr, size_t bytes)
    {
        if (!bytes) return true;
        const auto w = span(ptr, bytes);
        if (!contains(w)) return fail("workspace outside allocation");
        for (const auto &f : features)
            if (f.at(step) && overlaps(w, f.mem)) return fail("workspace overlaps live feature");
        return true;
    }

    template<class Fill>
    bool enter(int next, Fill fill, void **forwarded = nullptr)
    {
        if (!poison) return true;
        std::vector<Span> keep;
        for (const auto &f : features) {
            // A fused writer can produce the next block's input before the numbered steps wrap.
            if (forwarded == f.slot && f.at(previous) && f.at(next)) {
                keep.push_back(f.mem);
                continue;
            }
            for (auto r : f.live)
                if (r.first <= std::min(previous, next) && std::max(previous, next) <= r.second) {
                    keep.push_back(f.mem);
                    break;
                }
        }
        std::sort(keep.begin(), keep.end());
        for (const auto &r : regions) {
            uintptr_t pos = r.first;
            for (const auto &k : keep) {
                if (!overlaps(r, k)) continue;
                if (pos < k.first && !fill(reinterpret_cast<void *>(pos), k.first - pos)) return false;
                pos = std::max(pos, k.second);
            }
            if (pos < r.second && !fill(reinterpret_cast<void *>(pos), r.second - pos)) return false;
        }
        previous = next;
        return true;
    }

private:
    using Span = std::pair<uintptr_t, uintptr_t>;
    struct Feature {
        Span mem;
        std::vector<LifetimePlan::Range> live;
        void **slot;
        bool at(int step) const {
            for (auto r : live) if (r.first <= step && step <= r.second) return true;
            return false;
        }
    };
    static Span span(const void *p, size_t n) {
        const auto start = reinterpret_cast<uintptr_t>(p);
        return {start, start + n};
    }
    static bool overlaps(Span a, Span b) { return a.first < b.second && b.first < a.second; }
    bool contains(Span x) const {
        for (const auto &r : regions) if (r.first <= x.first && x.first < x.second && x.second <= r.second) return true;
        return false;
    }
    bool fail(const char *s) { error = s; return false; }
    std::vector<Span> regions;
    std::vector<Feature> features;
    int previous = -1;
};
} // namespace ajinative
