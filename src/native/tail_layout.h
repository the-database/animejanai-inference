// Host description used by Tail and by the allocation planner.
#pragma once
#include <cstddef>
#include <vector>
#include "native/gact.h"

namespace ajinative {
struct GOp;
class TailLayout {
public:
    static constexpr int kHalo = 3;
    // One 3x3 conv at the current scale (up2: its input nearest-upsampled by 2 first, so the scale doubles).
    void add(GOp *op, int out_stride, GAct act = GAct::None, float slope = 0.2f, bool up2 = false)
    {
        steps_.push_back({{op, nullptr}, false, up2, act, slope, out_stride, nullptr});
    }
    // A 2x pixel shuffle as its two phase-row convs (R 2): the scale doubles.
    void add_ps(GOp *op0, GOp *op1, int out_stride, GAct act = GAct::None, float slope = 0.2f)
    {
        steps_.push_back({{op0, op1}, true, false, act, slope, out_stride, nullptr});
    }
    int steps() const { return (int)steps_.size(); }
    // Step k's output buffer (the model lays them out); the last step's is the 16-channel output.
    void **out(int k) { return &steps_[k].dst; }
    // The scale after step k (S after the last).
    int scale_after(int k) const
    {
        int sc = 1;
        for (int i = 0; i <= k; ++i) if (steps_[i].ps || steps_[i].up2) sc *= 2;
        return sc;
    }
    int scale() const { return steps_.empty() ? 1 : scale_after(steps() - 1); }
    // Bytes of step k's output for `rows` body rows of a bw-wide body.
    size_t bytes(int k, int bw, int rows) const
    {
        const size_t sc = (size_t)scale_after(k);
        return sc * sc * bw * rows * steps_[k].out_stride * 2;
    }

protected:
    struct Step {
        GOp *op[2];
        bool ps, up2;
        GAct act;
        float slope;
        int out_stride;
        void *dst;
    };
    std::vector<Step> steps_;
};
} // namespace ajinative
