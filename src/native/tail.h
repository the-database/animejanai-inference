// native/tail.h — an image model's upscale tail as data, for ImgNet's body / tail split: its convs from
// the body's output (NHWC, at the body's size) up to the 16-channel output at S times it, planned on any
// band of the body's rows. A model lists its steps once (add, add_ps) and points each step's output at
// its buffer (out); run() and a split tail() then plan the same steps on the whole body or on a strip.
//
// A strip runs on its rows plus kHalo body rows of context on each side (clamped to the body), and keeps
// only its own rows: each 3x3 conv's zero padding at a band edge reaches a few output pixels per scale
// step into the band (at most 2 + 2 * S output rows over the steps here), under kHalo body rows.
#ifndef AJI_NATIVE_TAIL_H
#define AJI_NATIVE_TAIL_H

#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "native/gconv.h"
#include "native/netres.h"

namespace ajinative {

class Tail {
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
    int scale_after(int k) const;
    int scale() const { return steps_.empty() ? 1 : scale_after(steps() - 1); }
    // Bytes of step k's output for `rows` body rows of a bw-wide body.
    size_t bytes(int k, int bw, int rows) const;

    // The ops on body rows [t0, t0 + tn) of src (src_stride channels, bw wide), in order.
    void plan(const void *src, int src_stride, int bw, int t0, int tn, std::vector<std::pair<GOp *, GConvArgs>> *ops) const;
    // The body rows a strip of body rows [b0, b1) of a bh-tall body runs on: [*a, *z).
    static void band(int b0, int b1, int bh, int *a, int *z);
    // After a plan from body row a: output rows [b0, b1) x S into out (NCHW, 3 planes of S*bw x (b1-b0)*S).
    cudaError_t emit(int a, int b0, int b1, int bw, void *out, bool out_f32, bool clamp, cudaStream_t s) const;

private:
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

}  // namespace ajinative

#endif
