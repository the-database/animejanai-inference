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
#include "native/tail_layout.h"

namespace ajinative {

class Tail : public TailLayout {
public:
    // The ops on body rows [t0, t0 + tn) of src (src_stride channels, bw wide), in order.
    void plan(const void *src, int src_stride, int bw, int t0, int tn, std::vector<std::pair<GOp *, GConvArgs>> *ops) const;
    // The body rows a strip of body rows [b0, b1) of a bh-tall body runs on: [*a, *z).
    static void band(int b0, int b1, int bh, int *a, int *z);
    // After a plan from body row a: output rows [b0, b1) x S into out (NCHW, 3 planes of S*bw x (b1-b0)*S).
    cudaError_t emit(int a, int b0, int b1, int bw, void *out, bool out_f32, bool clamp, cudaStream_t s) const;
};

}  // namespace ajinative

#endif
