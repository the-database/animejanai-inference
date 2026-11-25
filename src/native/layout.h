/*
 * native/layout.h — the backend-neutral half of instantiating a lowered model (native/lower.h)
 * for one input size: the size checks, each step's channel padding, and the feature buffers
 * (one per live tensor, a buffer reused once its tensor's last reader has run). Host only: the
 * CUDA executor (native/executor) allocates and uploads from it, and so can any other backend.
 */
#ifndef AJI_NATIVE_LAYOUT_H
#define AJI_NATIVE_LAYOUT_H

#include <cstddef>
#include <string>
#include <vector>

#include "native/lower.h"

namespace ajinative {

struct Layout {
    int u = 1;                      // the plan's pixel-unshuffle factor
    int W = 0, H = 0;               // the conv resolution (input / u)
    int out_w = 0, out_h = 0;       // the model output (W, H times depth-to-space)
    // Per plan step, the padded channel counts its weights are packed to. A conv reading the
    // NCHW model input pads cin to 16, a depth-to-space 2 output (12 channels) pads cout to 16;
    // concats and attention weights are not padded.
    std::vector<int> cin_pad, cout_pad;
    // Per feature id (Plan::channels), the buffer that holds it, or -1 if no step writes it.
    std::vector<int> buffer;
    // Per buffer, its size: NHWC fp16 at W x H, in the order buffers are first needed.
    std::vector<size_t> buffer_bytes;
};

// Fails (false, *why says why) for an empty plan or a size the plan can't take.
bool layout(const Plan &plan, int in_w, int in_h, Layout *out, std::string *why);

// Apply once to a freshly computed layout for CUDA kernels that can overwrite a gate's residual.
void reuse_cuda_gate_buffers(const Plan &plan, Layout &layout);

}  // namespace ajinative

#endif
