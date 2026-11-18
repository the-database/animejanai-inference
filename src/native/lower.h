/*
 * native/lower.h — turn an ONNX graph into a plan of fused native kernel launches, or say why not.
 *
 * Lowering works op by op on the graph's data path (the tensors computed from the model input;
 * shape-only plumbing such as Shape/Gather/Concat-of-sizes is ignored) and folds what follows
 * each Conv into that Conv:
 *   - Relu / PRelu / LeakyRelu / SiLU (Sigmoid x Mul) after a conv -> its activation;
 *   - SPAN gates on a conv output o with the block input x:
 *       Mul(Add(x, o), Tanh(Conv1x1(o)))           -> Gate::Tanh1x1 (SPANF3)
 *       Mul(Add(o, x), Sub(Sigmoid(o), 0.5))        -> Gate::SigmoidHalf (SPAN)
 *       Add(x, o)                                   -> Gate::Residual;
 *   - Conv1x1(Concat(f1..fn)), n = 2..4             -> the concat kernel;
 *   - the model input, optionally normalized per channel by constants (Sub / Add / Mul / Div,
 *     as SPAN's (x - mean) * 255) and pixel-unshuffled (Reshape/Transpose/Reshape or
 *     SpaceToDepth), into the first conv: the normalization is applied where that conv loads
 *     the input, so its zero padding stays zero in normalized space;
 *   - DepthToSpace (CRD or DCR, by 1 to 4, or a last conv of 3 channels as the output itself)
 *     after the last conv, optionally plus the nearest-upsampled model input (SRVGG's skip: a
 *     Resize that is pixel replication, nearest_replicates in onnx_graph.h) and a final
 *     Clip(0, 1), into the output.
 * Casts between float types, Identity / Dropout, and Pad / Slice used to align the input to the unshuffle factor
 * and crop the output back, pass through (the executor then requires aligned sizes). Any other
 * op on the data path refuses the model. Host only.
 */
#ifndef AJI_NATIVE_LOWER_H
#define AJI_NATIVE_LOWER_H

#include <cstdint>
#include <string>
#include <vector>

#include "native/onnx_graph.h"
#include "native/ops.h"

namespace ajinative {

// Conv weights as the kernels take them: w is [taps][cin][cout] fp16 bits.
struct PlanWeights {
    int taps = 0, cin = 0, cout = 0;
    std::vector<uint16_t> w;
    std::vector<float> bias;   // cout
    std::vector<float> slope;  // cout (PReLU / LeakyReLU), else empty
};

// Tensor ids: >= 0 feature maps (NHWC fp16 at the conv resolution, Plan::channels[id] channels),
// or the model input / output (NCHW fp16 RGB).
constexpr int kModelInput = -2, kModelOutput = -3;

struct PlanStep {
    enum Kind { CONV = 0, CONCAT = 1 } kind = CONV;
    // CONV: values of Input / Act / Gate / Output (native/ops.h)
    int in = 0, act = 0, gate = 0, out = 0;
    int cin = 0, cout = 0;
    PlanWeights w, att;
    int src = -1, resid = -1, dst = -1;
    // CONV reading the model input: it is (x - in_mean[c]) * in_scale[c] (Plan's normalization)
    bool in_norm = false;
    float in_mean[3] = {0.f, 0.f, 0.f}, in_scale[3] = {1.f, 1.f, 1.f};
    // CONV writing the model output: clamp it to [0, 1] (a final Clip(0, 1))
    bool out_clamp = false;
    // CONCAT: 1x1 over the concat of nsrc feature maps of c channels each
    int c = 0, nsrc = 0;
    int srcs[4] = {-1, -1, -1, -1};
};

struct Plan {
    std::vector<PlanStep> steps;
    std::vector<int> channels;   // per feature id
    int unshuffle = 1;           // the input is pixel-unshuffled by this before the first conv
    int depth_to_space = 1;      // the output is depth-to-space'd by this after the last conv
    bool aligned_only = false;   // the graph pads/crops to the unshuffle factor: sizes must be multiples
    bool fp32_weights = false;   // the model was exported in fp32 (the kernels compute in fp16)
    bool in_norm = false;        // the input is normalized per channel (the steps reading it carry it)
    bool out_clamp = false;      // the output is clamped to [0, 1] (the output step carries it)
    // output = input * depth_to_space / unshuffle
};

bool lower(const OnnxGraph &g, Plan *plan, std::string *why);

// One line for the log, e.g. "18 kernels, 48 ch, pixel-unshuffle 2, depth-to-space 4".
std::string describe(const Plan &plan);

}  // namespace ajinative

#endif
