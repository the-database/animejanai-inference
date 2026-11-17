/*
 * native/rife_plan.h — RIFE v4 (IFNet, the vs-mlrt ONNX layout libaji ships) read from its ONNX:
 * the encoder and each block's convs, scale and wiring, every shape checked. Host only; the CUDA
 * (native/rife.cpp) and Vulkan (vk/rife.cpp) builders turn it into kernels. Anything else
 * (other layouts, ensembles with fed features) is refused with a reason.
 */
#ifndef AJI_NATIVE_RIFE_PLAN_H
#define AJI_NATIVE_RIFE_PLAN_H

#include <string>
#include <vector>

#include "native/onnx_graph.h"
#include "native/onnx_read.h"

namespace ajinative {

constexpr int kRifeMaxBlocks = 6;

struct RifeBlockPlan {
    int s = 1;                                // the block's scale (its input is downsampled by s)
    onnxread::Conv c0a, c0b;                  // conv0: two stride-2 convs, LeakyReLU
    std::vector<onnxread::Conv> res;          // convblock: lrelu(conv * beta + x)
    std::vector<std::vector<float>> beta;     // per res conv, cout values
    onnxread::Conv last;                      // lastconv: ConvTranspose 4x4/2, then DepthToSpace 2 (CRD)
    int K = 0;                                // output channels after the depth-to-space: flow 0-3, mask 4, features 5..K-1
    bool feeds = false;                       // the previous block's features are fed in
};

struct RifePlan {
    bool ens = false;                         // the input splits into [3, 3, 1, 2, 2] (an ensemble model)
    int F = 0;                                // encoder features per frame (0: no encoder)
    std::vector<onnxread::Conv> enc;          // the encoder's convs, the last a transposed conv
    std::vector<RifeBlockPlan> blocks;
};

// For W x H frames (multiples of 64). False and *why when the graph isn't one this reads.
bool rife_plan(const OnnxGraph &g, int W, int H, RifePlan *p, std::string *why);

}  // namespace ajinative

#endif
