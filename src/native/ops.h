/*
 * native/ops.h — the fused-conv vocabulary of the native path: what a lowered conv step reads,
 * which activation or SPAN gate it applies, and what it writes. Host only (no CUDA), so the
 * lowering (native/lower) and every backend that runs a Plan share it; native/conv.h is the
 * CUDA kernels' API over these values. PlanStep stores them as ints with these values.
 */
#ifndef AJI_NATIVE_OPS_H
#define AJI_NATIVE_OPS_H

namespace ajinative {

// NHWC feature map, the model's NCHW RGB input, or that input pixel-unshuffled by 2.
enum class Input { NHWC = 0, NCHW3 = 1, NCHW3Unshuffle2 = 2 };
// After bias; LeakyReLU is a PReLU with a constant slope.
enum class Act { None = 0, Relu = 1, PRelu = 2, SiLU = 3 };
// SPAN gates with the block input x on the conv output o: x + o, (x + o) * tanh(W1x1 o + b),
// (x + o) * (sigmoid(o) - 0.5). A gate takes no activation.
enum class Gate { None = 0, Residual = 1, Tanh1x1 = 2, SigmoidHalf = 3 };
// NHWC, or depth-to-space by 2 / 4 / 1 / 3 into NCHW RGB (optionally adding the upsampled model
// input; 1: the conv's 3 channels are the RGB output as they are).
enum class Output { NHWC = 0, DepthToSpace2 = 1, DepthToSpace4 = 2, DepthToSpace1 = 3, DepthToSpace3 = 4 };

// The depth-to-space factor of an output mode (0 for NHWC), and the mode of a factor (NHWC if none).
constexpr int depth_to_space_factor(Output o)
{
    return o == Output::DepthToSpace2 ? 2 : o == Output::DepthToSpace4 ? 4 : o == Output::DepthToSpace1 ? 1 : o == Output::DepthToSpace3 ? 3 : 0;
}
constexpr Output depth_to_space_output(int r)
{
    return r == 2 ? Output::DepthToSpace2 : r == 4 ? Output::DepthToSpace4 : r == 1 ? Output::DepthToSpace1 : r == 3 ? Output::DepthToSpace3 : Output::NHWC;
}

}  // namespace ajinative

#endif
