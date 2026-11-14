/*
 * native/rife.h — RIFE v4 (IFNet, the vs-mlrt ONNX layout libaji ships) on native kernels.
 *
 * The model is read from its ONNX by node names (/encode/..., /blockN/conv0/..., convblock.K,
 * lastconv) with every shape and wiring checked; anything else (ensemble models, other layouts)
 * is refused with a reason and runs on TensorRT.
 *
 * Input: libaji's 11-channel NCHW fp16 tensor (frame A in 0-2, frame B in 3-5; the timestep and
 * mesh planes are not read: t is passed in, the mesh is recomputed with the same fp16 rounding).
 * Output: NCHW fp16 RGB, the same size. Width and height must be multiples of 64 (libaji pads).
 *
 * Unlike the graph, no block works at full resolution unless its scale is 1: a block's input is
 * the bilinear downsample of [warped frames, warped features, timestep, mask, features, flow],
 * and the downsample of a warp is the warp evaluated at the downsample's sample points, with the
 * full-resolution flow (the sum of every earlier block's upsampled output) evaluated at those
 * same points (rife_kernels.cu, gather).
 */
#ifndef AJI_NATIVE_RIFE_H
#define AJI_NATIVE_RIFE_H

#include <memory>
#include <string>

#include <cuda_runtime.h>

namespace ajinative {

struct OnnxGraph;

class Rife {
public:
    // Parses and uploads; why explains a refusal. Builds buffers for w x h.
    static std::unique_ptr<Rife> create(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why);
    ~Rife();
    // in11: the NCHW fp16 input tensor; out3: NCHW fp16 RGB. t: the timestep.
    cudaError_t run(const void *in11, float t, void *out3, cudaStream_t s);
    std::string describe() const;  // e.g. "5 blocks, scales 16/8/4/2/1, 4 feature channels"
    size_t device_bytes() const;

    struct Impl;

private:
    Rife() = default;
    std::unique_ptr<Impl> d;
};

}  // namespace ajinative

#endif
