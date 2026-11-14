// native/rife_kernels.h — the resampling kernels of native/rife (internal).
#ifndef AJI_NATIVE_RIFE_KERNELS_H
#define AJI_NATIVE_RIFE_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

constexpr int kRifeMaxBlocks = 6;

struct RifeFrame {
    const void *in11 = nullptr;  // NCHW fp16, frames at planes 0-2 and 3-5
    int W = 0, H = 0;            // full (padded) size
    const void *feat[2] = {};    // per-frame features NHWC [H][W][fstride], or null
    int F = 0, fstride = 0;
};

struct RifeBlockOut {
    const void *b = nullptr;     // NHWC [H/s][W/s][kstride]: flow 0-3, mask 4, features 5..K-1
    int s = 1, kstride = 0;
};

struct RifeGather {
    int nprev = 0;               // blocks before this one (0: block 0's input)
    RifeBlockOut prev[kRifeMaxBlocks];
    bool feeds_feat = false;     // the previous block's features (channels 5..K-1) are fed in
    int K = 0;                   // the previous block's output channels
    const float *t = nullptr;    // the timestep, in device memory (graph replay)
    int s = 1;                   // this block's scale
    void *dst = nullptr;         // NHWC [H/s][W/s][dstride]
    int dstride = 0, cin = 0;
};

bool rife_gather_supported(const RifeFrame &f, const RifeGather &g);
cudaError_t rife_gather(const RifeFrame &f, const RifeGather &g, cudaStream_t s);
// out: NCHW fp16 RGB, full size. Uses every block (flow) and the last one's mask.
cudaError_t rife_final(const RifeFrame &f, int nblocks, const RifeBlockOut *blocks, void *out, cudaStream_t s);

}  // namespace ajinative

#endif
