// native/fdat_kernels.h — FDAT's channel attention fold and AIM gates (internal).
#ifndef AJI_NATIVE_FDAT_KERNELS_H
#define AJI_NATIVE_FDAT_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

// stats: tchanstats' layout; attn [heads][hd][hd].
cudaError_t fdat_chan_soft(const float *stats, const float *temp, int heads, int hd, float eps, float *attn, cudaStream_t s);
// fold [C][C] = per-head attention mixing composed with the projection wp [C][C].
cudaError_t fdat_chan_fold(const float *attn, const float *wp, int heads, int hd, float *fold, cudaStream_t s);
// Channel gate of the spatial blocks: cm[cpad] from the channel sums of the conv branch.
cudaError_t fdat_aim_cg(const float *sums, int M, int C, int r, const float *w1, const float *w2, float *cm, int cpad, cudaStream_t s);
// Spatial gate of the channel blocks: x += attn + conv * sigmoid(attn . wsg).
cudaError_t fdat_aim_sg(const void *attn, const void *conv, const float *wsg, void *x, int st, int M, int C, cudaStream_t s);

// Channel gates in general: cm = sigmoid(W2 act(W1 (in_scale * sums / M) + b1) + b2) (biases may be null;
// act 0 GELU, 1 ReLU; W1 [r][C], W2 [C][r]), zeros through cpad.
cudaError_t chan_gate(const float *sums, int M, int C, int r, const float *w1, const float *b1, const float *w2, const float *b2, int act,
                      float in_scale, float *cm, int cpad, cudaStream_t s);
// Subtracts m[0..2] from channels 0..2 of an fp16 token map (stride st).
cudaError_t tok_sub3(void *x, int st, int M, float m0, float m1, float m2, cudaStream_t s);

}  // namespace ajinative

#endif
