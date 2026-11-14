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

}  // namespace ajinative

#endif
