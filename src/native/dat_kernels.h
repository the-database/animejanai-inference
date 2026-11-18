// native/dat_kernels.h — DAT's adaptive interaction gates (internal).
#ifndef AJI_NATIVE_DAT_KERNELS_H
#define AJI_NATIVE_DAT_KERNELS_H

#include <cuda_runtime.h>

namespace ajinative {

// Per token: g[t] = sigmoid(w2 . gelu(W1 x[t] + b1) + b2) (x: fp16 [M][st], C channels; W1 [r][C], r <= 16).
cudaError_t dat_token_gate(const void *x, int st, int M, int C, int r, const float *w1, const float *b1, const float *w2, float b2,
                           float *g, cudaStream_t s);
// z = a * cg[c] + b * tg[t] (mode 0) or a * tg[t] + b * cg[c] (mode 1), zero through the padding to 16.
// dat_token_gate (gate source: a for mode 0, b for mode 1) and dat_mix in one pass, a thread per token
// (no tg written). cudaErrorNotSupported for strides / alignments it doesn't take: run the two then.
cudaError_t dat_gate_mix(const void *a, const void *b, int st, const float *cg, int M, int C, int r, const float *w1, const float *b1,
                         const float *w2, float b2, int mode, void *z, cudaStream_t s);
cudaError_t dat_mix(const void *a, const void *b, int st, const float *cg, const float *tg, int mode, void *z, int M, int C,
                    cudaStream_t s);

}  // namespace ajinative

#endif
