// Test-only replay for DAT channel statistics, depthwise channel sums and column sums.
#pragma once
#include <cuda_runtime.h>

namespace ajinative {
namespace {
__global__ void replay_tfm_partials(const float *parts, float *out, int blocks, int width)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < width)
        for (int i = 0; i < blocks; ++i) atomicAdd(out + c, parts[(size_t)i * width + c]);
}

struct OrderedTfmPartials {
    float *out, *parts = nullptr;
    int blocks, width;
    cudaStream_t stream;
    cudaError_t error = cudaSuccess;
    OrderedTfmPartials(float *dst, int n, int w, cudaStream_t s) : out(dst), blocks(n), width(w), stream(s)
    {
        if (out) error = cudaMalloc(&parts, (size_t)blocks * width * sizeof(float));
    }
    cudaError_t finish(cudaError_t e) const
    {
        if (e != cudaSuccess || !out) return e;
        replay_tfm_partials<<<(width + 127) / 128, 128, 0, stream>>>(parts, out, blocks, width);
        return cudaGetLastError();
    }
    ~OrderedTfmPartials() { if (parts) cudaFree(parts); }
};

// Shared-memory atomic sums use thread order inside a CTA. The final global replay uses CTA order.
template<int V>
__device__ void ordered_dw_sums(float *parts, float *shared, int c, int channels, bool live, const float (&values)[V])
{
    for (int lane = 0; lane < blockDim.x; ++lane) {
        if (threadIdx.x == lane && live)
            for (int j = 0; j < V && c + j < channels; ++j) atomicAdd(shared + c + j, values[j]);
        __syncthreads();
    }
    for (int k = threadIdx.x; k < channels; k += blockDim.x)
        parts[(size_t)blockIdx.x * channels + k] = shared[k];
}
} // namespace
} // namespace ajinative
