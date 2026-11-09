#include "aji_cuda_init.h"
#include <cstdio>

// Include CUDA registration like the production kernel modules.
__global__ void cuda_init_marker() {}

extern "C" __declspec(dllexport) int probe()
{
    cudaError_t result = aji_initialize_cuda_runtime();
    int device = -1;
    if (result == cudaSuccess) result = cudaGetDevice(&device);
    if (result != cudaSuccess || device < 0) {
        std::fprintf(stderr, "CUDA initialization failed: error=%d device=%d\n", (int)result, device);
        return 1;
    }
    return 0;
}
