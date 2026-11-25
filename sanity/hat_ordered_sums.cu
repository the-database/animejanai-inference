// HAT-L test executable only. Reconstruct each original CAB CTA's sum from the FP16 values
// that its epilogue stored. Both GConv layouts accumulate rows, then columns, in this order.
#include "native/gconv.h"
#include "native/gconv_workspace.h"
#include <cuda_fp16.h>

namespace ajinative {
namespace {
__global__ void partials(const half *map, int w, int h, int stride, int channels, int rows, float *parts)
{
    const int c = threadIdx.x + blockDim.x * blockIdx.z;
    if (c >= channels) return;
    const int x0 = blockIdx.x * gconv_tile_width, y0 = blockIdx.y * rows;
    float sum = 0.f;
    for (int y = y0; y < y0 + rows && y < h; ++y)
        for (int x = x0; x < x0 + gconv_tile_width && x < w; ++x)
            sum = __fadd_rn(sum, __half2float(map[((size_t)y * w + x) * stride + c]));
    parts[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * channels + c] = sum;
}

__global__ void replay(const float *parts, int tiles, int channels, float *out)
{
    const int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (c >= channels) return;
    // One writer per channel, in ascending CTA order. Use the same global atomic
    // operation as stock, including its FP32 subnormal handling.
    out[c] = 0.f;
    for (int i = 0; i < tiles; ++i) atomicAdd(out + c, parts[(size_t)i * channels + c]);
}
} // namespace

cudaError_t test_ordered_hat_sums(const GConvArgs &a, int channels, int rows, cudaStream_t s)
{
    const int nx = (a.W + gconv_tile_width - 1) / gconv_tile_width, ny = (a.H + rows - 1) / rows;
    float *parts = nullptr;
    auto e = cudaMalloc(&parts, (size_t)nx * ny * channels * sizeof(float));
    if (e != cudaSuccess) return e;
    partials<<<dim3(nx, ny, (channels + 127) / 128), 128, 0, s>>>
        ((const half *)a.dst, a.W, a.H, a.dst_stride, channels, rows, parts);
    e = cudaGetLastError();
    if (e == cudaSuccess) {
        replay<<<(channels + 127) / 128, 128, 0, s>>>(parts, nx * ny, channels, a.sums);
        e = cudaGetLastError();
    }
    const auto freed = cudaFree(parts);
    return e != cudaSuccess ? e : freed;
}
} // namespace ajinative
