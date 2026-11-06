/*
 * core/tiling_gpu.cu — see tiling_gpu.h.
 *
 * Every multiply and add below uses the round-to-nearest intrinsics
 * (__fmul_rn / __fadd_rn), which nvcc never contracts into an FMA: the
 * host path (core/tiling.cpp, compiled without contraction) rounds each
 * operation separately, and matching it bit for bit needs the same.
 */

#include "tiling_gpu.h"

#include <cuda_fp16.h>

namespace ajicore {

void GpuTileScratch::release()
{
    cudaFree(img);
    cudaFree(band);
    cudaFree(rows);
    cudaFree(wx);
    cudaFree(wy);
    *this = GpuTileScratch{};
}

namespace {

__device__ __forceinline__ int d_reflect(int i, int n)
{
    if (n == 1)
        return 0;
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0)
        i += period;
    return i < n ? i : period - i;
}

// core/half.h's f32_to_bf16 (round to nearest even, NaN kept quiet).
__device__ __forceinline__ uint16_t d_f32_to_bf16(float f)
{
    uint32_t x = __float_as_uint(f);
    if ((x & 0x7fffffffu) > 0x7f800000u)
        return (uint16_t)((x >> 16) | 0x40u);
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

__device__ __forceinline__ void d_store(void *dst, size_t i, float v,
                                        GpuDtype dt)
{
    if (dt == GpuDtype::F32)
        ((float *)dst)[i] = v;
    else if (dt == GpuDtype::F16)
        ((__half *)dst)[i] = __float2half_rn(v);
    else
        ((uint16_t *)dst)[i] = d_f32_to_bf16(v);
}

// fp16 and bf16 widen to fp32 exactly.
__device__ __forceinline__ float d_load(const void *src, size_t i,
                                        GpuDtype dt)
{
    if (dt == GpuDtype::F32)
        return ((const float *)src)[i];
    if (dt == GpuDtype::F16)
        return __half2float(((const __half *)src)[i]);
    return __uint_as_float((uint32_t)((const uint16_t *)src)[i] << 16);
}

__global__ void k_extract(const uint8_t *img, int W, int H, int ci, int ox,
                          int oy, int tw, int th, int in_ch, float inv255,
                          void *dst, GpuDtype dt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tw || y >= th)
        return;
    const uint8_t *p = img + ((size_t)d_reflect(oy + y, H) * W +
                              d_reflect(ox + x, W)) * ci;
    const size_t plane = (size_t)tw * th, i = (size_t)y * tw + x;
    if (in_ch == 3) {
        const int g = ci == 1;
        d_store(dst, i, __fmul_rn((float)p[0], inv255), dt);
        d_store(dst, i + plane, __fmul_rn((float)p[g ? 0 : 1], inv255), dt);
        d_store(dst, i + 2 * plane, __fmul_rn((float)p[g ? 0 : 2], inv255),
                dt);
    } else if (ci == 1) {
        d_store(dst, i, __fmul_rn((float)p[0], inv255), dt);
    } else {
        const float l = __fadd_rn(__fadd_rn(__fmul_rn(0.299f, (float)p[0]),
                                            __fmul_rn(0.587f, (float)p[1])),
                                  __fmul_rn(0.114f, (float)p[2]));
        d_store(dst, i, __fmul_rn(l, inv255), dt);
    }
}

__global__ void k_blend(const void *tile, GpuDtype dt, int out_ch, int tow,
                        int toh, const float *wx, const float *wy, int cols,
                        int rows, int ux0, int uy0, float *band, int co,
                        int band_rows, int OW)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= cols || v >= rows)
        return;
    const size_t plane = (size_t)tow * toh;
    const float w = __fmul_rn(wy[v], wx[u]);
    const int r = (uy0 + v) % band_rows;
    for (int c = 0; c < co; c++) {
        const int mc = (out_ch == 1 || co == 1) ? 0 : c;
        const float s = d_load(tile, mc * plane + (size_t)v * tow + u, dt);
        float *b = band + ((size_t)c * band_rows + r) * OW + ux0 + u;
        *b = __fadd_rn(*b, __fmul_rn(w, s));
    }
}

__global__ void k_flush(float *band, int co, int band_rows, int OW, int y0,
                        int n, uint8_t *dst)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= OW || k >= n)
        return;
    const int r = (y0 + k) % band_rows;
    for (int c = 0; c < co; c++) {
        float *b = band + ((size_t)c * band_rows + r) * OW + X;
        const float f = __fadd_rn(__fmul_rn(*b, 255.0f), 0.5f);
        dst[((size_t)k * OW + X) * co + c] =
            f <= 0.0f ? 0 : f >= 255.0f ? 255 : (uint8_t)f;
        *b = 0.0f;
    }
}

dim3 grid_for(int w, int h, dim3 block)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

} // namespace

cudaError_t gpu_tile_extract(const uint8_t *img, int W, int H, int ci,
                             int ox, int oy, int tw, int th, int in_ch,
                             float inv255, void *dst, GpuDtype dt,
                             cudaStream_t s)
{
    const dim3 block(32, 8);
    k_extract<<<grid_for(tw, th, block), block, 0, s>>>(
        img, W, H, ci, ox, oy, tw, th, in_ch, inv255, dst, dt);
    return cudaGetLastError();
}

cudaError_t gpu_tile_blend(const void *tile, GpuDtype dt, int out_ch,
                           int tow, int toh, const float *wx, const float *wy,
                           int cols, int rows, int ux0, int uy0, float *band,
                           int co, int band_rows, int OW, cudaStream_t s)
{
    const dim3 block(32, 8);
    k_blend<<<grid_for(cols, rows, block), block, 0, s>>>(
        tile, dt, out_ch, tow, toh, wx, wy, cols, rows, ux0, uy0, band, co,
        band_rows, OW);
    return cudaGetLastError();
}

cudaError_t gpu_band_flush(float *band, int co, int band_rows, int OW,
                           int y0, int n, uint8_t *dst, cudaStream_t s)
{
    const dim3 block(64, 4);
    k_flush<<<grid_for(OW, n, block), block, 0, s>>>(band, co, band_rows,
                                                      OW, y0, n, dst);
    return cudaGetLastError();
}

} // namespace ajicore
