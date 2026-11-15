// native/img_kernels.cu — see img_kernels.h.
#include "native/img_kernels.h"

#include <algorithm>

#include <cuda_fp16.h>

namespace ajinative {
namespace {

__device__ __forceinline__ float ld(const void *p, bool f32, size_t i)
{
    return f32 ? ((const float *)p)[i] : __half2float(((const half *)p)[i]);
}

__global__ void k_to_nhwc(const void *in, bool f32, int w, int h, int un, half *dst, int ds)
{
    const int ow = un ? w / 2 : w, oh = un ? h / 2 : h;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    const size_t plane = (size_t)w * h;
    half *d = dst + ((size_t)y * ow + x) * ds;
    __align__(16) half v[16];
#pragma unroll
    for (int k = 0; k < 16; k++) v[k] = __float2half(0.f);
    if (un) {
#pragma unroll
        for (int c = 0; c < 3; c++)
#pragma unroll
            for (int ry = 0; ry < 2; ry++)
#pragma unroll
                for (int rx = 0; rx < 2; rx++) v[c * 4 + ry * 2 + rx] = __float2half(ld(in, f32, c * plane + (size_t)(2 * y + ry) * w + 2 * x + rx));
    } else {
#pragma unroll
        for (int c = 0; c < 3; c++) v[c] = __float2half(ld(in, f32, c * plane + (size_t)y * w + x));
    }
    reinterpret_cast<uint4 *>(d)[0] = reinterpret_cast<uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(d)[1] = reinterpret_cast<uint4 *>(v)[1];
    for (int c = 16; c < ds; c += 8) *reinterpret_cast<uint4 *>(d + c) = make_uint4(0, 0, 0, 0);
}

__global__ void k_from_nhwc(const half *src, int ss, int w, int h, void *out, bool f32)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, pix = (size_t)y * w + x;
    const half *p = src + pix * ss;
#pragma unroll
    for (int c = 0; c < 3; c++) {
        if (f32) ((float *)out)[c * plane + pix] = __half2float(p[c]);
        else ((half *)out)[c * plane + pix] = p[c];
    }
}

__device__ __forceinline__ int reflect_idx(int i, int n)
{
    if (n == 1) return 0;
    const int period = 2 * (n - 1);
    i %= period;
    return i < n ? i : period - i;
}

__global__ void k_pad_reflect(const half *in, int w, int h, half *out, int pw, int ph)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const int sx = reflect_idx(x, w), sy = reflect_idx(y, h);
#pragma unroll
    for (int c = 0; c < 3; c++) out[(size_t)c * pw * ph + (size_t)y * pw + x] = in[(size_t)c * w * h + (size_t)sy * w + sx];
}

__global__ void k_cast(const void *in, bool in_f32, void *out, bool out_f32, size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = ld(in, in_f32, i);
        if (out_f32) ((float *)out)[i] = v;
        else ((half *)out)[i] = __float2half(v);
    }
}

}  // namespace

cudaError_t img_pad_reflect(const void *in, int w, int h, void *out, int pw, int ph, cudaStream_t s)
{
    if (w < 1 || h < 1 || pw < w || ph < h) return cudaErrorInvalidValue;
    const dim3 b(32, 8), g((pw + 31) / 32, (ph + 7) / 8);
    k_pad_reflect<<<g, b, 0, s>>>((const half *)in, w, h, (half *)out, pw, ph);
    return cudaGetLastError();
}

cudaError_t img_cast(const void *in, bool in_f32, void *out, bool out_f32, size_t n, cudaStream_t s)
{
    if (in_f32 == out_f32) return cudaMemcpyAsync(out, in, n * (in_f32 ? 4 : 2), cudaMemcpyDeviceToDevice, s);
    if (!n) return cudaSuccess;
    const unsigned blocks = (unsigned)std::min<size_t>((n + 255) / 256, 65535);
    k_cast<<<blocks, 256, 0, s>>>(in, in_f32, out, out_f32, n);
    return cudaGetLastError();
}

cudaError_t img_to_nhwc(const void *in, bool in_f32, int w, int h, bool unshuffle, void *dst, int dstride, cudaStream_t s)
{
    if (dstride < 16 || dstride % 8 || (unshuffle && (w % 2 || h % 2))) return cudaErrorInvalidValue;
    const int ow = unshuffle ? w / 2 : w, oh = unshuffle ? h / 2 : h;
    const dim3 b(16, 8), g((ow + 15) / 16, (oh + 7) / 8);
    k_to_nhwc<<<g, b, 0, s>>>(in, in_f32, w, h, unshuffle ? 1 : 0, (half *)dst, dstride);
    return cudaGetLastError();
}

cudaError_t img_from_nhwc(const void *src, int sstride, int w, int h, void *out, bool out_f32, cudaStream_t s)
{
    const dim3 b(16, 8), g((w + 15) / 16, (h + 7) / 8);
    k_from_nhwc<<<g, b, 0, s>>>((const half *)src, sstride, w, h, out, out_f32);
    return cudaGetLastError();
}

}  // namespace ajinative
