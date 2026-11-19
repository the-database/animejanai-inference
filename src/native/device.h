// native/device.h — per-device facts the kernels' launch choices depend on. Internal, host only.
#ifndef AJI_NATIVE_DEVICE_H
#define AJI_NATIVE_DEVICE_H

#include <atomic>

#include <cuda_runtime.h>

namespace ajinative {

// Whether the current device's L2 is 32 MB or more (RTX 4090 72 MB, 5090 96 MB; RTX A5000 / 30-series
// 6 MB). With a small L2, rows prefetched ahead of use are evicted again before they are read.
inline bool large_l2()
{
    constexpr int kMaxDev = 16;
    static std::atomic<int> state[kMaxDev] = {};  // 0 unknown, 1 small, 2 large
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDev) return false;
    int v = state[dev].load(std::memory_order_relaxed);
    if (!v) {
        int l2 = 0;
        if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev) != cudaSuccess) l2 = 0;
        v = l2 >= (32 << 20) ? 2 : 1;
        state[dev].store(v, std::memory_order_relaxed);
    }
    return v == 2;
}

// The current device's GPU generation, for the kernel choices measured per generation (gconv warps, conv3x3
// tiles): 0 Ampere (compute 8.0-8.7, and anything older), 1 Ada (8.9), 2 Blackwell (10.0 and up; RTX 50-series 12.0).
inline int gpu_gen()
{
    constexpr int kMaxDev = 16;
    static std::atomic<int> state[kMaxDev] = {};  // 0 unknown, else generation + 1
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDev) return 2;
    int v = state[dev].load(std::memory_order_relaxed);
    if (!v) {
        int ma = 0, mi = 0;
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
        v = 1 + (ma >= 10 ? 2 : ma == 8 && mi >= 9 ? 1 : 0);
        state[dev].store(v, std::memory_order_relaxed);
    }
    return v - 1;
}

}  // namespace ajinative

#endif
