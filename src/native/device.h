// native/device.h — per-device facts the kernels' launch choices depend on. Internal, host only.
#ifndef AJI_NATIVE_DEVICE_H
#define AJI_NATIVE_DEVICE_H

#include <atomic>

#include <cuda_runtime.h>

namespace ajinative {

// Whether the current device's L2 is 32 MB or more (RTX 4090 72 MB, 5090 96 MB; RTX A5000 / 30-series
// 6 MB). With a small L2, prefetching rows ahead of use and tiles that re-read more halo from L2 cost:
// they win measured alone but lose a few percent in a whole network.
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

}  // namespace ajinative

#endif
