// Count successful, model-owned cudaMalloc requests during a synchronous native load or probe.
#pragma once
#include <cuda_runtime.h>

namespace ajinative {
enum class AllocationKind { Weight, Activation, TileIO };
struct AllocationCount {
    size_t bytes[3] = {};
    // Test hook, null in production. Each successful tracked model allocation tests this pointer once.
    inline static thread_local AllocationCount *active = nullptr;
};
template<class T>
cudaError_t model_malloc(T **ptr, size_t bytes, AllocationKind kind)
{
    const cudaError_t e = cudaMalloc(ptr, bytes);
    if (e == cudaSuccess && AllocationCount::active)
        AllocationCount::active->bytes[static_cast<unsigned>(kind)] += bytes;
    return e;
}
} // namespace ajinative
