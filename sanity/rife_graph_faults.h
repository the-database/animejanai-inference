// Test-only forced include for a dedicated backend build. Never ship this DLL.
// Uses real CUDA errors without modifying production code or the CUDA runtime.
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

namespace aji_graph_test {
inline int begins = 0;
inline int failures = 0;
inline bool fail(const char *point) {
    const char *requested = std::getenv("AJI_TEST_RIFE_GRAPH_FAULT");
    if (!failures && requested && !std::strcmp(requested, point)) {
        ++failures;
        return true;
    }
    return false;
}
inline cudaError_t CUDARTAPI begin(cudaStream_t stream, cudaStreamCaptureMode mode) {
    ++begins;
    if (fail("begin"))
        return cudaGetDevice(nullptr); // invalid argument; sets the CUDA last error
    cudaError_t result = cudaStreamBeginCapture(stream, mode);
    if (result == cudaSuccess && fail("body"))
        cudaStreamSynchronize(stream); // next captured operation must fail
    return result;
}
inline cudaError_t CUDARTAPI end(cudaStream_t stream, cudaGraph_t *graph) {
    if (fail("end"))
        cudaStreamSynchronize(stream); // illegal during capture; invalidates it
    return cudaStreamEndCapture(stream, graph); // always terminate capture
}
inline cudaError_t CUDARTAPI instantiate(cudaGraphExec_t *exec, cudaGraph_t graph,
                                        unsigned long long flags) {
    if (fail("instantiate"))
        return cudaGetDevice(nullptr);
    return cudaGraphInstantiate(exec, graph, flags);
}
inline cudaError_t CUDARTAPI launch(cudaGraphExec_t exec, cudaStream_t stream) {
    if (fail("launch"))
        return cudaGetDevice(nullptr);
    return cudaGraphLaunch(exec, stream);
}
}

#ifdef _WIN32
#define AJI_GRAPH_TEST_EXPORT __declspec(dllexport)
#else
#define AJI_GRAPH_TEST_EXPORT __attribute__((visibility("default")))
#endif
extern "C" AJI_GRAPH_TEST_EXPORT inline int aji_test_graph_begins() {
    return aji_graph_test::begins;
}
extern "C" AJI_GRAPH_TEST_EXPORT inline int aji_test_graph_failures() {
    return aji_graph_test::failures;
}
// Query this backend's CUDA runtime (it may be statically linked).
extern "C" AJI_GRAPH_TEST_EXPORT inline int aji_test_graph_last_error() {
    return static_cast<int>(cudaGetLastError());
}
#define cudaStreamBeginCapture aji_graph_test::begin
#define cudaStreamEndCapture aji_graph_test::end
#define cudaGraphInstantiate aji_graph_test::instantiate
#define cudaGraphLaunch aji_graph_test::launch
