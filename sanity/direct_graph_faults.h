// Test-only forced include for a dedicated backend build. Never ship this DLL.
// The wrappers produce real CUDA errors; production sources stay unchanged.
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

namespace aji_graph_test {
inline int begins = 0, failures = 0;
inline bool fail(const char *point) {
    const char *requested = std::getenv("AJI_TEST_DIRECT_GRAPH_FAULT");
    if (!failures && requested && !std::strcmp(requested, point)) {
        ++failures;
        return true;
    }
    return false;
}
inline cudaError_t CUDARTAPI begin(cudaStream_t stream, cudaStreamCaptureMode mode) {
    ++begins;
    if (fail("begin")) return cudaGetDevice(nullptr);
    const auto result = cudaStreamBeginCapture(stream, mode);
    if (result == cudaSuccess && fail("body")) cudaStreamSynchronize(stream);
    return result;
}
inline cudaError_t CUDARTAPI end(cudaStream_t stream, cudaGraph_t *graph) {
    if (fail("end")) cudaStreamSynchronize(stream); // invalidate capture
    return cudaStreamEndCapture(stream, graph); // always end it
}
inline cudaError_t CUDARTAPI instantiate(cudaGraphExec_t *exec, cudaGraph_t graph,
                                        unsigned long long flags) {
    if (fail("instantiate")) return cudaGetDevice(nullptr);
    return cudaGraphInstantiate(exec, graph, flags);
}
inline cudaError_t CUDARTAPI sync(cudaStream_t stream) {
    if (fail("warmup")) return cudaGetDevice(nullptr);
    return cudaStreamSynchronize(stream);
}
inline cudaError_t CUDARTAPI launch(cudaGraphExec_t exec, cudaStream_t stream) {
    if (fail("launch")) return cudaGetDevice(nullptr);
    return cudaGraphLaunch(exec, stream);
}
}
#ifdef _WIN32
#define AJI_GRAPH_TEST_EXPORT __declspec(dllexport)
#else
#define AJI_GRAPH_TEST_EXPORT __attribute__((visibility("default")))
#endif
extern "C" AJI_GRAPH_TEST_EXPORT inline int aji_test_graph_begins() { return aji_graph_test::begins; }
extern "C" AJI_GRAPH_TEST_EXPORT inline int aji_test_graph_failures() { return aji_graph_test::failures; }
extern "C" AJI_GRAPH_TEST_EXPORT inline int aji_test_graph_last_error() { return cudaGetLastError(); }
#define cudaStreamBeginCapture aji_graph_test::begin
#define cudaStreamEndCapture aji_graph_test::end
#define cudaGraphInstantiate aji_graph_test::instantiate
#define cudaStreamSynchronize aji_graph_test::sync
#define cudaGraphLaunch aji_graph_test::launch
