// Usage: aji_native_gate_reuse_test HD_BALANCED.onnx HD_PERFORMANCE.onnx ILLUSTRATION_SPAN_S.onnx
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

#include "native/executor.h"
#include "native/pack.h"

using namespace ajinative;

static void check_cuda(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

static bool compare(const char *path, int channels, Gate gate)
{
    // Multiple CTAs and partial tiles in both dimensions, including after the HD models' unshuffle.
    constexpr int w = 130, h = 74;
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    OnnxGraph graph;
    Plan plan;
    std::string why;
    if (!parse_onnx(bytes.data(), bytes.size(), &graph, &why) || !lower(graph, &plan, &why, kernel_widths_cuda())) {
        std::fprintf(stderr, "%s: %s\n", path, why.c_str());
        return false;
    }
    if (std::none_of(plan.steps.begin(), plan.steps.end(), [&](const PlanStep &s) {
            return s.gate == (int)gate && s.cout == channels;
        })) {
        std::fprintf(stderr, "%s: expected a %d-channel gate of type %d\n", path, channels, (int)gate);
        return false;
    }
    auto reference = Executor::create(plan, w, h, nullptr, &why, false);
    auto candidate = Executor::create(plan, w, h, nullptr, &why);
    if (!reference || !candidate) {
        std::fprintf(stderr, "%s: %s\n", path, why.c_str());
        return false;
    }
    if (reference->feature_buffer_count() != 6 || candidate->feature_buffer_count() != 5) {
        std::fprintf(stderr, "%s: expected 6 -> 5 feature buffers, got %zu -> %zu\n", path,
                     reference->feature_buffer_count(), candidate->feature_buffer_count());
        return false;
    }

    std::vector<uint16_t> input(3 * w * h);
    std::vector<uint16_t> expected(3 * reference->out_w() * reference->out_h()), actual(expected.size());
    const size_t in_bytes = input.size() * sizeof(uint16_t), out_bytes = expected.size() * sizeof(uint16_t);
    void *in, *out;
    check_cuda(cudaMalloc(&in, in_bytes));
    check_cuda(cudaMalloc(&out, out_bytes));
    std::mt19937 rng(42);
    bool equal = true;
    for (int frame = 0; frame < 2 && equal; ++frame) {
        for (auto &v : input) v = float_to_half_bits((rng() % 1024) / 1023.f);
        check_cuda(cudaMemcpy(in, input.data(), in_bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemset(out, 0, out_bytes));
        check_cuda(reference->run(in, out, nullptr));
        check_cuda(cudaMemcpy(expected.data(), out, out_bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemset(out, 0xff, out_bytes));
        check_cuda(candidate->run(in, out, nullptr));
        check_cuda(cudaMemcpy(actual.data(), out, out_bytes, cudaMemcpyDeviceToHost));
        auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
        equal = mismatch.first == expected.end();
        if (!equal)
            std::fprintf(stderr, "%s: frame %d, FP16 sample %zu differs (%04x != %04x)\n", path, frame,
                         (size_t)(mismatch.first - expected.begin()), (unsigned)*mismatch.first, (unsigned)*mismatch.second);
    }
    check_cuda(cudaFree(out));
    check_cuda(cudaFree(in));
    if (equal) std::printf("%s: 6 -> 5 feature buffers, both FP16 outputs match\n", path);
    return equal;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s HD_BALANCED.onnx HD_PERFORMANCE.onnx ILLUSTRATION_SPAN_S.onnx\n", argv[0]);
        return 2;
    }
    return compare(argv[1], 64, Gate::Tanh1x1) && compare(argv[2], 48, Gate::Tanh1x1) &&
           compare(argv[3], 48, Gate::SigmoidHalf) ? 0 : 1;
}
