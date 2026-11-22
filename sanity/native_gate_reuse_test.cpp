// Usage: aji_native_gate_reuse_test HD_BALANCED.onnx HD_PERFORMANCE.onnx ILLUSTRATION_SPAN_S.onnx [default|fp16|fp8]
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
#include "native/span_fp8.h"

using namespace ajinative;

static void check_cuda(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

static bool compare(const char *path, int channels, Gate gate, const std::string &mode, bool sm120)
{
    // Use more CTAs than can run at once so a later CTA can still need a neighbor's source
    // halo when that neighbor writes. Both dimensions also leave partial tiles after unshuffle.
    constexpr int w = 1922, h = 1082;
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
    const bool fp8 = mode == "fp8", tuned = mode != "default";
    const bool fuse = tuned && sm120 && channels == 48;
    auto reference = Executor::create(plan, w, h, nullptr, &why, false, fp8);
    auto candidate = Executor::create(plan, w, h, nullptr, &why, true, fp8);
    // Also compare the fused 48-channel paths to the separate convolution + concat kernels.
    auto unfused = fuse ? Executor::create(plan, w, h, nullptr, &why, false, fp8) : nullptr;
    if (!reference || !candidate || (fuse && !unfused)) {
        std::fprintf(stderr, "%s [%s]: %s\n", path, mode.c_str(), why.c_str());
        return false;
    }
    const int links = fp8 ? (channels == 48 ? 9 : 15) : 0;
    for (Executor *ex : {reference.get(), candidate.get(), unfused.get()}) {
        if (!ex) continue;
        const size_t slots = ex == candidate.get() ? 5 : 6;
        if (ex->feature_buffer_count() != slots || ex->uses_fp8() != fp8 || ex->packed_links() != links) {
            std::fprintf(stderr, "%s [%s]: expected %zu slots, %d packed links, got %zu, %d\n",
                         path, mode.c_str(), slots, links, ex->feature_buffer_count(), ex->packed_links());
            return false;
        }
        const size_t map_bytes = size_t(w / plan.unshuffle) * (h / plan.unshuffle) * channels * sizeof(uint16_t);
        if (gate == Gate::Tanh1x1 && ex->device_bytes() - ex->weight_bytes() != slots * map_bytes) {
            std::fprintf(stderr, "%s [%s]: unexpected feature buffer capacity\n", path, mode.c_str());
            return false;
        }
        if (tuned && ((!fp8 && !ex->enable_fp16_cache_order()) ||
                      (ex != unfused.get() && ex->enable_gate_projection() != fuse))) {
            std::fprintf(stderr, "%s [%s]: unexpected kernel selection\n", path, mode.c_str());
            return false;
        }
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
        for (Executor *other : {candidate.get(), unfused.get()}) {
            if (!other) continue;
            check_cuda(cudaMemset(out, 0xff, out_bytes));
            check_cuda(other->run(in, out, nullptr));
            check_cuda(cudaMemcpy(actual.data(), out, out_bytes, cudaMemcpyDeviceToHost));
            auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
            if (mismatch.first != expected.end()) {
                std::fprintf(stderr, "%s [%s/%s]: frame %d, FP16 sample %zu differs (%04x != %04x)\n",
                             path, mode.c_str(), other == candidate.get() ? "reuse" : "unfused", frame,
                             size_t(mismatch.first - expected.begin()), unsigned(*mismatch.first), unsigned(*mismatch.second));
                equal = false;
                break;
            }
        }
    }
    check_cuda(cudaFree(out));
    check_cuda(cudaFree(in));
    if (equal) std::printf("%s [%s]: %zu -> %zu slots, %d packed links, 2 full frames bit-exact%s\n",
                           path, mode.c_str(), reference->feature_buffer_count(), candidate->feature_buffer_count(),
                           links, fuse ? " (also unfused)" : "");
    return equal;
}

int main(int argc, char **argv)
{
    const std::string mode = argc == 5 ? argv[4] : "all";
    if ((argc != 4 && argc != 5) || (mode != "all" && mode != "default" && mode != "fp16" && mode != "fp8")) {
        std::fprintf(stderr, "usage: %s HD_BALANCED.onnx HD_PERFORMANCE.onnx ILLUSTRATION_SPAN_S.onnx [default|fp16|fp8]\n", argv[0]);
        return 2;
    }
    int device, major, minor;
    check_cuda(cudaGetDevice(&device));
    check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    const bool sm120 = major == 12 && minor == 0;
    for (const std::string profile : {"default", "fp16", "fp8"}) {
        if (mode != "all" && mode != profile) continue;
        if ((profile == "fp16" && !sm120) || (profile == "fp8" && !span_fp8_available())) {
            std::printf("[%s]: unsupported on SM%d%d\n", profile.c_str(), major, minor);
            if (mode != "all") return 1;
            continue;
        }
        if (!compare(argv[1], 64, Gate::Tanh1x1, profile, sm120) ||
            !compare(argv[2], 48, Gate::Tanh1x1, profile, sm120) ||
            (profile == "default" && !compare(argv[3], 48, Gate::SigmoidHalf, profile, sm120))) return 1;
    }
    return 0;
}
