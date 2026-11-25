// Usage: aji_native_gate_reuse_test HD_BALANCED.onnx HD_PERFORMANCE.onnx ILLUSTRATION_SPAN_S.onnx [default|fp16|fp8]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
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

// A dark color ramp above a bright screentone, with thin diagonal outlines across both halves.
static void gradient_texture(std::vector<uint16_t> &input, int w, int h)
{
    const size_t plane = size_t(w) * h;
    for (int y = 0; y < h; ++y) {
        const float fy = float(y) / (h - 1);
        for (int x = 0; x < w; ++x) {
            const float fx = float(x) / (w - 1);
            const float shade = 0.035f + 0.085f * fx + 0.025f * fy;
            const float tone = ((x / 4 + y / 6) % 2) ? 0.70f : 0.23f;
            const float fill = y < h / 2 ? shade : tone + 0.10f * fx + 0.05f * fy;
            const float luma = (x + 2 * y) % 191 < 2 ? 0.025f : fill;
            const size_t i = size_t(y) * w + x;
            input[i] = float_to_half_bits(luma + 0.06f * fx);
            input[plane + i] = float_to_half_bits(luma);
            input[2 * plane + i] = float_to_half_bits(luma + 0.04f * (1.f - fx));
        }
    }
}

static bool check_fidelity(const std::vector<uint16_t> &fp16, const std::vector<uint16_t> &fp8,
                           size_t pixels_per_channel, const char *model, const char *fixture, double floor_db)
{
    double squared_error = 0.0;
    const size_t plane = fp16.size() / 3;
    for (size_t c = 0; c < 3; ++c) {
        const size_t start = c * plane;
        for (size_t i = start; i < start + pixels_per_channel; ++i) {
            const double delta = double(half_bits_to_float(fp16[i])) - half_bits_to_float(fp8[i]);
            squared_error += delta * delta;
        }
    }
    // Full-scale RGB peak is 1. Accumulate all channels at model output, before video conversion.
    const double psnr = squared_error == 0.0 ? std::numeric_limits<double>::infinity()
                                            : 10.0 * std::log10(3.0 * pixels_per_channel / squared_error);
    const bool pass = psnr >= floor_db;  // NaN from a non-finite output also fails.
    std::fprintf(pass ? stdout : stderr, "%s [FP8 vs FP16, %s]: %.3f dB (floor %.1f dB) %s\n",
                 model, fixture, psnr, floor_db, pass ? "PASS" : "FAIL");
    return pass;
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
    const bool fuse = tuned && sm120 && gate == Gate::Tanh1x1;
    auto reference = Executor::create(plan, w, h, nullptr, &why, false, fp8);
    auto candidate = Executor::create(plan, w, h, nullptr, &why, true, fp8);
    // Also compare the fused HD paths to the separate convolution + concat kernels.
    auto unfused = fuse ? Executor::create(plan, w, h, nullptr, &why, false, fp8) : nullptr;
    auto stock = fp8 ? Executor::create(plan, w, h, nullptr, &why, false, false) : nullptr;
    if (!reference || !candidate || (fuse && !unfused) || (fp8 && !stock)) {
        std::fprintf(stderr, "%s [%s]: %s\n", path, mode.c_str(), why.c_str());
        return false;
    }
    const int links = fp8 ? (channels == 48 ? 9 : 15) : 0;
    for (Executor *ex : {reference.get(), candidate.get(), unfused.get()}) {
        if (!ex) continue;
        const size_t slots = ex == candidate.get() ? 5 : 6;
        if (ex->feature_buffer_count() != slots || ex->uses_fp8() != fp8) {
            std::fprintf(stderr, "%s [%s]: unexpected precision or feature buffer count (got %zu slots)\n",
                         path, mode.c_str(), ex->feature_buffer_count());
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
    const size_t output_plane = size_t(reference->out_w()) * reference->out_h();
    std::vector<uint16_t> expected(3 * output_plane), actual(expected.size());
    const size_t in_bytes = input.size() * sizeof(uint16_t), out_bytes = expected.size() * sizeof(uint16_t);
    void *in, *out;
    check_cuda(cudaMalloc(&in, in_bytes));
    check_cuda(cudaMalloc(&out, out_bytes));
    auto read_output = [&](Executor &ex, std::vector<uint16_t> &samples, int fill) {
        check_cuda(cudaMemset(out, fill, out_bytes));
        check_cuda(ex.run(in, out, nullptr));
        check_cuda(cudaMemcpy(samples.data(), out, out_bytes, cudaMemcpyDeviceToHost));
    };
    std::mt19937 rng(42);
    bool passed = true;
    const int frames = fp8 ? 3 : 2;
    for (int frame = 0; frame < frames; ++frame) {
        const bool gradient = fp8 && frame == 0;
        if (gradient) {
            gradient_texture(input, w, h);
        } else {
            for (auto &v : input) v = float_to_half_bits((rng() % 1024) / 1023.f);
        }
        check_cuda(cudaMemcpy(in, input.data(), in_bytes, cudaMemcpyHostToDevice));
        read_output(*candidate, actual, 0xff);
        if (stock) {
            read_output(*stock, expected, 0);
            // About 3 dB below the RTX 5090 results for each HD model and fixture.
            const double gradient_floor = channels == 48 ? 40.0 : 42.0;
            const double dark_floor = channels == 48 ? 52.0 : 54.0;
            const double noise_floor = channels == 48 ? 29.0 : 31.0;
            const bool full = check_fidelity(expected, actual, output_plane, path,
                                             gradient ? "gradient + screentone" : "noise", gradient ? gradient_floor : noise_floor);
            // Bright texture dominates the frame's squared error. Check the dark shading separately.
            const bool dark = !gradient || check_fidelity(expected, actual, output_plane / 2, path, "dark upper half", dark_floor);
            if (!full || !dark) {
                passed = false;
                continue;
            }
        }
        read_output(*reference, expected, 0);
        auto bit_exact = [&](const char *variant) {
            auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
            if (mismatch.first == expected.end()) return true;
            std::fprintf(stderr, "%s [%s/%s]: frame %d, FP16 sample %zu differs (%04x != %04x)\n",
                         path, mode.c_str(), variant, frame, size_t(mismatch.first - expected.begin()),
                         unsigned(*mismatch.first), unsigned(*mismatch.second));
            return false;
        };
        if (!bit_exact("reuse")) passed = false;
        if (unfused) {
            read_output(*unfused, actual, 0xff);
            if (!bit_exact("unfused")) passed = false;
        }
    }
    check_cuda(cudaFree(out));
    check_cuda(cudaFree(in));
    if (!passed) return false;
    for (Executor *ex : {reference.get(), candidate.get(), unfused.get()}) {
        if (ex && ex->packed_links() != links) {
            std::fprintf(stderr, "%s [%s]: expected %d packed links, got %d\n", path, mode.c_str(), links, ex->packed_links());
            return false;
        }
    }
    std::printf("%s [%s]: %zu -> %zu slots, %d packed links, %d full frames bit-exact%s\n",
                path, mode.c_str(), reference->feature_buffer_count(), candidate->feature_buffer_count(),
                links, frames, fuse ? " (also unfused)" : "");
    return true;
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
        const bool balanced = compare(argv[1], 64, Gate::Tanh1x1, profile, sm120);
        const bool performance = compare(argv[2], 48, Gate::Tanh1x1, profile, sm120);
        if (!balanced || !performance || (profile == "default" && !compare(argv[3], 48, Gate::SigmoidHalf, profile, sm120))) return 1;
    }
    return 0;
}
