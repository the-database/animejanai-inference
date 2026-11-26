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

#include "native/device.h"
#include "native/executor.h"
#include "native/framenet.h"
#include "native/pack.h"
#include "native/span_fp8.h"

using namespace ajinative;

static_assert(hd_span_tuned_sm(8, 9) && hd_span_tuned_sm(12, 0));
static_assert(!hd_span_tuned_sm(8, 0) && !hd_span_tuned_sm(8, 6) && !hd_span_tuned_sm(9, 0));

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

// The first two HD body convolutions are identical ReLUs apart from their launch policy.
// Their pixels match under either policy, so inspect the actual captured kernels.
static bool check_body_launches(FrameNet &video, void *in, void *out, int feature_h, bool fp8, bool tuned)
{
    cudaStream_t stream{};
    cudaGraph_t graph{};
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check_cuda(video.run(in, out, stream));
    check_cuda(cudaStreamEndCapture(stream, &graph));
    cudaGraphNode_t node{};
    size_t count = 1;
    check_cuda(cudaGraphGetRootNodes(graph, &node, &count));
    cudaKernelNodeParams body[2]{};
    for (auto &launch : body) {
        if (count != 1) break;
        cudaGraphNode_t next{};
#if CUDART_VERSION >= 13000
        check_cuda(cudaGraphNodeGetDependentNodes(node, &next, nullptr, &count));
#else
        check_cuda(cudaGraphNodeGetDependentNodes(node, &next, &count));
#endif
        if (count == 1) check_cuda(cudaGraphKernelNodeGetParams(next, &launch));
        node = next;
    }
    check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(stream));
    if (count != 1) {
        std::fprintf(stderr, "Missing HD body convolution in captured graph\n");
        return false;
    }
    if (!fp8) {
        // Reversed FP16 rows use a different kernel specialization from ordinary rows.
        const bool alternating = body[0].func != body[1].func;
        if (alternating != tuned) std::fprintf(stderr, "Unexpected FP16 body row order\n");
        return alternating == tuned;
    }
    // Both write packed output. Only the second also reads packed input.
    const unsigned tile_rows = tuned ? 16 : 8;
    const unsigned grid_y = (feature_h + tile_rows - 1) / tile_rows;
    for (int i = 0; i < 2; ++i) {
        const unsigned threads = tuned && i == 1 ? 256 : 128;
        if (body[i].gridDim.y != grid_y || body[i].blockDim.x != threads) {
            std::fprintf(stderr, "FP8 body %d: grid.y %u, threads %u, expected %u and %u\n",
                         i + 1, body[i].gridDim.y, body[i].blockDim.x, grid_y, threads);
            return false;
        }
    }
    return true;
}

static bool compare(const char *path, int channels, Gate gate, const std::string &mode, bool expect_tuning)
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
    const bool fp8 = mode == "fp8";
    const bool hd = gate == Gate::Tanh1x1;
    const bool fuse = hd && expect_tuning;
    const SpanPrecision precision = fp8 ? SpanPrecision::FP8 :
                                    mode == "fp16" ? SpanPrecision::FP16 : SpanPrecision::Default;
    auto video = FrameNet::lowered(plan, w, h, nullptr, &why, precision);
    if (!video) { std::fprintf(stderr, "%s: %s\n", path, why.c_str()); return false; }
    const size_t operations = hd ? (channels == 64 ? 27 : 18) : plan.steps.size();
    const std::string label = video->describe();
    if (label.find(std::to_string(operations - size_t(fuse)) + " kernels, ") != 0 ||
        (label.find("fused gate and projection") != std::string::npos) != fuse ||
        (label.find("FP16 alternating tile rows") != std::string::npos) != (fuse && !fp8) ||
        (label.find("FP8 SPANF3 body") != std::string::npos) != fp8) {
        std::fprintf(stderr, "%s [%s]: unexpected video dispatch: %s\n", path, mode.c_str(), label.c_str());
        return false;
    }
    auto reference = Executor::create(plan, w, h, nullptr, &why, GateBuffers::Separate, fp8);
    auto candidate = Executor::create(plan, w, h, nullptr, &why, GateBuffers::Reuse, fp8);
    // Also compare the fused HD paths to the separate convolution + concat kernels.
    auto unfused = fuse ? Executor::create(plan, w, h, nullptr, &why, GateBuffers::Separate, fp8) : nullptr;
    auto stock = fp8 ? Executor::create(plan, w, h, nullptr, &why, GateBuffers::Separate, false) : nullptr;
    if (!reference || !candidate || (fuse && !unfused) || (fp8 && !stock)) {
        std::fprintf(stderr, "%s [%s]: %s\n", path, mode.c_str(), why.c_str());
        return false;
    }
    // The production 64-channel SM86 video path has one more full FP16 map than the in-place executor.
    const bool separate_video = hd && channels == 64 && current_cuda_sm() == 86;
    if (video->device_bytes() != (separate_video ? reference : candidate)->device_bytes()) {
        std::fprintf(stderr, "%s [%s]: unexpected video feature allocation\n", path, mode.c_str());
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
        const size_t feature_bytes = slots * map_bytes - (fp8 && ex == candidate.get() ? map_bytes / 2 : 0);
        if (gate == Gate::Tanh1x1 && ex->device_bytes() - ex->weight_bytes() != feature_bytes) {
            std::fprintf(stderr, "%s [%s]: unexpected feature buffer capacity\n", path, mode.c_str());
            return false;
        }
        // Keep the full-frame FP16 reference on ordinary kernels and row order.
        // FP8 selects its body tiles in create() and retains the fused reference.
        if (!fp8 && ex == reference.get()) continue;
        if (hd && ((!fp8 && ex->enable_fp16_cache_order() != expect_tuning) ||
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
    auto read_output = [&](auto &ex, std::vector<uint16_t> &samples, int fill) {
        check_cuda(cudaMemset(out, fill, out_bytes));
        check_cuda(ex.run(in, out, nullptr));
        check_cuda(cudaMemcpy(samples.data(), out, out_bytes, cudaMemcpyDeviceToHost));
    };
    std::mt19937 rng(42);
    bool passed = !hd || check_body_launches(*video, in, out, h / plan.unshuffle, fp8, expect_tuning);
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
        read_output(*video, actual, 0x5a);
        if (!bit_exact("video profile")) passed = false;
        if (unfused) {
            read_output(*unfused, actual, 0xff);
            if (!bit_exact("unfused")) passed = false;
        }
    }
    if (gate == Gate::Tanh1x1) {
        // The image tiler passes the active geometry to an executor created for a larger tile.
        // A separately sized, unfused reference cannot share a stale maximum-width stride.
        constexpr int tile_w = 322, tile_h = 182;
        auto tile_ref = Executor::create(plan, tile_w, tile_h, nullptr, &why, GateBuffers::Separate, fp8);
        if (!tile_ref) {
            std::fprintf(stderr, "tile reference: %s\n", why.c_str());
            passed = false;
        } else {
            const size_t n = size_t(3) * tile_ref->out_w() * tile_ref->out_h();
            check_cuda(tile_ref->run(in, out, nullptr));
            check_cuda(cudaMemcpy(expected.data(), out, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            check_cuda(cudaMemset(out, 0xa5, out_bytes));
            check_cuda(candidate->run(in, out, tile_w, tile_h, nullptr));
            check_cuda(cudaMemcpy(actual.data(), out, (n + 1) * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            if (!std::equal(expected.begin(), expected.begin() + n, actual.begin()) || actual[n] != 0xa5a5) {
                std::fprintf(stderr, "%s [%s]: 322x182 tile output or trailing guard differs\n", path, mode.c_str());
                passed = false;
            }
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
    const bool expect_tuning = hd_span_tuned_sm(major, minor);
    for (const std::string profile : {"default", "fp16", "fp8"}) {
        if (mode != "all" && mode != profile) continue;
        if (profile == "fp8" && !span_fp8_available()) {
            std::fprintf(stderr, "[%s]: FP8 kernel unavailable on SM%d%d\n", profile.c_str(), major, minor);
            if (expect_tuning || mode != "all") return 1;
            continue;
        }
        const bool balanced = compare(argv[1], 64, Gate::Tanh1x1, profile, expect_tuning);
        const bool performance = compare(argv[2], 48, Gate::Tanh1x1, profile, expect_tuning);
        if (!balanced || !performance || (profile == "default" && !compare(argv[3], 48, Gate::SigmoidHalf, profile, expect_tuning))) return 1;
    }
    return 0;
}
