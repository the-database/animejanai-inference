#include "aji_image.h"
#include "core/fileio.h"
#include "native/allocation_count.h"
#include "native/imgnet.h"
#include "native/lifetime_check.h"
#include "native/onnx_graph.h"
#include "native/pack.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace ajinative;

static void require(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}
static void gpu(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }

struct Device {
    uint16_t *p = nullptr;
    explicit Device(size_t elements) { gpu(cudaMalloc(&p, elements * sizeof(*p))); }
    ~Device() { cudaFree(p); }
};

static void stock(LifetimePlan &plan)
{
    for (auto &b : plan.buffers)
        if (b.name && (!std::strcmp(b.name, "D1") || !std::strcmp(b.name, "D2"))) b.live = {{2,4}};
}

using Shape = std::pair<int, int>;
using Output = std::vector<uint16_t>;
static std::vector<Output> render(const OnnxGraph &graph, Shape built, const std::vector<Shape> &shapes,
                                  int tail_rows, bool reference)
{
    LifetimeCheck check;
    check.edit = reference ? stock : nullptr;
    check.poison = !reference;
    LifetimeCheck::active = &check;
    std::string why;
    auto net = ImgNet::create(graph, built.first, built.second, nullptr, &why, tail_rows);
    LifetimeCheck::active = nullptr;
    require(net && net->scale() == 2 && net->tail_rows() == tail_rows, why + " (ESRGAN 2x / tail)");
    std::vector<Output> results;
    for (auto [w, h] : shapes) {
        const size_t area = size_t(w) * h, plane = 4 * area;
        Output input(3 * area), result(3 * plane);
        uint32_t state = 167;
        for (auto &pixel : input) {
            state = 1664525u * state + 1013904223u;
            pixel = float_to_half_bits(float(state >> 16) / 65535.0f);
        }
        Device in(input.size()), out(tail_rows ? size_t(12) * w * tail_rows : result.size());
        gpu(cudaMemcpy(in.p, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        if (tail_rows) {
            gpu(net->body(in.p, false, w, h, nullptr));
            for (int r = 0; r < h; r += tail_rows) {
                const int end = std::min(r + tail_rows, h);
                const size_t owned = size_t(4) * w * (end - r);
                gpu(net->tail(r, end, out.p, false, nullptr));
                for (int c = 0; c < 3; ++c)
                    gpu(cudaMemcpy(result.data() + c * plane + size_t(4) * w * r, out.p + c * owned,
                                   owned * 2, cudaMemcpyDeviceToHost));
            }
        } else {
            gpu(net->run(in.p, false, out.p, false, w, h, nullptr));
            gpu(cudaMemcpy(result.data(), out.p, result.size() * 2, cudaMemcpyDeviceToHost));
        }
        require(check.error.empty(), check.error);
        results.push_back(std::move(result));
    }
    return results;
}

static void exact(const OnnxGraph &graph, Shape built, const std::vector<Shape> &shapes, int strip)
{
    const auto baseline = render(graph, built, shapes, strip, true);
    const auto candidate = render(graph, built, shapes, strip, false);
    for (size_t j = 0; j < shapes.size(); ++j) {
        const auto &a = baseline[j], &b = candidate[j];
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i] || (b[i] & 0x7c00) == 0x7c00) {
                std::fprintf(stderr, "%dx%d built %dx%d tail %d element %zu: stock %04x, poisoned %04x\n",
                             shapes[j].first, shapes[j].second, built.first, built.second, strip, i, a[i], b[i]);
                throw std::runtime_error("FP16 output mismatch or nonfinite value");
            }
        std::printf("%dx%d built %dx%d tail %d: %zu FP16 values exact\n", shapes[j].first, shapes[j].second,
                    built.first, built.second, strip, a.size());
    }
}

static uint64_t account(aji_img_ctx *ctx, const std::string &path, int side, bool reference)
{
    const aji_img_model_params params{path.c_str(),side,side,32,AJI_IMG_LOAD_EVEN_TILES};
    aji_img_memory estimate{}, live{};
    LifetimeCheck probe_check, load_check;
    probe_check.edit = load_check.edit = reference ? stock : nullptr;
    LifetimeCheck::active = &probe_check;
    const int rc = aji_img_native_probe(ctx, &params, &estimate);
    LifetimeCheck::active = nullptr;
    require(rc == AJI_IMG_OK, aji_img_last_error(ctx));

    AllocationCount actual;
    AllocationCount::active = &actual;
    LifetimeCheck::active = &load_check;
    auto *model = aji_img_load(ctx, &params);
    LifetimeCheck::active = nullptr;
    AllocationCount::active = nullptr;
    require(model, aji_img_last_error(ctx));
    const int status = aji_img_memory_get(ctx, model, &live);
    aji_img_release(&model);
    require(status == AJI_IMG_OK && !live.activations_shared && live.model_activations, "expected a native model");
    const std::array<uint64_t,3> reported{estimate.model_weights,estimate.model_activations,estimate.model_buffers};
    const std::array<uint64_t,3> loaded{live.model_weights,live.model_activations,live.model_buffers};
    require(reported == loaded && std::equal(reported.begin(), reported.end(), actual.bytes),
            "native probe or live model differs from its successful CUDA allocation requests");
    const auto total = reported[0] + reported[1] + reported[2];
    std::printf("%s %d: weights %llu, activations %llu, I/O %llu, total %llu\n", reference ? "stock" : "reuse",
                side, (unsigned long long)reported[0], (unsigned long long)reported[1],
                (unsigned long long)reported[2], (unsigned long long)total);
    return total;
}

int main(int argc, char **argv)
{
    const char *dir = std::getenv("AJI_TEST_IMAGE_MODELS");
    if (!dir || !*dir) { std::puts("Set AJI_TEST_IMAGE_MODELS to the shipped ONNX directory"); return 77; }
    if (argc != 2) return 1;
    const int strip = std::atoi(argv[1]);
    try {
        const auto path = (std::filesystem::u8path(dir) / "2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.onnx").u8string();
        std::vector<char> blob;
        OnnxGraph graph;
        std::string why;
        require(ajicore::read_file_bytes(path, &blob), "cannot read the shipped ESRGAN fixture");
        const bool parsed = parse_onnx(blob.data(), blob.size(), &graph, &why);
        require(parsed, why);
        exact(graph, {1536,1536}, {{1536,1536}}, strip);
        exact(graph, {896,640}, {{896,640},{512,384}}, strip);

        aji_img_create_params make{};
        make.api_version = AJI_IMAGE_API_VERSION;
        make.backend = AJI_IMG_TENSORRT;
        char error[512]{};
        auto *ctx = aji_img_create(&make, error, sizeof(error));
        require(ctx, error);
        for (int side : {1536,2048}) {
            const uint64_t before = account(ctx, path, side, true), after = account(ctx, path, side, false);
            const uint64_t saved = strip ? uint64_t(side / 2) * (side / 2) * 64 * 2 : 0;
            require(before == after + saved, "unexpected ESRGAN arena size change");
        }
        aji_img_destroy(&ctx);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
