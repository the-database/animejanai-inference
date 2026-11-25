#include "transformer_lifetime_gpu.h"
#include "core/fileio.h"
#include "native/imgnet.h"
#include "native/onnx_graph.h"
#include "native/pack.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace ajinative;
using namespace ajinative::lifetime_test;

static void shorten_qkv(LifetimePlan &plan)
{
    for (auto &buffer : plan.buffers)
        if (buffer.name && std::string(buffer.name) == "QKV") {
            --buffer.live.front().second; // the HAB attention still reads QKV at step 15
            return;
        }
    throw std::runtime_error("missing HAT QKV mutation target");
}

static void exact(const OnnxGraph &graph, Shape built, const std::vector<Shape> &shapes)
{
    const auto reference = render("HAT", 4, graph, built, shapes, true, false);
    verify_stock("AJI_TEST_HAT_REFERENCE", built, shapes, reference);
    for (bool separate : {false, true}) {
        const auto actual = render("HAT", 4, graph, built, shapes, separate, true);
        for (size_t j = 0; j < shapes.size(); ++j) {
            const auto &a = actual[j], &b = reference[j];
            const auto diff = std::mismatch(a.begin(), a.end(), b.begin());
            if (diff.first != a.end()) {
                std::fprintf(stderr, "%s built %s, %s, element %zu: stock %04x, candidate %04x\n",
                             dimensions(shapes[j]).c_str(), dimensions(built).c_str(), separate ? "fallback" : "arena",
                             (size_t)(diff.first-a.begin()), *diff.second, *diff.first);
                throw std::runtime_error("poisoned HAT output differs from deterministic stock");
            }
            std::printf("HAT check %s built %s, %s: %zu FP16 values exact\n", dimensions(shapes[j]).c_str(),
                        dimensions(built).c_str(), separate ? "fallback" : "arena", a.size());
        }
    }
    if (shapes.back() != built) {
        const auto broken = render("HAT", 4, graph, built, {shapes.back()}, false, true, shorten_qkv);
        const auto &stock = reference.back();
        const auto diff = std::mismatch(stock.begin(), stock.end(), broken.front().begin());
        require(diff.first != stock.end(), "poison failed to expose the shortened HAT QKV lifetime");
        std::printf("HAT mutation %s, element %zu: stock %04x, shortened QKV %04x\n",
                    dimensions(shapes.back()).c_str(), size_t(diff.first - stock.begin()), *diff.first, *diff.second);
    }
}

static void allocations(aji_img_ctx *ctx, const std::string &path, Shape built)
{
    const aji_img_model_params params{path.c_str(), built.first, built.second, 32, AJI_IMG_LOAD_EVEN_TILES};
    LifetimeCheck separate_probe, separate_load, arena_probe, arena_load;
    separate_probe.workspace_arena_limit = separate_load.workspace_arena_limit = 0;
    const auto before = account_native(ctx, params, separate_probe, separate_load);
    const auto after = account_native(ctx, params, arena_probe, arena_load);
    const size_t workspace = size_t(built.first) * built.second * 192 * sizeof(float);
    require(before[0] == after[0] && before[2] == after[2] && before[1] == after[1] + workspace,
            "unexpected HAT fallback or workspace saving at " + dimensions(built));
    std::printf("HAT check %s: weights %llu, activations %llu -> %llu, I/O %llu, fallback %zu bytes\n",
                dimensions(built).c_str(), (unsigned long long)after[0], (unsigned long long)before[1],
                (unsigned long long)after[1], (unsigned long long)after[2], workspace);
}

int main()
{
    const char *dir = std::getenv("AJI_TEST_IMAGE_MODELS");
    if (!dir || !*dir) { std::puts("Set AJI_TEST_IMAGE_MODELS to the shipped ONNX directory"); return 77; }
    try {
        const auto path = (std::filesystem::u8path(dir) / "4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx").u8string();
        std::vector<char> blob;
        OnnxGraph graph;
        std::string why;
        require(ajicore::read_file_bytes(path, &blob), "cannot read the shipped HAT-L fixture");
        const bool parsed = parse_onnx(blob.data(), blob.size(), &graph, &why);
        require(parsed, why);
        exact(graph, {512,512}, {{512,512}});
        exact(graph, {768,768}, {{768,768}});
        exact(graph, {512,384}, {{512,384},{320,192}});

        aji_img_create_params make{};
        make.api_version = AJI_IMAGE_API_VERSION;
        make.backend = AJI_IMG_TENSORRT;
        char error[512]{};
        auto *ctx = aji_img_create(&make, error, sizeof(error));
        require(ctx, error);
        for (Shape shape : {Shape{512,512}, {768,768}, {512,384}}) allocations(ctx, path, shape);
        aji_img_destroy(&ctx);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
