#include "transformer_lifetime_gpu.h"
#include "core/fileio.h"

using namespace ajinative;
using namespace ajinative::lifetime_test;

static void qkv_end(LifetimePlan &plan, int end)
{
    for (auto &buffer : plan.buffers)
        if (buffer.name && std::string(buffer.name) == "QKV") {
            buffer.live.front().second = end;
            return;
        }
    throw std::runtime_error("missing DAT2 QKV");
}

static void exact(const OnnxGraph &graph, Shape built, const std::vector<Shape> &shapes)
{
    const auto stock = render("DAT", 4, graph, built, shapes, true, false, [](LifetimePlan &p) { qkv_end(p, 14); });
    verify_stock("AJI_TEST_DAT_REFERENCE", built, shapes, stock);
    for (bool separate : {false,true}) {
        const auto out = render("DAT", 4, graph, built, shapes, separate, true);
        for (size_t i = 0; i < shapes.size(); ++i) {
            const auto label = "DAT " + dimensions(shapes[i]) + " on " + dimensions(built) + (separate ? " fallback" : " arena");
            require_exact(out[i], stock[i], label);
            std::printf("%s: %zu FP16 values exact\n", label.c_str(), out[i].size());
        }
    }
    if (shapes.back() != built) {
        const auto broken = render("DAT", 4, graph, built, {shapes.back()}, false, true, [](LifetimePlan &p) { qkv_end(p, 12); });
        const auto &v = stock.back();
        const auto diff = std::mismatch(v.begin(), v.end(), broken.front().begin());
        require(diff.first != v.end(), "poison failed to expose QKV dying before attention");
        std::printf("DAT mutation %s, element %zu: stock %04x, shortened QKV %04x\n",
                    dimensions(shapes.back()).c_str(), size_t(diff.first-v.begin()), *diff.first, *diff.second);
    }
}

int main()
{
    const char *dir = std::getenv("AJI_TEST_IMAGE_MODELS");
    if (!dir || !*dir) return 77;
    try {
        const auto path = (std::filesystem::u8path(dir) / "4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx").u8string();
        std::vector<char> blob;
        OnnxGraph graph;
        std::string why;
        require(ajicore::read_file_bytes(path, &blob), "cannot read DAT2 fixture");
        const bool parsed = parse_onnx(blob.data(), blob.size(), &graph, &why);
        require(parsed, why);
        exact(graph, {512,512}, {{512,512}});
        exact(graph, {768,768}, {{768,768}});
        exact(graph, {512,384}, {{512,384},{320,192}});
        aji_img_create_params create{};
        create.api_version = AJI_IMAGE_API_VERSION;
        create.backend = AJI_IMG_TENSORRT;
        char error[512]{};
        auto *ctx = aji_img_create(&create, error, sizeof(error));
        require(ctx, error);
        for (Shape size : {Shape{512,512}, {768,768}, {512,384}}) {
            aji_img_model_params cfg{path.c_str(), size.first, size.second, 32, AJI_IMG_LOAD_EVEN_TILES};
            LifetimeCheck old_probe, old_load, probe, load;
            old_probe.edit = old_load.edit = [](LifetimePlan &p) { qkv_end(p,14); };
            old_probe.workspace_arena_limit = old_load.workspace_arena_limit = 0;
            const auto old = account_native(ctx, cfg, old_probe, old_load);
            const auto now = account_native(ctx, cfg, probe, load);
            require(old[0] == now[0] && old[2] == now[2] && old[1] > now[1], "DAT2 model memory did not fall");
            std::printf("DAT ledger %s weights %llu, activations %llu -> %llu, I/O %llu\n", dimensions(size).c_str(),
                        (unsigned long long)now[0], (unsigned long long)old[1], (unsigned long long)now[1],
                        (unsigned long long)now[2]);
        }
        aji_img_destroy(&ctx);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
