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

static std::array<uint64_t,3> model_bytes(aji_img_ctx *ctx, const aji_img_model_params &cfg,
                                          bool stock_qkv, bool separate)
{
    LifetimeCheck probe, load;
    if (stock_qkv) probe.edit = load.edit = [](LifetimePlan &p) { qkv_end(p, 14); };
    if (separate) probe.workspace_arena_limit = load.workspace_arena_limit = 0;
    return account_native(ctx, cfg, probe, load);
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
            const auto stock = model_bytes(ctx, cfg, true, true);
            const auto qkv = model_bytes(ctx, cfg, false, true);
            const auto workspace = model_bytes(ctx, cfg, true, false);
            const auto both = model_bytes(ctx, cfg, false, false);
            for (const auto &row : {qkv, workspace, both})
                require(row[0] == stock[0] && row[2] == stock[2], "DAT2 weight or I/O bytes changed");
            require(qkv[1] < stock[1] && workspace[1] <= stock[1] && both[1] <= qkv[1],
                    "DAT2 QKV or workspace allocation regression");
            std::printf("DAT ledger %s weights %llu, I/O %llu, activation bytes "
                        "stock %llu, QKV only %llu, workspace only %llu, both %llu\n", dimensions(size).c_str(),
                        (unsigned long long)both[0], (unsigned long long)both[2], (unsigned long long)stock[1],
                        (unsigned long long)qkv[1], (unsigned long long)workspace[1], (unsigned long long)both[1]);
        }
        aji_img_destroy(&ctx);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
