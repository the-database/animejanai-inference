#include "transformer_lifetime_gpu.h"
#include "core/fileio.h"

using namespace ajinative;
using namespace ajinative::lifetime_test;

static void shorten_cf(LifetimePlan &plan)
{
    for (auto &buffer : plan.buffers)
        if (buffer.name && std::string(buffer.name) == "CF") {
            buffer.live = {{6,8}};
            return;
        }
    throw std::runtime_error("missing FDAT padding mutation target");
}

static void exact(const OnnxGraph &graph, Shape built, const std::vector<Shape> &shapes)
{
    const auto stock = render("FDAT", 2, graph, built, shapes, true, false);
    verify_stock("AJI_TEST_FDAT_REFERENCE", built, shapes, stock);
    const auto actual = render("FDAT", 2, graph, built, shapes, false, true);
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto label = "FDAT " + dimensions(shapes[i]) + " on " + dimensions(built);
        require_exact(actual[i], stock[i], label);
        std::printf("%s: %zu FP16 values exact\n", label.c_str(), actual[i].size());
    }
    if (shapes.back() == built) return;
    const auto broken = render("FDAT", 2, graph, built, {shapes.back()}, false, true, shorten_cf);
    const auto &v = stock.back();
    const auto diff = std::mismatch(v.begin(), v.end(), broken.front().begin());
    require(diff.first != v.end(), "poison failed to expose a lost CF padding lane");
    std::printf("FDAT mutation %s on %s, element %zu: stock %04x, shortened CF %04x\n",
                dimensions(shapes.back()).c_str(), dimensions(built).c_str(), size_t(diff.first-v.begin()),
                *diff.first, *diff.second);
}

int main()
{
    const char *dir = std::getenv("AJI_TEST_IMAGE_MODELS");
    if (!dir || !*dir) return 77;
    try {
        const auto path = (std::filesystem::u8path(dir) / "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx").u8string();
        std::vector<char> blob;
        OnnxGraph graph;
        std::string why;
        require(ajicore::read_file_bytes(path, &blob), "cannot read FDAT fixture");
        const bool parsed = parse_onnx(blob.data(), blob.size(), &graph, &why);
        require(parsed, why);
        for (int side : {512,1536,2048}) exact(graph, {side,side}, {{side,side}});
        exact(graph, {512,384}, {{512,384},{320,192}});
        aji_img_create_params create{};
        create.api_version = AJI_IMAGE_API_VERSION;
        create.backend = AJI_IMG_TENSORRT;
        char error[512]{};
        auto *ctx = aji_img_create(&create, error, sizeof(error));
        require(ctx, error);
        for (Shape size : {Shape{512,512}, {1536,1536}, {2048,2048}, {512,384}}) {
            aji_img_model_params cfg{path.c_str(), size.first, size.second, 32, AJI_IMG_LOAD_EVEN_TILES};
            LifetimeCheck probe, load;
            const auto bytes = account_native(ctx, cfg, probe, load);
            std::printf("FDAT ledger %s weights %llu, activations %llu, I/O %llu\n", dimensions(size).c_str(),
                        (unsigned long long)bytes[0], (unsigned long long)bytes[1], (unsigned long long)bytes[2]);
        }
        aji_img_destroy(&ctx);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
