#include "aji_image.h"
#include "core/tiling.h"
#include "native_lifetime_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using ajinative::lifetime_test::require;
bool aji_test_force_tile_blend = false;
void test_tile_pack();

namespace {
struct Model {
    const char *file;
    int scale;
};
constexpr Model models[] = {
    {"2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.onnx", 2},
    {"2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.onnx", 2},
    {"2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx", 2},
    {"4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx", 4},
    {"4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx", 4}
};

struct Session {
    aji_img_ctx *ctx = nullptr;
    aji_img_model *model = nullptr;
    ~Session() {
        if (model) aji_img_release(&model);
        if (ctx) aji_img_destroy(&ctx);
    }
};
struct Page { int w, h, channels; };
struct Progress {
    std::vector<std::pair<int,int>> calls;
    bool cancel = false;
    static int report(void *opaque, int done, int total) {
        auto &p = *static_cast<Progress *>(opaque);
        p.calls.emplace_back(done, total);
        return p.cancel ? 1 : 0;
    }
};

using Bytes = std::vector<uint8_t>;
std::vector<Bytes> pages(const std::string &path, int scale, int maximum, bool even,
                         const std::vector<Page> &inputs, bool blend)
{
    Session session;
    aji_img_create_params cp{};
    cp.api_version = AJI_IMAGE_API_VERSION;
    cp.backend = AJI_IMG_TENSORRT;
    char err[512]{};
    session.ctx = aji_img_create(&cp, err, sizeof(err));
    require(session.ctx, err);
    constexpr int overlap = 32;
    const aji_img_model_params params{path.c_str(), maximum, maximum, overlap, even ? AJI_IMG_LOAD_EVEN_TILES : 0};
    session.model = aji_img_load(session.ctx, &params);
    require(session.model, aji_img_last_error(session.ctx));
    aji_img_memory mem{};
    require(aji_img_memory_get(session.ctx, session.model, &mem) == AJI_IMG_OK &&
            mem.model_activations && !mem.activations_shared && !mem.ctx_scratch, "expected fresh native model");
    aji_test_force_tile_blend = blend;
    size_t img = 0, packed = 0, band = 0, wx = 0, wy = 0;
    std::vector<Bytes> outputs;
    for (const auto &p : inputs) {
        int tw = maximum, th = maximum;
        auto xs = even ? ajicore::even_origins(p.w,maximum,overlap,&tw) : ajicore::tile_origins(p.w,tw,overlap);
        auto ys = even ? ajicore::even_origins(p.h,maximum,overlap,&th) : ajicore::tile_origins(p.h,th,overlap);
        const int tiles = int(xs.size()*ys.size());
        const int ow = p.w * scale, oh = p.h * scale;
        const int cols = std::min(p.w, tw)*scale, rows = std::min(p.h, th)*scale;
        const int input_stride = p.w*p.channels+7, output_stride = ow*p.channels+11;
        Bytes input(size_t(input_stride)*p.h);
        for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t((i*17 + (i>>4)*39) ^ (i>>7));
        constexpr uint8_t guard = 0xb6;
        Bytes output(size_t(output_stride)*oh, guard);
        aji_img_buffer in{p.w,p.h,p.channels,input_stride,input.data()};
        aji_img_buffer out{ow,oh,p.channels,output_stride,output.data()};
        Progress count;
        // Cancellation happens before host delivery, including when the tail emits several strips.
        if (outputs.empty()) {
            count.cancel = true;
            require(aji_img_upscale(session.model,&in,&out,Progress::report,&count) == AJI_IMG_CANCELLED &&
                    count.calls == std::vector<std::pair<int,int>>{{1,1}}, "single-tile cancellation callback");
            require(std::all_of(output.begin(),output.end(),[=](uint8_t v){ return v == guard; }), "cancelled output changed");
            count = {};
        }
        require(aji_img_upscale(session.model,&in,&out,Progress::report,&count) == AJI_IMG_OK,
                aji_img_last_error(session.ctx));
        require(count.calls.size() == size_t(tiles), "wrong callback count");
        for (int t = 0; t < tiles; ++t)
            require(count.calls[t] == std::make_pair(t+1,tiles), "wrong callback arguments");
        for (int y = 0; y < oh; ++y)
            require(std::all_of(output.begin()+size_t(y)*output_stride+ow*p.channels,
                                output.begin()+size_t(y+1)*output_stride,[=](uint8_t v){return v==guard;}),
                    "overwrote output row padding");
        img = std::max(img,size_t(p.w)*p.h*p.channels);
        packed = std::max(packed,size_t(rows)*ow*p.channels);
        if (blend || tiles > 1) {
            band = std::max(band,size_t(rows)*ow*p.channels*sizeof(float));
            wx = std::max(wx,xs.size()*cols*sizeof(float));
            wy = std::max(wy,ys.size()*rows*sizeof(float));
        }
        require(aji_img_memory_get(session.ctx,session.model,&mem) == AJI_IMG_OK &&
                mem.ctx_scratch == img+packed+band+wx+wy, "wrong retained context scratch");
        outputs.push_back(std::move(output));
    }
    return outputs;
}

void compare(const std::filesystem::path &dir, Model model, int maximum, bool even, const std::vector<Page> &inputs)
{
    const auto path = (dir / model.file).u8string();
    const auto ref = pages(path,model.scale,maximum,even,inputs,true);
    const auto got = pages(path,model.scale,maximum,even,inputs,false);
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto diff = std::mismatch(ref[i].begin(),ref[i].end(),got[i].begin());
        if (diff.first != ref[i].end()) {
            std::fprintf(stderr,"%s %dx%d byte %zu: blend %u, direct %u\n",model.file,inputs[i].w,inputs[i].h,
                         size_t(diff.first-ref[i].begin()),*diff.first,*diff.second);
            throw std::runtime_error("page output mismatch");
        }
    }
    std::printf("%s at %d, %s origins: %zu pages byte-exact, scratch and callbacks verified\n",
                model.file,maximum,even ? "even" : "fixed",inputs.size());
}
} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 2) return 1;
        const std::string mode = argv[1];
        if (mode == "pack") {
            test_tile_pack();
            return 0;
        }
        const char *dir = std::getenv("AJI_TEST_IMAGE_MODELS");
        if (!dir || !*dir) return 77;
        const auto directory = std::filesystem::u8path(dir);
        if (mode == "pages") {
            const std::vector<Page> sequence{{121,117,3}, {207,183,3}, {121,117,3}, {119,109,1}};
            for (const auto &model : models) compare(directory, model, 128, true, sequence);
            compare(directory, models[0], 128, false, sequence);
        } else if (mode == "large") {
            for (const auto &model : models) {
                const std::vector<int> sizes = model.scale == 4 ? std::vector<int>{512} : std::vector<int>{1536,2048};
                for (int side : sizes) compare(directory, model, side, true, {{side,side,3}});
            }
        } else return 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
    return 0;
}
