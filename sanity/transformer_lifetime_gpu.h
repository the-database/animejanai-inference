#pragma once
#include "native_lifetime_gpu.h"
#include "native/imgnet.h"
#include "native/onnx_graph.h"
#include "native/pack.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace ajinative::lifetime_test {
inline std::string dimensions(Shape s) { return std::to_string(s.first) + "x" + std::to_string(s.second); }

inline Output input_for(int w, int h)
{
    const size_t plane = size_t(w) * h;
    Output input(3 * plane);
    for (size_t i = 0; i < input.size(); ++i) {
        const auto v = uint8_t((i * 13 + (i >> 4) * 17 + (i >> 11) * 61) ^ ((i >> 3) * 7));
        input[(i % 3) * plane + i / 3] = float_to_half_bits(v * (1.f / 255.f));
    }
    return input;
}

inline std::vector<Output> render(const char *family, int scale, const OnnxGraph &graph, Shape built, const std::vector<Shape> &shapes,
                                  bool separate, bool poison, void (*edit)(LifetimePlan &) = nullptr)
{
    LifetimeCheck check;
    std::vector<LifetimeCheck::Workspace> launches, first_launches;
    check.launches = &launches;
    check.edit = edit;
    if (separate) {
        check.workspace_arena_limit = 0;
        check.disjoint_features = true;
    }
    check.poison = poison;
    LifetimeCheck::active = &check;
    std::string why;
    auto net = ImgNet::create(graph, built.first, built.second, nullptr, &why);
    LifetimeCheck::active = nullptr;
    require(net && net->scale() == scale && net->describe().find(family) != std::string::npos, std::string("expected ") + family + ": " + why);
    require(launches.empty(), "build unexpectedly recorded a dispatch");
    std::vector<Output> results;
    for (auto [w,h] : shapes) {
        const auto input = input_for(w,h);
        Output fp16(input.size() * scale * scale);
        Device in(input.size()), out(fp16.size());
        gpu(cudaMemcpy(in.p, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        // Exercise a second inference on the same allocations, including repeated groups and phase scatters.
        for (int i = 0; i < 2; ++i) {
            launches.clear();
            const auto rc = net->run(in.p, false, out.p, false, w, h, nullptr);
            require(check.error.empty(), check.error);
            gpu(rc);
            require(!launches.empty(), "no run-time workspace checks");
            if (first_launches.empty()) first_launches = launches;
            else require(launches == first_launches, "run-time GConv step, address or capacity changed on the same model");
            if (i == 0 && Shape{w,h} != built && !edit)
                std::printf("%s check %s on %s: %zu identical run-time workspace ranges\n",
                            family, dimensions({w,h}).c_str(), dimensions(built).c_str(), launches.size());
            gpu(cudaMemcpy(fp16.data(), out.p, fp16.size() * 2, cudaMemcpyDeviceToHost));
            if (i == 0) results.push_back(fp16);
            else require(results.back() == fp16, "fixed-order output did not repeat at " + dimensions({w,h}));
        }
    }
    return results;
}

inline void verify_stock(const char *reference_dir, Shape built, const std::vector<Shape> &shapes,
                         const std::vector<Output> &reference)
{
    const char *dir = std::getenv(reference_dir);
    for (size_t j = 0; j < shapes.size(); ++j) {
        const auto &v = reference[j];
        require(std::none_of(v.begin(), v.end(), [](uint16_t x) { return (x & 0x7c00) == 0x7c00; }), "nonfinite reference");
        // Optional independent pre-change build: raw NCHW FP16 before RGB8 conversion, using input_for().
        if (dir && *dir) {
            const auto name = dimensions(built) + "-" + dimensions(shapes[j]) + ".fp16";
            std::ifstream f(std::filesystem::u8path(dir) / name, std::ios::binary);
            Output old(v.size());
            require(f.read(reinterpret_cast<char *>(old.data()), old.size() * 2) &&
                    f.peek() == std::char_traits<char>::eof() && old == v, "independent stock mismatch at " + name);
        }
    }
}

inline void require_exact(const Output &actual, const Output &expected, const std::string &label)
{
    const auto diff = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (diff.first != actual.end()) {
        std::fprintf(stderr, "%s, element %zu: expected %04x, got %04x\n", label.c_str(),
                     size_t(diff.first - actual.begin()), *diff.second, *diff.first);
        throw std::runtime_error("FP16 output mismatch");
    }
}
} // namespace ajinative::lifetime_test
