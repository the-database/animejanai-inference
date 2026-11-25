#include "core/tiling_gpu.h"
#include "native_lifetime_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

using namespace ajicore;
using ajinative::lifetime_test::gpu;
using ajinative::lifetime_test::require;

namespace {
struct CudaFree {
    void operator()(void *p) const { cudaFree(p); }
};

template<class T> std::unique_ptr<T, CudaFree> buffer(size_t count)
{
    T *p = nullptr;
    gpu(cudaMalloc(&p, count * sizeof(T)));
    return std::unique_ptr<T, CudaFree>(p);
}
} // namespace

void test_tile_pack()
{
    constexpr int w = 256, h = 256, plane = w * h;
    std::vector<uint16_t> bits(3 * plane);
    std::vector<float> floats(3 * plane);
    for (size_t i = 0; i < bits.size(); ++i) {
        // Every 16-bit pattern appears in every channel, with a different phase.
        bits[i] = uint16_t(i + (i / plane) * 17777);
        const uint32_t word = uint32_t(i) * 2654435761u;
        std::memcpy(&floats[i], &word, sizeof(word));
        if (i % plane < 3 * 255) {
            const int j = int(i % plane);
            const float edge = (float((j / 3 + i / plane) % 255) + 0.5f) / 255.0f;
            floats[i] = j % 3 == 0 ? std::nextafter(edge, 0.0f) :
                        j % 3 == 1 ? edge : std::nextafter(edge, 1.0f);
        }
    }
    const uint32_t special[] = {0, 0x80000000u, 1, 0x80000001u, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0x7f800001u};
    std::memcpy(floats.data() + 1024, special, sizeof(special));
    auto src = buffer<float>(floats.size());
    auto weights = buffer<float>(w);
    std::vector<float> ones(w, 1.0f);
    gpu(cudaMemcpy(weights.get(), ones.data(), w * sizeof(float), cudaMemcpyHostToDevice));
    size_t checked = 0;
    for (GpuDtype dt : {GpuDtype::F16, GpuDtype::BF16, GpuDtype::F32}) {
        if (dt == GpuDtype::F32)
            gpu(cudaMemcpy(src.get(), floats.data(), floats.size() * sizeof(float), cudaMemcpyHostToDevice));
        else
            gpu(cudaMemcpy(src.get(), bits.data(), bits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        for (int co : {1,3}) for (int model_channels : {1,3}) for (int crop : {0,7}) {
            const int cols = w - crop, rows = h - crop, y0 = crop;
            const size_t stride = size_t(cols) * co, start = y0 * stride;
            const size_t bytes = start + rows * stride + 16;
            auto band = buffer<float>(h * stride);
            auto packed = buffer<uint8_t>(bytes);
            auto reference = buffer<uint8_t>(bytes);
            gpu(cudaMemset(band.get(), 0, h * stride * sizeof(float)));
            gpu(cudaMemset(packed.get(), 33, bytes));
            gpu(cudaMemset(reference.get(), 33, bytes));
            gpu(gpu_tile_blend(src.get(), dt, model_channels, w, h, weights.get(), weights.get(),
                               cols, rows, 0, y0, band.get(), co, h, cols, nullptr));
            gpu(gpu_band_flush(band.get(), co, h, cols, y0, rows, reference.get() + start, nullptr));
            gpu(gpu_tile_pack(src.get(), dt, model_channels, w, h, cols, rows, co, packed.get() + start, nullptr));
            std::vector<uint8_t> a(bytes), b(bytes);
            gpu(cudaMemcpy(a.data(), reference.get(), bytes, cudaMemcpyDeviceToHost));
            gpu(cudaMemcpy(b.data(), packed.get(), bytes, cudaMemcpyDeviceToHost));
            const auto diff = std::mismatch(a.begin(), a.end(), b.begin());
            if (diff.first != a.end()) {
                std::fprintf(stderr, "packing mismatch dtype=%d model=%d output=%d crop=%d byte=%zu blend=%u direct=%u\n",
                             int(dt), model_channels, co, crop, size_t(diff.first-a.begin()), *diff.first, *diff.second);
                throw std::runtime_error("tile packing differs from unit-weight blend");
            }
            require(std::all_of(b.begin(), b.begin() + start, [](uint8_t v){ return v == 33; }) &&
                    std::all_of(b.end() - 16, b.end(), [](uint8_t v){ return v == 33; }), "packing wrote outside the strip");
            checked += rows * stride;
        }
    }
    std::printf("Packing: %zu visible output bytes exact across three dtypes, four channel layouts and two crops\n", checked);
}
