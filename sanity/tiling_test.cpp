/*
 * tiling_test — CPU-only checks for core/tiling (no GPU, no models).
 *
 *  - nearest: a model that pixel-replicates its tile must reproduce an
 *    exact nearest-neighbour upscale at any image size, tile, overlap
 *    and channel layout (blend weights sum to one; no seams, no gaps).
 *  - constant: a model emitting 0.6 must give 153 everywhere (coverage).
 *  - fp16/bf16 round trips.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/half.h"
#include "core/tiling.h"

using namespace ajicore;

namespace {

class Nearest : public TileModel {
public:
    int calls = 0;
    int run_tile(const float *in, float *out, std::string *) override
    {
        calls++;
        const int ow = tile_w * scale, oh = tile_h * scale;
        for (int c = 0; c < out_ch; c++) {
            const float *src = in + (size_t)(in_ch == 1 ? 0 : c) * tile_w * tile_h;
            float *dst = out + (size_t)c * ow * oh;
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    dst[(size_t)y * ow + x] =
                        src[(size_t)(y / scale) * tile_w + x / scale];
        }
        return AJI_IMG_OK;
    }
};

class Constant : public TileModel {
public:
    int run_tile(const float *, float *out, std::string *) override
    {
        const size_t n = (size_t)out_ch * tile_w * scale * tile_h * scale;
        for (size_t i = 0; i < n; i++)
            out[i] = 0.6f;  // 153.0 exactly: clear of the .5 rounding edge
        return AJI_IMG_OK;
    }
};

int failures = 0;

void check(bool ok, const char *what, int w, int h, int ch, int tile, int ov, int s)
{
    if (!ok) {
        failures++;
        fprintf(stderr, "FAIL %s: %dx%d c%d tile %d ov %d scale %d\n", what, w, h,
                ch, tile, ov, s);
    }
}

void run_nearest(int w, int h, int ch, int model_ch, int tile, int ov, int s)
{
    std::vector<uint8_t> src((size_t)w * h * ch);
    for (size_t i = 0; i < src.size(); i++)
        src[i] = (uint8_t)((i * 2654435761u) >> 24);
    // pad the strides to catch stride/width mix-ups
    const ptrdiff_t is = (ptrdiff_t)w * ch + 7, os = (ptrdiff_t)w * s * ch + 5;
    std::vector<uint8_t> inb((size_t)is * h), outb((size_t)os * h * s, 0xCD);
    for (int y = 0; y < h; y++)
        memcpy(&inb[(size_t)y * is], &src[(size_t)y * w * ch], (size_t)w * ch);
    aji_img_buffer in{w, h, ch, is, inb.data()};
    aji_img_buffer out{w * s, h * s, ch, os, outb.data()};

    Nearest m;
    m.tile_w = m.tile_h = tile;
    m.scale = s;
    m.in_ch = m.out_ch = model_ch;
    std::string err;
    int rc = upscale_tiled(m, ov, &in, &out, nullptr, nullptr, &err);
    check(rc == AJI_IMG_OK, "nearest rc", w, h, ch, tile, ov, s);
    if (rc != AJI_IMG_OK)
        return;
    bool ok = true;
    for (int y = 0; y < h * s && ok; y++)
        for (int x = 0; x < w * s && ok; x++)
            for (int c = 0; c < ch && ok; c++) {
                // a 3-ch model fed gray replicates; channel 0 comes back
                const uint8_t want = src[((size_t)(y / s) * w + x / s) * ch + c];
                ok = outb[(size_t)y * os + (size_t)x * ch + c] == want;
            }
    check(ok, "nearest pixels", w, h, ch, tile, ov, s);
}

void run_constant(int w, int h, int tile, int ov, int s)
{
    std::vector<uint8_t> inb((size_t)w * h * 3, 0), outb((size_t)w * h * s * s * 3, 0);
    aji_img_buffer in{w, h, 3, (ptrdiff_t)w * 3, inb.data()};
    aji_img_buffer out{w * s, h * s, 3, (ptrdiff_t)w * s * 3, outb.data()};
    Constant m;
    m.tile_w = m.tile_h = tile;
    m.scale = s;
    std::string err;
    int rc = upscale_tiled(m, ov, &in, &out, nullptr, nullptr, &err);
    bool ok = rc == AJI_IMG_OK;
    for (size_t i = 0; ok && i < outb.size(); i++)
        ok = outb[i] == 153;
    check(ok, "constant coverage", w, h, 3, tile, ov, s);
}

} // namespace

int main()
{
    const int sizes[][2] = {{1, 1},     {7, 5},     {64, 64},   {63, 65},
                            {64, 129},  {200, 77},  {513, 300}, {1000, 1400},
                            {130, 2}};
    for (auto &sz : sizes) {
        for (int s : {1, 2, 4}) {
            for (int ov : {0, 8, 16}) {
                for (int tile : {32, 64}) {
                    run_nearest(sz[0], sz[1], 3, 3, tile, ov, s);
                    run_nearest(sz[0], sz[1], 1, 1, tile, ov, s);
                    run_nearest(sz[0], sz[1], 1, 3, tile, ov, s);
                    run_constant(sz[0], sz[1], tile, ov, s);
                }
            }
        }
    }

    // cancellation stops early with AJI_IMG_CANCELLED
    {
        std::vector<uint8_t> inb(300 * 300), outb(600 * 600);
        aji_img_buffer in{300, 300, 1, 300, inb.data()};
        aji_img_buffer out{600, 600, 1, 600, outb.data()};
        Nearest m;
        m.tile_w = m.tile_h = 64;
        m.scale = 2;
        m.in_ch = m.out_ch = 1;
        std::string err;
        auto stop_at_3 = [](void *, int done, int) -> int { return done >= 3; };
        int rc = upscale_tiled(m, 16, &in, &out, stop_at_3, nullptr, &err);
        check(rc == AJI_IMG_CANCELLED && m.calls == 3, "cancel", 300, 300, 1, 64, 16, 2);
    }

    // mismatched output dims are rejected
    {
        std::vector<uint8_t> b(64 * 64 * 4);
        aji_img_buffer in{32, 32, 1, 32, b.data()};
        aji_img_buffer out{63, 64, 1, 63, b.data()};
        Nearest m;
        m.tile_w = m.tile_h = 32;
        m.scale = 2;
        std::string err;
        check(upscale_tiled(m, 8, &in, &out, nullptr, nullptr, &err) == AJI_IMG_ERR_ARGS,
              "args", 32, 32, 1, 32, 8, 2);
    }

    // half conversions: exact for every 8-bit level, and round trips
    for (int i = 0; i < 256; i++) {
        const float f = i / 255.0f;
        const float h = f16_to_f32(f32_to_f16(f));
        const float b = bf16_to_f32(f32_to_bf16(f));
        if ((int)(h * 255.0f + 0.5f) != i || (int)(b * 255.0f + 0.5f) != i) {
            failures++;
            fprintf(stderr, "FAIL half level %d: f16 %f bf16 %f\n", i, h, b);
        }
    }
    for (uint32_t u = 0; u < 0x7c00; u++) {
        if (f32_to_f16(f16_to_f32((uint16_t)u)) != u) {
            failures++;
            fprintf(stderr, "FAIL f16 round trip 0x%04x\n", u);
            break;
        }
    }

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("tiling_test: all passed\n");
    return 0;
}
