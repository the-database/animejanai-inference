// aji_frame_diff's kernels (held-frame reuse in aji_encode): the largest 8x8
// tile mean |b - a| over all planes, and the signed luma means of a 4x4
// region grid, in 8-bit units, for each format aji_infer takes on CUDA.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "aji.h"
#include "kernels.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

namespace {

// A host frame in one format, uploaded with a row pitch wider than the
// row, so the kernels have to honour strides.
struct Frame {
    int w, h, fmt;
    bool wide, f444;
    std::vector<uint16_t> pl[3];   // samples; 8-bit formats use the low byte
    int pw[3], ph[3];              // samples per row (CbCr: both), rows
    void *dev[3] = {nullptr, nullptr, nullptr};
    size_t pitch[3] = {0, 0, 0};

    Frame(int w_, int h_, int fmt_) : w(w_), h(h_), fmt(fmt_)
    {
        wide = fmt != AJI_FMT_NV12;
        f444 = aji_format_is_444(fmt) != 0;
        const int n = f444 ? 3 : 2;
        for (int p = 0; p < 3; p++) {
            pw[p] = ph[p] = 0;
            if (p >= n) continue;
            const bool uv = !f444 && p == 1;
            pw[p] = uv ? (w + 1) / 2 * 2 : w;
            ph[p] = uv ? (h + 1) / 2 : h;
            pl[p].assign((size_t)pw[p] * ph[p], 0);
        }
    }
    ~Frame() { for (void *d : dev) cudaFree(d); }

    // a deterministic pseudo-random picture in the middle of the range
    void fill(unsigned seed)
    {
        for (int p = 0; p < 3; p++)
            for (size_t i = 0; i < pl[p].size(); i++) {
                seed = seed * 1664525u + 1013904223u;
                const unsigned v8 = 60 + (seed >> 24) % 120;     // 60..179
                pl[p][i] = (uint16_t)(wide ? v8 << 8 : v8);
            }
    }
    uint16_t &at(int p, int x, int y) { return pl[p][(size_t)y * pw[p] + x]; }

    bool upload()
    {
        const int bpp = wide ? 2 : 1;
        for (int p = 0; p < 3; p++) {
            if (pl[p].empty()) continue;
            const size_t row = (size_t)pw[p] * bpp;
            if (!dev[p] && cudaMallocPitch(&dev[p], &pitch[p], row + 64, ph[p])
                               != cudaSuccess)
                return false;
            std::vector<uint8_t> host(row * ph[p]);
            for (size_t i = 0; i < pl[p].size(); i++) {
                if (wide) std::memcpy(&host[i * 2], &pl[p][i], 2);
                else host[i] = (uint8_t)pl[p][i];
            }
            if (cudaMemcpy2D(dev[p], pitch[p], host.data(), row, row, ph[p],
                             cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
        }
        return true;
    }
    aji_frame frame() const
    {
        aji_frame f = {};
        f.width = w; f.height = h; f.format = fmt;
        f.matrix = AJI_MATRIX_BT709; f.range = AJI_RANGE_LIMITED;
        f.siting = AJI_SITING_LEFT;
        for (int p = 0; p < 3; p++) {
            f.plane[p] = dev[p];
            f.stride[p] = (ptrdiff_t)pitch[p];
        }
        return f;
    }
};

void *g_acc_dev, *g_acc_host;

int diff(const Frame &a, const Frame &b, aji_frame_diff_result *r)
{
    const aji_frame fa = a.frame(), fb = b.frame();
    if (aji_frame_diff_launch(&fa, &fb, g_acc_dev, nullptr) != 0) return -1;
    if (cudaMemcpy(g_acc_host, g_acc_dev, AJI_FRAME_DIFF_ACC_BYTES,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    aji_frame_diff_finish(&fa, g_acc_host, r);
    return 0;
}

bool near(float got, float want, float tol = 1e-4f)
{
    return std::fabs(got - want) <= tol;
}

float max_abs_region(const aji_frame_diff_result &r)
{
    float m = 0;
    for (float v : r.region_mean) m = std::fmax(m, std::fabs(v));
    return m;
}

// 10-bit sample v as P010 / YUV444P10MSB stores it
uint16_t msb10(int v) { return (uint16_t)(v << 6); }

} // namespace

int main()
{
    CHECK(cudaMalloc(&g_acc_dev, AJI_FRAME_DIFF_ACC_BYTES) == cudaSuccess);
    CHECK(cudaMallocHost(&g_acc_host, AJI_FRAME_DIFF_ACC_BYTES) == cudaSuccess);
    aji_frame_diff_result r;

    const int fmts[] = {AJI_FMT_NV12, AJI_FMT_P010, AJI_FMT_YUV444P16,
                        AJI_FMT_YUV444P10MSB, AJI_FMT_YUV444P12MSB};
    for (int fmt : fmts) {
        // 70x46: partial tiles at the right and bottom edges
        const int W = 70, H = 46;
        Frame a(W, H, fmt), b(W, H, fmt);
        a.fill(7);
        b.fill(7);
        const int unit = a.wide ? 256 : 1;   // one 8-bit level
        CHECK(a.upload() && b.upload());

        // identical frames
        CHECK(diff(a, b, &r) == 0);
        CHECK(r.max_tile == 0.0f && max_abs_region(r) == 0.0f);

        // one 8x8 luma tile +40: D = 40, one region moves by 40*64/area
        for (int y = 8; y < 16; y++)
            for (int x = 16; x < 24; x++)
                b.at(0, x, y) += 40 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 40.0f));
        // 9x6 tiles: region columns of tiles {0,1,2},{3,4},{5,6},{7,8} and
        // rows {0,1},{2},{3,4},{5}; tile (2,1) is in region (0,0), 24x16 px
        CHECK(near(r.region_mean[0], 40.0f * 64 / (24 * 16)));
        for (int i = 1; i < 16; i++) CHECK(r.region_mean[i] == 0.0f);

        // the same tile -40 instead: the signed mean flips, D doesn't
        for (int y = 8; y < 16; y++)
            for (int x = 16; x < 24; x++)
                b.at(0, x, y) -= 80 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 40.0f));
        CHECK(near(r.region_mean[0], -40.0f * 64 / (24 * 16)));

        // the partial bottom-right tile (6x6 px) +12: its mean, not /64
        b.fill(7);
        for (int y = 40; y < H; y++)
            for (int x = 64; x < W; x++)
                b.at(0, x, y) += 12 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 12.0f));
        CHECK(near(r.region_mean[15], 12.0f * 36 / (14 * 6)));   // tiles 7-8 x 5

        // a uniform +1 on every plane: D = 1, every region +1
        for (int p = 0; p < 3; p++)
            for (auto &v : b.pl[p]) v = a.pl[p][&v - b.pl[p].data()] + unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 1.0f));
        for (int i = 0; i < 16; i++) CHECK(near(r.region_mean[i], 1.0f));

        // a fade over one quadrant: +0.25 levels (16-bit) on the top-left
        // luma regions only; every tile stays far below a held-frame
        // threshold, the signed means show it
        if (a.wide) {
            b.fill(7);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    if (x < 40 && y < 32) b.at(0, x, y) += 64;   // 0.25 * 256
            CHECK(b.upload() && diff(a, b, &r) == 0);
            CHECK(near(r.max_tile, 0.25f));
            CHECK(near(r.region_mean[0], 0.25f) && near(r.region_mean[1], 0.25f));
            CHECK(near(r.region_mean[4], 0.25f) && near(r.region_mean[5], 0.25f));
            CHECK(r.region_mean[2] == 0.0f && r.region_mean[15] == 0.0f);
        }

        // a change in chroma only: one 8x8 tile of the second component
        // (Cr of 4:2:0's interleaved plane, or the Cr plane) +20; luma and
        // its regions see nothing
        b.fill(7);
        for (int y = 8; y < 16; y++)
            for (int x = 8; x < 16; x++) {
                if (a.f444) b.at(2, x, y) += 20 * unit;
                else b.at(1, x * 2 + 1, y) += 20 * unit;
            }
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 20.0f));
        CHECK(max_abs_region(r) == 0.0f);
        // and Cb: half a tile +20 -> 10
        b.fill(7);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 8; x++) {
                if (a.f444) b.at(1, x, y) += 20 * unit;
                else b.at(1, x * 2, y) += 20 * unit;
            }
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 10.0f));
    }

    // P010 / 10-bit MSB scaling: 4 10-bit levels are one 8-bit level, and
    // values below a whole 8-bit level stay fractional
    for (int fmt : {AJI_FMT_P010, AJI_FMT_YUV444P10MSB}) {
        Frame a(32, 32, fmt), b(32, 32, fmt);
        for (int p = 0; p < 3; p++)
            for (size_t i = 0; i < a.pl[p].size(); i++) {
                a.pl[p][i] = msb10(512);
                b.pl[p][i] = msb10(512);
            }
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++) b.at(0, x, y) = msb10(512 + 160);
        for (int y = 24; y < 32; y++)
            for (int x = 24; x < 32; x++) b.at(0, x, y) = msb10(512 - 2);
        CHECK(a.upload() && b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 40.0f));                          // 160 / 4
        CHECK(near(r.region_mean[0], 40.0f));                    // tile = region
        CHECK(near(r.region_mean[15], -0.5f));                   // -2 / 4
    }

    // mismatched or unreadable frames are refused
    {
        Frame a(16, 16, AJI_FMT_NV12), b(16, 16, AJI_FMT_P010), c(18, 16, AJI_FMT_NV12);
        CHECK(a.upload() && b.upload() && c.upload());
        const aji_frame fa = a.frame(), fb = b.frame(), fc = c.frame();
        CHECK(aji_frame_diff_launch(&fa, &fb, g_acc_dev, nullptr) == -1);
        CHECK(aji_frame_diff_launch(&fa, &fc, g_acc_dev, nullptr) == -1);
        aji_frame rgb = fa, rgb2 = fa;
        rgb.format = rgb2.format = AJI_FMT_RGB10A2;
        CHECK(aji_frame_diff_launch(&rgb, &rgb2, g_acc_dev, nullptr) == -1);
    }

    // 1080p: the grid regions cover the frame (a uniform +3 averages to 3)
    {
        Frame a(1920, 1080, AJI_FMT_NV12), b(1920, 1080, AJI_FMT_NV12);
        a.fill(3);
        b.fill(3);
        for (auto &v : b.pl[0]) v += 3;
        CHECK(a.upload() && b.upload() && diff(a, b, &r) == 0);
        CHECK(near(r.max_tile, 3.0f));
        for (int i = 0; i < 16; i++) CHECK(near(r.region_mean[i], 3.0f));
    }

    CHECK(cudaFreeHost(g_acc_host) == cudaSuccess);
    CHECK(cudaFree(g_acc_dev) == cudaSuccess);
    std::puts("frame diff tiles, regions and formats passed");
    return 0;
}
