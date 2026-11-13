// aji_frame_diff on DirectML (held-frame reuse in aji_encode): the cases of
// frame_diff.cu for every format the DirectML backend takes (NV12, P010,
// RGB10A2), through the dispatcher and shared D3D11 textures as aji_encode
// passes them. Every comparison runs twice, plain (waiting for all D3D11
// work) and with the frames' aji_d3d11_mark values (aji_frame_diff_marked,
// as aji_encode calls it), and both must agree. Where CUDA works, every
// NV12/P010 comparison is also run through kernels.cu's launcher on the
// same samples, and the results must be bit-identical.
//
//   aji_frame_diff_dml_test [--adapter N]
//       the synthetic cases (ctest -L gpu); N = DXGI adapter index
//   aji_frame_diff_dml_test [--adapter N] --raw FILE W H nv12|p010
//       real frames: each frame against the next and against the first,
//       DirectML vs CUDA (needs an NVIDIA GPU for the CUDA side)
//
// Runs from a folder holding aji.dll, aji_dml.dll, onnxruntime.dll and
// DirectML.dll (the build's dml-test folder); cudart is delay-loaded, so it
// runs on GPUs without the NVIDIA driver too (no cross-check then).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aji.h"
#include "kernels.h"

using Microsoft::WRL::ComPtr;

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

namespace {

ID3D11Device *g_dev;
ID3D11DeviceContext *g_ctx;
aji_ctx *g_aji;
bool g_cuda;                  // kernels.cu cross-check available
void *g_acc_dev, *g_acc_host;
int g_cross;                  // comparisons cross-checked bit-exactly

// A host frame in one format, uploaded into a shared D3D11 texture (and,
// for the cross-check, into pitched CUDA memory). NV12/P010: plane 0 luma,
// plane 1 interleaved CbCr; RGB10A2: plane 0 holds R, G, B per pixel.
// The texture may be larger than the frame (tw x th, as decoders pad their
// surfaces); the frame is its top-left corner and the rest holds `pad`.
struct Frame {
    int w, h, fmt;
    int tw, th;
    uint16_t pad = 0;              // padding samples (8-bit formats: low byte)
    bool wide, rgb;
    std::vector<uint16_t> pl[2];   // samples; 8-bit formats use the low byte
    int pw[2] = {0, 0}, ph[2] = {0, 0};
    ComPtr<ID3D11Texture2D> tex, st;
    void *dev[2] = {nullptr, nullptr};
    size_t pitch[2] = {0, 0};
    uint64_t mark = 0;             // aji_d3d11_mark after the last upload

    Frame(int w_, int h_, int fmt_, int tw_ = 0, int th_ = 0)
        : w(w_), h(h_), fmt(fmt_), tw(tw_ ? tw_ : w_), th(th_ ? th_ : h_)
    {
        wide = fmt == AJI_FMT_P010;
        rgb = fmt == AJI_FMT_RGB10A2;
        if (rgb) {
            pw[0] = w * 3;
            ph[0] = h;
        } else {
            pw[0] = w;
            ph[0] = h;
            pw[1] = w / 2 * 2;
            ph[1] = h / 2;
        }
        for (int p = 0; p < 2; p++)
            pl[p].assign((size_t)pw[p] * ph[p], 0);
    }
    ~Frame()
    {
        for (void *d : dev)
            if (d) cudaFree(d);
    }
    int unit() const { return wide ? 256 : rgb ? 4 : 1; }   // one 8-bit level

    // a deterministic pseudo-random picture in the middle of the range
    void fill(unsigned seed)
    {
        for (int p = 0; p < 2; p++)
            for (size_t i = 0; i < pl[p].size(); i++) {
                seed = seed * 1664525u + 1013904223u;
                const unsigned v8 = 60 + (seed >> 24) % 120;     // 60..179
                pl[p][i] = (uint16_t)(v8 * unit());
            }
    }
    uint16_t &at(int p, int x, int y) { return pl[p][(size_t)y * pw[p] + x]; }

    bool upload()
    {
        if (!tex) {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = tw;
            td.Height = th;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = rgb ? DXGI_FORMAT_R10G10B10A2_UNORM
                      : wide ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            td.MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                           D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
            if (FAILED(g_dev->CreateTexture2D(&td, NULL, &tex)))
                return false;
            td.Usage = D3D11_USAGE_STAGING;
            td.BindFlags = 0;
            td.MiscFlags = 0;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (FAILED(g_dev->CreateTexture2D(&td, NULL, &st)))
                return false;
        }
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(g_ctx->Map(st.Get(), 0, D3D11_MAP_WRITE, 0, &m)))
            return false;
        uint8_t *d = (uint8_t *)m.pData;
        for (int p = 0; p < (rgb ? 1 : 2); p++) {
            // the texture's plane: rows of tpw samples, tph of them
            const int tpw = rgb ? tw * 3 : p ? tw / 2 * 2 : tw;
            const int tph = p ? th / 2 : th;
            std::vector<uint16_t> line(tpw);
            for (int y = 0; y < tph; y++) {
                uint8_t *row = d + (size_t)y * m.RowPitch;
                for (int x = 0; x < tpw; x++)
                    line[x] = y < ph[p] && x < pw[p]
                                  ? pl[p][(size_t)y * pw[p] + x] : pad;
                const uint16_t *s = line.data();
                if (rgb) {
                    for (int x = 0; x < tw; x++) {
                        const uint32_t v = s[x * 3] | (s[x * 3 + 1] << 10) |
                                           ((uint32_t)s[x * 3 + 2] << 20) |
                                           (3u << 30);
                        std::memcpy(row + x * 4, &v, 4);
                    }
                } else if (wide) {
                    std::memcpy(row, s, (size_t)tpw * 2);
                } else {
                    for (int x = 0; x < tpw; x++)
                        row[x] = (uint8_t)s[x];
                }
            }
            d += (size_t)m.RowPitch * th;  // the CbCr plane follows the luma
        }
        g_ctx->Unmap(st.Get(), 0);
        g_ctx->CopyResource(tex.Get(), st.Get());
        mark = aji_d3d11_mark(g_aji);
        return rgb || !g_cuda || upload_cuda();
    }
    bool upload_cuda()
    {
        const int bpp = wide ? 2 : 1;
        for (int p = 0; p < 2; p++) {
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
        f.plane[0] = tex.Get();
        f.plane[1] = (void *)(intptr_t)0;
        return f;
    }
    aji_frame cuda_frame() const
    {
        aji_frame f = frame();
        for (int p = 0; p < 2; p++) {
            f.plane[p] = dev[p];
            f.stride[p] = (ptrdiff_t)pitch[p];
        }
        f.plane[2] = nullptr;
        return f;
    }
};

bool same(const aji_frame_diff_result &x, const aji_frame_diff_result &y)
{
    return std::memcmp(&x.max_tile, &y.max_tile, sizeof(float)) == 0 &&
           std::memcmp(x.region_mean, y.region_mean, sizeof(x.region_mean)) == 0;
}

void print(const char *what, const aji_frame_diff_result &r)
{
    std::fprintf(stderr, "  %s: D %.9g, regions", what, r.max_tile);
    for (float v : r.region_mean) std::fprintf(stderr, " %.9g", v);
    std::fputc('\n', stderr);
}

// DirectML's result in *r; with CUDA, kernels.cu's must be bit-identical
int diff(const Frame &a, const Frame &b, aji_frame_diff_result *r)
{
    const aji_frame fa = a.frame(), fb = b.frame();
    const int ret = aji_frame_diff(g_aji, &fa, &fb, r, nullptr);
    if (ret != AJI_OK) {
        std::fprintf(stderr, "aji_frame_diff: %d (%s)\n", ret,
                     aji_last_error(g_aji));
        return -1;
    }
    aji_frame_diff_result mr;
    const uint64_t m = a.mark > b.mark ? a.mark : b.mark;
    if (!m || aji_frame_diff_marked(g_aji, &fa, &fb, &mr, nullptr, m) != AJI_OK ||
        !same(*r, mr)) {
        std::fprintf(stderr, "aji_frame_diff_marked (mark %llu) differs: %s\n",
                     (unsigned long long)m, aji_last_error(g_aji));
        return -1;
    }
    if (!g_cuda || a.rgb)
        return 0;
    const aji_frame ca = a.cuda_frame(), cb = b.cuda_frame();
    aji_frame_diff_result cr;
    if (aji_frame_diff_launch(&ca, &cb, g_acc_dev, nullptr) != 0 ||
        cudaMemcpy(g_acc_host, g_acc_dev, AJI_FRAME_DIFF_ACC_BYTES,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    aji_frame_diff_finish(&ca, g_acc_host, &cr);
    if (!same(*r, cr)) {
        std::fprintf(stderr, "DirectML and CUDA differ (%dx%d format %d):\n",
                     a.w, a.h, a.fmt);
        print("DirectML", *r);
        print("CUDA", cr);
        return -1;
    }
    g_cross++;
    return 0;
}

bool approx(float got, float want, float tol = 1e-4f)
{
    return std::fabs(got - want) <= tol;
}

float max_abs_region(const aji_frame_diff_result &r)
{
    float m = 0;
    for (float v : r.region_mean) m = std::fmax(m, std::fabs(v));
    return m;
}

// 10-bit sample v as P010 stores it
uint16_t msb10(int v) { return (uint16_t)(v << 6); }

void log_cb(void *, int level, const char *msg)
{
    if (level <= 2 || std::getenv("AJI_DML_DEBUG"))
        std::fprintf(stderr, "[aji:%d] %s\n", level, msg);
}

bool cuda_available()
{
#ifdef AJI_CUDART_DLL
    if (!LoadLibraryA(AJI_CUDART_DLL))
        return false;
#endif
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0 &&
           cudaMalloc(&g_acc_dev, AJI_FRAME_DIFF_ACC_BYTES) == cudaSuccess &&
           cudaMallocHost(&g_acc_host, AJI_FRAME_DIFF_ACC_BYTES) == cudaSuccess;
}

int synthetic()
{
    aji_frame_diff_result r;
    const int fmts[] = {AJI_FMT_NV12, AJI_FMT_P010, AJI_FMT_RGB10A2};
    for (int fmt : fmts) {
        // 70x46: partial tiles at the right and bottom edges (35x23 chroma)
        const int W = 70, H = 46;
        Frame a(W, H, fmt), b(W, H, fmt);
        a.fill(7);
        b.fill(7);
        const int unit = a.unit();
        const int comps = a.rgb ? 3 : 1;   // RGB regions: mean over R, G, B
        const int c0 = a.rgb ? 3 : 1;      // samples per pixel in plane 0
        CHECK(a.upload() && b.upload());

        // identical frames
        CHECK(diff(a, b, &r) == 0);
        CHECK(r.max_tile == 0.0f && max_abs_region(r) == 0.0f);

        // one 8x8 luma (RGB: R) tile +40: D = 40, one region moves by
        // 40*64/area
        for (int y = 8; y < 16; y++)
            for (int x = 16; x < 24; x++)
                b.at(0, x * c0, y) += 40 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 40.0f));
        // 9x6 tiles: region columns of tiles {0,1,2},{3,4},{5,6},{7,8} and
        // rows {0,1},{2},{3,4},{5}; tile (2,1) is in region (0,0), 24x16 px
        CHECK(approx(r.region_mean[0], 40.0f * 64 / (24 * 16) / comps));
        for (int i = 1; i < 16; i++) CHECK(r.region_mean[i] == 0.0f);

        // the same tile -40 instead: the signed mean flips, D doesn't
        for (int y = 8; y < 16; y++)
            for (int x = 16; x < 24; x++)
                b.at(0, x * c0, y) -= 80 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 40.0f));
        CHECK(approx(r.region_mean[0], -40.0f * 64 / (24 * 16) / comps));

        // the partial bottom-right tile (6x6 px) +12: its mean, not /64
        b.fill(7);
        for (int y = 40; y < H; y++)
            for (int x = 64; x < W; x++)
                b.at(0, x * c0, y) += 12 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 12.0f));
        CHECK(approx(r.region_mean[15], 12.0f * 36 / (14 * 6) / comps));

        // a uniform +1 on every plane: D = 1, every region +1
        for (int p = 0; p < 2; p++)
            for (size_t i = 0; i < b.pl[p].size(); i++)
                b.pl[p][i] = (uint16_t)(a.pl[p][i] + unit);
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 1.0f));
        for (int i = 0; i < 16; i++) CHECK(approx(r.region_mean[i], 1.0f));

        // a fade over one quadrant: +0.25 levels on the top-left regions
        // only (P010: 64 of 256; RGB: one 10-bit level, every component);
        // every tile stays far below a held-frame threshold, the signed
        // means show it
        if (a.wide || a.rgb) {
            b.fill(7);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    if (x < 40 && y < 32)
                        for (int k = 0; k < c0; k++)
                            b.at(0, x * c0 + k, y) += (uint16_t)(unit / 4);
            CHECK(b.upload() && diff(a, b, &r) == 0);
            CHECK(approx(r.max_tile, 0.25f));
            CHECK(approx(r.region_mean[0], 0.25f) && approx(r.region_mean[1], 0.25f));
            CHECK(approx(r.region_mean[4], 0.25f) && approx(r.region_mean[5], 0.25f));
            CHECK(r.region_mean[2] == 0.0f && r.region_mean[15] == 0.0f);
        }

        if (a.rgb) {
            // G alone, one tile +20: D 20, its region a third of it
            b.fill(7);
            for (int y = 8; y < 16; y++)
                for (int x = 8; x < 16; x++)
                    b.at(0, x * 3 + 1, y) += 20 * unit;
            CHECK(b.upload() && diff(a, b, &r) == 0);
            CHECK(approx(r.max_tile, 20.0f));
            CHECK(approx(r.region_mean[0], 20.0f * 64 / (24 * 16) / 3));
            continue;
        }

        // a change in chroma only: one 8x8 tile of Cr in the interleaved
        // plane +20; luma and its regions see nothing
        b.fill(7);
        for (int y = 8; y < 16; y++)
            for (int x = 8; x < 16; x++)
                b.at(1, x * 2 + 1, y) += 20 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 20.0f));
        CHECK(max_abs_region(r) == 0.0f);
        // and Cb: half a tile +20 -> 10
        b.fill(7);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 8; x++)
                b.at(1, x * 2, y) += 20 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 10.0f));
        // the partial bottom-right chroma tile (3x7 of 35x23) +16
        b.fill(7);
        for (int y = 16; y < 23; y++)
            for (int x = 32; x < 35; x++)
                b.at(1, x * 2, y) += 16 * unit;
        CHECK(b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 16.0f));
        CHECK(max_abs_region(r) == 0.0f);
    }

    // P010 scaling: 4 10-bit levels are one 8-bit level, and values below
    // a whole 8-bit level stay fractional
    {
        Frame a(32, 32, AJI_FMT_P010), b(32, 32, AJI_FMT_P010);
        for (int p = 0; p < 2; p++)
            for (size_t i = 0; i < a.pl[p].size(); i++)
                a.pl[p][i] = b.pl[p][i] = msb10(512);
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++) b.at(0, x, y) = msb10(512 + 160);
        for (int y = 24; y < 32; y++)
            for (int x = 24; x < 32; x++) b.at(0, x, y) = msb10(512 - 2);
        CHECK(a.upload() && b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 40.0f));                          // 160 / 4
        CHECK(approx(r.region_mean[0], 40.0f));                    // tile = region
        CHECK(approx(r.region_mean[15], -0.5f));                   // -2 / 4
    }

    // mismatched or unreadable frames are refused
    {
        Frame a(16, 16, AJI_FMT_NV12), b(16, 16, AJI_FMT_P010), c(18, 16, AJI_FMT_NV12);
        CHECK(a.upload() && b.upload() && c.upload());
        const aji_frame fa = a.frame(), fb = b.frame(), fc = c.frame();
        CHECK(aji_frame_diff(g_aji, &fa, &fb, &r, nullptr) == AJI_ERR_FORMAT);
        CHECK(aji_frame_diff(g_aji, &fa, &fc, &r, nullptr) == AJI_ERR_SHAPE);
        aji_frame f444 = fa, g444 = fa;
        f444.format = g444.format = AJI_FMT_YUV444P16;   // CUDA only
        CHECK(aji_frame_diff(g_aji, &f444, &g444, &r, nullptr) == AJI_ERR_FORMAT);
        // a mark this context never issued
        const uint64_t next = aji_d3d11_mark(g_aji) + 1;
        CHECK(aji_frame_diff_marked(g_aji, &fa, &fa, &r, nullptr, next) == AJI_ERR);
        CHECK(aji_frame_diff_marked(g_aji, &fa, &fa, &r, nullptr, 0) == AJI_OK);
    }

    // marks: one increasing counter with the plain call's own signals; a
    // marked comparison sees every write before its mark, also with more
    // D3D11 work queued after the mark (which it need not wait for: the
    // times are printed, not checked); a mark that passed long ago is fine
    {
        Frame a(1920, 1080, AJI_FMT_P010), b(1920, 1080, AJI_FMT_P010);
        a.fill(5);
        b.fill(5);
        CHECK(a.upload() && b.upload());
        CHECK(b.mark > a.mark);
        const uint64_t m1 = aji_d3d11_mark(g_aji), m2 = aji_d3d11_mark(g_aji);
        CHECK(m1 > b.mark && m2 == m1 + 1);
        const aji_frame fa = a.frame(), fb = b.frame();
        CHECK(aji_frame_diff(g_aji, &fa, &fb, &r, nullptr) == AJI_OK);
        CHECK(r.max_tile == 0.0f);
        CHECK(aji_d3d11_mark(g_aji) == m2 + 2);   // the plain call took m2 + 1

        // filler: 4K RGBA16F copies on the context, after b's next upload
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = 3840;
        td.Height = 2160;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> f0, f1;
        CHECK(SUCCEEDED(g_dev->CreateTexture2D(&td, NULL, &f0)) &&
              SUCCEEDED(g_dev->CreateTexture2D(&td, NULL, &f1)));
        double t_marked = 0, t_plain = 0;
        for (int round = 1; round <= 3; round++) {
            for (auto &v : b.pl[0]) v = (uint16_t)(v + 2 * b.unit());
            CHECK(b.upload());
            for (int i = 0; i < 64; i++)
                g_ctx->CopyResource(i & 1 ? f0.Get() : f1.Get(),
                                    i & 1 ? f1.Get() : f0.Get());
            g_ctx->Flush();
            LARGE_INTEGER q0, q1, q2, fq;
            QueryPerformanceFrequency(&fq);
            QueryPerformanceCounter(&q0);
            aji_frame_diff_result mr;
            CHECK(aji_frame_diff_marked(g_aji, &fa, &fb, &mr, nullptr, b.mark) ==
                  AJI_OK);
            QueryPerformanceCounter(&q1);
            CHECK(approx(mr.max_tile, 2.0f * round));
            for (int i = 0; i < 64; i++)
                g_ctx->CopyResource(i & 1 ? f0.Get() : f1.Get(),
                                    i & 1 ? f1.Get() : f0.Get());
            CHECK(diff(a, b, &r) == 0 && same(r, mr));   // plain, then marked
            QueryPerformanceCounter(&q2);
            t_marked += 1e3 * (q1.QuadPart - q0.QuadPart) / fq.QuadPart;
            t_plain += 1e3 * (q2.QuadPart - q1.QuadPart) / fq.QuadPart;
        }
        CHECK(aji_frame_diff_marked(g_aji, &fa, &fb, &r, nullptr, 1) == AJI_OK);
        CHECK(approx(r.max_tile, 6.0f));
        std::printf("marks: behind 64 queued 4K copies, a marked comparison "
                    "took %.2f ms, a plain one (and a marked) %.2f ms\n",
                    t_marked / 3, t_plain / 3);
    }

    // 1080p: the grid regions cover the frame (a uniform +3 averages to 3);
    // the same staging, then another size
    for (int fmt : {AJI_FMT_NV12, AJI_FMT_P010}) {
        Frame a(1920, 1080, fmt), b(1920, 1080, fmt);
        a.fill(3);
        b.fill(3);
        for (auto &v : b.pl[0]) v = (uint16_t)(v + 3 * b.unit());
        CHECK(a.upload() && b.upload() && diff(a, b, &r) == 0);
        CHECK(approx(r.max_tile, 3.0f));
        for (int i = 0; i < 16; i++) CHECK(approx(r.region_mean[i], 3.0f));
    }

    // frames held in textures larger than the frame (decoder surfaces are
    // padded: 1920x1088 for 1080p): only the frame is read, so the results
    // equal those of exact-size textures bit for bit, whatever the padding
    // holds (a's padding near white, b's near black)
    {
        struct { int fmt, w, h, tw, th; } pads[] = {
            {AJI_FMT_NV12, 1920, 1080, 1920, 1088},
            {AJI_FMT_P010, 1920, 1080, 1920, 1088},
            {AJI_FMT_NV12, 1918, 1078, 1920, 1088},   // width padded too
            {AJI_FMT_P010, 70, 46, 96, 64},
            {AJI_FMT_NV12, 70, 46, 72, 46},           // width only
            {AJI_FMT_RGB10A2, 69, 45, 77, 51},        // odd padding
        };
        for (const auto &pc : pads) {
            Frame a(pc.w, pc.h, pc.fmt), b(pc.w, pc.h, pc.fmt);
            Frame pa(pc.w, pc.h, pc.fmt, pc.tw, pc.th),
                  pb(pc.w, pc.h, pc.fmt, pc.tw, pc.th);
            pa.pad = (uint16_t)(250 * pa.unit());
            pb.pad = (uint16_t)(5 * pb.unit());
            for (unsigned s = 1; s <= 3; s++) {
                // s 1: the same picture (D 0 only if no padding is read);
                // then noise against noise
                a.fill(s);
                b.fill(s == 1 ? s : s * 77);
                pa.pl[0] = a.pl[0]; pa.pl[1] = a.pl[1];
                pb.pl[0] = b.pl[0]; pb.pl[1] = b.pl[1];
                CHECK(a.upload() && b.upload() && pa.upload() && pb.upload());
                aji_frame_diff_result re, rp, rm;
                CHECK(diff(a, b, &re) == 0);
                CHECK(diff(pa, pb, &rp) == 0);
                CHECK(diff(pa, b, &rm) == 0);
                if (!same(re, rp) || !same(re, rm)) {
                    std::fprintf(stderr, "%dx%d format %d in %dx%d textures "
                                 "differs from exact-size textures:\n",
                                 pc.w, pc.h, pc.fmt, pc.tw, pc.th);
                    print("exact", re);
                    print("padded", rp);
                    print("padded a, exact b", rm);
                    return 1;
                }
                CHECK(s == 1 ? re.max_tile == 0.0f : re.max_tile > 0.0f);
            }
        }
    }

    // noise: random frames against each other, at odd tile counts (the
    // cross-check's main course: every tile and region differs)
    for (int fmt : {AJI_FMT_NV12, AJI_FMT_P010}) {
        Frame a(1366, 766, fmt), b(1366, 766, fmt);
        for (unsigned s = 1; s <= 4; s++) {
            a.fill(s);
            b.fill(s * 77);
            CHECK(a.upload() && b.upload() && diff(a, b, &r) == 0);
            CHECK(r.max_tile > 0.0f);
        }
    }
    std::printf("frame diff on DirectML: tiles, regions and formats passed; "
                "%d comparisons bit-identical to CUDA%s\n", g_cross,
                g_cuda ? "" : " (no CUDA GPU: not cross-checked)");
    return 0;
}

// Real frames from a raw file: each frame against the next and against the
// first, DirectML vs CUDA bit for bit.
int raw(const char *path, int w, int h, int fmt)
{
    if (!g_cuda) {
        std::fprintf(stderr, "--raw needs CUDA for the cross-check\n");
        return 1;
    }
    const size_t bpp = fmt == AJI_FMT_P010 ? 2 : 1;
    const size_t fbytes = (size_t)w * h * 3 / 2 * bpp;
    FILE *f = std::fopen(path, "rb");
    CHECK(f);
    std::vector<Frame *> frames;
    std::vector<uint8_t> buf(fbytes);
    while (frames.size() < 64 && std::fread(buf.data(), 1, fbytes, f) == fbytes) {
        Frame *fr = new Frame(w, h, fmt);
        for (int p = 0; p < 2; p++) {
            const uint8_t *s = buf.data() + (p ? (size_t)w * h * bpp : 0);
            for (size_t i = 0; i < fr->pl[p].size(); i++) {
                uint16_t v = 0;
                std::memcpy(&v, s + i * bpp, bpp);
                fr->pl[p][i] = v;
            }
        }
        CHECK(fr->upload());
        frames.push_back(fr);
    }
    std::fclose(f);
    CHECK(frames.size() >= 2);
    aji_frame_diff_result r;
    for (size_t i = 1; i < frames.size(); i++) {
        CHECK(diff(*frames[i - 1], *frames[i], &r) == 0);
        std::printf("frame %zu vs %zu: D %.4f\n", i - 1, i, r.max_tile);
        CHECK(diff(*frames[0], *frames[i], &r) == 0);
    }
    std::printf("%s: %zu frames, %d comparisons bit-identical to CUDA\n",
                path, frames.size(), g_cross);
    for (Frame *fr : frames) delete fr;
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    int adapter = -1;
    const char *raw_path = nullptr;
    int rw = 0, rh = 0, rfmt = AJI_FMT_NV12;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--adapter") && i + 1 < argc) {
            adapter = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--raw") && i + 4 < argc) {
            raw_path = argv[++i];
            rw = std::atoi(argv[++i]);
            rh = std::atoi(argv[++i]);
            rfmt = std::strcmp(argv[++i], "p010") ? AJI_FMT_NV12 : AJI_FMT_P010;
        } else {
            std::fprintf(stderr, "usage: aji_frame_diff_dml_test [--adapter N] "
                                 "[--raw FILE W H nv12|p010]\n");
            return 2;
        }
    }

    ComPtr<IDXGIFactory1> fac;
    ComPtr<IDXGIAdapter1> ad;
    CHECK(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac))));
    CHECK(SUCCEEDED(fac->EnumAdapters1(adapter < 0 ? 0 : (UINT)adapter, &ad)));
    DXGI_ADAPTER_DESC1 desc;
    ad->GetDesc1(&desc);
    std::printf("adapter: %ls\n", desc.Description);
    CHECK(SUCCEEDED(D3D11CreateDevice(ad.Get(), D3D_DRIVER_TYPE_UNKNOWN, NULL,
                                      0, NULL, 0, D3D11_SDK_VERSION, &g_dev,
                                      NULL, &g_ctx)));

    // a conf that only names the backend: aji_frame_diff needs no chain
    char tmp[MAX_PATH], conf[MAX_PATH];
    CHECK(GetTempPathA(MAX_PATH, tmp) > 0);
    std::snprintf(conf, sizeof conf, "%saji_frame_diff_dml_%lu.conf", tmp,
                  GetCurrentProcessId());
    FILE *cf = std::fopen(conf, "w");
    CHECK(cf);
    std::fputs("[global]\nconfig_version=2\nbackend=DirectML\n", cf);
    std::fclose(cf);
    aji_create_params p = {};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf;
    p.model_dir = tmp;
    p.slot = 1;
    p.d3d11_device = g_dev;
    p.log = log_cb;
    g_aji = aji_create(&p);
    std::remove(conf);
    CHECK(g_aji);

    g_cuda = cuda_available();
    const int rc = raw_path ? raw(raw_path, rw, rh, rfmt) : synthetic();
    aji_destroy(&g_aji);
    g_ctx->Release();
    g_dev->Release();
    return rc;
}
