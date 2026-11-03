/*
 * core/tiling.cpp — see tiling.h.
 */

#include "tiling.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace ajicore {

std::vector<int> tile_origins(int len, int tile, int overlap)
{
    if (len <= tile)
        return {0};
    const int step = std::max(1, tile - overlap);
    const int n = std::max(1, (len - overlap + step - 1) / step);
    std::vector<int> o;
    for (int i = 0; i < n; i++) {
        const int p = std::min(i * step, len - tile);
        if (o.empty() || p != o.back())
            o.push_back(p);
    }
    return o;
}

int reflect(int i, int n)
{
    if (n == 1)
        return 0;
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0)
        i += period;
    return i < n ? i : period - i;
}

std::vector<std::vector<float>> axis_weights(const std::vector<int> &org,
                                             int len, int tile, int scale)
{
    const int span = std::min(tile, len);
    const int n = (int)org.size();
    const int S = span * scale;
    std::vector<std::vector<float>> w(n, std::vector<float>(S, 1.0f));
    for (int t = 0; t < n; t++) {
        const int ol = t > 0 ? (org[t - 1] + span - org[t]) * scale : 0;
        const int orr = t + 1 < n ? (org[t] + span - org[t + 1]) * scale : 0;
        for (int u = 0; u < S; u++) {
            float v = 1.0f;
            if (u < ol)
                v = std::min(v, (u + 0.5f) / ol);
            if (u >= S - orr)
                v = std::min(v, (S - u - 0.5f) / orr);
            w[t][u] = v;
        }
    }
    std::vector<float> sum((size_t)len * scale, 0.0f);
    for (int t = 0; t < n; t++)
        for (int u = 0; u < S; u++)
            sum[(size_t)org[t] * scale + u] += w[t][u];
    for (int t = 0; t < n; t++)
        for (int u = 0; u < S; u++)
            w[t][u] /= sum[(size_t)org[t] * scale + u];
    return w;
}

namespace {

inline uint8_t to_u8(float v)
{
    const float f = v * 255.0f + 0.5f;
    return f <= 0.0f ? 0 : f >= 255.0f ? 255 : (uint8_t)f;
}

} // namespace

int upscale_tiled(TileModel &m, int overlap, const aji_img_buffer *in,
                  const aji_img_buffer *out, aji_img_progress_fn progress,
                  void *opaque, std::string *err)
{
    if (!in || !out || !in->data || !out->data) {
        *err = "null image buffer";
        return AJI_IMG_ERR_ARGS;
    }
    const int W = in->width, H = in->height, s = m.scale;
    const int Ci = in->channels, Co = out->channels;
    if (W < 1 || H < 1 || (Ci != 1 && Ci != 3) || Co != Ci) {
        *err = "image must be 1 or 3 channels, same on both sides";
        return AJI_IMG_ERR_ARGS;
    }
    if (out->width != W * s || out->height != H * s) {
        *err = "output must be " + std::to_string(W * s) + "x" +
               std::to_string(H * s) + " (scale " + std::to_string(s) + ")";
        return AJI_IMG_ERR_ARGS;
    }
    if (in->stride < (ptrdiff_t)W * Ci || out->stride < (ptrdiff_t)W * s * Co) {
        *err = "stride smaller than a row";
        return AJI_IMG_ERR_ARGS;
    }

    const int Tw = m.tile_w, Th = m.tile_h;
    const std::vector<int> ox = tile_origins(W, Tw, overlap);
    const std::vector<int> oy = tile_origins(H, Th, overlap);
    const int spw = std::min(Tw, W), sph = std::min(Th, H);
    const int OW = W * s, OH = H * s;
    const int tow = Tw * s, toh = Th * s;  // model output tile
    const int band_rows = sph * s;
    const size_t in_plane = (size_t)Tw * Th, out_plane = (size_t)tow * toh;

    using clk = std::chrono::steady_clock;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    TileProfile &P = m.prof;
    P = TileProfile{};
    P.on = std::getenv("AJI_IMG_PROFILE") != nullptr;
    const auto t_start = clk::now();
    auto t0 = t_start;

    try {
        const auto wx = axis_weights(ox, W, Tw, s);
        const auto wy = axis_weights(oy, H, Th, s);
        std::vector<float> tin(in_plane * m.in_ch), tout(out_plane * m.out_ch);
        std::vector<float> band((size_t)Co * band_rows * OW, 0.0f);
        std::vector<int> xs(Tw);
        const float inv255 = 1.0f / 255.0f;
        const int total = (int)(ox.size() * oy.size());
        int done = 0;
        int band_y0 = 0;  // output row of band row 0

        for (size_t j = 0; j < oy.size(); j++) {
            for (size_t i = 0; i < ox.size(); i++) {
                if (P.on)
                    t0 = clk::now();
                for (int x = 0; x < Tw; x++)
                    xs[x] = reflect(ox[i] + x, W);
                for (int y = 0; y < Th; y++) {
                    const uint8_t *row =
                        in->data + (ptrdiff_t)reflect(oy[j] + y, H) * in->stride;
                    float *d = tin.data() + (size_t)y * Tw;
                    for (int x = 0; x < Tw; x++) {
                        const uint8_t *p = row + (size_t)xs[x] * Ci;
                        if (m.in_ch == 3) {
                            const int g = Ci == 1;
                            d[x] = p[0] * inv255;
                            d[x + in_plane] = p[g ? 0 : 1] * inv255;
                            d[x + 2 * in_plane] = p[g ? 0 : 2] * inv255;
                        } else {
                            d[x] = Ci == 1 ? p[0] * inv255
                                           : (0.299f * p[0] + 0.587f * p[1] +
                                              0.114f * p[2]) * inv255;
                        }
                    }
                }

                if (P.on) {
                    const auto t1 = clk::now();
                    P.prep += ms(t0, t1);
                    t0 = t1;
                }
                const int rc = m.run_tile(tin.data(), tout.data(), err);
                if (rc != AJI_IMG_OK)
                    return rc;
                if (P.on)
                    t0 = clk::now();

                // The band is a ring of rows: image row Y lives at band row
                // Y % band_rows, so finished rows are reused without moving
                // the still-open ones.
                const int ux0 = ox[i] * s, uy0 = oy[j] * s;
                const std::vector<float> &wxi = wx[i], &wyj = wy[j];
                const int vrows = sph * s, cols = spw * s;
                parallel_for((size_t)Co * vrows, 64, [&](size_t b, size_t e) {
                    for (size_t k = b; k < e; k++) {
                        const int c = (int)(k / vrows), v = (int)(k % vrows);
                        const int mc = (m.out_ch == 1 || Co == 1) ? 0 : c;
                        const float *srow =
                            tout.data() + mc * out_plane + (size_t)v * tow;
                        const float wv = wyj[v];
                        const int r = (uy0 + v) % band_rows;
                        float *brow = band.data() +
                                      ((size_t)c * band_rows + r) * OW + ux0;
                        for (int u = 0; u < cols; u++)
                            brow[u] += wv * wxi[u] * srow[u];
                    }
                });

                if (P.on) {
                    P.blend += ms(t0, clk::now());
                    P.tiles++;
                }
                done++;
                if (progress && progress(opaque, done, total)) {
                    *err = "cancelled";
                    return AJI_IMG_CANCELLED;
                }
            }

            // Rows above the next tile row's top are final: write them out
            // (each output row front to back) and zero their ring rows.
            if (P.on)
                t0 = clk::now();
            const int next_y0 = j + 1 < oy.size() ? oy[j + 1] * s : OH;
            const int y_first = band_y0;
            parallel_for((size_t)(next_y0 - y_first), 32, [&](size_t b, size_t e) {
                for (int Y = y_first + (int)b; Y < y_first + (int)e; Y++) {
                    uint8_t *orow = out->data + (ptrdiff_t)Y * out->stride;
                    const int r = Y % band_rows;
                    float *b0 = band.data() + (size_t)r * OW;
                    if (Co == 3) {
                        float *b1 = b0 + (size_t)band_rows * OW;
                        float *b2 = b1 + (size_t)band_rows * OW;
                        for (int X = 0; X < OW; X++) {
                            orow[3 * (size_t)X] = to_u8(b0[X]);
                            orow[3 * (size_t)X + 1] = to_u8(b1[X]);
                            orow[3 * (size_t)X + 2] = to_u8(b2[X]);
                        }
                        std::fill(b1, b1 + OW, 0.0f);
                        std::fill(b2, b2 + OW, 0.0f);
                    } else {
                        for (int X = 0; X < OW; X++)
                            orow[X] = to_u8(b0[X]);
                    }
                    std::fill(b0, b0 + OW, 0.0f);
                }
            });
            band_y0 = next_y0;
            if (P.on)
                P.flush += ms(t0, clk::now());
        }
    } catch (const std::bad_alloc &) {
        *err = "out of host memory for tiling buffers";
        return AJI_IMG_ERR_NO_MEM;
    }
    if (P.on) {
        const double total_ms = ms(t_start, clk::now());
        const double gpu = P.gpu_h2d + P.gpu_infer + P.gpu_d2h;
        const double host = P.prep + P.pack + P.unpack + P.blend + P.flush;
        std::fprintf(stderr,
                     "aji-profile %dx%d x%d tiles=%d tile=%dx%d total=%.1fms | "
                     "host: prep=%.1f pack=%.1f unpack=%.1f blend=%.1f "
                     "flush=%.1f (sum %.1f, %.0f%%) | wait=%.1f | gpu: "
                     "h2d=%.1f infer=%.1f d2h=%.1f (sum %.1f, %.0f%% busy)\n",
                     W, H, s, P.tiles, Tw, Th, total_ms, P.prep, P.pack,
                     P.unpack, P.blend, P.flush, host,
                     100.0 * host / total_ms, P.wait, P.gpu_h2d, P.gpu_infer,
                     P.gpu_d2h, gpu, 100.0 * gpu / total_ms);
    }
    return AJI_IMG_OK;
}

} // namespace ajicore
