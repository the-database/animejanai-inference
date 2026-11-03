/*
 * core/tiling.h — tiled whole-image inference for the image front-end.
 *
 * A backend supplies a TileModel that runs one fixed-size tile
 * (planar float [0,1], host memory); upscale_tiled() covers the image
 * with overlapping tiles, reflect-pads images smaller than a tile,
 * blends overlaps with separable linear ramps and writes 8-bit output.
 *
 * Blending is exact at any overlap: each tile's 1-D ramp weights are
 * normalized by the per-axis weight sums, so the 2-D weights of all
 * tiles covering a pixel sum to one (corners included) — including the
 * flush-to-edge last tile, whose overlap is usually wider than the
 * nominal one. Output is accumulated one tile row at a time in a float
 * band (tile_h * scale rows), never as a full-size float image.
 */

#ifndef AJI_CORE_TILING_H
#define AJI_CORE_TILING_H

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "aji_image.h"

namespace ajicore {

// Run fn(begin, end) over [0, n) split across a few threads (the calling
// thread included), for the host path's per-element passes. Each element
// is computed exactly as it would be on one thread, so results don't
// depend on the split. Small jobs stay on the calling thread; at most half
// the hardware threads (and 8) are used, leaving the rest to the caller's
// own decode/encode workers.
template <typename F> void parallel_for(size_t n, size_t min_per_thread, F &&fn)
{
    static const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t k = std::min<size_t>(
        {std::max<size_t>(1, hw / 2), 8, n / std::max<size_t>(1, min_per_thread)});
    if (k <= 1) {
        fn((size_t)0, n);
        return;
    }
    std::vector<std::thread> ts;
    ts.reserve(k - 1);
    for (size_t t = 1; t < k; t++)
        ts.emplace_back([&, t] { fn(n * t / k, n * (t + 1) / k); });
    fn((size_t)0, n / k);
    for (auto &th : ts)
        th.join();
}

// Defaults for aji_img_model_params. 32 px of overlap keeps tile seams
// below visibility for ESRGAN-class receptive fields (16 leaves faint
// differences against a whole-image run on screentone).
const int DEFAULT_TILE = 512;
const int DEFAULT_OVERLAP = 32;

// Per-image phase timings (milliseconds), filled when AJI_IMG_PROFILE is
// set: host phases by upscale_tiled, the run_tile breakdown by a backend
// that reports it (gpu_* from CUDA events, wait = host time blocked on
// the stream).
struct TileProfile {
    bool on = false;
    int tiles = 0;
    double prep = 0, pack = 0, wait = 0, unpack = 0, blend = 0, flush = 0;
    double gpu_h2d = 0, gpu_infer = 0, gpu_d2h = 0;
};

class TileModel {
public:
    int tile_w = 0, tile_h = 0;  // input pixels
    int scale = 0;
    int in_ch = 3, out_ch = 3;
    TileProfile prof;

    virtual ~TileModel() = default;
    // in: in_ch planes of tile_h x tile_w; out: out_ch planes of
    // (tile_h*scale) x (tile_w*scale). Returns an aji_img_status.
    virtual int run_tile(const float *in, float *out, std::string *err) = 0;
};

// Tile origins along one axis of length `len` for tile size `tile`:
// step tile-overlap, the last tile flush with the far edge; a single
// origin 0 when the axis fits in one tile.
std::vector<int> tile_origins(int len, int tile, int overlap);

// Mirror an index into [0, n) without repeating the edge sample
// (numpy "reflect"), for any distance past the edge.
int reflect(int i, int n);

// Normalized 1-D blend weights per tile along one axis, in output pixels:
// w[t][u] for u in [0, span*scale), span = min(tile, len). Linear ramps
// across each actual overlap (none at the image border), divided by the
// per-position sum so the tiles covering any position sum to one.
std::vector<std::vector<float>> axis_weights(const std::vector<int> &org,
                                             int len, int tile, int scale);

int upscale_tiled(TileModel &m, int overlap, const aji_img_buffer *in,
                  const aji_img_buffer *out, aji_img_progress_fn progress,
                  void *opaque, std::string *err);

} // namespace ajicore

#endif
