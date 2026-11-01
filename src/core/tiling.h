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

#include <string>
#include <vector>

#include "aji_image.h"

namespace ajicore {

// Defaults for aji_img_model_params. 32 px of overlap keeps tile seams
// below visibility for ESRGAN-class receptive fields (16 leaves faint
// differences against a whole-image run on screentone).
const int DEFAULT_TILE = 512;
const int DEFAULT_OVERLAP = 32;

class TileModel {
public:
    int tile_w = 0, tile_h = 0;  // input pixels
    int scale = 0;
    int in_ch = 3, out_ch = 3;

    virtual ~TileModel() = default;
    // in: in_ch planes of tile_h x tile_w; out: out_ch planes of
    // (tile_h*scale) x (tile_w*scale). Returns an aji_img_status.
    virtual int run_tile(const float *in, float *out, std::string *err) = 0;
};

// Tile origins along one axis of length `len` for tile size `tile`:
// step tile-overlap, the last tile flush with the far edge; a single
// origin 0 when the axis fits in one tile.
std::vector<int> tile_origins(int len, int tile, int overlap);

int upscale_tiled(TileModel &m, int overlap, const aji_img_buffer *in,
                  const aji_img_buffer *out, aji_img_progress_fn progress,
                  void *opaque, std::string *err);

} // namespace ajicore

#endif
