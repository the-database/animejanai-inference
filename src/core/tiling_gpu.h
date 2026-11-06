/*
 * core/tiling_gpu.h — the image front-end's tiling on the GPU, for the
 * CUDA backends (aji_trt, aji_trtrtx).
 *
 * The same cover / reflect pad / overlap blend as core/tiling's
 * upscale_tiled, with the image data kept on the device: the 8-bit input
 * is uploaded once, each tile is cut out and converted to the engine's
 * input dtype by a kernel, blended into a float band by another, and
 * finished rows are converted to 8-bit and copied back once per tile row.
 * The host only launches work, so the GPU is not left idle while the CPU
 * converts and blends.
 *
 * The arithmetic matches upscale_tiled's step for step (explicitly
 * rounded multiplies and adds, never fused; the same fp16/bf16
 * conversions as core/half.h), so the output is bit-identical to the
 * CPU path's.
 */

#ifndef AJI_CORE_TILING_GPU_H
#define AJI_CORE_TILING_GPU_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ajicore {

// The engine tensors' element type.
enum class GpuDtype { F32 = 0, F16 = 1, BF16 = 2 };

// Device scratch for upscale_gpu, kept across calls (grown, never shrunk).
struct GpuTileScratch {
    uint8_t *img = nullptr;   // the 8-bit input image, tightly packed
    size_t img_cap = 0;
    float *band = nullptr;    // Co planes x band_rows x OW, a ring of rows
    size_t band_cap = 0;
    uint8_t *rows = nullptr;  // flushed 8-bit rows, interleaved
    size_t rows_cap = 0;
    float *wx = nullptr, *wy = nullptr;  // per-tile axis weights, packed
    size_t wx_cap = 0, wy_cap = 0;

    void release();
};

// Cut tile (ox, oy) out of the device image (W x H x ci, packed), reflect-
// padded to tw x th, and write it as the engine input: in_ch planes of
// dtype `dt`, values as upscale_tiled computes them.
cudaError_t gpu_tile_extract(const uint8_t *img, int W, int H, int ci,
                             int ox, int oy, int tw, int th, int in_ch,
                             float inv255, void *dst, GpuDtype dt,
                             cudaStream_t s);

// Blend the engine output (out_ch planes of tow x toh, dtype `dt`) into the
// band: for output columns [ux0, ux0 + cols) and rows [uy0, uy0 + rows)
// of the image, band += wy[v] * wx[u] * tile. Band rows are a ring:
// image row Y lives at band row Y % band_rows.
cudaError_t gpu_tile_blend(const void *tile, GpuDtype dt, int out_ch,
                           int tow, int toh, const float *wx, const float *wy,
                           int cols, int rows, int ux0, int uy0, float *band,
                           int co, int band_rows, int OW, cudaStream_t s);

// Convert image rows [y0, y0 + n) of the band to 8-bit interleaved rows in
// `dst` (n x OW x co) and zero them in the band for reuse.
cudaError_t gpu_band_flush(float *band, int co, int band_rows, int OW,
                           int y0, int n, uint8_t *dst, cudaStream_t s);

} // namespace ajicore

#endif
