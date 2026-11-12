/*
 * aji_frame_diff's host-side arithmetic, shared by the CUDA backends
 * (kernels.cu) and DirectML (aji_dml.cpp) so that the same frames give the
 * same floats on both: the per-tile scale table the GPU multiplies by, and
 * the region means from the GPU's integer sums.
 */
#ifndef AJI_FRAME_DIFF_H
#define AJI_FRAME_DIFF_H

#include "aji.h"

#ifdef __CUDACC__
#define AJI_FD_HD __host__ __device__
#else
#define AJI_FD_HD
#endif

/* 8-bit units: 16-bit containers are divided by 256, packed 10-bit RGB
 * (DirectML only) by 4. */
AJI_FD_HD static inline float aji_frame_diff_scale(int format)
{
    return format == AJI_FMT_NV12      ? 1.0f
         : format == AJI_FMT_RGB10A2   ? 1.0f / 4.0f
                                       : 1.0f / 256.0f;
}

/* A tile of n pixels (1-64) has mean |diff| = (float)sum * this, computed
 * as k_tile_diff computes its inv (one rounding of scale / n, then one of
 * the product). */
AJI_FD_HD static inline float aji_frame_diff_tile_inv(float scale, int n)
{
    return scale / (float)n;
}

/* The signed region means from the 16 integer sums of the luma plane (for
 * packed RGB, the sums over its `comps` components): sum * scale / pixels,
 * where region rx holds the tiles tx with tx * 4 / tiles_x == rx. */
static inline void aji_frame_diff_regions(int width, int height, int format,
                                          int comps, const long long sum[16],
                                          float out[16])
{
    const double scale = format == AJI_FMT_NV12    ? 1.0
                       : format == AJI_FMT_RGB10A2 ? 1.0 / 4.0
                                                   : 1.0 / 256.0;
    const int tiles[2] = {(width + 7) / 8, (height + 7) / 8};
    const int dims[2] = {width, height};
    int span[2][4];   /* pixels per grid column / row */
    for (int d = 0; d < 2; d++) {
        for (int i = 0; i < 4; i++) {
            /* the tiles t with t * 4 / tiles == i: [ceil(i*T/4), ceil((i+1)*T/4)) */
            const int lo = (i * tiles[d] + 3) / 4;
            const int hi = ((i + 1) * tiles[d] + 3) / 4;
            int end = hi * 8;
            if (end > dims[d])
                end = dims[d];
            span[d][i] = hi > lo ? end - lo * 8 : 0;
        }
    }
    for (int ry = 0; ry < 4; ry++) {
        for (int rx = 0; rx < 4; rx++) {
            const double n = (double)span[0][rx] * span[1][ry] * comps;
            out[ry * 4 + rx] =
                n > 0 ? (float)(sum[ry * 4 + rx] * scale / n) : 0.0f;
        }
    }
}

#endif
