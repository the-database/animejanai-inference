/*
 * Shared CPU-side resampling/colorspace math for the aji backends.
 *
 * Replicates zimg's compute_filter (what VapourSynth resize — the
 * reference pipeline — uses): per-output-pixel weight tables built in
 * double precision, kernel stretched by 1/scale on downscale,
 * taps = ceil(support/step)*2 (min 1), positions mirrored then clamped
 * at the borders, weights normalized to 1.
 *
 * Both the CUDA backend (kernels.cu) and the DirectML backend
 * (aji_dml.cpp) consume these tables; the device apply loops must match
 * each other exactly, so the table math lives here once.
 */
#pragma once

#include <math.h>

#include <vector>

#include "aji.h"
#include "kernels.h"

namespace aji_resample {

struct weights {
    int taps = 0, src = 0, dst = 0;
    std::vector<int> start;   // [dst] first source index per output
    std::vector<float> wt;    // [dst * taps], normalized
};

inline double spline36(double x)
{
    x = fabs(x);
    if (x < 1.0)
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
    if (x < 2.0) {
        x -= 1.0;
        return ((-6.0 / 11.0 * x + 270.0 / 209.0) * x - 156.0 / 209.0) * x;
    }
    if (x < 3.0) {
        x -= 2.0;
        return ((1.0 / 11.0 * x - 45.0 / 209.0) * x + 26.0 / 209.0) * x;
    }
    return 0.0;
}

inline double bilinear(double x)
{
    x = fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

inline weights compute(int src, int dst, double shift, int filter)
{
    const double scale = (double)dst / src;
    const double step = scale < 1.0 ? scale : 1.0;
    double (*f)(double) = filter == AJI_FILTER_BILINEAR ? bilinear : spline36;
    const double support = filter == AJI_FILTER_BILINEAR ? 1.0 : 3.0;
    int taps = (int)ceil(support / step) * 2;
    if (taps < 1)
        taps = 1;

    weights w;
    w.taps = taps;
    w.src = src;
    w.dst = dst;
    w.start.resize(dst);
    w.wt.resize((size_t)dst * taps);
    std::vector<double> row(taps);
    for (int i = 0; i < dst; i++) {
        const double pos = (i + 0.5) / scale + shift;
        // round_halfup(pos - taps/2) + 0.5 = first tap center
        const double begin = floor(pos - taps / 2.0 + 0.5) + 0.5;
        double sum = 0.0;
        for (int j = 0; j < taps; j++) {
            row[j] = f((begin + j - pos) * step);
            sum += row[j];
        }
        w.start[i] = (int)(begin - 0.5);
        for (int j = 0; j < taps; j++)
            w.wt[(size_t)i * taps + j] = (float)(row[j] / sum);
    }
    return w;
}

/* Chroma plane shifts in zimg convention (source-plane units): where an
 * output sample's center lands in the source grid, beyond the plain
 * (i + 0.5) / scale mapping. Derived from the siting offsets. */
inline void chroma_shifts(int siting, bool up, double *sx, double *sy)
{
    if (up) {  // 420 chroma -> luma grid
        *sx = siting == AJI_SITING_CENTER ? 0.0 : 0.25;
        *sy = siting == AJI_SITING_TOPLEFT ? 0.25 : 0.0;
    } else {   // luma grid -> 420 chroma
        *sx = siting == AJI_SITING_CENTER ? 0.0 : -0.5;
        *sy = siting == AJI_SITING_TOPLEFT ? -0.5 : 0.0;
    }
}

inline aji_csp make_csp(int format, int matrix, int range)
{
    aji_csp c = {};
    switch (matrix) {
    case AJI_MATRIX_BT601:  c.kr = 0.299f;  c.kb = 0.114f;  break;
    case AJI_MATRIX_BT2020: c.kr = 0.2627f; c.kb = 0.0593f; break;
    case AJI_MATRIX_BT709:
    default:                c.kr = 0.2126f; c.kb = 0.0722f; break;
    }
    const bool p010 = format == AJI_FMT_P010;
    const float m = p010 ? 256.0f : 1.0f;          // 8-bit-reference scale
    const float maxraw = p010 ? 65472.0f : 255.0f; // (2^bd - 1) << (16 - bd)
    if (range == AJI_RANGE_FULL) {
        c.yoff = 0.0f;
        c.yscale = maxraw;
        c.coff = p010 ? 32768.0f : 128.0f;
        c.cscale = maxraw;
    } else {
        c.yoff = 16.0f * m;
        c.yscale = 219.0f * m;
        c.coff = 128.0f * m;
        c.cscale = 224.0f * m;
    }
    return c;
}

} // namespace aji_resample
