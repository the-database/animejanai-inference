/*
 * Pre/post-processing and resize CUDA kernels for the aji shim.
 *
 * pre:    NV12/P010 -> fp16 NCHW RGB, Spline36 chroma upsample + matrix
 * post:   fp16 NCHW RGB -> NV12/P010, matrix + Spline36 chroma downsample
 * resize: Spline36 on fp16 NCHW RGB (chain resize steps)
 *
 * Resampling replicates zimg's compute_filter (what VapourSynth resize —
 * the reference pipeline — uses): per-output-pixel weight tables built
 * host-side in double precision, kernel stretched by 1/scale on downscale,
 * taps = max(ceil(2*support/step), 1), positions mirrored then clamped at
 * the borders, weights normalized to 1. Intermediates are fp32; values are
 * only clamped at final integer quantization (P010 at true 10-bit depth).
 */

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include <vector>

#include "aji.h"
#include "kernels.h"

extern "C" aji_csp aji_make_csp(int format, int matrix, int range)
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

/* ---------------- weight tables (zimg compute_filter) ---------------- */

namespace {

struct pass {
    int taps = 0, src = 0, dst = 0;
    const int *start = nullptr;   // [dst] first source index per output
    const float *wt = nullptr;    // [dst * taps], normalized
};

double spline36(double x)
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

double bilinear(double x)
{
    x = fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

bool build_pass(pass *p, int src, int dst, double shift, int filter)
{
    const double scale = (double)dst / src;
    const double step = scale < 1.0 ? scale : 1.0;
    double (*f)(double) = filter == AJI_FILTER_BILINEAR ? bilinear : spline36;
    const double support = filter == AJI_FILTER_BILINEAR ? 1.0 : 3.0;
    int taps = (int)ceil(support / step) * 2;
    if (taps < 1)
        taps = 1;

    std::vector<int> start(dst);
    std::vector<float> wt((size_t)dst * taps);
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
        start[i] = (int)(begin - 0.5);
        for (int j = 0; j < taps; j++)
            wt[(size_t)i * taps + j] = (float)(row[j] / sum);
    }

    int *d_start = nullptr;
    float *d_wt = nullptr;
    if (cudaMalloc(&d_start, dst * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_wt, wt.size() * sizeof(float)) != cudaSuccess) {
        cudaFree(d_start);
        return false;
    }
    cudaMemcpy(d_start, start.data(), dst * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, wt.data(), wt.size() * sizeof(float), cudaMemcpyHostToDevice);
    p->taps = taps;
    p->src = src;
    p->dst = dst;
    p->start = d_start;
    p->wt = d_wt;
    return true;
}

void free_pass(pass *p)
{
    cudaFree((void *)p->start);
    cudaFree((void *)p->wt);
    *p = {};
}

/* Chroma plane shifts in zimg convention (source-plane units): where an
 * output sample's center lands in the source grid, beyond the plain
 * (i + 0.5) / scale mapping. Derived from the siting offsets. */
void chroma_shifts(int siting, bool up, double *sx, double *sy)
{
    if (up) {  // 420 chroma -> luma grid
        *sx = siting == AJI_SITING_CENTER ? 0.0 : 0.25;
        *sy = siting == AJI_SITING_TOPLEFT ? 0.25 : 0.0;
    } else {   // luma grid -> 420 chroma
        *sx = siting == AJI_SITING_CENTER ? 0.0 : -0.5;
        *sy = siting == AJI_SITING_TOPLEFT ? -0.5 : 0.0;
    }
}

} // namespace

struct aji_plan {
    enum Kind { PRE, POST, RESIZE } kind;
    int format = 0;
    int w = 0, h = 0;          // pre/post luma dims; resize: src dims
    int dw = 0, dh = 0;        // resize: dst dims
    pass ph, pv;
    float *tmp0 = nullptr;     // after the first pass
    float *tmp1 = nullptr;     // after the second pass (pre only)
};

/* ---------------- device helpers ---------------- */

__device__ __forceinline__ int mirr(int i, int n)
{
    i = i < 0 ? -i - 1 : i;
    i = i >= n ? 2 * n - 1 - i : i;
    return min(max(i, 0), n - 1);
}

__device__ __forceinline__ float quant(float v, float qdiv, float qmax)
{
    return fminf(fmaxf(rintf(v / qdiv), 0.0f), qmax) * qdiv;
}

/* ---------------- pre: NV12/P010 -> RGB fp16 ---------------- */

// horizontal resample of interleaved raw chroma -> 2 planar fp32 (raw units)
template <typename T>
__global__ void k_uv_h(const uint8_t *uv, ptrdiff_t stride, int ch, pass px,
                       float *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= px.dst || y >= ch)
        return;
    const T *row = (const T *)(uv + (size_t)y * stride);
    const float *w = px.wt + (size_t)x * px.taps;
    const int s0 = px.start[x];
    float u = 0.0f, v = 0.0f;
    for (int j = 0; j < px.taps; j++) {
        const int sx = mirr(s0 + j, px.src);
        u += w[j] * row[2 * sx];
        v += w[j] * row[2 * sx + 1];
    }
    const size_t plane = (size_t)px.dst * ch, idx = (size_t)y * px.dst + x;
    dst[idx] = u;
    dst[plane + idx] = v;
}

// vertical resample of N planar fp32 planes (gridDim.z = N)
__global__ void k_v_f32(const float *src, int w, pass py, float *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= py.dst)
        return;
    const float *sp = src + (size_t)blockIdx.z * w * py.src;
    const float *wt = py.wt + (size_t)y * py.taps;
    const int s0 = py.start[y];
    float a = 0.0f;
    for (int j = 0; j < py.taps; j++)
        a += wt[j] * sp[(size_t)mirr(s0 + j, py.src) * w + x];
    dst[(size_t)blockIdx.z * w * py.dst + (size_t)y * w + x] = a;
}

template <typename T>
__global__ void k_pre_combine(const uint8_t *y_plane, ptrdiff_t y_stride,
                              const float *uvf, int w, int h, aji_csp csp,
                              __half *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const size_t plane = (size_t)w * h, idx = (size_t)y * w + x;

    const T *yrow = (const T *)(y_plane + (size_t)y * y_stride);
    const float Y = ((float)yrow[x] - csp.yoff) / csp.yscale;
    const float U = (uvf[idx] - csp.coff) / csp.cscale;
    const float V = (uvf[plane + idx] - csp.coff) / csp.cscale;

    const float kg = 1.0f - csp.kr - csp.kb;
    const float r = Y + 2.0f * (1.0f - csp.kr) * V;
    const float b = Y + 2.0f * (1.0f - csp.kb) * U;
    const float g = Y - (2.0f * csp.kb * (1.0f - csp.kb) * U +
                         2.0f * csp.kr * (1.0f - csp.kr) * V) / kg;

    dst[idx]             = __float2half(r);
    dst[plane + idx]     = __float2half(g);
    dst[2 * plane + idx] = __float2half(b);
}

/* ---------------- post: RGB fp16 -> NV12/P010 ---------------- */

// matrix + luma quantize; chroma (normalized units) to 2 planar fp32
template <typename T>
__global__ void k_post_matrix(const __half *src, int w, int h, aji_csp csp,
                              float qdiv, float qmax,
                              uint8_t *y_plane, ptrdiff_t y_stride, float *uvf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const size_t plane = (size_t)w * h, idx = (size_t)y * w + x;

    const float r = __half2float(src[idx]);
    const float g = __half2float(src[plane + idx]);
    const float b = __half2float(src[2 * plane + idx]);

    const float Y = csp.kr * r + (1.0f - csp.kr - csp.kb) * g + csp.kb * b;
    T *yrow = (T *)(y_plane + (size_t)y * y_stride);
    yrow[x] = (T)quant(Y * csp.yscale + csp.yoff, qdiv, qmax);

    uvf[idx]         = (b - Y) / (2.0f * (1.0f - csp.kb));
    uvf[plane + idx] = (r - Y) / (2.0f * (1.0f - csp.kr));
}

// horizontal resample of N planar fp32 planes (gridDim.z = N)
__global__ void k_h_f32(const float *src, int h, pass px, float *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= px.dst || y >= h)
        return;
    const float *sp = src + (size_t)blockIdx.z * px.src * h +
                      (size_t)y * px.src;
    const float *wt = px.wt + (size_t)x * px.taps;
    const int s0 = px.start[x];
    float a = 0.0f;
    for (int j = 0; j < px.taps; j++)
        a += wt[j] * sp[mirr(s0 + j, px.src)];
    dst[(size_t)blockIdx.z * px.dst * h + (size_t)y * px.dst + x] = a;
}

// vertical chroma resample + quantize + interleave
template <typename T>
__global__ void k_uv_v_store(const float *src, int cw, pass py, aji_csp csp,
                             float qdiv, float qmax,
                             uint8_t *uv, ptrdiff_t uv_stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= py.dst)
        return;
    const size_t plane = (size_t)cw * py.src;
    const float *wt = py.wt + (size_t)y * py.taps;
    const int s0 = py.start[y];
    float u = 0.0f, v = 0.0f;
    for (int j = 0; j < py.taps; j++) {
        const size_t row = (size_t)mirr(s0 + j, py.src) * cw + x;
        u += wt[j] * src[row];
        v += wt[j] * src[plane + row];
    }
    T *uvrow = (T *)(uv + (size_t)y * uv_stride);
    uvrow[2 * x]     = (T)quant(u * csp.cscale + csp.coff, qdiv, qmax);
    uvrow[2 * x + 1] = (T)quant(v * csp.cscale + csp.coff, qdiv, qmax);
}

/* ---------------- resize: RGB fp16 -> RGB fp16 ---------------- */

__global__ void k_rs_h(const __half *src, int h, pass px, float *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= px.dst || y >= h)
        return;
    const __half *sp = src + (size_t)blockIdx.z * px.src * h +
                       (size_t)y * px.src;
    const float *wt = px.wt + (size_t)x * px.taps;
    const int s0 = px.start[x];
    float a = 0.0f;
    for (int j = 0; j < px.taps; j++)
        a += wt[j] * __half2float(sp[mirr(s0 + j, px.src)]);
    dst[(size_t)blockIdx.z * px.dst * h + (size_t)y * px.dst + x] = a;
}

__global__ void k_rs_v(const float *src, int w, pass py, __half *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= py.dst)
        return;
    const float *sp = src + (size_t)blockIdx.z * w * py.src;
    const float *wt = py.wt + (size_t)y * py.taps;
    const int s0 = py.start[y];
    float a = 0.0f;
    for (int j = 0; j < py.taps; j++)
        a += wt[j] * sp[(size_t)mirr(s0 + j, py.src) * w + x];
    dst[(size_t)blockIdx.z * w * py.dst + (size_t)y * w + x] =
        __float2half(a);
}

/* ---------------- plans ---------------- */

static void plan_destroy_inner(aji_plan *p)
{
    free_pass(&p->ph);
    free_pass(&p->pv);
    cudaFree(p->tmp0);
    cudaFree(p->tmp1);
    p->tmp0 = p->tmp1 = nullptr;
}

extern "C" void aji_plan_destroy(aji_plan *p)
{
    if (!p)
        return;
    plan_destroy_inner(p);
    delete p;
}

extern "C" aji_plan *aji_pre_plan_create(int format, int w, int h, int siting,
                                         int filter)
{
    aji_plan *p = new aji_plan();
    p->kind = aji_plan::PRE;
    p->format = format;
    p->w = w;
    p->h = h;
    const int cw = w >> 1, ch = h >> 1;
    double sx, sy;
    chroma_shifts(siting, true, &sx, &sy);
    if (!build_pass(&p->ph, cw, w, sx, filter) ||
        !build_pass(&p->pv, ch, h, sy, filter) ||
        cudaMalloc(&p->tmp0, (size_t)2 * w * ch * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&p->tmp1, (size_t)2 * w * h * sizeof(float)) != cudaSuccess) {
        aji_plan_destroy(p);
        return nullptr;
    }
    return p;
}

extern "C" aji_plan *aji_post_plan_create(int format, int w, int h, int siting,
                                          int filter)
{
    aji_plan *p = new aji_plan();
    p->kind = aji_plan::POST;
    p->format = format;
    p->w = w;
    p->h = h;
    const int cw = w >> 1, ch = h >> 1;
    double sx, sy;
    chroma_shifts(siting, false, &sx, &sy);
    if (!build_pass(&p->ph, w, cw, sx, filter) ||
        !build_pass(&p->pv, h, ch, sy, filter) ||
        cudaMalloc(&p->tmp0, (size_t)2 * w * h * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&p->tmp1, (size_t)2 * cw * h * sizeof(float)) != cudaSuccess) {
        aji_plan_destroy(p);
        return nullptr;
    }
    return p;
}

extern "C" aji_plan *aji_resize_plan_create(int sw, int sh, int dw, int dh)
{
    aji_plan *p = new aji_plan();
    p->kind = aji_plan::RESIZE;
    p->w = sw;
    p->h = sh;
    p->dw = dw;
    p->dh = dh;
    if (!build_pass(&p->ph, sw, dw, 0.0, AJI_FILTER_SPLINE36) ||
        !build_pass(&p->pv, sh, dh, 0.0, AJI_FILTER_SPLINE36) ||
        cudaMalloc(&p->tmp0, (size_t)3 * dw * sh * sizeof(float)) != cudaSuccess) {
        aji_plan_destroy(p);
        return nullptr;
    }
    return p;
}

/* ---------------- runs ---------------- */

#define BLOCK_X 32
#define BLOCK_Y 8
#define GRID(w, h, d) dim3(((w) + BLOCK_X - 1) / BLOCK_X, \
                           ((h) + BLOCK_Y - 1) / BLOCK_Y, (d)), \
                      dim3(BLOCK_X, BLOCK_Y)

extern "C" int aji_run_pre(aji_plan *p,
                           const void *y_plane, ptrdiff_t y_stride,
                           const void *uv_plane, ptrdiff_t uv_stride,
                           const aji_csp *csp, void *dst_f16, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    const int w = p->w, h = p->h, ch = h >> 1;
    if (p->format == AJI_FMT_NV12) {
        k_uv_h<uint8_t><<<GRID(w, ch, 1), 0, s>>>(
            (const uint8_t *)uv_plane, uv_stride, ch, p->ph, p->tmp0);
    } else {
        k_uv_h<uint16_t><<<GRID(w, ch, 1), 0, s>>>(
            (const uint8_t *)uv_plane, uv_stride, ch, p->ph, p->tmp0);
    }
    k_v_f32<<<GRID(w, h, 2), 0, s>>>(p->tmp0, w, p->pv, p->tmp1);
    if (p->format == AJI_FMT_NV12) {
        k_pre_combine<uint8_t><<<GRID(w, h, 1), 0, s>>>(
            (const uint8_t *)y_plane, y_stride, p->tmp1, w, h, *csp,
            (__half *)dst_f16);
    } else {
        k_pre_combine<uint16_t><<<GRID(w, h, 1), 0, s>>>(
            (const uint8_t *)y_plane, y_stride, p->tmp1, w, h, *csp,
            (__half *)dst_f16);
    }
    return (int)cudaGetLastError();
}

extern "C" int aji_run_post(aji_plan *p, const void *src_f16,
                            const aji_csp *csp,
                            void *y_plane, ptrdiff_t y_stride,
                            void *uv_plane, ptrdiff_t uv_stride, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    const int w = p->w, h = p->h, cw = w >> 1, ch = h >> 1;
    const bool p010 = p->format == AJI_FMT_P010;
    const float qdiv = p010 ? 64.0f : 1.0f;
    const float qmax = p010 ? 1023.0f : 255.0f;
    if (p010) {
        k_post_matrix<uint16_t><<<GRID(w, h, 1), 0, s>>>(
            (const __half *)src_f16, w, h, *csp, qdiv, qmax,
            (uint8_t *)y_plane, y_stride, p->tmp0);
    } else {
        k_post_matrix<uint8_t><<<GRID(w, h, 1), 0, s>>>(
            (const __half *)src_f16, w, h, *csp, qdiv, qmax,
            (uint8_t *)y_plane, y_stride, p->tmp0);
    }
    k_h_f32<<<GRID(cw, h, 2), 0, s>>>(p->tmp0, h, p->ph, p->tmp1);
    if (p010) {
        k_uv_v_store<uint16_t><<<GRID(cw, ch, 1), 0, s>>>(
            p->tmp1, cw, p->pv, *csp, qdiv, qmax,
            (uint8_t *)uv_plane, uv_stride);
    } else {
        k_uv_v_store<uint8_t><<<GRID(cw, ch, 1), 0, s>>>(
            p->tmp1, cw, p->pv, *csp, qdiv, qmax,
            (uint8_t *)uv_plane, uv_stride);
    }
    return (int)cudaGetLastError();
}

extern "C" int aji_run_resize(aji_plan *p, const void *src_f16, void *dst_f16,
                              void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    k_rs_h<<<GRID(p->dw, p->h, 3), 0, s>>>(
        (const __half *)src_f16, p->h, p->ph, p->tmp0);
    k_rs_v<<<GRID(p->dw, p->dh, 3), 0, s>>>(
        p->tmp0, p->dw, p->pv, (__half *)dst_f16);
    return (int)cudaGetLastError();
}

/* ---------------- RIFE helpers ---------------- */

// The four constant planes of the RIFE v1 input layout (vsmlrt
// get_rife_input): mesh X = 2x/(w-1)-1, mesh Y = 2y/(h-1)-1, then the
// constant multipliers 2/(w-1) and 2/(h-1).
__global__ void k_rife_consts(__half *meshx, __half *meshy, __half *mulw,
                              __half *mulh, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const size_t idx = (size_t)y * w + x;
    meshx[idx] = __float2half(2.0f * x / (w - 1) - 1.0f);
    meshy[idx] = __float2half(2.0f * y / (h - 1) - 1.0f);
    mulw[idx] = __float2half(2.0f / (w - 1));
    mulh[idx] = __float2half(2.0f / (h - 1));
}

extern "C" int aji_rife_fill_consts(void *meshx_f16, void *meshy_f16,
                                    void *mulw_f16, void *mulh_f16,
                                    int w, int h, void *stream)
{
    k_rife_consts<<<GRID(w, h, 1), 0, (cudaStream_t)stream>>>(
        (__half *)meshx_f16, (__half *)meshy_f16, (__half *)mulw_f16,
        (__half *)mulh_f16, w, h);
    return (int)cudaGetLastError();
}

__global__ void k_fill_f16(__half *dst, size_t count, float value)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = __float2half(value);
}

extern "C" int aji_fill_f16(void *dst_f16, size_t count, float value,
                            void *stream)
{
    const int block = 256;
    const size_t grid = (count + block - 1) / block;
    k_fill_f16<<<(unsigned)grid, block, 0, (cudaStream_t)stream>>>(
        (__half *)dst_f16, count, value);
    return (int)cudaGetLastError();
}

template <typename T>
__global__ void k_fill_plane(uint8_t *plane, ptrdiff_t stride, int w, int h,
                             unsigned value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    ((T *)(plane + (size_t)y * stride))[x] = (T)value;
}

extern "C" int aji_fill_plane(int format, void *plane, ptrdiff_t stride,
                              int w, int h, unsigned value, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    if (format == AJI_FMT_NV12) {
        k_fill_plane<uint8_t><<<GRID(w, h, 1), 0, s>>>(
            (uint8_t *)plane, stride, w, h, value);
    } else {
        k_fill_plane<uint16_t><<<GRID(w, h, 1), 0, s>>>(
            (uint8_t *)plane, stride, w, h, value);
    }
    return (int)cudaGetLastError();
}

// Block-reduced sum of |Ya - Yb| / peak; one atomicAdd per block.
template <typename T>
__global__ void k_scd_diff(const uint8_t *ya, ptrdiff_t sa,
                           const uint8_t *yb, ptrdiff_t sb, int w, int h,
                           float norm, float *accum)
{
    __shared__ float partial[BLOCK_X * BLOCK_Y];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float d = 0.0f;
    if (x < w && y < h) {
        const float a = ((const T *)(ya + (size_t)y * sa))[x];
        const float b = ((const T *)(yb + (size_t)y * sb))[x];
        d = fabsf(a - b) * norm;
    }
    partial[tid] = d;
    __syncthreads();
    for (int n = BLOCK_X * BLOCK_Y / 2; n > 0; n >>= 1) {
        if (tid < n)
            partial[tid] += partial[tid + n];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(accum, partial[0]);
}

extern "C" int aji_scd_diff(int format, const void *ya, ptrdiff_t stride_a,
                            const void *yb, ptrdiff_t stride_b, int w, int h,
                            float *accum_dev, void *stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    if (format == AJI_FMT_NV12) {
        k_scd_diff<uint8_t><<<GRID(w, h, 1), 0, s>>>(
            (const uint8_t *)ya, stride_a, (const uint8_t *)yb, stride_b,
            w, h, 1.0f / 255.0f, accum_dev);
    } else {
        // P010 raw values are MSB-aligned 10-bit
        k_scd_diff<uint16_t><<<GRID(w, h, 1), 0, s>>>(
            (const uint8_t *)ya, stride_a, (const uint8_t *)yb, stride_b,
            w, h, 1.0f / 65472.0f, accum_dev);
    }
    return (int)cudaGetLastError();
}
