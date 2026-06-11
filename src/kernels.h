#ifndef AJI_KERNELS_H
#define AJI_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Colorspace constants, computed host-side from (format, matrix, range).
 * Normalization works in raw container units (P010 values are MSB-aligned
 * u16, so its scales include the <<6). */
typedef struct aji_csp {
    float kr, kb;          /* matrix coefficients */
    float yoff, yscale;    /* y = (raw - yoff) / yscale */
    float coff, cscale;    /* c = (raw - coff) / cscale */
} aji_csp;

aji_csp aji_make_csp(int format, int matrix, int range);

/* Precomputed separable Spline36 resampling plans, matching VapourSynth/
 * zimg resize semantics (the reference pipeline): kernel stretched by
 * 1/scale when downscaling, mirror boundary handling, fp32 intermediates
 * between passes, nothing clamped until final integer quantization
 * (out-of-gamut RGB flows through to the model, as zimg lets it).
 * Geometry and chroma siting are baked into the plan; matrix/range
 * constants are applied per run. P010 output is quantized at true 10-bit
 * depth, then shifted (low 6 bits zero).
 *
 * format: enum aji_format; siting: enum aji_siting (aji.h values).
 * All creates cudaMalloc against the current CUDA context; run/destroy
 * must be called with the same context current. Runs return 0 or a
 * cudaError_t. */
typedef struct aji_plan aji_plan;

/* Resampling kernel selection. The upscale chain uses Spline36 (what the
 * reference pipeline's resize.Spline36 calls do); the RIFE wrapper uses
 * Bilinear (resize.Bilinear in rife_cuda.py). */
enum {
    AJI_FILTER_SPLINE36 = 0,
    AJI_FILTER_BILINEAR = 1,
};

aji_plan *aji_pre_plan_create(int format, int w, int h, int siting, int filter);
aji_plan *aji_post_plan_create(int format, int w, int h, int siting, int filter);
aji_plan *aji_resize_plan_create(int sw, int sh, int dw, int dh);
void aji_plan_destroy(aji_plan *p);

/* NV12/P010 -> fp16 NCHW RGB at the plan's WxH. */
int aji_run_pre(aji_plan *p,
                const void *y_plane, ptrdiff_t y_stride,
                const void *uv_plane, ptrdiff_t uv_stride,
                const aji_csp *csp, void *dst_f16, void *stream);

/* fp16 NCHW RGB at the plan's WxH -> NV12/P010. */
int aji_run_post(aji_plan *p, const void *src_f16, const aji_csp *csp,
                 void *y_plane, ptrdiff_t y_stride,
                 void *uv_plane, ptrdiff_t uv_stride, void *stream);

/* Spline36 resize of fp16 NCHW RGB (3 planes), plan's SWxSH -> DWxDH. */
int aji_run_resize(aji_plan *p, const void *src_f16, void *dst_f16,
                   void *stream);

/* RIFE helpers. */

/* Fill the four constant fp16 planes of the RIFE v1 input layout
 * (channels 7..10: mesh X, mesh Y, 2/(w-1), 2/(h-1)), each w*h. */
int aji_rife_fill_consts(void *meshx_f16, void *meshy_f16, void *mulw_f16,
                         void *mulh_f16, int w, int h, void *stream);

/* Fill count fp16 elements with value (the per-call timestep plane). */
int aji_fill_f16(void *dst_f16, size_t count, float value, void *stream);

/* Fill a (strided) u8/u16 plane with a constant (pad border init). */
int aji_fill_plane(int format, void *plane, ptrdiff_t stride, int w, int h,
                   unsigned value, void *stream);

/* Mean absolute luma difference, normalized to [0,1] (misc.SCDetect's
 * metric). Accumulates into *accum_dev (one float, caller zeroes it);
 * divide by w*h after sync to get the mean. */
int aji_scd_diff(int format, const void *ya, ptrdiff_t stride_a,
                 const void *yb, ptrdiff_t stride_b, int w, int h,
                 float *accum_dev, void *stream);

#ifdef __cplusplus
}
#endif

#endif
