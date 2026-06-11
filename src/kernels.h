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

aji_plan *aji_pre_plan_create(int format, int w, int h, int siting);
aji_plan *aji_post_plan_create(int format, int w, int h, int siting);
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

#ifdef __cplusplus
}
#endif

#endif
