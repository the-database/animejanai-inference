/*
 * aji_kernel_test — drive individual aji kernels on raw files, for parity
 * testing against VapourSynth/zimg goldens. No TensorRT involved.
 *
 * Usage:
 *   aji_kernel_test pre    --input yuv.raw --output rgb.f16 --width W --height H
 *                          [--format nv12|p010] [--matrix 601|709|2020]
 *                          [--range limited|full] [--siting left|center|topleft]
 *   aji_kernel_test post   --input rgb.f16 --output yuv.raw --width W --height H
 *                          (W,H are the RGB dims; same colorimetry flags)
 *   aji_kernel_test resize --input rgb.f16 --output rgb.f16
 *                          --width SW --height SH --out-width DW --out-height DH
 *
 * Frames are processed until EOF. YUV raw layout: Y plane then interleaved
 * UV (NV12 u8 / P010 u16, packed strides). RGB layout: fp16 NCHW planar.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "aji.h"
#include "kernels.h"

#define CK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, \
                cudaGetErrorString(e_)); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: aji_kernel_test <pre|post|resize> ... (see header)\n");
        return 2;
    }
    const char *mode = argv[1];
    const char *input = NULL, *output = NULL;
    const char *format = "nv12", *matrix = "709", *range = "limited", *siting = "left";
    int w = 0, h = 0, dw = 0, dh = 0, bench = 0;

    for (int i = 2; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--input")) input = argv[++i];
        else if (!strcmp(argv[i], "--output")) output = argv[++i];
        else if (!strcmp(argv[i], "--format")) format = argv[++i];
        else if (!strcmp(argv[i], "--matrix")) matrix = argv[++i];
        else if (!strcmp(argv[i], "--range")) range = argv[++i];
        else if (!strcmp(argv[i], "--siting")) siting = argv[++i];
        else if (!strcmp(argv[i], "--width")) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height")) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out-width")) dw = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out-height")) dh = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) bench = atoi(argv[++i]);
    }
    if (!input || !output || w < 2 || h < 2) {
        fprintf(stderr, "missing required args\n");
        return 2;
    }

    const int fmt = strcmp(format, "p010") ? AJI_FMT_NV12 : AJI_FMT_P010;
    const int bpp = fmt == AJI_FMT_P010 ? 2 : 1;
    const int mat = !strcmp(matrix, "601") ? AJI_MATRIX_BT601 :
                    !strcmp(matrix, "2020") ? AJI_MATRIX_BT2020 : AJI_MATRIX_BT709;
    const int rng = !strcmp(range, "full") ? AJI_RANGE_FULL : AJI_RANGE_LIMITED;
    const int sit = !strcmp(siting, "center") ? AJI_SITING_CENTER :
                    !strcmp(siting, "topleft") ? AJI_SITING_TOPLEFT : AJI_SITING_LEFT;
    const aji_csp csp = aji_make_csp(fmt, mat, rng);

    const size_t y_sz = (size_t)w * h * bpp, uv_sz = y_sz / 2;
    const size_t yuv_sz = y_sz + uv_sz;
    const size_t rgb_sz = (size_t)w * h * 3 * 2;

    size_t in_sz, out_sz;
    if (!strcmp(mode, "pre")) {
        in_sz = yuv_sz; out_sz = rgb_sz;
    } else if (!strcmp(mode, "post")) {
        in_sz = rgb_sz; out_sz = yuv_sz;
    } else if (!strcmp(mode, "resize")) {
        if (dw < 2 || dh < 2) {
            fprintf(stderr, "resize needs --out-width/--out-height\n");
            return 2;
        }
        in_sz = rgb_sz; out_sz = (size_t)dw * dh * 3 * 2;
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    FILE *fin = fopen(input, "rb");
    if (!fin) { perror(input); return 1; }
    FILE *fout = fopen(output, "wb");
    if (!fout) { perror(output); return 1; }

    void *h_in = malloc(in_sz), *h_out = malloc(out_sz);
    void *d_in, *d_out;
    CK(cudaMalloc(&d_in, in_sz));
    CK(cudaMalloc(&d_out, out_sz));

    aji_plan *plan;
    if (!strcmp(mode, "pre"))
        plan = aji_pre_plan_create(fmt, w, h, sit, AJI_FILTER_SPLINE36);
    else if (!strcmp(mode, "post"))
        plan = aji_post_plan_create(fmt, w, h, sit, AJI_FILTER_SPLINE36);
    else
        plan = aji_resize_plan_create(w, h, dw, dh);
    if (!plan) {
        fprintf(stderr, "plan creation failed\n");
        return 1;
    }

    if (bench > 0) {
        // device-time microbenchmark on the first frame, no host I/O
        if (fread(h_in, 1, in_sz, fin) != in_sz) {
            fprintf(stderr, "bench: no full frame in input\n");
            return 1;
        }
        CK(cudaMemcpy(d_in, h_in, in_sz, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int i = 0; i < 10; i++) {  // warmup
            if (!strcmp(mode, "pre"))
                aji_run_pre(plan, d_in, (ptrdiff_t)w * bpp,
                            (char *)d_in + y_sz, (ptrdiff_t)w * bpp,
                            &csp, d_out, NULL);
            else if (!strcmp(mode, "post"))
                aji_run_post(plan, d_in, &csp, d_out, (ptrdiff_t)w * bpp,
                             (char *)d_out + y_sz, (ptrdiff_t)w * bpp, NULL);
            else
                aji_run_resize(plan, d_in, d_out, NULL);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0, 0));
        for (int i = 0; i < bench; i++) {
            if (!strcmp(mode, "pre"))
                aji_run_pre(plan, d_in, (ptrdiff_t)w * bpp,
                            (char *)d_in + y_sz, (ptrdiff_t)w * bpp,
                            &csp, d_out, NULL);
            else if (!strcmp(mode, "post"))
                aji_run_post(plan, d_in, &csp, d_out, (ptrdiff_t)w * bpp,
                             (char *)d_out + y_sz, (ptrdiff_t)w * bpp, NULL);
            else
                aji_run_resize(plan, d_in, d_out, NULL);
        }
        CK(cudaEventRecord(e1, 0));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("%s %dx%d: %.1f us/frame (%d iters)\n", mode, w, h,
               1000.0 * ms / bench, bench);
        return 0;
    }

    int n = 0;
    while (fread(h_in, 1, in_sz, fin) == in_sz) {
        CK(cudaMemcpy(d_in, h_in, in_sz, cudaMemcpyHostToDevice));

        int err;
        if (!strcmp(mode, "pre")) {
            err = aji_run_pre(plan, d_in, (ptrdiff_t)w * bpp,
                              (char *)d_in + y_sz, (ptrdiff_t)w * bpp,
                              &csp, d_out, NULL);
        } else if (!strcmp(mode, "post")) {
            err = aji_run_post(plan, d_in, &csp,
                               d_out, (ptrdiff_t)w * bpp,
                               (char *)d_out + y_sz, (ptrdiff_t)w * bpp,
                               NULL);
        } else {
            err = aji_run_resize(plan, d_in, d_out, NULL);
        }
        if (err) {
            fprintf(stderr, "kernel launch failed: %d\n", err);
            return 1;
        }
        CK(cudaDeviceSynchronize());

        CK(cudaMemcpy(h_out, d_out, out_sz, cudaMemcpyDeviceToHost));
        if (fwrite(h_out, 1, out_sz, fout) != out_sz) {
            perror("fwrite");
            return 1;
        }
        n++;
    }

    fprintf(stderr, "%s: %d frame(s)\n", mode, n);
    aji_plan_destroy(plan);
    fclose(fout);
    fclose(fin);
    return 0;
}
