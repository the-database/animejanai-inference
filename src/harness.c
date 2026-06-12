/*
 * aji_harness — CLI test harness for the aji shim. No player involved.
 *
 * Direct mode (--engine) or conf mode (--conf/--model-dir/--trtexec/--slot).
 * Reads raw NV12/P010 frames, runs aji_infer on the GPU, optionally writes
 * the upscaled raw frames, reports per-frame device timing.
 *
 * Usage:
 *   aji_harness --input in.raw --width W --height H [--format nv12|p010]
 *               [--matrix 601|709|2020] [--range limited|full]
 *               [--siting left|center|topleft] [--fps F] [--frames N]
 *               [--output out.raw] [--dump-rgb pre.f16]
 *               ( --engine E.engine | --conf animejanai.conf
 *                 --model-dir DIR --trtexec PATH [--trtexec-env K=V]
 *                 [--slot N] )
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "aji.h"

#define CK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, \
                cudaGetErrorString(e_)); \
        return 1; \
    } \
} while (0)

static void log_cb(void *opaque, int level, const char *msg)
{
    (void)opaque;
    if (level <= 2)
        fprintf(stderr, "[trt:%d] %s\n", level, msg);
}

int main(int argc, char **argv)
{
    const char *engine = NULL, *input = NULL, *output = NULL;
    const char *conf = NULL, *model_dir = NULL, *trtexec = NULL, *trtexec_env = NULL;
    const char *rife_model_dir = NULL;
    const char *format = "nv12", *matrix = "709", *range = "limited", *siting = "left";
    int w = 0, h = 0, max_frames = 1 << 30, slot = 1, rife = 0;

    double fps = 23.976;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rife")) { rife = 1; continue; }
        if (i >= argc - 1) break;
        if (!strcmp(argv[i], "--engine")) engine = argv[++i];
        else if (!strcmp(argv[i], "--conf")) conf = argv[++i];
        else if (!strcmp(argv[i], "--model-dir")) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--rife-model-dir")) rife_model_dir = argv[++i];
        else if (!strcmp(argv[i], "--trtexec")) trtexec = argv[++i];
        else if (!strcmp(argv[i], "--trtexec-env")) trtexec_env = argv[++i];
        else if (!strcmp(argv[i], "--slot")) slot = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--input")) input = argv[++i];
        else if (!strcmp(argv[i], "--output")) output = argv[++i];
        else if (!strcmp(argv[i], "--format")) format = argv[++i];
        else if (!strcmp(argv[i], "--matrix")) matrix = argv[++i];
        else if (!strcmp(argv[i], "--range")) range = argv[++i];
        else if (!strcmp(argv[i], "--siting")) siting = argv[++i];
        else if (!strcmp(argv[i], "--fps")) fps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--width")) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height")) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames")) max_frames = atoi(argv[++i]);
    }
    if ((!engine && !conf) || !input || w < 2 || h < 2) {
        fprintf(stderr, "missing required args (see header comment)\n");
        return 2;
    }

    const int fmt = strcmp(format, "p010") ? AJI_FMT_NV12 : AJI_FMT_P010;
    const int bpp = fmt == AJI_FMT_P010 ? 2 : 1;
    const size_t y_sz = (size_t)w * h * bpp;
    const size_t uv_sz = y_sz / 2;
    const size_t frame_sz = y_sz + uv_sz;

    aji_create_params params = {
        .api_version = AJI_API_VERSION,
        .cuda_context = NULL,
        .conf_path = conf,
        .model_dir = model_dir,
        .trtexec = trtexec,
        .trtexec_env = trtexec_env,
        .slot = slot,
        .rife_model_dir = rife_model_dir,
        .engine_path = engine,
        .max_width = w,
        .max_height = h,
        .log = log_cb,
    };
    aji_ctx *aji = aji_create(&params);
    if (!aji) {
        fprintf(stderr, "aji_create failed\n");
        return 1;
    }

    int ow = 0, oh = 0;
    int act = aji_configure(aji, w, h, fps, &ow, &oh);
    if (act < 0) {
        fprintf(stderr, "aji_configure: %d: %s\n", act, aji_last_error(aji));
        return 1;
    }
    printf("%s\n", aji_current_log(aji));

    if (rife) {
        // Interpolation-only test loop, replicating the reference's 2x
        // video_player ordering: f0, i01, f1, i12, f2 ... with the left
        // source frame substituted on scene changes.
        int rn = 0, rd = 0;
        if (!aji_rife_factor(aji, &rn, &rd) || rn != 2 * rd) {
            fprintf(stderr, "rife test needs an active 2/1 rife chain\n");
            return 1;
        }
        const int fmt2 = strcmp(format, "p010") ? AJI_FMT_NV12 : AJI_FMT_P010;
        const int mat2 = !strcmp(matrix, "601") ? AJI_MATRIX_BT601 :
                         !strcmp(matrix, "2020") ? AJI_MATRIX_BT2020
                                                 : AJI_MATRIX_BT709;
        const int rng2 = !strcmp(range, "full") ? AJI_RANGE_FULL
                                                : AJI_RANGE_LIMITED;
        const int sit2 = !strcmp(siting, "center") ? AJI_SITING_CENTER :
                         !strcmp(siting, "topleft") ? AJI_SITING_TOPLEFT
                                                    : AJI_SITING_LEFT;
        FILE *fin = fopen(input, "rb");
        if (!fin) { perror(input); return 1; }
        FILE *fout = output ? fopen(output, "wb") : NULL;
        if (output && !fout) { perror(output); return 1; }

        void *h_in, *h_out;
        CK(cudaMallocHost(&h_in, frame_sz));
        CK(cudaMallocHost(&h_out, frame_sz));
        aji_frame fr[3];  // two source slots + interp output
        for (int i = 0; i < 3; i++) {
            fr[i] = (aji_frame){.width = w, .height = h, .format = fmt2,
                                .matrix = mat2, .range = rng2, .siting = sit2,
                                .stride = {(ptrdiff_t)(w * bpp),
                                           (ptrdiff_t)(w * bpp)}};
            CK(cudaMalloc(&fr[i].plane[0], y_sz));
            CK(cudaMalloc(&fr[i].plane[1], uv_sz));
        }
        cudaStream_t stream;
        CK(cudaStreamCreate(&stream));

        int prev = -1, cur = 0, n = 0, interp = 0, scenes = 0;
        while (n < max_frames && fread(h_in, 1, frame_sz, fin) == frame_sz) {
            CK(cudaMemcpyAsync(fr[cur].plane[0], h_in, y_sz,
                               cudaMemcpyHostToDevice, stream));
            CK(cudaMemcpyAsync(fr[cur].plane[1], (char *)h_in + y_sz, uv_sz,
                               cudaMemcpyHostToDevice, stream));
            if (prev >= 0 && fout) {
                int ret = aji_infer_rife(aji, &fr[prev], &fr[cur], 0.5,
                                         &fr[2], stream);
                const aji_frame *emit;
                if (ret == AJI_OK) {
                    emit = &fr[2];
                    interp++;
                } else if (ret == AJI_SCENE) {
                    emit = &fr[prev];
                    scenes++;
                } else {
                    fprintf(stderr, "aji_infer_rife: %d: %s\n", ret,
                            aji_last_error(aji));
                    return 1;
                }
                CK(cudaMemcpyAsync(h_out, emit->plane[0], y_sz,
                                   cudaMemcpyDeviceToHost, stream));
                CK(cudaMemcpyAsync((char *)h_out + y_sz, emit->plane[1],
                                   uv_sz, cudaMemcpyDeviceToHost, stream));
                CK(cudaStreamSynchronize(stream));
                if (fwrite(h_out, 1, frame_sz, fout) != frame_sz) {
                    perror("fwrite");
                    return 1;
                }
            } else {
                CK(cudaStreamSynchronize(stream));
            }
            if (fout && fwrite(h_in, 1, frame_sz, fout) != frame_sz) {
                perror("fwrite");
                return 1;
            }
            if (prev < 0) {
                prev = cur;
                cur = 1;
            } else {
                int tmp = prev;
                prev = cur;
                cur = tmp;
            }
            n++;
        }
        printf("rife: %d source frames, %d interpolated, %d scene skips\n",
               n, interp, scenes);
        if (fout) fclose(fout);
        fclose(fin);
        aji_destroy(&aji);
        return 0;
    }

    if (act == 0) {
        printf("no chain active (passthrough); nothing to do\n");
        return 0;
    }
    printf("configured: %dx%d %s -> %dx%d\n", w, h, format, ow, oh);

    const size_t oy_sz = (size_t)ow * oh * bpp, ouv_sz = oy_sz / 2;

    FILE *fin = fopen(input, "rb");
    if (!fin) { perror(input); return 1; }
    FILE *fout = output ? fopen(output, "wb") : NULL;
    if (output && !fout) { perror(output); return 1; }

    void *h_in, *h_out;
    CK(cudaMallocHost(&h_in, frame_sz));
    CK(cudaMallocHost(&h_out, oy_sz + ouv_sz));

    const int mat = !strcmp(matrix, "601") ? AJI_MATRIX_BT601 :
                    !strcmp(matrix, "2020") ? AJI_MATRIX_BT2020 : AJI_MATRIX_BT709;
    const int rng = !strcmp(range, "full") ? AJI_RANGE_FULL : AJI_RANGE_LIMITED;
    const int sit = !strcmp(siting, "center") ? AJI_SITING_CENTER :
                    !strcmp(siting, "topleft") ? AJI_SITING_TOPLEFT : AJI_SITING_LEFT;

    aji_frame in = {.width = w, .height = h, .format = fmt,
                    .matrix = mat, .range = rng, .siting = sit,
                    .stride = {(ptrdiff_t)(w * bpp), (ptrdiff_t)(w * bpp)}};
    aji_frame out = {.width = ow, .height = oh, .format = fmt,
                     .matrix = mat, .range = rng, .siting = sit,
                     .stride = {(ptrdiff_t)(ow * bpp), (ptrdiff_t)(ow * bpp)}};
    CK(cudaMalloc(&in.plane[0], y_sz));
    CK(cudaMalloc(&in.plane[1], uv_sz));
    CK(cudaMalloc(&out.plane[0], oy_sz));
    CK(cudaMalloc(&out.plane[1], ouv_sz));

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));

    int n = 0, warmup = 3;
    double total_ms = 0;
    int timed = 0;

    while (n < max_frames) {
        if (fread(h_in, 1, frame_sz, fin) != frame_sz) {
            if (n == 0) {
                fprintf(stderr, "input shorter than one frame\n");
                return 1;
            }
            // loop the input so short seeds can drive long runs
            fseek(fin, 0, SEEK_SET);
            if (fread(h_in, 1, frame_sz, fin) != frame_sz)
                break;
        }
        CK(cudaMemcpyAsync(in.plane[0], h_in, y_sz, cudaMemcpyHostToDevice,
                           stream));
        CK(cudaMemcpyAsync(in.plane[1], (char *)h_in + y_sz, uv_sz,
                           cudaMemcpyHostToDevice, stream));

        CK(cudaEventRecord(ev0, stream));
        int ret = aji_infer(aji, &in, &out, stream);
        if (ret != AJI_OK) {
            fprintf(stderr, "aji_infer: %d: %s\n", ret, aji_last_error(aji));
            return 1;
        }
        CK(cudaEventRecord(ev1, stream));

        if (fout) {
            CK(cudaMemcpyAsync(h_out, out.plane[0], oy_sz,
                               cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync((char *)h_out + oy_sz, out.plane[1], ouv_sz,
                               cudaMemcpyDeviceToHost, stream));
        }
        CK(cudaStreamSynchronize(stream));

        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (n >= warmup) {
            total_ms += ms;
            timed++;
        }
        if (fout && fwrite(h_out, 1, oy_sz + ouv_sz, fout) != oy_sz + ouv_sz) {
            perror("fwrite");
            return 1;
        }
        n++;
    }

    printf("frames: %d, device chain time: %.3f ms/frame avg (%d timed)\n",
           n, timed ? total_ms / timed : 0.0, timed);

    if (fout) fclose(fout);
    fclose(fin);
    aji_destroy(&aji);
    return 0;
}
