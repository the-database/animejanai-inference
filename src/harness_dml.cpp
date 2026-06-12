/*
 * aji_harness_dml — CLI harness for the DirectML backend (Windows).
 *
 * Mirrors aji_harness's raw-frame interface, but frames travel as
 * shared D3D11 NV12/P010 textures, exactly like the mpv filter's D3D11
 * path: raw planes -> staging texture -> shared texture -> aji_infer
 * (which fences to D3D12/DirectML and back) -> staging -> raw out.
 *
 * Usage:
 *   aji_harness_dml --input in.raw --width W --height H
 *                   [--format nv12|p010] [--matrix 601|709|2020]
 *                   [--range limited|full] [--siting left|center|topleft]
 *                   [--fps F] [--frames N] [--output out.raw]
 *                   --conf animejanai.conf --model-dir DIR [--slot N]
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aji.h"

static void log_cb(void *opaque, int level, const char *msg)
{
    (void)opaque;
    if (level <= 2)
        fprintf(stderr, "[dml:%d] %s\n", level, msg);
}

static ID3D11Texture2D *make_tex(ID3D11Device *dev, int w, int h,
                                 DXGI_FORMAT fmt, bool staging, bool write)
{
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    if (staging) {
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = write ? D3D11_CPU_ACCESS_WRITE
                                  : D3D11_CPU_ACCESS_READ;
    } else {
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                       D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    }
    ID3D11Texture2D *tex = NULL;
    if (FAILED(dev->CreateTexture2D(&td, NULL, &tex)))
        return NULL;
    return tex;
}

int main(int argc, char **argv)
{
    const char *input = NULL, *output = NULL, *conf = NULL, *mdir = NULL;
    const char *rife_mdir = NULL;
    int w = 0, h = 0, frames = 1, slot = 1, rife = 0;
    int format = AJI_FMT_NV12, matrix = AJI_MATRIX_BT709;
    int range = AJI_RANGE_LIMITED, siting = AJI_SITING_LEFT;
    double fps = 23.976;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--rife")) { rife = 1; continue; }
        if (!strcmp(a, "--input")) { input = v; i++; }
        else if (!strcmp(a, "--output")) { output = v; i++; }
        else if (!strcmp(a, "--conf")) { conf = v; i++; }
        else if (!strcmp(a, "--model-dir")) { mdir = v; i++; }
        else if (!strcmp(a, "--rife-model-dir")) { rife_mdir = v; i++; }
        else if (!strcmp(a, "--width")) { w = atoi(v); i++; }
        else if (!strcmp(a, "--height")) { h = atoi(v); i++; }
        else if (!strcmp(a, "--frames")) { frames = atoi(v); i++; }
        else if (!strcmp(a, "--slot")) { slot = atoi(v); i++; }
        else if (!strcmp(a, "--fps")) { fps = atof(v); i++; }
        else if (!strcmp(a, "--format")) {
            format = strcmp(v, "p010") ? AJI_FMT_NV12 : AJI_FMT_P010; i++;
        } else if (!strcmp(a, "--matrix")) {
            matrix = !strcmp(v, "601") ? AJI_MATRIX_BT601 :
                     !strcmp(v, "2020") ? AJI_MATRIX_BT2020 :
                     AJI_MATRIX_BT709; i++;
        } else if (!strcmp(a, "--range")) {
            range = strcmp(v, "full") ? AJI_RANGE_LIMITED : AJI_RANGE_FULL;
            i++;
        } else if (!strcmp(a, "--siting")) {
            siting = !strcmp(v, "center") ? AJI_SITING_CENTER :
                     !strcmp(v, "topleft") ? AJI_SITING_TOPLEFT :
                     AJI_SITING_LEFT; i++;
        }
    }
    if (!input || !conf || !mdir || w < 2 || h < 2) {
        fprintf(stderr, "usage: see header (input/conf/model-dir/width/"
                        "height required)\n");
        return 2;
    }

    ID3D11Device *dev = NULL;
    ID3D11DeviceContext *ctx = NULL;
    if (FAILED(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0,
                                 NULL, 0, D3D11_SDK_VERSION, &dev, NULL,
                                 &ctx))) {
        fprintf(stderr, "D3D11CreateDevice failed\n");
        return 1;
    }

    aji_create_params p = {};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf;
    p.model_dir = mdir;
    p.rife_model_dir = rife_mdir;
    p.slot = slot;
    p.d3d11_device = dev;
    p.log = log_cb;
    aji_ctx *aji = aji_create(&p);
    if (!aji) {
        fprintf(stderr, "aji_create failed\n");
        return 1;
    }

    int ow = 0, oh = 0;
    int r = aji_configure(aji, w, h, fps, &ow, &oh);
    fputs(aji_current_log(aji), stdout);
    if (r < 0) {
        fprintf(stderr, "configure failed: %s\n", aji_last_error(aji));
        return 1;
    }
    printf("configured: %dx%d %s -> %dx%d\n", w, h,
           format == AJI_FMT_P010 ? "p010" : "nv12", ow, oh);

    if (rife) {
        // Interpolation-only loop replicating aji_harness --rife: emits
        // f0, i01, f1, i12, ... with the left frame substituted on scene
        // changes. Needs a 2/1 rife chain (and no scaling steps).
        int rn = 0, rd = 0;
        if (!aji_rife_factor(aji, &rn, &rd) || rn != 2 * rd || ow != w ||
            oh != h) {
            fprintf(stderr, "rife test needs an active scale-free 2/1 rife "
                            "chain\n");
            return 1;
        }
        const int bpp2 = format == AJI_FMT_P010 ? 2 : 1;
        const DXGI_FORMAT dxf = format == AJI_FMT_P010 ? DXGI_FORMAT_P010
                                                       : DXGI_FORMAT_NV12;
        ID3D11Texture2D *fr[3], *st_in, *st_out;
        for (int i = 0; i < 3; i++)
            fr[i] = make_tex(dev, w, h, dxf, false, false);
        st_in = make_tex(dev, w, h, dxf, true, true);
        st_out = make_tex(dev, w, h, dxf, true, false);
        if (!fr[0] || !fr[1] || !fr[2] || !st_in || !st_out) {
            fprintf(stderr, "rife texture creation failed\n");
            return 1;
        }
        FILE *fi = fopen(input, "rb");
        if (!fi) { fprintf(stderr, "cannot open %s\n", input); return 1; }
        FILE *fo = output ? fopen(output, "wb") : NULL;
        if (output && !fo) { fprintf(stderr, "cannot open %s\n", output);
                             return 1; }
        const size_t fb = (size_t)w * h * bpp2 * 3 / 2;
        char *raw = (char *)malloc(fb);

        aji_frame fd[3];
        for (int i = 0; i < 3; i++) {
            fd[i] = aji_frame{};
            fd[i].width = w;
            fd[i].height = h;
            fd[i].format = format;
            fd[i].matrix = matrix;
            fd[i].range = range;
            fd[i].siting = siting;
            fd[i].plane[0] = fr[i];
        }

        LARGE_INTEGER rfreq, rt0, rt1;
        QueryPerformanceFrequency(&rfreq);
        double rife_ms = 0;
        int rife_timed = 0;
        int prev = -1, cur = 0, n = 0, interp = 0, scenes = 0;
        while (n < frames && fread(raw, 1, fb, fi) == fb) {
            D3D11_MAPPED_SUBRESOURCE map = {};
            if (FAILED(ctx->Map(st_in, 0, D3D11_MAP_WRITE, 0, &map)))
                return 1;
            const char *s = raw;
            char *d = (char *)map.pData;
            for (int y = 0; y < h; y++)
                memcpy(d + (size_t)y * map.RowPitch,
                       s + (size_t)y * w * bpp2, (size_t)w * bpp2);
            s += (size_t)w * h * bpp2;
            d += (size_t)map.RowPitch * h;
            for (int y = 0; y < h / 2; y++)
                memcpy(d + (size_t)y * map.RowPitch,
                       s + (size_t)y * w * bpp2, (size_t)w * bpp2);
            ctx->Unmap(st_in, 0);
            ctx->CopyResource(fr[cur], st_in);

            if (prev >= 0 && fo) {
                QueryPerformanceCounter(&rt0);
                int ret = aji_infer_rife(aji, &fd[prev], &fd[cur], 0.5,
                                         &fd[2], NULL);
                QueryPerformanceCounter(&rt1);
                if (interp + scenes >= 1) {   // skip the lazy-init call
                    rife_ms += (double)(rt1.QuadPart - rt0.QuadPart) *
                               1000.0 / rfreq.QuadPart;
                    rife_timed++;
                }
                int emit;
                if (ret == AJI_OK) {
                    emit = 2;
                    interp++;
                } else if (ret == AJI_SCENE) {
                    emit = prev;
                    scenes++;
                } else {
                    fprintf(stderr, "aji_infer_rife: %d: %s\n", ret,
                            aji_last_error(aji));
                    return 1;
                }
                ctx->CopyResource(st_out, fr[emit]);
                if (FAILED(ctx->Map(st_out, 0, D3D11_MAP_READ, 0, &map)))
                    return 1;
                const char *ss = (const char *)map.pData;
                char *dd = raw;   // reuse as out scratch after the upload
                char *tmp = (char *)malloc(fb);
                dd = tmp;
                for (int y = 0; y < h; y++)
                    memcpy(dd + (size_t)y * w * bpp2,
                           ss + (size_t)y * map.RowPitch, (size_t)w * bpp2);
                ss += (size_t)map.RowPitch * h;
                dd += (size_t)w * h * bpp2;
                for (int y = 0; y < h / 2; y++)
                    memcpy(dd + (size_t)y * w * bpp2,
                           ss + (size_t)y * map.RowPitch, (size_t)w * bpp2);
                ctx->Unmap(st_out, 0);
                fwrite(tmp, 1, fb, fo);
                free(tmp);
            }
            if (fo)
                fwrite(raw, 1, fb, fo);
            if (prev < 0) {
                prev = cur;
                cur = 1;
            } else {
                int tmp2 = prev;
                prev = cur;
                cur = tmp2;
            }
            n++;
        }
        printf("rife: %d source frames, %d interpolated, %d scene skips"
               ", %.1f ms/interp avg (%d timed)\n",
               n, interp, scenes, rife_timed ? rife_ms / rife_timed : 0.0,
               rife_timed);
        if (fo)
            fclose(fo);
        fclose(fi);
        aji_destroy(&aji);
        return 0;
    }

    if (r == 0) {
        fprintf(stderr, "no chain active; nothing to do\n");
        return 0;
    }

    const int bpp = format == AJI_FMT_P010 ? 2 : 1;
    const DXGI_FORMAT dxfmt = format == AJI_FMT_P010 ? DXGI_FORMAT_P010
                                                     : DXGI_FORMAT_NV12;
    ID3D11Texture2D *in_tex = make_tex(dev, w, h, dxfmt, false, false);
    ID3D11Texture2D *out_tex = make_tex(dev, ow, oh, dxfmt, false, false);
    ID3D11Texture2D *in_st = make_tex(dev, w, h, dxfmt, true, true);
    ID3D11Texture2D *out_st = make_tex(dev, ow, oh, dxfmt, true, false);
    if (!in_tex || !out_tex || !in_st || !out_st) {
        fprintf(stderr, "texture creation failed (shared NV12/P010)\n");
        return 1;
    }

    FILE *fi = fopen(input, "rb");
    if (!fi) {
        fprintf(stderr, "cannot open %s\n", input);
        return 1;
    }
    FILE *fo = output ? fopen(output, "wb") : NULL;
    if (output && !fo) {
        fprintf(stderr, "cannot open %s\n", output);
        return 1;
    }

    const size_t in_frame_bytes = (size_t)w * h * bpp * 3 / 2;
    char *raw = (char *)malloc(in_frame_bytes);
    const size_t out_frame_bytes = (size_t)ow * oh * bpp * 3 / 2;
    char *raw_out = (char *)malloc(out_frame_bytes);

    aji_frame fin = {};
    fin.width = w;
    fin.height = h;
    fin.format = format;
    fin.matrix = matrix;
    fin.range = range;
    fin.siting = siting;
    fin.plane[0] = in_tex;
    fin.plane[1] = (void *)(intptr_t)0;
    aji_frame fout = fin;
    fout.width = ow;
    fout.height = oh;
    fout.plane[0] = out_tex;

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    double total_ms = 0;
    int timed = 0;

    for (int f = 0; f < frames; f++) {
        if (fread(raw, 1, in_frame_bytes, fi) != in_frame_bytes) {
            if (f == 0) {
                fprintf(stderr, "input shorter than one frame\n");
                return 1;
            }
            fseek(fi, 0, SEEK_SET);
            if (fread(raw, 1, in_frame_bytes, fi) != in_frame_bytes)
                return 1;
        }

        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(ctx->Map(in_st, 0, D3D11_MAP_WRITE, 0, &map))) {
            fprintf(stderr, "map in staging failed\n");
            return 1;
        }
        const char *src = raw;
        char *dst = (char *)map.pData;
        for (int y = 0; y < h; y++)        // Y plane
            memcpy(dst + (size_t)y * map.RowPitch, src + (size_t)y * w * bpp,
                   (size_t)w * bpp);
        src += (size_t)w * h * bpp;
        dst += (size_t)map.RowPitch * h;   // UV plane (interleaved)
        for (int y = 0; y < h / 2; y++)
            memcpy(dst + (size_t)y * map.RowPitch, src + (size_t)y * w * bpp,
                   (size_t)w * bpp);
        ctx->Unmap(in_st, 0);
        ctx->CopyResource(in_tex, in_st);

        QueryPerformanceCounter(&t0);
        int ir = aji_infer(aji, &fin, &fout, NULL);
        QueryPerformanceCounter(&t1);
        if (ir != AJI_OK) {
            fprintf(stderr, "aji_infer failed (%d): %s\n", ir,
                    aji_last_error(aji));
            return 1;
        }
        if (f >= 2) {
            total_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 /
                        freq.QuadPart;
            timed++;
        }

        if (fo) {
            ctx->CopyResource(out_st, out_tex);
            if (FAILED(ctx->Map(out_st, 0, D3D11_MAP_READ, 0, &map))) {
                fprintf(stderr, "map out staging failed\n");
                return 1;
            }
            const char *s = (const char *)map.pData;
            char *d = raw_out;
            for (int y = 0; y < oh; y++)
                memcpy(d + (size_t)y * ow * bpp,
                       s + (size_t)y * map.RowPitch, (size_t)ow * bpp);
            s += (size_t)map.RowPitch * oh;
            d += (size_t)ow * oh * bpp;
            for (int y = 0; y < oh / 2; y++)
                memcpy(d + (size_t)y * ow * bpp,
                       s + (size_t)y * map.RowPitch, (size_t)ow * bpp);
            ctx->Unmap(out_st, 0);
            fwrite(raw_out, 1, out_frame_bytes, fo);
        }
    }

    printf("frames: %d, infer time: %.3f ms/frame avg (%d timed)\n", frames,
           timed ? total_ms / timed : 0.0, timed);

    if (fo)
        fclose(fo);
    fclose(fi);
    aji_destroy(&aji);
    in_tex->Release();
    out_tex->Release();
    in_st->Release();
    out_st->Release();
    ctx->Release();
    dev->Release();
    return 0;
}
