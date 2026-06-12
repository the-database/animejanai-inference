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
    int w = 0, h = 0, frames = 1, slot = 1;
    int format = AJI_FMT_NV12, matrix = AJI_MATRIX_BT709;
    int range = AJI_RANGE_LIMITED, siting = AJI_SITING_LEFT;
    double fps = 23.976;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--input")) { input = v; i++; }
        else if (!strcmp(a, "--output")) { output = v; i++; }
        else if (!strcmp(a, "--conf")) { conf = v; i++; }
        else if (!strcmp(a, "--model-dir")) { mdir = v; i++; }
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
