/*
 * aji_image — CLI harness for the image ABI (aji_image.h).
 *
 * Usage:
 *   aji_image --list-devices [--backend trt|trtrtx|dml]
 *   aji_image --model M.onnx [--backend trt|trtrtx|dml] [--device ID] [--tile N]
 *             [--overlap N] [--cache DIR] [--trtexec PATH] [--gray]
 *             [--repeat N] [--async] IN.(pgm|ppm) OUT.(pgm|ppm)
 *
 * Images are binary PNM (P5 gray / P6 RGB, 8-bit). --gray feeds a P6
 * input as luma (1 channel), exercising the gray path. The model loads
 * blocking (a missing TensorRT engine is built first), or with --async
 * in the background while the CLI polls its state. Timing per run is
 * printed; --repeat N runs the same image N times (warm numbers).
 */

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "aji_image.h"

namespace {

struct Image {
    int w = 0, h = 0, ch = 0;
    std::vector<uint8_t> px;
};

int pnm_token(FILE *f)
{
    int c = fgetc(f);
    while (c == '#' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        if (c == '#')
            while (c != '\n' && c != EOF)
                c = fgetc(f);
        c = fgetc(f);
    }
    int v = 0;
    bool any = false;
    while (c >= '0' && c <= '9') {
        v = v * 10 + (c - '0');
        any = true;
        c = fgetc(f);
    }
    return any ? v : -1;  // the single whitespace after maxval is consumed
}

bool read_pnm(const char *path, Image *im)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;
    char magic[3] = {0};
    bool ok = fread(magic, 1, 2, f) == 2 && magic[0] == 'P' &&
              (magic[1] == '5' || magic[1] == '6');
    if (ok) {
        im->ch = magic[1] == '5' ? 1 : 3;
        im->w = pnm_token(f);
        im->h = pnm_token(f);
        ok = im->w > 0 && im->h > 0 && pnm_token(f) == 255;
    }
    if (ok) {
        im->px.resize((size_t)im->w * im->h * im->ch);
        ok = fread(im->px.data(), 1, im->px.size(), f) == im->px.size();
    }
    fclose(f);
    return ok;
}

bool write_pnm(const char *path, const Image &im)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    fprintf(f, "P%c\n%d %d\n255\n", im.ch == 1 ? '5' : '6', im.w, im.h);
    bool ok = fwrite(im.px.data(), 1, im.px.size(), f) == im.px.size();
    return fclose(f) == 0 && ok;
}

void log_cb(void *, int level, const char *msg)
{
    if (level <= 2)
        fprintf(stderr, "[aji:%d] %s\n", level, msg);
}

int progress_cb(void *, int done, int total)
{
    fprintf(stderr, "\r  tile %d/%d", done, total);
    if (done == total)
        fprintf(stderr, "\n");
    return 0;
}

double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int usage()
{
    fprintf(stderr,
            "usage: aji_image --list-devices [--backend trt|trtrtx|dml]\n"
            "       aji_image --model M.onnx [--backend trt|trtrtx|dml] [--device ID]\n"
            "                 [--tile N] [--overlap N] [--cache DIR] [--trtexec PATH]\n"
            "                 [--gray] [--repeat N] [--async] IN.(pgm|ppm) OUT.(pgm|ppm)\n");
    return 2;
}

} // namespace

int main(int argc, char **argv)
{
    int backend = AJI_IMG_TENSORRT;
    const char *model = nullptr, *cache = nullptr, *trtexec = nullptr;
    const char *in_path = nullptr, *out_path = nullptr;
    uint64_t device = 0;
    int tile = 0, overlap = 0, repeat = 1;
    bool list = false, gray = false, async = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--list-devices") list = true;
        else if (a == "--gray") gray = true;
        else if (a == "--async") async = true;
        else if (a == "--backend") {
            const char *v = next();
            if (!v) return usage();
            backend = !strcmp(v, "dml")      ? AJI_IMG_DIRECTML
                    : !strcmp(v, "trtrtx")   ? AJI_IMG_TENSORRT_RTX
                                             : AJI_IMG_TENSORRT;
        }
        else if (a == "--model") model = next();
        else if (a == "--cache") cache = next();
        else if (a == "--trtexec") trtexec = next();
        else if (a == "--device") { const char *v = next(); if (!v) return usage(); device = strtoull(v, nullptr, 0); }
        else if (a == "--tile") { const char *v = next(); if (!v) return usage(); tile = atoi(v); }
        else if (a == "--overlap") { const char *v = next(); if (!v) return usage(); overlap = atoi(v); }
        else if (a == "--repeat") { const char *v = next(); if (!v) return usage(); repeat = atoi(v); }
        else if (!in_path) in_path = argv[i];
        else if (!out_path) out_path = argv[i];
        else return usage();
    }

    if (list) {
        aji_img_device devs[16];
        int n = aji_img_enum_devices(backend, devs, 16);
        if (n < 0) {
            fprintf(stderr, "backend unavailable\n");
            return 1;
        }
        for (int i = 0; i < n && i < 16; i++)
            printf("0x%016" PRIx64 "  %s  %.1f GB  sm %d.%d\n", devs[i].id,
                   devs[i].name, devs[i].vram_bytes / 1073741824.0,
                   devs[i].sm_major, devs[i].sm_minor);
        return 0;
    }
    if (!model || !in_path || !out_path)
        return usage();

    Image src;
    if (!read_pnm(in_path, &src)) {
        fprintf(stderr, "cannot read %s (binary 8-bit PGM/PPM only)\n", in_path);
        return 1;
    }
    if (gray && src.ch == 3) {
        std::vector<uint8_t> l((size_t)src.w * src.h);
        for (size_t i = 0; i < l.size(); i++) {
            const uint8_t *p = &src.px[i * 3];
            l[i] = (uint8_t)((299 * p[0] + 587 * p[1] + 114 * p[2] + 500) / 1000);
        }
        src.px.swap(l);
        src.ch = 1;
    }

    aji_img_create_params cp = {};
    cp.api_version = AJI_IMAGE_API_VERSION;
    cp.backend = backend;
    cp.device_id = device;
    cp.cache_dir = cache;
    cp.trtexec = trtexec;
    cp.log = log_cb;
    char err[512] = {0};
    aji_img_ctx *c = aji_img_create(&cp, err, sizeof(err));
    if (!c) {
        fprintf(stderr, "create failed: %s\n", err);
        return 1;
    }

    aji_img_model_params mp = {};
    mp.onnx_path = model;
    mp.tile_w = mp.tile_h = tile;
    mp.overlap = overlap;
    mp.flags = async ? AJI_IMG_LOAD_ASYNC_BUILD : 0;
    double t0 = now_ms();
    aji_img_model *m = aji_img_load(c, &mp);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", aji_img_last_error(c));
        aji_img_destroy(&c);
        return 1;
    }
    int polls = 0;
    while (aji_img_model_state(m) == AJI_IMG_STATE_BUILDING) {
        if (polls++ % 50 == 0)
            fprintf(stderr, "  building engine... %.0f s\n", (now_ms() - t0) / 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    aji_img_model_info info = {};
    if (aji_img_model_info_get(m, &info) != AJI_IMG_OK) {
        fprintf(stderr, "model not ready: %s\n", aji_img_last_error(c));
        return 1;
    }
    fprintf(stderr, "model: %dx, in %dch out %dch, tile %dx%d overlap %d (load %.0f ms)\n",
            info.scale, info.in_channels, info.out_channels, info.tile_w,
            info.tile_h, info.overlap, now_ms() - t0);

    Image dst;
    dst.w = src.w * info.scale;
    dst.h = src.h * info.scale;
    dst.ch = src.ch;
    dst.px.resize((size_t)dst.w * dst.h * dst.ch);
    aji_img_buffer ib = {src.w, src.h, src.ch, (ptrdiff_t)src.w * src.ch, src.px.data()};
    aji_img_buffer ob = {dst.w, dst.h, dst.ch, (ptrdiff_t)dst.w * dst.ch, dst.px.data()};
    int rc = AJI_IMG_OK;
    for (int r = 0; r < repeat && rc == AJI_IMG_OK; r++) {
        t0 = now_ms();
        rc = aji_img_upscale(m, &ib, &ob, repeat == 1 ? progress_cb : nullptr, nullptr);
        fprintf(stderr, "upscale %dx%d -> %dx%d: %.1f ms\n", src.w, src.h, dst.w,
                dst.h, now_ms() - t0);
    }
    if (rc != AJI_IMG_OK) {
        fprintf(stderr, "upscale failed (%d): %s\n", rc, aji_img_last_error(c));
    } else if (!write_pnm(out_path, dst)) {
        fprintf(stderr, "cannot write %s\n", out_path);
        rc = AJI_IMG_ERR;
    }
    aji_img_release(&m);
    aji_img_destroy(&c);
    return rc == AJI_IMG_OK ? 0 : 1;
}
