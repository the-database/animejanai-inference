/*
 * aji_image.h — still-image super-resolution, C ABI (version 1).
 *
 * The image front-end of libaji, separate from the video ABI in aji.h:
 * no conf slots, chains, fps or YUV. Pixels travel in host memory as
 * 8-bit gray or RGB; tiling, overlap blending and dtype conversion
 * happen inside. Policy (when to upscale, which model, grayscale
 * detection, pre/post resizes) belongs to the caller.
 *
 * Exported by aji.dll / libaji.so, which forwards to the backend chosen
 * at aji_img_create (aji_trt / aji_dml, loaded from its own directory),
 * and by the backend libraries themselves.
 *
 * Models are ONNX files with one NCHW input and one NCHW output
 * (C = 1 or 3), an integer scale, and a dynamic or fixed spatial shape.
 * Each loaded model runs on fixed-size tiles: TensorRT builds one static
 * engine per (model, tile, GPU, TensorRT version), cached on disk;
 * DirectML compiles the tile shape when the model loads.
 *
 * Threading: a context may be used from any thread. Calls on one model
 * are serialized internally; different models may run concurrently.
 */

#ifndef AJI_IMAGE_H
#define AJI_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define AJI_IMG_EXPORT __declspec(dllexport)
#else
#  define AJI_IMG_EXPORT __attribute__((visibility("default")))
#endif

#define AJI_IMAGE_API_VERSION 1

typedef struct aji_img_ctx aji_img_ctx;
typedef struct aji_img_model aji_img_model;

enum aji_img_backend {
    AJI_IMG_TENSORRT = 1,
    AJI_IMG_DIRECTML = 2,   /* Windows only */
};

enum aji_img_status {
    AJI_IMG_OK         = 0,
    AJI_IMG_ERR        = -1,
    AJI_IMG_ERR_ARGS   = -2,  /* bad buffer dims/channels/stride */
    AJI_IMG_NOT_READY  = -3,  /* model still building (or failed) */
    AJI_IMG_CANCELLED  = -4,  /* progress callback asked to stop */
    AJI_IMG_ERR_NO_MEM = -5,  /* device or host allocation failed */
};

enum aji_img_model_state {
    AJI_IMG_STATE_READY    = 0,
    AJI_IMG_STATE_BUILDING = 1,  /* TensorRT engine build queued/running */
    AJI_IMG_STATE_FAILED   = -1, /* see aji_img_last_error */
};

typedef void (*aji_img_log_fn)(void *opaque, int level, const char *msg);

/* Return nonzero to cancel; called between tiles. */
typedef int (*aji_img_progress_fn)(void *opaque, int tiles_done,
                                   int tiles_total);

typedef struct aji_img_device {
    uint64_t id;            /* stable across runs: the PCI domain/bus/
                               device, the same value on both backends
                               on single-PCI-segment (desktop) systems
                               (DirectML falls back to the adapter LUID
                               if Windows reports no address). Pass to
                               aji_img_create. */
    char name[128];
    uint64_t vram_bytes;
    int sm_major, sm_minor; /* CUDA compute capability; 0 for DirectML */
} aji_img_device;

/* Number of usable devices for `backend`, filling up to `max` entries
 * of `out` (which may be NULL). Negative if the backend or its runtime
 * cannot be loaded. Each physical GPU is listed once (DirectML skips the
 * adapters Windows adds for virtual displays). */
AJI_IMG_EXPORT int aji_img_enum_devices(int backend, aji_img_device *out,
                                        int max);

typedef struct aji_img_create_params {
    uint32_t api_version;     /* AJI_IMAGE_API_VERSION */
    int backend;              /* enum aji_img_backend */
    uint64_t device_id;       /* aji_img_device.id; 0 = first device */
    const char *cache_dir;    /* TensorRT engine/timing cache; NULL = the
                                 model's own directory */
    const char *trtexec;      /* TensorRT: trtexec binary that builds
                                 missing engines; NULL = trtexec next to
                                 the backend library */
    const char *trtexec_env;  /* optional POSIX env prefix for trtexec */
    aji_img_log_fn log;
    void *log_opaque;
} aji_img_create_params;

/* NULL on failure, with the reason in err (if err_size > 0). */
AJI_IMG_EXPORT aji_img_ctx *aji_img_create(const aji_img_create_params *p,
                                           char *err, size_t err_size);

enum aji_img_load_flags {
    /* Build a missing TensorRT engine on a background thread instead of
     * inside aji_img_load: the model comes back BUILDING and becomes READY
     * later (poll aji_img_model_state). For interactive apps that must not
     * stall on a first-use build; the counterpart of the video ABI's
     * aji_create_params.async_build. No effect on DirectML. */
    AJI_IMG_LOAD_ASYNC_BUILD = 1,
};

typedef struct aji_img_model_params {
    const char *onnx_path;
    int tile_w, tile_h;  /* input-pixel tile size; 0 = 512. Ignored for
                            fixed-shape models (their shape is the tile) */
    int overlap;         /* input-pixel overlap between neighbouring
                            tiles, blended linearly; 0 = 32 */
    int flags;           /* enum aji_img_load_flags */
} aji_img_model_params;

/* Returns a handle, or NULL with the reason in aji_img_last_error(ctx).
 * By default a TensorRT model whose engine is not cached is built here
 * (minutes, once per model/tile/GPU/TensorRT version), so the model is
 * READY on return. With AJI_IMG_LOAD_ASYNC_BUILD it comes back BUILDING
 * instead; background builds run one at a time per context. */
AJI_IMG_EXPORT aji_img_model *aji_img_load(aji_img_ctx *c,
                                           const aji_img_model_params *p);

/* Current state; also advances background builds (cheap to poll). */
AJI_IMG_EXPORT int aji_img_model_state(aji_img_model *m);

typedef struct aji_img_model_info {
    int scale;                /* output = scale * input, both axes */
    int in_channels;          /* the model's own input channels (1 or 3) */
    int out_channels;
    int tile_w, tile_h;       /* effective tile (input pixels) */
    int overlap;
} aji_img_model_info;

/* Valid once the model is READY. */
AJI_IMG_EXPORT int aji_img_model_info_get(aji_img_model *m,
                                          aji_img_model_info *info);

typedef struct aji_img_buffer {
    int width, height;
    int channels;             /* 1 = gray, 3 = RGB (8 bits each) */
    ptrdiff_t stride;         /* bytes per row, >= width * channels */
    uint8_t *data;
} aji_img_buffer;

/* Upscale `in` into `out`. out->width/height must be scale * in's, and
 * out->channels must equal in->channels. A gray input is replicated to
 * a 3-channel model's input on the way in, and channel 0 of the output
 * is kept. Images smaller than the tile are reflect-padded to it.
 * `progress` may be NULL. Blocks until done. */
AJI_IMG_EXPORT int aji_img_upscale(aji_img_model *m, const aji_img_buffer *in,
                                   const aji_img_buffer *out,
                                   aji_img_progress_fn progress, void *opaque);

AJI_IMG_EXPORT void aji_img_release(aji_img_model **m);

/* Last error on this context (thread-unsafe snapshot; for diagnostics). */
AJI_IMG_EXPORT const char *aji_img_last_error(aji_img_ctx *c);

/* Releases any models still loaded and stops a running engine build. */
AJI_IMG_EXPORT void aji_img_destroy(aji_img_ctx **c);

#ifdef __cplusplus
}
#endif

#endif
