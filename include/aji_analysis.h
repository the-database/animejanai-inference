/* Optional bounded observation ABI; independent of the RGB upscale contract.
 * No code on the normal aji inference path calls these functions. */
#ifndef AJI_ANALYSIS_H
#define AJI_ANALYSIS_H
#include "aji.h"
#ifdef __cplusplus
extern "C" {
#endif
#define AJI_ANALYSIS_VERSION 1
typedef struct aji_observer aji_observer;
typedef int (*aji_observer_cancel_fn)(void *opaque);
typedef void (*aji_observer_state_fn)(void *opaque, int state);
/* Called by the optional sampling worker, never the playback thread. Model may
 * be NULL for color only. Non-NULL must be the host-verified isnetis model. */
AJI_EXPORT aji_observer *aji_observer_create(void *cuda_context, int width, int height,
    const char *model, aji_observer_cancel_fn cancel, aji_observer_state_fn state,
    void *opaque, char *error, size_t error_size);
/* Blocks only this optional worker until its private stream is complete.
 * Image planes stay on the GPU. Only <=320x180 BGRA + soft unorm16 mask return. */
AJI_EXPORT int aji_observer_run(aji_observer *observer, const aji_frame *frame,
    const int crop[4], void *decode_stream, unsigned char *bgra, uint16_t *mask,
    double *gpu_ms);
AJI_EXPORT void aji_observer_destroy(aji_observer *observer);
#ifdef __cplusplus
}
#endif
#endif
