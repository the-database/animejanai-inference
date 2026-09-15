// GPU regression driver: reuse the production harness's D3D11 upload/readback
// and call the optional dispatcher ABI for each real RIFE input pair.
// AJN_SCENE_DECISION=-1/0/1; AJN_SCENE_EXPECT=AJI_OK(0)/AJI_SCENE(2).
// AJN_SCENE_LEGACY=1 verifies a new dispatcher with an older backend.
#include <stdio.h>
#include <stdlib.h>
#include "aji.h"
static int checked_scene(aji_ctx *ctx, const aji_frame *a, const aji_frame *b,
                         double t, const aji_frame *out, void *stream)
{
    const char *decision = getenv("AJN_SCENE_DECISION");
    const char *expected = getenv("AJN_SCENE_EXPECT");
    if (!decision || !expected) { fputs("Scene test expectations required\n", stderr); exit(2); }
    const bool legacy = getenv("AJN_SCENE_LEGACY") != nullptr;
    if (!!aji_rife_scene_supported(ctx) == legacy) { fputs("Wrong scene ABI support\n", stderr); exit(1); }
    if (aji_infer_rife_with_scene(ctx, a, b, t, out, stream, 2) != AJI_ERR) {
        fputs("Invalid scene value was accepted\n", stderr); exit(1);
    }
    if (legacy && aji_infer_rife_with_scene(ctx, a, b, t, out, stream, 0) != AJI_ERR) {
        fputs("Legacy backend silently accepted an override\n", stderr); exit(1);
    }
    int result = aji_infer_rife_with_scene(ctx, a, b, t, out, stream, atoi(decision));
    if (result != atoi(expected)) {
        fprintf(stderr, "Scene decision %s returned %d, expected %s: %s\n",
                decision, result, expected, aji_last_error(ctx)); exit(1);
    }
    printf("PASS real RIFE scene=%s result=%d legacy=%d\n", decision, result, legacy);
    return result;
}
#define aji_infer_rife checked_scene
#include "../src/harness_dml.cpp"
