// The allocator's live pointers give a process-local baseline, even when another
// process uses the GPU or the driver retains pages after cudaFree.
#include "aji_image.h"

#include <cuda_runtime_api.h>
#include <cupti.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>

struct Allocations {
    std::map<void *, size_t> live;
    size_t allocations = 0;
};

static void CUPTIAPI allocation(void *user, CUpti_CallbackDomain, CUpti_CallbackId id, const void *data)
{
    const auto &cb = *static_cast<const CUpti_CallbackData *>(data);
    if (cb.callbackSite != CUPTI_API_EXIT || *static_cast<const cudaError_t *>(cb.functionReturnValue) != cudaSuccess) return;
    auto &state = *static_cast<Allocations *>(user);
    if (id == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020) {
        const auto &p = *static_cast<const cudaMalloc_v3020_params *>(cb.functionParams);
        state.live[*p.devPtr] = p.size;
        ++state.allocations;
    } else if (id == CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020) {
        const auto &p = *static_cast<const cudaFree_v3020_params *>(cb.functionParams);
        state.live.erase(p.devPtr);
    }
}

static bool exercise(aji_img_ctx *ctx, const char *path, const Allocations &state, bool probe)
{
    const aji_img_model_params params{path, 128, 128, 32, AJI_IMG_LOAD_EVEN_TILES};
    const auto baseline = state.live;
    for (int i = 0; i < 3; ++i) {
        const auto before = state.allocations;
        aji_img_memory info{};
        if (probe) {
            if (aji_img_native_probe(ctx, &params, &info) != AJI_IMG_OK) return false;
        } else {
            auto *model = aji_img_load(ctx, &params);
            if (!model) return false;
            const bool native = aji_img_memory_get(ctx, model, &info) == AJI_IMG_OK &&
                                info.model_activations && !info.activations_shared;
            aji_img_release(&model);
            if (!native) return false;
        }
        if (state.allocations == before || state.live != baseline) {
            size_t extra = 0;
            for (const auto &p : state.live)
                if (!baseline.count(p.first)) extra += p.second;
            std::fprintf(stderr, "%s iteration %d: %zu new live device bytes, %zu CUDA allocations observed\n",
                         probe ? "probe" : "release", i + 1, extra, state.allocations - before);
            return false;
        }
    }
    return true;
}

int main()
{
    const char *dir = std::getenv("AJI_TEST_IMAGE_MODELS");
    if (!dir || !*dir) {
        std::puts("Set AJI_TEST_IMAGE_MODELS to the shipped image ONNX directory");
        return 77;
    }
    const char *names[] = {
        "2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.onnx",
        "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx",
        "4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx",
        "4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx",
        "2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.onnx",
    };
    aji_img_create_params make{};
    make.api_version = AJI_IMAGE_API_VERSION;
    make.backend = AJI_IMG_TENSORRT;
    char err[512]{};
    auto *ctx = aji_img_create(&make, err, sizeof(err));
    if (!ctx) { std::fprintf(stderr, "%s\n", err); return 1; }

    Allocations state;
    CUpti_SubscriberHandle sub{};
    auto rc = cuptiSubscribe(&sub, allocation, &state);
    if (rc == CUPTI_SUCCESS)
        rc = cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020);
    if (rc == CUPTI_SUCCESS)
        rc = cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020);
    bool ok = rc == CUPTI_SUCCESS;
    if (!ok) std::fprintf(stderr, "CUPTI allocation tracing failed (%d)\n", (int)rc);
    if (ok) {
        for (const char *name : names) {
            const auto path = (std::filesystem::u8path(dir) / name).u8string();
            for (bool probe : {false, true}) {
                if (!exercise(ctx, path.c_str(), state, probe)) {
                    std::fprintf(stderr, "%s: %s failed, %s\n", name, probe ? "probe" : "release", aji_img_last_error(ctx));
                    ok = false;
                }
            }
        }
    }
    if (sub) cuptiUnsubscribe(sub);
    aji_img_destroy(&ctx);
    return ok ? 0 : 1;
}
