// Regression for NVDEC's planar MSB-aligned 4:4:4 surfaces (AJN issue #205).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include "aji.h"
#include "kernels.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

int main()
{
    uint16_t *yuv, *restored;
    __half *rgb;
    CHECK(cudaMalloc(&yuv, 12 * sizeof(*yuv)) == cudaSuccess);
    CHECK(cudaMalloc(&restored, 12 * sizeof(*restored)) == cudaSuccess);
    CHECK(cudaMalloc(&rgb, 12 * sizeof(*rgb)) == cudaSuccess);
    for (int bits : {10, 12, 16}) {
        const int fmt = bits == 10 ? AJI_FMT_YUV444P10MSB :
                        bits == 12 ? AJI_FMT_YUV444P12MSB : AJI_FMT_YUV444P16;
        const int shift = 16 - bits;
        const int peak = ((1 << bits) - 1) << shift;
        for (int range : {AJI_RANGE_LIMITED, AJI_RANGE_FULL}) {
            const bool full = range == AJI_RANGE_FULL;
            const int black = full ? 0 : 4096;
            const int white = full ? peak : 60160;
            const int mid = ((black + white) / 2) & ~((1 << shift) - 1);
            uint16_t src[12] = {
                (uint16_t)black, (uint16_t)white, (uint16_t)mid, (uint16_t)mid,
                32768, 32768, 36864, 28672,
                32768, 32768, 28672, 36864,
            };
            CHECK(cudaMemcpy(yuv, src, sizeof(src), cudaMemcpyHostToDevice) == cudaSuccess);
            const aji_csp csp = aji_make_csp(fmt, AJI_MATRIX_BT601, range);
            CHECK(aji_run_pre444(2, 2, yuv, 4, yuv + 4, 4, yuv + 8, 4,
                                &csp, rgb, nullptr) == 0);
            __half got[12];
            CHECK(cudaMemcpy(got, rgb, sizeof(got), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (int i = 0; i < 4; i++) {
                const float Y = float(src[i] - black) / (white - black);
                const float U = float(src[4 + i] - 32768) / (full ? peak : 57344);
                const float V = float(src[8 + i] - 32768) / (full ? peak : 57344);
                const float expected[3] = {Y + 1.402f * V,
                    Y - (0.114f * 1.772f * U + 0.299f * 1.402f * V) / 0.587f,
                    Y + 1.772f * U};
                for (int p = 0; p < 3; p++)
                    CHECK(std::fabs(__half2float(got[p * 4 + i]) - expected[p]) < 0.0005f);
            }
            CHECK(aji_run_post444(fmt, 2, 2, rgb, &csp, restored, 4,
                                  restored + 4, 4, restored + 8, 4, nullptr) == 0);
            uint16_t back[12];
            CHECK(cudaMemcpy(back, restored, sizeof(back), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (int i = 0; i < 12; i++) {
                CHECK((back[i] & ((1 << shift) - 1)) == 0);
                CHECK(back[i] <= peak);
                // fp16 model input/output adds rounding beyond integer quantization.
                CHECK(std::abs(int(back[i]) - src[i]) <= (1 << shift) + 32);
            }
            CHECK(back[0] == black && back[1] == white);
        }
    }
    CHECK(cudaFree(rgb) == cudaSuccess);
    CHECK(cudaFree(restored) == cudaSuccess);
    CHECK(cudaFree(yuv) == cudaSuccess);
    std::puts("MSB 4:4:4 GPU normalization and quantization passed");
}
