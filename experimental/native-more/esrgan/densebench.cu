// ESRGAN dense-block conv shapes on the existing k_conv3x3 template (no code changes): cin 64..192,
// cout 32 or 64, several tile shapes, shared memory opted in above 48 KB where needed.
// usage: densebench H W
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "native/conv.h"
#include "native/conv_kernels.cuh"
using namespace ajinative;

template <int CI, int CO, int WARPS, int WR>
void bench(int H, int W)
{
    constexpr int smem = dev::conv_smem_bytes<CI, CO, WARPS, WR>();
    auto k = dev::k_conv3x3<CI, CO, WARPS, WR, 0, 1, 0, 0>;
    if (smem > 48 * 1024 && cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
        printf("cin %3d cout %2d %d warps x %d rows: %6.1f KB smem: too big\n", CI, CO, WARPS, WR, smem / 1024.0);
        cudaGetLastError();
        return;
    }
    std::vector<uint16_t> w(9 * CI * CO);
    std::vector<float> b(CO, 0.01f);
    for (size_t i = 0; i < w.size(); i++) w[i] = float_to_half_bits(((int)(i * 2654435761u % 2001) - 1000) * 1e-5f);
    Weights wt;
    upload(wt, w.data(), b.data(), nullptr, 9, CI, CI, CO, CO);
    void *src, *dst;
    cudaMalloc(&src, (size_t)H * W * CI * 2); cudaMalloc(&dst, (size_t)H * W * CO * 2);
    cudaMemset(src, 0x31, (size_t)H * W * CI * 2);
    dev::ConvParams p{};
    p.src = (const half *)src; p.dst = (half *)dst; p.w = (const uint4 *)wt.frag; p.bias = (const half2 *)wt.bias; p.H = H; p.W = W;
    constexpr int TH = WARPS * WR;
    const dim3 grid((W + dev::TW - 1) / dev::TW, (H + TH - 1) / TH);
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    for (int i = 0; i < 20; i++) k<<<grid, WARPS * 32, smem>>>(p);
    const int N = 200;
    cudaEventRecord(a);
    for (int i = 0; i < N; i++) k<<<grid, WARPS * 32, smem>>>(p);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, a, e); ms /= N;
    cudaError_t err = cudaGetLastError();
    printf("cin %3d cout %2d %d warps x %d rows: %6.1f KB smem, %.4f ms, %6.1f TFLOPS%s\n", CI, CO, WARPS, WR, smem / 1024.0, ms,
           9.0 * CI * CO * 2.0 * H * W / ms / 1e9, err ? cudaGetErrorString(err) : "");
    cudaFree(src); cudaFree(dst); release(wt);
}

#define ALL(CI, CO) bench<CI, CO, 4, 1>(H, W); bench<CI, CO, 8, 1>(H, W); bench<CI, CO, 4, 2>(H, W); bench<CI, CO, 8, 2>(H, W); bench<CI, CO, 4, 4>(H, W);
int main(int argc, char **argv)
{
    const int H = argc > 1 ? atoi(argv[1]) : 256, W = argc > 2 ? atoi(argv[2]) : 256;
    ALL(64, 32) ALL(96, 32) ALL(128, 32) ALL(160, 32) ALL(192, 64) ALL(64, 64)
    return 0;
}
