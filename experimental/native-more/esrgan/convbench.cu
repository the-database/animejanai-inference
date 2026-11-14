// Throughput of the existing native 3x3 conv (NHWC fp16, C->C, ReLU) at ESRGAN body sizes.
// usage: convbench C H W
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "native/conv.h"
using namespace ajinative;
int main(int argc, char **argv)
{
    const int C = argc > 1 ? atoi(argv[1]) : 64, H = argc > 2 ? atoi(argv[2]) : 256, W = argc > 3 ? atoi(argv[3]) : 256;
    std::vector<uint16_t> w(9 * C * C);
    std::vector<float> b(C, 0.01f);
    for (size_t i = 0; i < w.size(); i++) w[i] = float_to_half_bits(((int)(i * 2654435761u % 2001) - 1000) * 1e-5f);
    Weights wt;
    if (upload(wt, w.data(), b.data(), nullptr, 9, C, C, C, C) != cudaSuccess) { printf("upload failed\n"); return 1; }
    void *src, *dst;
    cudaMalloc(&src, (size_t)H * W * C * 2); cudaMalloc(&dst, (size_t)H * W * C * 2);
    cudaMemset(src, 0x31, (size_t)H * W * C * 2);
    Conv3x3 op; op.act = Act::Relu; op.cin = C; op.cout = C; op.w = &wt;
    Tensors t; t.src = src; t.dst = dst;
    std::string why;
    if (!supported(op, &why)) { printf("unsupported: %s\n", why.c_str()); return 1; }
    cudaStream_t s; cudaStreamCreate(&s);
    float best = 0;
    int v = pick_variant(op, t, H, W, s, &best);
    // time 200 back-to-back launches of the best variant (ping-pong src/dst like a chain)
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    for (int i = 0; i < 20; i++) launch(op, t, H, W, v, s);
    cudaEventRecord(a, s);
    const int N = 200;
    for (int i = 0; i < N; i++) { Tensors u = t; if (i & 1) { u.src = dst; u.dst = src; } launch(op, u, H, W, v, s); }
    cudaEventRecord(e, s); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, a, e); ms /= N;
    const double flop = 9.0 * C * C * 2.0 * H * W;
    printf("C%d %dx%d variant %s: %.4f ms/conv, %.1f TFLOPS\n", C, W, H, variant_name(op, v), ms, flop / ms / 1e9);
    return 0;
}
