// aji_native_rife_run MODEL.onnx IN.f16 W H T OUT.f16 [ITERS]
// Runs native RIFE on a raw fp16 NCHW 11-channel tensor (libaji's input layout), writes the fp16
// NCHW RGB output, and prints the median GPU time over ITERS runs (default 30). Built on request.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "native/onnx_graph.h"
#include "native/rife.h"

using namespace ajinative;

static std::vector<char> slurp(const char *p)
{
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: %s MODEL.onnx IN.f16 W H T OUT.f16 [ITERS]\n", argv[0]);
        return 2;
    }
    const int W = atoi(argv[3]), H = atoi(argv[4]), iters = argc > 7 ? atoi(argv[7]) : 30;
    const float t = (float)atof(argv[5]);
    std::vector<char> model = slurp(argv[1]), in = slurp(argv[2]);
    const size_t plane = (size_t)W * H;
    if (in.size() != plane * 11 * 2) {
        fprintf(stderr, "input is %zu bytes, expected %zu\n", in.size(), plane * 22);
        return 1;
    }
    OnnxGraph g;
    std::string why;
    if (!parse_onnx(model.data(), model.size(), &g, &why)) {
        fprintf(stderr, "parse: %s\n", why.c_str());
        return 1;
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    auto r = Rife::create(g, W, H, s, &why);
    if (!r) {
        printf("refused: %s\n", why.c_str());
        return 3;
    }
    void *din, *dout;
    cudaMalloc(&din, in.size());
    cudaMalloc(&dout, plane * 3 * 2);
    cudaMemcpy(din, in.data(), in.size(), cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> ms;
    for (int i = 0; i < iters + 5; i++) {
        cudaEventRecord(a, s);
        cudaError_t e = r->run(din, t, dout, s);
        cudaEventRecord(b, s);
        if (e != cudaSuccess || cudaEventSynchronize(b) != cudaSuccess) {
            fprintf(stderr, "run failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        float m = 0;
        cudaEventElapsedTime(&m, a, b);
        if (i >= 5) ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    std::vector<char> out(plane * 3 * 2);
    cudaMemcpy(out.data(), dout, out.size(), cudaMemcpyDeviceToHost);
    std::ofstream(argv[6], std::ios::binary).write(out.data(), out.size());
    printf("%s: %s, %.1f MB; median %.3f ms (min %.3f)\n", argv[1], r->describe().c_str(), r->device_bytes() / 1e6,
           ms[ms.size() / 2], ms[0]);
    return 0;
}
