// aji_native_vid_run MODEL.onnx IN.f16 W H OUT.f16 [ITERS]
// Runs a native temporal model (src/native/vidnet.h) on one raw fp16 [3*frames][H][W] input, writes
// the fp16 NCHW output and prints the median GPU time over ITERS runs (default 10). Built on request.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "native/vidnet.h"
#include "native/onnx_graph.h"

using namespace ajinative;

static std::vector<char> slurp(const char *p)
{
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL.onnx IN.f16 W H OUT.f16 [ITERS]\n", argv[0]);
        return 2;
    }
    const int W = atoi(argv[3]), H = atoi(argv[4]), iters = argc > 6 ? atoi(argv[6]) : 10;
    std::vector<char> model = slurp(argv[1]), in = slurp(argv[2]);
    OnnxGraph g;
    std::string why;
    if (!parse_onnx(model.data(), model.size(), &g, &why)) {
        fprintf(stderr, "parse: %s\n", why.c_str());
        return 1;
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    auto n = VidNet::create(g, W, H, s, &why);
    if (!n) {
        printf("refused: %s\n", why.c_str());
        return 3;
    }
    const int S = n->scale();
    if (in.size() != (size_t)3 * n->frames() * W * H * 2) {
        fprintf(stderr, "input is %zu bytes, expected %zu\n", in.size(), (size_t)6 * n->frames() * W * H);
        return 1;
    }
    void *din, *dout;
    cudaMalloc(&din, in.size());
    cudaMalloc(&dout, (size_t)3 * W * H * S * S * 2);
    cudaMemcpy(din, in.data(), in.size(), cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> ms;
    for (int i = 0; i < iters + 2; i++) {
        cudaEventRecord(a, s);
        cudaError_t e = n->run(din, dout, s);
        cudaEventRecord(b, s);
        if (e != cudaSuccess || cudaEventSynchronize(b) != cudaSuccess) {
            fprintf(stderr, "run failed: %s\n", cudaGetErrorString(e != cudaSuccess ? e : cudaGetLastError()));
            return 1;
        }
        float m = 0;
        cudaEventElapsedTime(&m, a, b);
        if (i >= 2) ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    std::vector<char> out((size_t)3 * W * H * S * S * 2);
    cudaMemcpy(out.data(), dout, out.size(), cudaMemcpyDeviceToHost);
    std::ofstream(argv[5], std::ios::binary).write(out.data(), out.size());
    printf("%s: %s, %.1f MB; median %.3f ms (min %.3f)\n", argv[1], n->describe().c_str(), n->device_bytes() / 1e6,
           ms[ms.size() / 2], ms[0]);
    return 0;
}
