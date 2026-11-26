// aji_native_tfix_workspace_test S1.onnx S2.onnx S3.onnx
#include "native/onnx_graph.h"
#include "native/temporal_workspace.h"
#include "native/vidnet.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

using namespace ajinative;

static_assert(tfix_alignment_chains(500000, 89) == 6 && tfix_alignment_chains(500000, 86) == 6);
static_assert(tfix_alignment_chains(500001, 80) == 2 && tfix_alignment_chains(500001, 86) == 2);
static_assert(tfix_alignment_chains(500001, 89) == 1 && tfix_alignment_chains(500001, 120) == 1);

static size_t model_bytes(const OnnxGraph &graph, int w, int h, const char *chains)
{
#ifdef _WIN32
    _putenv_s("AJI_NATIVE_TFIX_CHAINS", chains);
#else
    if (*chains) setenv("AJI_NATIVE_TFIX_CHAINS", chains, 1);
    else unsetenv("AJI_NATIVE_TFIX_CHAINS");
#endif
    std::string why;
    auto net = VidNet::create(graph, w, h, nullptr, &why);
    if (!net) throw std::runtime_error(why);
    return net->device_bytes();
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s S1.onnx S2.onnx S3.onnx\n", argv[0]);
        return 2;
    }
    int device;
    cudaDeviceProp gpu{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&gpu, device) != cudaSuccess) return 1;
    const bool one_large_chain = (gpu.major == 8 && gpu.minor == 9) || (gpu.major == 12 && gpu.minor == 0);
    const bool measured_gpu = gpu.major == 12 && gpu.minor == 0 && gpu.multiProcessorCount == 170;
    for (int i = 1; i <= 3; ++i) {
        std::ifstream input(argv[i], std::ios::binary);
        const std::vector<char> data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        OnnxGraph graph;
        std::string why;
        if (!parse_onnx(data.data(), data.size(), &graph, &why)) {
            std::fprintf(stderr, "%s\n", why.c_str());
            return 1;
        }
        // Split-K workspace varies with SM count. Measure both explicit policies on this device.
        for (int h : {496, 504, 1080}) {
            const int w = h == 1080 ? 1920 : 1000;
            const size_t actual = model_bytes(graph, w, h, "");
            const size_t two = model_bytes(graph, w, h, "2");
            const size_t one = model_bytes(graph, w, h, "1");
            const size_t three = model_bytes(graph, w, h, "3");
            const size_t want = h == 496 ? one + 5 * (two - one) : (one_large_chain ? one : two);
            std::printf("S%d %dx%d default %zu bytes, one chain %zu, two chains %zu\n", i, w, h, actual, one, two);
            if (two <= one || three - two != two - one || actual != want) return 1;
            if (measured_gpu && h == 1080 && (one != 594477084 || two != 689862684)) return 1;
        }
    }
    return 0;
}
