#include "native/span_fp8.h"
#include "native/conv_kernels.cuh"
#include <cuda_fp8.h>
#include <vector>

namespace ajinative {
namespace {

template<int C, int WARPS, int ROWS, int ACT, int GATE, bool PACKED_IN = false, bool PACKED_OUT = false>
cudaError_t body(const dev::SpanConvParams &p, cudaStream_t stream) {
    constexpr int tile_h = WARPS * ROWS;
    constexpr int input_bytes = (tile_h + 2) * (dev::TW + 2) * dev::fp8_stride(C);
    constexpr int output_bytes = tile_h * dev::TW * (PACKED_OUT ? dev::fp8_stride(C) : 2 * dev::smem_stride(C));
    constexpr int shared_bytes = input_bytes > output_bytes ? input_bytes : output_bytes;
    const dim3 grid((p.W + dev::TW - 1) / dev::TW, (p.H + tile_h - 1) / tile_h);
    dev::k_conv3x3<C, C, WARPS, ROWS, 0, ACT, GATE, 0, true, PACKED_IN, PACKED_OUT>
        <<<grid, WARPS * 32, shared_bytes, stream>>>(p);
    return cudaGetLastError();
}

template<int C, int ACT, int GATE, bool PACKED_IN = false, bool PACKED_OUT = false>
cudaError_t tile(const dev::SpanConvParams &p, int variant, cudaStream_t s) {
    switch (variant) {
    case 0: return body<C,8,2,ACT,GATE,PACKED_IN,PACKED_OUT>(p,s);
    case 1: return body<C,4,4,ACT,GATE,PACKED_IN,PACKED_OUT>(p,s);
    default: return body<C,4,2,ACT,GATE,PACKED_IN,PACKED_OUT>(p,s);
    }
}

template<int C>
cudaError_t dispatch(const Conv3x3 &op,const dev::SpanConvParams &p,int variant,bool packed_input,bool packed_output,cudaStream_t s) {
    if (packed_output) {
        if (packed_input) return tile<C,1,0,true,true>(p,variant,s);
        return tile<C,1,0,false,true>(p,variant,s);
    }
    if (packed_input) {
        if (op.gate == Gate::Tanh1x1) return tile<C,0,2,true,false>(p,variant,s);
        return tile<C,1,0,true,false>(p,variant,s);
    }
    if (op.gate == Gate::Tanh1x1) return tile<C,0,2>(p,variant,s);
    return tile<C,1,0>(p,variant,s);
}

} // namespace

bool span_fp8_available() {
    cudaFuncAttributes attr{};
    return cudaFuncGetAttributes(&attr, dev::k_conv3x3<64,64,4,2,0,1,0,0,true,false,false,dev::SpanConvParams>) == cudaSuccess
        && attr.ptxVersion >= 89;
}

cudaError_t span_fp8_pack(const PlanWeights &w, void **device) {
    // FP16 weights are scaled by 16 before E4M3 conversion. The body removes the scale
    // after its last tensor-core instruction and before the FP16 bias and activation.
    const int c = w.cin, pairs = c / 16;
    std::vector<unsigned char> data(size_t(w.taps) * c * c);
    for (int tap = 0; tap < w.taps; ++tap) {
        size_t cursor = size_t(tap) * c * c;
        for (int k = 0; k < c; k += 32) {
            const int groups = c-k >= 32 ? 2 : 1;
            for (int pair = 0; pair < pairs; ++pair)
                for (int lane = 0; lane < 32; ++lane)
                    for (int half_tile = 0; half_tile < 2; ++half_tile)
                        for (int group = 0; group < groups; ++group)
                            for (int i = 0; i < 4; ++i) {
                                const int input = k + 16*group + 4*(lane%4) + i;
                                const int output = pair*16 + 8*half_tile + lane/4;
                                const float weight = half_bits_to_float(w.w[(size_t(tap) * c + input) * c + output]);
                                data[cursor++] = __nv_fp8_e4m3(weight * 16.f).__x;
                            }
        }
    }
    cudaError_t err = cudaMalloc(device, data.size());
    if (err == cudaSuccess) err = cudaMemcpy(*device, data.data(), data.size(), cudaMemcpyHostToDevice);
    return err;
}

cudaError_t span_fp8_run(const Conv3x3 &op,const Tensors &io,const void *packed,
                         int height,int width,int variant,bool packed_input,bool packed_output,cudaStream_t s) {
    dev::SpanConvParams p{};
    p.src = static_cast<const half*>(io.src);
    p.dst = static_cast<half*>(io.dst);
    p.resid = static_cast<const half*>(io.resid);
    p.w = static_cast<const uint4*>(packed);
    p.bias = static_cast<const half2*>(op.w->bias);
    p.watt = op.att ? static_cast<const uint4*>(op.att->frag) : nullptr;
    p.batt = op.att ? static_cast<const half2*>(op.att->bias) : nullptr;
    p.W = width;
    p.H = height;
    p.reverse_tile_rows = op.reverse_tile_rows;
    return op.cin == 64 ? dispatch<64>(op,p,variant,packed_input,packed_output,s)
                        : dispatch<48>(op,p,variant,packed_input,packed_output,s);
}
} // namespace ajinative
