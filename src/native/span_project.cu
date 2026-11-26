#include "native/conv.h"
#include "native/conv_kernels.cuh"

namespace ajinative {
namespace {
template<int C, bool FP8, bool READ_ONLY_RESIDUAL = false>
cudaError_t project_tile(const dev::GateProjectParams &p, cudaStream_t stream)
{
    constexpr int WARPS = 4, ROWS = 2;
    constexpr int height = WARPS * ROWS;
    constexpr int halo = (height + 2) * (dev::TW + 2) * (FP8 ? dev::fp8_stride(C) : 2 * dev::smem_stride(C));
    constexpr int stage = height * dev::TW * 2 * dev::smem_stride(C);
    constexpr int shared_bytes = halo > stage ? halo : stage;
    static_assert(shared_bytes <= 48 * 1024, "gate/projection tile exceeds the default shared-memory limit");
    const dim3 grid((p.W + dev::TW - 1) / dev::TW, (p.H + height - 1) / height);
    dev::k_conv3x3<C,C,WARPS,ROWS,0,0,2,0,FP8,FP8,false,dev::GateProjectParams,READ_ONLY_RESIDUAL>
        <<<grid,WARPS*32,shared_bytes,stream>>>(p);
    return cudaGetLastError();
}

} // namespace

cudaError_t span_gate_project(const Conv3x3 &op, const Tensors &io, const ConcatConv1x1 &project,
                                const void *const *maps, const void *fp8_weights, int height, int width,
                                cudaStream_t stream)
{
    dev::GateProjectParams p{};
    p.src = static_cast<const half *>(io.src);
    p.dst = static_cast<half *>(io.dst);
    p.resid = static_cast<const half *>(io.resid);
    p.w = static_cast<const uint4 *>(fp8_weights ? fp8_weights : op.w->frag);
    p.bias = static_cast<const half2 *>(op.w->bias);
    p.watt = static_cast<const uint4 *>(op.att->frag);
    p.batt = static_cast<const half2 *>(op.att->bias);
    p.H = height;
    p.W = width;
    p.reverse_tile_rows = op.reverse_tile_rows;
    for (int i = 0; i < 4; ++i) p.maps[i] = static_cast<const half *>(maps[i]);
    p.project_w = static_cast<const uint4 *>(project.w->frag);
    p.project_bias = static_cast<const half2 *>(project.w->bias);
    if (op.cin == 64 && !fp8_weights && op.read_only_residual && p.resid != p.dst)
        return project_tile<64,false,true>(p,stream);
    if (op.cin == 64)
        return fp8_weights ? project_tile<64,true>(p,stream) : project_tile<64,false>(p,stream);
    return fp8_weights ? project_tile<48,true>(p,stream) : project_tile<48,false>(p,stream);
}
} // namespace ajinative
