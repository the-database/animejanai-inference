#pragma once

#include "native/conv.h"
#include "native/lower.h"

namespace ajinative {

// E4M3 is used by the 3x3 body. An exclusive link between two body convolutions
// can store the converted bytes directly. FP16 readers always get the original feature map.
bool span_fp8_available();
cudaError_t span_fp8_pack(const PlanWeights &w, void **device);
cudaError_t span_fp8_run(const Conv3x3 &op, const Tensors &io, const void *packed,
                         int height, int width, int variant, bool packed_input, bool packed_output, cudaStream_t stream);

} // namespace ajinative
