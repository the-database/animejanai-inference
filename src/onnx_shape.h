/* Internal: read an ONNX model's graph input shape without an onnx dependency. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct AjiOnnxInput {
    int elem_type = 0;           // TensorProto.DataType (1 f32, 10 f16, 16 bf16)
    std::vector<int64_t> dims;   // -1 = symbolic/unknown
};

// Parses the first graph input that is not an initializer. Returns false if
// the buffer is not a parseable ModelProto or has no such input.
bool aji_onnx_input(const char *data, size_t size, AjiOnnxInput *out);

// Frames per inference of a temporal (multi-frame) model, from its input
// shape: [N, T*3, H, W] -> T, [N, T, 3, H, W] -> T. 1 for single-frame
// models or an unrecognized shape.
int aji_onnx_temporal_frames(const AjiOnnxInput &in);
