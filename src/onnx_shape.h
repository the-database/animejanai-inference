/* Internal: read an ONNX model's graph input shape without an onnx dependency. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct AjiOnnxInput {
    std::string name;            // graph input name (trtexec shape flags)
    int elem_type = 0;           // TensorProto.DataType (1 f32, 10 f16, 16 bf16)
    std::vector<int64_t> dims;   // -1 = symbolic/unknown
    std::vector<std::string> dim_params;  // per dim: symbolic name, or ""
};

// Parses the first graph input that is not an initializer. Returns false if
// the buffer is not a parseable ModelProto or has no such input.
bool aji_onnx_input(const char *data, size_t size, AjiOnnxInput *out);

// Free-dimension overrides that pin `in`'s symbolic dims to `want` (same
// rank; an entry <= 0 leaves that dim alone): the (name, value) pairs for
// ORT's AddFreeDimensionOverrideByName. A name that would need two
// different values (one symbol for both H and W, say) is left free.
std::vector<std::pair<std::string, int64_t>>
aji_onnx_pin_dims(const AjiOnnxInput &in, const std::vector<int64_t> &want);

// Frames per inference of a temporal (multi-frame) model, from its input
// shape: [N, T*3, H, W] -> T, [N, T, 3, H, W] -> T. 1 for single-frame
// models or an unrecognized shape.
int aji_onnx_temporal_frames(const AjiOnnxInput &in);
