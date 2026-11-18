/*
 * native/onnx_graph.h — an ONNX ModelProto decoded just far enough for the native lowering:
 * nodes (op, inputs, outputs, attributes), initializers (with their data), the graph's inputs
 * and outputs. No onnx/protobuf dependency; external data is not supported. Host only.
 */
#ifndef AJI_NATIVE_ONNX_GRAPH_H
#define AJI_NATIVE_ONNX_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ajinative {

enum OnnxType { ONNX_FLOAT = 1, ONNX_INT32 = 6, ONNX_INT64 = 7, ONNX_FLOAT16 = 10, ONNX_DOUBLE = 11, ONNX_BFLOAT16 = 16 };

struct OnnxTensor {
    std::string name;
    int dtype = 0;
    std::vector<int64_t> dims;
    std::vector<uint8_t> raw;  // little-endian element bytes, whatever field they came from

    // Elements per dims; kBadCount when a dim is negative or the product exceeds kMaxElems (the
    // file is untrusted: callers index the data by dims, so the two must agree before any use).
    static constexpr size_t kMaxElems = size_t(1) << 28, kBadCount = ~size_t(0);
    size_t count() const;
    // Element values as floats (FLOAT, FLOAT16, DOUBLE) or integers (INT32, INT64);
    // false for other types or a size mismatch.
    bool floats(std::vector<float> *out) const;
    bool ints(std::vector<int64_t> *out) const;
};

struct OnnxAttr {
    std::string name;
    int type = 0;  // AttributeProto.AttributeType: 1 FLOAT, 2 INT, 3 STRING, 4 TENSOR, 6 FLOATS, 7 INTS
    float f = 0.f;
    int64_t i = 0;
    std::string s;
    OnnxTensor t;
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

struct OnnxNode {
    std::string op, name, domain;
    std::vector<std::string> in, out;  // empty names = omitted optional inputs
    std::vector<OnnxAttr> attrs;
    const OnnxAttr *attr(const char *name) const;
    int64_t attr_i(const char *name, int64_t dflt) const;
    std::string attr_s(const char *name, const char *dflt) const;
    float attr_f(const char *name, float dflt) const
    {
        const OnnxAttr *a = attr(name);
        return a ? a->f : dflt;
    }
};

// A Cast between floating-point types (float, float16, double, bfloat16): on the data path only a
// precision change, which the kernels' own precision stands in for. A Cast to an integer or bool
// type changes the values.
inline bool float_cast(const OnnxNode &n)
{
    const int64_t to = n.attr_i("to", 0);
    return n.op == "Cast" && (to == 1 || to == 10 || to == 11 || to == 16);
}

// A nearest Resize by the integer factor r that is pixel replication (output x reads input x / r),
// the only kind the native kernels fold. For x = r i + j, ONNX samples i + (j + 0.5) / r - 0.5
// (half_pixel, pytorch_half_pixel), which rounding to nearest takes to i and floor / ceil do not, or
// i + j / r (asymmetric: the torch exporters' nearest), which floor takes to i and round_prefer_floor
// only up to r = 2.
inline bool nearest_replicates(const OnnxNode &rz, int r)
{
    if (rz.op != "Resize" || rz.attr_s("mode", "nearest") != "nearest" || r < 1) return false;
    const std::string ct = rz.attr_s("coordinate_transformation_mode", "half_pixel");
    const std::string nm = rz.attr_s("nearest_mode", "round_prefer_floor");
    if (ct == "half_pixel" || ct == "pytorch_half_pixel") return nm == "round_prefer_floor" || nm == "round_prefer_ceil";
    if (ct == "asymmetric") return nm == "floor" || (nm == "round_prefer_floor" && r <= 2);
    return false;
}

struct OnnxGraph {
    std::vector<OnnxNode> nodes;               // in file order (topological for valid models)
    std::map<std::string, OnnxTensor> inits;   // initializers by name
    std::vector<std::string> inputs;           // graph inputs that are not initializers
    std::vector<std::string> outputs;
};

bool parse_onnx(const void *data, size_t size, OnnxGraph *g, std::string *err);

}  // namespace ajinative

#endif
