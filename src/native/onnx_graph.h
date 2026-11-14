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

enum OnnxType { ONNX_FLOAT = 1, ONNX_INT32 = 6, ONNX_INT64 = 7, ONNX_FLOAT16 = 10, ONNX_DOUBLE = 11 };

struct OnnxTensor {
    std::string name;
    int dtype = 0;
    std::vector<int64_t> dims;
    std::vector<uint8_t> raw;  // little-endian element bytes, whatever field they came from

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
};

struct OnnxGraph {
    std::vector<OnnxNode> nodes;               // in file order (topological for valid models)
    std::map<std::string, OnnxTensor> inits;   // initializers by name
    std::vector<std::string> inputs;           // graph inputs that are not initializers
    std::vector<std::string> outputs;
};

bool parse_onnx(const void *data, size_t size, OnnxGraph *g, std::string *err);

}  // namespace ajinative

#endif
