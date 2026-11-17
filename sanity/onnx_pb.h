// sanity/onnx_pb.h — a minimal protobuf encoder for the small ONNX graphs the host-side tests write
// (native_lower_test, native_forms_test).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace onnx_pb {

struct Pb {
    std::string b;
    Pb &varint(uint64_t v)
    {
        while (v >= 0x80) { b.push_back((char)(v | 0x80)); v >>= 7; }
        b.push_back((char)v);
        return *this;
    }
    Pb &tag(int f, int wire) { return varint((uint64_t)f << 3 | wire); }
    Pb &bytes(int f, const std::string &s) { tag(f, 2).varint(s.size()); b += s; return *this; }
    Pb &msg(int f, const Pb &m) { return bytes(f, m.b); }
    Pb &i(int f, int64_t v) { tag(f, 0).varint((uint64_t)v); return *this; }
    Pb &f32(int f, float v) { tag(f, 5); char c[4]; std::memcpy(c, &v, 4); b.append(c, 4); return *this; }
};

inline Pb tensor(const std::string &name, std::vector<int64_t> dims, int dtype = 10, float fill = 0.f)
{
    Pb t;
    size_t n = 1;
    for (int64_t d : dims) { t.i(1, d); n *= (size_t)d; }
    t.i(2, dtype).bytes(8, name);
    std::string raw;
    if (dtype == 10) raw.assign(n * 2, '\0');
    if (dtype == 1) for (size_t k = 0; k < n; k++) { char c[4]; std::memcpy(c, &fill, 4); raw.append(c, 4); }
    if (dtype == 7) raw.assign(n * 8, '\0');
    return t.bytes(9, raw);
}
inline Pb attr_i(const char *name, int64_t v) { return Pb().bytes(1, name).i(3, v).i(20, 2); }
inline Pb attr_s(const char *name, const char *v) { return Pb().bytes(1, name).bytes(4, v).i(20, 3); }
inline Pb attr_ints(const char *name, std::vector<int64_t> v)
{
    Pb a;
    a.bytes(1, name);
    for (int64_t x : v) a.i(8, x);
    return a.i(20, 7);
}
inline Pb attr_f(const char *name, float v) { return Pb().bytes(1, name).f32(2, v).i(20, 1); }

struct G {
    Pb graph;
    int n = 0;
    std::string fresh(const char *p) { return std::string(p) + std::to_string(n++); }
    void node(const char *op, std::vector<std::string> in, std::vector<std::string> out, std::vector<Pb> attrs = {})
    {
        Pb nd;
        for (auto &s : in) nd.bytes(1, s);
        for (auto &s : out) nd.bytes(2, s);
        nd.bytes(3, fresh(op)).bytes(4, op);
        for (auto &a : attrs) nd.msg(5, a);
        graph.msg(1, nd);
    }
    void init(const Pb &t) { graph.msg(5, t); }
    // conv with fresh weights; k = 3 or 1
    std::string conv(const std::string &x, int ci, int co, int k)
    {
        const std::string w = fresh("w"), b = fresh("b"), y = fresh("conv");
        init(tensor(w, {co, ci, k, k}));
        init(tensor(b, {co}));
        node("Conv", {x, w, b}, {y}, {attr_ints("kernel_shape", {k, k}), attr_ints("pads", {k / 2, k / 2, k / 2, k / 2}),
                                      attr_ints("strides", {1, 1})});
        return y;
    }
    std::string op1(const char *op, const std::string &x, std::vector<Pb> attrs = {})
    {
        const std::string y = fresh(op);
        node(op, {x}, {y}, attrs);
        return y;
    }
    std::string op2(const char *op, const std::string &a, const std::string &b)
    {
        const std::string y = fresh(op);
        node(op, {a, b}, {y});
        return y;
    }
    std::string bytes(const std::string &in, const std::string &out)
    {
        graph.msg(11, Pb().bytes(1, in));
        graph.msg(12, Pb().bytes(1, out));
        return Pb().msg(7, graph).b;
    }
};

// an fp32 tensor with the given values
inline Pb tensor_v(const std::string &name, std::vector<int64_t> dims, const std::vector<float> &v)
{
    Pb t;
    for (int64_t d : dims) t.i(1, d);
    t.i(2, 1).bytes(8, name);
    std::string raw(v.size() * 4, '\0');
    if (!v.empty()) std::memcpy(&raw[0], v.data(), v.size() * 4);
    return t.bytes(9, raw);
}
// an int64 tensor with the given values
inline Pb tensor_i(const std::string &name, std::vector<int64_t> dims, const std::vector<int64_t> &v)
{
    Pb t;
    for (int64_t d : dims) t.i(1, d);
    t.i(2, 7).bytes(8, name);
    std::string raw(v.size() * 8, '\0');
    if (!v.empty()) std::memcpy(&raw[0], v.data(), v.size() * 8);
    return t.bytes(9, raw);
}

}  // namespace onnx_pb
