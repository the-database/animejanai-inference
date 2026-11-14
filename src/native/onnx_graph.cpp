/*
 * native/onnx_graph.cpp — see onnx_graph.h. The protobuf fields read:
 *   ModelProto.graph(7)
 *   GraphProto: node(1) initializer(5) input(11) output(12)    ValueInfoProto.name(1)
 *   NodeProto: input(1) output(2) name(3) op_type(4) attribute(5) domain(7)
 *   AttributeProto: name(1) f(2) i(3) s(4) t(5) floats(7) ints(8) type(20)
 *   TensorProto: dims(1) data_type(2) float_data(4) int32_data(5) int64_data(7) name(8)
 *                raw_data(9) double_data(10) data_location(14)
 * Repeated scalars are accepted packed or unpacked. Everything else is skipped by wire type.
 */
#include "native/onnx_graph.h"
#include "native/pack.h"

#include <cstring>

namespace ajinative {
namespace {

struct Reader {
    const uint8_t *p, *end;
    bool ok = true;
    bool more() const { return ok && p < end; }
    uint64_t varint()
    {
        uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p >= end) { ok = false; return 0; }
            const uint8_t b = *p++;
            v |= (uint64_t)(b & 0x7f) << shift;
            if (!(b & 0x80)) return v;
        }
        ok = false;
        return 0;
    }
    uint32_t tag(int *wire)
    {
        const uint64_t t = varint();
        *wire = (int)(t & 7);
        return (uint32_t)(t >> 3);
    }
    Reader sub()
    {
        const uint64_t n = varint();
        if (!ok || n > (uint64_t)(end - p)) { ok = false; return {end, end}; }
        Reader r{p, p + n};
        p += n;
        return r;
    }
    uint32_t fixed32()
    {
        if (end - p < 4) { ok = false; return 0; }
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    uint64_t fixed64()
    {
        if (end - p < 8) { ok = false; return 0; }
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    void skip(int wire)
    {
        switch (wire) {
        case 0: varint(); break;
        case 1: fixed64(); break;
        case 2: sub(); break;
        case 5: fixed32(); break;
        default: ok = false; break;
        }
    }
    std::string str()
    {
        Reader s = sub();
        return std::string((const char *)s.p, (const char *)s.end);
    }
};

// repeated varint field, packed (wire 2) or not (wire 0)
void read_varints(Reader &r, int wire, std::vector<int64_t> *out)
{
    if (wire == 2) {
        Reader s = r.sub();
        while (s.more()) out->push_back((int64_t)s.varint());
        if (!s.ok) r.ok = false;
    } else if (wire == 0) {
        out->push_back((int64_t)r.varint());
    } else {
        r.skip(wire);
    }
}

void append_bytes(std::vector<uint8_t> *raw, const void *v, size_t n)
{
    const uint8_t *b = static_cast<const uint8_t *>(v);
    raw->insert(raw->end(), b, b + n);
}

bool parse_tensor(Reader r, OnnxTensor *t)
{
    std::vector<int64_t> i32, i64;
    std::vector<uint8_t> f32, f64;
    bool external = false;
    while (r.more()) {
        int wire;
        const uint32_t f = r.tag(&wire);
        switch (f) {
        case 1: read_varints(r, wire, &t->dims); break;
        case 2: t->dtype = (int)r.varint(); break;
        case 4:
            if (wire == 2) { Reader s = r.sub(); append_bytes(&f32, s.p, (size_t)(s.end - s.p)); }
            else if (wire == 5) { const uint32_t v = r.fixed32(); append_bytes(&f32, &v, 4); }
            else r.skip(wire);
            break;
        case 5: read_varints(r, wire, &i32); break;
        case 7: read_varints(r, wire, &i64); break;
        case 8: t->name = r.str(); break;
        case 9: { Reader s = r.sub(); t->raw.assign(s.p, s.end); break; }
        case 10:
            if (wire == 2) { Reader s = r.sub(); append_bytes(&f64, s.p, (size_t)(s.end - s.p)); }
            else if (wire == 1) { const uint64_t v = r.fixed64(); append_bytes(&f64, &v, 8); }
            else r.skip(wire);
            break;
        case 14: external = r.varint() == 1; break;
        default: r.skip(wire); break;
        }
    }
    if (!r.ok || external) return false;
    if (!t->raw.empty()) return true;
    // typed fields -> little-endian raw bytes of the declared type
    switch (t->dtype) {
    case ONNX_FLOAT: t->raw = f32; break;
    case ONNX_DOUBLE: t->raw = f64; break;
    case ONNX_INT64:
        for (int64_t v : i64) append_bytes(&t->raw, &v, 8);
        break;
    case ONNX_INT32:
        for (int64_t v : i32) { const int32_t x = (int32_t)v; append_bytes(&t->raw, &x, 4); }
        break;
    case ONNX_FLOAT16:  // stored in int32_data, one element per entry
        for (int64_t v : i32) { const uint16_t x = (uint16_t)v; append_bytes(&t->raw, &x, 2); }
        break;
    default: break;
    }
    return true;
}

bool parse_attr(Reader r, OnnxAttr *a)
{
    while (r.more()) {
        int wire;
        const uint32_t f = r.tag(&wire);
        switch (f) {
        case 1: a->name = r.str(); break;
        case 2: { const uint32_t v = r.fixed32(); std::memcpy(&a->f, &v, 4); break; }
        case 3: a->i = (int64_t)r.varint(); break;
        case 4: a->s = r.str(); break;
        case 5: if (!parse_tensor(r.sub(), &a->t)) return false; break;
        case 7:
            if (wire == 2) {
                Reader s = r.sub();
                while (s.more()) { const uint32_t v = s.fixed32(); float x; std::memcpy(&x, &v, 4); a->floats.push_back(x); }
            } else if (wire == 5) {
                const uint32_t v = r.fixed32(); float x; std::memcpy(&x, &v, 4); a->floats.push_back(x);
            } else r.skip(wire);
            break;
        case 8: read_varints(r, wire, &a->ints); break;
        case 20: a->type = (int)r.varint(); break;
        default: r.skip(wire); break;
        }
    }
    return r.ok;
}

bool parse_node(Reader r, OnnxNode *n)
{
    while (r.more()) {
        int wire;
        const uint32_t f = r.tag(&wire);
        switch (f) {
        case 1: n->in.push_back(r.str()); break;
        case 2: n->out.push_back(r.str()); break;
        case 3: n->name = r.str(); break;
        case 4: n->op = r.str(); break;
        case 5: { n->attrs.emplace_back(); if (!parse_attr(r.sub(), &n->attrs.back())) return false; break; }
        case 7: n->domain = r.str(); break;
        default: r.skip(wire); break;
        }
    }
    return r.ok;
}

std::string value_info_name(Reader r)
{
    std::string name;
    while (r.more()) {
        int wire;
        if (r.tag(&wire) == 1 && wire == 2) name = r.str();
        else r.skip(wire);
    }
    return name;
}

bool parse_graph(Reader r, OnnxGraph *g)
{
    std::vector<std::string> inputs;
    while (r.more()) {
        int wire;
        const uint32_t f = r.tag(&wire);
        if (f == 1 && wire == 2) {
            g->nodes.emplace_back();
            if (!parse_node(r.sub(), &g->nodes.back())) return false;
        } else if (f == 5 && wire == 2) {
            OnnxTensor t;
            if (!parse_tensor(r.sub(), &t)) return false;
            std::string name = t.name;
            g->inits[name] = std::move(t);
        } else if (f == 11 && wire == 2) {
            inputs.push_back(value_info_name(r.sub()));
        } else if (f == 12 && wire == 2) {
            g->outputs.push_back(value_info_name(r.sub()));
        } else {
            r.skip(wire);
        }
    }
    for (std::string &s : inputs)
        if (!g->inits.count(s)) g->inputs.push_back(std::move(s));
    return r.ok;
}

}  // namespace

size_t OnnxTensor::count() const
{
    size_t n = 1;
    for (int64_t d : dims) n *= (size_t)(d > 0 ? d : 0);
    return n;
}

bool OnnxTensor::floats(std::vector<float> *out) const
{
    const size_t n = count();
    out->resize(n);
    if (dtype == ONNX_FLOAT && raw.size() == n * 4) {
        std::memcpy(out->data(), raw.data(), n * 4);
    } else if (dtype == ONNX_FLOAT16 && raw.size() == n * 2) {
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            std::memcpy(&h, &raw[i * 2], 2);
            (*out)[i] = half_bits_to_float(h);
        }
    } else if (dtype == ONNX_DOUBLE && raw.size() == n * 8) {
        for (size_t i = 0; i < n; i++) {
            double d;
            std::memcpy(&d, &raw[i * 8], 8);
            (*out)[i] = (float)d;
        }
    } else {
        return false;
    }
    return true;
}

bool OnnxTensor::ints(std::vector<int64_t> *out) const
{
    const size_t n = count();
    out->resize(n);
    if (dtype == ONNX_INT64 && raw.size() == n * 8) {
        std::memcpy(out->data(), raw.data(), n * 8);
    } else if (dtype == ONNX_INT32 && raw.size() == n * 4) {
        for (size_t i = 0; i < n; i++) {
            int32_t v;
            std::memcpy(&v, &raw[i * 4], 4);
            (*out)[i] = v;
        }
    } else {
        return false;
    }
    return true;
}

const OnnxAttr *OnnxNode::attr(const char *n) const
{
    for (const OnnxAttr &a : attrs)
        if (a.name == n) return &a;
    return nullptr;
}

int64_t OnnxNode::attr_i(const char *n, int64_t dflt) const
{
    const OnnxAttr *a = attr(n);
    return a ? a->i : dflt;
}

std::string OnnxNode::attr_s(const char *n, const char *dflt) const
{
    const OnnxAttr *a = attr(n);
    return a ? a->s : std::string(dflt);
}

bool parse_onnx(const void *data, size_t size, OnnxGraph *g, std::string *err)
{
    *g = OnnxGraph{};
    Reader r{static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + size};
    bool have_graph = false;
    while (r.more()) {
        int wire;
        const uint32_t f = r.tag(&wire);
        if (f == 7 && wire == 2) {
            if (!parse_graph(r.sub(), g)) {
                if (err) *err = "unreadable ONNX graph (or external data)";
                return false;
            }
            have_graph = true;
        } else {
            r.skip(wire);
        }
    }
    if (!r.ok || !have_graph) {
        if (err) *err = "not an ONNX model";
        return false;
    }
    return true;
}

}  // namespace ajinative
