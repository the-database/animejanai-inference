/*
 * Minimal protobuf walk over an ONNX ModelProto, just deep enough to read
 * the graph input's element type and dims:
 *   ModelProto.graph(7) -> GraphProto.input(11) / initializer(5)
 *   ValueInfoProto.name(1), type(2) -> TypeProto.tensor_type(1)
 *   Tensor.elem_type(1), shape(2) -> TensorShapeProto.dim(1)
 *   Dimension.dim_value(1) | dim_param(2)
 * Every other field is skipped by wire type, so the (large) initializer
 * payloads are never decoded.
 */

#include "onnx_shape.h"

#include <set>
#include <string>

namespace {

struct Reader {
    const uint8_t *p, *end;
    bool ok = true;

    bool more() const { return ok && p < end; }

    uint64_t varint()
    {
        uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p >= end) {
                ok = false;
                return 0;
            }
            uint8_t b = *p++;
            v |= (uint64_t)(b & 0x7f) << shift;
            if (!(b & 0x80))
                return v;
        }
        ok = false;
        return 0;
    }

    // Returns the field number; *wire gets the wire type.
    uint32_t tag(int *wire)
    {
        uint64_t t = varint();
        *wire = (int)(t & 7);
        return (uint32_t)(t >> 3);
    }

    Reader sub()
    {
        uint64_t n = varint();
        if (!ok || n > (uint64_t)(end - p)) {
            ok = false;
            return {end, end};
        }
        Reader r{p, p + n};
        p += n;
        return r;
    }

    void skip(int wire)
    {
        switch (wire) {
        case 0: varint(); break;
        case 1: p += 8; break;
        case 2: sub(); break;
        case 5: p += 4; break;
        default: ok = false; break;
        }
        if (p > end)
            ok = false;
    }
};

std::string read_string(Reader &r)
{
    Reader s = r.sub();
    return std::string((const char *)s.p, (const char *)s.end);
}

bool parse_shape(Reader r, std::vector<int64_t> *dims)
{
    while (r.more()) {
        int wire;
        uint32_t f = r.tag(&wire);
        if (f == 1 && wire == 2) {
            Reader d = r.sub();
            int64_t v = -1;
            while (d.more()) {
                int dw;
                uint32_t df = d.tag(&dw);
                if (df == 1 && dw == 0)
                    v = (int64_t)d.varint();
                else
                    d.skip(dw);
            }
            if (!d.ok)
                return false;
            dims->push_back(v);
        } else {
            r.skip(wire);
        }
    }
    return r.ok;
}

bool parse_value_info(Reader r, std::string *name, AjiOnnxInput *out)
{
    bool have_type = false;
    while (r.more()) {
        int wire;
        uint32_t f = r.tag(&wire);
        if (f == 1 && wire == 2) {
            *name = read_string(r);
        } else if (f == 2 && wire == 2) {
            Reader tp = r.sub();
            while (tp.more()) {
                int tw;
                uint32_t tf = tp.tag(&tw);
                if (tf != 1 || tw != 2) {
                    tp.skip(tw);
                    continue;
                }
                Reader tt = tp.sub();  // TypeProto.Tensor
                while (tt.more()) {
                    int w;
                    uint32_t ff = tt.tag(&w);
                    if (ff == 1 && w == 0) {
                        out->elem_type = (int)tt.varint();
                    } else if (ff == 2 && w == 2) {
                        out->dims.clear();
                        if (!parse_shape(tt.sub(), &out->dims))
                            return false;
                    } else {
                        tt.skip(w);
                    }
                }
                if (!tt.ok)
                    return false;
                have_type = true;
            }
            if (!tp.ok)
                return false;
        } else {
            r.skip(wire);
        }
    }
    return r.ok && have_type;
}

std::string tensor_name(Reader r)
{
    while (r.more()) {
        int wire;
        uint32_t f = r.tag(&wire);
        if (f == 8 && wire == 2)
            return read_string(r);
        r.skip(wire);
    }
    return std::string();
}

} // namespace

bool aji_onnx_input(const char *data, size_t size, AjiOnnxInput *out)
{
    Reader m{(const uint8_t *)data, (const uint8_t *)data + size};
    while (m.more()) {
        int wire;
        uint32_t f = m.tag(&wire);
        if (f != 7 || wire != 2) {
            m.skip(wire);
            continue;
        }
        Reader g = m.sub();
        std::set<std::string> inits;
        std::vector<std::pair<std::string, AjiOnnxInput>> inputs;
        while (g.more()) {
            int gw;
            uint32_t gf = g.tag(&gw);
            if (gf == 5 && gw == 2) {
                inits.insert(tensor_name(g.sub()));
            } else if (gf == 11 && gw == 2) {
                std::string name;
                AjiOnnxInput in;
                if (parse_value_info(g.sub(), &name, &in))
                    inputs.emplace_back(name, in);
            } else {
                g.skip(gw);
            }
        }
        if (!g.ok)
            return false;
        for (auto &kv : inputs) {
            if (!inits.count(kv.first)) {
                *out = kv.second;
                return true;
            }
        }
        return false;
    }
    return false;
}

int aji_onnx_temporal_frames(const AjiOnnxInput &in)
{
    if (in.dims.size() == 5 && in.dims[1] > 1 && in.dims[2] == 3)
        return (int)in.dims[1];
    if (in.dims.size() == 4 && in.dims[1] > 3 && in.dims[1] % 3 == 0)
        return (int)(in.dims[1] / 3);
    return 1;
}
