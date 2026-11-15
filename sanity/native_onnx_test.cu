// aji_native_onnx_test: the native ONNX reader and model builders against malformed input.
// Unit cases (CPU): tensor shapes that overflow, are negative, are too large or disagree with their
// data are refused before any allocation; nodes without outputs or with too few inputs fail the parse.
// With model files as arguments (GPU): each is mutated many times, structurally (tensor dims,
// truncated data, dropped node inputs and outputs, attribute values) and byte-wise (the protobuf
// itself), and every mutant goes through the whole native build (parse, then the image, video, RIFE
// and SPAN builders, as libaji runs them). Passing means no crash: every mutant is either refused
// or builds.
// usage: aji_native_onnx_test [--iters N] [MODEL.onnx ...]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "native/executor.h"
#include "native/guard.h"
#include "native/imgnet.h"
#include "native/lower.h"
#include "native/onnx_graph.h"
#include "native/rife.h"
#include "native/vidnet.h"

using namespace ajinative;

namespace {

int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); fails++; } } while (0)

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
};

Pb tensor(const std::string &name, const std::vector<int64_t> &dims, size_t raw_bytes, int dtype = 1)
{
    Pb t;
    for (int64_t d : dims) t.i(1, d);
    t.i(2, dtype).bytes(8, name);
    return t.bytes(9, std::string(raw_bytes, '\0'));
}

std::string model(const Pb &graph) { return Pb().i(1, 8).msg(7, graph).b; }

bool parse(const std::string &bytes, OnnxGraph *g, std::string *why) { return parse_onnx(bytes.data(), bytes.size(), g, why); }

void unit()
{
    std::vector<float> f;
    std::vector<int64_t> iv;
    OnnxGraph g;
    std::string why;
    // a sane tensor
    {
        Pb gr;
        gr.msg(5, tensor("w", {2, 3}, 24));
        CHECK(parse(model(gr), &g, &why));
        CHECK(g.inits.count("w") && g.inits["w"].floats(&f) && f.size() == 6);
    }
    // dims whose product wraps 64 bits, with a matching-looking small buffer
    {
        Pb gr;
        gr.msg(5, tensor("w", {int64_t(1) << 32, int64_t(1) << 32, 1}, 0));
        CHECK(parse(model(gr), &g, &why));
        CHECK(g.inits["w"].count() == OnnxTensor::kBadCount);
        CHECK(!g.inits["w"].floats(&f) && f.empty());
    }
    // a huge but non-wrapping count: refused before the allocation
    {
        Pb gr;
        gr.msg(5, tensor("w", {int64_t(1) << 40}, 4));
        CHECK(parse(model(gr), &g, &why));
        CHECK(!g.inits["w"].floats(&f) && f.empty());
    }
    // negative dims, and data that disagrees with dims
    {
        Pb gr;
        gr.msg(5, tensor("a", {-1, 4}, 16));
        gr.msg(5, tensor("b", {4, 4}, 60));
        gr.msg(5, tensor("c", {3}, 24, 7));
        CHECK(parse(model(gr), &g, &why));
        CHECK(!g.inits["a"].floats(&f));
        CHECK(!g.inits["b"].floats(&f));
        CHECK(g.inits["c"].ints(&iv) && iv.size() == 3);
        CHECK(!g.inits["c"].floats(&f));
    }
    // nodes: no outputs, and a Conv with one input
    {
        Pb gr, nd;
        nd.bytes(1, "x").bytes(3, "n").bytes(4, "Relu");
        gr.msg(1, nd);
        CHECK(!parse(model(gr), &g, &why));
    }
    {
        Pb gr, nd;
        nd.bytes(1, "x").bytes(2, "y").bytes(3, "n").bytes(4, "Conv");
        gr.msg(1, nd);
        CHECK(!parse(model(gr), &g, &why));
    }
    // a length that runs past the end, and a truncated varint
    {
        std::string m = model(Pb().msg(5, tensor("w", {2}, 8)));
        CHECK(!parse(m.substr(0, m.size() - 3), &g, &why));
        CHECK(!parse(std::string("\x3a\xff\xff\xff\xff\xff\xff\xff\xff\xff", 10), &g, &why));
    }
    std::printf("unit cases: %s\n", fails ? "FAIL" : "OK");
}

// The whole native build of a graph, as libaji runs it (every builder; each refuses what is not its
// model). Returns whether any builder accepted it.
bool build_all(const OnnxGraph &g, cudaStream_t s)
{
    bool any = false;
    std::string why;
    guarded(&why, [&] { auto n = ImgNet::create(g, 64, 64, s, &why); any |= n != nullptr; return true; });
    guarded(&why, [&] { auto n = VidNet::create(g, 64, 64, s, &why); any |= n != nullptr; return true; });
    guarded(&why, [&] { auto n = Rife::create(g, 64, 64, s, &why); any |= n != nullptr; return true; });
    guarded(&why, [&] {
        Plan p;
        if (lower(g, &p, &why)) {
            auto e = Executor::create(p, 64, 64, s, &why);
            any |= e != nullptr;
        }
        return true;
    });
    cudaDeviceSynchronize();
    cudaGetLastError();
    return any;
}

void mutate(OnnxGraph *g, std::mt19937_64 &rng)
{
    auto pick = [&](size_t n) { return n ? (size_t)(rng() % n) : 0; };
    const int64_t odd[] = {0, -1, 1, 2, 3, 7, 1 << 16, int64_t(1) << 31, int64_t(1) << 33, int64_t(1) << 62, -(int64_t(1) << 62)};
    const int kind = (int)(rng() % 6);
    if (kind <= 1 && !g->inits.empty()) {  // a tensor's dims, or its data
        auto it = g->inits.begin();
        std::advance(it, pick(g->inits.size()));
        OnnxTensor &t = it->second;
        if (kind == 0 && !t.dims.empty()) t.dims[pick(t.dims.size())] = odd[pick(sizeof(odd) / sizeof(odd[0]))];
        else if (kind == 0) t.dims.push_back(odd[pick(sizeof(odd) / sizeof(odd[0]))]);
        else t.raw.resize(t.raw.size() / 2);
    } else if (kind == 2 && !g->nodes.empty()) {  // drop an input or output
        OnnxNode &n = g->nodes[pick(g->nodes.size())];
        if (rng() % 2 && !n.in.empty()) n.in.erase(n.in.begin() + pick(n.in.size()));
        else if (!n.out.empty()) n.out.erase(n.out.begin() + pick(n.out.size()));
    } else if (kind == 3 && !g->nodes.empty()) {  // an attribute's value
        OnnxNode &n = g->nodes[pick(g->nodes.size())];
        if (!n.attrs.empty()) {
            OnnxAttr &a = n.attrs[pick(n.attrs.size())];
            a.i = odd[pick(sizeof(odd) / sizeof(odd[0]))];
            if (!a.ints.empty()) a.ints[pick(a.ints.size())] = odd[pick(sizeof(odd) / sizeof(odd[0]))];
            else if (rng() % 2) a.ints.push_back(odd[pick(sizeof(odd) / sizeof(odd[0]))]);
            if (!a.floats.empty()) a.floats.clear();
            a.f = (float)odd[pick(sizeof(odd) / sizeof(odd[0]))];
            if (a.type == 4 && !a.t.dims.empty()) a.t.dims[pick(a.t.dims.size())] = odd[pick(sizeof(odd) / sizeof(odd[0]))];
        }
    } else if (kind == 4 && !g->nodes.empty()) {  // rewire an input to another value
        OnnxNode &n = g->nodes[pick(g->nodes.size())], &m = g->nodes[pick(g->nodes.size())];
        if (!n.in.empty() && !m.out.empty()) n.in[pick(n.in.size())] = m.out[0];
    } else if (!g->nodes.empty()) {  // swap two nodes' ops
        OnnxNode &n = g->nodes[pick(g->nodes.size())], &m = g->nodes[pick(g->nodes.size())];
        std::swap(n.op, m.op);
    }
}

int fuzz(const char *path, int iters, cudaStream_t s)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    OnnxGraph base;
    std::string why;
    if (!parse_onnx(bytes.data(), bytes.size(), &base, &why)) {
        std::printf("%s: does not parse (%s)\n", path, why.c_str());
        return 1;
    }
    const bool ok0 = build_all(base, s);
    std::mt19937_64 rng(12345);
    int built = 0;
    for (int it = 0; it < iters; it++) {  // structural mutants
        OnnxGraph g = base;
        const int k = 1 + (int)(rng() % 3);
        for (int j = 0; j < k; j++) mutate(&g, rng);
        built += build_all(g, s);
    }
    // byte mutants: the protobuf reader (and whatever still parses); flips in the first 64 KB, where
    // the graph's structure is, plus truncations
    int parsed = 0;
    for (int it = 0; it < iters / 4; it++) {
        std::vector<char> b = bytes;
        const size_t span = std::min<size_t>(b.size(), 65536);
        for (int j = 0; j < 1 + (int)(rng() % 8); j++) b[rng() % span] = (char)(rng() & 0xff);
        if (rng() % 4 == 0) b.resize(rng() % b.size());
        OnnxGraph g;
        if (guarded(&why, [&] { return parse_onnx(b.data(), b.size(), &g, &why); })) {
            parsed++;
            build_all(g, s);
        }
    }
    std::printf("%s: original %s; %d structural mutants (%d still built), %d byte mutants (%d still parsed): no crash\n", path,
                ok0 ? "builds" : "REFUSED", iters, built, iters / 4, parsed);
    return ok0 ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv)
{
    unit();
    int iters = 200, rc = fails ? 1 : 0;
    std::vector<const char *> models;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else models.push_back(argv[i]);
    }
    if (!models.empty()) {
        cudaStream_t s = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
            std::printf("no CUDA device: skipping the model mutants\n");
            return rc;
        }
        for (const char *m : models) rc |= fuzz(m, iters, s);
    }
    std::printf("native onnx: %s\n", rc ? "FAIL" : "OK");
    return rc;
}
