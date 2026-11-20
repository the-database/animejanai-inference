// native/netres.h — device resources and step helpers shared by the native nets built from
// gconv and tfm steps (FDAT, TFDAT) (internal).
#ifndef AJI_NATIVE_NETRES_H
#define AJI_NATIVE_NETRES_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "native/gconv.h"
#include "native/onnx_read.h"
#include "native/tfm.h"

namespace ajinative {

// One gconv step: its weights and the kernel variant picked for its shape.
struct GOp {
    GWeights *w = nullptr;
    int variant = -1;
};

inline GConvArgs ga(const void *src, int ss, int IH, int IW, void *dst, int ds, int H, int W)
{
    GConvArgs a;
    a.src = src;
    a.src_stride = ss;
    a.IH = IH;
    a.IW = IW;
    a.dst = dst;
    a.dst_stride = ds;
    a.H = H;
    a.W = W;
    return a;
}

// Device memory and weights owned by a net; any failed allocation sets failed.
class DevRes {
public:
    DevRes() = default;
    DevRes(const DevRes &) = delete;
    DevRes &operator=(const DevRes &) = delete;
    ~DevRes()
    {
        for (GWeights *w : gw) {
            grelease(*w);
            delete w;
        }
        for (TMat *m : mats) {
            tmat_release(*m);
            delete m;
        }
        for (void *p : bufs) cudaFree(p);
    }
    // Zeroed device memory.
    void *alloc(size_t n)
    {
        void *p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) {
            failed = true;
            return nullptr;
        }
        cudaMemset(p, 0, n);
        bufs.push_back(p);
        bytes += n;
        return p;
    }
    // Weights from the model file are untrusted: every upload states how many values the kernels
    // read (need), and a shorter tensor fails the build (failed) instead of being read past.
    float *dfl(const std::vector<float> &v, size_t need)
    {
        if (v.size() < need || need == 0) {
            failed = true;
            return nullptr;
        }
        float *p = (float *)alloc(v.size() * 4);
        if (p) weight_bytes += v.size() * 4;
        if (p) cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
        return p;
    }
    GWeights *gup(const std::vector<uint16_t> &w, const std::vector<float> &b, int cin, int cin_pad, int cout, int cout_pad)
    {
        if (cin <= 0 || cout <= 0 || w.size() < (size_t)9 * cin * cout || (!b.empty() && b.size() < (size_t)cout)) {
            failed = true;
            return nullptr;
        }
        GWeights *g = new GWeights;
        if (gupload(*g, w.data(), b.empty() ? nullptr : b.data(), cin, cin_pad, cout, cout_pad) != cudaSuccess) {
            delete g;
            failed = true;
            return nullptr;
        }
        gw.push_back(g);
        bytes += (size_t)9 * cin_pad * cout_pad * 2;
        weight_bytes += (size_t)9 * cin_pad * cout_pad * 2;
        return g;
    }
    // A GEMM weight matrix, w [K][N] (row k, column n), bias N floats or null.
    TMat *mat(const std::vector<float> &w, const std::vector<float> *bias, int K, int N)
    {
        if (K <= 0 || N <= 0 || w.size() < (size_t)K * N || (bias && bias->size() < (size_t)N)) {
            failed = true;
            return nullptr;
        }
        TMat *m = new TMat;
        if (tmat_upload(*m, w.data(), bias ? bias->data() : nullptr, K, N) != cudaSuccess) {
            delete m;
            failed = true;
            return nullptr;
        }
        mats.push_back(m);
        bytes += (size_t)m->Kpad * m->Npad * 2;
        weight_bytes += (size_t)m->Kpad * m->Npad * 2;
        return m;
    }
    size_t bytes = 0;         // everything allocated
    size_t weight_bytes = 0;  // of which the uploaded weights (dfl, gup, mat)
    bool failed = false;

private:
    std::vector<void *> bufs;
    std::vector<GWeights *> gw;
    std::vector<TMat *> mats;
};

// Carves 256-byte aligned pieces out of an arena (base null: only counts the bytes).
struct Bump {
    char *base = nullptr;
    size_t off = 0;
    void *take(size_t n)
    {
        off = (off + 255) / 256 * 256;
        void *p = base ? base + off : nullptr;
        off += n;
        return p;
    }
};

// Buffers that share one arena where their lifetimes allow, as an inference engine's memory planner
// does: each buffer names the steps of a run during which it holds data (inclusive step ranges, a step
// that reads one buffer and writes another keeps both live), and two buffers overlap in memory only if
// no range of one meets a range of the other. Placed largest first at the lowest offset that fits
// (greedy by size). AJI_NATIVE_POISON fills the arena with 0xFF (NaN in fp16 and fp32) instead of zeros,
// so a read of anything a step didn't write in this run (another buffer's leftovers) shows in the output.
class Lifetimes {
public:
    using Range = std::pair<int, int>;
    void add(void **slot, size_t bytes, std::vector<Range> live) { bufs.push_back({slot, (bytes + 255) / 256 * 256, std::move(live), 0}); }
    // Lays the buffers out, allocates the arena from r and points each slot into it.
    bool assign(DevRes &r) { return assign([&](size_t n) { return r.alloc(n); }); }
    // The same with any allocator: alloc(bytes) returns device memory or null.
    template <class Alloc>
    bool assign(Alloc alloc)
    {
        std::vector<size_t> order(bufs.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return bufs[a].bytes > bufs[b].bytes; });
        std::vector<size_t> placed;
        size_t total = 0;
        for (size_t i : order) {
            Buf &b = bufs[i];
            std::vector<std::pair<size_t, size_t>> taken;  // [offset, end) of placed buffers live with this one
            for (size_t j : placed)
                if (meets(b, bufs[j])) taken.push_back({bufs[j].off, bufs[j].off + bufs[j].bytes});
            std::sort(taken.begin(), taken.end());
            size_t off = 0;
            for (auto &t : taken) {
                if (off + b.bytes <= t.first) break;
                off = std::max(off, t.second);
            }
            b.off = off;
            total = std::max(total, off + b.bytes);
            placed.push_back(i);
        }
        static const bool poison = std::getenv("AJI_NATIVE_POISON") != nullptr;
        char *base = (char *)alloc(total);
        if (!base) return false;
        if (poison) cudaMemset(base, 0xFF, total);
        for (Buf &b : bufs) *b.slot = base + b.off;
        return true;
    }

private:
    struct Buf {
        void **slot;
        size_t bytes;
        std::vector<Range> live;
        size_t off;
    };
    static bool meets(const Buf &a, const Buf &b)
    {
        for (const Range &x : a.live)
            for (const Range &y : b.live)
                if (x.first <= y.second && y.first <= x.second) return true;
        return false;
    }
    std::vector<Buf> bufs;
};

// Checks the ops are supported and sets each one's kernel variant (gvariant_for; no timing).
inline bool gprepare(std::vector<std::pair<GOp *, GConvArgs>> &ops, cudaStream_t s, std::string *why)
{
    (void)s;
    for (auto &o : ops) {
        std::string w2;
        if (!o.first->w) return onnxread::refuse(why, "weight upload failed");
        if (!gsupported(*o.first->w, o.second, &w2)) return onnxread::refuse(why, "kernel: " + w2);
        o.first->variant = gvariant_for(*o.first->w, o.second);
        // AJI_NATIVE_GVARIANT=n: every gconv step on variant n; with AJI_NATIVE_GVARIANT_OP=i only step
        // i (the others on variant 2); AJI_NATIVE_GDEBUG lists the steps (for testing the variants)
        static const char *fv = std::getenv("AJI_NATIVE_GVARIANT"), *fo = std::getenv("AJI_NATIVE_GVARIANT_OP");
        const int idx = (int)(&o - &ops[0]);
        if (fv && *fv && fo && *fo) o.first->variant = idx == std::atoi(fo) ? std::atoi(fv) : 2;
        if (std::getenv("AJI_NATIVE_GDEBUG"))
            std::fprintf(stderr, "gconv %d: %d->%d %dx%d R%d sy%d stride%d nchw%d up2%d r1 %d r1f %d outf %d act %d acc32 %d: variant %d (default %d) ss %d ds %d in %dx%d ws %p %zu src %p dst %p\n", idx,
                         o.first->w->cin_pad, o.first->w->cout_pad, o.second.W, o.second.H, o.second.R, o.second.sy, o.second.stride,
                         (int)o.second.nchw, (int)o.second.up2, o.second.r1 != nullptr, o.second.r1f != nullptr, (int)o.second.out_f32,
                         (int)o.second.act, (int)o.second.acc32, o.first->variant, gvariant_for(*o.first->w, o.second), o.second.src_stride, o.second.dst_stride, o.second.IW, o.second.IH, (void *)o.second.ws, o.second.ws_floats, o.second.src, o.second.dst);
    }
    return true;
}

// Split-K workspace floats the ops need (the max over them).
inline size_t gworkspace_max(const std::vector<std::pair<GOp *, GConvArgs>> &ops)
{
    size_t n = 0;
    for (auto &o : ops)
        if (o.first->w) n = std::max(n, gworkspace(*o.first->w, o.second));
    return n;
}

// AJI_NATIVE_PROFILE: GPU time per step kind, from events between steps (printed to stderr).
enum ProfKind {
    P_IO, P_CONV, P_LN, P_QKV, P_DW, P_WIN, P_GATE, P_PROJ, P_CSTATS, P_CFOLD, P_CPROJ, P_FC1, P_SMIX, P_FC2,
    P_ENC, P_FLOW, P_WARP, P_ALIGN, P_FUSE, P_UP, P_MLP, P_NKINDS
};
class StepProf {
public:
    explicit StepProf(cudaStream_t s_, bool allow = true) : s(s_)
    {
        static const bool env = std::getenv("AJI_NATIVE_PROFILE") != nullptr;
        on = env && allow;
        mark(P_IO);
    }
    void mark(int k)
    {
        if (!on) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s);
        evs.push_back({k, e});
    }
    void report(const char *what)
    {
        if (!on) return;
        cudaStreamSynchronize(s);
        static const char *const names[P_NKINDS] = {"io",   "conv3x3", "layernorm", "qkv",  "dwconv", "winattn", "gate",
                                                    "proj", "chanstats", "chanfold", "chanproj", "fc1", "smix", "fc2",
                                                    "encoder", "flow", "warp", "align", "fusion", "upsample", "mlp"};
        double t[P_NKINDS] = {}, total = 0;
        for (size_t i = 1; i < evs.size(); i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, evs[i - 1].second, evs[i].second);
            t[evs[i].first] += ms;
            total += ms;
        }
        for (auto &e : evs) cudaEventDestroy(e.second);
        evs.clear();
        fprintf(stderr, "%s: %.3f ms:", what, total);
        for (int k = 0; k < P_NKINDS; k++)
            if (t[k] > 0) fprintf(stderr, " %s %.3f", names[k], t[k]);
        fprintf(stderr, "\n");
    }

private:
    cudaStream_t s;
    bool on = false;
    std::vector<std::pair<int, cudaEvent_t>> evs;
};

}  // namespace ajinative

#endif
