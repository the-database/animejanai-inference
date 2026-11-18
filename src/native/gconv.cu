// native/gconv.cu — see gconv.h. Instantiates k_gconv for a fixed set of shapes.
#include "native/gconv.h"
#include "native/gconv.cuh"
#include "native/pack.h"

#include <algorithm>
#include <cstring>

namespace ajinative {
namespace {

using GFn = cudaError_t (*)(const gdev::GParams &, int, cudaStream_t);
constexpr int kGVariants = 3;
constexpr int kWarps[kGVariants] = {8, 4, 2};

template <int KCH, int COUTP, int S, int IN, int OUT, int WARPS, int ACC = 0, int SUMS = 0>
cudaError_t run_g(const gdev::GParams &p, int zdim, cudaStream_t s)
{
    constexpr int smem = gdev::gconv_smem_bytes<KCH, COUTP, S, WARPS>();
    auto k = gdev::k_gconv<KCH, COUTP, S, IN, OUT, WARPS, ACC, SUMS>;
    if constexpr (smem > 48 * 1024) {
        static bool set = false;
        if (!set) {
            cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess) return e;
            set = true;
        }
    }
    const dim3 grid((p.W + gdev::TW - 1) / gdev::TW, (p.H + WARPS - 1) / WARPS, zdim);
    k<<<grid, WARPS * 32, smem, s>>>(p);
    return cudaGetLastError();
}

struct GKey {
    int kch, coutp, s, in, out, acc, sums = 0;
    bool operator==(const GKey &o) const
    {
        return kch == o.kch && coutp == o.coutp && s == o.s && in == o.in && out == o.out && acc == o.acc && sums == o.sums;
    }
};
struct GEntry {
    GKey key;
    GFn fn[kGVariants];
};

#define AJI_G(K, C, S, IN, OUT) {{K, C, S, IN, OUT, 0}, {&run_g<K, C, S, IN, OUT, 8>, &run_g<K, C, S, IN, OUT, 4>, &run_g<K, C, S, IN, OUT, 2>}}
#define AJI_G32(K, C) {{K, C, 1, 0, 0, 1}, {&run_g<K, C, 1, 0, 0, 8, 1>, &run_g<K, C, 1, 0, 0, 4, 1>, &run_g<K, C, 1, 0, 0, 2, 1>}}
#define AJI_G32_COUT(K) AJI_G32(K, 16), AJI_G32(K, 32), AJI_G32(K, 48), AJI_G32(K, 64)
// with the channel sums (HAT's CAB: 64 -> 192 channels)
#define AJI_GS(K, C) {{K, C, 1, 0, 0, 0, 1}, {&run_g<K, C, 1, 0, 0, 8, 0, 1>, &run_g<K, C, 1, 0, 0, 4, 0, 1>, &run_g<K, C, 1, 0, 0, 2, 0, 1>}}
#define AJI_G_COUT(K, S, IN, OUT) AJI_G(K, 16, S, IN, OUT), AJI_G(K, 32, S, IN, OUT), AJI_G(K, 48, S, IN, OUT), AJI_G(K, 64, S, IN, OUT)

const GEntry kG[] = {
    // stride 1, NHWC (residuals are a runtime option)
    AJI_G_COUT(16, 1, 0, 0), AJI_G_COUT(32, 1, 0, 0), AJI_G_COUT(64, 1, 0, 0),
    // stride 1, phase scatter (transposed convs)
    AJI_G_COUT(16, 1, 0, 1), AJI_G_COUT(32, 1, 0, 1), AJI_G_COUT(64, 1, 0, 1),
    // stride 2, NHWC and NCHW planes
    AJI_G_COUT(16, 2, 0, 0), AJI_G_COUT(32, 2, 0, 0), AJI_G_COUT(16, 2, 1, 0),
    // stride 1, NHWC, fp32 accumulation
    AJI_G32_COUT(16), AJI_G32_COUT(32), AJI_G32_COUT(64),
    // stride 1, NHWC, with the channel sums
    AJI_GS(64, 64),
};

bool gfail(std::string *why, const char *m)
{
    if (why) *why = m;
    return false;
}

int kch_for(const GWeights &w, const GConvArgs &a)
{
    if (a.nchw) return 16;
    const int maxk = a.stride == 2 ? 32 : 64;
    for (int k = maxk; k >= 16; k /= 2)
        if (w.cin_pad % k == 0) return k;
    return 0;
}

const GEntry *gfind(const GWeights &w, const GConvArgs &a, std::string *why)
{
    if (!w.frag) return gfail(why, "no weights"), nullptr;
    if (a.stride != 1 && a.stride != 2) return gfail(why, "stride must be 1 or 2"), nullptr;
    if (a.nchw && (a.stride != 2 || w.cin != 3)) return gfail(why, "NCHW input: 3 channels, stride 2"), nullptr;
    if (a.up2 && (a.nchw || a.stride != 1 || a.IH % 2 || a.IW % 2)) return gfail(why, "upsampled input: NHWC, stride 1, even size"), nullptr;
    const bool res = a.r1 || a.r2;
    if (res && (a.stride != 1 || a.R)) return gfail(why, "residuals need stride 1 and NHWC output"), nullptr;
    if (a.R && (a.stride != 1 || (a.R != 2 && a.R != 4) || (w.cout_pad / a.R) % 8 || (w.cout_pad / a.R) % w.coutp && w.coutp % (w.cout_pad / a.R)))
        return gfail(why, "phase scatter: stride 1, R 2 or 4, 8-channel groups"), nullptr;
    if (!a.nchw && (a.src_stride % 8 || a.src_stride < w.cin_pad)) return gfail(why, "input pixel stride"), nullptr;
    if (a.dst_stride % 8 || a.dst_stride < (a.R ? w.cout_pad / a.R : w.coutp)) return gfail(why, "output pixel stride"), nullptr;
    if ((a.r1 && a.r1_stride % 2) || (a.r2 && a.r2_stride % 2)) return gfail(why, "residual stride"), nullptr;
    if (a.acc32 && (a.stride != 1 || a.nchw || a.R)) return gfail(why, "fp32 accumulation: stride 1, NHWC in and out"), nullptr;
    if (a.out_f32 && (!a.acc32 || a.r1 || a.r2 || a.act == GAct::Gelu)) return gfail(why, "fp32 output: fp32 accumulation, r1f only, no GELU"), nullptr;
    if (a.sums && (a.out_f32 || a.R)) return gfail(why, "channel sums: fp16 NHWC output"), nullptr;
    const GKey k{kch_for(w, a), w.coutp, a.stride, a.nchw ? 1 : 0, a.R ? 1 : 0, a.acc32 ? 1 : 0, a.sums ? 1 : 0};
    if (!k.kch) return gfail(why, "input channels"), nullptr;
    for (const GEntry &e : kG)
        if (e.key == k) return &e;
    gfail(why, "no gconv kernel for this shape");
    return nullptr;
}

int sm_count()
{
    static int n = 0;
    if (!n) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0) n = 80;
    }
    return n;
}

// K splits for a: enough CTAs to cover the SMs about twice, at most one split per (tap, K chunk).
int ksplit_for(const GWeights &w, const GConvArgs &a, int warps)
{
    const int kch = kch_for(w, a);
    if (!kch) return 1;
    const long ctas = (long)((a.W + gdev::TW - 1) / gdev::TW) * ((a.H + warps - 1) / warps) * (w.cout_pad / w.coutp);
    const long want = 2L * sm_count();
    if (ctas >= want) return 1;
    const int units = (a.nchw ? 1 : w.cin_pad / kch) * 9;
    int k = (int)((want + ctas - 1) / ctas);
    return k < units ? k : units;
}

gdev::GParams params(const GWeights &w, const GConvArgs &a, int ksplit)
{
    gdev::GParams p{};
    const int kch = kch_for(w, a);
    p.src = (const half *)a.src;
    p.sstride = a.src_stride;
    p.IH = a.IH;
    p.IW = a.IW;
    p.nchunks = a.nchw ? 1 : w.cin_pad / kch;
    p.kct = w.cin_pad / 16;
    p.dst = (half *)a.dst;
    p.dstride = a.dst_stride;
    p.H = a.H;
    p.W = a.W;
    p.w = (const uint4 *)w.frag;
    p.wchunk = (size_t)9 * (w.cin_pad / 16) * (w.coutp / 16) * 32;
    p.bias = (const half2 *)w.bias;
    p.r1 = (const half *)a.r1;
    p.r2 = (const half *)a.r2;
    p.r1s = a.r1_stride;
    p.r2s = a.r2_stride;
    p.c1 = __float2half2_rn(a.c1);
    p.c2 = __float2half2_rn(a.c2);
    p.slope = __float2half2_rn(a.slope);
    p.act = (int)a.act;
    p.R = a.R;
    p.sy = a.sy;
    p.coutp_all = w.cout_pad;
    p.ksplit = ksplit;
    p.ws = a.ws;
    p.up2 = a.up2 ? 1 : 0;
    p.outf = a.out_f32 ? 1 : 0;
    p.dstf = (float *)a.dst;
    p.r1f = a.r1f;
    p.r1fs = a.r1f_stride;
    p.sums = a.sums;
    p.slopef = a.slope;
    return p;
}

}  // namespace

cudaError_t gupload(GWeights &out, const uint16_t *w, const float *bias, int cin, int cin_pad, int cout, int cout_pad)
{
    out = GWeights{};
    if (cin_pad % 16 || cout_pad % 16 || cin > cin_pad || cout > cout_pad) return cudaErrorInvalidValue;
    int coutp = 16;
    for (int c : {64, 48, 32})
        if (cout_pad % c == 0) { coutp = c; break; }
    const int chunks = cout_pad / coutp;
    std::vector<uint32_t> all;
    std::vector<uint16_t> part((size_t)9 * cin * coutp);
    for (int ch = 0; ch < chunks; ch++) {
        std::fill(part.begin(), part.end(), (uint16_t)0);
        const int c0 = ch * coutp, cn = std::min(coutp, cout - c0);
        for (int t = 0; t < 9; t++)
            for (int i = 0; i < cin; i++)
                for (int o = 0; o < cn; o++) part[((size_t)t * cin + i) * coutp + o] = w[((size_t)t * cin + i) * cout + c0 + o];
        std::vector<uint32_t> f = pack_fragments(part.data(), 9, cin, cin_pad, coutp, coutp);
        all.insert(all.end(), f.begin(), f.end());
    }
    std::vector<uint16_t> b(cout_pad, 0);
    if (bias)
        for (int o = 0; o < cout; o++) b[o] = float_to_half_bits(bias[o]);
    cudaError_t e = cudaMalloc(&out.frag, all.size() * 4);
    if (e == cudaSuccess) e = cudaMemcpy(out.frag, all.data(), all.size() * 4, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMalloc(&out.bias, b.size() * 2);
    if (e == cudaSuccess) e = cudaMemcpy(out.bias, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        grelease(out);
        return e;
    }
    out.cin = cin;
    out.cin_pad = cin_pad;
    out.cout = cout;
    out.cout_pad = cout_pad;
    out.coutp = coutp;
    return cudaSuccess;
}

void grelease(GWeights &w)
{
    cudaFree(w.frag);
    cudaFree(w.bias);
    w = GWeights{};
}

bool gsupported(const GWeights &w, const GConvArgs &a, std::string *why) { return gfind(w, a, why) != nullptr; }
int gvariant_count() { return kGVariants; }

size_t gworkspace(const GWeights &w, const GConvArgs &a)
{
    size_t m = 0;
    for (int v = 0; v < kGVariants; v++) {
        const int k = ksplit_for(w, a, kWarps[v]);
        if (k > 1 || a.out_f32) m = std::max(m, (size_t)k * a.H * a.W * w.cout_pad);
    }
    return m;
}

cudaError_t glaunch(const GWeights &w, const GConvArgs &a, int variant, cudaStream_t s)
{
    const GEntry *e = gfind(w, a, nullptr);
    if (!e) return cudaErrorInvalidValue;
    if (variant < 0 || variant >= kGVariants) variant = 0;
    int ks = a.sums ? 1 : ksplit_for(w, a, kWarps[variant]);  // the sums come from the unsplit epilogue
    if (ks > 1 && (!a.ws || (size_t)ks * a.H * a.W * w.cout_pad > a.ws_floats)) ks = 1;
    if (a.out_f32 && (!a.ws || (size_t)ks * a.H * a.W * w.cout_pad > a.ws_floats)) return cudaErrorInvalidValue;
    const gdev::GParams p = params(w, a, ks);
    cudaError_t r = e->fn[variant](p, (w.cout_pad / w.coutp) * ks, s);
    if (r != cudaSuccess || (ks == 1 && !a.out_f32)) return r;
    const size_t n = (size_t)a.H * a.W * (w.cout_pad / 2);
    gdev::k_gconv_reduce<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(p);
    return cudaGetLastError();
}

int gpick(const GWeights &w, const GConvArgs &a, cudaStream_t s, float *best_ms)
{
    if (!gfind(w, a, nullptr)) return -1;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int best = -1;
    float bt = 1e30f;
    for (int v = 0; v < kGVariants; v++) {
        for (int i = 0; i < 2; i++) glaunch(w, a, v, s);
        cudaEventRecord(e0, s);
        for (int i = 0; i < 5; i++) glaunch(w, a, v, s);
        cudaEventRecord(e1, s);
        if (cudaEventSynchronize(e1) != cudaSuccess) break;
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < bt) {
            bt = ms;
            best = v;
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (best_ms) *best_ms = bt / 5;
    return best;
}

}  // namespace ajinative
