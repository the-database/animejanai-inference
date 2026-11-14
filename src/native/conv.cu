// native/conv.cu — see conv.h. The kernels are instantiated here for a fixed set of shapes
// (the kConv table); supported() reports what is in it.
#include "native/conv.h"
#include "native/pack.h"
#include "native/conv_kernels.cuh"

#include <cstdio>
#include <cstring>

namespace ajinative {
namespace {

using LaunchFn = cudaError_t (*)(const dev::ConvParams &, cudaStream_t);
constexpr int kVariants = 2;
// Tile shapes: (warps, rows per warp). 8x2 measured best at 48 channels, 4x4 at 64 (RTX 5090).
const char *const kVariantNames[kVariants] = {"8 warps x 2 rows", "4 warps x 4 rows"};

template <int CINP, int COUTP, int WARPS, int WR, int IN, int ACT, int GATE, int OUT>
cudaError_t run_conv(const dev::ConvParams &p, cudaStream_t s)
{
    constexpr int TH = WARPS * WR, smem = dev::conv_smem_bytes<CINP, COUTP, WARPS, WR>();
    static_assert(smem <= 48 * 1024, "keep tiles within the default shared-memory limit");
    const dim3 grid((p.W + dev::TW - 1) / dev::TW, (p.H + TH - 1) / TH);
    dev::k_conv3x3<CINP, COUTP, WARPS, WR, IN, ACT, GATE, OUT><<<grid, WARPS * 32, smem, s>>>(p);
    return cudaGetLastError();
}

struct Key {
    int cinp, coutp, in, act, gate, out;
    bool operator==(const Key &o) const
    {
        return cinp == o.cinp && coutp == o.coutp && in == o.in && act == o.act && gate == o.gate && out == o.out;
    }
};
struct Entry {
    Key key;
    LaunchFn fn[kVariants];
};

#define AJI_CONV(CI, CO, IN, ACT, GATE, OUT)                                         \
    {{CI, CO, IN, ACT, GATE, OUT},                                                   \
     {&run_conv<CI, CO, 8, 2, IN, ACT, GATE, OUT>, &run_conv<CI, CO, 4, 4, IN, ACT, GATE, OUT>}}
#define AJI_BODY(C)                                                                  \
    AJI_CONV(C, C, 0, 0, 0, 0), AJI_CONV(C, C, 0, 1, 0, 0), AJI_CONV(C, C, 0, 2, 0, 0), \
    AJI_CONV(C, C, 0, 3, 0, 0), AJI_CONV(C, C, 0, 0, 1, 0), AJI_CONV(C, C, 0, 0, 2, 0), \
    AJI_CONV(C, C, 0, 0, 3, 0)
#define AJI_HEAD(C)                                                                  \
    AJI_CONV(16, C, 1, 0, 0, 0), AJI_CONV(16, C, 1, 1, 0, 0), AJI_CONV(16, C, 1, 2, 0, 0), \
    AJI_CONV(16, C, 2, 0, 0, 0), AJI_CONV(16, C, 2, 1, 0, 0), AJI_CONV(16, C, 2, 2, 0, 0)
#define AJI_TAIL(C) AJI_CONV(C, 16, 0, 0, 0, 1), AJI_CONV(C, 48, 0, 0, 0, 2)

const Entry kConv[] = {AJI_BODY(32), AJI_BODY(48), AJI_BODY(64), AJI_HEAD(32), AJI_HEAD(48), AJI_HEAD(64),
                       AJI_TAIL(32), AJI_TAIL(48), AJI_TAIL(64)};

template <int C, int NSRC, int WARPS>
cudaError_t run_concat(const dev::ConcatParams &p, cudaStream_t s)
{
    constexpr int smem = dev::concat_smem_bytes<C, NSRC, WARPS>();
    static_assert(smem <= 48 * 1024, "keep tiles within the default shared-memory limit");
    dev::k_concat1x1<C, NSRC, WARPS><<<(p.npx + WARPS * 16 - 1) / (WARPS * 16), WARPS * 32, smem, s>>>(p);
    return cudaGetLastError();
}
using ConcatFn = cudaError_t (*)(const dev::ConcatParams &, cudaStream_t);
struct ConcatEntry {
    int c, nsrc;
    ConcatFn fn;
};
const ConcatEntry kConcat[] = {
    {32, 2, &run_concat<32, 2, 4>}, {32, 3, &run_concat<32, 3, 4>}, {32, 4, &run_concat<32, 4, 4>},
    {48, 2, &run_concat<48, 2, 4>}, {48, 3, &run_concat<48, 3, 4>}, {48, 4, &run_concat<48, 4, 4>},
    {64, 2, &run_concat<64, 2, 4>}, {64, 3, &run_concat<64, 3, 4>}, {64, 4, &run_concat<64, 4, 4>},
};

bool fail(std::string *why, const char *msg)
{
    if (why) *why = msg;
    return false;
}

// The kernel shape an op needs, with the checks that do not depend on the table.
bool key_of(const Conv3x3 &op, Key &k, std::string *why)
{
    if (!op.w) return fail(why, "no weights");
    if (op.w->taps != 9) return fail(why, "weights are not 3x3");
    k.in = (int)op.in;
    k.act = (int)op.act;
    k.gate = (int)op.gate;
    k.out = (int)op.out;
    switch (op.in) {
    case Input::NHWC:
        if (op.cin % 16) return fail(why, "NHWC input channels must be a multiple of 16");
        k.cinp = op.cin;
        break;
    case Input::NCHW3:
        if (op.cin != 3) return fail(why, "NCHW3 input has 3 channels");
        k.cinp = 16;
        break;
    case Input::NCHW3Unshuffle2:
        if (op.cin != 12) return fail(why, "unshuffled NCHW3 input has 12 channels");
        k.cinp = 16;
        break;
    }
    switch (op.out) {
    case Output::NHWC:
        if (op.cout % 16) return fail(why, "NHWC output channels must be a multiple of 16");
        k.coutp = op.cout;
        break;
    case Output::DepthToSpace2:
        if (op.cout != 12) return fail(why, "depth-to-space 2 output has 12 channels");
        k.coutp = 16;
        break;
    case Output::DepthToSpace4:
        if (op.cout != 48) return fail(why, "depth-to-space 4 output has 48 channels");
        k.coutp = 48;
        break;
    }
    if (op.w->cin != op.cin || op.w->cout != op.cout || op.w->cin_pad != k.cinp || op.w->cout_pad != k.coutp)
        return fail(why, "weights do not match the op's channels");
    if (op.gate != Gate::None) {
        if (op.act != Act::None) return fail(why, "gates take no activation");
        if (op.in != Input::NHWC || op.out != Output::NHWC || op.cin != op.cout)
            return fail(why, "gates need NHWC input and output with cin == cout");
    }
    if (op.gate == Gate::Tanh1x1) {
        if (!op.att || op.att->taps != 1 || op.att->cin != op.cout || op.att->cout != op.cout || op.att->cin_pad != op.cout ||
            op.att->cout_pad != op.cout)
            return fail(why, "tanh gate needs a 1x1 cout -> cout attention conv");
    }
    if (op.act == Act::PRelu && !op.w->slope) return fail(why, "PReLU without slopes");
    return true;
}

const Entry *find(const Conv3x3 &op, std::string *why)
{
    Key k{};
    if (!key_of(op, k, why)) return nullptr;
    for (const Entry &e : kConv)
        if (e.key == k) return &e;
    fail(why, "no kernel for this combination of channels, input, activation, gate and output");
    return nullptr;
}

const ConcatEntry *find(const ConcatConv1x1 &op, std::string *why)
{
    if (!op.w || op.w->taps != 1 || op.w->cin != op.nsrc * op.c || op.w->cout != op.c || op.w->cin_pad != op.w->cin ||
        op.w->cout_pad != op.c) {
        fail(why, "weights do not match a 1x1 nsrc*c -> c conv");
        return nullptr;
    }
    for (const ConcatEntry &e : kConcat)
        if (e.c == op.c && e.nsrc == op.nsrc) return &e;
    fail(why, "no concat kernel for this channel count and number of sources");
    return nullptr;
}

cudaError_t upload_half2(void **dst, const float *v, int n, int npad)
{
    std::vector<uint16_t> h(npad, 0);
    for (int i = 0; i < n; i++) h[i] = float_to_half_bits(v[i]);
    cudaError_t e = cudaMalloc(dst, npad * 2);
    if (e == cudaSuccess) e = cudaMemcpy(*dst, h.data(), npad * 2, cudaMemcpyHostToDevice);
    return e;
}

}  // namespace

cudaError_t upload(Weights &out, const uint16_t *w, const float *bias, const float *slope, int taps, int cin,
                   int cin_pad, int cout, int cout_pad)
{
    release(out);
    if (cin_pad % 16 || cout_pad % 16 || cin > cin_pad || cout > cout_pad) return cudaErrorInvalidValue;
    const std::vector<uint32_t> f = pack_fragments(w, taps, cin, cin_pad, cout, cout_pad);
    cudaError_t e = cudaMalloc(&out.frag, f.size() * 4);
    if (e == cudaSuccess) e = cudaMemcpy(out.frag, f.data(), f.size() * 4, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = upload_half2(&out.bias, bias, cout, cout_pad);
    if (e == cudaSuccess && slope) e = upload_half2(&out.slope, slope, cout, cout_pad);
    if (e != cudaSuccess) {
        release(out);
        return e;
    }
    out.taps = taps;
    out.cin = cin;
    out.cin_pad = cin_pad;
    out.cout = cout;
    out.cout_pad = cout_pad;
    return cudaSuccess;
}

void release(Weights &w)
{
    cudaFree(w.frag);
    cudaFree(w.bias);
    cudaFree(w.slope);
    w = Weights{};
}

bool device_supported(int device, std::string *why)
{
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return fail(why, "cannot query the device");
    if (prop.major < 8) return fail(why, "needs compute capability 8.0 or newer");
    return true;
}

bool supported(const Conv3x3 &op, std::string *why) { return find(op, why) != nullptr; }
bool supported(const ConcatConv1x1 &op, std::string *why) { return find(op, why) != nullptr; }
int variant_count(const Conv3x3 &op) { return find(op, nullptr) ? kVariants : 0; }
int variant_count(const ConcatConv1x1 &op) { return find(op, nullptr) ? 1 : 0; }
const char *variant_name(const Conv3x3 &op, int variant)
{
    return find(op, nullptr) && variant >= 0 && variant < kVariants ? kVariantNames[variant] : "";
}

cudaError_t launch(const Conv3x3 &op, const Tensors &t, int H, int W, int variant, cudaStream_t s)
{
    const Entry *e = find(op, nullptr);
    if (!e || variant < 0 || variant >= kVariants || H <= 0 || W <= 0 || !t.src || !t.dst) return cudaErrorInvalidValue;
    if (op.gate != Gate::None && !t.resid) return cudaErrorInvalidValue;
    dev::ConvParams p{};
    p.src = static_cast<const half *>(t.src);
    p.resid = static_cast<const half *>(t.resid);
    p.dst = static_cast<half *>(t.dst);
    p.w = static_cast<const uint4 *>(op.w->frag);
    p.bias = static_cast<const half2 *>(op.w->bias);
    p.slope = static_cast<const half2 *>(op.w->slope);
    p.watt = op.att ? static_cast<const uint4 *>(op.att->frag) : nullptr;
    p.batt = op.att ? static_cast<const half2 *>(op.att->bias) : nullptr;
    p.H = H;
    p.W = W;
    return e->fn[variant](p, s);
}

cudaError_t launch(const ConcatConv1x1 &op, const void *const *srcs, void *dst, int npx, int variant, cudaStream_t s)
{
    const ConcatEntry *e = find(op, nullptr);
    if (!e || variant != 0 || npx <= 0 || !dst) return cudaErrorInvalidValue;
    dev::ConcatParams p{};
    for (int i = 0; i < op.nsrc; i++) {
        if (!srcs[i]) return cudaErrorInvalidValue;
        p.src[i] = static_cast<const half *>(srcs[i]);
    }
    p.dst = static_cast<half *>(dst);
    p.w = static_cast<const uint4 *>(op.w->frag);
    p.bias = static_cast<const half2 *>(op.w->bias);
    p.npx = npx;
    return e->fn(p, s);
}

int pick_variant(const Conv3x3 &op, const Tensors &t, int H, int W, cudaStream_t s, float *best_ms)
{
    const int n = variant_count(op);
    if (n == 0) return -1;
    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) != cudaSuccess) return -1;
    if (cudaEventCreate(&e1) != cudaSuccess) {
        cudaEventDestroy(e0);
        return -1;
    }
    int best = -1;
    float best_t = 0.f;
    constexpr int kWarm = 3, kRuns = 10;
    for (int v = 0; v < n; v++) {
        bool ok = true;
        for (int i = 0; i < kWarm && ok; i++) ok = launch(op, t, H, W, v, s) == cudaSuccess;
        ok = ok && cudaEventRecord(e0, s) == cudaSuccess;
        for (int i = 0; i < kRuns && ok; i++) ok = launch(op, t, H, W, v, s) == cudaSuccess;
        ok = ok && cudaEventRecord(e1, s) == cudaSuccess && cudaEventSynchronize(e1) == cudaSuccess;
        float ms = 0.f;
        if (!ok || cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) {
            best = -1;
            break;
        }
        if (best < 0 || ms < best_t) {
            best = v;
            best_t = ms;
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (best >= 0 && best_ms) *best_ms = best_t / kRuns;
    return best;
}

}  // namespace ajinative
