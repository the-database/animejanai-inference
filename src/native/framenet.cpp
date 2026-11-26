// native/framenet.cpp — see framenet.h. Also the lowered conv nets as image models
// (build_lowered, for ImgNet::create): the image path's tiles on the executor.
#include "native/framenet.h"

#include <algorithm>
#include <cstdio>

#include "native/conv.h"
#include "native/device.h"
#include "native/executor.h"
#include "native/img_kernels.h"
#include "native/imgnet.h"
#include "native/lower.h"
#include "native/onnx_graph.h"

namespace ajinative {
namespace {

std::string gb(double bytes)
{
    char b[32];
    std::snprintf(b, sizeof(b), "%.1f GB", bytes / (1024.0 * 1024 * 1024));
    return b;
}

std::string mb(size_t bytes)
{
    return std::to_string((bytes + (1 << 20) - 1) >> 20) + " MB";
}

// Match the two V3.1 HD topologies. Other networks using the same channel width keep their precision.
bool full_spanf3(const Plan &plan)
{
    if (plan.unshuffle != 2 || plan.depth_to_space != 4 || !plan.model_channels.empty() ||
        (plan.steps.size() != 18 && plan.steps.size() != 27)) return false;
    const auto &steps = plan.steps;
    const int c = steps.size() == 18 ? 48 : 64;
    const PlanStep &head = steps.front(), &join = steps[steps.size() - 2], &tail = steps.back();
    if (head.kind != PlanStep::CONV || head.in != int(Input::NCHW3Unshuffle2) || head.cout != c ||
        join.kind != PlanStep::CONCAT || join.c != c || join.nsrc != 4 ||
        tail.kind != PlanStep::CONV || tail.src != join.dst || tail.out != int(Output::DepthToSpace4)) return false;
    for (size_t i = 1; i + 2 < steps.size(); ++i) {
        const PlanStep &st = steps[i];
        const bool gated = i % 3 == 0;
        if (st.kind != PlanStep::CONV || st.cin != c || st.cout != c || st.src != steps[i - 1].dst ||
            st.in != int(Input::NHWC) || st.out != int(Output::NHWC) ||
            st.act != int(gated ? Act::None : Act::Relu) || st.gate != int(gated ? Gate::Tanh1x1 : Gate::None) ||
            (gated && st.resid != steps[i - 3].dst)) return false;
    }
    return true;
}

// The executor at the frame size.
class LoweredFrame : public FrameNet {
public:
    LoweredFrame(std::unique_ptr<Executor> e, std::string w) : ex(std::move(e)), what(std::move(w)) {}
    cudaError_t run(const void *in, void *out, cudaStream_t s) override { return ex->run(in, out, s); }
    int out_w() const override { return ex->out_w(); }
    int out_h() const override { return ex->out_h(); }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return ex->device_bytes(); }

private:
    std::unique_ptr<Executor> ex;
    std::string what;
};

// An image model at the (padded) frame size.
class ImageFrame : public FrameNet {
public:
    ~ImageFrame() override
    {
        cudaFree(pin);
        cudaFree(pout);
    }
    cudaError_t run(const void *in, void *out, cudaStream_t s) override
    {
        if (!pin) return net->run(in, false, out, false, w, h, s);
        cudaError_t e = img_pad_reflect(in, w, h, pin, pw, ph, s);
        if (e == cudaSuccess) e = net->run(pin, false, pout, false, pw, ph, s);
        const size_t ow = (size_t)w * S, oh = (size_t)h * S, pow_ = (size_t)pw * S, poh = (size_t)ph * S;
        for (int c = 0; c < 3 && e == cudaSuccess; c++)
            e = cudaMemcpy2DAsync((char *)out + c * ow * oh * 2, ow * 2, (const char *)pout + c * pow_ * poh * 2, pow_ * 2,
                                  ow * 2, oh, cudaMemcpyDeviceToDevice, s);
        return e;
    }
    int out_w() const override { return w * S; }
    int out_h() const override { return h * S; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return net->device_bytes() + pad_bytes; }

    std::unique_ptr<ImgNet> net;
    int w = 0, h = 0, pw = 0, ph = 0, S = 1;
    void *pin = nullptr, *pout = nullptr;
    size_t pad_bytes = 0;
    std::string what;
};

// A lowered conv net behind the image-model interface: the image path's tiles on the executor.
class LoweredImg : public ImgNet {
public:
    ~LoweredImg() override
    {
        cudaFree(in16);
        cudaFree(out16);
    }
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override
    {
        if (!accepts(w, h)) return cudaErrorInvalidValue;
        const size_t n_in = (size_t)3 * w * h, n_out = n_in * sc * sc;
        // fp32 sides go through fp16 scratch (allocated on first use; libaji's image path is fp16)
        if ((in_f32 && !in16 && cudaMalloc(&in16, (size_t)3 * maxw * maxh * 2) != cudaSuccess) ||
            (out_f32 && !out16 && cudaMalloc(&out16, (size_t)3 * maxw * maxh * sc * sc * 2) != cudaSuccess))
            return cudaErrorMemoryAllocation;
        cudaError_t e = cudaSuccess;
        if (in_f32) e = img_cast(in, true, in16, false, n_in, s);
        if (e == cudaSuccess) e = ex->run(in_f32 ? in16 : in, out_f32 ? out16 : out, w, h, s);
        if (e == cudaSuccess && out_f32) e = img_cast(out16, false, out, true, n_out, s);
        return e;
    }
    int scale() const override { return sc; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override
    {
        return ex->device_bytes() + (in16 ? (size_t)3 * maxw * maxh * 2 : 0) + (out16 ? (size_t)3 * maxw * maxh * sc * sc * 2 : 0);
    }
    size_t weight_bytes() const override { return ex->weight_bytes(); }
    bool accepts(int w, int h) const override { return w > 0 && h > 0 && w <= maxw && h <= maxh && w % u == 0 && h % u == 0; }

    std::unique_ptr<Executor> ex;
    int sc = 1, u = 1, maxw = 0, maxh = 0;
    void *in16 = nullptr, *out16 = nullptr;
    std::string what;
};

}  // namespace

std::unique_ptr<ImgNet> build_lowered(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Plan plan;
    if (!lower(g, &plan, why, kernel_widths_cuda())) return nullptr;
    if (plan.unshuffle < 1 || plan.depth_to_space % plan.unshuffle) {
        if (why) *why = "output scale is not a whole number";
        return nullptr;
    }
    auto ex = Executor::create(plan, max_w, max_h, s, why);
    if (!ex) return nullptr;
    auto n = std::make_unique<LoweredImg>();
    n->ex = std::move(ex);
    n->sc = plan.depth_to_space / plan.unshuffle;
    n->u = plan.unshuffle;
    n->maxw = max_w;
    n->maxh = max_h;
    n->what = describe(plan);
    return n;
}

std::unique_ptr<FrameNet> FrameNet::lowered(const Plan &plan, int w, int h, cudaStream_t s, std::string *why,
                                            SpanPrecision precision)
{
    const bool hd = full_spanf3(plan);
    // On the RTX 3090 the 64-channel gates recover throughput with one extra feature map for distinct residuals.
    const auto gates = hd && plan.steps.front().cout == 64 && current_cuda_sm() == 86
        ? GateBuffers::ReadOnlyResidual : GateBuffers::Reuse;
    auto ex = Executor::create(plan, w, h, s, why, gates, hd && precision == SpanPrecision::FP8);
    if (!ex) return nullptr;
    const bool fused = hd && ex->enable_gate_projection();
    const bool reordered = hd && ex->enable_fp16_cache_order();
    std::string description = ajinative::describe(plan, plan.steps.size() - (fused ? 1 : 0));
    if (ex->uses_fp8()) description += ", FP8 SPANF3 body";
    else if (reordered || (hd && precision != SpanPrecision::Default)) description += ", FP16 SPANF3 body";
    if (ex->packed_links()) description += ", " + std::to_string(ex->packed_links()) + " packed feature maps";
    if (reordered) description += ", FP16 alternating tile rows";
    if (fused) description += ", fused gate and projection";
    return std::make_unique<LoweredFrame>(std::move(ex), std::move(description));
}

std::unique_ptr<FrameNet> FrameNet::image(const OnnxGraph &g, int w, int h, cudaStream_t s, std::string *why, bool *size_only)
{
    if (size_only) *size_only = false;
    auto fail = [&](const std::string &m, bool size) -> std::unique_ptr<FrameNet> {
        if (why) *why = m;
        if (size_only) *size_only = size;
        return nullptr;
    };
    int dev = 0;
    std::string reason;
    if (cudaGetDevice(&dev) != cudaSuccess || !device_supported(dev, &reason)) return fail(reason.empty() ? "no CUDA device" : reason, false);
    if (w < 1 || h < 1) return fail("bad frame size", true);

    // The probe: the model's size rule and its buffers per input pixel (they scale with the area;
    // what does not makes the estimate high, not low).
    constexpr int P = 128;
    std::unique_ptr<ImgNet> probe = ImgNet::create(g, P, P, s, why);
    if (!probe) return nullptr;
    int a = 1;
    while (a <= 64 && !probe->accepts(a, a)) a *= 2;
    if (a > 64) return fail("no input size rule found", false);
    const int pw = (w + a - 1) / a * a, ph = (h + a - 1) / a * a, S = probe->scale();
    const bool pad = pw != w || ph != h;
    const size_t pad_bytes = pad ? (size_t)3 * pw * ph * 2 * (1 + (size_t)S * S) : 0;
    const double per_px = (double)(probe->device_bytes() - probe->weight_bytes()) / ((double)P * P);
    const double need = (double)probe->weight_bytes() + per_px * pw * ph + (double)pad_bytes;
    probe.reset();

    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return fail("cannot read the free GPU memory", true);
    const double keep = (double)(1ull << 30) + total_b / 10.0;
    if (need + keep > (double)free_b)
        return fail("a whole " + std::to_string(w) + "x" + std::to_string(h) + " frame needs ~" + gb(need) + ", " + gb((double)free_b) +
                        " free (keeping " + gb(keep) + ")",
                    true);

    auto f = std::make_unique<ImageFrame>();
    f->net = ImgNet::create(g, pw, ph, s, why);
    if (!f->net) {
        if (size_only) *size_only = true;  // the probe built, so the size is what failed
        return nullptr;
    }
    if (!f->net->accepts(pw, ph)) return fail("frame size not accepted", true);
    f->w = w;
    f->h = h;
    f->pw = pw;
    f->ph = ph;
    f->S = S;
    if (pad) {
        if (cudaMalloc(&f->pin, (size_t)3 * pw * ph * 2) != cudaSuccess ||
            cudaMalloc(&f->pout, (size_t)3 * pw * ph * S * S * 2) != cudaSuccess) {
            cudaGetLastError();
            return fail("out of GPU memory for the padded frame", true);
        }
        f->pad_bytes = pad_bytes;
    }
    f->what = f->net->describe() + ", whole frame" + (pad ? " padded to " + std::to_string(pw) + "x" + std::to_string(ph) : std::string()) +
              ", " + mb(f->device_bytes());
    return f;
}

}  // namespace ajinative
