// native/executor.cpp — see executor.h.
#include "native/executor.h"

#include <algorithm>

#include "native/layout.h"

namespace ajinative {

namespace {

void reuse_gate_buffers(const Plan &plan, Layout &layout)
{
    if (std::none_of(plan.steps.begin(), plan.steps.end(), [](const PlanStep &s) { return s.gate != 0; }))
        return;

    std::vector<int> last_reader(plan.channels.size(), -1);
    for (int i = 0; i < (int)plan.steps.size(); ++i) {
        const PlanStep &s = plan.steps[i];
        if (s.src >= 0) last_reader[s.src] = i;
        if (s.resid >= 0) last_reader[s.resid] = i;
        if (s.kind == PlanStep::CONCAT)
            for (int src : s.srcs)
                if (src >= 0) last_reader[src] = i;
    }

    std::vector<size_t> sizes;
    std::vector<int> busy_through;
    for (int i = 0; i < (int)plan.steps.size(); ++i) {
        const PlanStep &s = plan.steps[i];
        if (s.dst < 0) continue;
        // Each feature has one producer. Its original slot still gives its physical size here.
        const size_t bytes = layout.buffer_bytes[layout.buffer[s.dst]];
        const Gate gate = (Gate)s.gate;
        int slot;
        if (s.kind == PlanStep::CONV && s.in == (int)Input::NHWC && s.out == (int)Output::NHWC &&
            (gate == Gate::Residual || gate == Gate::Tanh1x1 || gate == Gate::SigmoidHalf) &&
            s.resid >= 0 && s.resid != s.src && last_reader[s.resid] == i &&
            plan.channels[s.resid] == plan.channels[s.dst]) {
            // Each CTA reads its own residual pixels before the barrier preceding its writes.
            // The convolution source supplies a halo to other CTAs and must be a distinct tensor.
            slot = layout.buffer[s.resid];
        } else {
            for (slot = 0; slot < (int)sizes.size(); ++slot)
                if (busy_through[slot] < i && sizes[slot] == bytes) break;
            if (slot == (int)sizes.size()) {
                sizes.push_back(bytes);
                busy_through.push_back(-1);
            }
        }
        layout.buffer[s.dst] = slot;
        busy_through[slot] = last_reader[s.dst];
    }
    layout.buffer_bytes.swap(sizes);
}

}  // namespace

Executor::~Executor()
{
    for (void *b : buffers_) cudaFree(b);
    for (auto &w : weights_) release(*w);
}

const void *Executor::ptr(int t, const void *in, void *out) const
{
    if (t == kModelInput) return in;
    if (t == kModelOutput) return out;
    return t >= 0 ? tensor_[t] : nullptr;
}

std::unique_ptr<Executor> Executor::create(const Plan &plan, int in_w, int in_h, cudaStream_t s, std::string *why,
                                           bool reuse_gate_residuals)
{
    auto fail = [&](const std::string &m) -> std::unique_ptr<Executor> {
        if (why) *why = m;
        return nullptr;
    };
    int dev = 0;
    std::string reason;
    if (cudaGetDevice(&dev) != cudaSuccess || !device_supported(dev, &reason)) return fail(reason.empty() ? "no CUDA device" : reason);
    Layout L;
    if (!layout(plan, in_w, in_h, &L, &reason)) return fail(reason);
    if (reuse_gate_residuals) reuse_gate_buffers(plan, L);

    std::unique_ptr<Executor> e(new Executor());
    e->u_ = L.u;
    e->W_ = L.W;
    e->H_ = L.H;
    e->out_w_ = L.out_w;
    e->out_h_ = L.out_h;

    // weights and ops
    auto upload_w = [&](const PlanWeights &pw, int cin_pad, int cout_pad, const Weights **out) -> bool {
        e->weights_.push_back(std::make_unique<Weights>());
        Weights &w = *e->weights_.back();
        const cudaError_t ce = upload(w, pw.w.data(), pw.bias.data(), pw.slope.empty() ? nullptr : pw.slope.data(), pw.taps,
                                      pw.cin, cin_pad, pw.cout, cout_pad);
        if (ce != cudaSuccess) { reason = std::string("weight upload failed: ") + cudaGetErrorString(ce); return false; }
        const size_t wb = (size_t)pw.taps * cin_pad * cout_pad * 2 + (size_t)cout_pad * 4;
        e->bytes_ += wb;
        e->weight_bytes_ += wb;
        *out = &w;
        return true;
    };
    for (size_t i = 0; i < plan.steps.size(); i++) {
        const PlanStep &ps = plan.steps[i];
        Step st;
        st.src = ps.src;
        st.resid = ps.resid;
        st.dst = ps.dst;
        if (ps.kind == PlanStep::CONCAT) {
            st.concat = true;
            const Weights *w;
            if (!upload_w(ps.w, L.cin_pad[i], L.cout_pad[i], &w)) return fail(reason);
            st.cop = ConcatConv1x1{ps.c, ps.nsrc, w};
            for (int k = 0; k < 4; k++) st.srcs[k] = ps.srcs[k];
            if (!supported(st.cop, &reason)) return fail("concat: " + reason);
        } else {
            st.op.in = (Input)ps.in;
            st.op.act = (Act)ps.act;
            st.op.gate = (Gate)ps.gate;
            st.op.out = (Output)ps.out;
            st.op.cin = ps.cin;
            st.op.cout = ps.cout;
            st.op.in_norm = ps.in_norm;
            st.op.out_clamp = ps.out_clamp;
            for (int c = 0; c < 3; c++) {
                st.op.in_mean[c] = ps.in_mean[c];
                st.op.in_scale[c] = ps.in_scale[c];
            }
            if (!upload_w(ps.w, L.cin_pad[i], L.cout_pad[i], &st.op.w)) return fail(reason);
            if (ps.gate == 2 && !upload_w(ps.att, ps.att.cin, ps.att.cout, &st.op.att)) return fail(reason);
            if (!supported(st.op, &reason)) return fail("conv " + std::to_string(ps.cin) + "->" + std::to_string(ps.cout) + ": " + reason);
        }
        e->steps_.push_back(st);
    }

    // feature buffers, as the layout assigned them
    for (size_t need : L.buffer_bytes) {
        void *b = nullptr;
        const cudaError_t ce = cudaMalloc(&b, need);
        if (ce != cudaSuccess) return fail(std::string("feature buffer allocation failed: ") + cudaGetErrorString(ce));
        e->buffers_.push_back(b);
        e->bytes_ += need;
    }
    e->tensor_.assign(plan.channels.size(), nullptr);
    for (size_t t = 0; t < L.buffer.size(); t++)
        if (L.buffer[t] >= 0) e->tensor_[t] = e->buffers_[L.buffer[t]];

    // tile variants from the ops' shapes
    for (Step &st : e->steps_)
        if (!st.concat) st.variant = tile_variant(st.op);
    (void)s;
    return e;
}

cudaError_t Executor::run(const void *in, void *out, int w, int h, cudaStream_t s) const
{
    if (w < 1 || h < 1 || w % u_ || h % u_ || w / u_ > W_ || h / u_ > H_) return cudaErrorInvalidValue;
    const int W = w / u_, H = h / u_;
    for (const Step &st : steps_) {
        cudaError_t ce;
        if (st.concat) {
            const void *srcs[4];
            for (int k = 0; k < 4; k++) srcs[k] = ptr(st.srcs[k], in, out);
            ce = launch(st.cop, srcs, const_cast<void *>(ptr(st.dst, in, out)), W * H, 0, s);
        } else {
            Tensors t;
            t.src = ptr(st.src, in, out);
            t.resid = ptr(st.resid, in, out);
            t.dst = const_cast<void *>(ptr(st.dst, in, out));
            ce = launch(st.op, t, H, W, st.variant, s);
        }
        if (ce != cudaSuccess) return ce;
    }
    return cudaSuccess;
}

}  // namespace ajinative
