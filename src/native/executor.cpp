// native/executor.cpp — see executor.h.
#include "native/executor.h"

#include <cstdio>

namespace ajinative {

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

std::unique_ptr<Executor> Executor::create(const Plan &plan, int in_w, int in_h, cudaStream_t s, std::string *why)
{
    auto fail = [&](const std::string &m) -> std::unique_ptr<Executor> {
        if (why) *why = m;
        return nullptr;
    };
    int dev = 0;
    std::string reason;
    if (cudaGetDevice(&dev) != cudaSuccess || !device_supported(dev, &reason)) return fail(reason.empty() ? "no CUDA device" : reason);
    if (plan.steps.empty()) return fail("empty plan");
    const int u = plan.unshuffle, r = plan.depth_to_space;
    if (in_w % u || in_h % u) return fail("this model needs even video dimensions");
    if (plan.aligned_only && (in_w % u || in_h % u)) return fail("this model needs aligned video dimensions");

    std::unique_ptr<Executor> e(new Executor());
    e->W_ = in_w / u;
    e->H_ = in_h / u;
    e->out_w_ = e->W_ * r;
    e->out_h_ = e->H_ * r;
    const size_t px = (size_t)e->W_ * e->H_;

    // weights and ops
    auto upload_w = [&](const PlanWeights &pw, int cin_pad, int cout_pad, const Weights **out) -> bool {
        e->weights_.push_back(std::make_unique<Weights>());
        Weights &w = *e->weights_.back();
        const cudaError_t ce = upload(w, pw.w.data(), pw.bias.data(), pw.slope.empty() ? nullptr : pw.slope.data(), pw.taps,
                                      pw.cin, cin_pad, pw.cout, cout_pad);
        if (ce != cudaSuccess) { reason = std::string("weight upload failed: ") + cudaGetErrorString(ce); return false; }
        e->bytes_ += (size_t)pw.taps * cin_pad * cout_pad * 2 + (size_t)cout_pad * 4;
        *out = &w;
        return true;
    };
    for (const PlanStep &ps : plan.steps) {
        Step st;
        st.src = ps.src;
        st.resid = ps.resid;
        st.dst = ps.dst;
        if (ps.kind == PlanStep::CONCAT) {
            st.concat = true;
            const Weights *w;
            if (!upload_w(ps.w, ps.w.cin, ps.w.cout, &w)) return fail(reason);
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
            const int cin_pad = ps.in == 0 ? ps.cin : 16;
            const int cout_pad = ps.out == 0 ? ps.cout : ps.out == 1 ? 16 : ps.cout;
            if (!upload_w(ps.w, cin_pad, cout_pad, &st.op.w)) return fail(reason);
            if (ps.gate == 2 && !upload_w(ps.att, ps.att.cin, ps.att.cout, &st.op.att)) return fail(reason);
            if (!supported(st.op, &reason)) return fail("conv " + std::to_string(ps.cin) + "->" + std::to_string(ps.cout) + ": " + reason);
        }
        e->steps_.push_back(st);
    }

    // feature buffers: reuse a buffer of the same size once its tensor's last reader has run
    const int nt = (int)plan.channels.size();
    std::vector<int> last(nt, -1), born(nt, -1);
    for (int i = 0; i < (int)e->steps_.size(); i++) {
        const Step &st = e->steps_[i];
        auto read = [&](int t) { if (t >= 0) last[t] = i; };
        read(st.src);
        read(st.resid);
        for (int t : st.srcs) read(t);
        if (st.dst >= 0) born[st.dst] = i;
    }
    e->tensor_.assign(nt, nullptr);
    std::vector<std::pair<size_t, void *>> free_list;  // (bytes, buffer)
    for (int i = 0; i < (int)e->steps_.size(); i++) {
        const int d = e->steps_[i].dst;
        if (d >= 0 && !e->tensor_[d]) {
            const size_t need = px * plan.channels[d] * 2;
            void *b = nullptr;
            for (size_t k = 0; k < free_list.size(); k++)
                if (free_list[k].first == need) { b = free_list[k].second; free_list.erase(free_list.begin() + k); break; }
            if (!b) {
                const cudaError_t ce = cudaMalloc(&b, need);
                if (ce != cudaSuccess) return fail(std::string("feature buffer allocation failed: ") + cudaGetErrorString(ce));
                e->buffers_.push_back(b);
                e->bytes_ += need;
            }
            e->tensor_[d] = b;
        }
        for (int t = 0; t < nt; t++)
            if (e->tensor_[t] && (last[t] == i || (last[t] < 0 && born[t] == i)))
                free_list.push_back({px * plan.channels[t] * 2, e->tensor_[t]});
    }

    // tile variants, timed on scratch input/output in step order
    void *in = nullptr, *out = nullptr;
    const size_t in_bytes = (size_t)3 * in_w * in_h * 2, out_bytes = (size_t)3 * e->out_w_ * e->out_h_ * 2;
    if (cudaMalloc(&in, in_bytes) != cudaSuccess || cudaMalloc(&out, out_bytes) != cudaSuccess) {
        cudaFree(in);
        return fail("scratch allocation failed");
    }
    cudaMemsetAsync(in, 0, in_bytes, s);
    for (Step &st : e->steps_) {
        if (st.concat) {
            const void *srcs[4];
            for (int k = 0; k < 4; k++) srcs[k] = e->ptr(st.srcs[k], in, out);
            if (launch(st.cop, srcs, const_cast<void *>(e->ptr(st.dst, in, out)), e->W_ * e->H_, 0, s) != cudaSuccess) {
                reason = "concat launch failed";
                break;
            }
            continue;
        }
        Tensors t;
        t.src = e->ptr(st.src, in, out);
        t.resid = e->ptr(st.resid, in, out);
        t.dst = const_cast<void *>(e->ptr(st.dst, in, out));
        st.variant = pick_variant(st.op, t, e->H_, e->W_, s);
        if (st.variant < 0) { reason = "timing the tile variants failed"; break; }
    }
    cudaStreamSynchronize(s);
    cudaFree(in);
    cudaFree(out);
    if (!reason.empty()) return fail(reason);
    return e;
}

cudaError_t Executor::run(const void *in, void *out, cudaStream_t s) const
{
    for (const Step &st : steps_) {
        cudaError_t ce;
        if (st.concat) {
            const void *srcs[4];
            for (int k = 0; k < 4; k++) srcs[k] = ptr(st.srcs[k], in, out);
            ce = launch(st.cop, srcs, const_cast<void *>(ptr(st.dst, in, out)), W_ * H_, 0, s);
        } else {
            Tensors t;
            t.src = ptr(st.src, in, out);
            t.resid = ptr(st.resid, in, out);
            t.dst = const_cast<void *>(ptr(st.dst, in, out));
            ce = launch(st.op, t, H_, W_, st.variant, s);
        }
        if (ce != cudaSuccess) return ce;
    }
    return cudaSuccess;
}

}  // namespace ajinative
