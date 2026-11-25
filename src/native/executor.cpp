// native/executor.cpp — see executor.h.
#include "native/executor.h"

#include "native/layout.h"
#include "native/span_fp8.h"

namespace ajinative {
namespace {

bool is_sm120()
{
    int device, major, minor;
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
           major == 12 && minor == 0;
}

}  // namespace

Executor::~Executor()
{
    for (void *p : fp8_owned_) cudaFree(p);
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
                                           bool reuse_gate_residuals, bool use_fp8)
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

    std::unique_ptr<Executor> e(new Executor());
    e->u_ = L.u;
    e->W_ = L.W;
    e->H_ = L.H;
    e->out_w_ = L.out_w;
    e->out_h_ = L.out_h;

    const bool fp8 = use_fp8 && span_fp8_available();
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
            // full_spanf3 validated each body convolution. The stem, attention, join and tail retain FP16 weights.
            if (fp8 && i > 0 && i + 2 < plan.steps.size()) {
                const auto err = span_fp8_pack(ps.w, &st.fp8_weights);
                if (st.fp8_weights) e->fp8_owned_.push_back(st.fp8_weights);
                if (err != cudaSuccess) return fail(cudaGetErrorString(err));
                e->bytes_ += ps.w.w.size();
                e->weight_bytes_ += ps.w.w.size();
            }
            if (!supported(st.op, &reason)) return fail("conv " + std::to_string(ps.cin) + "->" + std::to_string(ps.cout) + ": " + reason);
        }
        e->steps_.push_back(st);
    }

    std::vector<bool> packed;
    if (fp8) {
        // A residual or concat needs the complete FP16 map. Pack only a private
        // ReLU output whose sole reader would immediately convert it to E4M3.
        packed.resize(plan.channels.size());
        std::vector<int> readers(plan.channels.size());
        auto count = [&](int tensor) { if (tensor >= 0) ++readers[tensor]; };
        for (const Step &st : e->steps_) {
            if (st.concat) {
                for (int i = 0; i < st.cop.nsrc; ++i) count(st.srcs[i]);
            } else {
                count(st.src);
                count(st.resid);
            }
        }
        for (size_t i = 1; i < e->steps_.size(); ++i) {
            Step &producer = e->steps_[i - 1], &consumer = e->steps_[i];
            if (producer.fp8_weights && consumer.fp8_weights && producer.dst >= 0 &&
                consumer.src == producer.dst && readers[producer.dst] == 1 &&
                producer.op.gate == Gate::None && producer.op.act == Act::Relu &&
                (consumer.op.act == Act::Relu || consumer.op.gate == Gate::Tanh1x1)) {
                producer.packed_output = consumer.packed_input = true;
                packed[producer.dst] = true;
                ++e->packed_links_;
            }
        }
    }

    if (reuse_gate_residuals) reuse_cuda_gate_buffers(plan, L, packed);
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
    const bool tune_packed_tiles = e->packed_links_ && is_sm120();
    int body_index = 0;
    for (Step &st : e->steps_) {
        if (st.concat) continue;
        if (!st.fp8_weights) {
            st.variant = tile_variant(st.op);
            continue;
        }
        st.variant = 3;
        ++body_index;
        if (tune_packed_tiles && (st.packed_input || st.packed_output)) {
            // Measured on SM120. Packed ReLU outputs fit a 16-row tile in shared memory.
            if (st.packed_output) st.variant = st.packed_input ? 0 : 1;
            st.op.reverse_tile_rows = body_index % 2 == 0;
        }
    }
    (void)s;
    return e;
}

bool Executor::enable_fp16_cache_order()
{
    if (uses_fp8() || !is_sm120()) return false;
    for (size_t i = 1; i + 2 < steps_.size(); i += 2) steps_[i].op.reverse_tile_rows = true;
    return true;
}

bool Executor::enable_gate_projection()
{
    const size_t count = steps_.size();
    if ((count != 18 && count != 27) || u_ != 2 || out_w_ != 4 * W_ || !is_sm120()) return false;
    const int channels = count == 18 ? 48 : 64;
    const Step &gate = steps_[count - 3], &proj = steps_[count - 2];
    Step &tail = steps_.back();
    if (gate.concat || gate.op.in != Input::NHWC || gate.op.out != Output::NHWC || gate.op.gate != Gate::Tanh1x1 ||
        gate.op.cin != channels || gate.op.cout != channels || gate.resid < 0 || gate.dst < 0 || (gate.fp8_weights && !gate.packed_input) ||
        !proj.concat || proj.cop.c != channels || proj.cop.nsrc != 4 || proj.srcs[1] != gate.dst ||
        tail.concat || tail.src != proj.dst || tail.op.out != Output::DepthToSpace4 || tail.resid != -1 || tail.dst != kModelOutput)
        return false;
    // A neighbor CTA may still need the convolution source halo when this CTA stores.
    if (gate.src < 0 || tensor_[gate.src] == tensor_[gate.dst]) return false;
    for (int i = 0; i < 4; ++i)
        if (i != 1 && (proj.srcs[i] < 0 || tensor_[proj.srcs[i]] == tensor_[gate.dst])) return false;
    // The CTA reads only its own residual pixels, then projects map 1 from registers.
    // Maps 0, 2 and 3 live in distinct storage from the destination. The CTA barrier
    // before global stores retires every residual read even when gate.dst == gate.resid physically.
    tail.src = gate.dst;
    fused_projection_ = true;
    return true;
}

cudaError_t Executor::run(const void *in, void *out, int w, int h, cudaStream_t s) const
{
    if (w < 1 || h < 1 || w % u_ || h % u_ || w / u_ > W_ || h / u_ > H_) return cudaErrorInvalidValue;
    const int W = w / u_, H = h / u_;
    for (size_t i = 0; i < steps_.size(); ++i) {
        const Step &st = steps_[i];
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
            if (fused_projection_ && i + 3 == steps_.size()) {
                const Step &project = steps_[++i];
                const void *maps[4];
                for (int j = 0; j < 4; ++j) maps[j] = ptr(project.srcs[j], in, out);
                ce = span_gate_project(st.op, t, project.cop, maps, st.fp8_weights, H, W, s);
            } else if (st.fp8_weights) {
                ce = span_fp8_run(st.op, t, st.fp8_weights, H, W, st.variant, st.packed_input, st.packed_output, s);
            } else {
                ce = launch(st.op, t, H, W, st.variant, s);
            }
        }
        if (ce != cudaSuccess) return ce;
    }
    return cudaSuccess;
}

}  // namespace ajinative
