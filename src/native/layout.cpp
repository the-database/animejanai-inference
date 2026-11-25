// native/layout.cpp — see layout.h.
#include "native/layout.h"

#include <algorithm>
#include <utility>

#include "native/ops.h"

namespace ajinative {

bool layout(const Plan &plan, int in_w, int in_h, Layout *out, std::string *why)
{
    auto fail = [&](const char *m) {
        if (why) *why = m;
        return false;
    };
    if (plan.steps.empty()) return fail("empty plan");
    const int u = plan.unshuffle, r = plan.depth_to_space;
    if (in_w % u || in_h % u) return fail("this model needs even video dimensions");
    if (plan.aligned_only && (in_w % u || in_h % u)) return fail("this model needs aligned video dimensions");

    Layout L;
    L.u = u;
    L.W = in_w / u;
    L.H = in_h / u;
    L.out_w = L.W * r;
    L.out_h = L.H * r;
    const size_t px = (size_t)L.W * L.H;

    const int ns = (int)plan.steps.size();
    L.cin_pad.resize(ns);
    L.cout_pad.resize(ns);
    for (int i = 0; i < ns; i++) {
        const PlanStep &ps = plan.steps[i];
        if (ps.kind == PlanStep::CONCAT) {
            L.cin_pad[i] = ps.w.cin;
            L.cout_pad[i] = ps.w.cout;
        } else {
            L.cin_pad[i] = ps.in == (int)Input::NHWC ? ps.cin : 16;
            // depth-to-space outputs pad to the kernel's channel tile: 3 / 12 -> 16, 27 -> 32
            L.cout_pad[i] = ps.out == (int)Output::NHWC ? ps.cout : (ps.cout + 15) / 16 * 16;
        }
    }

    // feature buffers: reuse a buffer of the same size once its tensor's last reader has run
    const int nt = (int)plan.channels.size();
    std::vector<int> last(nt, -1), born(nt, -1);
    for (int i = 0; i < ns; i++) {
        const PlanStep &st = plan.steps[i];
        auto read = [&](int t) { if (t >= 0) last[t] = i; };
        read(st.src);
        read(st.resid);
        if (st.kind == PlanStep::CONCAT)
            for (int t : st.srcs) read(t);
        if (st.dst >= 0) born[st.dst] = i;
    }
    L.buffer.assign(nt, -1);
    std::vector<std::pair<size_t, int>> free_list;  // (bytes, buffer)
    for (int i = 0; i < ns; i++) {
        const int d = plan.steps[i].dst;
        if (d >= 0 && L.buffer[d] < 0) {
            const size_t need = px * plan.channels[d] * 2;
            int b = -1;
            for (size_t k = 0; k < free_list.size(); k++)
                if (free_list[k].first == need) { b = free_list[k].second; free_list.erase(free_list.begin() + k); break; }
            if (b < 0) {
                b = (int)L.buffer_bytes.size();
                L.buffer_bytes.push_back(need);
            }
            L.buffer[d] = b;
        }
        for (int t = 0; t < nt; t++)
            if (L.buffer[t] >= 0 && (last[t] == i || (last[t] < 0 && born[t] == i)))
                free_list.push_back({px * plan.channels[t] * 2, L.buffer[t]});
    }
    *out = std::move(L);
    return true;
}

void reuse_cuda_gate_buffers(const Plan &plan, Layout &layout)
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

}  // namespace ajinative
