// native/layout: the size checks, channel padding and feature-buffer reuse of a lowered model,
// on plans written here (the shapes native/lower emits). CPU only.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "native/layout.h"
#include "native/ops.h"
#include "native/temporal_workspace.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

using namespace ajinative;

namespace {

PlanStep conv(int src, int dst, int cin, int cout, Input in = Input::NHWC, Output out = Output::NHWC)
{
    PlanStep s;
    s.kind = PlanStep::CONV;
    s.in = (int)in;
    s.out = (int)out;
    s.cin = cin;
    s.cout = cout;
    s.w.taps = 9;
    s.w.cin = cin;
    s.w.cout = cout;
    s.src = src;
    s.dst = dst;
    return s;
}

PlanStep concat(std::vector<int> srcs, int dst, int c)
{
    PlanStep s;
    s.kind = PlanStep::CONCAT;
    s.c = c;
    s.nsrc = (int)srcs.size();
    for (size_t k = 0; k < srcs.size(); k++) s.srcs[k] = srcs[k];
    s.w.taps = 1;
    s.w.cin = c * s.nsrc;
    s.w.cout = c;
    s.dst = dst;
    return s;
}

int test_refusals()
{
    Plan p;
    Layout L;
    std::string why;
    CHECK(!layout(p, 64, 64, &L, &why) && why == "empty plan");
    p.steps.push_back(conv(kModelInput, kModelOutput, 12, 12, Input::NCHW3Unshuffle2, Output::DepthToSpace2));
    p.unshuffle = 2;
    p.depth_to_space = 2;
    CHECK(!layout(p, 63, 64, &L, &why) && why == "this model needs even video dimensions");
    CHECK(!layout(p, 64, 31, &L, &why));
    CHECK(layout(p, 64, 32, &L, &why));
    CHECK(L.u == 2 && L.W == 32 && L.H == 16 && L.out_w == 64 && L.out_h == 32);
    // NCHW input pads cin to 16; a depth-to-space 2 output (12 channels) pads cout to 16
    CHECK(L.cin_pad[0] == 16 && L.cout_pad[0] == 16);
    CHECK(L.buffer_bytes.empty());
    return 0;
}

// SPANF3-shaped: unshuffled head, a block with a residual gate, a concat of three feature maps,
// a depth-to-space 4 tail.
int test_span_chain()
{
    Plan p;
    p.unshuffle = 2;
    p.depth_to_space = 4;
    p.channels = {48, 48, 48, 48, 48};
    p.steps.push_back(conv(kModelInput, 0, 12, 48, Input::NCHW3Unshuffle2));  // t0, read by s2 and s4
    p.steps.push_back(conv(0, 1, 48, 48));                                  // t1, read by s2
    PlanStep g = conv(1, 2, 48, 48);
    g.gate = (int)Gate::Residual;
    g.resid = 0;
    p.steps.push_back(g);                                                   // t2, read by s3, s4
    p.steps.push_back(conv(2, 3, 48, 48));                                  // t3, read by s4
    p.steps.push_back(concat({0, 2, 3}, 4, 48));                            // t4, read by s5
    p.steps.push_back(conv(4, kModelOutput, 48, 48, Input::NHWC, Output::DepthToSpace4));
    Layout L;
    std::string why;
    CHECK(layout(p, 1920, 1080, &L, &why));
    CHECK(L.W == 960 && L.H == 540 && L.out_w == 3840 && L.out_h == 2160);
    CHECK(L.cin_pad[0] == 16 && L.cout_pad[0] == 48);
    CHECK(L.cin_pad[4] == 144 && L.cout_pad[4] == 48);  // concat: not padded
    CHECK(L.cin_pad[5] == 48 && L.cout_pad[5] == 48);   // depth-to-space 4: 48 channels as they are
    // t1's buffer is free after s2 (its last reader), so t3 takes it; t4 needs a fourth, since
    // t0, t2 and t3 are still read by s4 itself
    CHECK(L.buffer.size() == 5);
    CHECK(L.buffer[0] == 0 && L.buffer[1] == 1 && L.buffer[2] == 2 && L.buffer[3] == 1 && L.buffer[4] == 3);
    CHECK(L.buffer_bytes.size() == 4);
    for (size_t b : L.buffer_bytes) CHECK(b == (size_t)960 * 540 * 48 * 2);
    return 0;
}

// Buffers are reused only at the same size; a tensor nobody reads is freed right after its step.
int test_sizes_and_unread()
{
    Plan p;
    p.channels = {64, 32, 64, 32};
    p.steps.push_back(conv(kModelInput, 0, 3, 64, Input::NCHW3));  // t0 (64), read by s1
    p.steps.push_back(conv(0, 1, 64, 32));                       // t1 (32), never read
    p.steps.push_back(conv(0, 2, 64, 64));                       // t2 (64): t0 is still read here
    p.steps.push_back(conv(2, 3, 64, 32));                       // t3 (32): takes t1's buffer
    p.steps.push_back(conv(3, kModelOutput, 32, 3));
    Layout L;
    std::string why;
    CHECK(layout(p, 100, 50, &L, &why));
    CHECK(L.u == 1 && L.W == 100 && L.H == 50 && L.out_w == 100 && L.out_h == 50);
    CHECK(L.cin_pad[0] == 16);  // NCHW3 input
    CHECK(L.buffer[0] == 0 && L.buffer[1] == 1 && L.buffer[2] == 2 && L.buffer[3] == 1);
    CHECK(L.buffer_bytes.size() == 3);
    CHECK(L.buffer_bytes[0] == (size_t)100 * 50 * 64 * 2 && L.buffer_bytes[1] == (size_t)100 * 50 * 32 * 2);
    // a feature id no step writes gets no buffer
    p.channels.push_back(16);
    CHECK(layout(p, 100, 50, &L, &why));
    CHECK(L.buffer.size() == 5 && L.buffer[4] == -1);
    return 0;
}

// Walk backwards from consumers. A write must spare every feature that survives the step.
// Of the inputs that die here, only the named gate outputs may take their residual's slot.
int check_live_slots(const Plan &p, const Layout &L, const std::set<int> &in_place_outputs)
{
    std::set<int> live;
    for (int i = (int)p.steps.size() - 1; i >= 0; --i) {
        const PlanStep &s = p.steps[i];
        if (s.dst >= 0) {
            live.erase(s.dst);
            for (int t : live) CHECK(L.buffer[t] != L.buffer[s.dst]);
        }
        for (int t : {s.src, s.resid, s.srcs[0], s.srcs[1], s.srcs[2], s.srcs[3]}) {
            if (t < 0) continue;
            if (s.dst >= 0 && L.buffer[t] == L.buffer[s.dst] &&
                !(in_place_outputs.count(s.dst) && t == s.resid && t != s.src)) {
                std::fprintf(stderr, "step %d: output t%d overwrites input t%d in slot %d\n", i, s.dst, t, L.buffer[t]);
                return 1;
            }
            live.insert(t);
        }
    }
    return 0;
}

// Five SPAN blocks with the stem, last gate, first gate and last block's first conv kept for the join.
int test_cuda_span_gates(Gate gate_type)
{
    Plan p;
    p.unshuffle = 2;
    p.depth_to_space = 4;
    p.channels.assign(17, 48);
    p.steps.push_back(conv(kModelInput, 0, 12, 48, Input::NCHW3Unshuffle2));
    for (int block = 0; block < 5; ++block) {
        const int input = 3 * block;
        p.steps.push_back(conv(input, input + 1, 48, 48));
        p.steps.push_back(conv(input + 1, input + 2, 48, 48));
        PlanStep gate = conv(input + 2, input + 3, 48, 48);
        gate.gate = (int)gate_type;
        gate.resid = input;
        p.steps.push_back(gate);
    }
    p.steps.push_back(concat({0, 15, 3, 13}, 16, 48));
    p.steps.push_back(conv(16, kModelOutput, 48, 48, Input::NHWC, Output::DepthToSpace4));

    Layout L;
    std::string why;
    CHECK(layout(p, 1920, 1080, &L, &why));
    CHECK(L.buffer_bytes.size() == 6);
    CHECK(check_live_slots(p, L, {}) == 0);
    reuse_cuda_gate_buffers(p, L);
    CHECK(check_live_slots(p, L, {9, 12, 15}) == 0);
    CHECK(L.buffer_bytes.size() == 5);
    CHECK(L.buffer[3] != L.buffer[6]);  // the join still needs block one's output
    for (int dst : {9, 12, 15}) CHECK(L.buffer[dst] == L.buffer[p.steps[dst].resid]);
    const std::vector<int> main_slots{0, 1, 2, 1, 2, 3, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 4};
    CHECK(L.buffer == main_slots);
    if (gate_type == Gate::Tanh1x1) {
        // The last block's first ReLU (t13) is also a projection input and stays FP16.
        std::vector<bool> packed(p.channels.size());
        for (int t : {1, 2, 4, 5, 7, 8, 10, 11, 14}) packed[t] = true;
        for (bool reuse : {false, true}) {
            CHECK(layout(p, 1920, 1080, &L, &why));
            if (reuse) reuse_cuda_gate_buffers(p, L, packed);
            CHECK(L.buffer_bytes.size() == (reuse ? 5 : 6));
            CHECK(check_live_slots(p, L, reuse ? std::set<int>{9, 12, 15} : std::set<int>{}) == 0);
            for (size_t bytes : L.buffer_bytes) CHECK(bytes == size_t(960) * 540 * 48 * 2);
            for (int dst : {9, 12, 15}) CHECK((L.buffer[dst] == L.buffer[p.steps[dst].resid]) == reuse);
        }
        // Step 9 can overwrite t6 only when both t6 and t9 hold FP16.
        for (int tensor : {6, 9}) {
            CHECK(layout(p, 1920, 1080, &L, &why));
            packed[tensor] = true;
            reuse_cuda_gate_buffers(p, L, packed);
            CHECK(L.buffer[9] != L.buffer[6]);
            packed[tensor] = false;
        }
    }
    return 0;
}

int test_excluded_gate(int residual, int stem_channels, Output output)
{
    Plan p;
    p.channels = {stem_channels, 48, 48};
    p.steps.push_back(conv(kModelInput, 0, 3, stem_channels, Input::NCHW3));
    p.steps.push_back(conv(0, 1, stem_channels, 48));
    // Keep a feature destination even for depth-to-space to exercise the output-mode guard independently.
    PlanStep gate = conv(1, 2, 48, 48, Input::NHWC, output);
    gate.gate = (int)Gate::Residual;
    gate.resid = residual;
    p.steps.push_back(gate);
    p.steps.push_back(conv(2, kModelOutput, 48, 3, Input::NHWC, Output::DepthToSpace1));

    Layout L;
    std::string why;
    CHECK(layout(p, 64, 32, &L, &why));
    reuse_cuda_gate_buffers(p, L);
    CHECK(L.buffer[gate.dst] != L.buffer[residual]);
    return check_live_slots(p, L, {});
}

}  // namespace

int main()
{
    const auto batch = ping_pong_layout(96, 32);
    CHECK(batch.size == 96 && batch.second_offset == 32);
    const auto two_lanes = ping_pong_layout(48, 32);
    CHECK(two_lanes.size == 64 && two_lanes.second_offset == 32);
    const auto cached = temporal_input_layout(7, 7, true);
    CHECK(cached.input_frames == 1 && cached.cache_entries == 7);
    for (auto full : {temporal_input_layout(7, 7, false), temporal_input_layout(7, 6, true),
                      temporal_input_layout(7, 0, true)}) {
        CHECK(full.input_frames == 7 && full.cache_entries == 0);
    }
    if (test_refusals()) return 1;
    if (test_span_chain()) return 1;
    if (test_sizes_and_unread()) return 1;
    for (Gate gate : {Gate::Residual, Gate::Tanh1x1, Gate::SigmoidHalf})
        if (test_cuda_span_gates(gate)) return 1;
    if (test_excluded_gate(1, 48, Output::NHWC)) return 1;  // residual is the conv source
    if (test_excluded_gate(0, 32, Output::NHWC)) return 1;  // residual and destination widths differ
    if (test_excluded_gate(0, 48, Output::DepthToSpace4)) return 1;
    std::printf("native layout: all checks passed\n");
    return 0;
}
