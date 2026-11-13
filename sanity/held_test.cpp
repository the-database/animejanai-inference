/*
 * held_test — CPU-only checks for core/held, the held-frame reuse rule
 * (aji_held_*), against synthetic comparisons (no GPU, no frames).
 *
 * Each frame has a position and a luma level: D(a, b) = |pos b - pos a|
 * (or a value set for the pair), region mean 7 = lum b - lum a, the other
 * region means 0. The driver decides frames in order with the two after
 * each as the look-ahead, and makes every undamaged upscaled frame the
 * anchor, as aji_encode does.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/held.h"

using namespace ajicore;

namespace {

int failures = 0;

void check(bool ok, const char *what)
{
    if (!ok) {
        printf("FAIL: %s\n", what);
        failures++;
    }
}

void check_eq(const std::string &got, const std::string &want, const char *what)
{
    if (got != want) {
        printf("FAIL: %s\n  got  %s\n  want %s\n", what, got.c_str(), want.c_str());
        failures++;
    }
}

struct Synth {
    std::vector<float> pos, lum;
    std::map<std::pair<uint64_t, uint64_t>, float> d;   // D set for a pair
    std::vector<std::pair<uint64_t, uint64_t>> calls;   // (a.id, b.id)
    int fail_call = -1;                                 // this call fails

    int diff(const aji_held_frame &a, const aji_held_frame &b,
             aji_frame_diff_result *r)
    {
        calls.push_back({a.id, b.id});
        if ((int)calls.size() - 1 == fail_call)
            return AJI_ERR_CUDA;
        memset(r, 0, sizeof(*r));
        auto it = d.find({a.id, b.id});
        r->max_tile = it != d.end() ? it->second : std::fabs(pos[b.id] - pos[a.id]);
        r->region_mean[7] = lum[b.id] - lum[a.id];
        return AJI_OK;
    }
};

aji_held_params params(float t = AJI_HELD_THRESHOLD_DEFAULT,
                       float f = AJI_HELD_FADE_DEFAULT,
                       int run = AJI_HELD_MAX_RUN_DEFAULT)
{
    aji_held_params p{};
    p.struct_size = sizeof(p);
    p.threshold = t;
    p.fade_threshold = f;
    p.max_run = run;
    return p;
}

struct Run {
    std::string decisions;           // U or R per frame
    std::vector<int> cmps;           // comparisons per decision
    std::vector<int> errors;
};

aji_held_frame frame(uint64_t id, bool damaged = false)
{
    aji_held_frame f{};
    f.id = id;
    f.damaged = damaged ? 1 : 0;
    return f;
}

// decide frames [from, to) of s; frames in `damaged` are damaged
Run drive(HeldRule &rule, Synth &s, size_t from, size_t to,
          const std::vector<uint64_t> &damaged = {})
{
    Run out;
    std::vector<aji_held_frame> f;
    for (size_t i = 0; i < s.pos.size(); i++) {
        bool dm = false;
        for (uint64_t x : damaged) dm |= x == i;
        f.push_back(frame(i, dm));
    }
    for (size_t i = from; i < to; i++) {
        const size_t before = s.calls.size();
        aji_held_decision d;
        rule.decide(f[i], i + 1 < to ? &f[i + 1] : nullptr,
                    i + 2 < to ? &f[i + 2] : nullptr, &d);
        out.decisions += d.action == AJI_HELD_REUSE ? 'R' : 'U';
        out.cmps.push_back((int)(s.calls.size() - before));
        out.errors.push_back(d.error);
        if (d.action == AJI_HELD_UPSCALE && !f[i].damaged && !d.error)
            rule.set_anchor(f[i]);
    }
    return out;
}

HeldRule::Diff bind(Synth &s)
{
    return [&s](const aji_held_frame &a, const aji_held_frame &b,
                aji_frame_diff_result *r) { return s.diff(a, b, r); };
}

Synth flat(size_t n, float p = 0.0f)
{
    Synth s;
    s.pos.assign(n, p);
    s.lum.assign(n, 0.0f);
    return s;
}

// one decision of frame 1 after anchor 0, look-ahead 2 and 3
std::string one(Synth s, aji_held_params p = params())
{
    HeldRule rule(p, bind(s));
    return drive(rule, s, 0, 4).decisions.substr(1, 1);
}

std::string repeat(char c, int n) { return std::string((size_t)n, c); }

std::string trace_line(const char *frame, const char *pair, const char *d,
                       const char *m7)
{
    std::string s = std::string(frame) + "," + pair + "," + d;
    for (int i = 0; i < 16; i++)
        s += std::string(",") + (i == 7 ? m7 : "0.0000");
    return s + "\n";
}

} // namespace

int main()
{
    // parameters
    {
        aji_held_params p = params();
        check(HeldRule::valid_params(&p), "default params valid");
        check(!HeldRule::valid_params(nullptr), "NULL params invalid");
        p.struct_size = sizeof(p) - 4;
        check(!HeldRule::valid_params(&p), "short struct_size invalid");
        p = params(-1.0f);
        check(!HeldRule::valid_params(&p), "negative threshold invalid");
        p = params(NAN);
        check(!HeldRule::valid_params(&p), "NaN threshold invalid");
        p = params(2.5f, -0.1f);
        check(!HeldRule::valid_params(&p), "negative fade threshold invalid");
        p = params(2.5f, 0.15f, -1);
        check(!HeldRule::valid_params(&p), "negative max_run invalid");
    }

    // no anchor: upscale, nothing compared
    {
        Synth s = flat(3);
        HeldRule rule(params(), bind(s));
        aji_held_frame f0 = frame(0), f1 = frame(1), f2 = frame(2);
        aji_held_decision d;
        rule.decide(f0, &f1, &f2, &d);
        check(d.action == AJI_HELD_UPSCALE && d.error == AJI_OK && s.calls.empty(),
              "no anchor: upscale without comparisons");
    }

    // an identical stream: 24 reused, an upscale, and the last two upscaled
    {
        Synth s = flat(40);
        HeldRule rule(params(), bind(s));
        Run r = drive(rule, s, 0, 40);
        check_eq(r.decisions,
                 "U" + repeat('R', 24) + "U" + repeat('R', 12) + "UU",
                 "identical stream");
        aji_held_stats st{};
        st.struct_size = sizeof(st);
        rule.stats(&st);
        // 5 for the first reuse after an anchor, 4 with the look-ahead's D
        check(st.seen == 40 && st.reused == 36 && st.comparisons == 146,
              "identical stream: stats");
        check(r.cmps[0] == 0 && r.cmps[1] == 5 && r.cmps[2] == 4 &&
              r.cmps[25] == 0 && r.cmps[26] == 5 && r.cmps[38] == 0,
              "identical stream: comparisons per decision");
    }

    // max_run 0 never reuses
    {
        Synth s = flat(10);
        HeldRule rule(params(2.5f, 0.15f, 0), bind(s));
        check_eq(drive(rule, s, 0, 10).decisions, repeat('U', 10), "max_run 0");
        check(s.calls.empty(), "max_run 0: nothing compared");
    }

    // D and fade against T and F (equality reuses)
    {
        Synth s = flat(4);
        s.pos = {0.0f, 2.5f, 2.5f, 2.5f};
        check_eq(one(s), "R", "D == T reuses");
        s.pos = {0.0f, 2.6f, 2.6f, 2.6f};
        check_eq(one(s), "U", "D > T upscales");
        s = flat(4);
        s.lum = {0.0f, 0.15f, 0.15f, 0.15f};
        check_eq(one(s), "R", "fade == F reuses");
        s.lum = {0.0f, 0.16f, 0.16f, 0.16f};
        check_eq(one(s), "U", "fade > F upscales");
        s.lum = {0.0f, -0.16f, -0.16f, -0.16f};
        check_eq(one(s), "U", "fade < -F upscales");
        s.lum = {0.0f, 0.16f, 0.16f, 0.16f};
        check_eq(one(s, params(2.5f, 0.2f)), "R", "fade threshold is used");
        s = flat(4);
        s.pos = {0.0f, 2.6f, 2.6f, 2.6f};
        check_eq(one(s, params(3.0f)), "R", "threshold is used");
    }

    // the look-ahead vetoes, each alone, for N+1 and for N+2
    {
        // the change starts at N: D(a,X) = 3 > T, D(N,X) = 2 <= T
        // (3 > 1.5 * max(1, 2, 0.5) is false: growth alone wouldn't veto)
        Synth s = flat(4);
        s.pos = {0.0f, 1.0f, 3.0f, 3.0f};
        check_eq(one(s), "U", "change starts at N (N+1)");
        s.pos = {0.0f, 1.0f, 1.0f, 3.0f};
        check_eq(one(s), "U", "change starts at N (N+2)");
        // D(N,X) > T too (X differs from N and the anchor alike), and
        // 3 <= 1.5 * max(0, 3, 0.5): neither test vetoes
        s.pos = {0.0f, 0.0f, 0.0f, 0.0f};
        s.d[{0, 2}] = 3.0f;
        s.d[{1, 2}] = 3.0f;
        check_eq(one(s), "R", "D(a,X) = D(N,X) = 3: no veto");
    }
    {
        // growth: D(a,X) = 1 > 1.5 * max(0.4, 0.6, 0.5) = 0.9
        Synth s = flat(4);
        s.pos = {0.0f, 0.4f, 1.0f, 1.0f};
        s.d[{0, 3}] = 0.4f;
        s.d[{1, 3}] = 0.0f;
        check_eq(one(s), "U", "D from the anchor grows (N+1)");
        s.d.clear();
        s.pos = {0.0f, 0.4f, 0.4f, 1.0f};
        check_eq(one(s), "U", "D from the anchor grows (N+2)");
        // the 0.5 floor: D(a,N) = D(N,X) = 0.2, D(a,X) = 0.7 <= 0.75
        s = flat(4);
        s.d[{0, 1}] = 0.2f;
        s.d[{0, 2}] = 0.7f;
        s.d[{1, 2}] = 0.2f;
        check_eq(one(s), "R", "the 0.5 floor (N+1)");
        s.d[{0, 2}] = 0.76f;
        check_eq(one(s), "U", "past the 0.5 floor (N+1)");
        s = flat(4);
        s.d[{0, 1}] = 0.2f;
        s.d[{0, 3}] = 0.7f;
        s.d[{1, 3}] = 0.2f;
        check_eq(one(s), "R", "the 0.5 floor (N+2)");
        s.d[{0, 3}] = 0.76f;
        check_eq(one(s), "U", "past the 0.5 floor (N+2)");
    }

    // damaged frames are never compared or reused, and never the anchor
    {
        for (uint64_t dmg = 1; dmg <= 3; dmg++) {
            Synth s = flat(4);
            HeldRule rule(params(), bind(s));
            Run r = drive(rule, s, 0, 4, {dmg});
            check(r.decisions[1] == 'U' && r.cmps[1] == 0,
                  "damaged p, n or n2: upscale without comparisons");
        }
        Synth s = flat(6);
        HeldRule rule(params(), bind(s));
        check_eq(drive(rule, s, 0, 6, {0}).decisions, "UURRUU",
                 "a damaged frame is not the anchor");
    }

    // n or n2 missing: upscale
    {
        Synth s = flat(3);
        HeldRule rule(params(), bind(s));
        aji_held_frame f0 = frame(0), f1 = frame(1), f2 = frame(2);
        aji_held_decision d;
        rule.decide(f0, &f1, &f2, &d);
        rule.set_anchor(f0);
        rule.decide(f1, nullptr, &f2, &d);
        check(d.action == AJI_HELD_UPSCALE, "n NULL: upscale");
        rule.set_anchor(f1);
        rule.decide(f2, &f1, nullptr, &d);
        check(d.action == AJI_HELD_UPSCALE && s.calls.empty(), "n2 NULL: upscale");
    }

    // comparison order and short circuits
    {
        typedef std::vector<std::pair<uint64_t, uint64_t>> Calls;
        Synth s = flat(5);
        HeldRule rule(params(), bind(s));
        Run r = drive(rule, s, 0, 5);
        check_eq(r.decisions, "URRUU", "order: decisions");
        check(s.calls == Calls({{0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3},
                                {0, 3}, {2, 3}, {0, 4}, {2, 4}}),
              "order: a-p, a-n, p-n, a-n2, p-n2; then the look-ahead's a-p skipped");
        check(r.cmps[1] == 5 && r.cmps[2] == 4, "order: 5, then 4");

        // a-p over T: one comparison
        s = flat(4);
        s.pos = {0.0f, 3.0f, 3.0f, 3.0f};
        HeldRule r2(params(), bind(s));
        check(drive(r2, s, 0, 4).cmps[1] == 1, "a-p over T: one comparison");

        // a veto at N+1 skips N+2's comparisons
        s = flat(4);
        s.pos = {0.0f, 1.0f, 3.0f, 3.0f};
        HeldRule r3(params(), bind(s));
        check(drive(r3, s, 0, 4).cmps[1] == 3, "veto at N+1: three comparisons");
        s = flat(4);
        s.pos = {0.0f, 1.0f, 1.0f, 3.0f};
        HeldRule r4(params(), bind(s));
        check(drive(r4, s, 0, 4).cmps[1] == 5, "veto at N+2: five comparisons");
    }

    // the look-ahead's D is used only for the frame it was measured on
    {
        Synth s = flat(6);
        HeldRule rule(params(), bind(s));
        aji_held_frame f[6];
        for (int i = 0; i < 6; i++) f[i] = frame((uint64_t)i);
        aji_held_decision d;
        rule.decide(f[0], &f[1], &f[2], &d);
        rule.set_anchor(f[0]);
        rule.decide(f[1], &f[2], &f[3], &d);
        check(d.action == AJI_HELD_REUSE && d.run == 1 && d.anchor_id == 0,
              "reuse: run and anchor id");
        const size_t before = s.calls.size();
        rule.decide(f[3], &f[4], &f[5], &d);    // not frame 2: a-p again
        check(d.action == AJI_HELD_REUSE && d.run == 2 &&
                  s.calls.size() - before == 5 && s.calls[before] ==
                  std::make_pair((uint64_t)0, (uint64_t)3),
              "id mismatch: a-p compared");
    }

    // a failed comparison turns the engine off until reset
    {
        Synth s = flat(12);
        s.fail_call = 2;        // frame 1's p-n
        HeldRule rule(params(), bind(s));
        Run r = drive(rule, s, 0, 5);
        check_eq(r.decisions, "UUUUU", "failed comparison: upscale");
        check(r.cmps[1] == 3 && r.errors[0] == AJI_OK &&
                  r.errors[1] == AJI_ERR_CUDA && r.errors[2] == AJI_ERR_CUDA &&
                  r.cmps[2] == 0,
              "failed comparison: error kept, nothing compared");
        aji_held_frame f4 = frame(4);
        rule.set_anchor(f4);
        Run r2 = drive(rule, s, 5, 9);
        check(r2.decisions == "UUUU" && r2.errors[0] == AJI_ERR_CUDA &&
                  r2.cmps[0] == 0,
              "off: a new anchor doesn't turn it on");
        rule.reset();
        Run r3 = drive(rule, s, 9, 12);
        check(r3.errors[0] == AJI_OK, "reset clears the error");
        Synth t = flat(6);
        t.fail_call = 0;        // frame 1's a-p
        HeldRule r4(params(), bind(t));
        Run r5 = drive(r4, t, 0, 4);
        check(r5.decisions == "UUUU" && r5.errors[1] == AJI_ERR_CUDA,
              "failed a-p: upscale");
        r4.reset();
        Run r6 = drive(r4, t, 0, 6);
        check_eq(r6.decisions, "URRRUU", "after reset: reuse again");
        aji_held_stats st{};
        st.struct_size = sizeof(st);
        r4.stats(&st);
        // the failed comparison counts, as in aji_encode before the engine
        check(st.seen == 10 && st.reused == 3 && st.comparisons == 1 + 5 + 4 + 4,
              "reset keeps the stats");
    }

    // reset forgets the anchor and the run
    {
        Synth s = flat(6);
        HeldRule rule(params(), bind(s));
        drive(rule, s, 0, 3);
        rule.reset();
        Run r = drive(rule, s, 3, 6);
        check(r.decisions[0] == 'U' && r.cmps[0] == 0, "reset: no anchor");
    }

    // the trace
    {
        const char *path = "aji_held_test_trace.csv";
        Synth s = flat(5);
        s.pos = {0.0f, 0.5f, 0.25f, 3.0f, 3.0f};
        s.lum = {0.0f, 0.0625f, 0.125f, 0.0f, 0.0f};
        {
            HeldRule rule(params(), bind(s));
            std::FILE *f = std::fopen(path, "w");
            check(f != nullptr, "trace: open");
            rule.set_trace(f);
            check_eq(drive(rule, s, 0, 5).decisions, "UURUU", "trace: decisions");
        }   // closes the trace
        std::string got;
        if (std::FILE *f = std::fopen(path, "r")) {
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) got.append(buf, n);
            std::fclose(f);
        }
        std::remove(path);
        const std::string want =
            "0,upscale\n" +
            trace_line("1", "ap", "0.5000", "0.0625") +
            trace_line("1", "an", "0.2500", "0.1250") +
            trace_line("1", "pn", "0.2500", "0.0625") +
            trace_line("1", "an2", "3.0000", "0.0000") +
            trace_line("1", "pn2", "2.5000", "-0.0625") +
            "1,upscale\n" +
            trace_line("2", "ap", "0.2500", "0.0625") +
            trace_line("2", "an", "2.5000", "-0.0625") +
            trace_line("2", "pn", "2.7500", "-0.1250") +
            trace_line("2", "an2", "2.5000", "-0.0625") +
            trace_line("2", "pn2", "2.7500", "-0.1250") +
            "2,reuse\n3,upscale\n4,upscale\n";
        check_eq(got, want, "trace text");
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("held_test: all passed\n");
    return 0;
}
