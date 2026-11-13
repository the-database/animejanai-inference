/*
 * core/held.h — the held-frame reuse rule (aji_held_*), backend-neutral.
 *
 * Part of the internal core (no ABI of its own). The dispatcher's
 * aji_held.cpp wraps it in the C ABI and binds the comparison to
 * aji_frame_diff_marked; aji_held_test binds it to synthetic diffs. The
 * rule and its reasoning are in held.cpp, the contract in aji.h.
 */

#ifndef AJI_CORE_HELD_H
#define AJI_CORE_HELD_H

#include "aji.h"

#include <cstdint>
#include <cstdio>
#include <functional>

namespace ajicore {

class HeldRule {
public:
    // D and region means of (a, b) into *r: an aji_status
    using Diff = std::function<int(const aji_held_frame &a, const aji_held_frame &b,
                                   aji_frame_diff_result *r)>;

    // p must be valid (see valid_params)
    HeldRule(const aji_held_params &p, Diff diff);
    ~HeldRule();
    HeldRule(const HeldRule &) = delete;
    HeldRule &operator=(const HeldRule &) = delete;

    static bool valid_params(const aji_held_params *p);

    // Takes the file (closes the one before; NULL: no trace).
    void set_trace(std::FILE *f);
    void decide(const aji_held_frame &p, const aji_held_frame *n,
                const aji_held_frame *n2, aji_held_decision *out);
    void set_anchor(const aji_held_frame &copy);
    void reset();
    void stats(aji_held_stats *out) const;

private:
    int cmp(const aji_held_frame &a, const aji_held_frame &b, const char *pair,
            float *d, float *fade);
    bool ahead(const aji_held_frame &p, float d_ap, const aji_held_frame &n,
               const char *an, const char *pn, float *d_an, float *f_an);

    Diff diff_;
    float T_, F_;
    int max_run_;
    bool have_anchor_ = false;
    aji_held_frame anchor_{};
    int skip_run_ = 0;
    bool next_known_ = false;     // look-ahead: D and fade of the next frame
    float next_d_ = 0.0f, next_fade_ = 0.0f;  // against the anchor, if known
    uint64_t next_id_ = 0;
    int off_ = 0;                 // a comparison failed: its aji_status
    int64_t seen_ = 0, reused_ = 0;
    int64_t cmp_n_ = 0, cmp_ns_ = 0;
    std::FILE *trace_ = nullptr;
};

} // namespace ajicore

#endif
