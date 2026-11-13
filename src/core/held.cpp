/*
 * core/held.cpp — the held-frame reuse rule (rule version 2).
 *
 * A frame that is the same picture as the last frame that was upscaled (the
 * anchor) need not be inferred: the anchor's output can be shown or encoded
 * again under the frame's timestamp. Anime holds drawings for 2-3 frames;
 * slides, screen recordings and still shots repeat frames exactly.
 *
 * D(x, y) is aji_frame_diff's largest 8x8-tile mean |y - x| over all planes,
 * in 8-bit levels; "fade" its largest |signed mean| over a 4x4 luma region
 * grid. With a the anchor, T = 2.5 and F = 0.15 (AJI_HELD_*_DEFAULT; the
 * caller may override them), frame N is reused iff
 *   D(a, N) <= T and fade(a, N) <= F,
 *   fewer than max_run (24) frames in a row were reused (long stills pick
 *   up the codec's quality refinements now and then),
 *   N, N+1 and N+2 are undamaged and N+1 and N+2 exist (the last two frames
 *   of a stream are always upscaled), and none of these holds:
 *     D(a, N+1) > T and D(N, N+1) <= T              the change starts at N
 *     D(a, N+1) > 1.5 * max(D(a, N), D(N, N+1), 0.5) D from the anchor grows
 *     D(a, N+2) > T and D(N, N+2) <= T
 *     D(a, N+2) > 1.5 * max(D(a, N), D(N, N+2), 0.5)
 *
 * Why: codec noise between held frames gives D of about 0.5-3, slow real
 * motion more (a soft cloud shadow drifting by 3.5-5 a frame). Comparing
 * with the anchor rather than with the previous frame keeps a slow drift
 * from creeping through in small steps. Motion slower than T/2 a frame
 * still stays under T for a frame or two from the anchor, and with only the
 * first test every other frame of a slow pan was reused (60 of 120 frames
 * of a blurred still panning 0.1 px a frame): the trend tests see D from
 * the anchor grow, which noise doesn't do. The 0.5 floor keeps tiny noise
 * from tripping them. Fade guard: signed means don't separate noise from
 * drift (codec DC shifts move whole blocks), but a fade of 0.5 levels a
 * frame would step; noise-only region means stay under 0.13. Signed region
 * means between frames whose tile D is within 2.5 (noise only), on the
 * calibration clips' first 2 minutes: p99 0.061 / 0.101, max 0.071 / 0.127
 * (the larger clip's top values are a real drift of ~0.1 a frame in one
 * region, which the look-ahead rejects anyway); frame-wide, p99 0.036 /
 * 0.038, max 0.043 / 0.055. F = 0.15 is about 2x the noise maximum, and
 * well under a fade of 0.25 levels a frame.
 *
 * Evidence (aji_encode, RTX 5090): on full episodes 16-57% of frames are
 * reused (a 1999 DVD 16.2%, three 1080p web episodes 29.5-56.7%). Frame
 * counts and timestamps match an encode without reuse; with a lossless
 * NVENC every upscaled frame is bit-identical to one, and a reused frame is
 * 48.8 dB or more from a fresh upscale. A blind A/B of the closest calls
 * (clouds, a title-card fade, smoke, a blurred concert background) showed
 * no stall, and clean synthetic pans of 0.1-1 px a frame (sharp and
 * blurred) and drifting patches reuse no frame inside the motion.
 * Known limits: slow motion under codec noise can still be held for single
 * frames (with x264 noise added, a blurred pan of 0.1 px a frame held 26 of
 * 120 frames and a 0.25 px drifting patch 5, each alone: a sub-pixel
 * shift); a faint one-pixel change at half contrast or less (an
 * anti-aliased caret) can be held for up to max_run frames. Speed: where
 * inference is the limit (-preset p1) wall time fell 24-31%, about in
 * proportion to the reuse rate; where NVENC is (VideoJaNai's p7 at 4K on a
 * fast GPU), it changes little (CUDA: none; DirectML: 1-4% faster).
 *
 * The comparisons are made in a fixed order (a-N unless known from the
 * last decision's look-ahead, then a-N+1, N-N+1, a-N+2, N-N+2), each
 * written to the trace as it is made, with the same short circuits as the
 * rule's text: the trace (and every caller's comparison count) depends on
 * it. The arithmetic is float, as aji_encode's was, so traces from before
 * the engine moved here are byte-identical; keep this file free of
 * fast-math options.
 */

#include "held.h"

#include <chrono>
#include <cmath>

// FFMAX's form, not std::max: the same result on NaN as before the move
#define HELD_MAX(a, b) ((a) > (b) ? (a) : (b))

namespace ajicore {

bool HeldRule::valid_params(const aji_held_params *p)
{
    return p && p->struct_size >= sizeof(aji_held_params) &&
           p->threshold >= 0.0f && p->fade_threshold >= 0.0f && p->max_run >= 0;
}

HeldRule::HeldRule(const aji_held_params &p, Diff diff)
    : diff_(std::move(diff)), T_(p.threshold), F_(p.fade_threshold),
      max_run_(p.max_run)
{
}

HeldRule::~HeldRule()
{
    if (trace_)
        std::fclose(trace_);
}

void HeldRule::set_trace(std::FILE *f)
{
    if (trace_)
        std::fclose(trace_);
    trace_ = f;
}

// D(a, b) and fade(a, b) (NULL: not wanted); 0, or -1 if the comparison
// failed (the engine is off from then on)
int HeldRule::cmp(const aji_held_frame &a, const aji_held_frame &b,
                  const char *pair, float *d, float *fade)
{
    aji_frame_diff_result r;
    const auto t0 = std::chrono::steady_clock::now();
    const int ret = diff_(a, b, &r);
    cmp_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - t0).count();
    cmp_n_++;
    if (ret != AJI_OK) {
        off_ = ret;
        return -1;
    }
    float m = 0.0f;
    for (int i = 0; i < 16; i++)
        m = HELD_MAX(m, fabsf(r.region_mean[i]));
    *d = r.max_tile;
    if (fade) *fade = m;
    if (trace_) {
        /* frame (0-based count of decisions), pair, D, the 16 region means */
        std::fprintf(trace_, "%lld,%s,%.4f", (long long)seen_ - 1, pair,
                     r.max_tile);
        for (int i = 0; i < 16; i++)
            std::fprintf(trace_, ",%.4f", r.region_mean[i]);
        std::fputc('\n', trace_);
    }
    return 0;
}

// The look-ahead tests for frame p against frame n (n = N+1 or N+2): true
// when p may still be reused, false when not (or a comparison failed).
// d_ap is D(anchor, p); D(anchor, n) and fade(anchor, n) go to *d_an and
// *f_an.
bool HeldRule::ahead(const aji_held_frame &p, float d_ap, const aji_held_frame &n,
                     const char *an, const char *pn, float *d_an, float *f_an)
{
    float d_pn;
    if (cmp(anchor_, n, an, d_an, f_an) < 0 || cmp(p, n, pn, &d_pn, nullptr) < 0)
        return false;
    if (*d_an > T_ && d_pn <= T_)
        return false;   // the change starts at p
    if (*d_an > 1.5f * HELD_MAX(HELD_MAX(d_ap, d_pn), 0.5f))
        return false;   // D from the anchor grows: motion
    return true;
}

void HeldRule::decide(const aji_held_frame &p, const aji_held_frame *n,
                      const aji_held_frame *n2, aji_held_decision *out)
{
    bool skip = false;
    const bool known = next_known_ && p.id == next_id_;
    float d_ap = next_d_, f_ap = next_fade_;
    next_known_ = false;
    seen_++;
    if (!off_ && have_anchor_ && skip_run_ < max_run_ && n && n2 &&
        !p.damaged && !n->damaged && !n2->damaged &&
        (known || cmp(anchor_, p, "ap", &d_ap, &f_ap) == 0) &&
        d_ap <= T_ && f_ap <= F_) {
        float d_an, f_an, d_an2, f_an2;
        skip = ahead(p, d_ap, *n, "an", "pn", &d_an, &f_an) &&
               ahead(p, d_ap, *n2, "an2", "pn2", &d_an2, &f_an2);
        if (skip) {     // the anchor stays: D(anchor, n) is known
            next_known_ = true;
            next_d_ = d_an;
            next_fade_ = f_an;
            next_id_ = n->id;
        }
    }
    if (trace_)
        std::fprintf(trace_, "%lld,%s\n", (long long)seen_ - 1,
                     skip ? "reuse" : "upscale");
    *out = aji_held_decision{};
    out->error = off_;
    if (skip) {
        skip_run_++;
        reused_++;
        out->action = AJI_HELD_REUSE;
        out->run = skip_run_;
        out->anchor_id = anchor_.id;
        return;
    }
    out->action = AJI_HELD_UPSCALE;
    skip_run_ = 0;
    have_anchor_ = false;
}

void HeldRule::set_anchor(const aji_held_frame &copy)
{
    anchor_ = copy;
    have_anchor_ = true;
    next_known_ = false;    // the look-ahead was against the anchor before
}

void HeldRule::reset()
{
    have_anchor_ = false;
    anchor_ = aji_held_frame{};
    skip_run_ = 0;
    next_known_ = false;
    off_ = 0;
}

void HeldRule::stats(aji_held_stats *out) const
{
    out->seen = seen_;
    out->reused = reused_;
    out->comparisons = cmp_n_;
    out->compare_ns = cmp_ns_;
}

} // namespace ajicore
