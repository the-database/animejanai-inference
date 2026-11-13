/*
 * aji_held.cpp — the held-frame reuse engine's C ABI (aji_held_*), in the
 * dispatcher only.
 *
 * The rule is core/held.cpp. Its comparisons go through the dispatcher's
 * own aji_frame_diff_marked, so they reach every backend the dispatcher
 * can load (a backend without marks gets the plain aji_frame_diff) and
 * follow a TensorRT-RTX context that fell back to TensorRT. Backends
 * export nothing for this.
 */

#include "aji.h"
#include "core/fileio.h"
#include "core/held.h"

#include <cstddef>
#include <cstring>
#include <new>

// aji_dispatch.cpp: whether c's backend can compare frames
bool aji_dispatch_has_frame_diff(aji_ctx *c);

struct aji_held {
    aji_ctx *c;
    void *cu_stream = nullptr;      // aji_held_decide's, during the call
    ajicore::HeldRule rule;

    aji_held(aji_ctx *ctx, const aji_held_params &p)
        : c(ctx),
          rule(p, [this](const aji_held_frame &a, const aji_held_frame &b,
                         aji_frame_diff_result *r) {
              // wait for the later of the two frames' marks (0, either
              // without one: for everything, as the plain call does)
              const uint64_t mark =
                  a.mark && b.mark ? (a.mark > b.mark ? a.mark : b.mark) : 0;
              return aji_frame_diff_marked(c, &a.frame, &b.frame, r, cu_stream,
                                           mark);
          })
    {
    }
};

extern "C" AJI_EXPORT int aji_held_version(void)
{
    return AJI_HELD_RULE_VERSION;
}

extern "C" AJI_EXPORT aji_held *aji_held_create(aji_ctx *c, const aji_held_params *p)
{
    if (!c || !ajicore::HeldRule::valid_params(p) || !aji_dispatch_has_frame_diff(c))
        return nullptr;
    return new (std::nothrow) aji_held(c, *p);
}

extern "C" AJI_EXPORT int aji_held_set_trace(aji_held *h, const char *path)
{
    if (!h)
        return AJI_ERR;
    if (!path) {
        h->rule.set_trace(nullptr);
        return AJI_OK;
    }
    std::FILE *f = ajicore::fopen_utf8(path, "w");
    if (!f)
        return AJI_ERR;
    h->rule.set_trace(f);
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_held_decide(aji_held *h, const aji_held_frame *p,
                                          const aji_held_frame *n,
                                          const aji_held_frame *n2,
                                          void *cu_stream, aji_held_decision *out)
{
    if (!h || !p || !out)
        return AJI_ERR;
    h->cu_stream = cu_stream;
    h->rule.decide(*p, n, n2, out);
    h->cu_stream = nullptr;
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_held_set_anchor(aji_held *h, const aji_held_frame *copy)
{
    if (!h || !copy || copy->damaged)
        return AJI_ERR;
    h->rule.set_anchor(*copy);
    return AJI_OK;
}

extern "C" AJI_EXPORT void aji_held_reset(aji_held *h)
{
    if (h)
        h->rule.reset();
}

extern "C" AJI_EXPORT void aji_held_get_stats(aji_held *h, aji_held_stats *out)
{
    if (!h || !out)
        return;
    aji_held_stats s{};
    h->rule.stats(&s);
    // the fields the caller's struct has room for
    const size_t first = offsetof(aji_held_stats, seen);
    const size_t n = out->struct_size < sizeof(s) ? out->struct_size : sizeof(s);
    if (n > first)
        memcpy((char *)out + first, (const char *)&s + first, n - first);
}

extern "C" AJI_EXPORT void aji_held_destroy(aji_held **h)
{
    if (!h || !*h)
        return;
    delete *h;
    *h = nullptr;
}
