#pragma once
#include "native/lifetime_check.h"
#include "native/tail_layout.h"

namespace ajinative {
// The model and the host verifier bind the same body, Tail and workspace slots.
struct EsrganBuffers {
    void *in16 = nullptr, *F0 = nullptr, *D[3] = {}, *T = nullptr;

    float *ws = nullptr;
    size_t ws_floats = 0;

    // Whole-tile launch order: first, the RRDB convolutions, body conv, tail convolutions.
    static int step(size_t op, size_t dense_ops)
    {
        if (op == 0) return 2;
        if (op <= dense_ops) return 3;
        return 4 + (int)(op - dense_ops - 1);
    }

    void declare(LifetimePlan &lt, int bw, int bh, int in_stride, int nf, int ds, int strip_body_rows, TailLayout &tail)
    {
        const size_t px = size_t(bw) * bh;
        const int rows = strip_body_rows ? std::min(bh, strip_body_rows + 2 * TailLayout::kHalo) : bh;
        lt.add(&in16, px * in_stride * 2, {{1, 2}}, "input");
        lt.add(&F0, px * nf * 2, {{2, 4}}, "first");
        lt.add(&D[0], px * ds * 2, {{2, 4}}, "D0");
        // Only D0 is initialized by the first conv's copy. D1 and D2 are first written in the
        // RRDBs, and only D0 and F0 feed the final body conv. T can reuse either dead buffer.
        lt.add(&D[1], px * ds * 2, {{3, 3}}, "D1");
        lt.add(&D[2], px * ds * 2, {{3, 3}}, "D2");
        lt.add(&T, px * nf * 2, {{4, strip_body_rows ? 6 + tail.steps() : 5}}, "trunk");
        for (int k = 0; k < tail.steps(); ++k)
            lt.add(tail.out(k), tail.bytes(k, bw, rows), {{5 + k, 6 + k}}, "tail");
    }

    template<class Alloc>
    bool assign_workspace(Alloc alloc, LifetimeCheck *check)
    {
        if (!ws_floats) return true;
        const size_t bytes = ws_floats * sizeof(float);
        ws = static_cast<float *>(alloc(bytes));
        if (ws && check) check->region(ws, bytes);
        return ws != nullptr;
    }
};
} // namespace ajinative
