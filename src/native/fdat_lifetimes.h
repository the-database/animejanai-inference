#pragma once
#include "native/lifetime_plan.h"
#include <initializer_list>

namespace ajinative {
// The body and image head/tail execute sequentially in one allocation. Each phase starts at
// a known offset. The high-water mark covers every phase even when their sizes differ.
struct FdatPhaseLayout {
    char *base;
    LifetimePlan *view;
    bool disjoint = false;
    size_t cursor = 0, high = 0;

    void take(void **slot, size_t n, std::initializer_list<LifetimePlan::Range> live, const char *name)
    {
        const size_t offset = ((disjoint ? high : cursor) + 255) / 256 * 256;
        if (base) *slot = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) + offset);
        if (view) view->buffers.push_back({slot, n, live, offset, name});
        cursor = offset + n;
        high = std::max(high, cursor);
    }
};

struct FdatImageBuffers {
    void *IN16 = nullptr, *XS = nullptr, *T = nullptr, *U1 = nullptr, *U2 = nullptr, *O16 = nullptr;
    void phases(FdatPhaseLayout &p, size_t m, int cp)
    {
        const size_t row = m * cp * 2, rgb_map = m * 16 * 32;
        // G is at offset zero. T must survive G at 12 and U1 at 13. U2 must survive U1 at 14 and O16 at 15.
        const size_t right = std::max(4 * row, rgb_map);
        p.cursor = 0;
        p.take(&IN16, m * 32, {{1,2}}, "IN16");
        p.cursor = 0;
        p.take(&U1, 4 * row, {{13,14}}, "U1");
        p.cursor = 0;
        p.take(&O16, rgb_map, {{15,16}}, "O16");
        p.cursor = right;
        p.take(&T, row, {{12,13}}, "T");
        p.cursor = right;
        p.take(&U2, rgb_map, {{14,15}}, "U2");
    }
};

struct FdatBodyBuffers {
    void *G = nullptr, *X = nullptr, *N1 = nullptr, *QKV = nullptr, *CF = nullptr, *AT = nullptr, *H1 = nullptr, *H2 = nullptr;
    void phases(FdatPhaseLayout &p, size_t m, int cp, int qs, int hs)
    {
        const size_t row = m * cp * 2;
        p.cursor = 0;
        p.take(&G, row, {{3,12}}, "G");
        p.take(&X, row, {{4,11}}, "X");
        // The body clears both maps once per call. Preserve padding that later writers skip,
        // including the CF lanes read by the projected residual.
        p.take(&CF, row, {{3,11}}, "CF");
        p.take(&AT, row, {{3,11}}, "AT");
        const size_t phase = p.cursor;
        p.take(&QKV, m * qs * 2, {{5,7}}, "QKV");
        p.take(&N1, row, {{5,6},{9,9}}, "N1");
        p.cursor = phase;
        p.take(&H1, m * hs * 2, {{9,10}}, "H1");
        p.take(&H2, m * hs * 2, {{10,10}}, "H2");
    }
};
} // namespace ajinative
