#pragma once
#include "native/gconv_workspace.h"

namespace ajinative {
struct HatBuffers {
    void *IN16 = nullptr, *XF = nullptr, *N1 = nullptr, *QKV = nullptr, *AT = nullptr, *T60 = nullptr, *Y = nullptr, *H1 = nullptr;
    void *S16 = nullptr, *TT = nullptr, *U0 = nullptr, *V1 = nullptr, *V2 = nullptr, *O16 = nullptr;
    float *S = nullptr, *GIN = nullptr;

    void declare(LifetimePlan &lt, size_t M, int CP, int crp, int QS, int HS, int nup)
    {
        lt.add(&IN16, M * 32, {{1, 2}});
        lt.add(&XF, M * CP * 2, {{2, 31}});
        lt.add((void **)&S, M * CP * 4, {{3, 30}});
        lt.add((void **)&GIN, M * CP * 4, {{9, 28}});
        lt.add(&N1, M * CP * 2, {{10, 14}, {17, 19}, {20, 21}, {24, 25}, {30, 31}});  // 19: the MLP makes the next LN1
        lt.add(&T60, M * crp * 2, {{11, 12}});
        lt.add(&Y, M * CP * 2, {{12, 16}});
        lt.add(&QKV, M * QS * 2, {{14, 15}, {21, 22}});
        lt.add(&AT, M * CP * 2, {{15, 16}, {22, 23}});
        lt.add(&H1, M * HS * 2, {{18, 19}, {25, 26}});
        lt.add(&S16, M * CP * 2, {{27, 28}});
        lt.add(&TT, M * CP * 2, {{31, 32}});
        // (conv_last reads the last of U0 / V1 / V2 at step 35)
        lt.add(&U0, M * 64 * 2, {{32, nup == 0 ? 35 : 33}});
        const size_t S2 = (size_t)1 << (2 * nup);  // output pixels per input pixel
        if (nup >= 1) lt.add(&V1, 4 * M * 64 * 2, {{33, nup == 1 ? 35 : 34}});
        if (nup >= 2) lt.add(&V2, 16 * M * 64 * 2, {{34, 35}});
        lt.add(&O16, S2 * M * 16 * 2, {{35, 36}});
    }
};
} // namespace ajinative
