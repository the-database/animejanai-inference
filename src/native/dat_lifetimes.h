#pragma once
#include "native/gconv_workspace.h"

namespace ajinative {
struct DatBuffers {
    void *IN16 = nullptr, *XF = nullptr, *N1 = nullptr, *QKV = nullptr, *AT = nullptr, *CX = nullptr, *Z = nullptr, *H1 = nullptr, *TG = nullptr;
    void *N2 = nullptr, *PR = nullptr, *S16 = nullptr, *TT = nullptr, *U0 = nullptr, *V1 = nullptr, *V2 = nullptr, *O16 = nullptr;
    float *LST = nullptr;  // the SGFN's LN3 statistics per token (fc1's epilogue -> the gated depthwise GEMM)
    float *S = nullptr, *GIN = nullptr;

    // Both attention branches finish reading QKV at 13. Step 14 mixes AT and CX.
    void declare(LifetimePlan &lt, size_t M, int CP, int QS, int FS)
    {
        lt.add(&IN16, M * 32, {{1, 2}});
        lt.add(&XF, M * CP * 2, {{2, 31}});
        lt.add((void **)&S, M * CP * 4, {{3, 30}});
        lt.add((void **)&GIN, M * CP * 4, {{9, 22}});
        lt.add(&N1, M * CP * 2, {{10, 11}, {16, 17}, {30, 31}});
        lt.add(&QKV, M * QS * 2, {{11, 13}}, "QKV");
        lt.add(&CX, M * CP * 2, {{12, 14}});
        lt.add(&AT, M * CP * 2, {{13, 14}});
        lt.add(&Z, M * CP * 2, {{14, 15}});
        lt.add(&TG, M * 16 * 2, {{13, 14}});  // the token gate's hidden layer
        lt.add(&H1, M * FS * 2, {{17, 19}});
        lt.add(&N2, M * CP * 2, {{18, 19}});
        lt.add((void **)&LST, M * 8, {{17, 19}});
        lt.add(&PR, M * CP * 2, {{19, 20}});
        lt.add(&S16, M * CP * 2, {{21, 22}});
        lt.add(&TT, M * CP * 2, {{31, 32}});
        lt.add(&U0, M * 64 * 2, {{32, 33}});
        lt.add(&V1, 4 * M * 64 * 2, {{33, 34}});
        lt.add(&V2, 16 * M * 64 * 2, {{34, 35}});
        lt.add(&O16, 16 * M * 16 * 2, {{35, 36}});
    }
};
} // namespace ajinative
