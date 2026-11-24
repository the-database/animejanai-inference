// native/tail.cpp — see tail.h.
#include "native/tail.h"

#include <algorithm>
#include <cstdint>

#include "native/img_kernels.h"

namespace ajinative {

int Tail::scale_after(int k) const
{
    int sc = 1;
    for (int i = 0; i <= k; i++)
        if (steps_[i].ps || steps_[i].up2) sc *= 2;
    return sc;
}

size_t Tail::bytes(int k, int bw, int rows) const
{
    const size_t sc = (size_t)scale_after(k);
    return sc * sc * bw * rows * steps_[k].out_stride * 2;
}

void Tail::plan(const void *src, int src_stride, int bw, int t0, int tn, std::vector<std::pair<GOp *, GConvArgs>> *ops) const
{
    const void *in = (const uint16_t *)src + (size_t)t0 * bw * src_stride;
    int is = src_stride, sc = 1;
    for (const Step &st : steps_) {
        if (st.ps) {
            for (int py = 0; py < 2; py++) {
                GConvArgs a = ga(in, is, sc * tn, sc * bw, st.dst, st.out_stride, sc * tn, sc * bw);
                a.R = 2;
                a.sy = py;
                a.act = st.act;
                a.slope = st.slope;
                ops->push_back({st.op[py], a});
            }
            sc *= 2;
        } else {
            if (st.up2) sc *= 2;
            GConvArgs a = ga(in, is, sc * tn, sc * bw, st.dst, st.out_stride, sc * tn, sc * bw);
            a.up2 = st.up2;
            a.act = st.act;
            a.slope = st.slope;
            ops->push_back({st.op[0], a});
        }
        in = st.dst;
        is = st.out_stride;
    }
}

void Tail::band(int b0, int b1, int bh, int *a, int *z)
{
    *a = std::max(0, b0 - kHalo);
    *z = std::min(bh, b1 + kHalo);
}

cudaError_t Tail::emit(int a, int b0, int b1, int bw, void *out, bool out_f32, bool clamp, cudaStream_t s) const
{
    const int S = scale();
    const uint16_t *src = (const uint16_t *)steps_.back().dst + (size_t)(b0 - a) * S * S * bw * 16;
    return img_from_nhwc(src, 16, S * bw, (b1 - b0) * S, out, out_f32, s, clamp);
}

}  // namespace ajinative
