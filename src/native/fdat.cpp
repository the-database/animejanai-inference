// native/fdat.cpp — FDAT (traiNNer-redux fdat_arch: IllustrationJaNai's FDAT_M/XL) on native kernels:
// conv_first (after the 2x pixel-unshuffle for the 2x models), the body (fdat_body.h), conv_after
// plus the shallow features, and the transpose+conv upsampler (two stride-2 transposed convs as
// phase convs, a 3x3 conv).
#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include "native/fdat_body.h"
#include "native/img_kernels.h"
#include "native/imgnet.h"

namespace ajinative {
namespace {

using namespace onnxread;

class Fdat : public ImgNet {
public:
    static std::unique_ptr<ImgNet> build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why);
    cudaError_t run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s) override;
    int scale() const override { return unshuffle ? 2 : 4; }
    std::string describe() const override { return what; }
    size_t device_bytes() const override { return res.bytes + body.device_bytes(); }
    bool accepts(int w, int h) const override
    {
        return w > 0 && h > 0 && w <= maxw && h <= maxh && (!unshuffle || (w % 2 == 0 && h % 2 == 0));
    }

private:
    DevRes res;
    FdatBody body;
    int C = 0, CP = 0, maxw = 0, maxh = 0;
    bool unshuffle = false;
    GOp first, after, up0[2], up2[2], up3;
    void *IN16 = nullptr, *XS = nullptr, *T = nullptr, *U1 = nullptr, *U2 = nullptr, *O16 = nullptr;
    float *wsp = nullptr;
    size_t ws_floats = 0;
    std::string what;

    // The gconv ops of a run around the body: [0] conv_first, then conv_after and the upsampler.
    void gplan(int bw, int bh, std::vector<std::pair<GOp *, GConvArgs>> *ops);
};

void Fdat::gplan(int bw, int bh, std::vector<std::pair<GOp *, GConvArgs>> *ops)
{
    ops->clear();
    ops->push_back({&first, ga(IN16, 16, bh, bw, XS, CP, bh, bw)});
    GConvArgs a = ga(body.out(), CP, bh, bw, T, CP, bh, bw);
    a.r1 = XS;
    a.r1_stride = CP;
    ops->push_back({&after, a});
    for (int py = 0; py < 2; py++) {
        GConvArgs u = ga(T, CP, bh, bw, U1, CP, bh, bw);
        u.R = 2;
        u.sy = py;
        u.act = GAct::Gelu;
        ops->push_back({&up0[py], u});
    }
    for (int py = 0; py < 2; py++) {
        GConvArgs u = ga(U1, CP, 2 * bh, 2 * bw, U2, 16, 2 * bh, 2 * bw);
        u.R = 2;
        u.sy = py;
        ops->push_back({&up2[py], u});
    }
    ops->push_back({&up3, ga(U2, 16, 4 * bh, 4 * bw, O16, 16, 4 * bh, 4 * bw)});
    for (auto &o : *ops) {
        o.second.ws = wsp;
        o.second.ws_floats = ws_floats;
    }
}

std::unique_ptr<ImgNet> Fdat::build(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    Reader rd(g);
    if (!FdatBody::present(rd, "/")) return refuse(why, "not an FDAT graph"), nullptr;
    std::unique_ptr<Fdat> f(new Fdat);
    Conv cfirst, cafter, cu0, cu2, cu3;
    if (rd.node("/conv_first/conv_first.1/Conv")) {
        if (!rd.conv("/conv_first/conv_first.1/Conv", &cfirst, why)) return nullptr;
        f->unshuffle = true;
        if (cfirst.cin != 12) return refuse(why, "FDAT unshuffle input"), nullptr;
    } else {
        if (!rd.conv("/conv_first/Conv", &cfirst, why)) return nullptr;
        if (cfirst.cin != 3) return refuse(why, "FDAT input channels"), nullptr;
    }
    const int C = f->C = cfirst.cout;
    f->CP = pad16(C);
    if (!rd.conv("/conv_after/Conv", &cafter, why) || !rd.conv("/upsampler/upsampler.0/ConvTranspose", &cu0, why) ||
        !rd.conv("/upsampler/upsampler.2/ConvTranspose", &cu2, why) || !rd.conv("/upsampler/upsampler.3/Conv", &cu3, why))
        return nullptr;
    if (cu0.cin != C || cu0.cout != C || cu2.cin != C || cu2.cout != 3 || cu3.cin != 3 || cu3.cout != 3)
        return refuse(why, "FDAT upsampler shape (expected transpose+conv)"), nullptr;
    if (!f->body.parse(rd, "/", C, why)) return nullptr;
    const int CP = f->CP;
    f->first.w = f->res.gup(conv_taps(cfirst, nullptr), cfirst.b, cfirst.cin, 16, C, CP);
    f->after.w = f->res.gup(conv_taps(cafter, nullptr), cafter.b, C, CP, C, CP);
    for (int py = 0; py < 2; py++) {
        std::vector<uint16_t> w;
        std::vector<float> b;
        phase_weights(cu0, 2, py, CP, &w, &b);
        f->up0[py].w = f->res.gup(w, b, C, CP, 2 * CP, 2 * CP);
        phase_weights(cu2, 2, py, 16, &w, &b);
        f->up2[py].w = f->res.gup(w, b, C, CP, 32, 32);
    }
    f->up3.w = f->res.gup(conv_taps(cu3, nullptr), cu3.b, 3, 16, 3, 16);

    // buffers for the largest tile
    f->maxw = max_w;
    f->maxh = max_h;
    const int bw = f->unshuffle ? max_w / 2 : max_w, bh = f->unshuffle ? max_h / 2 : max_h;
    const size_t M = (size_t)bw * bh, CPb = (size_t)CP * 2;
    if (!f->body.setup(bw, bh, s, why)) return nullptr;
    f->IN16 = f->res.alloc(M * 32);
    f->XS = f->res.alloc(M * CPb);
    f->T = f->res.alloc(M * CPb);
    f->U1 = f->res.alloc(M * 4 * CPb);
    f->U2 = f->res.alloc(M * 16 * 32);
    f->O16 = f->res.alloc(M * 16 * 32);
    if (f->res.failed) return refuse(why, "out of GPU memory"), nullptr;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    for (int sz : {0, 1}) {
        f->gplan(sz ? std::min(bw, 32) : bw, sz ? std::min(bh, 32) : bh, &ops);
        f->ws_floats = std::max(f->ws_floats, gworkspace_max(ops));
    }
    if (f->ws_floats && !(f->wsp = (float *)f->res.alloc(f->ws_floats * 4))) return refuse(why, "out of GPU memory"), nullptr;
    f->gplan(bw, bh, &ops);
    if (!gprepare(ops, s, why)) return nullptr;
    if (cudaStreamSynchronize(s) != cudaSuccess) return refuse(why, "CUDA error during setup"), nullptr;
    f->what = "FDAT " + std::to_string(f->scale()) + "x, " + f->body.describe();
    return std::unique_ptr<ImgNet>(f.release());
}

cudaError_t Fdat::run(const void *in, bool in_f32, void *out, bool out_f32, int w, int h, cudaStream_t s)
{
    if (!accepts(w, h)) return cudaErrorInvalidValue;
    const int bw = unshuffle ? w / 2 : w, bh = unshuffle ? h / 2 : h;
    std::vector<std::pair<GOp *, GConvArgs>> ops;
    gplan(bw, bh, &ops);
    StepProf pr(s);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return e_; } while (0)
    auto gl = [&](size_t i) { return glaunch(*ops[i].first->w, ops[i].second, ops[i].first->variant, s); };
    CK(img_to_nhwc(in, in_f32, w, h, unshuffle, IN16, 16, s));
    pr.mark(P_IO);
    CK(gl(0));  // conv_first -> XS
    pr.mark(P_CONV);
    CK(body.run(XS, bw, bh, s, &pr));
    CK(gl(1));  // conv_after + XS
    pr.mark(P_CONV);
    for (size_t i = 2; i < ops.size(); i++) CK(gl(i));
    pr.mark(P_UP);
    CK(img_from_nhwc(O16, 16, 4 * bw, 4 * bh, out, out_f32, s));
    pr.mark(P_IO);
#undef CK
    char label[48];
    snprintf(label, sizeof(label), "fdat %dx%d", w, h);
    pr.report(label);
    return cudaSuccess;
}

}  // namespace

std::unique_ptr<ImgNet> build_fdat(const OnnxGraph &g, int max_w, int max_h, cudaStream_t s, std::string *why)
{
    return Fdat::build(g, max_w, max_h, s, why);
}

}  // namespace ajinative
