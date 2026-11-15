// native/fdat_body.h — the FDAT body (traiNNer-redux fdat_arch's residual groups) on the tfm and
// gconv kernels, shared by FDAT (image path) and TFDAT (video path) (internal).
//
// Tokens are NHWC fp16 maps with the channels padded to 16 (CP). Per block:
//   n1 = LN(x); qkv = n1 Wqkv; conv = GELU(dw3x3(n1))
//   spatial: a = window attention (learned bias); cm = cg(mean(conv)); x += (a Wp + bp) * cm + conv
//   channel: stats of q, k over all tokens -> per-head softmax -> folded with Wp into one matrix on the
//            GPU; a = v M + bp; x += a + conv * sigmoid(a . wsg)
//   x += dw3x3(GELU(LN(x) W1)) W2
// and per group, after its blocks, the group conv plus the group's input.
#ifndef AJI_NATIVE_FDAT_BODY_H
#define AJI_NATIVE_FDAT_BODY_H

#include <string>
#include <vector>

#include "native/netres.h"

namespace ajinative {

class FdatBody {
public:
    // The groups under prefix (e.g. "/" or "/model/"): <prefix>groups/groups.<g>/...
    static bool present(const onnxread::Reader &rd, const std::string &prefix);
    bool parse(const onnxread::Reader &rd, const std::string &prefix, int C, std::string *why);
    // Buffers for bw x bh tokens and the group convs' kernels. With scratch (scratch_bytes(bw, bh)
    // bytes, which the caller may use for other things between runs), the large buffers live there.
    bool setup(int bw, int bh, cudaStream_t s, std::string *why, void *scratch = nullptr);
    size_t scratch_bytes(int bw, int bh) const;
    // x0 [M][CP] -> out() [M][CP] at bw x bh (<= the setup size; same kernels, any smaller size).
    cudaError_t run(const void *x0, int bw, int bh, cudaStream_t s, StepProf *prof);
    void *out() const { return G; }
    std::string describe() const;
    size_t device_bytes() const { return res.bytes; }
    size_t weight_bytes() const { return res.weight_bytes; }
    int C = 0, CP = 0;

private:
    struct Block {
        bool spatial = true;
        TMat *qkv = nullptr, *proj = nullptr, *fc1 = nullptr, *fc2 = nullptr;
        float *ln1g = nullptr, *ln1b = nullptr, *ln2g = nullptr, *ln2b = nullptr, *dw = nullptr, *smix = nullptr;
        float *bias = nullptr, *temp = nullptr, *cg1 = nullptr, *cg2 = nullptr, *sg = nullptr, *wp = nullptr;
        float scale = 1.f, eps1 = 1e-5f, eps2 = 1e-5f;
        // channel attention: q, k / max(|q|, clip * N^-pre) over N tokens (the export pre-scales q and k by
        // N^pre, pre = -0.5 or 0, before the L2 norm and its clip)
        float clip = 1.1920928955078125e-07f, pre = 0.f;
        int shift = 0;  // spatial: cyclic window shift (torch.roll by -shift before, +shift after; no mask)
    };
    DevRes res;
    int QS = 0, HS = 0, heads = 0, hd = 0, ws = 8, cgr = 0, ngroups = 0, nblk = 0, maxm = 0;
    std::vector<Block> blocks;
    std::vector<GOp> gconvs;
    TMat foldm;
    void *G = nullptr, *X = nullptr, *N1 = nullptr, *QKV = nullptr, *CF = nullptr, *AT = nullptr, *H1 = nullptr, *H2 = nullptr;
    float *sums = nullptr, *cm = nullptr, *stats = nullptr, *attnb = nullptr, *fold = nullptr, *wsp = nullptr;
    bool shared = false;  // the large buffers are in a caller's scratch (not zeroed between runs)
    void carve(Bump &b, size_t M);
    size_t ws_floats = 0;

public:
    FdatBody() = default;
    FdatBody(const FdatBody &) = delete;
    FdatBody &operator=(const FdatBody &) = delete;
    ~FdatBody() { tmat_release(foldm); }
};

}  // namespace ajinative

#endif
