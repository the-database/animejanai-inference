// vk/fdat_body.h — the FDAT body (its residual groups of spatial / channel blocks and group convs)
// on vk/tnet, shared by vk/fdat.cpp and vk/tfdat.cpp as native/fdat_body.h is by their CUDA
// counterparts.
#ifndef AJI_VK_FDAT_BODY_H
#define AJI_VK_FDAT_BODY_H

#include <string>
#include <vector>

#include "native/onnx_read.h"
#include "vk/tnet.h"

namespace ajivk {

// One block's weights as the CUDA parse reads them (fdat_body.cpp), on the host.
struct FdatBlock {
    bool spatial = true;
    std::vector<float> ln1g, ln1b, ln2g, ln2b, qkv, wp, bp, dw, bias, cg1, cg2, temp, sg, fc1, fc2, smix;
    float scale = 1.f, eps1 = 1e-5f, eps2 = 1e-5f, clip = 1.1920928955078125e-07f, pre = 0.f;
    int shift = 0;
};

struct FdatBody {
    int C = 0, CP = 0, QS = 0, HS = 0, heads = 0, hd = 0, ws = 8, cgr = 0, ngroups = 0, nblk = 0;
    std::vector<FdatBlock> blocks;
    std::vector<ajinative::onnxread::Conv> gconvs;
};

// The body under prefix (groups/groups.N/...) of a C-channel model.
bool fdat_parse_body(const ajinative::onnxread::Reader &rd, const std::string &prefix, int C, FdatBody *b, std::string *why);
// Records the body on G (fp16 [bh * bw][CP] in n's arena) in place; its scratch is placed in the
// arena, or taken from scratch (fdat_body_scratch bytes).
void fdat_record_body(TNet &n, const FdatBody &B, VkDeviceSize G, int bh, int bw, TCarve *scratch = nullptr);
VkDeviceSize fdat_body_scratch(const TNet &n, const FdatBody &B, int bh, int bw);

}  // namespace ajivk

#endif
