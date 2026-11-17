// native/rife_plan.cpp — see rife_plan.h. The checks are native/rife.cpp's, moved here.
#include "native/rife_plan.h"

#include <cmath>

namespace ajinative {

using namespace onnxread;

bool rife_plan(const OnnxGraph &g, int W, int H, RifePlan *p, std::string *why)
{
    *p = RifePlan{};
    if (W % 64 || H % 64 || W < 64 || H < 64) return refuse(why, "size must be a multiple of 64");
    Reader rd(g);
    // input layout: Split(input, [6, 1, 2, 2])
    const OnnxNode *sp = rd.node("/Split");
    std::vector<int64_t> parts;
    if (sp && sp->in.size() > 1 && rd.tensor(sp->in[1])) rd.tensor(sp->in[1])->ints(&parts);
    if (sp && parts.empty() && sp->attr("split")) parts = sp->attr("split")->ints;
    if (!sp || g.inputs.size() != 1 || sp->in[0] != g.inputs[0]) return refuse(why, "not the vs-mlrt RIFE input layout");
    p->ens = parts == std::vector<int64_t>{3, 3, 1, 2, 2};
    if (!p->ens && parts != std::vector<int64_t>{6, 1, 2, 2}) return refuse(why, "not the vs-mlrt RIFE input layout");

    // encoder: the /encode/ convs in file order
    for (const OnnxNode &n : g.nodes)
        if (n.name.rfind("/encode/", 0) == 0 && (n.op == "Conv" || n.op == "ConvTranspose")) {
            Conv c;
            if (!rd.conv(n.name, &c, why)) return false;
            p->enc.push_back(c);
        }
    if (!p->enc.empty()) {
        const std::vector<Conv> &enc = p->enc;
        if (enc.front().transposed || enc.front().stride != 2 || enc.front().cin != 3 || !enc.back().transposed)
            return refuse(why, "unexpected encoder");
        for (size_t i = 1; i + 1 < enc.size(); i++)
            if (enc[i].transposed || enc[i].stride != 1) return refuse(why, "unexpected encoder");
        for (size_t i = 1; i < enc.size(); i++)
            if (enc[i].cin != enc[i - 1].cout) return refuse(why, "encoder channels");
        p->F = enc.back().cout;
        if (p->F > 16 || enc.back().lrelu) return refuse(why, "encoder output");
    }
    const int F = p->F;

    int nb = 0;
    while (rd.node("/block" + std::to_string(nb) + "/conv0/conv0.0/conv0.0.0/Conv")) nb++;
    if (nb < 1 || nb > kRifeMaxBlocks) return refuse(why, "unexpected number of blocks");
    int prevK = 0;
    for (int b = 0; b < nb; b++) {
        const std::string pre = "/block" + std::to_string(b) + "/";
        RifeBlockPlan bp;
        if (!rd.conv(pre + "conv0/conv0.0/conv0.0.0/Conv", &bp.c0a, why) || !rd.conv(pre + "conv0/conv0.1/conv0.1.0/Conv", &bp.c0b, why) ||
            !rd.conv(pre + "lastconv/lastconv.0/ConvTranspose", &bp.last, why))
            return false;
        const Conv &c0a = bp.c0a, &c0b = bp.c0b, &last = bp.last;
        if (c0a.stride != 2 || c0b.stride != 2 || !c0a.lrelu || !c0b.lrelu || c0b.cin != c0a.cout || last.lrelu)
            return refuse(why, pre + "conv0");
        const int c = c0b.cout;
        for (int k = 0;; k++) {
            const std::string rp = pre + "convblock/convblock." + std::to_string(k) + "/";
            if (!rd.node(rp + "conv/Conv")) break;
            Conv rc;
            if (!rd.conv(rp + "conv/Conv", &rc, why)) return false;
            const OnnxNode *mul = rd.node(rp + "Mul"), *add = rd.node(rp + "Add"), *act = rd.node(rp + "relu/LeakyRelu");
            std::vector<float> bt;
            if (!mul || !add || !act || mul->in.size() != 2 || !rd.floats(mul->in[1], &bt) || (int)bt.size() != c || rc.cin != c ||
                rc.cout != c || rc.stride != 1 || rc.lrelu)
                return refuse(why, rp + ": not conv * beta + x, LeakyReLU");
            bp.res.push_back(rc);
            bp.beta.push_back(bt);
        }
        if (last.cin != c || last.cout % 4 || last.cout / 4 < 5) return refuse(why, pre + "lastconv");
        const OnnxNode *d2s = rd.node(pre + "lastconv/lastconv.1/DepthToSpace");
        if (!d2s || d2s->attr_i("blocksize", 0) != 2 || d2s->attr_s("mode", "DCR") != "CRD")
            return refuse(why, pre + "lastconv depth-to-space");
        bp.K = last.cout / 4;
        // scale from the block's input Resize (absent: 1)
        int s = 1;
        if (const OnnxNode *rz = rd.node(pre + "Resize")) {
            std::vector<float> sc;
            if (rz->in.size() < 3 || !rd.floats(rz->in[2], &sc) || sc.size() != 4 || sc[2] != sc[3] || sc[2] <= 0)
                return refuse(why, pre + "Resize scales");
            if (rz->attr_s("mode", "") != "linear" || rz->attr_s("coordinate_transformation_mode", "") != "half_pixel")
                return refuse(why, pre + "Resize mode");
            s = (int)std::lround(1.0 / sc[2]);
            if (s < 2 || s > 64 || (s & (s - 1)) || std::fabs(sc[2] * s - 1.f) > 1e-6f) return refuse(why, pre + "scale");
        }
        if (W % (4 * s) || H % (4 * s)) return refuse(why, "size not divisible by a block scale");
        bp.s = s;
        // input channels and whether the previous block's features are fed in
        const int base0 = 6 + 2 * F + 1, base = base0 + 1 + 4;
        if (b == 0) {
            if (c0a.cin != base0) return refuse(why, "block 0 input channels");
        } else if (c0a.cin == base + prevK - 5 && prevK > 5) {
            bp.feeds = true;
        } else if (c0a.cin != base) {
            return refuse(why, pre + "input channels");
        }
        if (!p->blocks.empty() && s > p->blocks.back().s) return refuse(why, "block scales must not increase");
        if (p->ens && (bp.feeds || bp.K > 8)) return refuse(why, "ensemble with fed block features");
        if (bp.K > 16) return refuse(why, pre + "too many output channels");
        prevK = bp.K;
        p->blocks.push_back(std::move(bp));
    }
    if (p->blocks.back().s != 1) return refuse(why, "the last block must run at full resolution");
    return true;
}

}  // namespace ajinative
