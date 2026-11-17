// vk/gconv.cpp — see gconv.h. Module names must match src/vk/shaders/shaders.txt.
#include "vk/gconv.h"

#include <algorithm>
#include <cstdlib>
#include <map>

#include "vk/spirv.h"

namespace ajivk {
namespace {

bool fail(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

// Tiles per family and stride, and the most input channels a halo chunk holds (KMAX).
struct Tile {
    const char *name;
    int a, b;  // cm16: subgroups, rows per subgroup; sg16 / plain: rows, rows per group
};
const Tile kCm1[kConvVariants] = {{"w8r2", 8, 2}, {"w4r4", 4, 4}}, kCm2[kConvVariants] = {{"w4r2", 4, 2}, {"w2r4", 2, 4}};
const Tile kSg[kConvVariants] = {{"t8r1", 8, 1}, {"t4r1", 4, 1}};

}  // namespace

bool gconv_dispatch(const Device &dev, const GConvShape &s, int cinp, int v, Dispatch *d, int *kch, std::string *why)
{
    if (v < 0 || v >= kConvVariants) return fail(why, "no such variant");
    if (cinp < 16 || cinp % 16) return fail(why, "input channels not a multiple of 16");
    *d = Dispatch{};
    const std::string ops = "_o" + std::to_string(s.coutp) + "_n" + std::to_string(s.in) + "a" + std::to_string(s.act) + "r" +
                            std::to_string(s.nres) + "u" + std::to_string(s.up2) + "s" + std::to_string(s.stride) + "o" +
                            std::to_string(s.out);
    int kmax = 64;
    switch (dev.family()) {
    case Family::CM16: {
        const Tile &t = (s.stride == 2 ? kCm2 : kCm1)[v];
        d->module = std::string("g3_cm16") + ops + "_" + t.name;
        d->subgroup = 32;
        d->local_size = (uint32_t)t.a * 32;
        d->threads = d->local_size;
        d->tile_h = t.a * t.b;
        kmax = s.stride == 2 ? 32 : 64;
        break;
    }
    case Family::SG16:
    case Family::PLAIN: {
        const Tile &t = kSg[v];
        const bool sg = dev.family() == Family::SG16;
        d->module = std::string(sg ? "g3_sg16" : "g3_plain") + ops + "_" + t.name;
        d->subgroup = sg ? 16 : 0;
        d->tile_h = t.a;
        d->threads = (uint32_t)((sg ? 16 : 8) * (s.coutp / 16) * t.a);
        kmax = s.stride == 2 ? 16 : 64;
        break;
    }
    default:
        return fail(why, "no kernel family");
    }
    // the chunk: the largest of 64 / 32 / 16 within KMAX that divides cinp (NCHW inputs: 16)
    *kch = 16;
    for (int k : {64, 32})
        if (k <= kmax && cinp % k == 0 && s.in == 0) {
            *kch = k;
            break;
        }
    const int hr = (d->tile_h - 1) * s.stride + 3, hc = 15 * s.stride + 3;
    const int halo = hr * hc * (kmax + 8) / 8, stage = d->tile_h * 16 * (s.coutp + 8) / 8;
    d->shared_bytes = (uint32_t)(halo > stage ? halo : stage) * 16;
    if (!spirv_find(d->module.c_str())) return fail(why, "no kernel " + d->module + " in this build");
    if (d->shared_bytes > dev.info().max_shared)
        return fail(why, d->module + " needs " + std::to_string(d->shared_bytes) + " bytes of shared memory");
    if (d->threads > dev.info().max_invocations)
        return fail(why, d->module + " needs workgroups of " + std::to_string(d->threads) + " invocations");
    return true;
}

std::vector<uint16_t> gconv_pack(Family f, const uint16_t *w, int cin, int cinp, int cout, int co0, int coutp)
{
    // the slice of output channels [co0, co0 + coutp), as a conv of its own
    const int n = cout - co0 < coutp ? cout - co0 : coutp;
    std::vector<uint16_t> sub((size_t)9 * cin * n);
    for (int t = 0; t < 9; t++)
        for (int i = 0; i < cin; i++)
            for (int o = 0; o < n; o++) sub[((size_t)t * cin + i) * n + o] = w[((size_t)t * cin + i) * cout + co0 + o];
    return pack_weights(f, sub.data(), 9, cin, cinp, n, coutp);
}

std::vector<uint16_t> gconv_pack_slices(Family f, const uint16_t *w, int cin, int cinp, int cout, int coutp, int nz)
{
    std::vector<uint16_t> all;
    for (int z = 0; z < nz; z++) {
        const std::vector<uint16_t> p = gconv_pack(f, w, cin, cinp, cout, z * coutp, coutp);
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<uint16_t> gconv_bias_slices(const float *b, int cout, int coutp, int nz)
{
    std::vector<float> v((size_t)nz * coutp, 0.f);
    for (int o = 0; o < cout && o < nz * coutp; o++) v[o] = b[o];
    return pack_vector(v.data(), nz * coutp, nz * coutp);
}

bool gconv_prepare(const Device &dev, GConvOp *o, std::string *why)
{
    o->nvar = 0;
    for (int v = 0; v < kConvVariants; v++) {
        std::string w;
        int kch = 16;
        if (!gconv_dispatch(dev, o->s, o->p.cinp, v, &o->disp[v], &kch, &w)) {
            if (v == 0) return fail(why, w);
            break;
        }
        o->p.kch = kch;  // the same for every variant of a family and stride
        o->nvar = v + 1;
    }
    return true;
}

void gconv_record(const Device &dev, VkCommandBuffer cb, const GConvOp &o, int v)
{
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, o.pipe[v]);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev.pipeline_layout(), 0, 1, &o.set, 0, nullptr);
    const int32_t pc[4] = {o.entry, 0, 0, 0};
    vkCmdPushConstants(cb, dev.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    const Dispatch &d = o.disp[v];
    vkCmdDispatch(cb, (uint32_t)((o.p.W + 15) / 16), (uint32_t)((o.p.H + d.tile_h - 1) / d.tile_h), (uint32_t)o.nz);
}

bool gconv_choose(Device &dev, const std::vector<GConvOp *> &ops, std::string *why)
{
    const char *fix = std::getenv("AJI_VK_VARIANT");
    const bool can_time = dev.info().timestamp_ns > 0;
    std::map<std::string, int> picked;
    VkQueryPool qp = VK_NULL_HANDLE;
    if (can_time && !fix) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        vkCreateQueryPool(dev.device(), &qpi, nullptr, &qp);
    }
    bool ok = true;
    for (GConvOp *o : ops) {
        const std::string key = "g3:" + o->disp[0].module + ":" + std::to_string(o->p.W) + "x" + std::to_string(o->p.H) +
                                (o->nz > 1 ? "z" + std::to_string(o->nz) : "");
        int v = fix ? std::min(std::max(std::atoi(fix), 0), o->nvar - 1) : -1;
        if (v < 0) {
            auto it = picked.find(key);
            if (it != picked.end()) v = it->second;
            else if ((v = dev.cached_variant(key)) >= o->nvar) v = -1;
        }
        const int lo = v >= 0 ? v : 0, hi = v >= 0 ? v + 1 : o->nvar;
        for (int k = lo; k < hi && ok; k++) {
            o->pipe[k] = dev.pipeline(o->disp[k].module.c_str(), o->disp[k].local_size, o->disp[k].subgroup, why);
            ok = o->pipe[k] != VK_NULL_HANDLE;
        }
        if (!ok) break;
        if (v < 0) {
            v = 0;
            if (qp && o->nvar > 1) {
                double best = -1;
                for (int k = 0; k < o->nvar && ok; k++) {
                    VkCommandBuffer cb = dev.begin();
                    vkCmdResetQueryPool(cb, qp, 0, 2);
                    gconv_record(dev, cb, *o, k);  // warm-up
                    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                    for (int r = 0; r < 3; r++) gconv_record(dev, cb, *o, k);
                    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                    if (!(ok = dev.submit_wait(cb, why))) break;
                    uint64_t t[2];
                    vkGetQueryPoolResults(dev.device(), qp, 0, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                    const double ms = (double)(t[1] - t[0]);
                    if (best < 0 || ms < best) {
                        best = ms;
                        v = k;
                    }
                }
                if (!ok) break;
            }
            dev.store_variant(key, v);
        }
        picked[key] = v;
        o->variant = v;
    }
    if (qp) vkDestroyQueryPool(dev.device(), qp, nullptr);
    return ok;
}

}  // namespace ajivk
