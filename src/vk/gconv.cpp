// vk/gconv.cpp — see gconv.h. Module names must match src/vk/shaders/shaders.txt.
#include "vk/gconv.h"

#include "vk/spirv.h"

namespace ajivk {
namespace {

bool fail(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

const int kCmTile[kConvVariants][2] = {{8, 2}, {4, 4}};  // subgroups, rows per subgroup
const int kRows[kConvVariants] = {8, 4};                 // sg16 / plain tile rows

// Shared memory of a gconv module (gconv_common.glsl): max(halo chunk, staging).
uint32_t gshared(const GConvShape &s, int th)
{
    const int kch = s.cinp % 64 == 0 ? 64 : s.cinp % 32 == 0 ? 32 : 16;
    const int halo = (th + 2) * 18 * (kch + 8) / 8, stage = th * 16 * (s.coutp + 8) / 8;
    return (uint32_t)(halo > stage ? halo : stage) * 16;
}

}  // namespace

bool gconv_dispatch(const Device &dev, const GConvShape &s, int v, Dispatch *d, std::string *why)
{
    if (v < 0 || v >= kConvVariants) return fail(why, "no such variant");
    *d = Dispatch{};
    const std::string ops = "_i" + std::to_string(s.cinp) + "_o" + std::to_string(s.coutp) + "_n" + std::to_string(s.in) + "a" +
                            std::to_string(s.act) + "r" + std::to_string(s.nres) + "u" + std::to_string(s.up2) + "o" +
                            std::to_string(s.out);
    switch (dev.family()) {
    case Family::CM16: {
        const int warps = kCmTile[v][0], wr = kCmTile[v][1];
        d->module = "g3_cm16" + ops + "_w" + std::to_string(warps) + "r" + std::to_string(wr);
        d->subgroup = 32;
        d->local_size = (uint32_t)warps * 32;
        d->threads = d->local_size;
        d->tile_h = warps * wr;
        break;
    }
    case Family::SG16:
        d->module = "g3_sg16" + ops + "_t" + std::to_string(kRows[v]) + "r1";
        d->subgroup = 16;
        d->tile_h = kRows[v];
        d->threads = (uint32_t)(16 * (s.coutp / 16) * d->tile_h);
        break;
    case Family::PLAIN:
        d->module = "g3_plain" + ops + "_t" + std::to_string(kRows[v]) + "r1";
        d->subgroup = 0;
        d->tile_h = kRows[v];
        d->threads = (uint32_t)(8 * (s.coutp / 16) * d->tile_h);
        break;
    default:
        return fail(why, "no kernel family");
    }
    d->shared_bytes = gshared(s, d->tile_h);
    if (!spirv_find(d->module.c_str())) return fail(why, "no kernel " + d->module + " in this build");
    if (d->shared_bytes > dev.info().max_shared)
        return fail(why, d->module + " needs " + std::to_string(d->shared_bytes) + " bytes of shared memory");
    if (d->threads > dev.info().max_invocations)
        return fail(why, d->module + " needs workgroups of " + std::to_string(d->threads) + " invocations");
    return true;
}

std::vector<uint16_t> gconv_pack(Family f, const uint16_t *w, int cin, const GConvShape &s, int cout)
{
    return pack_weights(f, w, 9, cin, s.cinp, cout, s.coutp);
}

}  // namespace ajivk
