// vk/kernels.cpp — see kernels.h. Module names must match src/vk/shaders/shaders.txt.
#include "vk/kernels.h"

#include "native/pack.h"
#include "vk/spirv.h"

namespace ajivk {
namespace {

bool fail(std::string *why, const std::string &m)
{
    if (why) *why = m;
    return false;
}

// cm16 tiles: (subgroups, rows per subgroup); sg16 tiles: rows (one per subgroup).
const int kCmTile[kConvVariants][2] = {{8, 2}, {4, 4}};
const int kSgRows[kConvVariants] = {8, 4};

// Shared memory of a conv3x3 module (conv_common.glsl): max(halo, staging + extra).
uint32_t conv_shared(const ConvKey &k, int th, int extra_uvec4)
{
    const int halo = (th + 2) * 18 * (k.cinp + 8) / 8, stage = th * 16 * (k.coutp + 8) / 8 + extra_uvec4;
    return (uint32_t)(halo > stage ? halo : stage) * 16;
}

bool finish(const Device &dev, Dispatch *d, std::string *why)
{
    if (!spirv_find(d->module.c_str())) return fail(why, "no kernel " + d->module + " in this build");
    if (d->shared_bytes > dev.info().max_shared)
        return fail(why, d->module + " needs " + std::to_string(d->shared_bytes) + " bytes of shared memory");
    if (d->threads > dev.info().max_invocations)
        return fail(why, d->module + " needs workgroups of " + std::to_string(d->threads) + " invocations");
    return true;
}

}  // namespace

const char *conv_variant_name(Family f, int v)
{
    static const char *cm[kConvVariants] = {"8 subgroups x 2 rows", "4 subgroups x 4 rows"};
    static const char *sg[kConvVariants] = {"8 rows", "4 rows"};
    if (v < 0 || v >= kConvVariants) return "";
    return f == Family::CM16 ? cm[v] : f == Family::SG16 ? sg[v] : "";
}

bool conv_dispatch(const Device &dev, const ConvKey &k, int v, Dispatch *d, std::string *why)
{
    if (v < 0 || v >= kConvVariants) return fail(why, "no such variant");
    *d = Dispatch{};
    const std::string ops = "_i" + std::to_string(k.cinp) + "_o" + std::to_string(k.coutp) + "_n" + std::to_string(k.in) + "a" +
                            std::to_string(k.act) + "g" + std::to_string(k.gate) + "o" + std::to_string(k.out);
    switch (dev.family()) {
    case Family::CM16: {
        const int warps = kCmTile[v][0], wr = kCmTile[v][1];
        d->module = "c3_cm16" + ops + "_w" + std::to_string(warps) + "r" + std::to_string(wr);
        d->subgroup = 32;
        d->local_size = (uint32_t)warps * d->subgroup;
        d->threads = d->local_size;
        d->tile_h = warps * wr;
        d->shared_bytes = conv_shared(k, d->tile_h, 0);
        break;
    }
    case Family::SG16:
        d->module = "c3_sg16" + ops + "_t" + std::to_string(kSgRows[v]) + "r1";
        d->subgroup = 16;
        d->tile_h = kSgRows[v];
        d->threads = (uint32_t)(16 * (k.coutp / 16) * d->tile_h);  // a subgroup per row and 16-channel group
        d->shared_bytes = conv_shared(k, d->tile_h, 0);
        break;
    default:
        return fail(why, "no kernel family");
    }
    return finish(dev, d, why);
}

bool concat_dispatch(const Device &dev, int c, int nsrc, Dispatch *d, std::string *why)
{
    *d = Dispatch{};
    const std::string base = "_c" + std::to_string(c) + "_s" + std::to_string(nsrc);
    d->tile_px = 64;
    d->shared_bytes = (uint32_t)(nsrc * 64 * (c + 8) * 2);
    switch (dev.family()) {
    case Family::CM16:
        d->module = "cat_cm16" + base + "_w4";
        d->subgroup = 32;
        d->local_size = 4 * d->subgroup;
        d->threads = d->local_size;
        break;
    case Family::SG16:
        d->module = "cat_sg16" + base + "_b4";
        d->subgroup = 16;
        d->threads = (uint32_t)(16 * (c / 16) * 4);
        break;
    default:
        return fail(why, "no kernel family");
    }
    return finish(dev, d, why);
}

std::vector<uint16_t> pack_weights(Family f, const uint16_t *w, int taps, int cin, int cin_pad, int cout, int cout_pad)
{
    std::vector<uint16_t> out((size_t)taps * cin_pad * cout_pad, 0);
    if (f == Family::SG16) {
        const int KC = cin_pad / 16, NG = cout_pad / 16;
        for (int t = 0; t < taps; t++)
            for (int kc = 0; kc < KC; kc++)
                for (int g = 0; g < NG; g++)
                    for (int l = 0; l < 16; l++)
                        for (int k = 0; k < 16; k++) {
                            const int ci = kc * 16 + k, co = g * 16 + l;
                            if (ci < cin && co < cout)
                                out[((((size_t)t * KC + kc) * NG + g) * 16 + l) * 16 + k] = w[((size_t)t * cin + ci) * cout + co];
                        }
    } else {
        for (int t = 0; t < taps; t++)
            for (int ci = 0; ci < cin; ci++)
                for (int co = 0; co < cout; co++)
                    out[((size_t)t * cin_pad + ci) * cout_pad + co] = w[((size_t)t * cin + ci) * cout + co];
    }
    return out;
}

std::vector<uint16_t> pack_vector(const float *v, int n, int npad)
{
    std::vector<uint16_t> out(npad, 0);
    for (int i = 0; i < n; i++) out[i] = ajinative::float_to_half_bits(v[i]);
    return out;
}

}  // namespace ajivk
