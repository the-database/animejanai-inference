// aji_vk_gconv_test: the general conv (vk/gconv.h) in each mode the networks use, against a CPU
// reference on random data: channel prefixes and slices of wider buffers, runtime input channels
// (every halo chunk size), stride 2, NCHW planes at an offset, residuals, LeakyReLU, and the
// phase scatters (R = 2, 4) of the transposed convs, GELU, output slices. Each tile variant, at a
// ragged size.
// Fails above --max-err (default 0.02: fp16 sums, outputs of order 1).
//   aji_vk_gconv_test [--device N]   (AJI_VK_FAMILY picks the kernel family)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "native/pack.h"
#include "vk/device.h"
#include "vk/gconv.h"
#include "vk/kernels.h"

using namespace ajinative;
using namespace ajivk;

namespace {

uint32_t g_seed = 0x2468aceu;
float urand(float lo, float hi)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((g_seed >> 8) & 0xffffff) / 16777215.f;
}
float rh(float v) { return half_bits_to_float(float_to_half_bits(v)); }

struct Case {
    const char *name;
    GConvShape s;
    int cin, cout;     // the conv's real channels (cout: of this dispatch, R * CHP with phases)
    int cinp;          // input channels read
    int IH, IW;        // input size
    int src_stride, src_off, dst_stride, dst_off;
    float c1;          // residual scale
    int nz = 1;        // output slices of s.coutp (cout = nz * coutp)
};

// One case: random input / weights / residual, the dispatch per variant, the CPU result.
bool run_case(Device &dev, const Case &k, double max_err, int *runs)
{
    std::string why;
    const GConvShape &s = k.s;
    const int S = s.stride, R = s.out == 4 ? 2 : s.out == 5 ? 4 : 1;
    const int H = k.IH / S, W = k.IW / S;  // the conv's output size (phases: before the scatter)
    const int nin = s.in == 1 ? 3 : k.cin;
    // input: NHWC [IH][IW][src_stride] at src_off, or three NCHW planes at src_off
    const size_t in_len = s.in == 1 ? (size_t)k.src_off + 3 * k.IH * k.IW : (size_t)k.IH * k.IW * k.src_stride;
    std::vector<float> in(in_len);
    for (float &v : in) v = rh(urand(-1.f, 1.f));
    const auto x_at = [&](int y, int x, int c) -> float {
        if (y < 0 || x < 0 || y >= k.IH || x >= k.IW || c >= nin) return 0.f;
        if (s.in == 1) return in[(size_t)k.src_off + ((size_t)c * k.IH + y) * k.IW + x];
        return in[((size_t)y * k.IW + x) * k.src_stride + c];
    };
    const float ws = 1.5f / std::sqrt(9.f * nin);
    std::vector<uint16_t> taps((size_t)9 * nin * k.cout);
    std::vector<float> wf(taps.size()), bias(k.cout);
    for (size_t i = 0; i < taps.size(); i++) {
        taps[i] = float_to_half_bits(urand(-ws, ws));
        wf[i] = half_bits_to_float(taps[i]);
    }
    for (float &b : bias) b = rh(urand(-0.3f, 0.3f));
    // residual: NHWC [H][W][dst_stride] at dst_off, as the networks pass it
    const int hasr = s.nres > 0;
    std::vector<float> res(hasr ? (size_t)H * W * k.dst_stride : 0);
    for (float &v : res) v = rh(urand(-1.f, 1.f));
    // reference
    const int CHP = k.cout / R, DH = H * R, DW = W * R;
    const size_t out_len = (size_t)DH * DW * k.dst_stride;
    std::vector<float> ref(out_len, 0.f);
    std::vector<char> mine(out_len, 0);
    const int sy = 0;  // set per phase below
    (void)sy;
    std::vector<float> acc(k.cout);
    for (int phase = 0; phase < R; phase++) {
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                for (int o = 0; o < k.cout; o++) acc[o] = 0.f;
                for (int t = 0; t < 9; t++)
                    for (int i = 0; i < nin; i++) {
                        const float xv = x_at(y * S + t / 3 - 1, x * S + t % 3 - 1, i);
                        if (xv == 0.f) continue;
                        const float *wr = &wf[((size_t)t * nin + i) * k.cout];
                        for (int o = 0; o < k.cout; o++) acc[o] += xv * wr[o];
                    }
                for (int o = 0; o < k.cout; o++) {
                    float v = acc[o] + bias[o];
                    if (hasr) v += k.c1 * res[((size_t)y * W + x) * k.dst_stride + k.dst_off + o];
                    if (s.act == 2 && v < 0) v *= 0.2f;
                    if (s.act == 1) v = 0.5f * v * (1.f + std::erf(v / 1.4140625f));  // GELU, as CUDA's
                    size_t d;
                    if (R > 1) {
                        const int sx = o / CHP, ch = o % CHP;
                        d = ((size_t)(R * y + phase) * DW + R * x + sx) * k.dst_stride + k.dst_off + ch;
                    } else {
                        d = ((size_t)y * W + x) * k.dst_stride + k.dst_off + o;
                    }
                    ref[d] = v;  // the phase weights are the same per phase here: one conv per row phase
                    mine[d] = 1;
                }
            }
    }
    // device buffers
    Buffer bin, bout, bres, bw, bb, bp;
    std::vector<uint16_t> hin(in.size());
    for (size_t i = 0; i < in.size(); i++) hin[i] = float_to_half_bits(in[i]);
    const std::vector<uint16_t> pw = gconv_pack_slices(dev.family(), taps.data(), nin, k.cinp, k.cout, s.coutp, k.nz);
    const std::vector<uint16_t> pb = gconv_bias_slices(bias.data(), k.cout, s.coutp, k.nz);
    std::vector<uint16_t> hres(res.size());
    for (size_t i = 0; i < res.size(); i++) hres[i] = float_to_half_bits(res[i]);
    const uint16_t sentinel = 0x7e00;  // NaN: words the conv must not write stay NaN
    std::vector<uint16_t> init(out_len, sentinel);
    GConvOp op;
    op.s = s;
    op.nz = k.nz;
    if (!dev.make_buffer(&bin, hin.size() * 2 + 64, false, &why) || !dev.make_buffer(&bout, out_len * 2, false, &why) ||
        !dev.make_buffer(&bw, pw.size() * 2, false, &why) || !dev.make_buffer(&bb, pb.size() * 2, false, &why) ||
        !dev.make_buffer(&bp, sizeof(GConvParams) * R, false, &why) || (hasr && !dev.make_buffer(&bres, hres.size() * 2, false, &why)) ||
        !dev.upload(bin, hin.data(), hin.size() * 2, &why) || !dev.upload(bw, pw.data(), pw.size() * 2, &why) ||
        !dev.upload(bb, pb.data(), pb.size() * 2, &why) || (hasr && !dev.upload(bres, hres.data(), hres.size() * 2, &why))) {
        std::printf("%s: buffers: %s\n", k.name, why.c_str());
        return false;
    }
    std::vector<GConvParams> prm(R);
    for (int p = 0; p < R; p++) {
        GConvParams &q = prm[p];
        q.H = H;
        q.W = W;
        q.IH = k.IH;
        q.IW = k.IW;
        q.cinp = k.cinp;
        q.src_off = k.src_off;
        q.src_stride = k.src_stride;
        q.dst_off = k.dst_off;
        q.dst_stride = k.dst_stride;
        q.r1_off = k.dst_off;
        q.r1_stride = k.dst_stride;
        q.c1 = k.c1;
        q.sy = p;
    }
    op.p = prm[0];
    if (!gconv_prepare(dev, &op, &why)) {
        std::printf("%s: %s\n", k.name, why.c_str());
        return false;
    }
    for (auto &q : prm) q.kch = op.p.kch;
    dev.upload(bp, prm.data(), prm.size() * sizeof(GConvParams), &why);
    VkDescriptorPool pool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)Device::kBindings};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        vkCreateDescriptorPool(dev.device(), &pi, nullptr, &pool);
        VkDescriptorSetLayout l = dev.set_layout();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &l;
        vkAllocateDescriptorSets(dev.device(), &ai, &op.set);
        VkDescriptorBufferInfo bi[Device::kBindings];
        for (auto &b : bi) b = {dev.dummy(), 0, VK_WHOLE_SIZE};
        bi[0] = {bin.buf, 0, VK_WHOLE_SIZE};
        if (hasr) bi[1] = {bres.buf, 0, VK_WHOLE_SIZE};
        bi[2] = {bout.buf, 0, VK_WHOLE_SIZE};
        bi[3] = {bw.buf, 0, VK_WHOLE_SIZE};
        bi[4] = {bb.buf, 0, VK_WHOLE_SIZE};
        bi[7] = {bp.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[Device::kBindings];
        for (int j = 0; j < Device::kBindings; j++) {
            wr[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[j].dstSet = op.set;
            wr[j].dstBinding = (uint32_t)j;
            wr[j].descriptorCount = 1;
            wr[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[j].pBufferInfo = &bi[j];
        }
        vkUpdateDescriptorSets(dev.device(), Device::kBindings, wr, 0, nullptr);
    }
    bool ok = true;
    for (int v = 0; v < op.nvar; v++) {
        op.pipe[v] = dev.pipeline(op.disp[v].module.c_str(), op.disp[v].local_size, op.disp[v].subgroup, &why);
        if (!op.pipe[v]) {
            std::printf("%s: %s\n", k.name, why.c_str());
            ok = false;
            break;
        }
        dev.upload(bout, init.data(), init.size() * 2, &why);
        VkCommandBuffer cb = dev.begin();
        for (int p = 0; p < R; p++) {
            op.entry = p;
            gconv_record(dev, cb, op, v);
        }
        std::vector<uint16_t> got(out_len);
        if (!dev.submit_wait(cb, &why) || !dev.download(bout, got.data(), got.size() * 2, &why)) {
            std::printf("%s: run: %s\n", k.name, why.c_str());
            ok = false;
            break;
        }
        double maxd = 0;
        size_t bad = 0, stray = 0, first = (size_t)-1;
        for (size_t i = 0; i < out_len; i++) {
            if (!mine[i]) {
                stray += got[i] != sentinel;
                continue;
            }
            const float g = half_bits_to_float(got[i]);
            const double d = std::isfinite(g) ? std::fabs((double)g - ref[i]) : 1e9;
            if (d > max_err) {
                bad++;
                if (first == (size_t)-1) first = i;
            }
            if (d > maxd) maxd = d;
        }
        const bool pass = bad == 0 && stray == 0;
        std::printf("%-34s %-44s max err %.4f%s", k.name, op.disp[v].module.c_str(), maxd, pass ? "  ok\n" : "");
        if (!pass) {
            std::printf("  FAIL: %zu wrong, %zu written outside the slice", bad, stray);
            if (first != (size_t)-1) {
                const size_t px = first / k.dst_stride;
                std::printf("; first at pixel (%zu, %zu) channel %zu: %.4f, want %.4f", px / DW, px % DW, first % k.dst_stride,
                            half_bits_to_float(got[first]), ref[first]);
            }
            std::printf("\n");
        }
        ok = ok && pass;
        (*runs)++;
    }
    vkDestroyDescriptorPool(dev.device(), pool, nullptr);
    for (Buffer *b : {&bin, &bout, &bres, &bw, &bb, &bp}) dev.destroy(*b);
    return ok;
}

GConvShape shape(int coutp, int in, int act, int nres, int stride, int out)
{
    GConvShape s;
    s.coutp = coutp;
    s.in = in;
    s.act = act;
    s.nres = nres;
    s.stride = stride;
    s.out = out;
    return s;
}

}  // namespace

int main(int argc, char **argv)
{
    int device = -1;
    double max_err = 0.02;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--device")) device = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--max-err")) max_err = std::atof(argv[i + 1]);
    }
    std::string why;
    auto dev = Device::create(device, nullptr, &why);
    if (!dev) {
        std::printf("aji_vk_gconv_test: skipped (%s)\n", why.c_str());
        return 0;
    }
    std::printf("device %s, family %s\n", dev->info().name.c_str(), family_name(dev->family()));
    // name, shape, cin, cout, cinp, IH, IW, src stride, src off, dst stride, dst off, c1
    const Case cases[] = {
        {"nhwc s1 prefix/slice 48->32 lrelu", shape(32, 0, 2, 0, 1, 0), 40, 32, 48, 37, 45, 64, 0, 96, 32, 1.f},
        {"nhwc s1 64->64 residual", shape(64, 0, 2, 1, 1, 0), 64, 64, 64, 29, 50, 64, 0, 64, 0, 1.f},
        {"nhwc s1 192->64 slice residual", shape(64, 0, 2, 1, 1, 0), 192, 64, 192, 21, 35, 192, 0, 192, 64, 1.f},
        {"nhwc s1 96->48", shape(48, 0, 0, 0, 1, 0), 96, 48, 96, 19, 33, 96, 0, 48, 0, 1.f},
        {"nhwc s2 32->32 lrelu", shape(32, 0, 2, 0, 2, 0), 23, 32, 32, 38, 66, 32, 0, 32, 0, 1.f},
        {"nhwc s2 96->64 slice lrelu", shape(64, 0, 2, 0, 2, 0), 96, 64, 96, 34, 62, 96, 0, 192, 128, 1.f},
        {"nhwc s2 48->48 lrelu", shape(48, 0, 2, 0, 2, 0), 48, 48, 48, 22, 40, 48, 0, 48, 0, 1.f},
        {"nchw s2 at plane offset 3->32", shape(32, 1, 2, 0, 2, 0), 3, 32, 16, 34, 70, 0, 3 * 34 * 70, 32, 0, 1.f},
        {"nchw s2 3->16", shape(16, 1, 0, 0, 2, 0), 3, 16, 16, 30, 46, 0, 0, 16, 0, 1.f},
        {"nhwc s1 192->192 in 12 slices res", shape(16, 0, 2, 1, 1, 0), 192, 192, 192, 21, 35, 192, 0, 256, 32, 1.f, 12},
        {"nhwc s1 128->128 in 2 slices res", shape(64, 0, 2, 1, 1, 0), 128, 128, 128, 19, 37, 128, 0, 128, 0, 1.f, 2},
        {"nhwc s2 96->96 in 2 slices", shape(48, 0, 2, 0, 2, 0), 96, 96, 96, 30, 42, 96, 0, 96, 0, 1.f, 2},
        {"phase R2 GELU 2x32 in 2 slices", shape(32, 0, 1, 0, 1, 4), 48, 64, 48, 13, 27, 48, 0, 32, 0, 1.f, 2},
        {"phase R2 32->2x8", shape(16, 0, 0, 0, 1, 4), 32, 16, 32, 13, 27, 32, 0, 8, 0, 1.f},
        {"phase R2 32->2x16", shape(32, 0, 0, 0, 1, 4), 32, 32, 32, 13, 27, 32, 0, 16, 0, 1.f},
        {"phase R4 64->4x8", shape(32, 0, 0, 0, 1, 5), 64, 32, 64, 11, 19, 64, 0, 8, 0, 1.f},
        {"phase R4 192->4x16", shape(64, 0, 0, 0, 1, 5), 192, 64, 192, 9, 21, 192, 0, 16, 0, 1.f},
    };
    int failed = 0, runs = 0;
    for (const Case &k : cases) failed += !run_case(*dev, k, max_err, &runs);
    std::printf("%d runs, %d cases failed\n", runs, failed);
    return failed ? 1 : 0;
}
