/*
 * vk/tnet.h — what the transformer image models on Vulkan (vk/fdat.cpp; HAT, DAT and TFDAT to come)
 * are built from: one arena of tensors addressed as maps (fp16 or fp32, offset / stride / channel
 * offset, as tfm.comp reads them), fp32 and fp16 weight stores, a buffer of partial sums, and an
 * ordered list of steps (general convs on vk/gconv.h, tfm.comp passes, arena copies and fills),
 * recorded with a barrier between each two. The network's input and output are buffers it owns
 * (NCHW fp16) until bind_io points them at the caller's.
 */
#ifndef AJI_VK_TNET_H
#define AJI_VK_TNET_H

#include <cstdint>
#include <string>
#include <vector>

#include "vk/device.h"
#include "vk/gconv.h"

namespace ajivk {

// A map into the arena (tfm.comp's Map): byte offset, elements per pixel, first channel, fp32.
struct TMap {
    VkDeviceSize bytes = 0;
    int stride = 0, off = 0;
    bool f32 = false;
    bool none = true;
};
inline TMap hmap(VkDeviceSize bytes, int stride, int off = 0) { return TMap{bytes, stride, off, false, false}; }
inline TMap fmap(VkDeviceSize bytes, int stride, int off = 0) { return TMap{bytes, stride, off, true, false}; }

// One tfm.comp parameter entry (64 ints).
struct TArgs {
    int32_t v[64] = {};
    TArgs();
    void map(int k, const TMap &m);
    void f(int k, float x);
};

class TNet {
public:
    explicit TNet(Device &dev) : dev_(dev) {}
    ~TNet();
    TNet(const TNet &) = delete;
    TNet &operator=(const TNet &) = delete;

    // ---- layout (before finish)
    VkDeviceSize place(VkDeviceSize bytes);  // arena bytes, aligned for descriptors and uvec4s
    int add_wf(const std::vector<float> &v);  // fp32 store: offset in floats
    int add_wh(const std::vector<uint16_t> &h);  // fp16 store (GEMM weights): offset in halfs
    int add_wh_zero(size_t halfs);              // a zeroed fp16 region (written on the GPU)
    int add_part(size_t floats);                // partial sums: offset in floats
    // GEMM weights: w [K][N] floats, padded to [pad16(K)][pad16(N)] fp16; bias (N floats) or empty.
    struct Mat {
        int w = 0, bias = -1, K = 0, N = 0, Kp = 0, Np = 0;
    };
    Mat add_mat(const std::vector<float> &w, const std::vector<float> &bias, int K, int N);
    // A 3x3 conv as GEMM weights for conv_gemm: taps [9][cin][cout] (fp16 bits) into [9 * cinp][pad16(cout)].
    Mat add_conv_mat(const std::vector<uint16_t> &taps, const std::vector<float> &bias, int cin, int cinp, int cout);
    VkDeviceSize arena_bytes() const { return ar_; }

    // ---- steps
    // A 3x3 conv (conv c with bias b, taps [9][cin][cout]): output slices of the widths given.
    struct ConvIO {
        int in = 0, act = 0, stride = 1, out = 0;  // gconv's IN / ACT / STRIDE / OUT
        float slope = 0.2f;                        // ACT 2: the LeakyReLU slope
        bool src_in = false, dst_out = false;      // src the network input / dst the output
        VkDeviceSize src = 0, dst = 0, r1 = 0;     // arena offsets (r1: when nres)
        VkDeviceSize src_bytes = 0, dst_bytes = 0, r1_bytes = 0;
        int nres = 0;
        int src_off = 0, src_stride = 0, dst_stride = 0, r1_stride = 0;
        int IH = 0, IW = 0, H = 0, W = 0, cinp = 16, sy = 0;
    };
    void conv(const std::vector<uint16_t> &taps, const std::vector<float> &bias, int cin, int cout, const ConvIO &io,
              std::initializer_list<int> widths, const void *key);
    void pass(int pass, const TArgs &a, uint32_t gx, uint32_t gy, const std::string &label);
    void pass_threads(int pass, const TArgs &a, long long threads, const std::string &label);
    void copy(VkDeviceSize src, VkDeviceSize dst, VkDeviceSize bytes);
    void zero(VkDeviceSize off, VkDeviceSize bytes);

    // tfm.comp passes with their usual geometry
    void gemm(const TMap &a, int M, const Mat &w, const TMap &out, int act = 0, bool act_last = false, int colscale = -1,
              const TMap &add1 = TMap{}, const TMap &add2 = TMap{}, float slope = 0.2f, int add2scale = -1);
    void layernorm(const TMap &in, const TMap &out, int M, int C, int cpad, int gamma, int beta, float eps);
    // dw 3x3 + bias, GELU; with sums >= 0 (arena float offset), the channel sums of its output there
    void dwconv(const TMap &in, const TMap &out, int H, int W, int C, int w, int bias, bool gelu, VkDeviceSize sums = ~0ull);
    // A 3x3 conv (zero padding) as an im2col GEMM, fp32 sums: in (cinp channels) at H x W
    void conv_gemm(const TMap &in, int H, int W, int cinp, const Mat &w, const TMap &out, const TMap &add1 = TMap{});
    // The network input (NCHW fp16, h x w) minus a mean into an NHWC map (16 channels, zero beyond 3)
    // of ph x pw, reflect-padded
    void imgin(const TMap &out, int h, int w, int ph, int pw, const float mean[3]);
    // A map's first 3 channels (its width mw) cropped to h x w into the network output (NCHW fp16)
    void imgout(const TMap &in, int h, int w, int mw);
    // The channel sums of a map (C channels over M tokens) into sums (arena float offset)
    void colsums(const TMap &in, int M, int C, VkDeviceSize sums);
    // A map into another of n channels (fp32 <-> fp16)
    void convert(const TMap &in, const TMap &out, int M, int n);
    // cm = sigmoid(W2 act(W1 (sums / M * in_scale) + b1) + b2), act GELU or ReLU (b1, b2 < 0: none)
    void chan_gate(VkDeviceSize sums, int M, int C, int r, int w1, int b1, int w2, int b2, bool relu, float in_scale, VkDeviceSize cm,
                   int cpad);

    // ---- build: input / output (NCHW fp16) of the given bytes, uploads, pipelines, sets, variants
    bool finish(VkDeviceSize in_bytes, VkDeviceSize out_bytes, std::string *why);
    const Buffer &input() const { return in_; }
    const Buffer &output() const { return out_; }
    void bind_io(VkBuffer in, VkBuffer out);
    void record(VkCommandBuffer cb) const;
    std::string profile(Device &dev) const;
    size_t device_bytes() const { return bytes_; }
    size_t weight_bytes() const { return wbytes_; }

private:
    struct Op : GConvOp {
        int w = 0;
        int nres = 0;
        bool src_in = false, dst_out = false;
        VkDeviceSize src = 0, dst = 0, r1 = 0, src_bytes = 0, dst_bytes = 0, r1_bytes = 0;
    };
    struct W1 {
        VkDeviceSize woff = 0, wsize = 0, boff = 0, bsize = 0;
    };
    enum Kind { GCONV, PASS, COPY, ZERO };
    struct Step {
        Kind kind = GCONV;
        int idx = 0, pass = 0;
        uint32_t gx = 1, gy = 1;
        VkDeviceSize a = 0, b = 0, n = 0;  // COPY src, dst, bytes / ZERO off, -, bytes
        VkPipeline pipe = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::string label;
    };
    void record_step(VkCommandBuffer cb, const Step &st) const;
    void step_sets(VkBuffer in, VkBuffer out);

    Device &dev_;
    VkDeviceSize ar_ = 0;
    std::vector<float> wf_;
    std::vector<uint16_t> wh_, gw_, gb_;  // fp16 store; gconv weights and biases
    size_t part_ = 0;
    std::vector<W1> wv_;
    std::vector<Op> ops_;
    std::vector<TArgs> ta_;
    std::vector<Step> steps_;
    std::vector<std::pair<const void *, int>> wkeys_;  // conv key -> wv_ index
    Buffer in_, out_, arena_, wfb_, whb_, gwb_, gbb_, partb_, gprm_, tprm_;
    bool io_external_ = false;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::string failed_;  // a step the kernels can't run (finish refuses with it)
    size_t bytes_ = 0, wbytes_ = 0;
};

}  // namespace ajivk

#endif
