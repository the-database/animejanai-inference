/*
 * native/gconv.h — a general fp16 tensor-core 3x3 convolution for the larger native networks
 * (RIFE, ESRGAN, TemporalFix, ...). Separate from conv.h (SPAN/Compact), whose fused kernels stay
 * untouched.
 *
 * One kernel family, k_gconv (gconv.cuh): mma.sync m16n8k16 with fp16 accumulation, the input
 * tile with its halo in shared memory, K split into chunks of 16/32/64 channels so any input
 * width works, output channels in launches of up to 64. Runtime pixel strides on the input,
 * the output and the residuals, so a conv can read a channel prefix of a wider buffer and write
 * into a channel slice of another (concatenations cost nothing).
 *   - input: NHWC fp16, stride 1 or 2 (pad 1), or the planes of an NCHW tensor (3 channels);
 *   - epilogue: bias, optional v += c1*r1 + c2*r2 (NHWC residuals), then none/ReLU/LeakyReLU;
 *   - output: NHWC, or a phase scatter dst[R*y + sy][R*x + sx][c] for sx in [0,R), which with
 *     rearranged weights turns a stride-2 transposed conv (and a following depth-to-space) into
 *     R plain 3x3 convs, one per output row phase sy.
 */
#ifndef AJI_NATIVE_GCONV_H
#define AJI_NATIVE_GCONV_H

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ajinative {

enum class GAct { None = 0, Relu = 1, LRelu = 2, Gelu = 3 };  // GELU: exact (erf), divisor 1.4140625

// Packed weights of one conv (all output chunks), device memory.
struct GWeights {
    void *frag = nullptr;      // per output chunk: uint4[9][cin_pad/16][coutp/16][32]
    void *bias = nullptr;      // half2[cout_pad/2]
    int cin = 0, cin_pad = 0, cout = 0, cout_pad = 0, coutp = 0;  // coutp: channels per launch
};

// w: [9][cin][cout] fp16 bits (taps ky*3+kx; ONNX OIHW transposed). bias: cout floats or null.
// cin_pad: a multiple of 16 (the input buffer's channels that are read); cout_pad: a multiple of
// 16; output launches of min(cout_pad, 64) channels.
cudaError_t gupload(GWeights &out, const uint16_t *w, const float *bias, int cin, int cin_pad, int cout,
                    int cout_pad);
void grelease(GWeights &w);

struct GConvArgs {
    const void *src = nullptr;   // NHWC (or the first plane of NCHW)
    int src_stride = 0;          // halfs per input pixel (NHWC) | halfs per plane (NCHW)
    int IH = 0, IW = 0;          // input size
    void *dst = nullptr;         // NHWC; for a phase scatter, the full-size output buffer
    int dst_stride = 0;          // halfs per output pixel
    int H = 0, W = 0;            // conv output size (before any phase scatter)
    const void *r1 = nullptr, *r2 = nullptr;  // residuals, NHWC at H x W
    int r1_stride = 0, r2_stride = 0;
    float c1 = 1.f, c2 = 1.f;
    GAct act = GAct::None;
    float slope = 0.2f;          // LeakyReLU
    int stride = 1;              // 1 or 2
    bool nchw = false;           // input is 3 NCHW planes (stride 2 only)
    bool up2 = false;            // input is nearest-upsampled by 2 on the fly (IH, IW: the upsampled size)
    bool acc32 = false;          // fp32 accumulation (stride 1, NHWC output): deep sums that fp16 loses
    int R = 0, sy = 0;           // phase scatter when R > 0 (2 or 4); dst is (R*H) x (R*W)
    float *ws = nullptr;         // split-K workspace (gworkspace floats); null: no split
    size_t ws_floats = 0;
    // fp32 output (dst float NHWC, dst_stride in floats) for sums that outgrow fp16: needs acc32 and
    // the workspace (the partials go through it); r1f: an fp32 residual added in fp32 (r1/r2 unused).
    bool out_f32 = false;
    const float *r1f = nullptr;
    int r1f_stride = 0;
    // Optional: the output's channel sums (of the stored fp16 values) added to sums[cout_pad] (float,
    // zeroed by the caller); fp16 NHWC output only; the launch then runs unsplit.
    float *sums = nullptr;
};

// Floats of workspace glaunch would use to split K for a (0: it would not split).
size_t gworkspace(const GWeights &w, const GConvArgs &a);

bool gsupported(const GWeights &w, const GConvArgs &a, std::string *why = nullptr);
int gvariant_count();
// Runs all output chunks. variant < 0 picks the default.
cudaError_t glaunch(const GWeights &w, const GConvArgs &a, int variant, cudaStream_t s);
// Time the variants on a (writes a.dst) and return the fastest, or -1.
int gpick(const GWeights &w, const GConvArgs &a, cudaStream_t s, float *best_ms = nullptr);

}  // namespace ajinative

#endif
