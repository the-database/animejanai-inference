/*
 * native/tfm.h — transformer building blocks on fp16 token maps (NHWC [H][W][stride]), for the
 * native image transformers (FDAT, and the window-attention models built on the same pieces).
 *
 *   - LayerNorm over the first C channels of each token (padded channels written zero);
 *   - GEMM: out = epi(A[M][K] x W[K][N]), W packed in mma fragment order, fp32 accumulation; the
 *     epilogue adds a bias, applies GELU, scales by a per-column vector, adds up to two tensors;
 *   - the MLP fc1 -> GELU -> fc2 (plus residual) in one kernel, the hidden kept on chip;
 *   - window attention: per (window, head), softmax(q k^T * scale + bias) v, flash style (online
 *     softmax over 64-key blocks), windows of any shape over the image with an optional cyclic
 *     shift, tokens outside the image treated as zero (zero padding before attention);
 *   - depthwise 3x3 (optional GELU), channel sums (global average pools), the channel attention's
 *     Gram matrix and norms, and small device-side folds.
 * GELU is exact (erf) with the divisor 1.4140625, as the fp16 exports have it.
 */
#ifndef AJI_NATIVE_TFM_H
#define AJI_NATIVE_TFM_H

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ajinative {

struct TMat {
    void *frag = nullptr;  // uint4[Kpad/16][Npad/16][32] (pack.h, taps = 1)
    void *bias = nullptr;  // float[Npad] or null
    int K = 0, Kpad = 0, N = 0, Npad = 0;
};
// w: [K][N] floats (row k, column n). bias: N floats or null.
cudaError_t tmat_upload(TMat &m, const float *w, const float *bias, int K, int N);
void tmat_release(TMat &m);
// Allocate a TMat for K x N with no weights yet (frag zeroed; bias copied from device if given).
cudaError_t tmat_alloc(TMat &m, int K, int N, const float *bias_dev);
// Pack w_dev ([K][N] floats, device) into m.frag on stream s (weights computed on the GPU).
cudaError_t tmat_pack(TMat &m, const float *w_dev, cudaStream_t s);

enum class TAct { None = 0, Gelu = 1, LRelu = 2 };

struct TGemm {
    const void *a = nullptr;   // [M][sa] fp16, the first Kpad channels read (padding must be zero)
    int sa = 0, M = 0;
    void *out = nullptr;       // [M][so] fp16, columns [0, Npad) written at out + col0
    int so = 0;
    TAct act = TAct::None;
    const float *colscale = nullptr;          // [Npad] device, or null
    const void *add1 = nullptr, *add2 = nullptr;  // [M][s] fp16, or null (added after the scale)
    int s1 = 0, s2 = 0;
    float slope = 0.2f;     // LRelu
    bool act_last = false;  // apply act after the adds instead of before the scale
    // fp32 residual streams (bf16-trained models whose stream outgrows fp16): out and/or add1 as float
    bool out_f32 = false, add1_f32 = false;
    const float *add2scale = nullptr;  // [Npad] device: add2 scaled per column before it is added, or null
    // LayerNorm of A's rows over the first K channels (gamma, beta: K floats, device) applied on load, as
    // tlayernorm; only without adds and for K padded to 128 / 192 / 224 / 256 (cudaErrorNotSupported
    // otherwise, nothing launched: normalise first then)
    const float *ln_gamma = nullptr, *ln_beta = nullptr;
    float ln_eps = 1e-5f;
};
cudaError_t tgemm(const TMat &w, const TGemm &g, cudaStream_t s);

// out = A W + bias + add1 with A = (dwconv3x3(in) + dwb) * mul (FDAT's smix then fc2; DAT's SGFN gate
// then fc2) in one kernel: the depthwise 3x3 (zero padding; dw: [dwc][9] floats, dwb: [dwc] or null) over
// the H x W map in, times mul (or not), as tdwconv2 computes and rounds it, never stored. out and add1
// fp16 or fp32; add1 may be out. W a multiple of 64. cudaErrorNotSupported (nothing launched) for other
// shapes or AJI_NATIVE_DWMM=0: tdwconv2 then tgemm.
struct TDwGemm {
    const void *in = nullptr;  // [H*W][sin] fp16, W's Kpad channels read
    int sin = 0, H = 0, W = 0;
    const float *dw = nullptr, *dwb = nullptr;
    int dwc = 0;  // channels of dw / dwb (0: W's K); those past it count as zero
    const void *mul = nullptr;  // [H*W][smul] fp16, or null
    int smul = 0;
    void *out = nullptr;
    int so = 0;
    bool out_f32 = false;
    const void *add1 = nullptr;
    int s1 = 0;
    bool add1_f32 = false;
};
cudaError_t tdwgemm(const TMat &w, const TDwGemm &g, cudaStream_t s);

// The MLP out = epi(GELU(A W1 + b1) W2 + b2) in one kernel, the hidden kept on chip; the epilogue as
// tgemm's (add1, then out; fp32 or fp16 each). Both biases required. With ln_gamma, A is an fp32 map
// normalised on load (LayerNorm over W1's K channels, as tlayernorm2). cudaErrorNotSupported (nothing
// launched) for the shapes without a fused kernel, or per AJI_NATIVE_FMLP (0: none, 1: not with the
// LayerNorm): normalise / run the two GEMMs then.
struct TMlp {
    const void *a = nullptr;  // [M][sa] fp16, or fp32 with ln_gamma
    int sa = 0, M = 0;
    void *out = nullptr;
    int so = 0;
    bool out_f32 = false;
    const void *add1 = nullptr;
    int s1 = 0;
    bool add1_f32 = false;
    const float *ln_gamma = nullptr, *ln_beta = nullptr;  // [K] device
    float ln_eps = 1e-5f;
    // Optional: the LayerNorm of the output rows (over W2's N channels; zeros up to Npad) as fp16 at
    // lno [M][slno], e.g. the next block's input.
    void *lno = nullptr;
    int slno = 0;
    const float *lno_gamma = nullptr, *lno_beta = nullptr;
    float lno_eps = 1e-5f;
};
cudaError_t tmlp(const TMat &w1, const TMat &w2, const TMlp &g, cudaStream_t s);

// LayerNorm of the first C channels (gamma, beta: C floats in device memory), zeros to pad.
cudaError_t tlayernorm(const void *in, int sin, void *out, int sout, int M, int C, int cpad, const float *gamma,
                       const float *beta, float eps, cudaStream_t s);
// The same with fp32 input and/or output (strides in elements; cpad a multiple of 8).
cudaError_t tlayernorm2(const void *in, bool in_f32, int sin, void *out, bool out_f32, int sout, int M, int C, int cpad,
                        const float *gamma, const float *beta, float eps, cudaStream_t s);
// Element copies between fp32 and fp16 token maps: n channels per token (a multiple of 8).
cudaError_t tconvert(const void *in, bool in_f32, int sin, void *out, bool out_f32, int sout, int M, int n, cudaStream_t s,
                     float scale = 1.f);
// The residual stream's fp16 copy for a group conv, scaled down so a stream past fp16's range (a
// pretrained DAT-2's last group reaches ~1.4e5) stays finite; the conv's weights carry the inverse.
constexpr float kStreamScale = 1.f / 32;

struct TWinAttn {
    const void *qkv = nullptr;  // tokens [H][W][sq]: q at qo + h*hd, k at ko + h*hd, v at vo + h*hd
    int sq = 0, qo = 0, ko = 0, vo = 0;
    int H = 0, W = 0, heads = 0, hd = 0;  // hd <= 32
    int wh = 0, ww = 0;                   // window size (wh*ww tokens, a multiple of 16, <= 1024)
    int shift_y = 0, shift_x = 0;         // cyclic shift (HAT-style), 0 for none
    float scale = 1.f;
    const float *bias = nullptr;          // [heads][T][Tk] (T = wh*ww), or null
    void *out = nullptr;                  // tokens [H][W][so], head h at h*hd
    int so = 0;
    // General form (another kernel): overlapping key windows of kwh x kww tokens at (wy*wh - kpad,
    // wx*ww - kpad), zero outside the image (HAT's OCAB; Tk = kwh*kww <= 576); Swin's mask for
    // shifted windows (-100 between the regions the roll joins); and the bias as a relative-position
    // table: bias(q, k) = rtab[h][(ky - kpad - qy + rt_off) * rt_w + (kx - kpad - qx + rt_off)].
    int kwh = 0, kww = 0, kpad = 0;
    bool mask = false;
    const float *rtab = nullptr;
    int rt_w = 0, rt_offy = 0, rt_offx = 0;  // index (ky - kpad - qy + rt_offy) * rt_w + (kx - kpad - qx + rt_offx)
    // A padded size the windows, the shift and the mask are laid out on (0: H, W): tokens beyond H x W
    // are zero (DAT pads q, k, v to its windows' multiple with zeros).
    int Hp = 0, Wp = 0;
};
// The relative-position table behind a full bias [heads][wh*ww][Tk] (key windows kwh x kww with kpad,
// or kwh = 0: the query window), if the bias is one (each relative offset one value).
bool tbias_table(const std::vector<float> &bias, int heads, int wh, int ww, int kwh, int kww, int kpad, std::vector<float> *tab,
                 int *rt_w, int *rt_offy, int *rt_offx);
cudaError_t twinattn(const TWinAttn &a, cudaStream_t s);

// Depthwise 3x3 (zero padding, no bias) over the first C channels; w: [C][9] floats (device).
cudaError_t tdwconv(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, bool gelu, cudaStream_t s);
// The same, also adding the channel sums of the output to sums[C] (float device, zeroed by the caller).
cudaError_t tdwconv(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, bool gelu, float *sums,
                    cudaStream_t s);
// With a bias [C] (device, or null) before the GELU, and the result multiplied by mul (fp16 [H*W][smul],
// or null) after it; C a multiple of 8, strides too.
cudaError_t tdwconv2(const void *in, int sin, void *out, int sout, int H, int W, int C, const float *w, const float *bias, bool gelu,
                     const void *mul, int smul, float *sums, cudaStream_t s);
// sums[c] += sum over tokens of in[t][c] for c < C (sums: float device, zeroed by the caller).
cudaError_t tcolsum(const void *in, int sin, int M, int C, float *sums, cudaStream_t s);

// Channel attention statistics per head (q, k at qo/ko + h*hd of each token): gram[h][d][e] =
// sum_t q_d k_e, nq[h][d] = sum_t q_d^2, nk[h][e] = sum_t k_e^2 (float device, zeroed by the caller;
// layout gram [heads][hd][hd], then nq [heads][hd], then nk [heads][hd]).
cudaError_t tchanstats(const void *qkv, int sq, int qo, int ko, int M, int heads, int hd, float *stats, cudaStream_t s);

}  // namespace ajinative

#endif
