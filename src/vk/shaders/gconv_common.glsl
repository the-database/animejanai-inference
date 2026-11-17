// Shared by the general 3x3 conv kernels (gconv_cm16, gconv_sg16, gconv_plain), the Vulkan
// counterpart of libaji's CUDA gconv (src/native/gconv.h) for the larger networks (ESRGAN, RIFE): a
// conv reads a channel prefix of a wider NHWC buffer and writes a channel slice of another, at
// strides and offsets given per dispatch, so dense blocks and concatenations cost nothing.
//
// Compile-time (the module): COUTP (output channels computed, 16..64), IN (0 NHWC at a stride, 1
// three NCHW fp16 planes at an offset, 2 / 3 the 2x / 4x pixel-unshuffle of three NCHW planes:
// 12 / 48 channels, one chunk), ACT (0 none,
// 2 LeakyReLU), NRES (residuals added: 0, 1 or 2), UP2 (1: the input is nearest-upsampled by 2 on
// the fly), STRIDE (1 or 2), OUT (0 NHWC at a stride, 3 three NCHW fp16 planes from channels 0..2,
// 4 / 5 a phase scatter by R = 2 / 4: channel sx*COUTP/R + c of output pixel (y, x) lands at
// (R*y + sy, R*x + sx), channel c, for the dispatch's row phase sy), the tile rows TH and KMAX, the
// most input channels a halo chunk holds.
//
// Runtime (the parameter entry): the input channels read (cinp, a multiple of the chunk) and the
// chunk (kch, a multiple of 16 up to KMAX dividing cinp), the input size, offsets and strides. The
// input tile and its halo go through shared memory kch channels at a time; the GEMM sums over the
// chunks. The conv sum (fp16) is staged in shared memory, and the writer applies the epilogue per
// element in fp32, as gconv does: + bias, + c1 * r1 + c2 * r2, then the activation; one rounding.
//
// Output slices: a dispatch's z index z computes output channels [z*COUTP, (z+1)*COUTP) of a wider
// conv (NHWC output only): its weights follow slice 0's (9 * cinp * COUTP halfs each), its bias
// slice 0's, and its channels (and residual channels) are offset by z*COUTP.
//
// ACT 1 is GELU (erf). A phase scatter's channels per phase are its dst stride, so a transposed conv
// of any width runs as output slices too.
//
// Parameters: entry pc.entry of the Params buffer, 32 ints (offsets and strides in halfs,
// multiples of 8): 0 H, 1 W (the conv's output size), 2 src offset, 3 src stride, 4 dst offset,
// 5 dst stride, 6 r1 offset, 7 r1 stride, 8 r2 offset, 9 r2 stride, 10 c1, 11 c2, 12 slope (float
// bits), 13 cinp, 14 IH, 15 IW (the input size: with UP2 the upsampled one), 16 sy, 17 kch, 18 clamp
// (OUT 3: the output clamped to [0, 1], a final Clip).

const int TW = 16, S = STRIDE;
const int HR = (TH - 1) * S + 3, HC = (TW - 1) * S + 3, HP = HR * HC;  // halo rows, columns
const int SP = COUTP + 8;
const int SM_HALO = HP * (KMAX + 8) / 8, SM_STAGE = TH * TW * SP / 8;
const int SM_TOTAL = SM_HALO > SM_STAGE ? SM_HALO : SM_STAGE;
// the phase scatter's factor (a macro: the writer branches on it in the preprocessor)
#if OUT == 4
#define R 2
#elif OUT == 5
#define R 4
#else
#define R 1
#endif

layout(push_constant) uniform PC { int entry; int u0; int u1; int u2; } pc;
layout(std430, binding = 7) readonly buffer Params { ivec4 params[]; };
int prm(int k) { return params[pc.entry * 8 + (k >> 2)][k & 3]; }
float prmf(int k) { return intBitsToFloat(prm(k)); }
int zs() { return int(gl_WorkGroupID.z); }

#if IN == 0
layout(std430, binding = 0) readonly buffer Src { uvec4 src_v[]; };
#elif IN == 1
layout(std430, binding = 0) readonly buffer Src { float16_t src_h[]; };  // NCHW [3][IH][IW] at the src offset
#else
layout(std430, binding = 0) readonly buffer Src { uint src_u[]; };       // NCHW [3][2IH][2IW], half pairs
#endif
#if NRES >= 1
layout(std430, binding = 1) readonly buffer R1 { uvec4 r1_v[]; };
#endif
#if NRES >= 2
layout(std430, binding = 5) readonly buffer R2 { uvec4 r2_v[]; };
#endif
#if OUT == 3
layout(std430, binding = 2) writeonly buffer Dst { float16_t dst_h[]; };  // NCHW [3][H][W]
#else
layout(std430, binding = 2) writeonly buffer Dst { uvec4 dst_v[]; };
#endif
layout(std430, binding = 4) readonly buffer Bias { float16_t bias_h[]; };  // [COUTP]

shared uvec4 sm[SM_TOTAL];

uint sm_pair(int h) { return sm[h >> 3][(h >> 1) & 3]; }
void sm_set_pair(int h, uint v) { sm[h >> 3][(h >> 1) & 3] = v; }

// The halo pixel of output tile row r, column m, tap (dy, dx): sm index (in uvec4) of its chunk.
int halo_px(int r, int m, int dy, int dx) { return (r * S + dy) * HC + m * S + dx; }

// Channels [chunk*kch, chunk*kch + kch) of the halo tile: input pixel (ty0*S - 1 + py,
// tx0*S - 1 + px) at sm[(py*HC + px) * kp/8], kp = kch + 8. Zero outside the input.
void fill_chunk(int chunk, int kch, int tid, int nth, int tx0, int ty0)
{
    const int IH = prm(14), IW = prm(15), kp8 = (kch + 8) / 8;
    const int iy0 = ty0 * S - 1, ix0 = tx0 * S - 1;
#if IN == 0
    const int VPI = kch / 8, soff = prm(2), sstride = prm(3);
    for (int i = tid; i < HP * VPI; i += nth) {
        const int px = i / VPI, v = i % VPI;
        const int yy = iy0 + px / HC, xx = ix0 + px % HC;
        uvec4 a = uvec4(0);
        if (yy >= 0 && yy < IH && xx >= 0 && xx < IW) {
#if UP2
            const int sp = (yy >> 1) * (IW >> 1) + (xx >> 1);
#else
            const int sp = yy * IW + xx;
#endif
            a = src_v[(sp * sstride + soff + chunk * kch) / 8 + v];
        }
        sm[px * kp8 + v] = a;
    }
#elif IN == 1
    const int soff = prm(2), plane = IH * IW;
    for (int px = tid; px < HP; px += nth) {
        const int yy = iy0 + px / HC, xx = ix0 + px % HC;
        uvec4 a = uvec4(0);
        if (yy >= 0 && yy < IH && xx >= 0 && xx < IW) {
            const int o = soff + yy * IW + xx;
            a.x = packFloat2x16(f16vec2(src_h[o], src_h[plane + o]));
            a.y = packFloat2x16(f16vec2(src_h[2 * plane + o], float16_t(0.0)));
        }
        sm[px * kp8] = a;
        sm[px * kp8 + 1] = uvec4(0);
    }
#elif IN == 3
    // pixel-unshuffle by 4: channel c*16 + ry*4 + rx (PyTorch order); (rx, rx + 1) is one pair
    for (int px = tid; px < HP; px += nth) {
        const int yy = iy0 + px / HC, xx = ix0 + px % HC;
        uvec4 q[6] = uvec4[6](uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0));
        if (yy >= 0 && yy < IH && xx >= 0 && xx < IW) {
            const int W4 = 4 * IW, H4 = 4 * IH, x4 = 4 * xx, y4 = 4 * yy;
            for (int c = 0; c < 3; c++)
                for (int ry = 0; ry < 4; ry++)
                    for (int hh = 0; hh < 2; hh++) {
                        const int k = c * 8 + ry * 2 + hh;
                        q[k >> 2][k & 3] = src_u[((c * H4 + y4 + ry) * W4 + x4 + 2 * hh) >> 1];
                    }
        }
        for (int k = 0; k < 6; k++) sm[px * kp8 + k] = q[k];
    }
#else
    // pixel-unshuffle by 2: channel c*4 + ry*2 + rx (PyTorch order); (rx 0, 1) is one pair
    for (int px = tid; px < HP; px += nth) {
        const int yy = iy0 + px / HC, xx = ix0 + px % HC;
        uvec4 a = uvec4(0), b = uvec4(0);
        if (yy >= 0 && yy < IH && xx >= 0 && xx < IW) {
            const int W2 = 2 * IW, H2 = 2 * IH, x2 = 2 * xx, y2 = 2 * yy;
            a.x = src_u[((0 * H2 + y2) * W2 + x2) >> 1];
            a.y = src_u[((0 * H2 + y2 + 1) * W2 + x2) >> 1];
            a.z = src_u[((1 * H2 + y2) * W2 + x2) >> 1];
            a.w = src_u[((1 * H2 + y2 + 1) * W2 + x2) >> 1];
            b.x = src_u[((2 * H2 + y2) * W2 + x2) >> 1];
            b.y = src_u[((2 * H2 + y2 + 1) * W2 + x2) >> 1];
        }
        sm[px * kp8] = a;
        sm[px * kp8 + 1] = b;
    }
#endif
}

#if ACT == 1
// erf (Abramowitz & Stegun 7.1.26, |error| < 1.2e-7) for GELU as CUDA's erff gives it, far below
// the fp16 output's step
float erf_(float x)
{
    const float s = sign(x), a = abs(x);
    const float t = 1.0 / (1.0 + 0.3275911 * a);
    return s * (1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a));
}
#endif
float act(float v)
{
#if ACT == 2
    return v >= 0.0 ? v : v * prmf(12);
#elif ACT == 1
    return 0.5 * v * (1.0 + erf_(v / 1.4140625));
#else
    return v;
#endif
}

// The staging tile [TH*TW][SP] (the conv sum, fp16) out through the epilogue.
void write_out(int tid, int nth, int tx0, int ty0, int H, int W)
{
#if OUT == 3
    // channels 0..2 to NCHW planes
    for (int q = tid; q < TH * TW * 3; q += nth) {
        const int p = q / 3, c = q % 3, oy = ty0 + p / TW, ox = tx0 + p % TW;
        if (oy >= H || ox >= W) continue;
        const vec2 pr = vec2(unpackFloat2x16(sm_pair(p * SP + (c & ~1))));
        const float x = ((c & 1) != 0 ? pr.y : pr.x) + float(bias_h[c]);
        const float y = prm(18) != 0 ? clamp(act(x), 0.0, 1.0) : act(x);
        dst_h[(c * H + oy) * W + ox] = float16_t(y);
    }
#else
    const int zc = zs() * COUTP;
    const int VPO = COUTP / 8, doff = prm(4) + zc, dstride = prm(5);
#if NRES >= 1
    const int r1off = prm(6) + zc, r1stride = prm(7);
    const float c1 = prmf(10);
#endif
#if NRES >= 2
    const int r2off = prm(8) + zc, r2stride = prm(9);
    const float c2 = prmf(11);
#endif
#if R > 1
    const int sy = prm(16);
#endif
    for (int q = tid; q < TH * TW * VPO; q += nth) {
        const int p = q / VPO, v = q % VPO, oy = ty0 + p / TW, ox = tx0 + p % TW;
        if (oy >= H || ox >= W) continue;
        const int op = oy * W + ox;
        const uvec4 s = sm[p * (SP / 8) + v];
#if NRES >= 1
        const uvec4 a = r1_v[(op * r1stride + r1off) / 8 + v];
#endif
#if NRES >= 2
        const uvec4 b2 = r2_v[(op * r2stride + r2off) / 8 + v];
#endif
        uvec4 o;
        [[unroll]] for (int j = 0; j < 4; j++) {
            vec2 x = vec2(unpackFloat2x16(s[j]));
            const int c = v * 8 + 2 * j;
            x += vec2(float(bias_h[zc + c]), float(bias_h[zc + c + 1]));
#if NRES >= 1
            x += c1 * vec2(unpackFloat2x16(a[j]));
#endif
#if NRES >= 2
            x += c2 * vec2(unpackFloat2x16(b2[j]));
#endif
            o[j] = packFloat2x16(f16vec2(act(x.x), act(x.y)));
        }
#if R > 1
        // output channel zc + 8v of the transposed conv: phase sx = it / dstride (the channels per
        // phase), channel it % dstride of pixel (R*oy + sy, R*ox + sx); with slices, any slice width
        const int gch = zc + v * 8, sx = gch / dstride, ch = gch % dstride;
        const int dp = (R * oy + sy) * (R * W) + R * ox + sx;
        dst_v[(dp * dstride + prm(4) + ch) / 8] = o;
#else
        dst_v[(op * dstride + doff) / 8 + v] = o;
#endif
    }
#endif
}
