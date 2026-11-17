// Shared by the general 3x3 conv kernels (gconv_cm16, gconv_sg16, gconv_plain), the Vulkan
// counterpart of libaji's CUDA gconv (src/native/gconv.h) for the larger image networks (ESRGAN):
// a conv reads a channel prefix of a wider NHWC buffer and writes a channel slice of another, at
// strides and offsets given per dispatch, so dense blocks need no concatenation.
//
// The including kernel defines CINP (input channels read, a multiple of 16), COUTP (output
// channels computed, 16, 32 or 64), IN (0 NHWC fp16 at a stride, 1 three NCHW fp16 planes,
// 2 the 2x pixel-unshuffle of three NCHW planes), ACT (0 none, 2 LeakyReLU), NRES (residuals
// added: 0, 1 or 2), UP2 (1: the input is nearest-upsampled by 2 on the fly; it is H/2 x W/2) and
// OUT (0 NHWC at a stride, 3 three NCHW fp16 planes from channels 0..2), and the tile rows TH.
//
// The input tile and its one-pixel halo go through shared memory KCH channels at a time (the
// widest ESRGAN input is 192 channels, more than a 32 KB tile holds at once); the GEMM sums over
// the chunks. The conv sum (fp16) is staged in shared memory, and the writer applies the epilogue
// per element in fp32, as gconv does: + bias, + c1 * r1 + c2 * r2, then the activation; one
// rounding to fp16.
//
// Per-dispatch parameters (entry pc.entry of the Params buffer, 16 ints; offsets and strides in
// halfs, multiples of 8): 0 H, 1 W (the conv's output size), 2 src offset, 3 src stride, 4 dst
// offset, 5 dst stride, 6 r1 offset, 7 r1 stride, 8 r2 offset, 9 r2 stride, 10 c1, 11 c2,
// 12 slope (float bits).

const int TW = 16;
const int KCH = CINP % 64 == 0 ? 64 : CINP % 32 == 0 ? 32 : 16;  // channels per halo chunk
const int NCHUNK = CINP / KCH;
const int KP = KCH + 8, SP = COUTP + 8, HP = (TH + 2) * (TW + 2);
const int SM_HALO = HP * KP / 8, SM_STAGE = TH * TW * SP / 8;
const int SM_TOTAL = SM_HALO > SM_STAGE ? SM_HALO : SM_STAGE;

layout(push_constant) uniform PC { int entry; int u0; int u1; int u2; } pc;
layout(std430, binding = 7) readonly buffer Params { ivec4 params[]; };
int prm(int k) { return params[pc.entry * 4 + (k >> 2)][k & 3]; }
float prmf(int k) { return intBitsToFloat(prm(k)); }

#if IN == 0
layout(std430, binding = 0) readonly buffer Src { uvec4 src_v[]; };
#elif IN == 1
layout(std430, binding = 0) readonly buffer Src { float16_t src_h[]; };  // NCHW [3][H][W]
#else
layout(std430, binding = 0) readonly buffer Src { uint src_u[]; };       // NCHW [3][2H][2W], half pairs
#endif
#if NRES >= 1
layout(std430, binding = 1) readonly buffer R1 { uvec4 r1_v[]; };
#endif
#if NRES >= 2
layout(std430, binding = 5) readonly buffer R2 { uvec4 r2_v[]; };
#endif
#if OUT == 0
layout(std430, binding = 2) writeonly buffer Dst { uvec4 dst_v[]; };
#else
layout(std430, binding = 2) writeonly buffer Dst { float16_t dst_h[]; };  // NCHW [3][H][W]
#endif
layout(std430, binding = 4) readonly buffer Bias { float16_t bias_h[]; };  // [COUTP]

shared uvec4 sm[SM_TOTAL];

uint sm_pair(int h) { return sm[h >> 3][(h >> 1) & 3]; }
void sm_set_pair(int h, uint v) { sm[h >> 3][(h >> 1) & 3] = v; }

// Channels [chunk*KCH, chunk*KCH + KCH) of the halo tile: pixel (ty0 - 1 + py, tx0 - 1 + px) at
// sm[(py*(TW+2) + px) * KP/8]. Zero outside the image.
void fill_chunk(int chunk, int tid, int nth, int tx0, int ty0, int H, int W)
{
#if IN == 0
    const int VPI = KCH / 8, soff = prm(2), sstride = prm(3);
    for (int i = tid; i < HP * VPI; i += nth) {
        const int px = i / VPI, v = i % VPI;
        const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
        uvec4 a = uvec4(0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
#if UP2
            const int sp = (yy >> 1) * (W >> 1) + (xx >> 1);
#else
            const int sp = yy * W + xx;
#endif
            a = src_v[(sp * sstride + soff + chunk * KCH) / 8 + v];
        }
        sm[px * (KP / 8) + v] = a;
    }
#elif IN == 1
    for (int px = tid; px < HP; px += nth) {
        const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
        uvec4 a = uvec4(0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
            const int o = yy * W + xx, plane = H * W;
            a.x = packFloat2x16(f16vec2(src_h[o], src_h[plane + o]));
            a.y = packFloat2x16(f16vec2(src_h[2 * plane + o], float16_t(0.0)));
        }
        sm[px * (KP / 8)] = a;
        sm[px * (KP / 8) + 1] = uvec4(0);
    }
#else
    // pixel-unshuffle by 2: channel c*4 + ry*2 + rx (PyTorch order); (rx 0, 1) is one pair
    for (int px = tid; px < HP; px += nth) {
        const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
        uvec4 a = uvec4(0), b = uvec4(0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
            const int W2 = 2 * W, H2 = 2 * H, x2 = 2 * xx, y2 = 2 * yy;
            a.x = src_u[((0 * H2 + y2) * W2 + x2) >> 1];
            a.y = src_u[((0 * H2 + y2 + 1) * W2 + x2) >> 1];
            a.z = src_u[((1 * H2 + y2) * W2 + x2) >> 1];
            a.w = src_u[((1 * H2 + y2 + 1) * W2 + x2) >> 1];
            b.x = src_u[((2 * H2 + y2) * W2 + x2) >> 1];
            b.y = src_u[((2 * H2 + y2 + 1) * W2 + x2) >> 1];
        }
        sm[px * (KP / 8)] = a;
        sm[px * (KP / 8) + 1] = b;
    }
#endif
}

float act(float v)
{
#if ACT == 2
    return v >= 0.0 ? v : v * prmf(12);
#else
    return v;
#endif
}

// The staging tile [TH*TW][SP] (the conv sum, fp16) out through the epilogue.
void write_out(int tid, int nth, int tx0, int ty0, int H, int W)
{
#if OUT == 0
    const int VPO = COUTP / 8, doff = prm(4), dstride = prm(5);
#if NRES >= 1
    const int r1off = prm(6), r1stride = prm(7);
    const float c1 = prmf(10);
#endif
#if NRES >= 2
    const int r2off = prm(8), r2stride = prm(9);
    const float c2 = prmf(11);
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
            x += vec2(float(bias_h[c]), float(bias_h[c + 1]));
#if NRES >= 1
            x += c1 * vec2(unpackFloat2x16(a[j]));
#endif
#if NRES >= 2
            x += c2 * vec2(unpackFloat2x16(b2[j]));
#endif
            o[j] = packFloat2x16(f16vec2(act(x.x), act(x.y)));
        }
        dst_v[(op * dstride + doff) / 8 + v] = o;
    }
#else
    // channels 0..2 to NCHW planes
    for (int q = tid; q < TH * TW * 3; q += nth) {
        const int p = q / 3, c = q % 3, oy = ty0 + p / TW, ox = tx0 + p % TW;
        if (oy >= H || ox >= W) continue;
        const vec2 pr = vec2(unpackFloat2x16(sm_pair(p * SP + (c & ~1))));
        const float x = ((c & 1) != 0 ? pr.y : pr.x) + float(bias_h[c]);
        dst_h[(c * H + oy) * W + ox] = float16_t(act(x));
    }
#endif
}
