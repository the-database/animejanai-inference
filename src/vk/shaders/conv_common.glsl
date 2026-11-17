// Shared by the 3x3 conv kernels (conv3x3_cm16.comp, conv3x3_sg16.comp): the buffers, the
// input tile with its one-pixel halo, the epilogue (bias, activation, SPAN gates) and the output
// writers (NHWC, or depth-to-space into NCHW RGB plus the model input). The semantics are
// libaji's CUDA k_conv3x3 (src/native/conv_kernels.cuh); epilogue math runs in fp32 and rounds
// to fp16 once.
//
// The including kernel defines CINP, COUTP (padded channels), IN, ACT, GATE, OUT (the values of
// native/ops.h) and, before the include, the constants TH (tile rows) and SM_EXTRA (uvec4s of
// shared memory it needs after the staging tile).
//
// Shared memory `sm` is first the input tile [TH+2][TW+2][CP] (CP = CINP + 8 halfs per pixel),
// then, once the GEMM has read it, the staging tile [TH*TW][SP] (SP = COUTP + 8).

const int TW = 16, CP = CINP + 8, SP = COUTP + 8, HP = (TH + 2) * (TW + 2);
const int SM_HALO = HP * CP / 8, SM_STAGE = TH * TW * SP / 8;
const int SM_TOTAL = SM_HALO > SM_STAGE + SM_EXTRA ? SM_HALO : SM_STAGE + SM_EXTRA;

// H, W: the conv resolution. has_resid: a depth-to-space output adds the model input. in_norm:
// an NCHW input is normalized as it is loaded, (x - mean[c]) * scale[c], the six fp16 values after
// the bias (bias_h[COUTP + c], bias_h[COUTP + 3 + c]).
layout(push_constant) uniform PC { int H; int W; int has_resid; int in_norm; } pc;

#if IN == 0
layout(std430, binding = 0) readonly buffer Src { uvec4 src_v[]; };      // NHWC [H][W][CINP]
#elif IN == 1
layout(std430, binding = 0) readonly buffer Src { float16_t src_h[]; };  // NCHW [3][H][W]
#else
layout(std430, binding = 0) readonly buffer Src { uint src_u[]; };       // NCHW [3][2H][2W], half pairs
#endif
#if GATE != 0
layout(std430, binding = 1) readonly buffer Resid { uint resid_u[]; };   // NHWC [H][W][COUTP], half pairs
#elif OUT != 0
layout(std430, binding = 1) readonly buffer Resid { float16_t resid_h[]; };  // NCHW [3][H][W]
#endif
#if OUT == 0
layout(std430, binding = 2) writeonly buffer Dst { uvec4 dst_v[]; };     // NHWC [H][W][COUTP]
#else
layout(std430, binding = 2) writeonly buffer Dst { uint dst_u[]; };      // NCHW [3][R*H][R*W], half pairs
#endif
layout(std430, binding = 4) readonly buffer Bias { float16_t bias_h[]; };  // [COUTP]
#if ACT == 2
layout(std430, binding = 5) readonly buffer Slope { float16_t slope_h[]; };
#endif
#if GATE == 2
layout(std430, binding = 7) readonly buffer Batt { float16_t batt_h[]; };
#endif

shared uvec4 sm[SM_TOTAL];

// The fp16 pair at half index h (even) of shared memory.
uint sm_pair(int h) { return sm[h >> 3][(h >> 1) & 3]; }
void sm_set_pair(int h, uint v) { sm[h >> 3][(h >> 1) & 3] = v; }

// The input tile with halo: pixel (ty0 - 1 + py, tx0 - 1 + px) at sm[(py*(TW+2) + px) * CP/8].
// NCHW inputs fill channels 0..2 (or the 12 unshuffled ones) and zero the rest of the 16.
void fill_halo(int tid, int nth, int tx0, int ty0, int H, int W)
{
#if IN == 0
    const int VPI = CINP / 8;
    for (int i = tid; i < HP * VPI; i += nth) {
        const int px = i / VPI, v = i % VPI;
        const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
        const bool ok = yy >= 0 && yy < H && xx >= 0 && xx < W;
        sm[px * (CP / 8) + v] = ok ? src_v[(yy * W + xx) * VPI + v] : uvec4(0);
    }
#elif IN == 1
    for (int px = tid; px < HP; px += nth) {
        const int yy = ty0 - 1 + px / (TW + 2), xx = tx0 - 1 + px % (TW + 2);
        uvec4 a = uvec4(0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
            const int o = yy * W + xx, plane = H * W;
            vec3 v = vec3(float(src_h[o]), float(src_h[plane + o]), float(src_h[2 * plane + o]));
            if (pc.in_norm != 0)
                for (int c = 0; c < 3; c++) v[c] = (v[c] - float(bias_h[COUTP + c])) * float(bias_h[COUTP + 3 + c]);
            a.x = packFloat2x16(f16vec2(v.x, v.y));
            a.y = packFloat2x16(f16vec2(v.z, 0.0));
        }
        sm[px * (CP / 8)] = a;
        sm[px * (CP / 8) + 1] = uvec4(0);
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
            if (pc.in_norm != 0) {
                // pairs (a.x, a.y): channel 0, (a.z, a.w): 1, (b.x, b.y): 2
                const float m0 = float(bias_h[COUTP]), m1 = float(bias_h[COUTP + 1]), m2 = float(bias_h[COUTP + 2]);
                const float s0 = float(bias_h[COUTP + 3]), s1 = float(bias_h[COUTP + 4]), s2 = float(bias_h[COUTP + 5]);
                a.x = packFloat2x16(f16vec2((unpackFloat2x16(a.x) - m0) * s0));
                a.y = packFloat2x16(f16vec2((unpackFloat2x16(a.y) - m0) * s0));
                a.z = packFloat2x16(f16vec2((unpackFloat2x16(a.z) - m1) * s1));
                a.w = packFloat2x16(f16vec2((unpackFloat2x16(a.w) - m1) * s1));
                b.x = packFloat2x16(f16vec2((unpackFloat2x16(b.x) - m2) * s2));
                b.y = packFloat2x16(f16vec2((unpackFloat2x16(b.y) - m2) * s2));
            }
        }
        sm[px * (CP / 8)] = a;
        sm[px * (CP / 8) + 1] = b;
    }
#endif
}

#if GATE != 0
// The block input x at (oy, ox), channel c; zero outside the image.
float resid_at(int oy, int ox, int c, int H, int W)
{
    if (oy >= H || ox >= W) return 0.0;
    const f16vec2 p = unpackFloat2x16(resid_u[((oy * W + ox) * COUTP + c) >> 1]);
    return float((c & 1) != 0 ? p.y : p.x);
}
#endif

// The epilogue of channel c at (oy, ox) for o = accumulator + bias. GATE 2 (the tanh gate) is
// finished by the kernel, which has the attention conv: here it returns o unchanged.
float epilogue(float o, int c, int oy, int ox, int H, int W)
{
#if GATE == 0
#if ACT == 1
    o = max(o, 0.0);
#elif ACT == 2
    o = o >= 0.0 ? o : o * float(slope_h[c]);
#elif ACT == 3
    o = o / (1.0 + exp(-o));
#endif
    return o;
#elif GATE == 1
    return resid_at(oy, ox, c, H, W) + o;
#elif GATE == 3
    return (resid_at(oy, ox, c, H, W) + o) * (1.0 / (1.0 + exp(-o)) - 0.5);
#else
    return o;
#endif
}

#if GATE == 2
// (x + o) * tanh(W1x1 o + b): ga is the attention conv's accumulator, without its bias.
float tanh_gate(float o, float ga, int c, int oy, int ox, int H, int W)
{
    return (resid_at(oy, ox, c, H, W) + o) * tanh(ga + float(batt_h[c]));
}
#endif

// Writes the staging tile out: NHWC rows, or depth-to-space by R (CRD: channel c*R*R + ry*R + rx
// -> dst[c][R*y + ry][R*x + rx]) plus the model input at (y, x) when has_resid.
void write_out(int tid, int nth, int tx0, int ty0, int H, int W)
{
#if OUT == 0
    const int VPO = COUTP / 8, VPR = TW * VPO;
    for (int q = tid; q < TH * VPR; q += nth) {
        const int r = q / VPR, w = q % VPR, px = w / VPO, v = w % VPO, oy = ty0 + r, ox = tx0 + px;
        if (oy >= H || ox >= W) continue;
        dst_v[(oy * W + ox) * VPO + v] = sm[(r * TW + px) * (SP / 8) + v];
    }
#else
    const int R = OUT == 1 ? 2 : 4, CPO = TW * R / 2;  // output pairs per tile row
    const int OH = R * H, OW = R * W;
    const bool add = pc.has_resid != 0;
    for (int q = tid; q < 3 * TH * R * CPO; q += nth) {
        const int k = q % CPO, ry = (q / CPO) % R, r = (q / (CPO * R)) % TH, c = q / (CPO * R * TH);
        const int oy = ty0 + r, e = 2 * k, lx = e / R, rx = e % R, ox = tx0 + lx;
        if (oy >= H || ox >= W) continue;
        vec2 v = vec2(unpackFloat2x16(sm_pair((r * TW + lx) * SP + c * R * R + ry * R + rx)));
#if GATE == 0
        if (add) v += float(resid_h[(c * H + oy) * W + ox]);
#endif
        dst_u[((c * OH + R * oy + ry) * OW + R * tx0 + e) >> 1] = packFloat2x16(f16vec2(v));
    }
#endif
}
