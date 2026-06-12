/*
 * HLSL compute kernels for the DirectML backend — exact ports of the
 * CUDA kernels in kernels.cu (same weight tables from resample.h, same
 * mirror boundaries, same round-half-even quantization). Compiled at
 * runtime with FXC (d3dcompiler_47, cs_5_0) with IEEE strictness.
 *
 * Data lives in raw D3D12 buffers (DML tensors are buffers; video
 * planes are staged through buffers with 256-aligned row pitches).
 * Device-local data buffers are bound as UAVs for both reads and
 * writes, so they live in UNORDERED_ACCESS state while kernels run and
 * only the texture-staging copies need state transitions. The small
 * static weight tables sit in upload-heap buffers bound as SRVs
 * (GENERIC_READ covers shader reads — no transitions ever).
 *
 * Root signature (all root descriptors, no descriptor heaps):
 *   b0: 16 DWORD root constants
 *   t0: starts (StructuredBuffer<int>), t1: weights (StructuredBuffer<float>)
 *   u0,u1,u2: RWByteAddressBuffer data (roles per kernel, see entries)
 *
 * Defines per PSO variant:
 *   T16=0|1  raw video sample type: u8 (NV12) / u16 (P010)
 *   IO16=0|1 tensor element type: fp32 / fp16
 *   TPX=n    pixels per thread where dword-granular stores require it
 */
#pragma once

static const char DML_HLSL[] = R"hlsl(
cbuffer CB : register(b0) {
    int4 di;        // kernel-specific dims (see entries)
    int4 dj;        // kernel-specific (strides etc.)
    float kr, kb, yoff, yscale;
    float coff, cscale, qdiv, qmax;
};

StructuredBuffer<int>   starts  : register(t0);
StructuredBuffer<float> weights : register(t1);
RWByteAddressBuffer d0 : register(u0);
RWByteAddressBuffer d1 : register(u1);
RWByteAddressBuffer d2 : register(u2);

int mirr(int i, int n)
{
    i = i < 0 ? -i - 1 : i;
    i = i >= n ? 2 * n - 1 - i : i;
    return clamp(i, 0, n - 1);
}

// round-half-even (CUDA rintf), exact for our integer-scaled ranges
float rint_even(float v)
{
    float fl = floor(v);
    float fr = v - fl;
    float r = fl + (fr > 0.5f ? 1.0f : 0.0f);
    if (fr == 0.5f)
        r = fl + (fmod(fl, 2.0f) != 0.0f ? 1.0f : 0.0f);
    return r;
}

float quant(float v, float d, float m)
{
    return min(max(rint_even(v / d), 0.0f), m) * d;
}

uint load_u8(RWByteAddressBuffer b, uint o)
{
    return (b.Load(o & ~3u) >> ((o & 3u) * 8u)) & 0xffu;
}

uint load_u16(RWByteAddressBuffer b, uint o)  // o must be even
{
    return (b.Load(o & ~3u) >> ((o & 2u) * 8u)) & 0xffffu;
}

float load_f16e(RWByteAddressBuffer b, uint elem)
{
    uint o = elem * 2u;
    return f16tof32((b.Load(o & ~3u) >> ((o & 2u) * 8u)) & 0xffffu);
}

float load_f32e(RWByteAddressBuffer b, uint elem)
{
    return asfloat(b.Load(elem * 4u));
}

#if T16
#define RAWLOAD(buf, off) float(load_u16(buf, off))
#define RAWSIZE 2u
#else
#define RAWLOAD(buf, off) float(load_u8(buf, off))
#define RAWSIZE 1u
#endif

#if IO16
#define TLOAD(buf, elem) load_f16e(buf, elem)
#else
#define TLOAD(buf, elem) load_f32e(buf, elem)
#endif

// ---- cs_uv_h: horizontal resample of interleaved raw chroma ----
// -> 2 planar fp32 (raw units). di = {src(cw), dst(W), taps, rows(ch)},
// dj.x = uv row stride bytes. d0 = uv raw in, d1 = f32 out.
[numthreads(32, 8, 1)]
void cs_uv_h(uint3 id : SV_DispatchThreadID)
{
    int x = id.x, y = id.y;
    if (x >= di.y || y >= di.w)
        return;
    uint row = (uint)y * (uint)dj.x;
    int s0 = starts[x];
    float u = 0.0f, v = 0.0f;
    for (int j = 0; j < di.z; j++) {
        int sx = mirr(s0 + j, di.x);
        float w = weights[x * di.z + j];
        u += w * RAWLOAD(d0, row + (2u * sx) * RAWSIZE);
        v += w * RAWLOAD(d0, row + (2u * sx + 1u) * RAWSIZE);
    }
    uint plane = (uint)di.y * (uint)di.w, idx = (uint)y * (uint)di.y + x;
    d1.Store(idx * 4u, asuint(u));
    d1.Store((plane + idx) * 4u, asuint(v));
}

// ---- cs_v_f32: vertical resample of N planar fp32 planes ----
// di = {w, src rows, taps, dst rows}. d0 = f32 in, d1 = f32 out. z = plane.
[numthreads(32, 8, 1)]
void cs_v_f32(uint3 id : SV_DispatchThreadID)
{
    int x = id.x, y = id.y;
    if (x >= di.x || y >= di.w)
        return;
    uint sp = id.z * (uint)di.x * (uint)di.y;
    uint dp = id.z * (uint)di.x * (uint)di.w;
    int s0 = starts[y];
    float a = 0.0f;
    for (int j = 0; j < di.z; j++)
        a += weights[y * di.z + j] *
             load_f32e(d0, sp + (uint)mirr(s0 + j, di.y) * (uint)di.x + x);
    d1.Store((dp + (uint)y * (uint)di.x + x) * 4u, asuint(a));
}

// ---- cs_pre_combine: Y raw + upsampled chroma f32 -> RGB tensor ----
// 2 px per thread (x covers w/2; w is always even). di = {w, h},
// dj.x = y row stride bytes, dj.y = destination element offset (RIFE
// writes channel blocks of a shared input tensor). d0 = y raw,
// d1 = uvf f32 (2 planes w*h), d2 = tensor out.
[numthreads(32, 8, 1)]
void cs_pre_combine(uint3 id : SV_DispatchThreadID)
{
    int w = di.x, h = di.y;
    int x0 = id.x * 2, y = id.y;
    if (x0 >= w || y >= h)
        return;
    uint plane = (uint)w * (uint)h;
    float rgb[6];
    [unroll]
    for (int k = 0; k < 2; k++) {
        int x = x0 + k;
        uint idx = (uint)y * (uint)w + x;
        float Y = (RAWLOAD(d0, (uint)y * (uint)dj.x + (uint)x * RAWSIZE)
                   - yoff) / yscale;
        float U = (load_f32e(d1, idx) - coff) / cscale;
        float V = (load_f32e(d1, plane + idx) - coff) / cscale;
        float kg = 1.0f - kr - kb;
        rgb[k * 3 + 0] = Y + 2.0f * (1.0f - kr) * V;
        rgb[k * 3 + 2] = Y + 2.0f * (1.0f - kb) * U;
        rgb[k * 3 + 1] = Y - (2.0f * kb * (1.0f - kb) * U +
                              2.0f * kr * (1.0f - kr) * V) / kg;
    }
    uint base = (uint)dj.y + (uint)y * (uint)w + (uint)x0;
    [unroll]
    for (int p = 0; p < 3; p++) {
        uint e = p * plane + base;
#if IO16
        d2.Store(e * 2u, f32tof16(rgb[p]) | (f32tof16(rgb[3 + p]) << 16u));
#else
        d2.Store(e * 4u, asuint(rgb[p]));
        d2.Store((e + 1u) * 4u, asuint(rgb[3 + p]));
#endif
    }
}

// ---- cs_fill: fill a dword range with a constant ----
// di.x = dword count, dj.x = start dword, dj.y = value bits. d0 = dst.
[numthreads(256, 1, 1)]
void cs_fill(uint3 id : SV_DispatchThreadID)
{
    if (id.x < (uint)di.x)
        d0.Store(((uint)dj.x + id.x) * 4u, (uint)dj.y);
}

// ---- cs_window: copy a byte window between two pitched buffers ----
// Used to embed frames into the RIFE pad interior and crop them back
// out. Partial dwords at the window edges read-modify-write so the
// black pad border survives. di = {window bytes per row, rows, dwords
// per row}, dj = {src byte offset, src pitch, dst byte offset, dst
// pitch}. d0 = src, d1 = dst. Offsets/pitches are even; pitches %4==0.
[numthreads(32, 8, 1)]
void cs_window(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)di.z || id.y >= (uint)di.y)
        return;
    uint row_src = (uint)dj.x + id.y * (uint)dj.y;
    uint row_dst = (uint)dj.z + id.y * (uint)dj.w;
    uint wstart = row_dst, wend = row_dst + (uint)di.x;
    uint base = (row_dst & ~3u) + id.x * 4u;
    uint val = 0u, mask = 0u;
    [unroll]
    for (uint b = 0; b < 4; b++) {
        uint db = base + b;
        if (db >= wstart && db < wend) {
            val |= load_u8(d0, row_src + (db - wstart)) << (b * 8u);
            mask |= 0xffu << (b * 8u);
        }
    }
    if (mask != 0xffffffffu)
        val |= d1.Load(base) & ~mask;
    d1.Store(base, val);
}

// ---- cs_rife_consts: mesh + multiplier channels (RIFE input 7..10) ----
// 2 px per thread on the fp16 path (w = padded width, always even).
// di = {w, h}, dj.x = base element offset (7 * plane). d0 = tensor.
[numthreads(32, 8, 1)]
void cs_rife_consts(uint3 id : SV_DispatchThreadID)
{
    int w = di.x, h = di.y;
#if IO16
    int x0 = id.x * 2, y = id.y;
    if (x0 >= w || y >= h)
        return;
    uint plane = (uint)w * (uint)h;
    uint base = (uint)dj.x + (uint)y * (uint)w + (uint)x0;
    float vx0 = 2.0f * x0 / (w - 1) - 1.0f;
    float vx1 = 2.0f * (x0 + 1) / (w - 1) - 1.0f;
    float vy = 2.0f * y / (h - 1) - 1.0f;
    d0.Store(base * 2u, f32tof16(vx0) | (f32tof16(vx1) << 16u));
    d0.Store((plane + base) * 2u, f32tof16(vy) | (f32tof16(vy) << 16u));
    uint mw = f32tof16(2.0f / (w - 1));
    uint mh = f32tof16(2.0f / (h - 1));
    d0.Store((2u * plane + base) * 2u, mw | (mw << 16u));
    d0.Store((3u * plane + base) * 2u, mh | (mh << 16u));
#else
    int x = id.x, y = id.y;
    if (x >= w || y >= h)
        return;
    uint plane = (uint)w * (uint)h;
    uint base = (uint)dj.x + (uint)y * (uint)w + (uint)x;
    d0.Store(base * 4u, asuint(2.0f * x / (w - 1) - 1.0f));
    d0.Store((plane + base) * 4u, asuint(2.0f * y / (h - 1) - 1.0f));
    d0.Store((2u * plane + base) * 4u, asuint(2.0f / (w - 1)));
    d0.Store((3u * plane + base) * 4u, asuint(2.0f / (h - 1)));
#endif
}

// ---- cs_scd_partial: per-group sum of |Ya - Yb| * norm ----
// (misc.SCDetect's metric; HLSL has no float atomics, so this is a
// two-pass reduction.) di = {w, h, groups_x}, dj = {pitch_a, pitch_b},
// kr = norm. d0 = plane a, d1 = plane b, d2 = partials (one float per
// group).
groupshared float scd_gs[256];
[numthreads(32, 8, 1)]
void cs_scd_partial(uint3 id : SV_DispatchThreadID,
                    uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    float d = 0.0f;
    if (id.x < (uint)di.x && id.y < (uint)di.y) {
        float a = RAWLOAD(d0, id.y * (uint)dj.x + id.x * RAWSIZE);
        float b = RAWLOAD(d1, id.y * (uint)dj.y + id.x * RAWSIZE);
        d = abs(a - b) * kr;
    }
    scd_gs[gi] = d;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint n = 128; n > 0; n >>= 1) {
        if (gi < n)
            scd_gs[gi] += scd_gs[gi + n];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0)
        d2.Store((gid.y * (uint)di.z + gid.x) * 4u, asuint(scd_gs[0]));
}

// ---- cs_scd_final: one group sums all partials ----
// di.x = partial count. d0 = partials, d1 = result (one float).
[numthreads(256, 1, 1)]
void cs_scd_final(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    float a = 0.0f;
    for (uint i = gi; i < (uint)di.x; i += 256u)
        a += load_f32e(d0, i);
    scd_gs[gi] = a;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint n = 128; n > 0; n >>= 1) {
        if (gi < n)
            scd_gs[gi] += scd_gs[gi + n];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0)
        d1.Store(0u, asuint(scd_gs[0]));
}

// ---- cs_post_matrix: RGB tensor -> quantized Y raw + chroma f32 ----
// TPX px per thread (4 for u8, 2 for u16: one dword Y store; spill into
// row padding is safe). di = {w, h}, dj.x = y row stride bytes.
// d0 = tensor in, d1 = y raw out, d2 = uvf f32 out (2 planes w*h).
[numthreads(32, 8, 1)]
void cs_post_matrix(uint3 id : SV_DispatchThreadID)
{
    int w = di.x, h = di.y;
    int x0 = id.x * TPX, y = id.y;
    if (x0 >= w || y >= h)
        return;
    uint plane = (uint)w * (uint)h;
    uint ydword = 0u;
    [unroll]
    for (int k = 0; k < TPX; k++) {
        int x = min(x0 + k, w - 1);   // clamp: spill goes to row padding
        uint idx = (uint)y * (uint)w + x;
        float r = TLOAD(d0, idx);
        float g = TLOAD(d0, plane + idx);
        float b = TLOAD(d0, 2u * plane + idx);
        float Y = kr * r + (1.0f - kr - kb) * g + kb * b;
        uint q = (uint)quant(Y * yscale + yoff, qdiv, qmax);
#if T16
        ydword |= (q & 0xffffu) << (k * 16);
#else
        ydword |= (q & 0xffu) << (k * 8);
#endif
        if (x0 + k < w) {
            d2.Store(idx * 4u, asuint((b - Y) / (2.0f * (1.0f - kb))));
            d2.Store((plane + idx) * 4u,
                     asuint((r - Y) / (2.0f * (1.0f - kr))));
        }
    }
    d1.Store((uint)y * (uint)dj.x + (uint)x0 * RAWSIZE, ydword);
}

// ---- cs_h_f32: horizontal resample of N planar fp32 planes ----
// di = {src w, dst w, taps, rows}. d0 = f32 in, d1 = f32 out. z = plane.
[numthreads(32, 8, 1)]
void cs_h_f32(uint3 id : SV_DispatchThreadID)
{
    int x = id.x, y = id.y;
    if (x >= di.y || y >= di.w)
        return;
    uint sp = id.z * (uint)di.x * (uint)di.w + (uint)y * (uint)di.x;
    uint dp = id.z * (uint)di.y * (uint)di.w;
    int s0 = starts[x];
    float a = 0.0f;
    for (int j = 0; j < di.z; j++)
        a += weights[x * di.z + j] * load_f32e(d0, sp + mirr(s0 + j, di.x));
    d1.Store((dp + (uint)y * (uint)di.y + x) * 4u, asuint(a));
}

// ---- cs_uv_v_store: vertical chroma resample + quantize + interleave ----
// TPX chroma px per thread (2 for u8, 1 for u16: one dword UV store).
// di = {cw, src rows(h), taps, dst rows(ch)}, dj.x = uv row stride bytes.
// d0 = uvf f32 (2 planes cw*h), d1 = uv raw out.
[numthreads(32, 8, 1)]
void cs_uv_v_store(uint3 id : SV_DispatchThreadID)
{
    int cw = di.x;
    int x0 = id.x * TPX, y = id.y;
    if (x0 >= cw || y >= di.w)
        return;
    uint plane = (uint)cw * (uint)di.y;
    int s0 = starts[y];
    uint dword = 0u;
    [unroll]
    for (int k = 0; k < TPX; k++) {
        int x = min(x0 + k, cw - 1);  // clamp: spill goes to row padding
        float u = 0.0f, v = 0.0f;
        for (int j = 0; j < di.z; j++) {
            uint row = (uint)mirr(s0 + j, di.y) * (uint)cw + x;
            float w = weights[y * di.z + j];
            u += w * load_f32e(d0, row);
            v += w * load_f32e(d0, plane + row);
        }
        uint qu = (uint)quant(u * cscale + coff, qdiv, qmax);
        uint qv = (uint)quant(v * cscale + coff, qdiv, qmax);
#if T16
        dword = qu | (qv << 16u);
#else
        dword |= (qu | (qv << 8u)) << (k * 16);
#endif
    }
    d1.Store((uint)y * (uint)dj.x + (uint)x0 * 2u * RAWSIZE, dword);
}

// ---- cs_rs_h: horizontal resize, tensor -> planar f32, 3 planes ----
// di = {src w, dst w, taps, rows}. d0 = tensor in, d1 = f32 out. z = plane.
[numthreads(32, 8, 1)]
void cs_rs_h(uint3 id : SV_DispatchThreadID)
{
    int x = id.x, y = id.y;
    if (x >= di.y || y >= di.w)
        return;
    uint sp = id.z * (uint)di.x * (uint)di.w + (uint)y * (uint)di.x;
    uint dp = id.z * (uint)di.y * (uint)di.w;
    int s0 = starts[x];
    float a = 0.0f;
    for (int j = 0; j < di.z; j++)
        a += weights[x * di.z + j] * TLOAD(d0, sp + mirr(s0 + j, di.x));
    d1.Store((dp + (uint)y * (uint)di.y + x) * 4u, asuint(a));
}

// ---- cs_rs_v: vertical resize, planar f32 -> tensor, 3 planes ----
// 2 px per thread (x covers w/2; w is always even).
// di = {w, src rows, taps, dst rows}. d0 = f32 in, d1 = tensor out.
[numthreads(32, 8, 1)]
void cs_rs_v(uint3 id : SV_DispatchThreadID)
{
    int w = di.x;
    int x0 = id.x * 2, y = id.y;
    if (x0 >= w || y >= di.w)
        return;
    uint sp = id.z * (uint)w * (uint)di.y;
    uint dp = id.z * (uint)w * (uint)di.w;
    int s0 = starts[y];
    float a[2];
    [unroll]
    for (int k = 0; k < 2; k++) {
        a[k] = 0.0f;
        for (int j = 0; j < di.z; j++)
            a[k] += weights[y * di.z + j] *
                    load_f32e(d0, sp + (uint)mirr(s0 + j, di.y) * (uint)w +
                                  (uint)(x0 + k));
    }
    uint e = dp + (uint)y * (uint)w + (uint)x0;
#if IO16
    d1.Store(e * 2u, f32tof16(a[0]) | (f32tof16(a[1]) << 16u));
#else
    d1.Store(e * 4u, asuint(a[0]));
    d1.Store((e + 1u) * 4u, asuint(a[1]));
#endif
}
)hlsl";
