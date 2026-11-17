// Shared by the video frame kernels (frame_*.comp), the Vulkan ports of the DirectML backend's
// HLSL pre/post kernels (src/dml_shaders.h), themselves ports of the CUDA ones (kernels.cu): the
// same weight tables (resample.h), mirror boundaries and round-half-even quantization. Not
// bit-identical across APIs (division rounds differently): within one level of the DirectML path,
// as that is of the CUDA one. Planes and tensors are raw buffers of 32-bit words;
// the model tensor is fp16 NCHW.
//
// Parameters: entry pc.entry of the Params buffer, laid out as the HLSL KernArgs: di[4], dj[4],
// kr, kb, yoff, yscale, coff, cscale, qdiv, qmax. Bindings: 0 starts (int), 1 weights (float),
// 2 d0, 3 d1, 4 d2 (the HLSL u0..u2), 7 Params. Workgroups of 32 x 8 (x, y), z per plane.
// Defines: T16 (raw video samples u16 / P010, else u8 / NV12), TPX where a kernel needs it.

layout(local_size_x = 32, local_size_y = 8) in;

layout(push_constant) uniform PC { int entry; int u0; int u1; int u2; } pc;
layout(std430, binding = 7) readonly buffer Params { ivec4 params[]; };
layout(std430, binding = 0) readonly buffer Starts { int starts[]; };
layout(std430, binding = 1) readonly buffer Weights { float weights[]; };
layout(std430, binding = 2) buffer D0 { uint d0[]; };
layout(std430, binding = 3) buffer D1 { uint d1[]; };
layout(std430, binding = 4) buffer D2 { uint d2[]; };

ivec4 DI() { return params[pc.entry * 4]; }
ivec4 DJ() { return params[pc.entry * 4 + 1]; }
float kr() { return intBitsToFloat(params[pc.entry * 4 + 2].x); }
float kb() { return intBitsToFloat(params[pc.entry * 4 + 2].y); }
float yoff() { return intBitsToFloat(params[pc.entry * 4 + 2].z); }
float yscale() { return intBitsToFloat(params[pc.entry * 4 + 2].w); }
float coff() { return intBitsToFloat(params[pc.entry * 4 + 3].x); }
float cscale() { return intBitsToFloat(params[pc.entry * 4 + 3].y); }
float qdiv() { return intBitsToFloat(params[pc.entry * 4 + 3].z); }
float qmax() { return intBitsToFloat(params[pc.entry * 4 + 3].w); }

int mirr(int i, int n)
{
    i = i < 0 ? -i - 1 : i;
    i = i >= n ? 2 * n - 1 - i : i;
    return clamp(i, 0, n - 1);
}

// round-half-even (CUDA rintf), as the HLSL rint_even
float rint_even(float v)
{
    const float fl = floor(v);
    const float fr = v - fl;
    float r = fl + (fr > 0.5 ? 1.0 : 0.0);
    if (fr == 0.5) r = fl + ((fl - 2.0 * trunc(fl / 2.0)) != 0.0 ? 1.0 : 0.0);
    return r;
}

float quant(float v, float d, float m) { return min(max(rint_even(v / d), 0.0), m) * d; }

// core/half.h's conversions, exactly (GLSL leaves fp16 rounding and denormals open)
uint f32_to_f16(float f)
{
    const uint x = floatBitsToUint(f);
    const uint sign = (x >> 16) & 0x8000u;
    uint mant = x & 0x7fffffu;
    int e = int((x >> 23) & 0xffu);
    if (e == 0xff) return sign | 0x7c00u | (mant != 0u ? 0x200u : 0u);
    e = e - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00u;
    if (e <= 0) {
        if (e < -10) return sign;
        mant |= 0x800000u;
        const uint shift = uint(14 - e);
        uint h = mant >> shift;
        const uint rem = mant & ((1u << shift) - 1u);
        const uint mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (h & 1u) != 0u)) h++;
        return sign | h;
    }
    uint h = sign | (uint(e) << 10) | (mant >> 13);
    const uint rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h++;
    return h;
}

float f16_to_f32(uint h)
{
    const uint sign = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    if (e == 0u) {
        if (mant == 0u) return uintBitsToFloat(sign);
        uint k = 0u;
        while ((mant & 0x400u) == 0u) {
            mant <<= 1;
            k++;
        }
        return uintBitsToFloat(sign | ((113u - k) << 23) | ((mant & 0x3ffu) << 13));
    }
    if (e == 31u) return uintBitsToFloat(sign | 0x7f800000u | (mant << 13));
    return uintBitsToFloat(sign | ((e + 112u) << 23) | (mant << 13));
}

#if T16
#define RAWLOAD(buf, off) float((buf[(off) >> 2] >> (((off) & 2u) * 8u)) & 0xffffu)
#define RAWSIZE 2u
#else
#define RAWLOAD(buf, off) float((buf[(off) >> 2] >> (((off) & 3u) * 8u)) & 0xffu)
#define RAWSIZE 1u
#endif
#define F32(buf, elem) uintBitsToFloat(buf[elem])
#define TLOAD(buf, elem) f16_to_f32((buf[(elem) >> 1] >> (((elem) & 1u) * 16u)) & 0xffffu)
