// Shared by the image path's tiling kernels (tile_extract, tile_blend, band_flush): the GPU side
// of core/tiling's upscale_tiled on Vulkan (aji_img_vk.cpp). Matches the host path bit for bit:
// every multiply and add is `precise` (never fused), and fp16 converts by integer arithmetic as
// core/half.h does (GLSL leaves fp16 rounding and denormals to the implementation). 8- and 16-bit
// values travel packed in 32-bit words, each word written by one invocation.
//
// Each dispatch reads its parameters from entry pc.entry of the Params buffer (the shared pipeline
// layout has 16 bytes of push constants).

layout(push_constant) uniform PC { int entry; int u0; int u1; int u2; } pc;
layout(std430, binding = 7) readonly buffer Params { ivec4 params[]; };  // 4 ivec4 per entry

int prm(int k) { return params[pc.entry * 4 + (k >> 2)][k & 3]; }

int reflect_i(int i, int n)
{
    if (n == 1) return 0;
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0) i += period;
    return i < n ? i : period - i;
}

// core/half.h f32_to_f16: round to nearest even, denormals and NaN as the host does.
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

// fp16 bits to fp32, exactly (fp16 denormals are normal fp32 values).
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

// Flat invocation index over a 2-D grid of 256-invocation groups (one dimension caps at 65535).
uint flat_id()
{
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationIndex;
}
