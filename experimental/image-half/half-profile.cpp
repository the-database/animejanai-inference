// Isolated image-ABI conversion experiment; not a playback or model benchmark.
#include "../../src/core/half.h"
#include <immintrin.h>
#include <intrin.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>

__declspec(noinline) void scalar_pack(const float *s, uint16_t *d, size_t n) {
    for (size_t i=0; i<n; ++i) d[i]=ajicore::f32_to_f16(s[i]);
}
__declspec(noinline) void scalar_unpack(const uint16_t *s, float *d, size_t n) {
    for (size_t i=0; i<n; ++i) d[i]=ajicore::f16_to_f32(s[i]);
}
__declspec(noinline) void vector_pack(const float *s, uint16_t *d, size_t n) {
    size_t i=0;
    for (; i+8<=n; i+=8)
        _mm_storeu_si128((__m128i*)(d+i), _mm256_cvtps_ph(_mm256_loadu_ps(s+i), _MM_FROUND_TO_NEAREST_INT));
    for (; i<n; ++i) d[i]=ajicore::f32_to_f16(s[i]);
}
__declspec(noinline) void vector_unpack(const uint16_t *s, float *d, size_t n) {
    size_t i=0;
    for (; i+8<=n; i+=8)
        _mm256_storeu_ps(d+i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s+i))));
    for (; i<n; ++i) d[i]=ajicore::f16_to_f32(s[i]);
}
template<class F> double elapsed(F f) {
    auto begin=std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
int main() {
    int caps[4]; __cpuid(caps,1);
    if (!(caps[2]&(1<<29)) || !(caps[2]&(1<<27)) || (_xgetbv(0)&6)!=6) return 2;
    __cpuidex(caps,7,0);
    if (!(caps[1]&(1<<5))) return 2; // This standalone binary is built with /arch:AVX2.
    char cpu[49]={};
    for (int i=0;i<3;++i) {__cpuid(caps,0x80000002+i); memcpy(cpu+16*i,caps,16);}
    printf("{\"cpu\":\"%s\",\"scope\":\"CPU image-ABI FP16 conversion only\"}\n",cpu);

    // Check every finite half, including subnormals, signs and boundary values.
    std::vector<uint16_t> h;
    for (unsigned i=0;i<65536;++i) if ((i&0x7c00)!=0x7c00) h.push_back((uint16_t)i);
    std::vector<float> a(h.size()),b(h.size());
    scalar_unpack(h.data(),a.data(),h.size()); vector_unpack(h.data(),b.data(),h.size());
    if (memcmp(a.data(),b.data(),a.size()*sizeof(float))) return 3;
    std::vector<uint16_t> half_a(h.size()),half_b(h.size());
    scalar_pack(a.data(),half_a.data(),a.size()); vector_pack(a.data(),half_b.data(),a.size());
    if (half_a!=half_b || half_a!=h) return 4;

    // Report the known semantic gap rather than hiding non-finite inputs.
    std::vector<uint16_t> nf;
    for (unsigned i=0;i<65536;++i) if ((i&0x7c00)==0x7c00) nf.push_back((uint16_t)i);
    std::vector<float> nfa(nf.size()),nfb(nf.size());
    std::vector<uint16_t> nfha(nf.size()),nfhb(nf.size());
    scalar_unpack(nf.data(),nfa.data(),nf.size());
    vector_unpack(nf.data(),nfb.data(),nf.size());
    scalar_pack(nfa.data(),nfha.data(),nf.size());
    vector_pack(nfa.data(),nfhb.data(),nf.size());
    size_t pack_differences=0,unpack_differences=0;
    for(size_t i=0;i<nf.size();++i) {
        pack_differences += nfha[i]!=nfhb[i];
        unpack_differences += memcmp(&nfa[i],&nfb[i],sizeof(float))!=0;
    }
    printf("{\"nonfinite_patterns\":%zu,\"pack_bit_differences\":%zu,"
           "\"unpack_bit_differences\":%zu}\n",nf.size(),pack_differences,unpack_differences);

    for (int width : {512,1920}) {
        int height=width==512 ? 512 : 1080;
        size_t in_n=size_t(width)*height*3, out_n=in_n*4;
        std::vector<float> src(out_n), unpacked(out_n), reference(out_n);
        std::vector<uint16_t> packed(out_n), expected(out_n);
        uint32_t rng=7711;
        for (auto &v:src) {rng=rng*1664525u+1013904223u; v=float(rng>>8)*(1.2f/16777215.f)-.1f;}
        scalar_pack(src.data(),expected.data(),out_n); vector_pack(src.data(),packed.data(),out_n);
        if (expected!=packed) return 5;
        scalar_unpack(expected.data(),reference.data(),out_n); vector_unpack(packed.data(),unpacked.data(),out_n);
        if (memcmp(reference.data(),unpacked.data(),out_n*sizeof(float))) return 6;
        std::vector<double> sp,vp,su,vu;
        for (int round=-2;round<10;++round) {
            double p[2],u[2];
            for (int k=0;k<2;++k) {
                int order=(round&1)?1-k:k;
                p[order]=elapsed([&]{(order?vector_pack:scalar_pack)(src.data(),packed.data(),in_n);});
                u[order]=elapsed([&]{(order?vector_unpack:scalar_unpack)(packed.data(),unpacked.data(),out_n);});
            }
            if (round>=0) {sp.push_back(p[0]);vp.push_back(p[1]);su.push_back(u[0]);vu.push_back(u[1]);}
        }
        auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return (v[4]+v[5])*.5;};
        printf("{\"input_w\":%d,\"input_h\":%d,\"scale\":2,\"scalar_pack_ms\":%.6f,\"f16c_pack_ms\":%.6f,\"scalar_unpack_ms\":%.6f,\"f16c_unpack_ms\":%.6f,\"finite_parity\":true,\"image_values_parity\":true}\n",width,height,median(sp),median(vp),median(su),median(vu));
    }
    return 0;
}
