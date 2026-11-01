// Direct four-input concatenation + 1x1 convolution. FP16 NHWC storage.
// The fixed weights remain OI (64 x 256); no concatenated activation exists.
typedef unsigned short half_bits;
__device__ __forceinline__ float hf(half_bits x) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(x)); return f;
}
__device__ __forceinline__ half_bits fh(float x) {
    half_bits h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x)); return h;
}
struct Acc32 {
    float d[4];
    __device__ void clear() { d[0]=d[1]=d[2]=d[3]=0; }
    __device__ void mma(unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    __device__ float get(int i) { return d[i]; }
};
struct Acc16 {
    unsigned d[2];
    __device__ void clear() { d[0]=d[1]=0; }
    __device__ void mma(unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
            : "+r"(d[0]),"+r"(d[1]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    __device__ float get(int i) { return hf((half_bits)(d[i/2] >> (16*(i%2)))); }
};
__device__ __forceinline__ void async16(half_bits *dst,const half_bits *src,int bytes) {
    unsigned address=(unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(src),"r"(bytes) : "memory");
}
template<int M>
__device__ __forceinline__ void load_stage(half_bits *a,half_bits *b,const half_bits *x,
    const half_bits *weights,int stage) {
    #pragma unroll
    for(int j=0;j<2;j++) {
        int vector=threadIdx.x+j*256,row=vector/4,k=(vector%4)*8;
        int pixel=blockIdx.x*128+row;
        const half_bits *src=pixel<M ? x+pixel*64+(stage%2)*32+k : x;
        async16(a+row*40+k,src,pixel<M?16:0);
    }
    int oc=threadIdx.x/4,k=(threadIdx.x%4)*8;
    async16(b+oc*40+k,weights+oc*256+stage*32+k,16);
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ unsigned pair(const half_bits *p) { return *(const unsigned*)p; }
template<int M,class Acc>
__device__ __forceinline__ void body(const half_bits *x0,const half_bits *x1,
    const half_bits *x2,const half_bits *x3,const half_bits *weights,const half_bits *bias,
    half_bits *output,half_bits *storage) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int wm=(warp/2)*32,wn=(warp%2)*32,ar=lane/4,ak=2*(lane%4);
    Acc accum[2][4];
    #pragma unroll
    for(int m=0;m<2;m++) {
        #pragma unroll
        for(int n=0;n<4;n++) accum[m][n].clear();
    }
    load_stage<M>(storage,storage+128*40,x0,weights,0);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    #pragma unroll 1
    for(int stage=0;stage<8;stage++) {
        if(stage<7) {
            int next=stage+1;
            const half_bits *x=next<2?x0:(next<4?x1:(next<6?x2:x3));
            half_bits *s=storage+(next%2)*192*40;
            load_stage<M>(s,s+128*40,x,weights,next);
        }
        const half_bits *a=storage+(stage%2)*192*40;
        const half_bits *b=a+128*40;
        #pragma unroll
        for(int k=0;k<32;k+=16) {
            #pragma unroll
            for(int m=0;m<2;m++) {
                const half_bits *ap=a+(wm+m*16+ar)*40+k+ak;
                unsigned a0=pair(ap),a1=pair(ap+8*40),a2=pair(ap+8),a3=pair(ap+8*40+8);
                #pragma unroll
                for(int n=0;n<4;n++) {
                    const half_bits *bp=b+(wn+n*8+ar)*40+k+ak;
                    accum[m][n].mma(a0,a1,a2,a3,pair(bp),pair(bp+8));
                }
            }
        }
        if(stage<7) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
    #pragma unroll
    for(int m=0;m<2;m++) {
        #pragma unroll
        for(int n=0;n<4;n++) {
            #pragma unroll
            for(int r=0;r<2;r++) {
                int pixel=blockIdx.x*128+wm+m*16+ar+r*8,col=wn+n*8+ak;
                if(pixel<M) {
                    unsigned lo=fh(accum[m][n].get(r*2)+hf(bias[col]));
                    unsigned hi=fh(accum[m][n].get(r*2+1)+hf(bias[col+1]));
                    *(unsigned*)(output+pixel*64+col)=lo|(hi<<16);
                }
            }
        }
    }
}
#define ENTRY(name,M,Acc) \
extern "C" __global__ __launch_bounds__(256) void name(const half_bits *x0,const half_bits *x1, \
    const half_bits *x2,const half_bits *x3,const half_bits *weights,const half_bits *bias,half_bits *output) { \
    extern __shared__ __align__(16) unsigned char scratch[]; \
    body<M,Acc>(x0,x1,x2,x3,weights,bias,output,(half_bits*)scratch); \
}
ENTRY(concat32,540*960,Acc32)
ENTRY(concat16,540*960,Acc16)
ENTRY(concat16_540p,270*480,Acc16)
ENTRY(concat16_720p,360*640,Acc16)
ENTRY(test32,17,Acc32)
ENTRY(test16,17,Acc16)

ENTRY(concat16_360p,180*240,Acc16)
ENTRY(concat16_480p,240*320,Acc16)
ENTRY(concat16_576p,288*384,Acc16)
