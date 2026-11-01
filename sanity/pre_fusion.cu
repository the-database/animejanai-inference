// Fused preprocessing must produce exactly the original fp16 model input.
// Include the CUDA implementation to exercise its private filter plan without
// adding a testing API to the library. Build this translation unit alone.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../src/kernels.cu"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); exit(1); } } while(0)

// vertical resample of N planar fp32 planes (gridDim.z = N)
__global__ void reference_v_f32(const float *src, int w, pass py, float *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= py.dst)
        return;
    const float *sp = src + (size_t)blockIdx.z * w * py.src;
    const float *wt = py.wt + (size_t)y * py.taps;
    const int s0 = py.start[y];
    float a = 0.0f;
    for (int j = 0; j < py.taps; j++)
        a += wt[j] * sp[(size_t)mirr(s0 + j, py.src) * w + x];
    dst[(size_t)blockIdx.z * w * py.dst + (size_t)y * w + x] = a;
}

template <typename T>
__global__ void reference_pre_combine(const uint8_t *y_plane, ptrdiff_t y_stride,
                              const float *uvf, int w, int h, aji_csp csp,
                              __half *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const size_t plane = (size_t)w * h, idx = (size_t)y * w + x;

    const T *yrow = (const T *)(y_plane + (size_t)y * y_stride);
    const float Y = ((float)yrow[x] - csp.yoff) / csp.yscale;
    const float U = (uvf[idx] - csp.coff) / csp.cscale;
    const float V = (uvf[plane + idx] - csp.coff) / csp.cscale;

    const float kg = 1.0f - csp.kr - csp.kb;
    const float r = Y + 2.0f * (1.0f - csp.kr) * V;
    const float b = Y + 2.0f * (1.0f - csp.kb) * U;
    const float g = Y - (2.0f * csp.kb * (1.0f - csp.kb) * U +
                         2.0f * csp.kr * (1.0f - csp.kr) * V) / kg;

    dst[idx]             = __float2half(r);
    dst[plane + idx]     = __float2half(g);
    dst[2 * plane + idx] = __float2half(b);
}

void reference_pre(aji_plan *p, uint8_t *src, size_t stride, aji_csp csp,
                   float *uv, __half *rgb, cudaStream_t s)
{
    int w=p->w,h=p->h,ch=h/2;
    if(p->format==AJI_FMT_NV12)
        k_uv_h<uint8_t><<<GRID(w,ch,1),0,s>>>(src+stride*h,stride,ch,p->ph,p->tmp0);
    else
        k_uv_h<uint16_t><<<GRID(w,ch,1),0,s>>>(src+stride*h,stride,ch,p->ph,p->tmp0);
    reference_v_f32<<<GRID(w,h,2),0,s>>>(p->tmp0,w,p->pv,uv);
    if(p->format==AJI_FMT_NV12)
        reference_pre_combine<uint8_t><<<GRID(w,h,1),0,s>>>(src,stride,uv,w,h,csp,rgb);
    else
        reference_pre_combine<uint16_t><<<GRID(w,h,1),0,s>>>(src,stride,uv,w,h,csp,rgb);
    CK(cudaGetLastError());
}

void pre_case(int w,int h,int format,int sit,int matrix,int range,int filter)
{
    cudaStream_t s; CK(cudaStreamCreate(&s));
    aji_plan *p=aji_pre_plan_create(format,w,h,sit,filter);
    if(!p) exit(2);
    const int bpp=format==AJI_FMT_NV12?1:2;
    size_t stride=w*bpp+64, bytes=stride*(h+h/2), n=size_t(w)*h*3;
    std::vector<uint8_t> input(bytes);
    uint32_t rng=18773;
    for(size_t i=0;i<bytes;i++) { rng=rng*1664525u+1013904223u; input[i]=rng>>24; }
    if(bpp==2) for(size_t i=0;i<bytes;i+=2) input[i]&=0xc0;
    uint8_t *src; __half *a,*b; float *uv;
    CK(cudaMalloc(&src,bytes)); CK(cudaMemcpy(src,input.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a,n*2)); CK(cudaMalloc(&b,n*2)); CK(cudaMalloc(&uv,size_t(w)*h*8));
    const auto csp=aji_make_csp(format,matrix,range);
    auto old=[&]{reference_pre(p,src,stride,csp,uv,a,s);};
    auto fused=[&]{CK((cudaError_t)aji_run_pre(p,src,stride,src+stride*h,stride,&csp,b,s));};
    old(); fused(); CK(cudaStreamSynchronize(s));
    std::vector<uint16_t> ha(n),hb(n);
    CK(cudaMemcpy(ha.data(),a,n*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(),b,n*2,cudaMemcpyDeviceToHost));
    size_t mismatches=0;
    for(size_t i=0;i<n;i++) mismatches+=ha[i]!=hb[i];
    printf("{\"kind\":\"pre\",\"w\":%d,\"h\":%d,\"format\":%d,\"siting\":%d,\"matrix\":%d,\"range\":%d,\"filter\":%d,\"mismatches\":%zu",w,h,format,sit,matrix,range,filter,mismatches);
    printf("}\n"); fflush(stdout);
    if(mismatches) exit(3);
    CK(cudaFree(src)); CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(uv));
    aji_plan_destroy(p); CK(cudaStreamDestroy(s));
}

int main()
{
    for(int h:{540,720,1080})
        pre_case(h*16/9,h,AJI_FMT_NV12,AJI_SITING_LEFT,AJI_MATRIX_BT709,AJI_RANGE_LIMITED,AJI_FILTER_SPLINE36);
    pre_case(1920,1080,AJI_FMT_P010,AJI_SITING_LEFT,AJI_MATRIX_BT709,AJI_RANGE_LIMITED,AJI_FILTER_SPLINE36);
    pre_case(66,34,AJI_FMT_NV12,AJI_SITING_CENTER,AJI_MATRIX_BT601,AJI_RANGE_FULL,AJI_FILTER_SPLINE36);
    pre_case(130,18,AJI_FMT_P010,AJI_SITING_TOPLEFT,AJI_MATRIX_BT2020,AJI_RANGE_FULL,AJI_FILTER_SPLINE36);
    pre_case(66,34,AJI_FMT_NV12,AJI_SITING_TOPLEFT,AJI_MATRIX_BT2020,AJI_RANGE_LIMITED,AJI_FILTER_BILINEAR);
    pre_case(130,18,AJI_FMT_P010,AJI_SITING_CENTER,AJI_MATRIX_BT601,AJI_RANGE_LIMITED,AJI_FILTER_BILINEAR);

}
