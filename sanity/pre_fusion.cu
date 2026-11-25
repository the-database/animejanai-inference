// Fused preprocessing must produce exactly the original fp16 model input.
// Include the CUDA implementation to exercise its private filter plan without
// adding a testing API to the library. Build this translation unit alone.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../src/kernels.cu"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); exit(1); } } while(0)

// horizontal resample of interleaved raw chroma -> 2 planar fp32 (raw units)
template <typename T>
__global__ void reference_uv_h(const uint8_t *uv, ptrdiff_t stride, int ch, pass px,
                       float *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= px.dst || y >= ch)
        return;
    const T *row = (const T *)(uv + (size_t)y * stride);
    const float *w = px.wt + (size_t)x * px.taps;
    const int s0 = px.start[x];
    float u = 0.0f, v = 0.0f;
    for (int j = 0; j < px.taps; j++) {
        const int sx = mirr(s0 + j, px.src);
        u += w[j] * row[2 * sx];
        v += w[j] * row[2 * sx + 1];
    }
    const size_t plane = (size_t)px.dst * ch, idx = (size_t)y * px.dst + x;
    dst[idx] = u;
    dst[plane + idx] = v;
}

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

void reference_pre(aji_plan *p, uint8_t *src, size_t stride, size_t uv_stride, aji_csp csp,
                   float *horizontal, float *uv, __half *rgb, cudaStream_t s)
{
    int w=p->w,h=p->h,ch=h/2;
    if(p->format==AJI_FMT_NV12)
        reference_uv_h<uint8_t><<<GRID(w,ch,1),0,s>>>(src+stride*h,uv_stride,ch,p->ph,horizontal);
    else
        reference_uv_h<uint16_t><<<GRID(w,ch,1),0,s>>>(src+stride*h,uv_stride,ch,p->ph,horizontal);
    reference_v_f32<<<GRID(w,h,2),0,s>>>(horizontal,w,p->pv,uv);
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
    if(!p || p->tmp0) { fprintf(stderr,"expected a pre plan without a frame temporary\n"); exit(2); }
    const int bpp=format==AJI_FMT_NV12?1:2;
    size_t stride=w*bpp+64, uv_stride=w*bpp+128, bytes=stride*h+uv_stride*(h/2), n=size_t(w)*h*3;
    std::vector<uint8_t> input(bytes);
    uint32_t rng=18773;
    for(size_t i=0;i<bytes;i++) { rng=rng*1664525u+1013904223u; input[i]=rng>>24; }
    if(bpp==2) for(size_t i=0;i<bytes;i+=2) input[i]&=0xc0;
    uint8_t *src; __half *a,*b; float *horizontal,*uv;
    CK(cudaMalloc(&src,bytes)); CK(cudaMemcpy(src,input.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a,n*2)); CK(cudaMalloc(&b,n*2)); CK(cudaMalloc(&uv,size_t(w)*h*8));
    CK(cudaMalloc(&horizontal,2*size_t(w)*(h/2)*sizeof(float)));
    const auto csp=aji_make_csp(format,matrix,range);
    auto old=[&]{reference_pre(p,src,stride,uv_stride,csp,horizontal,uv,a,s);};
    auto fused=[&]{CK((cudaError_t)aji_run_pre(p,src,stride,src+stride*h,uv_stride,&csp,b,s));};
    old(); fused(); CK(cudaStreamSynchronize(s));
    std::vector<uint16_t> ha(n),hb(n);
    CK(cudaMemcpy(ha.data(),a,n*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(),b,n*2,cudaMemcpyDeviceToHost));
    size_t mismatches=0;
    for(size_t i=0;i<n;i++) mismatches+=ha[i]!=hb[i];
    printf("{\"kind\":\"pre\",\"w\":%d,\"h\":%d,\"format\":%d,\"siting\":%d,\"matrix\":%d,\"range\":%d,\"filter\":%d,\"mismatches\":%zu",w,h,format,sit,matrix,range,filter,mismatches);
    printf("}\n"); fflush(stdout);
    if(mismatches) exit(3);
    CK(cudaFree(src)); CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(horizontal)); CK(cudaFree(uv));
    aji_plan_destroy(p); CK(cudaStreamDestroy(s));
}

void pre_tile_bounds()
{
    // The final block has two output rows. Its span determines the allocation.
    auto w = aji_resample::compute(5, 10, 0.0, AJI_FILTER_SPLINE36);
    w.start = {-2, -2, -2, -1, -1, -1, -1, -1, 0, 2};
    pass vertical;
    int rows = 0;
    if (!build_pre_vertical(&vertical, w, rows) || rows != w.taps + 2) {
        fprintf(stderr, "wrong shared row count for partial input tile\n");
        exit(2);
    }
    free_pass(&vertical);

    auto reject = [&](int y, int start, const char *label) {
        const int saved = w.start[y];
        w.start[y] = start;
        if (build_pre_vertical(&vertical, w, rows)) {
            fprintf(stderr, "accepted invalid vertical table: %s\n", label);
            exit(2);
        }
        w.start[y] = saved;
        printf("rejected invalid vertical table: %s\n", label);
    };
    reject(4, -2, "decrease inside tile");
    reject(8, -2, "decrease at tile boundary");
    reject(9, BLOCK_Y / 2 + 1, "excess span in final partial tile");
}

int main()
{
    pre_tile_bounds();
    for(int h:{540,720,1080})
        pre_case(h*16/9,h,AJI_FMT_NV12,AJI_SITING_LEFT,AJI_MATRIX_BT709,AJI_RANGE_LIMITED,AJI_FILTER_SPLINE36);
    pre_case(1920,1080,AJI_FMT_P010,AJI_SITING_LEFT,AJI_MATRIX_BT709,AJI_RANGE_LIMITED,AJI_FILTER_SPLINE36);
    pre_case(66,34,AJI_FMT_NV12,AJI_SITING_CENTER,AJI_MATRIX_BT601,AJI_RANGE_FULL,AJI_FILTER_SPLINE36);
    pre_case(130,18,AJI_FMT_P010,AJI_SITING_TOPLEFT,AJI_MATRIX_BT2020,AJI_RANGE_FULL,AJI_FILTER_SPLINE36);
    pre_case(66,34,AJI_FMT_NV12,AJI_SITING_TOPLEFT,AJI_MATRIX_BT2020,AJI_RANGE_LIMITED,AJI_FILTER_BILINEAR);
    pre_case(130,18,AJI_FMT_P010,AJI_SITING_CENTER,AJI_MATRIX_BT601,AJI_RANGE_LIMITED,AJI_FILTER_BILINEAR);
    pre_case(1918,1078,AJI_FMT_P010,AJI_SITING_CENTER,AJI_MATRIX_BT709,AJI_RANGE_LIMITED,AJI_FILTER_SPLINE36);
    pre_case(5,3,AJI_FMT_NV12,AJI_SITING_TOPLEFT,AJI_MATRIX_BT709,AJI_RANGE_FULL,AJI_FILTER_SPLINE36);
}
