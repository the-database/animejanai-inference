#include <cuda_runtime.h>
#include <cmath>
#include "analysis_kernels.h"

namespace {
__device__ float at(const void *plane, ptrdiff_t pitch, int x, int y, int w, int h, bool word, int channels = 1, int channel = 0)
{
    x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1));
    const unsigned char *row = (const unsigned char *)plane + y * pitch;
    return word ? ((const uint16_t *)row)[x * channels + channel] : row[x * channels + channel];
}
__device__ float bilinear(const void *plane, ptrdiff_t pitch, float x, float y, int w, int h, bool word, int channels = 1, int channel = 0)
{
    int ix = floorf(x), iy = floorf(y); float fx = x - ix, fy = y - iy;
    float a = at(plane,pitch,ix,iy,w,h,word,channels,channel), b = at(plane,pitch,ix+1,iy,w,h,word,channels,channel);
    float c = at(plane,pitch,ix,iy+1,w,h,word,channels,channel), d = at(plane,pitch,ix+1,iy+1,w,h,word,channels,channel);
    return (a+(b-a)*fx)*(1-fy)+(c+(d-c)*fx)*fy;
}
__device__ float3 rgb(aji_frame f, aji_csp c, float x, float y)
{
    bool word = f.format != AJI_FMT_NV12;
    bool planar = f.format == AJI_FMT_YUV444P16 || f.format == AJI_FMT_YUV444P10MSB || f.format == AJI_FMT_YUV444P12MSB;
    float Y = (bilinear(f.plane[0],f.stride[0],x,y,f.width,f.height,word)-c.yoff)/c.yscale;
    float u, v;
    if (planar) {
        u = bilinear(f.plane[1],f.stride[1],x,y,f.width,f.height,true);
        v = bilinear(f.plane[2],f.stride[2],x,y,f.width,f.height,true);
    } else {
        float cx = (x - (f.siting == AJI_SITING_CENTER ? 0.5f : 0.f)) * 0.5f;
        float cy = (y - (f.siting == AJI_SITING_TOPLEFT ? 0.f : 0.5f)) * 0.5f;
        u = bilinear(f.plane[1],f.stride[1],cx,cy,(f.width+1)/2,(f.height+1)/2,word,2,0);
        v = bilinear(f.plane[1],f.stride[1],cx,cy,(f.width+1)/2,(f.height+1)/2,word,2,1);
    }
    u=(u-c.coff)/c.cscale; v=(v-c.coff)/c.cscale;
    float r=Y+2*(1-c.kr)*v, b=Y+2*(1-c.kb)*u;
    float g=(Y-c.kr*r-c.kb*b)/(1-c.kr-c.kb);
    return make_float3(fminf(1,fmaxf(0,r)),fminf(1,fmaxf(0,g)),fminf(1,fmaxf(0,b)));
}
__global__ void color_kernel(aji_frame f, aji_csp c, int4 crop, int w, int h, uchar4 *out)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=w*h)return;
    float x=crop.x+(i%w+0.5f)*crop.z/w-0.5f, y=crop.y+(i/w+0.5f)*crop.w/h-0.5f;
    float3 v=rgb(f,c,x,y); out[i]=make_uchar4(lrintf(v.z*255),lrintf(v.y*255),lrintf(v.x*255),255);
}
__global__ void input_kernel(aji_frame f, aji_csp c, int4 crop, int iw, int ih, float *out)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=1024*1024)return;
    int x=i%1024-(1024-iw)/2,y=i/1024-(1024-ih)/2;
    float3 v=make_float3(0,0,0);
    if(x>=0&&x<iw&&y>=0&&y<ih)v=rgb(f,c,crop.x+(x+0.5f)*crop.z/iw-0.5f,crop.y+(y+0.5f)*crop.w/ih-0.5f);
    out[i]=v.x;out[i+1024*1024]=v.y;out[i+2*1024*1024]=v.z;
}
__global__ void mask_kernel(const float *input, int w, int h, int iw, int ih, uint16_t *output)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
    // Crop padding BEFORE resizing. Clamp within the unpadded mask rectangle.
    int left=(1024-iw)/2,top=(1024-ih)/2;
    float x=fminf(iw-1.f,fmaxf(0.f,(i%w+0.5f)*iw/w-0.5f))+left;
    float y=fminf(ih-1.f,fmaxf(0.f,(i/w+0.5f)*ih/h-0.5f))+top;
    int x0=floorf(x),y0=floorf(y),x1=min(x0+1,left+iw-1),y1=min(y0+1,top+ih-1);
    float fx=x-x0,fy=y-y0;
    float a=input[y0*1024+x0]*(1-fx)+input[y0*1024+x1]*fx;
    float b=input[y1*1024+x0]*(1-fx)+input[y1*1024+x1]*fx;
    float value=a*(1-fy)+b*fy;
    output[i]=(uint16_t)lrintf(fminf(1.f,fmaxf(0.f,value))*65535.f);
}
}
int aji_analysis_pre(const aji_frame *frame,const int crop[4],int w,int h,int iw,int ih,float *input,unsigned char *color,void *stream)
{
    aji_csp c=aji_make_csp(frame->format,frame->matrix,frame->range);
    int4 rect=make_int4(crop[0],crop[1],crop[2],crop[3]);
    color_kernel<<<(w*h+255)/256,256,0,(cudaStream_t)stream>>>(*frame,c,rect,w,h,(uchar4*)color);
    if(input)input_kernel<<<4096,256,0,(cudaStream_t)stream>>>(*frame,c,rect,iw,ih,input);
    return cudaGetLastError();
}
int aji_analysis_mask(const float *input,int w,int h,int iw,int ih,uint16_t *out,void *stream)
{
    mask_kernel<<<(w*h+255)/256,256,0,(cudaStream_t)stream>>>(input,w,h,iw,ih,out);
    return cudaGetLastError();
}
