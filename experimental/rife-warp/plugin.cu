#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <new>

using namespace nvinfer1;

// RIFE's ONNX GridSample: bilinear, border padding, align_corners=1.
// The source is FP16 followed by a lossless cast to FP32, and the grid
// result is immediately cast back to FP16. Keep the arithmetic FP32.
__global__ void warp(const __half *src, const float *grid, __half *out,
                     int channels, int h, int w)
{
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=h*w)return;
    int channel=blockIdx.y, batch=channel/channels;
    const float *g=grid+2*(batch*h*w+pixel);
    float x=((g[0]+1.0f)*0.5f)*(w-1);
    float y=((g[1]+1.0f)*0.5f)*(h-1);
    x=fminf(fmaxf(x,0.0f),float(w-1));
    y=fminf(fmaxf(y,0.0f),float(h-1));
    int x0=int(floorf(x)),y0=int(floorf(y));
    int x1=min(x0+1,w-1),y1=min(y0+1,h-1);
    float dx=x-x0,dy=y-y0;
    const __half *s=src+channel*h*w;
    float v=__half2float(s[y0*w+x0])*((1.0f-dx)*(1.0f-dy));
    v=fmaf(__half2float(s[y0*w+x1]),dx*(1.0f-dy),v);
    v=fmaf(__half2float(s[y1*w+x0]),(1.0f-dx)*dy,v);
    v=fmaf(__half2float(s[y1*w+x1]),dx*dy,v);
    out[channel*h*w+pixel]=__float2half_rn(v);
}

static PluginFieldCollection fields{0,nullptr};
class Warp final: public IPluginV3,public IPluginV3OneCore,
                  public IPluginV3OneBuild,public IPluginV3OneRuntime {
public:
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        switch(t){case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);
        default:return nullptr;}
    }
    IPluginV3* clone() noexcept override{return new(std::nothrow) Warp;}
    char const* getPluginName() const noexcept override{return "AJRifeWarp";}
    char const* getPluginVersion() const noexcept override{return "1";}
    char const* getPluginNamespace() const noexcept override{return "";}
    int32_t getNbOutputs() const noexcept override{return 1;}
    int32_t getOutputDataTypes(DataType *out,int32_t no,DataType const *in,int32_t ni) const noexcept override {
        if(ni!=2||no!=1||in[0]!=DataType::kHALF||in[1]!=DataType::kFLOAT)return 1;
        out[0]=DataType::kHALF;return 0;
    }
    int32_t getOutputShapes(DimsExprs const *in,int32_t ni,DimsExprs const*,int32_t ns,
                           DimsExprs *out,int32_t no,IExprBuilder&) noexcept override {
        if(ni!=2||no!=1||ns||in[0].nbDims!=4||in[1].nbDims!=4)return 1;
        out[0]=in[0];out[0].d[2]=in[1].d[1];out[0].d[3]=in[1].d[2];return 0;
    }
    bool supportsFormatCombination(int32_t pos,DynamicPluginTensorDesc const *io,int32_t ni,int32_t no) noexcept override {
        return ni==2&&no==1&&pos>=0&&pos<3&&io[pos].desc.format==TensorFormat::kLINEAR&&
            io[pos].desc.type==(pos==1?DataType::kFLOAT:DataType::kHALF);
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*,int32_t ni,DynamicPluginTensorDesc const*,int32_t no) noexcept override {
        return ni==2&&no==1?0:1;
    }
    int32_t onShapeChange(PluginTensorDesc const *in,int32_t ni,PluginTensorDesc const *out,int32_t no) noexcept override {
        if(ni!=2||no!=1||in[0].dims.nbDims!=4||in[1].dims.nbDims!=4)return 1;
        auto a=in[0].dims,b=in[1].dims;
        return a.d[0]==b.d[0]&&a.d[2]==b.d[1]&&a.d[3]==b.d[2]&&b.d[3]==2?0:1;
    }
    int32_t enqueue(PluginTensorDesc const *desc,PluginTensorDesc const*,void const *const *in,
                    void *const *out,void*,cudaStream_t stream) noexcept override {
        auto d=desc[0].dims;int n=int(d.d[0]),c=int(d.d[1]),h=int(d.d[2]),w=int(d.d[3]);
        warp<<<dim3((h*w+255)/256,n*c),256,0,stream>>>(
            (const __half*)in[0],(const float*)in[1],(__half*)out[0],c,h,w);
        return int(cudaGetLastError());
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override{return clone();}
    PluginFieldCollection const *getFieldsToSerialize() noexcept override{return &fields;}
};
class Creator final:public IPluginCreatorV3One {
public:
    IPluginV3* createPlugin(char const*,PluginFieldCollection const*,TensorRTPhase) noexcept override{return new(std::nothrow) Warp;}
    PluginFieldCollection const* getFieldNames() noexcept override{return &fields;}
    char const* getPluginName() const noexcept override{return "AJRifeWarp";}
    char const* getPluginVersion() const noexcept override{return "1";}
    char const* getPluginNamespace() const noexcept override{return "";}
};
static Creator creator;
extern "C" __declspec(dllexport) int initRifePlugin() {
    static bool registered=false;if(registered)return 1;
    using GetRegistry=IPluginRegistry*(*)() noexcept;
    auto fn=(GetRegistry)GetProcAddress(GetModuleHandleW(L"nvinfer_11.dll"),"getPluginRegistry");
    registered=fn&&fn()->registerCreator(creator,"");return registered;
}
extern "C" __declspec(dllexport) void setLoggerFinder(ILoggerFinder*) {}
extern "C" __declspec(dllexport) IPluginCreatorInterface* const* getCreators(int32_t& count) {
    initRifePlugin();static IPluginCreatorInterface* creators[]={&creator};count=1;return creators;
}
