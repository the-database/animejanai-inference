#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <cuda.h>
#include <NvInferRuntime.h>
#include <cstdio>
#include <new>
#include <string>
#include <cstdint>
#include "parameters.h"
#include "kernel_ptx.h"

using namespace nvinfer1;
static PluginFieldCollection fields{0,nullptr};
static bool shape(Dims const& d) {
    return d.nbDims==4 && d.d[0]==1 && d.d[1]==48 &&
        ((d.d[2]==180 && d.d[3]==240) || (d.d[2]==240 && d.d[3]==320) ||
         (d.d[2]==288 && d.d[3]==384) || (d.d[2]==270 && d.d[3]==480) || (d.d[2]==360 && d.d[3]==640) ||
         (d.d[2]==540 && d.d[3]==960));
}
static bool sameShape(Dims const& a,Dims const& b) {
    return shape(a)&&shape(b)&&a.d[2]==b.d[2]&&a.d[3]==b.d[3];
}
static bool ok(CUresult r) {
    if(r==CUDA_SUCCESS)return true;
    const char *s=nullptr;cuGetErrorString(r,&s);
    std::fprintf(stderr,"[performance-plugin] CUDA %d: %s\n",int(r),s?s:"unknown");return false;
}
class Performance final: public IPluginV3,public IPluginV3OneCore,public IPluginV3OneBuild,public IPluginV3OneRuntime {
    CUmodule module=nullptr;CUfunction kernel[6]={};CUdeviceptr weights=0,bias=0;
public:
    ~Performance() override {if(weights)cuMemFree(weights);if(bias)cuMemFree(bias);if(module)cuModuleUnload(module);}
    bool init() {
        return ok(cuModuleLoadDataEx(&module,kPtx,0,nullptr,nullptr)) &&
            ok(cuModuleGetFunction(&kernel[0],module,"concat16_540p")) &&
            ok(cuModuleGetFunction(&kernel[1],module,"concat16_720p")) &&
            ok(cuModuleGetFunction(&kernel[2],module,"concat16")) &&
            ok(cuModuleGetFunction(&kernel[3],module,"concat16_360p")) &&
            ok(cuModuleGetFunction(&kernel[4],module,"concat16_480p")) &&
            ok(cuModuleGetFunction(&kernel[5],module,"concat16_576p")) &&
            ok(cuMemAlloc(&weights,sizeof(kWeights))) && ok(cuMemAlloc(&bias,sizeof(kBias))) &&
            ok(cuMemcpyHtoD(weights,kWeights,sizeof(kWeights))) && ok(cuMemcpyHtoD(bias,kBias,sizeof(kBias)));
    }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        switch(t){case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);default:return nullptr;}
    }
    IPluginV3* clone() noexcept override {return new(std::nothrow) Performance;}
    char const* getPluginName() const noexcept override{return "AJPerformanceConcatFP16";}
    char const* getPluginVersion() const noexcept override{return "2";}
    char const* getPluginNamespace() const noexcept override{return "";}
    int32_t getNbOutputs() const noexcept override{return 1;}
    int32_t getOutputDataTypes(DataType* out,int32_t no,DataType const* in,int32_t ni) const noexcept override {
        if(ni!=4||no!=1)return 1;
        for(int i=0;i<4;i++)if(in[i]!=DataType::kHALF)return 1;
        out[0]=DataType::kHALF;return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in,int32_t ni,DimsExprs const*,int32_t ns,
        DimsExprs* out,int32_t no,IExprBuilder&) noexcept override {
        if(ni!=4||no!=1||ns!=0)return 1;out[0]=in[0];return 0;
    }
    bool supportsFormatCombination(int32_t pos,DynamicPluginTensorDesc const* io,int32_t ni,int32_t no) noexcept override {
        return ni==4&&no==1&&pos>=0&&pos<5&&io[pos].desc.type==DataType::kHALF&&io[pos].desc.format==TensorFormat::kHWC;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in,int32_t ni,
        DynamicPluginTensorDesc const* out,int32_t no) noexcept override {
        if(ni!=4||no!=1)return 1;
        for(int i=0;i<4;i++)if(!sameShape(in[i].min,out[0].min)||!sameShape(in[i].max,out[0].max))return 1;
        return shape(out[0].min)&&shape(out[0].max)?0:1;
    }
    int32_t onShapeChange(PluginTensorDesc const* in,int32_t ni,PluginTensorDesc const* out,int32_t no) noexcept override {
        if(ni!=4||no!=1)return 1;
        for(int i=0;i<4;i++)if(!sameShape(in[i].dims,out[0].dims)||in[i].type!=DataType::kHALF||in[i].format!=TensorFormat::kHWC)return 1;
        return shape(out[0].dims)&&out[0].type==DataType::kHALF&&out[0].format==TensorFormat::kHWC?0:1;
    }
    int32_t enqueue(PluginTensorDesc const* desc,PluginTensorDesc const*,void const* const* in,
        void* const* out,void*,cudaStream_t stream) noexcept override {
        CUdeviceptr x[4],y=(CUdeviceptr)out[0];for(int i=0;i<4;i++)x[i]=(CUdeviceptr)in[i];
        void* args[]={&x[0],&x[1],&x[2],&x[3],&weights,&bias,&y};
        const int h=int(desc[0].dims.d[2]),w=int(desc[0].dims.d[3]);
        const int index=h==180?3:(h==240?4:(h==288?5:(h==270?0:(h==360?1:2))));
        return kernel[index]&&ok(cuLaunchKernel(kernel[index],(h*w+127)/128,1,1,256,1,1,16896,(CUstream)stream,args,nullptr))?0:1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
        auto* p=new(std::nothrow) Performance;if(!p)return nullptr;
        if(!p->init()){delete p;return nullptr;}return p;
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override{return &fields;}
};
class Creator final:public IPluginCreatorV3One {
public:
    IPluginV3* createPlugin(char const*,PluginFieldCollection const*,TensorRTPhase) noexcept override {return new(std::nothrow) Performance;}
    PluginFieldCollection const* getFieldNames() noexcept override{return &fields;}
    char const* getPluginName() const noexcept override{return "AJPerformanceConcatFP16";}
    char const* getPluginVersion() const noexcept override{return "2";}
    char const* getPluginNamespace() const noexcept override{return "";}
};
static Creator creator;
extern "C" __declspec(dllexport) int initPerformancePlugin();
extern "C" __declspec(dllexport) void setLoggerFinder(ILoggerFinder*) {}
extern "C" __declspec(dllexport) IPluginCreatorInterface* const* getCreators(int32_t& count) {
    // Tiling evaluates candidate graphs through the runtime registry as well as
    // the builder registry populated by trtexec's dynamicPlugins option.
    initPerformancePlugin();
    static IPluginCreatorInterface* creators[]={&creator};count=1;return creators;
}
extern "C" __declspec(dllexport) int initPerformancePlugin() {
    static bool registered=false;
    if(registered)return 1;
    HMODULE trt=GetModuleHandleW(L"nvinfer_11.dll");if(!trt)return 0;
    using GetRegistry=IPluginRegistry*(*)() noexcept;
    auto get=(GetRegistry)GetProcAddress(trt,"getPluginRegistry");
    registered=get && get()->registerCreator(creator,"");
    return registered;
}
