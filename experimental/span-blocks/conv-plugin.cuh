#include <cuda_runtime.h>
#include <type_traits>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "conv-parameters.h"

using ConvHalf=cutlass::half_t;
using ConvLayout=cutlass::layout::TensorNHWC;
struct ConvState {
    virtual ~ConvState()=default;
    virtual bool launch(void const*,void*,cudaStream_t)=0;
};
template<bool Relu,int TM,int ST,int WN> struct ConvStateImpl:ConvState {
    using Epilogue=std::conditional_t<Relu,
        cutlass::epilogue::thread::LinearCombinationRelu<ConvHalf,8,ConvHalf,float>,
        cutlass::epilogue::thread::LinearCombination<ConvHalf,8,ConvHalf,float>>;
    using Kernel=typename cutlass::conv::kernel::DefaultConv2dFprop<
        ConvHalf,ConvLayout,ConvHalf,ConvLayout,ConvHalf,ConvLayout,ConvHalf,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TM,64,32>,cutlass::gemm::GemmShape<64,WN,32>,cutlass::gemm::GemmShape<16,8,16>,
        Epilogue,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,ST,cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,cutlass::conv::StrideSupport::kUnity>::Kernel;
    using Op=cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
    Op op;
    typename Op::Arguments args;
    bool init(int h,int w,ConvHalf* weights,ConvHalf* bias){
        cutlass::Tensor4DCoord shape(1,h,w,64),filter(64,3,3,64);
        cutlass::conv::Conv2dProblemSize problem(shape,filter,{1,1,1,1},{1,1},{1,1},shape,cutlass::conv::Mode::kCrossCorrelation,1);
        args=typename Op::Arguments{problem,{nullptr,ConvLayout::packed(shape)},{weights,ConvLayout::packed(filter)},
            {bias,ConvLayout(ConvLayout::Stride(0))},{nullptr,ConvLayout::packed(shape)},{1.f,1.f}};
        return op.can_implement(args)==cutlass::Status::kSuccess&&op.get_workspace_size(args)==0&&op.initialize(args)==cutlass::Status::kSuccess;
    }
    bool launch(void const* input,void* output,cudaStream_t stream)override{
        args.ref_A.reset((ConvHalf*)input);args.ref_D.reset((ConvHalf*)output);
        return op.update(args)==cutlass::Status::kSuccess&&op.run(stream)==cutlass::Status::kSuccess;
    }
};
template<bool R,int M,int S,int N> ConvState* makeConvState(int h,int w,ConvHalf* weights,ConvHalf* bias){
    auto* s=new(std::nothrow) ConvStateImpl<R,M,S,N>;
    if(s&&!s->init(h,w,weights,bias)){delete s;return nullptr;}return s;
}
static bool convShape(Dims const& d){
    return d.nbDims==4&&d.d[0]==1&&d.d[1]==64&&
        ((d.d[2]==270&&d.d[3]==480)||(d.d[2]==360&&d.d[3]==640)||(d.d[2]==540&&d.d[3]==960));
}
class ConvPlugin final:public IPluginV3,public IPluginV3OneCore,public IPluginV3OneBuild,public IPluginV3OneRuntime {
    int index;ConvHalf *weights=nullptr,*bias=nullptr;ConvState* state=nullptr;
public:
    explicit ConvPlugin(int i):index(i){}
    ~ConvPlugin()override{delete state;if(weights)cudaFree(weights);if(bias)cudaFree(bias);}
    bool init(){
        return cudaMalloc(&weights,sizeof(kConvWeights[0]))==cudaSuccess&&cudaMalloc(&bias,sizeof(kConvBias[0]))==cudaSuccess&&
            cudaMemcpy(weights,kConvWeights[index],sizeof(kConvWeights[0]),cudaMemcpyHostToDevice)==cudaSuccess&&
            cudaMemcpy(bias,kConvBias[index],sizeof(kConvBias[0]),cudaMemcpyHostToDevice)==cudaSuccess;
    }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t)noexcept override{
        switch(t){case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);default:return nullptr;}
    }
    IPluginV3* clone()noexcept override{return new(std::nothrow) ConvPlugin(index);}
    char const* getPluginName()const noexcept override{return kConvNames[index];}
    char const* getPluginVersion()const noexcept override{return "1";}
    char const* getPluginNamespace()const noexcept override{return "";}
    int32_t getNbOutputs()const noexcept override{return 1;}
    int32_t getOutputDataTypes(DataType* out,int32_t no,DataType const* in,int32_t ni)const noexcept override{
        if(ni!=1||no!=1||in[0]!=DataType::kHALF)return 1;out[0]=DataType::kHALF;return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in,int32_t ni,DimsExprs const*,int32_t ns,DimsExprs* out,int32_t no,IExprBuilder&)noexcept override{
        if(ni!=1||no!=1||ns)return 1;out[0]=in[0];return 0;
    }
    bool supportsFormatCombination(int32_t pos,DynamicPluginTensorDesc const* io,int32_t ni,int32_t no)noexcept override{
        return ni==1&&no==1&&pos>=0&&pos<2&&io[pos].desc.type==DataType::kHALF&&io[pos].desc.format==TensorFormat::kHWC;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in,int32_t ni,DynamicPluginTensorDesc const* out,int32_t no)noexcept override{
        return ni==1&&no==1&&convShape(in[0].min)&&convShape(in[0].max)&&convShape(out[0].min)&&convShape(out[0].max)?0:1;
    }
    int32_t onShapeChange(PluginTensorDesc const* in,int32_t ni,PluginTensorDesc const* out,int32_t no)noexcept override{
        if(ni!=1||no!=1||!convShape(in[0].dims)||!convShape(out[0].dims))return 1;
        if(!weights||!bias)return 1;
        delete state;state=nullptr;
        int h=int(in[0].dims.d[2]),w=int(in[0].dims.d[3]);
#ifdef AJ_ADA_M
        state=kConvRelu[index]?makeConvState<true,AJ_ADA_M,AJ_ADA_S,AJ_ADA_N>(h,w,weights,bias):makeConvState<false,AJ_ADA_M,AJ_ADA_S,AJ_ADA_N>(h,w,weights,bias);
#else
        if(h==270)state=kConvRelu[index]?makeConvState<true,256,3,64>(h,w,weights,bias):makeConvState<false,256,3,64>(h,w,weights,bias);
        else {
#ifdef AJ_CONV_720_M256
            state=kConvRelu[index]?makeConvState<true,256,3,64>(h,w,weights,bias):makeConvState<false,256,3,64>(h,w,weights,bias);
#else
            state=kConvRelu[index]?makeConvState<true,128,4,32>(h,w,weights,bias):makeConvState<false,128,4,32>(h,w,weights,bias);
#endif
        }
#endif
        return state?0:1;
    }
    int32_t enqueue(PluginTensorDesc const*,PluginTensorDesc const*,void const* const* in,void* const* out,void*,cudaStream_t stream)noexcept override{
        return state&&state->launch(in[0],out[0],stream)?0:1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*)noexcept override{
        auto* p=new(std::nothrow) ConvPlugin(index);if(p&&!p->init()){delete p;return nullptr;}return p;
    }
    PluginFieldCollection const* getFieldsToSerialize()noexcept override{return &fields;}
};
class ConvCreator final:public IPluginCreatorV3One {
    int index;
public:
    explicit ConvCreator(int i):index(i){}
    IPluginV3* createPlugin(char const*,PluginFieldCollection const*,TensorRTPhase)noexcept override{return new(std::nothrow) ConvPlugin(index);}
    PluginFieldCollection const* getFieldNames()noexcept override{return &fields;}
    char const* getPluginName()const noexcept override{return kConvNames[index];}
    char const* getPluginVersion()const noexcept override{return "1";}
    char const* getPluginNamespace()const noexcept override{return "";}
};
