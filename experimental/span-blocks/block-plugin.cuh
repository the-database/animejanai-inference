#include "block-parameters.h"
class BlockPlugin final:public IPluginV3,public IPluginV3OneCore,public IPluginV3OneBuild,public IPluginV3OneRuntime {
    int block;ConvPlugin* conv[3]={};
    int count()const{return kBlockExpose[block]?2:1;}
public:
    explicit BlockPlugin(int b):block(b){}
    ~BlockPlugin()override{for(auto* c:conv)delete c;}
    bool init(){for(int i=0;i<3;++i){conv[i]=new(std::nothrow) ConvPlugin(block*3+i);if(!conv[i]||!conv[i]->init())return false;}return true;}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t)noexcept override{
        switch(t){case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);default:return nullptr;}
    }
    IPluginV3* clone()noexcept override{return new(std::nothrow) BlockPlugin(block);}
    char const* getPluginName()const noexcept override{return kBlockNames[block];}
    char const* getPluginVersion()const noexcept override{return "1";}
    char const* getPluginNamespace()const noexcept override{return kBlockNamespace;}
    int32_t getNbOutputs()const noexcept override{return count();}
    int32_t getOutputDataTypes(DataType* out,int32_t no,DataType const* in,int32_t ni)const noexcept override{
        if(ni!=1||no!=count()||in[0]!=DataType::kHALF)return 1;for(int i=0;i<no;++i)out[i]=DataType::kHALF;return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in,int32_t ni,DimsExprs const*,int32_t ns,DimsExprs* out,int32_t no,IExprBuilder&)noexcept override{
        if(ni!=1||no!=count()||ns)return 1;for(int i=0;i<no;++i)out[i]=in[0];return 0;
    }
    bool supportsFormatCombination(int32_t pos,DynamicPluginTensorDesc const* io,int32_t ni,int32_t no)noexcept override{
        return ni==1&&no==count()&&pos>=0&&pos<ni+no&&io[pos].desc.type==DataType::kHALF&&io[pos].desc.format==TensorFormat::kHWC;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in,int32_t ni,DynamicPluginTensorDesc const*,int32_t no)noexcept override{
        return ni==1&&no==count()&&convShape(in[0].min)&&convShape(in[0].max)?0:1;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in,int32_t,DynamicPluginTensorDesc const*,int32_t)const noexcept override{
        return size_t(in[0].max.d[2])*size_t(in[0].max.d[3])*64*2;
    }
    int32_t onShapeChange(PluginTensorDesc const* in,int32_t ni,PluginTensorDesc const* out,int32_t no)noexcept override{
        if(ni!=1||no!=count()||!convShape(in[0].dims))return 1;
        for(auto* c:conv)if(!c||c->onShapeChange(in,1,out,1))return 1;return 0;
    }
    int32_t enqueue(PluginTensorDesc const* desc,PluginTensorDesc const* od,void const* const* in,void* const* out,void* workspace,cudaStream_t stream)noexcept override{
        void* first=out[0];
        void* second=workspace;
        void* last=out[count()-1];void const* a[]={in[0]};void* b[]={first};
        if(conv[0]->enqueue(desc,od,a,b,nullptr,stream))return 1;
        a[0]=first;b[0]=second;if(conv[1]->enqueue(desc,od,a,b,nullptr,stream))return 1;
        a[0]=second;b[0]=last;return conv[2]->enqueue(desc,od,a,b,nullptr,stream);
    }
    IPluginV3* attachToContext(IPluginResourceContext*)noexcept override{
        auto* p=new(std::nothrow) BlockPlugin(block);if(p&&!p->init()){delete p;return nullptr;}return p;
    }
    PluginFieldCollection const* getFieldsToSerialize()noexcept override{return &fields;}
};
class BlockCreator final:public IPluginCreatorV3One {
    int block;
public:
    explicit BlockCreator(int b):block(b){}
    IPluginV3* createPlugin(char const*,PluginFieldCollection const*,TensorRTPhase)noexcept override{return new(std::nothrow) BlockPlugin(block);}
    PluginFieldCollection const* getFieldNames()noexcept override{return &fields;}
    char const* getPluginName()const noexcept override{return kBlockNames[block];}
    char const* getPluginVersion()const noexcept override{return "1";}
    char const* getPluginNamespace()const noexcept override{return kBlockNamespace;}
};
