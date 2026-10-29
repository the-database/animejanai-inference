// Optional addon observer. All setup and GPU waits run on its own worker;
// the normal upscaler and RIFE paths never construct or invoke this object.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "aji_analysis.h"
#include "analysis_kernels.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {
struct Logger : nvinfer1::ILogger {
    std::string error;
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kERROR) error = message;
    }
};
struct Current {
    bool valid;
    explicit Current(CUcontext context) : valid(cuCtxPushCurrent(context) == CUDA_SUCCESS) {
        if (!valid) throw std::runtime_error("Could not enter the source CUDA context.");
    }
    ~Current() { if (valid) { CUcontext previous; cuCtxPopCurrent(&previous); } }
};
void check(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
struct Progress : nvinfer1::IProgressMonitor {
    aji_observer_cancel_fn cancel; void *opaque;
    Progress(aji_observer_cancel_fn cancel, void *opaque) : cancel(cancel), opaque(opaque) {}
    void phaseStart(const char *, const char *, int32_t) noexcept override {}
    bool stepComplete(const char *, int32_t) noexcept override { return !cancel || !cancel(opaque); }
    void phaseFinish(const char *) noexcept override {}
};
std::vector<char> read(const std::filesystem::path &path) {
    auto size = std::filesystem::file_size(path);
    require(size > 0 && size < 2ULL*1024*1024*1024, "Invalid analysis engine/model file size.");
    std::vector<char> bytes((size_t)size); std::ifstream stream(path, std::ios::binary);
    require(bool(stream.read(bytes.data(), bytes.size())), "Could not read analysis engine/model."); return bytes;
}
bool shape(nvinfer1::Dims d, int c) { return d.nbDims == 4 && d.d[0] == 1 && d.d[1] == c && d.d[2] == 1024 && d.d[3] == 1024; }
}

struct aji_observer {
    CUcontext context;
    int width, height;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> execution;
    cudaStream_t stream = nullptr;
    cudaEvent_t decoded = nullptr, start = nullptr, finish = nullptr;
    float *input = nullptr, *mask = nullptr;
    unsigned char *color = nullptr;
    uint16_t *small_mask = nullptr;
    void *download = nullptr;
    aji_observer(CUcontext context, int w, int h) : context(context), width(w), height(h) {}
    ~aji_observer() {
        CUcontext previous;
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        if (stream) cudaStreamSynchronize(stream);
        execution.reset(); engine.reset(); runtime.reset();
        cudaFree(input); cudaFree(mask); cudaFree(color); cudaFree(small_mask);
        if (download) cudaFreeHost(download);
        if (decoded) cudaEventDestroy(decoded); if (start) cudaEventDestroy(start); if (finish) cudaEventDestroy(finish);
        if (stream) cudaStreamDestroy(stream);
        cuCtxPopCurrent(&previous);
    }
};

extern "C" AJI_EXPORT aji_observer *aji_observer_create(void *context, int width, int height,
    const char *model, aji_observer_cancel_fn cancelled, aji_observer_state_fn state,
    void *opaque, char *error, size_t error_size)
{
    try {
        require(width > 0 && width <= 320 && height > 0 && height <= 180, "Invalid analysis sample dimensions.");
        Current current((CUcontext)context);
        auto p = std::make_unique<aji_observer>((CUcontext)context, width, height);
        check(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&p->decoded, cudaEventDisableTiming));
        check(cudaEventCreate(&p->start)); check(cudaEventCreate(&p->finish));
        check(cudaMalloc(&p->color, width*height*4));
        check(cudaMallocHost(&p->download, width*height*(model ? 6 : 4)));
        if (model) {
            std::filesystem::path path = std::filesystem::u8path(model);
            require(std::filesystem::exists(path), "Approved isnetis.onnx is missing; provision it through the addon host.");
            int device, driver; check(cudaGetDevice(&device)); check(cudaDriverGetVersion(&driver));
            cudaDeviceProp gpu{}; check(cudaGetDeviceProperties(&gpu,device));
            std::string key = ".trt-"+std::to_string(getInferLibVersion())+".driver-"+std::to_string(driver)+".gpu-";
            for (unsigned char b : gpu.uuid.bytes) { char hex[3]; std::snprintf(hex,sizeof hex,"%02x",b); key += hex; }
            auto cached = path; cached += key + ".engine";
            p->runtime.reset(nvinfer1::createInferRuntime(p->logger));
            require(bool(p->runtime), "Could not create the TensorRT analysis runtime.");
            // OS lock prevents two observed players from building/writing the
            // same engine concurrently. Waiting happens only on this worker.
#ifdef _WIN32
            auto lock_path = cached; lock_path += ".lock";
            HANDLE lock = INVALID_HANDLE_VALUE;
            while (lock == INVALID_HANDLE_VALUE) {
                require(!cancelled || !cancelled(opaque), "Analysis setup cancelled.");
                lock = CreateFileW(lock_path.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_TEMPORARY|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
                if (lock == INVALID_HANDLE_VALUE) {
                    require(GetLastError() == ERROR_SHARING_VIOLATION, "Cannot lock the analysis engine cache directory.");
                    if(state)state(opaque,9); Sleep(100);
                }
            }
            struct Unlock { HANDLE handle; ~Unlock(){CloseHandle(handle);} } unlock{lock};
#endif
            if(std::filesystem::exists(cached)) {
                auto bytes=read(cached); p->engine.reset(p->runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
            }
            if(!p->engine) {
                if(state)state(opaque,9);
                auto bytes=read(path);
                std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(p->logger));
                require(bool(builder),"TensorRT engine builder is unavailable.");
                std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
                std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network,p->logger));
                require(bool(parser) && parser->parse(bytes.data(),bytes.size()), "Could not parse the approved character-mask model.");
                require(network->getNbInputs()==1 && network->getNbOutputs()==1 &&
                    std::string(network->getInput(0)->getName())=="img" && shape(network->getInput(0)->getDimensions(),3) &&
                    std::string(network->getOutput(0)->getName())=="mask" && shape(network->getOutput(0)->getDimensions(),1),
                    "Character-mask model must expose img [1,3,1024,1024] and mask [1,1,1024,1024].");
                std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
                config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL*1024*1024);
                config->setBuilderOptimizationLevel(3);
#if NV_TENSORRT_MAJOR < 11
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
                Progress progress(cancelled,opaque); config->setProgressMonitor(&progress);
                std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));
                require(bool(plan),p->logger.error.empty()?"Character-mask engine build failed.":p->logger.error.c_str());
                require(!cancelled||!cancelled(opaque),"Analysis setup cancelled.");
                p->engine.reset(p->runtime->deserializeCudaEngine(plan->data(),plan->size()));
                require(bool(p->engine),"New character-mask engine did not load.");
                auto temporary=cached;temporary+=".partial";
                { std::ofstream out(temporary,std::ios::binary|std::ios::trunc); require(bool(out.write((const char*)plan->data(),plan->size())),"Could not save the analysis engine."); }
                if(std::filesystem::exists(cached))std::filesystem::remove(cached);
                std::filesystem::rename(temporary,cached);
            }
            require(p->engine->getNbIOTensors()==2 && shape(p->engine->getTensorShape("img"),3) && shape(p->engine->getTensorShape("mask"),1) &&
                p->engine->getTensorDataType("img")==nvinfer1::DataType::kFLOAT && p->engine->getTensorDataType("mask")==nvinfer1::DataType::kFLOAT,
                "Character-mask engine must have static float RGB input and one-channel float output.");
            p->execution.reset(p->engine->createExecutionContext()); require(bool(p->execution),"Could not create the character-mask execution context.");
            check(cudaMalloc(&p->input,3*1024*1024*sizeof(float)));check(cudaMalloc(&p->mask,1024*1024*sizeof(float)));
            check(cudaMalloc(&p->small_mask,width*height*2));
            require(p->execution->setTensorAddress("img",p->input)&&p->execution->setTensorAddress("mask",p->mask),"Could not bind character-mask tensors.");
        }
        return p.release();
    } catch(const std::exception &e) { if(error && error_size)std::snprintf(error,error_size,"%s",e.what());return nullptr; }
}

extern "C" AJI_EXPORT int aji_observer_run(aji_observer *p,const aji_frame *frame,const int crop[4],void *decode_stream,
    unsigned char *bgra,uint16_t *mask,double *gpu_ms)
{
    try {
        Current current(p->context);
        require(crop[0]>=0&&crop[1]>=0&&crop[2]>0&&crop[3]>0&&crop[0]<=frame->width-crop[2]&&crop[1]<=frame->height-crop[3],"Invalid source crop.");
        require(frame->format==AJI_FMT_NV12||frame->format==AJI_FMT_P010||aji_format_is_444(frame->format),"Unsupported CUDA source format.");
        check(cudaEventRecord(p->decoded,(cudaStream_t)decode_stream));check(cudaStreamWaitEvent(p->stream,p->decoded,0));
        check(cudaEventRecord(p->start,p->stream));
        int iw=1024,ih=1024;
        if(crop[2]>=crop[3])ih=std::max(1,(int)std::lround(1024.0*crop[3]/crop[2]));
        else iw=std::max(1,(int)std::lround(1024.0*crop[2]/crop[3]));
        check((cudaError_t)aji_analysis_pre(frame,crop,p->width,p->height,iw,ih,p->input,p->color,p->stream));
        if(p->execution) {
            require(p->execution->enqueueV3(p->stream),"Character-mask inference failed.");
            check((cudaError_t)aji_analysis_mask(p->mask,p->width,p->height,iw,ih,p->small_mask,p->stream));
        }
        check(cudaMemcpyAsync(p->download,p->color,p->width*p->height*4,cudaMemcpyDeviceToHost,p->stream));
        if(p->execution)check(cudaMemcpyAsync((char*)p->download+p->width*p->height*4,p->small_mask,p->width*p->height*2,cudaMemcpyDeviceToHost,p->stream));
        check(cudaEventRecord(p->finish,p->stream));check(cudaEventSynchronize(p->finish));
        float elapsed;check(cudaEventElapsedTime(&elapsed,p->start,p->finish));*gpu_ms=elapsed;
        std::memcpy(bgra,p->download,p->width*p->height*4);
        if(p->execution)std::memcpy(mask,(char*)p->download+p->width*p->height*4,p->width*p->height*2);
        return AJI_OK;
    }catch(...){ // Retire queued reads before the caller releases the source.
        CUcontext previous;if(cuCtxPushCurrent(p->context)==CUDA_SUCCESS){cudaStreamSynchronize(p->stream);cuCtxPopCurrent(&previous);}return AJI_ERR_CUDA;
    }
}
extern "C" AJI_EXPORT void aji_observer_destroy(aji_observer *p){delete p;}
