/*
 * encode_cuda_dl.h — aji_encode's CUDA, loaded when it is first used (Linux).
 *
 * The Linux counterpart of the Windows build's /DELAYLOAD of nvcuda.dll and cudart: the same
 * aji_encode starts on a machine without the NVIDIA driver or the CUDA runtime (--backend vulkan),
 * and the TensorRT path finds libcuda.so.1 and libcudart.so.<major> as the loader would have (the
 * executable's RUNPATH, $ORIGIN in the release, then the usual search). The CUDA headers give the
 * types and constants only: each call encode.c makes is a macro onto a table filled by dlsym, and a
 * call this table misses fails to link (aji_encode no longer links CUDA). Not loadable: the driver
 * calls return CUDA_ERROR_NOT_INITIALIZED, the runtime calls cudaErrorInitializationError.
 */
#ifndef AJI_ENCODE_CUDA_DL_H
#define AJI_ENCODE_CUDA_DL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stddef.h>

#ifndef AJI_CUDART_SOVER
#error "AJI_CUDART_SOVER (the CUDA runtime's soname major) is set by CMakeLists.txt"
#endif

typedef struct {
    int tried, ok;
    CUresult (*Init)(unsigned int);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
    CUresult (*CtxPushCurrent)(CUcontext);
    CUresult (*CtxPopCurrent)(CUcontext *);
    CUresult (*GetErrorString)(CUresult, const char **);
    cudaError_t (*StreamCreate)(cudaStream_t *);
    cudaError_t (*StreamCreateWithPriority)(cudaStream_t *, unsigned int, int);
    cudaError_t (*StreamDestroy)(cudaStream_t);
    cudaError_t (*StreamSynchronize)(cudaStream_t);
    cudaError_t (*StreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int);
    cudaError_t (*EventCreateWithFlags)(cudaEvent_t *, unsigned int);
    cudaError_t (*EventDestroy)(cudaEvent_t);
    cudaError_t (*EventRecord)(cudaEvent_t, cudaStream_t);
    cudaError_t (*DeviceGetStreamPriorityRange)(int *, int *);
    cudaError_t (*Memcpy2DAsync)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
    const char *(*GetErrorStringRt)(cudaError_t);
} aji_cuda_dl;

static aji_cuda_dl g_cudl;

static int aji_cuda_dl_load(void)
{
    if (g_cudl.tried)
        return g_cudl.ok;
    g_cudl.tried = 1;
    void *drv = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    void *rt = dlopen("libcudart.so." AJI_CUDART_SOVER, RTLD_NOW | RTLD_GLOBAL);
    if (!drv || !rt)
        return 0;
#define AJI_CUDL(h, field, sym) \
    if (!(*(void **)&g_cudl.field = dlsym(h, sym))) return 0;
    AJI_CUDL(drv, Init, "cuInit")
    AJI_CUDL(drv, DeviceGetCount, "cuDeviceGetCount")
    AJI_CUDL(drv, DeviceGet, "cuDeviceGet")
    AJI_CUDL(drv, DeviceGetName, "cuDeviceGetName")
    AJI_CUDL(drv, DeviceGetAttribute, "cuDeviceGetAttribute")
    AJI_CUDL(drv, CtxPushCurrent, "cuCtxPushCurrent_v2")
    AJI_CUDL(drv, CtxPopCurrent, "cuCtxPopCurrent_v2")
    AJI_CUDL(drv, GetErrorString, "cuGetErrorString")
    AJI_CUDL(rt, StreamCreate, "cudaStreamCreate")
    AJI_CUDL(rt, StreamCreateWithPriority, "cudaStreamCreateWithPriority")
    AJI_CUDL(rt, StreamDestroy, "cudaStreamDestroy")
    AJI_CUDL(rt, StreamSynchronize, "cudaStreamSynchronize")
    AJI_CUDL(rt, StreamWaitEvent, "cudaStreamWaitEvent")
    AJI_CUDL(rt, EventCreateWithFlags, "cudaEventCreateWithFlags")
    AJI_CUDL(rt, EventDestroy, "cudaEventDestroy")
    AJI_CUDL(rt, EventRecord, "cudaEventRecord")
    AJI_CUDL(rt, DeviceGetStreamPriorityRange, "cudaDeviceGetStreamPriorityRange")
    AJI_CUDL(rt, Memcpy2DAsync, "cudaMemcpy2DAsync")
    AJI_CUDL(rt, GetErrorStringRt, "cudaGetErrorString")
#undef AJI_CUDL
    g_cudl.ok = 1;
    return 1;
}

static inline CUresult aji_cu_get_error_string(CUresult e, const char **s)
{
    if (aji_cuda_dl_load())
        return g_cudl.GetErrorString(e, s);
    *s = "the NVIDIA driver (libcuda.so.1) or the CUDA runtime is not available";
    return CUDA_SUCCESS;
}

static inline const char *aji_cuda_get_error_string(cudaError_t e)
{
    return aji_cuda_dl_load() ? g_cudl.GetErrorStringRt(e)
                              : "the NVIDIA driver (libcuda.so.1) or the CUDA runtime is not available";
}

/* cuda.h maps some names onto their _v2 entry points: replace those too */
#undef cuInit
#undef cuDeviceGetCount
#undef cuDeviceGet
#undef cuDeviceGetName
#undef cuDeviceGetAttribute
#undef cuCtxPushCurrent
#undef cuCtxPopCurrent
#undef cuGetErrorString
#undef cudaStreamCreate
#undef cudaStreamCreateWithPriority
#undef cudaStreamDestroy
#undef cudaStreamSynchronize
#undef cudaStreamWaitEvent
#undef cudaEventCreateWithFlags
#undef cudaEventDestroy
#undef cudaEventRecord
#undef cudaDeviceGetStreamPriorityRange
#undef cudaMemcpy2DAsync
#undef cudaGetErrorString
#define AJI_CU(f, ...) (aji_cuda_dl_load() ? g_cudl.f(__VA_ARGS__) : CUDA_ERROR_NOT_INITIALIZED)
#define AJI_RT(f, ...) (aji_cuda_dl_load() ? g_cudl.f(__VA_ARGS__) : cudaErrorInitializationError)
#define cuInit(...)                AJI_CU(Init, __VA_ARGS__)
#define cuDeviceGetCount(...)      AJI_CU(DeviceGetCount, __VA_ARGS__)
#define cuDeviceGet(...)           AJI_CU(DeviceGet, __VA_ARGS__)
#define cuDeviceGetName(...)       AJI_CU(DeviceGetName, __VA_ARGS__)
#define cuDeviceGetAttribute(...)  AJI_CU(DeviceGetAttribute, __VA_ARGS__)
#define cuCtxPushCurrent(...)      AJI_CU(CtxPushCurrent, __VA_ARGS__)
#define cuCtxPopCurrent(...)       AJI_CU(CtxPopCurrent, __VA_ARGS__)
#define cuGetErrorString(e, s)     aji_cu_get_error_string((e), (s))
#define cudaStreamCreate(...)      AJI_RT(StreamCreate, __VA_ARGS__)
#define cudaStreamCreateWithPriority(...) AJI_RT(StreamCreateWithPriority, __VA_ARGS__)
#define cudaStreamDestroy(...)     AJI_RT(StreamDestroy, __VA_ARGS__)
#define cudaStreamSynchronize(...) AJI_RT(StreamSynchronize, __VA_ARGS__)
#define cudaStreamWaitEvent(...)   AJI_RT(StreamWaitEvent, __VA_ARGS__)
#define cudaEventCreateWithFlags(...) AJI_RT(EventCreateWithFlags, __VA_ARGS__)
#define cudaEventDestroy(...)      AJI_RT(EventDestroy, __VA_ARGS__)
#define cudaEventRecord(...)       AJI_RT(EventRecord, __VA_ARGS__)
#define cudaDeviceGetStreamPriorityRange(...) AJI_RT(DeviceGetStreamPriorityRange, __VA_ARGS__)
#define cudaMemcpy2DAsync(...)     AJI_RT(Memcpy2DAsync, __VA_ARGS__)
#define cudaGetErrorString(e)      aji_cuda_get_error_string(e)

#endif
