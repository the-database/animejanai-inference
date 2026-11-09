#pragma once

#include <cuda_runtime_api.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cwchar>
#endif

// CUDA's Windows hybrid loader can fail with cudaErrorInsufficientDriver
// when two DLLs first enter it concurrently. Initialize it outside DllMain
// under a process-wide lock shared by the upscaler and optional observer.
// The lock covers runtime loading only; inference and analysis stay concurrent.
inline cudaError_t aji_initialize_cuda_runtime()
{
#ifdef _WIN32
    wchar_t name[80];
    std::swprintf(name, 80, L"Local\\AnimeJaNai.CudaRuntimeInit.%lu", GetCurrentProcessId());
    HANDLE mutex = CreateMutexW(nullptr, FALSE, name);
    if (!mutex) return cudaErrorOperatingSystem;
    DWORD wait = WaitForSingleObject(mutex, INFINITE);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) {
        CloseHandle(mutex);
        return cudaErrorOperatingSystem;
    }
#endif
    int version;
    cudaError_t result = cudaRuntimeGetVersion(&version);
#ifdef _WIN32
    ReleaseMutex(mutex);
    CloseHandle(mutex);
#endif
    return result;
}
