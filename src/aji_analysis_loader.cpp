// aji_analysis.dll: the optional addon observer's entry point. It imports no
// CUDA or TensorRT library itself. On the first observer it loads what the
// observer imports (cudart, nvinfer, the ONNX parser) and then the observer,
// aji_analysis_trt.dll, beside it, and forwards the aji_analysis.h ABI to it.
//
// With aji-runtime.json the TensorRT runtime lives in the shared runtime packs,
// not in this folder, so those libraries are loaded first by full path and the
// observer's imports bind to them: the same order aji.dll uses for aji_trt.dll.
// Without the file the observer's imports resolve from this folder, as before.
//
// The observer is unloaded again when its last instance is destroyed, so an
// unsubscribed player keeps no analysis kernels or engine loaded.
#include "aji_analysis.h"
#include "core/runtime_packs.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {

using create_fn = aji_observer *(*)(void *, int, int, const char *, aji_observer_cancel_fn,
                                    aji_observer_state_fn, void *, char *, size_t);
using run_fn = int (*)(aji_observer *, const aji_frame *, const int[4], void *, unsigned char *,
                       uint16_t *, double *);
using destroy_fn = void (*)(aji_observer *);

std::mutex mu;
HMODULE observer_lib = nullptr;
create_fn create_impl = nullptr;
std::atomic<run_fn> run_impl{nullptr};  // read without the lock, per sample
destroy_fn destroy_impl = nullptr;
unsigned live = 0;  // observers created and not yet destroyed

// This library's folder (wide: it may not be representable in the ANSI code page).
std::wstring own_dir_w()
{
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&own_dir_w), &self))
        return L"";
    std::wstring path(32768, L'\0');
    DWORD n = GetModuleFileNameW(self, &path[0], (DWORD)path.size());
    if (!n || n >= path.size())
        return L"";
    path.resize(n);
    const size_t slash = path.find_last_of(L'\\');
    return slash == std::wstring::npos ? L"" : path.substr(0, slash);
}

void report(char *error, size_t error_size, const std::string &message)
{
    if (error && error_size)
        std::snprintf(error, error_size, "%s", message.c_str());
}

void unload_locked()
{
    if (observer_lib)
        FreeLibrary(observer_lib);
    observer_lib = nullptr;
    create_impl = nullptr;
    run_impl = nullptr;
    destroy_impl = nullptr;
}

// Loads the observer if it isn't loaded. Called with `mu` held.
bool load_locked(std::string *err)
{
    if (observer_lib)
        return true;
    if (!ajicore::preload_backend_deps("aji_analysis_trt", err, nullptr))
        return false;
    const std::wstring dir = own_dir_w();
    if (dir.empty()) {
        *err = "Could not locate the analysis runtime directory.";
        return false;
    }
    HMODULE lib = LoadLibraryExW((dir + L"\\aji_analysis_trt.dll").c_str(), nullptr,
                                 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!lib) {
        *err = "aji_analysis_trt.dll failed to load (Windows error " +
               std::to_string(GetLastError()) + ").";
        return false;
    }
    create_impl = reinterpret_cast<create_fn>(GetProcAddress(lib, "aji_observer_create"));
    auto run = reinterpret_cast<run_fn>(GetProcAddress(lib, "aji_observer_run"));
    destroy_impl = reinterpret_cast<destroy_fn>(GetProcAddress(lib, "aji_observer_destroy"));
    observer_lib = lib;
    if (!create_impl || !run || !destroy_impl) {
        unload_locked();
        *err = "aji_analysis_trt.dll does not export the analysis observer ABI.";
        return false;
    }
    run_impl = run;
    return true;
}

}  // namespace

extern "C" AJI_EXPORT aji_observer *aji_observer_create(void *cuda_context, int width, int height,
    const char *model, aji_observer_cancel_fn cancel, aji_observer_state_fn state,
    void *opaque, char *error, size_t error_size)
{
    std::lock_guard<std::mutex> lock(mu);
    std::string err;
    if (!load_locked(&err)) {
        report(error, error_size, err);
        return nullptr;
    }
    // The lock also keeps a concurrent last destroy from unloading the
    // observer while this create runs. Creates are rare and run only on the
    // optional sampling worker.
    aji_observer *observer = create_impl(cuda_context, width, height, model, cancel, state,
                                         opaque, error, error_size);
    if (observer)
        live++;
    else if (!live)
        unload_locked();
    return observer;
}

extern "C" AJI_EXPORT int aji_observer_run(aji_observer *observer, const aji_frame *frame,
    const int crop[4], void *decode_stream, unsigned char *bgra, uint16_t *mask, double *gpu_ms)
{
    // A live observer keeps the library loaded; no lock on the per-sample path.
    run_fn run = run_impl.load();
    if (!observer || !run)
        return AJI_ERR;
    return run(observer, frame, crop, decode_stream, bgra, mask, gpu_ms);
}

extern "C" AJI_EXPORT void aji_observer_destroy(aji_observer *observer)
{
    if (!observer)
        return;
    std::lock_guard<std::mutex> lock(mu);
    if (!destroy_impl)
        return;
    destroy_impl(observer);
    if (live && !--live)
        unload_locked();
}
