// Opt-in Windows/TensorRT 11 experiment; no plugin dependency for stock RIFE.
#pragma once
#include <filesystem>
#include <mutex>
#include <string>
#include "core/plugin_cache.h"

static constexpr const char *AJI_RIFE_WARP_MODEL = "rife_v4.22_fp16_warp_v2";

#if defined(_WIN32) && !defined(AJI_TRTRTX) && NV_TENSORRT_MAJOR == 11
static std::string aji_rife_warp_plugin_path()
{
    HMODULE self = nullptr;
    wchar_t filename[32768];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&aji_rife_warp_plugin_path), &self))
        return {};
    DWORD n = GetModuleFileNameW(self, filename, 32768);
    if (!n || n >= 32768)
        return {};
    return (std::filesystem::path(filename).parent_path() / "rife-warp-plugin.dll").u8string();
}

static bool aji_register_rife_warp_plugin(std::string *identity = nullptr)
{
    static std::mutex mutex;
    static bool ready = false;
    static HMODULE mod = nullptr;
    static std::string loaded_id;
    std::lock_guard<std::mutex> lock(mutex);
    const auto path = aji_rife_warp_plugin_path();
    const auto id = ajicore::plugin_binary_id(path);
    if (id.empty()) return false;
    if (mod) {
        if (!ready || id != loaded_id) return false;
        if (identity) *identity = loaded_id;
        return true;
    }
    mod = LoadLibraryExW(std::filesystem::u8path(path).c_str(),
        nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    loaded_id = id;
    auto init = mod ? reinterpret_cast<int(*)()>(GetProcAddress(mod, "initRifePlugin")) : nullptr;
    // The registry may retain creator pointers even after failed registration.
    if (!init || !init())
        return false;
    ready = true; // Creator stays registered for the lifetime of this module.
    if (identity) *identity = loaded_id;
    return true;
}
#endif
