// Opt-in Windows/TensorRT 11 experiment; no plugin dependency for stock RIFE.
#pragma once
#include <filesystem>
#include <mutex>
#include <string>

static constexpr const char *AJI_RIFE_WARP_MODEL = "rife_v4.22_fp16_warp_v1";

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

static bool aji_register_rife_warp_plugin()
{
    static std::mutex mutex;
    static bool ready = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (ready)
        return true;
    HMODULE mod = LoadLibraryExW(std::filesystem::u8path(aji_rife_warp_plugin_path()).c_str(),
        nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    auto init = mod ? reinterpret_cast<int(*)()>(GetProcAddress(mod, "initRifePlugin")) : nullptr;
    if (!init || !init()) {
        if (mod) FreeLibrary(mod);
        return false;
    }
    ready = true; // Creator stays registered for the lifetime of this module.
    return true;
}
#endif
