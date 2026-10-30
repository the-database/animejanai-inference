/*
 * core/fileio.cpp — see fileio.h.
 */

#include "fileio.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <libgen.h>
#endif

namespace fs = std::filesystem;

namespace ajicore {

#ifdef _WIN32
std::wstring widen_utf8(const std::string &s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0)
        return std::wstring();
    std::wstring w((size_t)n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

std::wstring win_long_path(const std::string &path)
{
    std::wstring w = widen_utf8(path);
    if (w.empty() || w.rfind(L"\\\\?\\", 0) == 0 ||
        w.rfind(L"\\\\.\\", 0) == 0)
        return w;

    DWORD need = GetFullPathNameW(w.c_str(), 0, nullptr, nullptr);
    if (need > 0) {
        std::vector<wchar_t> full(need);
        DWORD n = GetFullPathNameW(w.c_str(), need, full.data(), nullptr);
        if (n > 0 && n < need)
            w.assign(full.data(), n);
    }

    if (w.rfind(L"\\\\", 0) == 0)
        return L"\\\\?\\UNC" + w.substr(1);
    if (w.size() >= 2 && w[1] == L':')
        return L"\\\\?\\" + w;
    return w;
}

bool file_exists(const std::string &path)
{
    DWORD attrs = GetFileAttributesW(win_long_path(path).c_str());
    return attrs != INVALID_FILE_ATTRIBUTES &&
           !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

void remove_file(const std::string &path)
{
    DeleteFileW(win_long_path(path).c_str());
}

long long file_size(const std::string &path)
{
    WIN32_FILE_ATTRIBUTE_DATA a = {};
    if (!GetFileAttributesExW(win_long_path(path).c_str(), GetFileExInfoStandard,
                              &a))
        return -1;
    return ((long long)a.nFileSizeHigh << 32) | a.nFileSizeLow;
}

bool move_file(const std::string &from, const std::string &to)
{
    return MoveFileExW(win_long_path(from).c_str(), win_long_path(to).c_str(),
                       MOVEFILE_REPLACE_EXISTING) != 0;
}

bool read_file_bytes(const std::string &path, std::vector<char> *out)
{
    HANDLE h = CreateFileW(win_long_path(path).c_str(), GENERIC_READ,
                           FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return false;
    LARGE_INTEGER size = {};
    if (!GetFileSizeEx(h, &size) || size.QuadPart < 0 ||
        size.QuadPart > SIZE_MAX) {
        CloseHandle(h);
        return false;
    }
    out->resize((size_t)size.QuadPart);
    char *dst = out->data();
    size_t left = out->size();
    while (left > 0) {
        DWORD chunk = (DWORD)std::min<size_t>(left, 1u << 30);
        DWORD got = 0;
        if (!ReadFile(h, dst, chunk, &got, nullptr) || got == 0) {
            CloseHandle(h);
            return false;
        }
        dst += got;
        left -= got;
    }
    CloseHandle(h);
    return true;
}

void append_file_text(const std::string &path, const std::string &text)
{
    HANDLE h = CreateFileW(win_long_path(path).c_str(), FILE_APPEND_DATA,
                           FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return;
    DWORD written = 0;
    WriteFile(h, text.data(), (DWORD)text.size(), &written, nullptr);
    CloseHandle(h);
}

std::string own_dir()
{
    HMODULE self = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&own_dir, &self);
    char path[MAX_PATH] = {0};
    GetModuleFileNameA(self, path, sizeof(path));
    char *slash = strrchr(path, '\\');
    if (slash)
        *slash = 0;
    return path;
}
#else
bool file_exists(const std::string &path)
{
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

void remove_file(const std::string &path)
{
    std::error_code ec;
    fs::remove(path, ec);
}

long long file_size(const std::string &path)
{
    std::error_code ec;
    auto n = fs::file_size(path, ec);
    return ec ? -1 : (long long)n;
}

bool move_file(const std::string &from, const std::string &to)
{
    std::error_code ec;
    fs::rename(from, to, ec);
    return !ec;
}

bool read_file_bytes(const std::string &path, std::vector<char> *out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out->assign(std::istreambuf_iterator<char>(f),
                std::istreambuf_iterator<char>());
    return true;
}

void append_file_text(const std::string &path, const std::string &text)
{
    if (FILE *f = fopen(path.c_str(), "a")) {
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
    }
}

std::string own_dir()
{
    Dl_info info = {};
    if (!dladdr((void *)&own_dir, &info) || !info.dli_fname)
        return ".";
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s", info.dli_fname);
    return dirname(buf);
}
#endif

} // namespace ajicore
