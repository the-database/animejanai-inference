/*
 * core/runtime_packs.cpp — see runtime_packs.h.
 */

#include "runtime_packs.h"
#include "fileio.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <libgen.h>
#endif

// Library names the backends import, from CMake (they follow the SDKs the
// build used). Empty = not built / nothing to preload.
#ifndef AJI_PACK_CUDART
#  define AJI_PACK_CUDART ""
#endif
#ifndef AJI_PACK_NVINFER
#  define AJI_PACK_NVINFER ""
#endif
#ifndef AJI_PACK_NVONNXPARSER
#  define AJI_PACK_NVONNXPARSER ""
#endif
#ifndef AJI_PACK_TRTRTX
#  define AJI_PACK_TRTRTX ""
#endif
#ifndef AJI_PACK_TRTRTX_PARSER
#  define AJI_PACK_TRTRTX_PARSER ""
#endif

namespace ajicore {

namespace {

#ifdef _WIN32
const char SEP = '\\';
#else
const char SEP = '/';
#endif

// --- a minimal JSON reader: just enough for aji-runtime.json ---------------

struct Json {
    const char *p, *end;
    std::string err;

    void ws()
    {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            p++;
    }
    bool fail(const char *what)
    {
        if (err.empty())
            err = what;
        return false;
    }
    bool lit(const char *s)
    {
        size_t n = strlen(s);
        if ((size_t)(end - p) < n || memcmp(p, s, n) != 0)
            return fail("invalid JSON");
        p += n;
        return true;
    }
    static void put_utf8(std::string *o, uint32_t c)
    {
        if (c < 0x80) {
            o->push_back((char)c);
        } else if (c < 0x800) {
            o->push_back((char)(0xC0 | (c >> 6)));
            o->push_back((char)(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            o->push_back((char)(0xE0 | (c >> 12)));
            o->push_back((char)(0x80 | ((c >> 6) & 0x3F)));
            o->push_back((char)(0x80 | (c & 0x3F)));
        } else {
            o->push_back((char)(0xF0 | (c >> 18)));
            o->push_back((char)(0x80 | ((c >> 12) & 0x3F)));
            o->push_back((char)(0x80 | ((c >> 6) & 0x3F)));
            o->push_back((char)(0x80 | (c & 0x3F)));
        }
    }
    bool hex4(uint32_t *v)
    {
        if (end - p < 4)
            return fail("invalid JSON string escape");
        *v = 0;
        for (int i = 0; i < 4; i++) {
            char ch = *p++;
            *v <<= 4;
            if (ch >= '0' && ch <= '9') *v |= (uint32_t)(ch - '0');
            else if (ch >= 'a' && ch <= 'f') *v |= (uint32_t)(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') *v |= (uint32_t)(ch - 'A' + 10);
            else return fail("invalid JSON string escape");
        }
        return true;
    }
    bool str(std::string *o)
    {
        ws();
        if (p >= end || *p != '"')
            return fail("expected a JSON string");
        p++;
        while (p < end && *p != '"') {
            char ch = *p++;
            if ((unsigned char)ch < 0x20)
                return fail("control character in JSON string");
            if (ch != '\\') {
                o->push_back(ch);
                continue;
            }
            if (p >= end)
                break;
            char e = *p++;
            switch (e) {
            case '"': o->push_back('"'); break;
            case '\\': o->push_back('\\'); break;
            case '/': o->push_back('/'); break;
            case 'b': o->push_back('\b'); break;
            case 'f': o->push_back('\f'); break;
            case 'n': o->push_back('\n'); break;
            case 'r': o->push_back('\r'); break;
            case 't': o->push_back('\t'); break;
            case 'u': {
                uint32_t c;
                if (!hex4(&c))
                    return false;
                if (c >= 0xD800 && c < 0xDC00) {  // surrogate pair
                    uint32_t lo;
                    if (end - p < 2 || p[0] != '\\' || p[1] != 'u')
                        return fail("invalid JSON surrogate pair");
                    p += 2;
                    if (!hex4(&lo) || lo < 0xDC00 || lo > 0xDFFF)
                        return fail("invalid JSON surrogate pair");
                    c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
                }
                put_utf8(o, c);
                break;
            }
            default:
                return fail("invalid JSON string escape");
            }
        }
        if (p >= end)
            return fail("unterminated JSON string");
        p++;
        return true;
    }
    bool num(double *v)
    {
        ws();
        const char *s = p;
        if (p < end && *p == '-')
            p++;
        while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' ||
                           *p == 'E' || *p == '+' || *p == '-'))
            p++;
        if (p == s)
            return fail("invalid JSON");
        *v = strtod(std::string(s, p).c_str(), nullptr);
        return true;
    }
    // Skips any value (for keys this version doesn't know).
    bool skip(int depth = 0)
    {
        if (depth > 32)
            return fail("JSON nested too deeply");
        ws();
        if (p >= end)
            return fail("invalid JSON");
        if (*p == '"') {
            std::string s;
            return str(&s);
        }
        if (*p == '{' || *p == '[') {
            const char close = *p == '{' ? '}' : ']';
            const bool obj = *p == '{';
            p++;
            ws();
            if (p < end && *p == close) {
                p++;
                return true;
            }
            for (;;) {
                if (obj) {
                    std::string k;
                    if (!str(&k))
                        return false;
                    ws();
                    if (p >= end || *p++ != ':')
                        return fail("invalid JSON");
                }
                if (!skip(depth + 1))
                    return false;
                ws();
                if (p < end && *p == ',') {
                    p++;
                    continue;
                }
                if (p < end && *p == close) {
                    p++;
                    return true;
                }
                return fail("invalid JSON");
            }
        }
        if (*p == 't')
            return lit("true");
        if (*p == 'f')
            return lit("false");
        if (*p == 'n')
            return lit("null");
        double d;
        return num(&d);
    }
};

bool is_absolute(const std::string &p)
{
#ifdef _WIN32
    return (p.size() >= 2 && p[1] == ':') ||
           (p.size() >= 2 && (p[0] == '\\' || p[0] == '/') &&
            (p[1] == '\\' || p[1] == '/'));
#else
    return !p.empty() && p[0] == '/';
#endif
}

std::string resolve(const std::string &dir, std::string p)
{
#ifdef _WIN32
    for (char &ch : p)
        if (ch == '/')
            ch = '\\';
#endif
    while (p.size() > 1 && (p.back() == '\\' || p.back() == '/'))
        p.pop_back();
    if (p.empty() || is_absolute(p))
        return p;
    return dir + SEP + p;
}

// Loads `file` from pack folder `dir` by full path, once; global on POSIX
// so later loads by soname (the backend's DT_NEEDED) bind to it.
bool load_from(const std::string &dir, const char *file, const char *pack,
               std::string *err, std::string *what)
{
    if (!*file)
        return true;
    const std::string path = dir + SEP + file;
    if (!file_exists(path)) {
        *err = "aji-runtime.json: " + std::string(pack) + " pack " + dir +
               " has no " + file;
        return false;
    }
#ifdef _WIN32
    // DLL_LOAD_DIR: the library's own dependencies resolve from the pack
    HMODULE h = LoadLibraryExW(widen_utf8(path).c_str(), NULL,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h) {
        *err = "aji-runtime.json: " + path + " failed to load (error " +
               std::to_string(GetLastError()) + ")";
        return false;
    }
#else
    void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        const char *e = dlerror();
        *err = "aji-runtime.json: " + std::string(e ? e : path.c_str());
        return false;
    }
#endif
    // never unloaded: the backends stay loaded for the process too
    if (what) {
        if (!what->empty())
            *what += ", ";
        *what += path;
    }
    return true;
}

std::string sh_quote(const std::string &s)
{
    std::string o = "'";
    for (char ch : s) {
        if (ch == '\'')
            o += "'\\''";
        else
            o.push_back(ch);
    }
    return o + "'";
}

} // namespace

RuntimePacks parse_runtime_packs(const std::string &text, const std::string &dir)
{
    RuntimePacks r;
    r.present = true;
    Json j{text.data(), text.data() + text.size(), {}};
    // a UTF-8 BOM (Windows editors) is tolerated
    if (j.end - j.p >= 3 && memcmp(j.p, "\xEF\xBB\xBF", 3) == 0)
        j.p += 3;
    double format = -1;
    RuntimePacks v;
    bool ok = true;
    j.ws();
    if (j.p >= j.end || *j.p != '{')
        ok = j.fail("not a JSON object");
    else
        j.p++;
    j.ws();
    if (ok && j.p < j.end && *j.p == '}') {
        j.p++;
    } else {
        while (ok) {
            std::string key;
            if (!(ok = j.str(&key)))
                break;
            j.ws();
            if (j.p >= j.end || *j.p++ != ':') {
                ok = j.fail("invalid JSON");
                break;
            }
            if (key == "format") {
                ok = j.num(&format);
            } else if (key == "trt" || key == "trtrtx") {
                std::string s;
                ok = j.str(&s);
                (key == "trt" ? v.trt : v.trtrtx) = resolve(dir, s);
            } else if (key == "trt_builder") {
                j.ws();
                if (j.p >= j.end || *j.p != '[') {
                    ok = j.fail("trt_builder must be an array of strings");
                    break;
                }
                j.p++;
                j.ws();
                if (j.p < j.end && *j.p == ']') {
                    j.p++;
                } else {
                    while (ok) {
                        std::string s;
                        if (!(ok = j.str(&s)))
                            break;
                        v.trt_builder.push_back(resolve(dir, s));
                        j.ws();
                        if (j.p < j.end && *j.p == ',') {
                            j.p++;
                            continue;
                        }
                        if (j.p < j.end && *j.p == ']') {
                            j.p++;
                            break;
                        }
                        ok = j.fail("invalid JSON");
                    }
                }
            } else {
                ok = j.skip();  // a key from a later format revision
            }
            if (!ok)
                break;
            j.ws();
            if (j.p < j.end && *j.p == ',') {
                j.p++;
                continue;
            }
            if (j.p < j.end && *j.p == '}') {
                j.p++;
                break;
            }
            ok = j.fail("invalid JSON");
        }
    }
    if (ok) {
        j.ws();
        if (j.p != j.end)
            ok = j.fail("trailing data after the JSON object");
    }
    if (!ok) {
        r.error = j.err;
        return r;
    }
    if (format != 1) {
        r.error = format < 0 ? "no \"format\"" :
                  "format " + std::to_string((long long)format) +
                      " is not supported by this libaji (it reads format 1)";
        return r;
    }
    v.present = true;
    return v;
}

// AJI_RUNTIME_JSON names the file instead (UTF-8), for a libaji whose own folder can't be
// written, such as inside an AppImage. Empty: not set.
static std::string runtime_json_env()
{
#ifdef _WIN32
    const wchar_t *w = _wgetenv(L"AJI_RUNTIME_JSON");
    if (!w || !*w)
        return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(len > 0 ? len - 1 : 0, '\0');
    if (len > 1)
        WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], len, nullptr, nullptr);
    return s;
#else
    const char *e = getenv("AJI_RUNTIME_JSON");
    return e ? e : "";
#endif
}

const RuntimePacks &runtime_packs()
{
    static const RuntimePacks packs = [] {
        std::string file = runtime_json_env();
        std::string dir;
        if (file.empty()) {
            dir = own_dir();
            file = dir + SEP + "aji-runtime.json";
        } else {
            // relative paths in it are relative to its own folder
            const size_t slash = file.find_last_of("/\\");
            dir = slash == std::string::npos ? "." : file.substr(0, slash);
        }
        RuntimePacks r;
        std::vector<char> bytes;
        if (!file_exists(file) || !read_file_bytes(file, &bytes))
            return r;  // no file: everything from our own folder
        r = parse_runtime_packs(std::string(bytes.begin(), bytes.end()), dir);
        r.file = file;
        if (!r.error.empty())
            r.error = file + ": " + r.error;
        return r;
    }();
    return packs;
}

bool preload_backend_deps(const char *stem, std::string *err, std::string *what)
{
    // aji_dml loads nothing from the packs, so the file can't break it
    const bool analysis = !strcmp(stem, "aji_analysis_trt");
    if (strcmp(stem, "aji_trt") != 0 && strcmp(stem, "aji_trtrtx") != 0 && !analysis)
        return true;
    const RuntimePacks &rp = runtime_packs();
    if (!rp.present)
        return true;
    if (!rp.error.empty()) {
        *err = rp.error;
        return false;
    }
    // serialized: the video and image dispatchers may load concurrently
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!strcmp(stem, "aji_trt") || analysis) {
        if (rp.trt.empty())
            return true;
        // the observer builds its engine in-process, so it imports the ONNX
        // parser too (aji_trt leaves parsing to trtexec)
        return load_from(rp.trt, AJI_PACK_CUDART, "trt", err, what) &&
               load_from(rp.trt, AJI_PACK_NVINFER, "trt", err, what) &&
               (!analysis || load_from(rp.trt, AJI_PACK_NVONNXPARSER, "trt", err, what));
    }
    if (!strcmp(stem, "aji_trtrtx")) {
        if (rp.trtrtx.empty())
            return true;
        // aji_trtrtx imports cudart too: from the TensorRT-RTX pack, else
        // from the TensorRT one
        const char *cudart = AJI_PACK_CUDART;
        std::string cudart_dir = rp.trtrtx;
        if (*cudart && !file_exists(rp.trtrtx + SEP + cudart) && !rp.trt.empty())
            cudart_dir = rp.trt;
        return load_from(cudart_dir, cudart, "trtrtx", err, what) &&
               load_from(rp.trtrtx, AJI_PACK_TRTRTX, "trtrtx", err, what) &&
               load_from(rp.trtrtx, AJI_PACK_TRTRTX_PARSER, "trtrtx", err, what);
    }
    return true;
}

std::string trt_builder_dir(const std::string &file)
{
    const RuntimePacks &rp = runtime_packs();
    if (!rp.present || !rp.error.empty())
        return "";
    for (const std::string &d : rp.trt_builder)
        if (file_exists(d + SEP + file))
            return d;
    return "";
}

void resolve_trtexec(const std::string &host_trtexec,
                     const std::string &host_env, std::string *trtexec,
                     std::string *env_prefix, std::string *path_prefix)
{
    *trtexec = host_trtexec;
    *env_prefix = host_env;
    path_prefix->clear();
    const RuntimePacks &rp = runtime_packs();
    if (!rp.present || !rp.error.empty() || rp.trt.empty())
        return;
#ifdef _WIN32
    *trtexec = rp.trt + "\\trtexec.exe";
    // trtexec finds the runtime beside it; the builder resources only
    // through PATH (nvinfer loads them by name)
    for (const std::string &d : rp.trt_builder)
        *path_prefix += d + ";";
    *path_prefix += rp.trt;
#else
    *trtexec = rp.trt + "/trtexec";
    std::string lib = rp.trt;
    for (const std::string &d : rp.trt_builder)
        lib += ":" + d;
    // after the host's prefix, so this assignment is the one that counts
    *env_prefix = (host_env.empty() ? "" : host_env + " ") +
                  "LD_LIBRARY_PATH=" + sh_quote(lib);
#endif
}

} // namespace ajicore
