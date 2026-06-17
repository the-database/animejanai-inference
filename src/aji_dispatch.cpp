/*
 * aji_dispatch.cpp — the aji.dll the mpv filter loads.
 *
 * Thin dispatcher honoring animejanai.conf's [global] backend key
 * (3.3.x semantics: case-insensitive; "directml" -> DML, "ncnn" -> DML
 * with a warning (NCNN is retired; DirectML serves its audience),
 * anything else incl. absent/no conf -> TensorRT). Loads the sibling
 * backend library (aji_trt / aji_dml) from its own directory and
 * forwards the whole C ABI. Keeps nvinfer/onnxruntime out of the
 * process until the chosen backend actually needs them.
 */

#include "aji.h"
#include "aji_conf.h"

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
typedef HMODULE aji_lib;
#else
#  include <dlfcn.h>
#  include <libgen.h>
typedef void *aji_lib;
#endif

struct aji_backend {
    aji_lib lib;
    aji_ctx *(*create)(const aji_create_params *);
    int (*set_slot)(aji_ctx *, int);
    int (*configure)(aji_ctx *, int, int, double, int *, int *);
    int (*infer)(aji_ctx *, const aji_frame *, const aji_frame *, void *);
    uint64_t (*flush)(aji_ctx *, void *);
    int (*done)(aji_ctx *, uint64_t);
    int (*wait)(aji_ctx *, uint64_t);
    const char *(*current_log)(aji_ctx *);
    int (*scale_factor)(aji_ctx *);
    int (*rife_factor)(aji_ctx *, int *, int *);
    int (*rife_before_upscale)(aji_ctx *);
    int (*poll)(aji_ctx *);
    int (*infer_rife)(aji_ctx *, const aji_frame *, const aji_frame *,
                      double, const aji_frame *, void *);
    const char *(*last_error)(aji_ctx *);
    void (*destroy)(aji_ctx **);
};

/* The dispatcher's aji_ctx wraps the backend's. */
struct aji_ctx {
    aji_backend be;
    aji_ctx *inner;
};

static void logf_to(aji_log_fn log, void *opaque, int level,
                    const char *fmt, ...)
{
    if (!log)
        return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log(opaque, level, buf);
}

static std::string own_dir(void)
{
#ifdef _WIN32
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
#else
    Dl_info info = {};
    if (!dladdr((void *)&own_dir, &info) || !info.dli_fname)
        return ".";
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s", info.dli_fname);
    return dirname(buf);
#endif
}

static aji_lib lib_open(const std::string &path)
{
#ifdef _WIN32
    /* full path + search-own-dir flags so the backend's dependencies
     * (nvinfer_*, onnxruntime, DirectML) resolve from the same dir */
    return LoadLibraryExA(path.c_str(), NULL,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                          LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *lib_sym(aji_lib lib, const char *name)
{
#ifdef _WIN32
    return (void *)GetProcAddress(lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void lib_close(aji_lib lib)
{
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
}

static bool load_backend(const char *stem, aji_backend *be,
                         std::string *err)
{
#ifdef _WIN32
    std::string path = own_dir() + "\\" + stem + ".dll";
#else
    std::string path = own_dir() + "/lib" + stem + ".so";
#endif
    be->lib = lib_open(path);
    if (!be->lib) {
#ifdef _WIN32
        *err = path + " failed to load (error " +
               std::to_string(GetLastError()) + ")";
#else
        *err = std::string(dlerror());
#endif
        return false;
    }
#define SYM(field, name) \
    do { \
        *(void **)&be->field = lib_sym(be->lib, name); \
        if (!be->field) { \
            *err = path + ": missing symbol " name; \
            lib_close(be->lib); \
            be->lib = nullptr; \
            return false; \
        } \
    } while (0)
    SYM(create,       "aji_create");
    SYM(set_slot,     "aji_set_slot");
    SYM(configure,    "aji_configure");
    SYM(infer,        "aji_infer");
    SYM(flush,        "aji_flush");
    SYM(done,         "aji_done");
    SYM(wait,         "aji_wait");
    SYM(current_log,  "aji_current_log");
    SYM(scale_factor, "aji_scale_factor");
    SYM(rife_factor,  "aji_rife_factor");
    SYM(rife_before_upscale, "aji_rife_before_upscale");
    SYM(poll,         "aji_poll");
    SYM(infer_rife,   "aji_infer_rife");
    SYM(last_error,   "aji_last_error");
    SYM(destroy,      "aji_destroy");
#undef SYM
    return true;
}

extern "C" AJI_EXPORT aji_ctx *aji_create(const aji_create_params *params)
{
    if (!params || params->api_version != AJI_API_VERSION)
        return nullptr;

    /* 3.3.x semantics: backend lives in [global] of animejanai.conf;
     * direct mode (no conf) is the TRT harness path */
    std::string backend = "TensorRT";
    if (params->conf_path) {
        AjiConf conf;
        std::string err;
        if (aji_conf_load(params->conf_path, &conf, &err))
            backend = conf.backend;
        else
            logf_to(params->log, params->log_opaque, 2,
                    "conf parse failed (%s), using TensorRT", err.c_str());
    }

    std::string lower;
    for (char ch : backend)
        lower.push_back((char)tolower((unsigned char)ch));

    const char *stem = "aji_trt";
    if (lower == "directml") {
        stem = "aji_dml";
    } else if (lower == "ncnn") {
        logf_to(params->log, params->log_opaque, 2,
                "backend=NCNN is retired; using DirectML instead");
        stem = "aji_dml";
    }

    aji_backend be = {};
    std::string err;
    if (!load_backend(stem, &be, &err)) {
        logf_to(params->log, params->log_opaque, 0,
                "backend %s (%s): %s", backend.c_str(), stem, err.c_str());
        return nullptr;
    }
    logf_to(params->log, params->log_opaque, 3,
            "backend: %s (%s)", backend.c_str(), stem);

    aji_ctx *inner = be.create(params);
    if (!inner) {
        lib_close(be.lib);
        return nullptr;
    }
    aji_ctx *c = new aji_ctx{be, inner};
    return c;
}

extern "C" AJI_EXPORT int aji_set_slot(aji_ctx *c, int slot)
{
    return c->be.set_slot(c->inner, slot);
}

extern "C" AJI_EXPORT int aji_configure(aji_ctx *c, int w, int h, double fps,
                                        int *out_w, int *out_h)
{
    return c->be.configure(c->inner, w, h, fps, out_w, out_h);
}

extern "C" AJI_EXPORT int aji_infer(aji_ctx *c, const aji_frame *in,
                                    const aji_frame *out, void *cu_stream)
{
    return c->be.infer(c->inner, in, out, cu_stream);
}

extern "C" AJI_EXPORT uint64_t aji_flush(aji_ctx *c, void *cu_stream)
{
    return c->be.flush(c->inner, cu_stream);
}

extern "C" AJI_EXPORT int aji_done(aji_ctx *c, uint64_t ticket)
{
    return c->be.done(c->inner, ticket);
}

extern "C" AJI_EXPORT int aji_wait(aji_ctx *c, uint64_t ticket)
{
    return c->be.wait(c->inner, ticket);
}

extern "C" AJI_EXPORT const char *aji_current_log(aji_ctx *c)
{
    return c->be.current_log(c->inner);
}

extern "C" AJI_EXPORT int aji_scale_factor(aji_ctx *c)
{
    return c->be.scale_factor(c->inner);
}

extern "C" AJI_EXPORT int aji_rife_factor(aji_ctx *c, int *num, int *den)
{
    return c->be.rife_factor(c->inner, num, den);
}

extern "C" AJI_EXPORT int aji_rife_before_upscale(aji_ctx *c)
{
    return c->be.rife_before_upscale(c->inner);
}

extern "C" AJI_EXPORT int aji_poll(aji_ctx *c)
{
    return c->be.poll(c->inner);
}

extern "C" AJI_EXPORT int aji_infer_rife(aji_ctx *c, const aji_frame *a,
                                         const aji_frame *b, double t,
                                         const aji_frame *out, void *cu_stream)
{
    return c->be.infer_rife(c->inner, a, b, t, out, cu_stream);
}

extern "C" AJI_EXPORT const char *aji_last_error(aji_ctx *c)
{
    return c ? c->be.last_error(c->inner) : "no context";
}

extern "C" AJI_EXPORT void aji_destroy(aji_ctx **c)
{
    if (!c || !*c)
        return;
    aji_ctx *ctx = *c;
    ctx->be.destroy(&ctx->inner);
    lib_close(ctx->be.lib);
    delete ctx;
    *c = nullptr;
}
