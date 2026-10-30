/*
 * aji_img_dispatch.cpp — the image ABI (aji_image.h) in aji.dll.
 *
 * Forwards to the backend named in aji_img_create_params (aji_trt /
 * aji_dml from our own directory), so a consumer loads one library and
 * never links TensorRT or ONNX Runtime. Backend libraries are loaded on
 * first use and stay loaded for the life of the process.
 */

#include "aji_image.h"
#include "core/fileio.h"

#include <cstdio>
#include <mutex>
#include <string>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

struct ImgBackend {
    bool tried = false;
    std::string err;
    void *lib = nullptr;
    int (*enum_devices)(int, aji_img_device *, int);
    aji_img_ctx *(*create)(const aji_img_create_params *, char *, size_t);
    aji_img_model *(*load)(aji_img_ctx *, const aji_img_model_params *);
    int (*state)(aji_img_model *);
    int (*info)(aji_img_model *, aji_img_model_info *);
    int (*upscale)(aji_img_model *, const aji_img_buffer *,
                   const aji_img_buffer *, aji_img_progress_fn, void *);
    void (*release)(aji_img_model **);
    const char *(*last_error)(aji_img_ctx *);
    void (*destroy)(aji_img_ctx **);
};

std::mutex g_mu;
ImgBackend g_trt, g_dml;

void *sym(void *lib, const char *name)
{
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

// Resolves the backend for `backend` once; NULL (with *err) if absent.
ImgBackend *backend_for(int backend, std::string *err)
{
    const char *stem = backend == AJI_IMG_TENSORRT ? "aji_trt"
                     : backend == AJI_IMG_DIRECTML ? "aji_dml"
                                                   : nullptr;
    if (!stem) {
        *err = "unknown backend " + std::to_string(backend);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    ImgBackend &b = backend == AJI_IMG_TENSORRT ? g_trt : g_dml;
    if (!b.tried) {
        b.tried = true;
#ifdef _WIN32
        const std::string path = ajicore::own_dir() + "\\" + stem + ".dll";
        // search-own-dir flags so nvinfer/onnxruntime/DirectML resolve
        // from the backend's directory, not PATH
        b.lib = LoadLibraryExA(path.c_str(), NULL,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!b.lib)
            b.err = path + " failed to load (error " +
                    std::to_string(GetLastError()) + ")";
#else
        const std::string path = ajicore::own_dir() + "/lib" + stem + ".so";
        b.lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!b.lib)
            b.err = dlerror();
#endif
        bool ok = b.lib != nullptr;
#define SYM(field, name) \
        if (ok && !(*(void **)&b.field = sym(b.lib, name))) { \
            b.err = path + ": missing symbol " name " (backend predates the image ABI?)"; \
            ok = false; \
        }
        SYM(enum_devices, "aji_img_enum_devices")
        SYM(create, "aji_img_create")
        SYM(load, "aji_img_load")
        SYM(state, "aji_img_model_state")
        SYM(info, "aji_img_model_info_get")
        SYM(upscale, "aji_img_upscale")
        SYM(release, "aji_img_release")
        SYM(last_error, "aji_img_last_error")
        SYM(destroy, "aji_img_destroy")
#undef SYM
        if (!ok)
            b.lib = nullptr;  // leaked handle is harmless; never retried
    }
    if (!b.lib) {
        *err = b.err;
        return nullptr;
    }
    return &b;
}

} // namespace

/* Dispatcher handles wrap the backend's. */
struct aji_img_ctx {
    ImgBackend *be;
    aji_img_ctx *inner;
};

struct aji_img_model {
    aji_img_ctx *c;
    aji_img_model *inner;
};

extern "C" AJI_IMG_EXPORT int aji_img_enum_devices(int backend,
                                                   aji_img_device *out, int max)
{
    std::string err;
    ImgBackend *be = backend_for(backend, &err);
    return be ? be->enum_devices(backend, out, max) : -1;
}

extern "C" AJI_IMG_EXPORT aji_img_ctx *aji_img_create(const aji_img_create_params *p,
                                                      char *err, size_t err_size)
{
    std::string e;
    ImgBackend *be = p ? backend_for(p->backend, &e) : nullptr;
    if (!be) {
        if (err && err_size)
            snprintf(err, err_size, "%s", p ? e.c_str() : "no params");
        return nullptr;
    }
    aji_img_ctx *inner = be->create(p, err, err_size);
    if (!inner)
        return nullptr;
    return new aji_img_ctx{be, inner};
}

extern "C" AJI_IMG_EXPORT aji_img_model *aji_img_load(aji_img_ctx *c,
                                                      const aji_img_model_params *p)
{
    if (!c)
        return nullptr;
    aji_img_model *inner = c->be->load(c->inner, p);
    return inner ? new aji_img_model{c, inner} : nullptr;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_state(aji_img_model *m)
{
    return m ? m->c->be->state(m->inner) : AJI_IMG_STATE_FAILED;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_info_get(aji_img_model *m,
                                                     aji_img_model_info *info)
{
    return m ? m->c->be->info(m->inner, info) : AJI_IMG_ERR_ARGS;
}

extern "C" AJI_IMG_EXPORT int aji_img_upscale(aji_img_model *m,
                                              const aji_img_buffer *in,
                                              const aji_img_buffer *out,
                                              aji_img_progress_fn progress,
                                              void *opaque)
{
    return m ? m->c->be->upscale(m->inner, in, out, progress, opaque)
             : AJI_IMG_ERR_ARGS;
}

extern "C" AJI_IMG_EXPORT void aji_img_release(aji_img_model **pm)
{
    if (!pm || !*pm)
        return;
    (*pm)->c->be->release(&(*pm)->inner);
    delete *pm;
    *pm = nullptr;
}

extern "C" AJI_IMG_EXPORT const char *aji_img_last_error(aji_img_ctx *c)
{
    return c ? c->be->last_error(c->inner) : "no context";
}

extern "C" AJI_IMG_EXPORT void aji_img_destroy(aji_img_ctx **pc)
{
    if (!pc || !*pc)
        return;
    // the backend releases any models still loaded; their dispatcher
    // wrappers are the caller's to have released first (documented leak
    // otherwise, same as forgetting aji_img_release)
    (*pc)->be->destroy(&(*pc)->inner);
    delete *pc;
    *pc = nullptr;
}
