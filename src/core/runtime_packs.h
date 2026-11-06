/*
 * core/runtime_packs.h — shared runtime packs (aji-runtime.json).
 *
 * The TensorRT / TensorRT-RTX runtimes can live in a per-user store of
 * runtime packs, shared between apps, instead of in libaji's own folder.
 * The app's installer (aji-runtime ensure) then writes aji-runtime.json
 * beside aji.dll / libaji.so, naming the pack folders to load from:
 *
 *   { "format": 1,
 *     "trt": "C:\\...\\packs\\trt-runtime\\11.4.0.106-cuda13.4",
 *     "trt_builder": ["C:\\...\\packs\\trt-sm120\\11.4.0.106"],
 *     "trtrtx": "C:\\...\\packs\\trtrtx-runtime\\1.7.1.116-cuda13.4" }
 *
 * Every key but "format" is optional; a relative path is relative to the
 * file's folder. Keys this reader doesn't know are skipped (aji-runtime also
 * writes "models", the app's model pack folders, for the app itself).
 * Without the file everything loads from libaji's own folder, as before.
 *
 * - The dispatchers preload what a backend imports (cudart + nvinfer for
 *   aji_trt; cudart + the TensorRT-RTX runtime for aji_trtrtx) by full
 *   path before loading it, so the backend's imports bind to them.
 * - aji_trt runs the trt pack's trtexec, with the trt and trt_builder
 *   folders on its PATH (Windows) / LD_LIBRARY_PATH (Linux): nvinfer loads
 *   the per-GPU builder resource by file name, which only a search path
 *   finds (see plans/shared-runtime-packs.md in the animejan.ai repo,
 *   "Phase 0 results"). The host's trtexec parameter is then ignored.
 *
 * Part of the internal core: compiled into aji and the TensorRT backends.
 */

#ifndef AJI_CORE_RUNTIME_PACKS_H
#define AJI_CORE_RUNTIME_PACKS_H

#include <string>
#include <vector>

namespace ajicore {

struct RuntimePacks {
    bool present = false;   // aji-runtime.json exists beside the library
    std::string error;      // present but unusable (then nothing below is set)
    std::string file;       // its path
    std::string trt;        // TensorRT runtime pack folder ("" = own folder)
    std::vector<std::string> trt_builder;  // per-GPU builder resource folders
    std::string trtrtx;     // TensorRT-RTX runtime pack folder
};

// Parses <own_dir>/aji-runtime.json once per library and process; the AJI_RUNTIME_JSON environment
// variable names another file instead (for a read-only libaji folder, such as an AppImage's), whose
// own folder then resolves relative paths. A missing file means no packs, as without the variable.
const RuntimePacks &runtime_packs();

// Parses aji-runtime.json text (`dir` resolves relative paths). Exposed
// for tests; runtime_packs() uses it.
RuntimePacks parse_runtime_packs(const std::string &text, const std::string &dir);

// Before loading backend `stem` (aji_trt / aji_trtrtx / aji_dml), loads
// the libraries it imports from the packs aji-runtime.json names. True if
// there was nothing to do or everything loaded; false with *err otherwise.
// *what (if not NULL) gets a log line describing what was loaded.
bool preload_backend_deps(const char *stem, std::string *err,
                          std::string *what);

// The trtexec to run and how: the host's values without aji-runtime.json
// (or without its "trt"), else the trt pack's trtexec with the pack
// folders on its library search path. *path_prefix is for Windows (dirs to
// put in front of the child's PATH); on POSIX the search path goes into
// *env_prefix, after the host's own trtexec_env.
void resolve_trtexec(const std::string &host_trtexec,
                     const std::string &host_env, std::string *trtexec,
                     std::string *env_prefix, std::string *path_prefix);

} // namespace ajicore

#endif
