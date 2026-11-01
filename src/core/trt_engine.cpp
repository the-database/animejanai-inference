/*
 * core/trt_engine.cpp — see trt_engine.h.
 */

#include "trt_engine.h"
#include "fileio.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#ifdef AJI_TRTRTX
#include <NvOnnxParser.h>
#endif

namespace fs = std::filesystem;

namespace ajicore {

#ifdef AJI_TRTRTX
const char *const ENGINE_EXT = ".rtxengine";
static const char *const VERSION_TAG = ".trtrtx-";
// JIT (runtime) cache beside each engine: {engine}.jitcache
static const char *const JIT_CACHE_EXT = ".jitcache";
#else
const char *const ENGINE_EXT = ".engine";
static const char *const VERSION_TAG = ".trt-";
#endif

void TrtLogger::log(Severity sev, const char *msg) noexcept
{
    if (fn)
        fn(opaque, (int)sev, msg);
    else if (sev <= Severity::kWARNING)
        fprintf(stderr, "[aji-trt] %s\n", msg);
}

uint32_t crc32_z(const std::string &data)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (unsigned char ch : data)
        c = table[(c ^ ch) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::string sanitize_token(std::string s)
{
    for (auto &ch : s) {
        if (ch == ' ')
            ch = '-';
    }
    std::string out;
    for (char ch : s) {
        if (isalnum((unsigned char)ch) || ch == '.' || ch == '_' || ch == '-')
            out += ch;
    }
    return out.empty() ? "device0" : out;
}

std::string trt_version_token()
{
#ifdef AJI_TRTRTX
    // TensorRT-RTX's packed getInferLibVersion() has its own encoding;
    // the split getters are unambiguous
    char rbuf[32];
    snprintf(rbuf, sizeof(rbuf), "%d.%d.%d", getInferLibMajorVersion(),
             getInferLibMinorVersion(), getInferLibPatchVersion());
    return rbuf;
#else
    int32_t v = getInferLibVersion();
    int major, minor, patch;
    if (v < 10000) {
        major = v / 1000; minor = (v / 100) % 10; patch = v % 100;
    } else {
        major = v / 10000; minor = (v / 100) % 100; patch = v % 100;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d", major, minor, patch);
    return buf;
#endif
}

std::string gpu_token(int dev)
{
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return "unknown";
    std::string tok = sanitize_token(prop.name);
    return tok + "-sm" + std::to_string(prop.major);
}

std::string engine_suffix(int dev)
{
    return VERSION_TAG + trt_version_token() + ".gpu-" + gpu_token(dev) +
           ENGINE_EXT;
}

std::string engine_path_for(const std::string &model_dir,
                            const std::string &onnx_name,
                            const std::string &settings, int dev)
{
    return (fs::path(model_dir) /
            (onnx_name + "." + std::to_string(crc32_z(settings)) + engine_suffix(dev)))
        .string();
}

std::string short_engine_path_for(const std::string &model_dir,
                                  const std::string &onnx_name,
                                  const std::string &settings, int dev)
{
    char model_hash[9];
    snprintf(model_hash, sizeof(model_hash), "%08x", crc32_z(onnx_name));
    return (fs::path(model_dir) /
            ("aji-" + std::string(model_hash) + "." +
             std::to_string(crc32_z(settings)) + engine_suffix(dev)))
        .string();
}

void clean_stale_engines(const std::string &dir, int fallback_dev,
                         std::vector<std::string> *removed)
{
    // Keep the engines of every GPU in the system, not just the active one:
    // a multi-GPU user moving work between GPUs would otherwise rebuild
    // everything on each switch.
    std::vector<std::string> suffixes;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
        ndev = 0;
    for (int d = 0; d < ndev; d++)
        suffixes.push_back(engine_suffix(d));
    if (suffixes.empty())
        suffixes.push_back(engine_suffix(fallback_dev));
    auto ends_with = [](const std::string &s, const std::string &tail) {
        return s.size() >= tail.size() &&
               s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
    };
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        std::string key = name;  // the engine name this file belongs to
#ifdef AJI_TRTRTX
        if (ends_with(key, JIT_CACHE_EXT))  // an engine's JIT cache goes with it
            key.resize(key.size() - strlen(JIT_CACHE_EXT));
#endif
        if (!ends_with(key, ENGINE_EXT))
            continue;
        bool current = false;
        for (const std::string &suffix : suffixes)
            current |= ends_with(key, suffix);
        if (current)
            continue;
        std::error_code rec;
        if (fs::remove(e.path(), rec))
            removed->push_back(name);
    }
}

std::string shapes_label(const std::string &settings)
{
    auto dims = [&](const char *key) -> std::string {
        size_t p = settings.find(key);
        if (p == std::string::npos)
            return "";
        p = settings.find(':', p);  // --xxxShapes=input:NxCxHxW
        if (p == std::string::npos)
            return "";
        long v[8];
        int n = 0;
        size_t i = p + 1;
        while (n < 8 && i < settings.size() &&
               isdigit((unsigned char)settings[i])) {
            v[n++] = strtol(settings.c_str() + i, nullptr, 10);
            while (i < settings.size() && isdigit((unsigned char)settings[i]))
                i++;
            if (i < settings.size() && settings[i] == 'x')
                i++;
        }
        if (n < 2)
            return "";
        return std::to_string(v[n - 1]) + "x" + std::to_string(v[n - 2]);
    };
    const std::string mn = dims("--minShapes");
    const std::string mx = dims("--maxShapes");
    const std::string op = dims("--optShapes");
    if (!mn.empty() && !mx.empty() && mn != mx)
        return mn + "-" + mx;
    if (!op.empty())
        return op;
    return !mx.empty() ? mx : mn;
}

bool parse_shape_wh(const std::string &settings, const char *key, int *w, int *h)
{
    size_t p = settings.find(key);
    if (p == std::string::npos)
        return false;
    p = settings.find(':', p);
    if (p == std::string::npos)
        return false;
    long v[8];
    int n = 0;
    size_t i = p + 1;
    while (n < 8 && i < settings.size() &&
           isdigit((unsigned char)settings[i])) {
        v[n++] = strtol(settings.c_str() + i, nullptr, 10);
        while (i < settings.size() && isdigit((unsigned char)settings[i]))
            i++;
        if (i < settings.size() && settings[i] == 'x')
            i++;
    }
    if (n < 2)
        return false;
    *w = (int)v[n - 1];  // ...HxW
    *h = (int)v[n - 2];
    return true;
}

int run_build_process(const BuildSpec &spec, std::atomic<intptr_t> *child)
{
#ifdef AJI_TRTRTX
    // On a thread of its own, joined: the build selects its device
    // (cudaSetDevice) and makes contexts current, which must not leak into
    // the caller's thread (a host app's runtime calls would land on the
    // wrong device). Background builds already run on AsyncBuild's worker;
    // the extra thread costs nothing next to a build.
    int rc = 1;
    std::thread t([&]() { rc = run_build_inprocess(spec, child); });
    t.join();
    return rc;
#elif defined(_WIN32)
    SECURITY_ATTRIBUTES sa = {sizeof(sa), nullptr, TRUE};
    HANDLE log = CreateFileW(win_long_path(spec.log_path).c_str(), GENERIC_WRITE,
                             FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE)
        return -1;
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = log;
    si.hStdError = log;
    PROCESS_INFORMATION pi = {};
    // The command line carries UTF-8 paths (model dir, engine, timing cache).
    // CreateProcessA reinterprets those bytes through the process ANSI code
    // page, handing trtexec a mangled path on any install whose path is not
    // representable there. Widen and spawn wide, so the child gets the real
    // path whatever the system locale is.
    std::wstring cmd = widen_utf8(spec.cmdline);  // CreateProcess may modify it
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    if (log != INVALID_HANDLE_VALUE)
        CloseHandle(log);
    if (!ok)
        return -1;
    CloseHandle(pi.hThread);
    child->store((intptr_t)pi.hProcess);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    child->store(0);
    CloseHandle(pi.hProcess);
    return (int)code;
#else
    std::string full = (spec.env_prefix.empty() ? "" : spec.env_prefix + " ") +
                       spec.cmdline + " >\"" + spec.log_path + "\" 2>&1";
    pid_t pid = fork();
    if (pid < 0)
        return -1;
    if (pid == 0) {
        execl("/bin/sh", "sh", "-c", full.c_str(), (char *)nullptr);
        _exit(127);
    }
    child->store((intptr_t)pid);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    child->store(0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

// trtexec creates its --saveEngine file empty when it starts and only fills
// it at the end, so a build that is killed (the app's Cancel, closing the
// player, a crash) used to leave an empty engine under the real cache name.
// It builds into tmp_path instead, renamed into place only after a successful
// build, so the cache only ever holds complete engines.
bool run_build_spec(const BuildSpec &spec, std::atomic<intptr_t> *child)
{
#ifdef AJI_TRTRTX
    if (spec.warm_only)  // the engine exists; on failure it stays
        return run_build_process(spec, child) == 0;
#endif
    remove_file(spec.tmp_path);  // a previous build's leftover
    int rc = run_build_process(spec, child);
    if (rc != 0 || file_size(spec.tmp_path) <= 0 ||
        !move_file(spec.tmp_path, spec.engine_path)) {
        remove_file(spec.tmp_path);
        remove_file(spec.engine_path);
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "\n[aji] build process exit code: %d (0x%x)\n", rc,
                 (unsigned)rc);
        append_file_text(spec.log_path, msg);
        return false;
    }
    return true;
}

#if NV_TENSORRT_MAJOR >= 11
namespace {
// trtexec 11 dropped the weak-typing era flags; user confs migrated
// from 3.3.x may still carry them. Strip from the command line only -
// the engine-name CRC keeps hashing the original settings string.
std::string sanitize_settings_trt11(std::string s)
{
    static const char *dropped[] = {"--fp16", "--bf16", "--int8", "--best",
                                    "--buildOnly"};
    for (const char *flag : dropped) {
        size_t pos = 0;
        while ((pos = s.find(flag, pos)) != std::string::npos) {
            size_t end = pos + strlen(flag);
            // only whole flags (--fp16 but not --fp16something)
            if (end < s.size() && s[end] != ' ') {
                pos = end;
                continue;
            }
            while (end < s.size() && s[end] == ' ')
                end++;
            s.erase(pos, end - pos);
        }
    }
    // typed IO formats are weak-typing flags: types now come from the
    // network, and the fp16:/fp32: value prefixes no longer parse
    // the cuDNN/cuBLAS tactic-source enum values are gone too (their
    // tactics were already removed during 10.x)
    static const char *dropped_kv[] = {"--inputIOFormats=",
                                       "--outputIOFormats=",
                                       "--tacticSources="};
    for (const char *flag : dropped_kv) {
        size_t pos;
        while ((pos = s.find(flag)) != std::string::npos) {
            size_t end = s.find(' ', pos);
            end = end == std::string::npos ? s.size() : end + 1;
            s.erase(pos, end - pos);
        }
    }
    // --workspace=N became --memPoolSize=workspace:NM
    size_t pos = s.find("--workspace=");
    if (pos != std::string::npos) {
        size_t vs = pos + strlen("--workspace=");
        size_t ve = s.find(' ', vs);
        std::string val = s.substr(vs, ve == std::string::npos ? ve
                                                               : ve - vs);
        s.replace(pos, (ve == std::string::npos ? s.size() : ve) - pos,
                  "--memPoolSize=workspace:" + val + "M");
    }
    return s;
}
} // namespace
#endif

bool make_build_spec(const std::string &trtexec, const std::string &trtexec_env,
                     const std::string &dir, const std::string &onnx_name,
                     const std::string &settings, const std::string &engine_path,
                     int dev, BuildSpec *spec, std::string *err)
{
    // Persistent per-model timing cache: tactic timings get reused across
    // builds (other resolutions, rebuilds), which speeds builds up and is
    // NVIDIA's mitigation for timing noise when the GPU isn't idle - our
    // background builds run alongside playback.
    return make_build_spec_paths(
        trtexec, trtexec_env, (fs::path(dir) / (onnx_name + ".onnx")).string(),
        (fs::path(dir) / (onnx_name + ".timing.cache")).string(), onnx_name,
        settings, engine_path, dev, spec, err);
}

bool make_build_spec_paths(const std::string &trtexec,
                           const std::string &trtexec_env,
                           const std::string &onnx_path,
                           const std::string &tcache,
                           const std::string &name, const std::string &settings,
                           const std::string &engine_path, int dev,
                           BuildSpec *spec, std::string *err)
{
    if (!file_exists(onnx_path)) {
        *err = "model not found: " + onnx_path;
        return false;
    }
    // The timing cache is appended to the command line only: the settings
    // string feeds the engine filename CRC and must stay Python-compatible.
#if NV_TENSORRT_MAJOR >= 11
    const std::string cmd_settings = sanitize_settings_trt11(settings);
#else
    const std::string &cmd_settings = settings;
#endif
#ifdef AJI_TRTRTX
    // built in-process: no trtexec, no timing cache (TensorRT-RTX has no
    // autotuning to cache); the command line is only a log description
    (void)trtexec;
    (void)tcache;
    (void)cmd_settings;
    spec->cmdline = "in-process TensorRT-RTX build of \"" + onnx_path +
                    "\" (" + settings + ")";
    if (dev != 0)
        spec->cmdline += " on device " + std::to_string(dev);
#else
    spec->cmdline = "\"" + trtexec + "\" --onnx=\"" + onnx_path +
                    "\" --saveEngine=\"" + engine_path + ".tmp\" " +
                    cmd_settings +
                    " --timingCacheFile=\"" + tcache + "\"";
    // trtexec builds on device 0 by default; build on the GPU that runs
    // the engine (left off for device 0 to keep the usual command line)
    if (dev != 0)
        spec->cmdline += " --device=" + std::to_string(dev);
#endif
    spec->env_prefix = trtexec_env;
    spec->engine_path = engine_path;
    spec->tmp_path = engine_path + ".tmp";
    spec->log_path = engine_path + ".build.log";
    spec->name = name;
    spec->res = shapes_label(settings);
    spec->onnx_path = onnx_path;
    spec->settings = settings;
    spec->dev = dev;
    return true;
}

void AsyncBuild::start(const BuildSpec &spec)
{
    if (thread.joinable())
        thread.join();  // already-finished worker awaiting reap
    epath = spec.engine_path;
    done_flag = 0;
    ok = false;
    child = 0;
    running = true;
    thread = std::thread([this, spec]() {
        bool r = run_build_spec(spec, &child);
        ok = r;
        running = false;
        done_flag = 1;
    });
}

void AsyncBuild::stop()
{
    if (running.load()) {
#ifdef AJI_TRTRTX
        child = 1;  // in-process build: the progress monitor cancels
#else
        intptr_t h = child.load();
        if (h) {
#ifdef _WIN32
            TerminateProcess((HANDLE)h, 1);
#else
            kill((pid_t)h, SIGKILL);
#endif
        }
#endif
    }
    if (thread.joinable())
        thread.join();
#ifdef AJI_TRTRTX
    child = 0;  // the cancel flag; a later synchronous build must not see it
#endif
}

#ifdef AJI_TRTRTX
namespace {

// Builder log -> the engine's .build.log, like trtexec's output. One
// handle for the whole build, opened (truncating) at the first message at
// or above `min_sev`, flushed per line so a crash keeps what was logged.
class FileLogger : public nvinfer1::ILogger {
public:
    std::string path;
    Severity min_sev = Severity::kINFO;
    ~FileLogger() override
    {
        if (f)
            fclose(f);
    }
    void log(Severity sev, const char *msg) noexcept override
    {
        if (sev > min_sev)
            return;
        static const char *tags[] = {"F", "E", "W", "I", "V"};
        std::lock_guard<std::mutex> lock(mu);
        if (!f && !opened) {
            opened = true;
            f = fopen_utf8(path, "w");
        }
        if (f) {
            fprintf(f, "[%s] %s\n", tags[(int)sev], msg);
            fflush(f);
        }
    }

private:
    std::mutex mu;
    std::FILE *f = nullptr;
    bool opened = false;
};

class CancelMonitor : public nvinfer1::IProgressMonitor {
public:
    std::atomic<intptr_t> *cancel = nullptr;
    void phaseStart(const char *, const char *, int32_t) noexcept override {}
    bool stepComplete(const char *, int32_t) noexcept override
    {
        return !cancel || cancel->load() == 0;
    }
    void phaseFinish(const char *) noexcept override {}
};

// trtexec shape flag value "name:AxBxC[,name2:...]" -> dims of `input`.
bool shape_flag_dims(const std::string &settings, const char *flag,
                     const std::string &input, nvinfer1::Dims *out)
{
    size_t p = settings.find(flag);
    if (p == std::string::npos)
        return false;
    p += strlen(flag);
    size_t end = settings.find(' ', p);
    std::string val = settings.substr(p, end == std::string::npos ? end : end - p);
    size_t start = 0;
    while (start <= val.size()) {
        size_t comma = val.find(',', start);
        std::string item = val.substr(start, comma == std::string::npos
                                                 ? comma : comma - start);
        size_t colon = item.rfind(':');
        if (colon != std::string::npos) {
            std::string name = item.substr(0, colon);
            if (name.size() >= 2 && name.front() == '\'' && name.back() == '\'')
                name = name.substr(1, name.size() - 2);
            if (name == input) {
                nvinfer1::Dims d{};
                const char *q = item.c_str() + colon + 1;
                while (*q && d.nbDims < nvinfer1::Dims::MAX_DIMS) {
                    char *e = nullptr;
                    d.d[d.nbDims++] = strtoll(q, &e, 10);
                    if (e == q)
                        return false;
                    if (*e != 'x')
                        break;
                    q = e + 1;
                }
                *out = d;
                return d.nbDims > 0;
            }
        }
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return false;
}

bool int_flag(const std::string &settings, const char *flag, long long *v)
{
    size_t p = settings.find(flag);
    if (p == std::string::npos)
        return false;
    const char *b = settings.c_str() + p + strlen(flag);
    char *e = nullptr;
    *v = strtoll(b, &e, 10);
    return e != b;
}

} // namespace

bool forced_failure(const char *stage)
{
    const char *v = getenv("AJI_TRTRTX_FAIL");
    return v && !strcmp(v, stage);
}

namespace {

size_t dtype_bytes(nvinfer1::DataType t)
{
    switch (t) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT64: return 8;
    default: return 1;
    }
}

// Fill rc->cache from `path` (if it holds a compatible cache) and point rc
// at it; a mismatched file (other driver/GPU/version) is removed.
void load_runtime_cache(RuntimeCache *rc, const std::string &path)
{
    std::vector<char> blob;
    if (read_file_bytes(path, &blob) && !blob.empty()) {
        if (rc->cache->deserialize(blob.data(), blob.size())) {
            rc->loaded_size = blob.size();
            rc->loaded_crc = crc32_z(std::string(blob.data(), blob.size()));
        } else {
            remove_file(path);
        }
    }
    rc->path = path;
}

// Load a just-built plan with the runtime cache at `jit_cache`, warm it up
// (at the profile's opt shapes, i.e. the build shape) and save the cache.
bool warm_built_plan(nvinfer1::ILogger &log, const void *plan, size_t size,
                     const std::string &jit_cache, std::string *err)
{
    // declared so that they are destroyed context, engine, cache, runtime -
    // the order RuntimeCache needs - on every return path
    std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(log));
    RuntimeCache rc;
    std::unique_ptr<nvinfer1::ICudaEngine> eng(
        rt ? rt->deserializeCudaEngine(plan, size) : nullptr);
    if (!eng) {
        *err = "cannot load the engine";
        return false;
    }
    rc.config.reset(eng->createRuntimeConfig());
    if (rc.config) {
        rc.cache.reset(rc.config->createRuntimeCache());
        if (rc.cache) {
            load_runtime_cache(&rc, jit_cache);
            rc.config->setRuntimeCache(*rc.cache);
        }
    }
    std::unique_ptr<nvinfer1::IExecutionContext> exec(
        rc.config ? eng->createExecutionContext(rc.config.get())
                  : eng->createExecutionContext());
    if (!exec) {
        *err = "createExecutionContext failed";
        return false;
    }
    bool ok = warm_up(exec.get(), err);
    exec.reset();
    eng.reset();
    rc.save();  // still on this thread's (current) context
    return ok;
}

// Warm-only job (BuildSpec::warm_only): load the cached engine and warm it.
int run_warm_inprocess(const BuildSpec &spec)
{
    FileLogger log;
    log.path = spec.log_path;
    log.min_sev = nvinfer1::ILogger::Severity::kWARNING;  // only if trouble
    if (cudaSetDevice(spec.dev) != cudaSuccess) {
        log.log(nvinfer1::ILogger::Severity::kERROR, "cudaSetDevice failed");
        return 1;
    }
    std::vector<char> blob;
    std::string err;
    if (!read_file_bytes(spec.engine_path, &blob) ||
        !warm_built_plan(log, blob.data(), blob.size(),
                         spec.engine_path + JIT_CACHE_EXT, &err)) {
        log.log(nvinfer1::ILogger::Severity::kERROR,
                ("warm-up of " + spec.engine_path + " failed: " + err).c_str());
        return 1;
    }
    return 0;
}

} // namespace

bool warm_up(nvinfer1::IExecutionContext *exec, std::string *err)
{
    const nvinfer1::ICudaEngine &eng = exec->getEngine();
    const int n = eng.getNbIOTensors();
    // dynamic inputs not set yet: the profile's opt shape
    for (int i = 0; i < n; i++) {
        const char *name = eng.getIOTensorName(i);
        if (eng.getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;
        nvinfer1::Dims d = exec->getTensorShape(name);
        bool unset = d.nbDims < 0;
        for (int k = 0; k < d.nbDims; k++)
            unset |= d.d[k] < 0;
        if (unset &&
            !exec->setInputShape(name, eng.getProfileShape(
                                           name, 0, nvinfer1::OptProfileSelector::kOPT))) {
            *err = std::string("warm-up: cannot set the shape of ") + name;
            return false;
        }
    }
    std::vector<void *> bufs;
    cudaStream_t stream = nullptr;
    bool ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
    for (int i = 0; ok && i < n; i++) {
        const char *name = eng.getIOTensorName(i);
        nvinfer1::Dims d = exec->getTensorShape(name);
        size_t bytes = dtype_bytes(eng.getTensorDataType(name));
        for (int k = 0; k < d.nbDims; k++)
            bytes *= (size_t)(d.d[k] > 0 ? d.d[k] : 1);
        void *p = nullptr;
        ok = cudaMalloc(&p, bytes) == cudaSuccess;
        if (ok) {
            bufs.push_back(p);
            ok = cudaMemsetAsync(p, 0, bytes, stream) == cudaSuccess &&
                 exec->setTensorAddress(name, p);
        }
    }
    if (!ok)
        *err = "warm-up: scratch buffer setup failed";
    if (ok && !exec->enqueueV3(stream)) {
        *err = "warm-up inference (enqueueV3) failed";
        ok = false;
    }
    if (stream && cudaStreamSynchronize(stream) != cudaSuccess && ok) {
        *err = "warm-up inference failed on the GPU";
        ok = false;
    }
    for (void *p : bufs)
        cudaFree(p);
    if (stream)
        cudaStreamDestroy(stream);
    // the scratch addresses are gone: callers bind their own before every
    // enqueue
    return ok;
}

int run_build_inprocess(const BuildSpec &spec, std::atomic<intptr_t> *cancel)
{
    if (spec.warm_only)
        return run_warm_inprocess(spec);
    FileLogger log;
    log.path = spec.log_path;
    remove_file(spec.log_path);
    auto fail = [&](const std::string &msg) {
        log.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return 1;
    };
    log.log(nvinfer1::ILogger::Severity::kINFO, spec.cmdline.c_str());
    if (forced_failure("build"))
        return fail("forced build failure (AJI_TRTRTX_FAIL=build)");

    // The builder targets the current device (ComputeCapability::kCURRENT).
    // This always runs on a thread of its own (run_build_process, or
    // AsyncBuild's worker), so selecting the device here touches no one
    // else's state.
    if (cudaSetDevice(spec.dev) != cudaSuccess)
        return fail("cudaSetDevice(" + std::to_string(spec.dev) + ") failed");

    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(log));
    if (!builder)
        return fail("createInferBuilder failed");
    // TensorRT-RTX networks are always strongly typed
    std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(
        net ? nvonnxparser::createParser(*net, log) : nullptr);
    if (!parser)
        return fail("cannot create the network/ONNX parser");
    // read through fileio (UTF-8 / long paths), not parseFromFile
    std::vector<char> blob;
    if (!read_file_bytes(spec.onnx_path, &blob))
        return fail("cannot read " + spec.onnx_path);
    if (!parser->parse(blob.data(), blob.size())) {
        for (int i = 0; i < parser->getNbErrors(); i++)
            log.log(nvinfer1::ILogger::Severity::kERROR,
                    parser->getError(i)->desc());
        return fail("ONNX parse failed: " + spec.onnx_path);
    }
    blob.clear();
    blob.shrink_to_fit();

    std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
    if (!cfg)
        return fail("createBuilderConfig failed");
    cfg->setNbComputeCapabilities(1);
    cfg->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT, 0);
    long long v;
    if (int_flag(spec.settings, "--builderOptimizationLevel=", &v))
        cfg->setBuilderOptimizationLevel((int32_t)v);
    if (int_flag(spec.settings, "--memPoolSize=workspace:", &v))
        cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                (size_t)v << 20);  // trtexec's default unit: MiB
    CancelMonitor mon;
    mon.cancel = cancel;
    cfg->setProgressMonitor(&mon);

    // trtexec's shape semantics: a missing opt is the max (else the min), a
    // missing min/max is the opt; without any shape flag every dynamic
    // dimension becomes 1
    nvinfer1::IOptimizationProfile *prof = nullptr;
    for (int i = 0; i < net->getNbInputs(); i++) {
        nvinfer1::ITensor *t = net->getInput(i);
        nvinfer1::Dims d = t->getDimensions();
        bool dynamic = false;
        for (int k = 0; k < d.nbDims; k++)
            dynamic |= d.d[k] < 0;
        nvinfer1::Dims mn{}, op{}, mx{};
        const bool hmn = shape_flag_dims(spec.settings, "--minShapes=", t->getName(), &mn);
        const bool hop = shape_flag_dims(spec.settings, "--optShapes=", t->getName(), &op);
        const bool hmx = shape_flag_dims(spec.settings, "--maxShapes=", t->getName(), &mx);
        if (!dynamic)
            continue;  // a static input takes no profile entry
        if (!hmn && !hop && !hmx) {
            for (int k = 0; k < d.nbDims; k++)
                if (d.d[k] < 0)
                    d.d[k] = 1;
            mn = op = mx = d;
        } else {
            if (!hop)
                op = hmx ? mx : mn;
            if (!hmn)
                mn = op;
            if (!hmx)
                mx = op;
        }
        if (!prof)
            prof = builder->createOptimizationProfile();
        if (!prof ||
            !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMIN, mn) ||
            !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kOPT, op) ||
            !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMAX, mx))
            return fail(std::string("bad shape profile for input ") + t->getName());
    }
    if (prof && cfg->addOptimizationProfile(prof) < 0)
        return fail("addOptimizationProfile failed");

    std::unique_ptr<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*net, *cfg));
    if (!plan || !plan->size())
        return fail(cancel && cancel->load() ? "build cancelled"
                                             : "buildSerializedNetwork failed");
    if (!write_file_bytes(spec.tmp_path, plan->data(), plan->size()))
        return fail("cannot write " + spec.tmp_path);

    // Warm the new engine here, on the build thread, so the kernels it
    // compiles at first use land in the JIT caches before anyone plays it.
    // A failure here fails the build: the engine would not run anyway.
    // Release the builder first (its memory is no longer needed). TensorRT
    // requires everything a builder created to be destroyed before it, so
    // keep a copy of the plan and release the rest in reverse order.
    std::vector<char> engine((const char *)plan->data(),
                             (const char *)plan->data() + plan->size());
    plan.reset();
    cfg.reset();
    parser.reset();
    net.reset();
    builder.reset();
    std::string werr;
    const auto t0 = std::chrono::steady_clock::now();
    if (!warm_built_plan(log, engine.data(), engine.size(),
                         spec.engine_path + JIT_CACHE_EXT, &werr))
        return fail("warm-up of the built engine failed: " + werr);
    char msg[96];
    snprintf(msg, sizeof(msg), "warm-up inference done (%.2f s)",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                 .count());
    log.log(nvinfer1::ILogger::Severity::kINFO, msg);
    return 0;
}

void RuntimeCache::save()
{
    if (cache && !path.empty()) {
        // serializing is cheap (in memory); rewriting the file on every
        // configure / slot switch is not worth it when nothing compiled
        std::unique_ptr<nvinfer1::IHostMemory> blob(cache->serialize());
        const bool changed =
            blob && blob->size() &&
            (blob->size() != loaded_size ||
             crc32_z(std::string((const char *)blob->data(), blob->size())) !=
                 loaded_crc);
        if (changed &&
            write_file_bytes(path + ".tmp", blob->data(), blob->size()) &&
            !move_file(path + ".tmp", path))
            remove_file(path + ".tmp");
    }
    config.reset();
    cache.reset();
}
#endif

bool load_engine(nvinfer1::IRuntime *runtime, const std::string &engine_path,
                 ModelEngine *me, int in_w, int in_h, int in_ch, int rank5_t,
                 std::string *err, bool *keep)
{
    if (keep)
        *keep = false;
    std::vector<char> blob;
    if (!read_file_bytes(engine_path, &blob)) {
        *err = "cannot open engine: " + engine_path;
        return false;
    }
#ifdef AJI_TRTRTX
    me->exec.reset();  // a previous engine's context goes before its cache
    me->engine.reset();
    me->rt = RuntimeCache();
#endif
    me->engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!me->engine) {
        *err = "engine deserialization failed: " + engine_path;
        return false;
    }
    // from here on a failure is not the file's fault
    if (keep)
        *keep = true;
    for (int i = 0; i < me->engine->getNbIOTensors(); i++) {
        const char *name = me->engine->getIOTensorName(i);
        if (me->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            me->in_name = name;
        else
            me->out_name = name;
    }
    if (!me->in_name || !me->out_name) {
        *err = "engine I/O tensors not found";
        return false;
    }
    me->in_fp32 = me->engine->getTensorDataType(me->in_name) ==
                  nvinfer1::DataType::kFLOAT;
    me->out_fp32 = me->engine->getTensorDataType(me->out_name) ==
                   nvinfer1::DataType::kFLOAT;
#ifdef AJI_TRTRTX
    // Kernels compile for this GPU here (JIT), at context creation; the
    // runtime cache beside the engine makes that a load after first use.
    me->rt.config.reset(me->engine->createRuntimeConfig());
    if (me->rt.config) {
        me->rt.cache.reset(me->rt.config->createRuntimeCache());
        if (me->rt.cache) {
            load_runtime_cache(&me->rt, engine_path + JIT_CACHE_EXT);
            me->rt.config->setRuntimeCache(*me->rt.cache);
        }
    }
    if (!forced_failure("context"))
        me->exec.reset(me->rt.config
                           ? me->engine->createExecutionContext(me->rt.config.get())
                           : me->engine->createExecutionContext());
#else
    me->exec.reset(me->engine->createExecutionContext());
#endif
    if (!me->exec) {
        *err = "createExecutionContext failed";
        return false;
    }
    nvinfer1::Dims in_dims = nvinfer1::Dims4{1, in_ch, in_h, in_w};
    if (rank5_t > 0) {
        in_dims.nbDims = 5;
        in_dims.d[0] = 1;
        in_dims.d[1] = rank5_t;
        in_dims.d[2] = 3;
        in_dims.d[3] = in_h;
        in_dims.d[4] = in_w;
    }
    if (!me->exec->setInputShape(me->in_name, in_dims)) {
        *err = "engine rejects input " + std::to_string(in_w) + "x" +
               std::to_string(in_h) + " (" + engine_path + ")";
        return false;
    }
    nvinfer1::Dims od = me->exec->getTensorShape(me->out_name);
    if (od.nbDims != 4) {
        *err = "unexpected output rank from " + engine_path;
        return false;
    }
    me->in_w = in_w;
    me->in_h = in_h;
    me->out_w = (int)od.d[3];
    me->out_h = (int)od.d[2];
#ifdef AJI_TRTRTX
    // One throwaway inference at the configured shape, so the first real
    // frame doesn't pay TensorRT-RTX's first-enqueue compiles (cheap when
    // the build thread already warmed the caches).
    std::string werr;
    if (!warm_up(me->exec.get(), &werr)) {
        *err = werr + " (" + engine_path + ")";
        return false;
    }
#endif
    return true;
}

} // namespace ajicore
