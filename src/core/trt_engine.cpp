/*
 * core/trt_engine.cpp — see trt_engine.h.
 */

#include "trt_engine.h"
#include "fileio.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

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

namespace fs = std::filesystem;

namespace ajicore {

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
    return ".trt-" + trt_version_token() + ".gpu-" + gpu_token(dev) + ".engine";
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
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.size() < 7 || name.substr(name.size() - 7) != ".engine")
            continue;
        bool current = false;
        for (const std::string &suffix : suffixes) {
            current |= name.size() >= suffix.size() &&
                       name.compare(name.size() - suffix.size(), suffix.size(),
                                    suffix) == 0;
        }
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
#ifdef _WIN32
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
    spec->cmdline = "\"" + trtexec + "\" --onnx=\"" + onnx_path +
                    "\" --saveEngine=\"" + engine_path + ".tmp\" " +
                    cmd_settings +
                    " --timingCacheFile=\"" + tcache + "\"";
    // trtexec builds on device 0 by default; build on the GPU that runs
    // the engine (left off for device 0 to keep the usual command line)
    if (dev != 0)
        spec->cmdline += " --device=" + std::to_string(dev);
    spec->env_prefix = trtexec_env;
    spec->engine_path = engine_path;
    spec->tmp_path = engine_path + ".tmp";
    spec->log_path = engine_path + ".build.log";
    spec->name = name;
    spec->res = shapes_label(settings);
    return true;
}

void AsyncBuild::start(const BuildSpec &spec)
{
    if (thread.joinable())
        thread.join();  // already-finished worker awaiting reap
    epath = spec.engine_path;
    done_flag = 0;
    ok = false;
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
        intptr_t h = child.load();
        if (h) {
#ifdef _WIN32
            TerminateProcess((HANDLE)h, 1);
#else
            kill((pid_t)h, SIGKILL);
#endif
        }
    }
    if (thread.joinable())
        thread.join();
}

bool load_engine(nvinfer1::IRuntime *runtime, const std::string &engine_path,
                 ModelEngine *me, int in_w, int in_h, int in_ch, int rank5_t,
                 std::string *err)
{
    std::vector<char> blob;
    if (!read_file_bytes(engine_path, &blob)) {
        *err = "cannot open engine: " + engine_path;
        return false;
    }
    me->engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!me->engine) {
        *err = "engine deserialization failed: " + engine_path;
        return false;
    }
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
    me->exec.reset(me->engine->createExecutionContext());
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
    return true;
}

} // namespace ajicore
