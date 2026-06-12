/*
 * aji_dml.cpp — DirectML backend for the aji shim (Windows only).
 *
 * Mirrors aji_trt.cpp's conf-mode semantics (chain selection, resize
 * steps, stats text) with ONNX Runtime's DirectML EP replacing TensorRT
 * and D3D12 compute shaders (dml_shaders.h, ports of kernels.cu)
 * replacing the CUDA kernels. There is no engine cache: sessions
 * compile in well under a second at configure time.
 *
 * Frames arrive as shared D3D11 textures (aji_frame.plane[0] =
 * ID3D11Texture2D*, plane[1] = subresource index). Everything runs on
 * one D3D12 DIRECT queue created on the same adapter as the caller's
 * D3D11 device: input-plane copies + pre kernels, then ORT's submitted
 * work (the session shares our queue via the _DML1 EP append), then
 * post kernels + output-plane copies. Cross-API ordering uses one
 * shared fence: D3D11 signals after the caller's copy into the input
 * texture, our queue waits; v1 completes synchronously (CPU wait at the
 * end of infer), which also keeps command-allocator reuse trivial.
 *
 * Session recipe per the DML EP requirements and vs-mlrt's vsort:
 * ORT_SEQUENTIAL + DisableMemPattern, DirectML.dll loaded from our own
 * directory *before* onnxruntime.dll, first inference run twice.
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <DirectML.h>

#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "aji.h"
#include "aji_conf.h"
#include "dml_shaders.h"
#include "resample.h"

using Microsoft::WRL::ComPtr;

/* ---------------- module loading (DirectML before onnxruntime) -------- */

namespace {

const OrtApi *g_ort = nullptr;
const OrtDmlApi *g_dml_api = nullptr;
HRESULT(WINAPI *g_dml_create_device)(ID3D12Device *, DML_CREATE_DEVICE_FLAGS,
                                     REFIID, void **) = nullptr;

std::string own_dir(void)
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

// Loads DirectML.dll then onnxruntime.dll from our own directory (the
// order matters: onnxruntime delay-loads DirectML, and preloading ours
// keeps the System32 copy out of the picture). Idempotent.
bool load_ort(std::string *err)
{
    if (g_ort && g_dml_api && g_dml_create_device)
        return true;
    const std::string dir = own_dir();
    const DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    HMODULE dml = LoadLibraryExA((dir + "\\DirectML.dll").c_str(), NULL, flags);
    if (!dml) {
        *err = "DirectML.dll not found next to aji_dml.dll (error " +
               std::to_string(GetLastError()) + ")";
        return false;
    }
    HMODULE ort = LoadLibraryExA((dir + "\\onnxruntime.dll").c_str(), NULL,
                                 flags);
    if (!ort) {
        *err = "onnxruntime.dll not found next to aji_dml.dll (error " +
               std::to_string(GetLastError()) + ")";
        return false;
    }
    auto get_base = (const OrtApiBase *(ORT_API_CALL *)(void))
        GetProcAddress(ort, "OrtGetApiBase");
    if (!get_base) {
        *err = "onnxruntime.dll has no OrtGetApiBase";
        return false;
    }
    g_ort = get_base()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        *err = "onnxruntime.dll is older than the ORT headers we built "
               "against (API " + std::to_string(ORT_API_VERSION) + ")";
        return false;
    }
    if (g_ort->GetExecutionProviderApi("DML", ORT_API_VERSION,
                                       (const void **)&g_dml_api)) {
        *err = "this onnxruntime.dll has no DirectML execution provider";
        return false;
    }
    *(FARPROC *)&g_dml_create_device = GetProcAddress(dml, "DMLCreateDevice");
    if (!g_dml_create_device) {
        *err = "DirectML.dll has no DMLCreateDevice";
        return false;
    }
    return true;
}

std::wstring widen(const std::string &s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, NULL, 0);
    std::wstring w(n ? n - 1 : 0, L'\0');
    if (n)
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

/* ---------------- D3D12 plumbing ---------------- */

constexpr uint32_t align256(uint32_t v) { return (v + 255u) & ~255u; }

// Root constants: must match cbuffer CB in dml_shaders.h (16 DWORDs).
struct KernArgs {
    int32_t di[4];
    int32_t dj[4];
    float kr, kb, yoff, yscale;
    float coff, cscale, qdiv, qmax;
};

enum Kern {
    K_UV_H = 0, K_V_F32, K_PRE_COMBINE, K_POST_MATRIX,
    K_H_F32, K_UV_V_STORE, K_RS_H, K_RS_V,
    K_FILL, K_WINDOW, K_RIFE_CONSTS, K_SCD_PARTIAL, K_SCD_FINAL,
};
const char *const KERN_ENTRY[] = {
    "cs_uv_h", "cs_v_f32", "cs_pre_combine", "cs_post_matrix",
    "cs_h_f32", "cs_uv_v_store", "cs_rs_h", "cs_rs_v",
    "cs_fill", "cs_window", "cs_rife_consts", "cs_scd_partial",
    "cs_scd_final",
};

struct DmlPass {
    int taps = 0, src = 0, dst = 0;
    ComPtr<ID3D12Resource> starts, wt;   // upload heap, GENERIC_READ
};

struct DmlPlan {
    enum Kind { PRE, POST, RESIZE } kind = PRE;
    int format = 0;
    int w = 0, h = 0;       // pre/post luma dims; resize: src dims
    int dw = 0, dh = 0;     // resize: dst dims
    DmlPass ph, pv;
    ComPtr<ID3D12Resource> tmp0, tmp1;   // f32 intermediates
};

struct DmlModel {
    OrtSession *session = nullptr;
    std::string in_name, out_name;
    bool fp16 = true;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    OrtIoBinding *binding = nullptr;
    OrtValue *in_val = nullptr, *out_val = nullptr;
};

struct Step {
    enum Kind { RESIZE, MODEL } kind;
    int out_w, out_h;
    int model_idx = -1;                  // MODEL
    std::unique_ptr<DmlPlan> plan;       // RESIZE
    int in_buf = 0;                      // chain buffer ping-pong index
    bool elem16_in = true, elem16_out = true;
};

} // namespace

struct aji_ctx {
    // logging / errors
    aji_log_fn log_fn = nullptr;
    void *log_opaque = nullptr;
    char errbuf[512] = {0};

    // conf mode state (mirrors aji_trt)
    std::string conf_path, model_dir, rife_model_dir;
    AjiConf conf;
    int slot = 1;
    bool active = false;
    int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    std::string current_log;
    std::vector<std::string> log_info, log_steps;

    // D3D11 side (caller's device)
    ComPtr<ID3D11Device> dev11;
    ComPtr<ID3D11DeviceContext> ctx11;
    ComPtr<ID3D11DeviceContext4> ctx11_4;
    ComPtr<ID3D11Fence> fence11;

    // D3D12 side
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> queue;
    // Command allocators/lists are pooled: every recording segment gets
    // its own pair, because an allocator must not be reset while its
    // prior work is still executing. Frames are pipelined (no CPU wait
    // per infer), so entries retire by fence: a recorded entry sits in
    // cl_busy tagged with the done_fence value that covers it (UINT64_MAX
    // until the frame's end-of-frame signal) and returns to cl_free once
    // that value completes.
    struct ClEntry {
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> cl;
    };
    ComPtr<ID3D12CommandAllocator> alloc;       // current segment's pair
    ComPtr<ID3D12GraphicsCommandList> cl;
    std::vector<ClEntry> cl_free;
    std::vector<std::pair<ClEntry, uint64_t>> cl_busy;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = NULL;
    uint64_t fence_val = 0;
    // End-of-frame markers (aji_flush/done/wait tickets + cl retirement)
    // live on a queue-private fence, deliberately separate from `fence`:
    // that one is also signaled from the D3D11 side (input handoffs), and
    // a fence's completed value is its maximum, so a later D3D11 signal
    // would falsely complete an earlier end-of-frame marker.
    ComPtr<ID3D12Fence> done_fence;
    uint64_t done_val = 0;
    ComPtr<ID3D12RootSignature> rootsig;
    std::map<uint32_t, ComPtr<ID3D12PipelineState>> psos;
    ComPtr<IDMLDevice> dml_dev;

    // ORT
    OrtEnv *env = nullptr;
    OrtMemoryInfo *mi_dml = nullptr;

    // chain
    std::vector<DmlModel> models;
    std::vector<Step> steps;
    ComPtr<ID3D12Resource> buf[2];
    void *buf_alloc[2] = {nullptr, nullptr};  // OrtDmlApi GPU allocations
    size_t buf_bytes = 0;
    bool first_elem16 = true;   // element type cs_pre_combine writes
    bool last_elem16 = true;    // element type cs_post_matrix reads

    // lazy per-frame-format state
    std::unique_ptr<DmlPlan> pre_plan, post_plan;
    int pre_key[4] = {0}, post_key[4] = {0};
    ComPtr<ID3D12Resource> in_y, in_uv, out_y, out_uv;  // plane staging
    uint32_t in_pitch = 0, out_pitch = 0;
    int stage_key[3] = {0};     // format, in dims valid for staging

    // shared-texture cache (cleared on configure)
    std::map<void *, ComPtr<ID3D12Resource>> shared;

    // RIFE: interpolation between already-upscaled frames, mirroring
    // aji_trt (centered mod-64 pad, bilinear 709 conversions, vsmlrt v1
    // 11-channel fp32 input). Format-dependent staging builds lazily on
    // the first aji_infer_rife.
    struct {
        bool enabled = false;
        int num = 1, den = 1;
        double scd_threshold = 0.150;
        DmlModel model;
        int w = 0, h = 0;            // unpadded = chain output dims
        int pw = 0, ph = 0;          // padded to mod-64
        int pad_l = 0, pad_t = 0;    // centered, rounded down to even
        int format = 0;              // staging/plans built for this format
        std::unique_ptr<DmlPlan> pre, post;   // bilinear, 709
        // tight-pitch (pw*bpp) padded YUV planes
        ComPtr<ID3D12Resource> pad_y[3], pad_uv[3];   // a, b, out
        // 256-pitch texture staging at w x h
        ComPtr<ID3D12Resource> st_y[3], st_uv[3];
        uint32_t st_pitch = 0;
        ComPtr<ID3D12Resource> in_tensor, out_tensor; // 11 / 3 planes
        void *in_alloc = nullptr, *out_alloc = nullptr;
        ComPtr<ID3D12Resource> scd_part, scd_sum, scd_read;
        int scd_groups = 0;
    } rife;

    void set_error(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(errbuf, sizeof(errbuf), fmt, ap);
        va_end(ap);
        if (log_fn)
            log_fn(log_opaque, 1, errbuf);
    }
    void verbose(const char *fmt, ...) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (log_fn)
            log_fn(log_opaque, 3, buf);
    }
    bool ort_ck(OrtStatus *st, const char *what) {
        if (!st)
            return true;
        set_error("%s: %s", what, g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return false;
    }
};

namespace {

/* ---------------- small helpers ---------------- */

int round_even(double v)
{
    int i = (int)(v + 0.5);
    return i & ~1;
}

// scale_to_1080: fit into (box_w, box_h) preserving aspect.
void fit_box(int w, int h, double box_w, double box_h, int *ow, int *oh)
{
    if ((double)w / h > 16.0 / 9.0) {
        *ow = round_even(box_w);
        *oh = round_even(box_w * h / w);
    } else {
        *ow = round_even(box_h * w / h);
        *oh = round_even(box_h);
    }
}

std::string fmt_num(double v)
{
    if (v >= 1e300)
        return "inf";
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

void finalize_log(aji_ctx *c)
{
    std::string out;
    for (auto &l : c->log_info)
        out += l + "\n";
    out += "\n";
    for (size_t i = 0; i < c->log_steps.size(); i++)
        out += std::to_string(i + 1) + ". " + c->log_steps[i] + "\n";
    c->current_log = out;
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device *dev, uint64_t bytes,
                                   bool upload)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes ? bytes : 4;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = upload ? D3D12_RESOURCE_FLAG_NONE
                      : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> res;
    if (FAILED(dev->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd,
            upload ? D3D12_RESOURCE_STATE_GENERIC_READ
                   : D3D12_RESOURCE_STATE_COMMON,
            NULL, IID_PPV_ARGS(&res))))
        return nullptr;
    return res;
}

bool build_pass(aji_ctx *c, DmlPass *p, int src, int dst, double shift,
                int filter)
{
    aji_resample::weights w = aji_resample::compute(src, dst, shift, filter);
    p->taps = w.taps;
    p->src = src;
    p->dst = dst;
    p->starts = make_buffer(c->dev.Get(), (uint64_t)dst * 4, true);
    p->wt = make_buffer(c->dev.Get(), w.wt.size() * 4, true);
    if (!p->starts || !p->wt)
        return false;
    void *m = nullptr;
    D3D12_RANGE none = {0, 0};
    if (FAILED(p->starts->Map(0, &none, &m)))
        return false;
    memcpy(m, w.start.data(), (size_t)dst * 4);
    p->starts->Unmap(0, NULL);
    if (FAILED(p->wt->Map(0, &none, &m)))
        return false;
    memcpy(m, w.wt.data(), w.wt.size() * 4);
    p->wt->Unmap(0, NULL);
    return true;
}

std::unique_ptr<DmlPlan> pre_plan_create(aji_ctx *c, int format, int w, int h,
                                         int siting, int filter)
{
    auto p = std::make_unique<DmlPlan>();
    p->kind = DmlPlan::PRE;
    p->format = format;
    p->w = w;
    p->h = h;
    const int cw = w >> 1, ch = h >> 1;
    double sx, sy;
    aji_resample::chroma_shifts(siting, true, &sx, &sy);
    p->tmp0 = make_buffer(c->dev.Get(), (uint64_t)2 * w * ch * 4, false);
    p->tmp1 = make_buffer(c->dev.Get(), (uint64_t)2 * w * h * 4, false);
    if (!build_pass(c, &p->ph, cw, w, sx, filter) ||
        !build_pass(c, &p->pv, ch, h, sy, filter) || !p->tmp0 || !p->tmp1)
        return nullptr;
    return p;
}

std::unique_ptr<DmlPlan> post_plan_create(aji_ctx *c, int format, int w, int h,
                                          int siting, int filter)
{
    auto p = std::make_unique<DmlPlan>();
    p->kind = DmlPlan::POST;
    p->format = format;
    p->w = w;
    p->h = h;
    const int cw = w >> 1, ch = h >> 1;
    double sx, sy;
    aji_resample::chroma_shifts(siting, false, &sx, &sy);
    p->tmp0 = make_buffer(c->dev.Get(), (uint64_t)2 * w * h * 4, false);
    p->tmp1 = make_buffer(c->dev.Get(), (uint64_t)2 * cw * h * 4, false);
    if (!build_pass(c, &p->ph, w, cw, sx, filter) ||
        !build_pass(c, &p->pv, h, ch, sy, filter) || !p->tmp0 || !p->tmp1)
        return nullptr;
    return p;
}

std::unique_ptr<DmlPlan> resize_plan_create(aji_ctx *c, int sw, int sh,
                                            int dw, int dh)
{
    auto p = std::make_unique<DmlPlan>();
    p->kind = DmlPlan::RESIZE;
    p->w = sw;
    p->h = sh;
    p->dw = dw;
    p->dh = dh;
    p->tmp0 = make_buffer(c->dev.Get(), (uint64_t)3 * dw * sh * 4, false);
    if (!build_pass(c, &p->ph, sw, dw, 0.0, AJI_FILTER_SPLINE36) ||
        !build_pass(c, &p->pv, sh, dh, 0.0, AJI_FILTER_SPLINE36) || !p->tmp0)
        return nullptr;
    return p;
}

/* PSO variant key: kern | t16<<8 | io16<<9 */
ID3D12PipelineState *get_pso(aji_ctx *c, Kern k, bool t16, bool io16)
{
    const uint32_t key = (uint32_t)k | (t16 ? 0x100u : 0) | (io16 ? 0x200u : 0);
    auto it = c->psos.find(key);
    if (it != c->psos.end())
        return it->second.Get();

    char t16s[2] = {char('0' + t16), 0}, io16s[2] = {char('0' + io16), 0};
    // dword-granular stores: u8 packs 4 luma / 2 chroma px, u16 2 / 1
    const char *tpx = "1";
    if (k == K_POST_MATRIX)
        tpx = t16 ? "2" : "4";
    else if (k == K_UV_V_STORE)
        tpx = t16 ? "1" : "2";
    const D3D_SHADER_MACRO defs[] = {
        {"T16", t16s}, {"IO16", io16s}, {"TPX", tpx}, {NULL, NULL}};

    ComPtr<ID3DBlob> blob, errs;
    HRESULT hr = D3DCompile(DML_HLSL, sizeof(DML_HLSL) - 1, "aji_dml",
                            defs, NULL, KERN_ENTRY[k], "cs_5_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3 |
                                D3DCOMPILE_IEEE_STRICTNESS,
                            0, &blob, &errs);
    if (FAILED(hr)) {
        c->set_error("HLSL compile %s failed: %s", KERN_ENTRY[k],
                     errs ? (const char *)errs->GetBufferPointer() : "?");
        return nullptr;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = c->rootsig.Get();
    pd.CS = {blob->GetBufferPointer(), blob->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(c->dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)))) {
        c->set_error("compute PSO %s creation failed", KERN_ENTRY[k]);
        return nullptr;
    }
    c->psos[key] = pso;
    return c->psos[key].Get();
}

bool make_rootsig(aji_ctx *c)
{
    D3D12_ROOT_PARAMETER prm[6] = {};
    prm[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    prm[0].Constants = {0, 0, 16};      // b0, space0, 16 DWORDs
    for (int i = 0; i < 2; i++) {       // t0 starts, t1 weights
        prm[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        prm[1 + i].Descriptor = {(UINT)i, 0};
    }
    for (int i = 0; i < 3; i++) {       // u0..u2 data
        prm[3 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        prm[3 + i].Descriptor = {(UINT)i, 0};
    }
    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 6;
    rs.pParameters = prm;
    ComPtr<ID3DBlob> blob, errs;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &blob, &errs))) {
        c->set_error("root signature serialize failed: %s",
                     errs ? (const char *)errs->GetBufferPointer() : "?");
        return false;
    }
    if (FAILED(c->dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                           blob->GetBufferSize(),
                                           IID_PPV_ARGS(&c->rootsig)))) {
        c->set_error("root signature creation failed");
        return false;
    }
    return true;
}

/* ---------------- command recording ---------------- */

struct Recorder {
    aji_ctx *c;
    bool open = false;

    bool begin() {
        if (open)
            return true;
        // retire recordings whose covering end-of-frame marker completed
        const uint64_t done = c->done_fence->GetCompletedValue();
        for (size_t i = 0; i < c->cl_busy.size();) {
            if (c->cl_busy[i].second <= done) {
                c->cl_free.push_back(std::move(c->cl_busy[i].first));
                c->cl_busy.erase(c->cl_busy.begin() + i);
            } else {
                i++;
            }
        }
        if (c->cl_free.empty()) {
            // pool grows to the live working set: segments/frame x
            // pipelined frames
            aji_ctx::ClEntry e;
            if (FAILED(c->dev->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&e.alloc))) ||
                FAILED(c->dev->CreateCommandList(
                    0, D3D12_COMMAND_LIST_TYPE_DIRECT, e.alloc.Get(), NULL,
                    IID_PPV_ARGS(&e.cl))) ||
                FAILED(e.cl->Close())) {
                c->set_error("command allocator/list creation failed");
                return false;
            }
            c->cl_free.push_back(std::move(e));
        }
        c->alloc = c->cl_free.back().alloc;
        c->cl = c->cl_free.back().cl;
        c->cl_busy.emplace_back(std::move(c->cl_free.back()), UINT64_MAX);
        c->cl_free.pop_back();
        if (FAILED(c->alloc->Reset()) ||
            FAILED(c->cl->Reset(c->alloc.Get(), NULL))) {
            c->set_error("command list reset failed");
            return false;
        }
        c->cl->SetComputeRootSignature(c->rootsig.Get());
        open = true;
        return true;
    }
    bool exec() {
        if (!open)
            return true;
        if (FAILED(c->cl->Close())) {
            c->set_error("command list close failed");
            return false;
        }
        ID3D12CommandList *lists[] = {c->cl.Get()};
        c->queue->ExecuteCommandLists(1, lists);
        open = false;
        return true;
    }
    void uav_barrier() {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        c->cl->ResourceBarrier(1, &b);
    }
    void transition(ID3D12Resource *r, D3D12_RESOURCE_STATES from,
                    D3D12_RESOURCE_STATES to, UINT sub) {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {r, sub, from, to};
        c->cl->ResourceBarrier(1, &b);
    }
    bool dispatch(Kern k, bool t16, bool io16, const KernArgs &args,
                  ID3D12Resource *s0, ID3D12Resource *s1, ID3D12Resource *u0,
                  ID3D12Resource *u1, ID3D12Resource *u2,
                  int gx, int gy, int gz) {
        ID3D12PipelineState *pso = get_pso(c, k, t16, io16);
        if (!pso)
            return false;
        c->cl->SetPipelineState(pso);
        c->cl->SetComputeRoot32BitConstants(0, 16, &args, 0);
        if (s0) c->cl->SetComputeRootShaderResourceView(1, s0->GetGPUVirtualAddress());
        if (s1) c->cl->SetComputeRootShaderResourceView(2, s1->GetGPUVirtualAddress());
        if (u0) c->cl->SetComputeRootUnorderedAccessView(3, u0->GetGPUVirtualAddress());
        if (u1) c->cl->SetComputeRootUnorderedAccessView(4, u1->GetGPUVirtualAddress());
        if (u2) c->cl->SetComputeRootUnorderedAccessView(5, u2->GetGPUVirtualAddress());
        c->cl->Dispatch((gx + 31) / 32, (gy + 7) / 8, gz);
        return true;
    }
};

// Queue an end-of-frame marker on the done fence; its value is the ticket
// aji_flush/done/wait operate on, and it retires every command recording
// made since the previous marker.
uint64_t signal_done(aji_ctx *c)
{
    if (!c->queue || !c->done_fence)
        return 0;
    c->done_val++;
    c->queue->Signal(c->done_fence.Get(), c->done_val);
    for (auto &b : c->cl_busy) {
        if (b.second == UINT64_MAX)
            b.second = c->done_val;
    }
    return c->done_val;
}

bool wait_done_val(aji_ctx *c, uint64_t v, DWORD timeout_ms)
{
    if (!c->done_fence || c->done_fence->GetCompletedValue() >= v)
        return true;
    c->done_fence->SetEventOnCompletion(v, c->fence_event);
    return WaitForSingleObject(c->fence_event, timeout_ms) == WAIT_OBJECT_0;
}

// CPU-wait the queue idle (teardown, warmups whose temp resources die).
bool drain_queue(aji_ctx *c, DWORD timeout_ms)
{
    return wait_done_val(c, signal_done(c), timeout_ms);
}

// Tags this frame's recordings (and unblocks aji_flush tickets) on every
// exit from a pipelined entry point, error paths included.
struct DoneGuard {
    aji_ctx *c;
    ~DoneGuard() { signal_done(c); }
};

KernArgs csp_args(const aji_csp &csp, float qdiv, float qmax)
{
    KernArgs a = {};
    a.kr = csp.kr;
    a.kb = csp.kb;
    a.yoff = csp.yoff;
    a.yscale = csp.yscale;
    a.coff = csp.coff;
    a.cscale = csp.cscale;
    a.qdiv = qdiv;
    a.qmax = qmax;
    return a;
}

// pre: y/uv plane buffers -> tensor in dst at dst_off elements
// (cs_uv_h, cs_v_f32, cs_pre_combine)
bool record_pre(aji_ctx *c, Recorder &r, DmlPlan *p, const aji_csp &csp,
                ID3D12Resource *y_buf, ID3D12Resource *uv_buf, uint32_t pitch,
                ID3D12Resource *dst, int dst_off, bool io16)
{
    const bool t16 = p->format == AJI_FMT_P010;
    const int w = p->w, h = p->h, ch = h >> 1;
    KernArgs a = csp_args(csp, 0, 0);

    a.di[0] = p->ph.src; a.di[1] = p->ph.dst; a.di[2] = p->ph.taps;
    a.di[3] = ch; a.dj[0] = (int)pitch;
    if (!r.dispatch(K_UV_H, t16, io16, a, p->ph.starts.Get(), p->ph.wt.Get(),
                    uv_buf, p->tmp0.Get(), NULL, w, ch, 1))
        return false;
    r.uav_barrier();

    a.di[0] = w; a.di[1] = ch; a.di[2] = p->pv.taps; a.di[3] = h;
    if (!r.dispatch(K_V_F32, t16, io16, a, p->pv.starts.Get(), p->pv.wt.Get(),
                    p->tmp0.Get(), p->tmp1.Get(), NULL, w, h, 2))
        return false;
    r.uav_barrier();

    a.di[0] = w; a.di[1] = h; a.dj[0] = (int)pitch; a.dj[1] = dst_off;
    if (!r.dispatch(K_PRE_COMBINE, t16, io16, a, NULL, NULL, y_buf,
                    p->tmp1.Get(), dst, w / 2, h, 1))
        return false;
    r.uav_barrier();
    return true;
}

// post: tensor src -> y/uv plane buffers
bool record_post(aji_ctx *c, Recorder &r, DmlPlan *p, const aji_csp &csp,
                 ID3D12Resource *src, ID3D12Resource *y_buf,
                 ID3D12Resource *uv_buf, uint32_t pitch, bool io16)
{
    const bool t16 = p->format == AJI_FMT_P010;
    const int w = p->w, h = p->h, cw = w >> 1, ch = h >> 1;
    const float qdiv = t16 ? 64.0f : 1.0f;
    const float qmax = t16 ? 1023.0f : 255.0f;
    KernArgs a = csp_args(csp, qdiv, qmax);
    const int tpx_y = t16 ? 2 : 4, tpx_uv = t16 ? 1 : 2;

    a.di[0] = w; a.di[1] = h; a.dj[0] = (int)pitch;
    if (!r.dispatch(K_POST_MATRIX, t16, io16, a, NULL, NULL, src,
                    y_buf, p->tmp0.Get(), (w + tpx_y - 1) / tpx_y, h, 1))
        return false;
    r.uav_barrier();

    a.di[0] = w; a.di[1] = cw; a.di[2] = p->ph.taps; a.di[3] = h;
    if (!r.dispatch(K_H_F32, t16, io16, a, p->ph.starts.Get(), p->ph.wt.Get(),
                    p->tmp0.Get(), p->tmp1.Get(), NULL, cw, h, 2))
        return false;
    r.uav_barrier();

    a.di[0] = cw; a.di[1] = h; a.di[2] = p->pv.taps; a.di[3] = ch;
    a.dj[0] = (int)pitch;
    if (!r.dispatch(K_UV_V_STORE, t16, io16, a, p->pv.starts.Get(),
                    p->pv.wt.Get(), p->tmp1.Get(), uv_buf, NULL,
                    (cw + tpx_uv - 1) / tpx_uv, ch, 1))
        return false;
    r.uav_barrier();
    return true;
}

// dword-fill a range of a buffer
bool record_fill(aji_ctx *c, Recorder &r, ID3D12Resource *dst,
                 uint32_t start_dword, uint32_t dwords, uint32_t value)
{
    KernArgs a = {};
    a.di[0] = (int)dwords;
    a.dj[0] = (int)start_dword;
    a.dj[1] = (int)value;
    if (!r.dispatch(K_FILL, false, false, a, NULL, NULL, dst, NULL, NULL,
                    32 * ((dwords + 255) / 256) /* gx*32 = groups*256 thr */,
                    1, 1))
        return false;
    return true;
}

// copy a w_bytes x rows window between pitched buffers
bool record_window(aji_ctx *c, Recorder &r, ID3D12Resource *src,
                   uint32_t src_off, uint32_t src_pitch, ID3D12Resource *dst,
                   uint32_t dst_off, uint32_t dst_pitch, uint32_t w_bytes,
                   uint32_t rows)
{
    KernArgs a = {};
    a.di[0] = (int)w_bytes;
    a.di[1] = (int)rows;
    a.di[2] = (int)(((dst_off & 3u) + w_bytes + 3u) / 4u);
    a.dj[0] = (int)src_off;
    a.dj[1] = (int)src_pitch;
    a.dj[2] = (int)dst_off;
    a.dj[3] = (int)dst_pitch;
    return r.dispatch(K_WINDOW, false, false, a, NULL, NULL, src, dst, NULL,
                      a.di[2], rows, 1);
}

bool record_resize(aji_ctx *c, Recorder &r, DmlPlan *p, ID3D12Resource *src,
                   ID3D12Resource *dst, bool in16, bool out16)
{
    KernArgs a = {};
    a.di[0] = p->ph.src; a.di[1] = p->ph.dst; a.di[2] = p->ph.taps;
    a.di[3] = p->h;
    if (!r.dispatch(K_RS_H, false, in16, a, p->ph.starts.Get(), p->ph.wt.Get(),
                    src, p->tmp0.Get(), NULL, p->dw, p->h, 3))
        return false;
    r.uav_barrier();
    a.di[0] = p->dw; a.di[1] = p->h; a.di[2] = p->pv.taps; a.di[3] = p->dh;
    if (!r.dispatch(K_RS_V, false, out16, a, p->pv.starts.Get(),
                    p->pv.wt.Get(), p->tmp0.Get(), dst, NULL, p->dw / 2,
                    p->dh, 3))
        return false;
    r.uav_barrier();
    return true;
}

/* ---------------- shared textures + plane copies ---------------- */

void diagnose_device(aji_ctx *c, const char *where);

ID3D12Resource *open_shared(aji_ctx *c, void *tex11)
{
    auto it = c->shared.find(tex11);
    if (it != c->shared.end())
        return it->second.Get();
    ComPtr<IDXGIResource1> dxgi;
    if (FAILED(((IUnknown *)tex11)
                   ->QueryInterface(IID_PPV_ARGS(&dxgi)))) {
        c->set_error("frame texture is not a DXGI resource");
        return nullptr;
    }
    HANDLE h = NULL;
    if (FAILED(dxgi->CreateSharedHandle(
            NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
            NULL, &h))) {
        c->set_error("CreateSharedHandle failed (frame textures must be "
                     "created with SHARED | SHARED_NTHANDLE)");
        return nullptr;
    }
    ComPtr<ID3D12Resource> res;
    HRESULT hr = c->dev->OpenSharedHandle(h, IID_PPV_ARGS(&res));
    CloseHandle(h);
    if (FAILED(hr)) {
        c->set_error("D3D12 OpenSharedHandle failed (0x%08x)", (unsigned)hr);
        diagnose_device(c, "OpenSharedHandle");
        return nullptr;
    }
    c->shared[tex11] = res;
    return c->shared[tex11].Get();
}

DXGI_FORMAT plane_fmt(int format, int plane)
{
    if (format == AJI_FMT_P010)
        return plane ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R16_UNORM;
    return plane ? DXGI_FORMAT_R8G8_UNORM : DXGI_FORMAT_R8_UNORM;
}

// Transition both planes of one array slice. Frames can be slices of a
// single pool array texture (a/b/out of aji_infer_rife in particular),
// so whole-resource barriers would double-transition shared resources.
void transition_planes(aji_ctx *c, ID3D12Resource *tex, UINT subres,
                       D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_DESC td = tex->GetDesc();
    D3D12_RESOURCE_BARRIER bs[2] = {};
    for (int p = 0; p < 2; p++) {
        bs[p].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bs[p].Transition.pResource = tex;
        bs[p].Transition.Subresource = subres + (UINT)p * td.DepthOrArraySize;
        bs[p].Transition.StateBefore = from;
        bs[p].Transition.StateAfter = to;
    }
    c->cl->ResourceBarrier(2, bs);
}

// On any failure, distinguish a removed device (and say why) from a
// plain error; with the debug layer on (AJI_DML_DEBUG=1) also drain the
// info queue so validation messages reach the player log.
void diagnose_device(aji_ctx *c, const char *where)
{
    if (!c->dev)
        return;
    HRESULT rr = c->dev->GetDeviceRemovedReason();
    if (FAILED(rr)) {
        c->set_error("device removed (reason 0x%08x) detected at %s",
                     (unsigned)rr, where);
        ComPtr<ID3D12DeviceRemovedExtendedData> dred;
        if (SUCCEEDED(c->dev.As(&dred))) {
            D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc = {};
            if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&bc))) {
                for (auto *n = bc.pHeadAutoBreadcrumbNode; n;
                     n = n->pNext) {
                    UINT done = n->pLastBreadcrumbValue
                                    ? *n->pLastBreadcrumbValue : 0;
                    if (done == n->BreadcrumbCount)
                        continue;   // this command list completed fine
                    c->verbose("dred: CL stopped at op %u/%u (last op %d)",
                               done, n->BreadcrumbCount,
                               done < n->BreadcrumbCount
                                   ? (int)n->pCommandHistory[done] : -1);
                }
            }
            D3D12_DRED_PAGE_FAULT_OUTPUT pf = {};
            if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&pf)) &&
                pf.PageFaultVA)
                c->verbose("dred: page fault at VA 0x%llx",
                           (unsigned long long)pf.PageFaultVA);
        }
    }
    ComPtr<ID3D12InfoQueue> iq;
    if (SUCCEEDED(c->dev.As(&iq))) {
        UINT64 n = iq->GetNumStoredMessages();
        for (UINT64 i = 0; i < n; i++) {
            SIZE_T len = 0;
            iq->GetMessage(i, NULL, &len);
            if (!len)
                continue;
            std::vector<char> buf(len);
            D3D12_MESSAGE *m = (D3D12_MESSAGE *)buf.data();
            if (SUCCEEDED(iq->GetMessage(i, m, &len)) && m->pDescription)
                c->verbose("d3d12 debug: %s", m->pDescription);
        }
        iq->ClearStoredMessages();
    }
}

ComPtr<ID3D12Resource> make_readback(ID3D12Device *dev, uint64_t bytes)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> res;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                            D3D12_RESOURCE_STATE_COPY_DEST,
                                            NULL, IID_PPV_ARGS(&res))))
        return nullptr;
    return res;
}

// copy one plane between a shared texture and a staging buffer
void copy_plane(aji_ctx *c, bool to_buffer, ID3D12Resource *tex, UINT subres,
                int plane, int format, int w, int h, ID3D12Resource *buf,
                uint32_t pitch)
{
    D3D12_RESOURCE_DESC td = tex->GetDesc();
    D3D12_TEXTURE_COPY_LOCATION t = {};
    t.pResource = tex;
    t.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    t.SubresourceIndex = subres + (UINT)plane * td.DepthOrArraySize;

    D3D12_TEXTURE_COPY_LOCATION b = {};
    b.pResource = buf;
    b.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    b.PlacedFootprint.Footprint.Format = plane_fmt(format, plane);
    b.PlacedFootprint.Footprint.Width = plane ? w / 2 : w;
    b.PlacedFootprint.Footprint.Height = plane ? h / 2 : h;
    b.PlacedFootprint.Footprint.Depth = 1;
    b.PlacedFootprint.Footprint.RowPitch = pitch;

    if (to_buffer)
        c->cl->CopyTextureRegion(&b, 0, 0, 0, &t, NULL);
    else
        c->cl->CopyTextureRegion(&t, 0, 0, 0, &b, NULL);
}

/* ---------------- ORT session handling ---------------- */

void model_release(DmlModel *m)
{
    if (m->in_val)
        g_ort->ReleaseValue(m->in_val);
    if (m->out_val)
        g_ort->ReleaseValue(m->out_val);
    if (m->binding)
        g_ort->ReleaseIoBinding(m->binding);
    if (m->session)
        g_ort->ReleaseSession(m->session);
    m->in_val = m->out_val = nullptr;
    m->binding = nullptr;
    m->session = nullptr;
}

void rife_teardown(aji_ctx *c);

void chain_teardown(aji_ctx *c)
{
    // GPU must be idle before buffers/sessions go away (in-flight
    // pipelined frames included)
    drain_queue(c, 10000);
    // teardown waited the queue idle just above
    for (auto &b : c->cl_busy)
        c->cl_free.push_back(std::move(b.first));
    c->cl_busy.clear();
    for (auto &m : c->models)
        model_release(&m);
    c->models.clear();
    c->steps.clear();
    for (int i = 0; i < 2; i++) {
        if (c->buf_alloc[i]) {
            g_dml_api->FreeGPUAllocation(c->buf_alloc[i]);
            c->buf_alloc[i] = nullptr;
        }
        c->buf[i].Reset();
    }
    c->buf_bytes = 0;
    c->pre_plan.reset();
    c->post_plan.reset();
    memset(c->pre_key, 0, sizeof(c->pre_key));
    memset(c->post_key, 0, sizeof(c->post_key));
    c->in_y.Reset();
    c->in_uv.Reset();
    c->out_y.Reset();
    c->out_uv.Reset();
    memset(c->stage_key, 0, sizeof(c->stage_key));
    c->shared.clear();
    rife_teardown(c);
}

// Create an ORT tensor over `bytes` of a D3D12 buffer (DML allocation).
OrtValue *make_tensor(aji_ctx *c, void *dml_alloc, size_t bytes, int n, int ch,
                      int h, int w, bool fp16)
{
    int64_t dims[4] = {n, ch, h, w};
    OrtValue *v = nullptr;
    if (!c->ort_ck(g_ort->CreateTensorWithDataAsOrtValue(
                       c->mi_dml, dml_alloc, bytes, dims, 4,
                       fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                            : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                       &v),
                   "CreateTensorWithDataAsOrtValue"))
        return nullptr;
    return v;
}

// Create the session + discover the model's output dims for (in_w, in_h).
// Runs the model twice on a temporary input (the DML EP builds its
// reusable command list during the first executions — vsort's "replay").
bool model_open(aji_ctx *c, DmlModel *m, const std::string &onnx_path,
                int in_w, int in_h, int in_ch)
{
    OrtSessionOptions *so = nullptr;
    if (!c->ort_ck(g_ort->CreateSessionOptions(&so), "CreateSessionOptions"))
        return false;
    g_ort->SetSessionExecutionMode(so, ORT_SEQUENTIAL);
    g_ort->DisableMemPattern(so);
    g_ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    OrtStatus *st = g_dml_api->SessionOptionsAppendExecutionProvider_DML1(
        so, c->dml_dev.Get(), c->queue.Get());
    if (st) {
        c->set_error("DML EP append failed: %s", g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        g_ort->ReleaseSessionOptions(so);
        return false;
    }

    std::wstring wpath = widen(onnx_path);
    st = g_ort->CreateSession(c->env, wpath.c_str(), so, &m->session);
    g_ort->ReleaseSessionOptions(so);
    if (st) {
        c->set_error("failed to load %s: %s", onnx_path.c_str(),
                     g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return false;
    }

    OrtAllocator *alloc = nullptr;
    g_ort->GetAllocatorWithDefaultOptions(&alloc);
    size_t n_in = 0, n_out = 0;
    g_ort->SessionGetInputCount(m->session, &n_in);
    g_ort->SessionGetOutputCount(m->session, &n_out);
    if (n_in != 1 || n_out != 1) {
        c->set_error("%s: expected 1 input/1 output, has %zu/%zu",
                     onnx_path.c_str(), n_in, n_out);
        return false;
    }
    char *name = nullptr;
    g_ort->SessionGetInputName(m->session, 0, alloc, &name);
    m->in_name = name;
    alloc->Free(alloc, name);
    g_ort->SessionGetOutputName(m->session, 0, alloc, &name);
    m->out_name = name;
    alloc->Free(alloc, name);

    OrtTypeInfo *ti = nullptr;
    const OrtTensorTypeAndShapeInfo *tti = nullptr;
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (!c->ort_ck(g_ort->SessionGetInputTypeInfo(m->session, 0, &ti),
                   "GetInputTypeInfo"))
        return false;
    g_ort->CastTypeInfoToTensorInfo(ti, &tti);
    g_ort->GetTensorElementType(tti, &et);
    g_ort->ReleaseTypeInfo(ti);
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
        et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        c->set_error("%s: unsupported input element type %d",
                     onnx_path.c_str(), (int)et);
        return false;
    }
    m->fp16 = et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    m->in_w = in_w;
    m->in_h = in_h;

    // discovery: bind a zeroed temp input, let ORT allocate the output
    const size_t elem = m->fp16 ? 2 : 4;
    const size_t in_bytes = (size_t)in_ch * in_w * in_h * elem;
    ComPtr<ID3D12Resource> tmp = make_buffer(c->dev.Get(), in_bytes, false);
    if (!tmp) {
        c->set_error("discovery buffer allocation failed");
        return false;
    }
    void *tmp_alloc = nullptr;
    if (!c->ort_ck(g_dml_api->CreateGPUAllocationFromD3DResource(tmp.Get(),
                                                                 &tmp_alloc),
                   "CreateGPUAllocationFromD3DResource"))
        return false;
    bool ok = false;
    OrtValue *in_val = nullptr;
    OrtValue **outs = nullptr;
    size_t n_outs = 0;
    do {
        in_val = make_tensor(c, tmp_alloc, in_bytes, 1, in_ch, in_h, in_w,
                             m->fp16);
        if (!in_val)
            break;
        if (!c->ort_ck(g_ort->CreateIoBinding(m->session, &m->binding),
                       "CreateIoBinding"))
            break;
        if (!c->ort_ck(g_ort->BindInput(m->binding, m->in_name.c_str(),
                                        in_val), "BindInput"))
            break;
        if (!c->ort_ck(g_ort->BindOutputToDevice(m->binding,
                                                 m->out_name.c_str(),
                                                 c->mi_dml),
                       "BindOutputToDevice"))
            break;
        // run twice (command-list warmup), then read the output shape
        if (!c->ort_ck(g_ort->RunWithBinding(m->session, NULL, m->binding),
                       "RunWithBinding (discovery)"))
            break;
        if (!c->ort_ck(g_ort->RunWithBinding(m->session, NULL, m->binding),
                       "RunWithBinding (warmup)"))
            break;
        if (!c->ort_ck(g_ort->GetBoundOutputValues(m->binding, alloc, &outs,
                                                   &n_outs),
                       "GetBoundOutputValues") || n_outs != 1)
            break;
        OrtTensorTypeAndShapeInfo *si = nullptr;
        if (!c->ort_ck(g_ort->GetTensorTypeAndShape(outs[0], &si),
                       "GetTensorTypeAndShape"))
            break;
        int64_t dims[4] = {0};
        size_t nd = 0;
        g_ort->GetDimensionsCount(si, &nd);
        if (nd == 4)
            g_ort->GetDimensions(si, dims, 4);
        g_ort->ReleaseTensorTypeAndShapeInfo(si);
        if (nd != 4 || dims[1] != 3 || dims[2] < 2 || dims[3] < 2) {
            c->set_error("%s: unexpected output shape", onnx_path.c_str());
            break;
        }
        m->out_w = (int)dims[3];
        m->out_h = (int)dims[2];
        ok = true;
    } while (0);

    if (outs) {
        for (size_t i = 0; i < n_outs; i++)
            g_ort->ReleaseValue(outs[i]);
        alloc->Free(alloc, outs);
    }
    if (in_val)
        g_ort->ReleaseValue(in_val);
    g_ort->ClearBoundInputs(m->binding);
    g_ort->ClearBoundOutputs(m->binding);
    g_dml_api->FreeGPUAllocation(tmp_alloc);
    // tmp buffer: the queue is idle here only after the runs complete;
    // RunWithBinding with device outputs returns after submission, so
    // wait before the temp buffer dies.
    drain_queue(c, 30000);
    return ok;
}

// Bind a model's IO to the chain buffers (called once buffers exist).
bool model_bind(aji_ctx *c, DmlModel *m, int in_buf)
{
    const size_t elem = m->fp16 ? 2 : 4;
    m->in_val = make_tensor(c, c->buf_alloc[in_buf],
                            (size_t)3 * m->in_w * m->in_h * elem, 1, 3,
                            m->in_h, m->in_w, m->fp16);
    m->out_val = make_tensor(c, c->buf_alloc[in_buf ^ 1],
                             (size_t)3 * m->out_w * m->out_h * elem, 1, 3,
                             m->out_h, m->out_w, m->fp16);
    if (!m->in_val || !m->out_val)
        return false;
    if (!c->ort_ck(g_ort->BindInput(m->binding, m->in_name.c_str(),
                                    m->in_val), "BindInput") ||
        !c->ort_ck(g_ort->BindOutput(m->binding, m->out_name.c_str(),
                                     m->out_val), "BindOutput"))
        return false;
    return true;
}

bool push_resize_step(aji_ctx *c, int *cw, int *ch, int nw, int nh)
{
    Step st;
    st.kind = Step::RESIZE;
    st.out_w = nw;
    st.out_h = nh;
    st.plan = resize_plan_create(c, *cw, *ch, nw, nh);
    if (!st.plan) {
        c->set_error("resize plan %dx%d -> %dx%d allocation failed",
                     *cw, *ch, nw, nh);
        return false;
    }
    c->steps.push_back(std::move(st));
    *cw = nw;
    *ch = nh;
    return true;
}

/* ---------------- RIFE ---------------- */

std::string rife_model_name(int code, bool ensemble)
{
    std::string s = std::to_string(code);
    if (s.size() < 2)
        return "";
    std::string dec = s.size() == 2 ? s.substr(1, 1) : s.substr(1, 2);
    std::string name = "rife_v" + s.substr(0, 1) + "." + dec;
    if (s.size() == 4 && s.back() == '1')
        name += "_lite";
    if (ensemble)
        name += "_ensemble";
    return name;
}

void rife_teardown(aji_ctx *c)
{
    auto &R = c->rife;
    model_release(&R.model);
    if (R.in_alloc)
        g_dml_api->FreeGPUAllocation(R.in_alloc);
    if (R.out_alloc)
        g_dml_api->FreeGPUAllocation(R.out_alloc);
    R.in_alloc = R.out_alloc = nullptr;
    R = {};
}

// Configure-time RIFE setup: geometry, session, bindings, the constant
// input channels. Format-dependent staging builds lazily in
// aji_infer_rife. Mirrors aji_trt's setup_rife (and its stats text).
bool setup_rife(aji_ctx *c, const AjiChainConf *chain, int w, int h,
                double fps)
{
    auto &R = c->rife;
    R.w = w;
    R.h = h;
    R.pw = (w + 63) / 64 * 64;
    R.ph = (h + 63) / 64 * 64;
    // centered like animejanai_core's AddBorders split, but rounded down
    // to even so 4:2:0 chroma stays aligned
    R.pad_l = ((R.pw - w) / 2) & ~1;
    R.pad_t = ((R.ph - h) / 2) & ~1;
    R.num = chain->rife_factor_num;
    R.den = chain->rife_factor_den;
    R.scd_threshold = chain->rife_scd_threshold;

    const std::string model =
        rife_model_name(chain->rife_model, chain->rife_ensemble);
    if (model.empty()) {
        c->set_error("invalid rife model code %d", chain->rife_model);
        return false;
    }
    const std::string onnx = c->rife_model_dir + "\\" + model + ".onnx";
    if (!model_open(c, &R.model, onnx, R.pw, R.ph, 11))
        return false;
    if (R.model.out_w != R.pw || R.model.out_h != R.ph) {
        c->set_error("%s: output %dx%d does not match padded input %dx%d",
                     onnx.c_str(), R.model.out_w, R.model.out_h, R.pw, R.ph);
        return false;
    }

    const size_t plane = (size_t)R.pw * R.ph;
    const size_t elem = R.model.fp16 ? 2 : 4;
    R.in_tensor = make_buffer(c->dev.Get(), plane * 11 * elem, false);
    R.out_tensor = make_buffer(c->dev.Get(), plane * 3 * elem, false);
    if (!R.in_tensor || !R.out_tensor) {
        c->set_error("rife tensor allocation failed");
        return false;
    }
    if (!c->ort_ck(g_dml_api->CreateGPUAllocationFromD3DResource(
                       R.in_tensor.Get(), &R.in_alloc), "rife in alloc") ||
        !c->ort_ck(g_dml_api->CreateGPUAllocationFromD3DResource(
                       R.out_tensor.Get(), &R.out_alloc), "rife out alloc"))
        return false;
    R.model.in_val = make_tensor(c, R.in_alloc, plane * 11 * elem, 1, 11,
                                 R.ph, R.pw, R.model.fp16);
    R.model.out_val = make_tensor(c, R.out_alloc, plane * 3 * elem, 1, 3,
                                  R.ph, R.pw, R.model.fp16);
    if (!R.model.in_val || !R.model.out_val)
        return false;
    if (!c->ort_ck(g_ort->BindInput(R.model.binding,
                                    R.model.in_name.c_str(),
                                    R.model.in_val), "rife BindInput") ||
        !c->ort_ck(g_ort->BindOutput(R.model.binding,
                                     R.model.out_name.c_str(),
                                     R.model.out_val), "rife BindOutput"))
        return false;

    // scene-detect reduction buffers (unpadded luma)
    const int gx = (w + 31) / 32, gy = (h + 7) / 8;
    R.scd_groups = gx * gy;
    R.scd_part = make_buffer(c->dev.Get(), (uint64_t)R.scd_groups * 4, false);
    R.scd_sum = make_buffer(c->dev.Get(), 4, false);
    R.scd_read = make_readback(c->dev.Get(), 4);
    if (!R.scd_part || !R.scd_sum || !R.scd_read) {
        c->set_error("rife scd buffer allocation failed");
        return false;
    }

    // constant channels 7..10 (mesh + multipliers), one-time fill
    {
        Recorder r{c};
        if (!r.begin())
            return false;
        KernArgs a = {};
        a.di[0] = R.pw;
        a.di[1] = R.ph;
        a.dj[0] = (int)(plane * 7);
        if (!r.dispatch(K_RIFE_CONSTS, false, R.model.fp16, a, NULL, NULL,
                        R.in_tensor.Get(), NULL, NULL,
                        R.model.fp16 ? R.pw / 2 : R.pw, R.ph, 1))
            return false;
        r.uav_barrier();
        if (!r.exec())
            return false;
    }
    // warm the rebound command list once (the discovery runs used the
    // temporary binding), then drain the queue so the consts/warmup work
    // is done before any later segment recycles the command allocators
    if (!c->ort_ck(g_ort->RunWithBinding(R.model.session, NULL,
                                         R.model.binding), "rife warmup"))
        return false;
    drain_queue(c, 30000);

    char buf[200];
    const double factor = (double)R.num / R.den;
    if (R.pw != w || R.ph != h) {
        snprintf(buf, sizeof(buf),
                 "Padded to %dx%d, applied RIFE v%d Interpolation %.3fx, "
                 "cropped back to %dx%d;    New Video FPS: %.3f",
                 R.pw, R.ph, chain->rife_model, factor, w, h, fps * factor);
    } else {
        snprintf(buf, sizeof(buf),
                 "Applied RIFE v%d Interpolation %.3fx;    "
                 "New Video FPS: %.3f",
                 chain->rife_model, factor, fps * factor);
    }
    c->log_steps.push_back(buf);
    R.enabled = true;
    return true;
}

} // namespace

/* ---------------- C ABI ---------------- */

extern "C" AJI_EXPORT aji_ctx *aji_create(const aji_create_params *params)
{
    if (!params || params->api_version != AJI_API_VERSION)
        return nullptr;

    auto c = std::make_unique<aji_ctx>();
    c->log_fn = params->log;
    c->log_opaque = params->log_opaque;

    std::string err;
    if (!load_ort(&err)) {
        c->set_error("%s", err.c_str());
        return nullptr;
    }
    if (!params->conf_path) {
        c->set_error("the DirectML backend has no direct (engine_path) mode");
        return nullptr;
    }
    if (!params->d3d11_device) {
        c->set_error("the DirectML backend needs the player's D3D11 device "
                     "(hwdec=d3d11va; CUDA frames go to the TensorRT "
                     "backend)");
        return nullptr;
    }

    c->conf_path = params->conf_path;
    c->model_dir = params->model_dir ? params->model_dir : "";
    c->rife_model_dir = params->rife_model_dir ? params->rife_model_dir : "";
    c->slot = params->slot ? params->slot : 1;
    if (!aji_conf_load(c->conf_path.c_str(), &c->conf, &err))
        c->verbose("conf load: %s", err.c_str());

    // D3D11 side
    c->dev11 = (ID3D11Device *)params->d3d11_device;
    c->dev11->GetImmediateContext(&c->ctx11);
    if (FAILED(c->ctx11.As(&c->ctx11_4))) {
        c->set_error("ID3D11DeviceContext4 unavailable (needs Windows 10 "
                     "1703+)");
        return nullptr;
    }

    if (getenv("AJI_DML_DEBUG")) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            c->verbose("D3D12 debug layer enabled");
        } else {
            c->verbose("D3D12 debug layer unavailable (Graphics Tools "
                       "not installed?)");
        }
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            c->verbose("DRED breadcrumbs + page-fault tracking enabled");
        }
    }

    // D3D12 device on the same adapter
    ComPtr<IDXGIDevice> dxgi_dev;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(c->dev11.As(&dxgi_dev)) ||
        FAILED(dxgi_dev->GetAdapter(&adapter))) {
        c->set_error("could not get the D3D11 device's adapter");
        return nullptr;
    }
    DXGI_ADAPTER_DESC ad = {};
    adapter->GetDesc(&ad);
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&c->dev)))) {
        c->set_error("D3D12CreateDevice failed on %ls (DirectML needs a "
                     "DX12 GPU)", ad.Description);
        return nullptr;
    }
    c->verbose("DirectML on %ls", ad.Description);

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
    if (FAILED(c->dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&c->queue)))) {
        c->set_error("D3D12 queue creation failed");
        return nullptr;
    }
    if (FAILED(c->dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                   IID_PPV_ARGS(&c->fence)))) {
        c->set_error("D3D12 fence creation failed");
        return nullptr;
    }
    if (FAILED(c->dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                   IID_PPV_ARGS(&c->done_fence)))) {
        c->set_error("D3D12 done-fence creation failed");
        return nullptr;
    }
    c->fence_event = CreateEventA(NULL, FALSE, FALSE, NULL);

    // share the fence into D3D11 for the input-ready handoff
    HANDLE fh = NULL;
    ComPtr<ID3D11Device5> dev11_5;
    if (FAILED(c->dev->CreateSharedHandle(c->fence.Get(), NULL, GENERIC_ALL,
                                          NULL, &fh)) ||
        FAILED(c->dev11.As(&dev11_5)) ||
        FAILED(dev11_5->OpenSharedFence(fh, IID_PPV_ARGS(&c->fence11)))) {
        if (fh)
            CloseHandle(fh);
        c->set_error("shared fence D3D11<->D3D12 setup failed");
        return nullptr;
    }
    CloseHandle(fh);

    if (FAILED(g_dml_create_device(c->dev.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                   IID_PPV_ARGS(&c->dml_dev)))) {
        c->set_error("DMLCreateDevice failed");
        return nullptr;
    }
    if (!make_rootsig(c.get()))
        return nullptr;

    if (!c->ort_ck(g_ort->CreateEnv(getenv("AJI_ORT_VERBOSE")
                                        ? ORT_LOGGING_LEVEL_VERBOSE
                                        : ORT_LOGGING_LEVEL_WARNING,
                                    "aji_dml", &c->env), "CreateEnv"))
        return nullptr;
    if (!c->ort_ck(g_ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0,
                                           OrtMemTypeDefault, &c->mi_dml),
                   "CreateMemoryInfo"))
        return nullptr;

    return c.release();
}

extern "C" AJI_EXPORT int aji_set_slot(aji_ctx *c, int slot)
{
    if (!c)
        return AJI_ERR;
    c->slot = slot;
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_configure(aji_ctx *c, int w, int h, double fps,
                                        int *out_w, int *out_h)
{
    if (!c || w < 2 || h < 2 || (w & 1) || (h & 1))
        return AJI_ERR_SHAPE;

    // re-read the conf each configure so edits apply without restart
    std::string err;
    aji_conf_load(c->conf_path.c_str(), &c->conf, &err);

    chain_teardown(c);
    c->log_info.clear();
    c->log_steps.clear();
    c->active = false;
    c->in_w = w;
    c->in_h = h;
    c->out_w = w;
    c->out_h = h;

    auto sit = c->conf.slots.find(c->slot);
    std::string profile = sit != c->conf.slots.end()
                              ? sit->second.profile_name : "(missing slot)";
    if (c->slot == 0)
        profile = "Off";
    else if (c->slot < 10)
        profile = std::to_string(c->slot) + ". " + profile;
    c->log_info.push_back("Upscale Profile: " + profile);
    {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "Original Video Resolution: %dx%d;    Original Video FPS: %.3f",
                 w, h, fps);
        c->log_info.push_back(buf);
    }

    const AjiChainConf *chain = nullptr;
    if (sit != c->conf.slots.end()) {
        const double px = (double)w * h;
        for (const auto &ch : sit->second.chains) {
            if (ch.min_px <= px && px <= ch.max_px &&
                ch.min_fps <= fps && fps <= ch.max_fps) {
                chain = &ch;
                break;
            }
        }
    }
    if (!chain) {
        c->log_info.push_back("No Chains Activated");
        finalize_log(c);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;
    }

    c->log_info.push_back(
        "Active Upscale Chain: " + std::to_string(chain->index) +
        ";    Resolution Range: " + chain->min_resolution + " - " +
        chain->max_resolution + ";    FPS Range: " + fmt_num(chain->min_fps) +
        " - " + fmt_num(chain->max_fps));

    int cw = w, ch = h;
    bool any_model = false;

    for (const auto &m : chain->models) {
        double factor = m.resize_factor_before_upscale;
        if (m.resize_height_before_upscale != 0)
            factor = 100;

        char buf[160];
        if (factor != 100) {
            int nw = round_even(cw * factor / 100.0);
            int nh = round_even(ch * factor / 100.0);
            if (nw >= 2 && nh >= 2 && (nw != cw || nh != ch)) {
                if (!push_resize_step(c, &cw, &ch, nw, nh))
                    return AJI_ERR;
                snprintf(buf, sizeof(buf),
                         "Applied Resize Factor Before Upscale: %g%%;    "
                         "New Video Resolution: %dx%d", factor, cw, ch);
                c->log_steps.push_back(buf);
            }
        }
        if (m.resize_height_before_upscale != 0 &&
            (int)m.resize_height_before_upscale != ch) {
            int nw, nh;
            fit_box(cw, ch, m.resize_height_before_upscale * 16.0 / 9.0,
                    m.resize_height_before_upscale, &nw, &nh);
            if (!push_resize_step(c, &cw, &ch, nw, nh))
                return AJI_ERR;
            snprintf(buf, sizeof(buf),
                     "Applied Resize Height Before Upscale: %gpx;    "
                     "New Video Resolution: %dx%d",
                     m.resize_height_before_upscale, cw, ch);
            c->log_steps.push_back(buf);
        } else if (ch > 1080) {
            int nw, nh;
            fit_box(cw, ch, 1920, 1080, &nw, &nh);
            if (!push_resize_step(c, &cw, &ch, nw, nh))
                return AJI_ERR;
            snprintf(buf, sizeof(buf),
                     "Applied Resize to Video Larger than 1080p;    "
                     "New Video Resolution: %dx%d", cw, ch);
            c->log_steps.push_back(buf);
        }

        if (m.name.empty())
            continue;

        DmlModel dm;
        const std::string onnx = c->model_dir + "\\" + m.name + ".onnx";
        if (!model_open(c, &dm, onnx, cw, ch, 3)) {
            model_release(&dm);
            finalize_log(c);
            return AJI_ERR_ENGINE;
        }
        cw = dm.out_w;
        ch = dm.out_h;
        c->models.push_back(std::move(dm));
        Step st;
        st.kind = Step::MODEL;
        st.out_w = cw;
        st.out_h = ch;
        st.model_idx = (int)c->models.size() - 1;
        c->steps.push_back(std::move(st));
        any_model = true;
        snprintf(buf, sizeof(buf),
                 "Applied Model: %s;    New Video Resolution: %dx%d",
                 m.name.c_str(), cw, ch);
        c->log_steps.push_back(buf);
    }

    if (chain->rife) {
        if (!c->rife_model_dir.empty()) {
            if (!setup_rife(c, chain, cw, ch, fps)) {
                finalize_log(c);
                return AJI_ERR_ENGINE;
            }
        } else {
            c->log_steps.push_back(
                "RIFE requested by the chain but no rife model dir is "
                "configured; interpolation disabled");
        }
    }

    finalize_log(c);

    if (!any_model && c->steps.empty()) {
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;
    }

    // assign chain-buffer ping-pong + element types, then allocate
    size_t max_bytes = 0;
    {
        int cur = 0;
        bool elem16 = c->models.empty() ? true : c->models[0].fp16;
        c->first_elem16 = elem16;
        int sw = w, sh = h;
        size_t in_elem = c->models.empty() ? 2 : (c->models[0].fp16 ? 2 : 4);
        max_bytes = (size_t)3 * sw * sh * in_elem;
        for (size_t i = 0; i < c->steps.size(); i++) {
            Step &st = c->steps[i];
            st.in_buf = cur;
            st.elem16_in = elem16;
            if (st.kind == Step::MODEL) {
                DmlModel &m = c->models[st.model_idx];
                if (m.fp16 != elem16) {
                    c->set_error("mixed fp16/fp32 models in one chain are "
                                 "not supported by the DirectML backend");
                    return AJI_ERR_CONF;
                }
                st.elem16_out = m.fp16;
            } else {
                // a resize feeding a model adopts that model's IO type
                bool next16 = elem16;
                for (size_t j = i + 1; j < c->steps.size(); j++) {
                    if (c->steps[j].kind == Step::MODEL) {
                        next16 = c->models[c->steps[j].model_idx].fp16;
                        break;
                    }
                }
                st.elem16_out = next16;
            }
            elem16 = st.elem16_out;
            cur ^= 1;
            sw = st.out_w;
            sh = st.out_h;
            max_bytes = (size_t)3 * sw * sh * (elem16 ? 2 : 4) > max_bytes
                            ? (size_t)3 * sw * sh * (elem16 ? 2 : 4)
                            : max_bytes;
        }
        c->last_elem16 = elem16;
        if (!c->models.empty() && c->first_elem16 != c->models[0].fp16) {
            c->set_error("internal: pre-kernel element type mismatch");
            return AJI_ERR;
        }
    }

    for (int i = 0; i < 2; i++) {
        c->buf[i] = make_buffer(c->dev.Get(), max_bytes, false);
        if (!c->buf[i]) {
            c->set_error("chain buffer allocation (%zu bytes) failed",
                         max_bytes);
            return AJI_ERR;
        }
        if (!c->ort_ck(g_dml_api->CreateGPUAllocationFromD3DResource(
                           c->buf[i].Get(), &c->buf_alloc[i]),
                       "CreateGPUAllocationFromD3DResource"))
            return AJI_ERR;
    }
    c->buf_bytes = max_bytes;

    for (Step &st : c->steps) {
        if (st.kind != Step::MODEL)
            continue;
        if (!model_bind(c, &c->models[st.model_idx], st.in_buf))
            return AJI_ERR_ENGINE;
    }

    c->out_w = cw;
    c->out_h = ch;
    c->active = true;
    if (out_w) *out_w = cw;
    if (out_h) *out_h = ch;
    return 1;
}

extern "C" AJI_EXPORT int aji_infer(aji_ctx *c, const aji_frame *in,
                                    const aji_frame *out, void *cu_stream)
{
    (void)cu_stream;
    if (!c || !in || !out)
        return AJI_ERR;
    if (!c->active) {
        c->set_error("aji_infer without an active configuration");
        return AJI_ERR;
    }
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010) {
        c->set_error("unsupported input format %d", in->format);
        return AJI_ERR_FORMAT;
    }
    if (out->format != in->format) {
        c->set_error("output format must match input");
        return AJI_ERR_FORMAT;
    }
    if (in->width != c->in_w || in->height != c->in_h ||
        out->width != c->out_w || out->height != c->out_h) {
        c->set_error("frame dims %dx%d->%dx%d do not match configured "
                     "%dx%d->%dx%d", in->width, in->height, out->width,
                     out->height, c->in_w, c->in_h, c->out_w, c->out_h);
        return AJI_ERR_SHAPE;
    }

    if (FAILED(c->dev->GetDeviceRemovedReason())) {
        diagnose_device(c, "aji_infer entry");
        return AJI_ERR;
    }
    // Pipelined: everything below only submits; the end-of-frame marker
    // (queued by the guard on every exit) is what callers gate on via
    // aji_flush/aji_done/aji_wait before consuming the output texture.
    DoneGuard done_guard{c};

    const int bpp = in->format == AJI_FMT_P010 ? 2 : 1;
    const aji_csp csp =
        aji_resample::make_csp(in->format, in->matrix, in->range);

    // (re)build staging + plans on format/dim changes
    const int skey[3] = {in->format, in->width, in->height};
    if (memcmp(skey, c->stage_key, sizeof(skey)) != 0) {
        c->in_pitch = align256((uint32_t)(in->width * bpp));
        c->out_pitch = align256((uint32_t)(out->width * bpp));
        c->in_y = make_buffer(c->dev.Get(),
                              (uint64_t)c->in_pitch * in->height, false);
        c->in_uv = make_buffer(c->dev.Get(),
                               (uint64_t)c->in_pitch * (in->height / 2),
                               false);
        c->out_y = make_buffer(c->dev.Get(),
                               (uint64_t)c->out_pitch * out->height, false);
        c->out_uv = make_buffer(c->dev.Get(),
                                (uint64_t)c->out_pitch * (out->height / 2),
                                false);
        if (!c->in_y || !c->in_uv || !c->out_y || !c->out_uv) {
            c->set_error("plane staging allocation failed");
            return AJI_ERR;
        }
        memcpy(c->stage_key, skey, sizeof(skey));
    }
    const int pkey[4] = {in->format, in->width, in->height, in->siting};
    if (!c->pre_plan || memcmp(pkey, c->pre_key, sizeof(pkey)) != 0) {
        c->pre_plan = pre_plan_create(c, in->format, in->width, in->height,
                                      in->siting, AJI_FILTER_SPLINE36);
        if (!c->pre_plan) {
            c->set_error("pre plan allocation failed");
            return AJI_ERR;
        }
        memcpy(c->pre_key, pkey, sizeof(pkey));
    }
    // the reference pipeline's final 4:2:0 subsample is always LEFT-sited
    const int qkey[4] = {out->format, c->out_w, c->out_h, AJI_SITING_LEFT};
    if (!c->post_plan || memcmp(qkey, c->post_key, sizeof(qkey)) != 0) {
        c->post_plan = post_plan_create(c, out->format, c->out_w, c->out_h,
                                        AJI_SITING_LEFT, AJI_FILTER_SPLINE36);
        if (!c->post_plan) {
            c->set_error("post plan allocation failed");
            return AJI_ERR;
        }
        memcpy(c->post_key, qkey, sizeof(qkey));
    }

    ID3D12Resource *in_tex = open_shared(c, in->plane[0]);
    ID3D12Resource *out_tex = open_shared(c, out->plane[0]);
    if (!in_tex || !out_tex)
        return AJI_ERR;
    const UINT in_sub = (UINT)(intptr_t)in->plane[1];
    const UINT out_sub = (UINT)(intptr_t)out->plane[1];

    // input-ready handoff: D3D11's queued work (the caller's copy into
    // the input texture) must land before our queue reads it
    c->fence_val++;
    c->ctx11_4->Signal(c->fence11.Get(), c->fence_val);
    c->ctx11->Flush();
    c->queue->Wait(c->fence.Get(), c->fence_val);

    Recorder r{c};
    if (!r.begin())
        return AJI_ERR;
    transition_planes(c, in_tex, in_sub, D3D12_RESOURCE_STATE_COMMON,
                      D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_plane(c, true, in_tex, in_sub, 0, in->format, in->width, in->height,
               c->in_y.Get(), c->in_pitch);
    copy_plane(c, true, in_tex, in_sub, 1, in->format, in->width, in->height,
               c->in_uv.Get(), c->in_pitch);
    transition_planes(c, in_tex, in_sub, D3D12_RESOURCE_STATE_COPY_SOURCE,
                      D3D12_RESOURCE_STATE_COMMON);
    {
        D3D12_RESOURCE_BARRIER bs[2] = {};
        for (int i = 0; i < 2; i++) {
            bs[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bs[i].Transition.Subresource = 0;
            bs[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            bs[i].Transition.StateAfter =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        bs[0].Transition.pResource = c->in_y.Get();
        bs[1].Transition.pResource = c->in_uv.Get();
        c->cl->ResourceBarrier(2, bs);
    }
    if (!record_pre(c, r, c->pre_plan.get(), csp, c->in_y.Get(),
                    c->in_uv.Get(), c->in_pitch, c->buf[0].Get(), 0,
                    c->first_elem16))
        return AJI_ERR;
    if (!r.exec())
        return AJI_ERR;

    int cur = 0;
    for (Step &st : c->steps) {
        if (st.kind == Step::MODEL) {
            DmlModel &m = c->models[st.model_idx];
            if (!c->ort_ck(g_ort->RunWithBinding(m.session, NULL, m.binding),
                           "RunWithBinding"))
                return AJI_ERR_ENGINE;
        } else {
            if (!r.begin() ||
                !record_resize(c, r, st.plan.get(), c->buf[cur].Get(),
                               c->buf[cur ^ 1].Get(), st.elem16_in,
                               st.elem16_out) ||
                !r.exec())
                return AJI_ERR;
        }
        cur ^= 1;
    }

    if (!r.begin())
        return AJI_ERR;
    if (!record_post(c, r, c->post_plan.get(), csp, c->buf[cur].Get(),
                     c->out_y.Get(), c->out_uv.Get(), c->out_pitch,
                     c->last_elem16))
        return AJI_ERR;
    {
        D3D12_RESOURCE_BARRIER bs[2] = {};
        for (int i = 0; i < 2; i++) {
            bs[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bs[i].Transition.Subresource = 0;
            bs[i].Transition.StateBefore =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            bs[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }
        bs[0].Transition.pResource = c->out_y.Get();
        bs[1].Transition.pResource = c->out_uv.Get();
        c->cl->ResourceBarrier(2, bs);
    }
    transition_planes(c, out_tex, out_sub, D3D12_RESOURCE_STATE_COMMON,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    copy_plane(c, false, out_tex, out_sub, 0, out->format, out->width,
               out->height, c->out_y.Get(), c->out_pitch);
    copy_plane(c, false, out_tex, out_sub, 1, out->format, out->width,
               out->height, c->out_uv.Get(), c->out_pitch);
    transition_planes(c, out_tex, out_sub, D3D12_RESOURCE_STATE_COPY_DEST,
                      D3D12_RESOURCE_STATE_COMMON);
    if (!r.exec())
        return AJI_ERR;

    // No CPU wait: the guard queues this frame's end-of-frame marker and
    // the caller synchronizes through the ticket API.
    return AJI_OK;
}

extern "C" AJI_EXPORT const char *aji_current_log(aji_ctx *c)
{
    return c ? c->current_log.c_str() : "";
}

extern "C" AJI_EXPORT int aji_scale_factor(aji_ctx *c)
{
    (void)c;
    return 0;   // conf mode only
}

extern "C" AJI_EXPORT int aji_rife_factor(aji_ctx *c, int *num, int *den)
{
    if (!c || !c->rife.enabled)
        return 0;
    if (num)
        *num = c->rife.num;
    if (den)
        *den = c->rife.den;
    return 1;
}

extern "C" AJI_EXPORT int aji_poll(aji_ctx *c)
{
    (void)c;
    return 0;   // no background builds on DirectML
}

extern "C" AJI_EXPORT uint64_t aji_flush(aji_ctx *c, void *cu_stream)
{
    (void)cu_stream;
    if (!c || !c->done_fence)
        return 0;
    // every aji_infer exit already queued a marker; a fresh one also
    // covers any submissions made since (rife, warmups)
    return signal_done(c);
}

extern "C" AJI_EXPORT int aji_done(aji_ctx *c, uint64_t ticket)
{
    if (!c)
        return AJI_ERR;
    if (ticket == 0)
        return 1;
    if (!c->done_fence || ticket > c->done_val)
        return AJI_ERR;
    return c->done_fence->GetCompletedValue() >= ticket ? 1 : 0;
}

extern "C" AJI_EXPORT int aji_wait(aji_ctx *c, uint64_t ticket)
{
    if (!c)
        return AJI_ERR;
    if (ticket == 0)
        return AJI_OK;
    if (!c->done_fence || ticket > c->done_val)
        return AJI_ERR;
    if (!wait_done_val(c, ticket, 10000)) {
        c->set_error("GPU timeout waiting for ticket %llu",
                     (unsigned long long)ticket);
        diagnose_device(c, "ticket wait");
        return AJI_ERR;
    }
    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_infer_rife(aji_ctx *c, const aji_frame *a,
                                         const aji_frame *b, double t,
                                         const aji_frame *out, void *cu_stream)
{
    (void)cu_stream;
    if (!c || !a || !b || !out)
        return AJI_ERR;
    auto &R = c->rife;
    if (!R.enabled) {
        c->set_error("aji_infer_rife without an active RIFE configuration");
        return AJI_ERR;
    }
    if (a->format != b->format || a->format != out->format ||
        (a->format != AJI_FMT_NV12 && a->format != AJI_FMT_P010)) {
        c->set_error("rife frame formats must match (nv12/p010)");
        return AJI_ERR_FORMAT;
    }
    if (a->width != R.w || a->height != R.h || b->width != R.w ||
        b->height != R.h || out->width != R.w || out->height != R.h) {
        c->set_error("rife frame dims do not match configured %dx%d",
                     R.w, R.h);
        return AJI_ERR_SHAPE;
    }

    // Synchronous by contract (scene-change readback): recordings still
    // retire through the marker the guard queues on every exit.
    DoneGuard done_guard{c};

    const int fmt = a->format;
    const bool t16 = fmt == AJI_FMT_P010;
    const bool io16 = R.model.fp16;
    const int bpp = t16 ? 2 : 1;
    const uint32_t prow = (uint32_t)(R.pw * bpp);   // tight padded pitch
    const size_t plane = (size_t)R.pw * R.ph;

    // format-dependent staging + plans, built on first use; pad borders
    // get a one-time studio-black fill (interiors are overwritten)
    if (R.format != fmt) {
        R.pre = pre_plan_create(c, fmt, R.pw, R.ph, AJI_SITING_LEFT,
                                AJI_FILTER_BILINEAR);
        R.post = post_plan_create(c, fmt, R.pw, R.ph, AJI_SITING_LEFT,
                                  AJI_FILTER_BILINEAR);
        R.st_pitch = align256((uint32_t)(R.w * bpp));
        bool ok = R.pre && R.post;
        for (int i = 0; i < 3 && ok; i++) {
            R.pad_y[i] = make_buffer(c->dev.Get(),
                                     (uint64_t)prow * R.ph, false);
            R.pad_uv[i] = make_buffer(c->dev.Get(),
                                      (uint64_t)prow * (R.ph / 2), false);
            R.st_y[i] = make_buffer(c->dev.Get(),
                                    (uint64_t)R.st_pitch * R.h, false);
            R.st_uv[i] = make_buffer(c->dev.Get(),
                                     (uint64_t)R.st_pitch * (R.h / 2), false);
            ok = R.pad_y[i] && R.pad_uv[i] && R.st_y[i] && R.st_uv[i];
        }
        if (!ok) {
            c->set_error("rife staging allocation failed");
            R.format = 0;
            return AJI_ERR;
        }
        const uint32_t ypat = t16 ? 0x10001000u : 0x10101010u;
        const uint32_t cpat = t16 ? 0x80008000u : 0x80808080u;
        Recorder rf{c};
        if (!rf.begin())
            return AJI_ERR;
        for (int i = 0; i < 2; i++) {
            record_fill(c, rf, R.pad_y[i].Get(), 0, prow * R.ph / 4, ypat);
            record_fill(c, rf, R.pad_uv[i].Get(), 0, prow * (R.ph / 2) / 4,
                        cpat);
        }
        rf.uav_barrier();
        if (!rf.exec())
            return AJI_ERR;
        R.format = fmt;
    }

    ID3D12Resource *a_tex = open_shared(c, a->plane[0]);
    ID3D12Resource *b_tex = open_shared(c, b->plane[0]);
    ID3D12Resource *o_tex = open_shared(c, out->plane[0]);
    if (!a_tex || !b_tex || !o_tex)
        return AJI_ERR;
    const UINT a_sub = (UINT)(intptr_t)a->plane[1];
    const UINT b_sub = (UINT)(intptr_t)b->plane[1];
    const UINT o_sub = (UINT)(intptr_t)out->plane[1];

    // input-ready handoff (no-op when a/b were written by our own queue)
    c->fence_val++;
    c->ctx11_4->Signal(c->fence11.Get(), c->fence_val);
    c->ctx11->Flush();
    c->queue->Wait(c->fence.Get(), c->fence_val);

    // stage both frames + scene-detect on the unpadded luma
    Recorder r{c};
    if (!r.begin())
        return AJI_ERR;
    ID3D12Resource *tex[2] = {a_tex, b_tex};
    const UINT sub[2] = {a_sub, b_sub};
    for (int i = 0; i < 2; i++) {
        transition_planes(c, tex[i], sub[i], D3D12_RESOURCE_STATE_COMMON,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
        copy_plane(c, true, tex[i], sub[i], 0, fmt, R.w, R.h,
                   R.st_y[i].Get(), R.st_pitch);
        copy_plane(c, true, tex[i], sub[i], 1, fmt, R.w, R.h,
                   R.st_uv[i].Get(), R.st_pitch);
        transition_planes(c, tex[i], sub[i],
                          D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_COMMON);
    }
    {
        D3D12_RESOURCE_BARRIER bs[4] = {};
        ID3D12Resource *st[4] = {R.st_y[0].Get(), R.st_uv[0].Get(),
                                 R.st_y[1].Get(), R.st_uv[1].Get()};
        for (int i = 0; i < 4; i++) {
            bs[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bs[i].Transition.pResource = st[i];
            bs[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            bs[i].Transition.StateAfter =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        c->cl->ResourceBarrier(4, bs);
    }
    {
        KernArgs sa = {};
        sa.di[0] = R.w;
        sa.di[1] = R.h;
        sa.di[2] = (R.w + 31) / 32;
        sa.dj[0] = (int)R.st_pitch;
        sa.dj[1] = (int)R.st_pitch;
        sa.kr = t16 ? 1.0f / 65472.0f : 1.0f / 255.0f;
        if (!r.dispatch(K_SCD_PARTIAL, t16, false, sa, NULL, NULL,
                        R.st_y[0].Get(), R.st_y[1].Get(), R.scd_part.Get(),
                        R.w, R.h, 1))
            return AJI_ERR;
        r.uav_barrier();
        KernArgs sf = {};
        sf.di[0] = R.scd_groups;
        if (!r.dispatch(K_SCD_FINAL, false, false, sf, NULL, NULL,
                        R.scd_part.Get(), R.scd_sum.Get(), NULL, 32, 1, 1))
            return AJI_ERR;
        r.transition(R.scd_sum.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_SOURCE, 0);
        c->cl->CopyBufferRegion(R.scd_read.Get(), 0, R.scd_sum.Get(), 0, 4);
    }
    if (!r.exec())
        return AJI_ERR;
    if (!wait_done_val(c, signal_done(c), 10000)) {
        c->set_error("GPU timeout in rife scene detect");
        return AJI_ERR;
    }
    float sum = 0.0f;
    {
        void *m = nullptr;
        D3D12_RANGE rr = {0, 4};
        if (FAILED(R.scd_read->Map(0, &rr, &m)))
            return AJI_ERR;
        memcpy(&sum, m, 4);
        D3D12_RANGE none = {0, 0};
        R.scd_read->Unmap(0, &none);
    }
    // The reference runs SCDetect on the padded clip; the constant borders
    // contribute zero difference, so its mean divides by the padded area.
    if (sum / ((double)R.pw * R.ph) > R.scd_threshold)
        return AJI_SCENE;

    // embed the frames into the pad interiors, convert, run, convert back
    const uint32_t doff_y = (uint32_t)R.pad_t * prow +
                            (uint32_t)(R.pad_l * bpp);
    const uint32_t doff_uv = (uint32_t)(R.pad_t / 2) * prow +
                             (uint32_t)(R.pad_l * bpp);
    if (!r.begin())
        return AJI_ERR;
    for (int i = 0; i < 2; i++) {
        if (!record_window(c, r, R.st_y[i].Get(), 0, R.st_pitch,
                           R.pad_y[i].Get(), doff_y, prow,
                           (uint32_t)(R.w * bpp), R.h) ||
            !record_window(c, r, R.st_uv[i].Get(), 0, R.st_pitch,
                           R.pad_uv[i].Get(), doff_uv, prow,
                           (uint32_t)(R.w * bpp), R.h / 2))
            return AJI_ERR;
    }
    r.uav_barrier();
    // padded YUV -> RGB into channels 0-2 (a) and 3-5 (b), bilinear
    // chroma, hardcoded BT.709 like rife_cuda.py
    const aji_csp csp =
        aji_resample::make_csp(fmt, AJI_MATRIX_BT709, a->range);
    for (int i = 0; i < 2; i++) {
        if (!record_pre(c, r, R.pre.get(), csp, R.pad_y[i].Get(),
                        R.pad_uv[i].Get(), prow, R.in_tensor.Get(),
                        (int)(plane * 3 * i), io16))
            return AJI_ERR;
    }
    // channel 6: the timestep plane
    {
        uint32_t value, dwords, start;
        if (io16) {
            uint32_t x;
            float ft = (float)t;
            memcpy(&x, &ft, 4);
            uint32_t sign = (x >> 16) & 0x8000;
            int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
            uint32_t half = e <= 0 ? sign
                            : e >= 31 ? (sign | 0x7c00)
                            : (sign | (e << 10) | ((x & 0x7fffff) >> 13));
            value = half | (half << 16);
            start = (uint32_t)(plane * 6 / 2);
            dwords = (uint32_t)(plane / 2);
        } else {
            float ft = (float)t;
            memcpy(&value, &ft, 4);
            start = (uint32_t)(plane * 6);
            dwords = (uint32_t)plane;
        }
        if (!record_fill(c, r, R.in_tensor.Get(), start, dwords, value))
            return AJI_ERR;
        r.uav_barrier();
    }
    if (!r.exec())
        return AJI_ERR;

    if (!c->ort_ck(g_ort->RunWithBinding(R.model.session, NULL,
                                         R.model.binding),
                   "rife RunWithBinding"))
        return AJI_ERR_ENGINE;

    // RGB -> padded YUV (bilinear, 709), crop the window out, hand back
    if (!r.begin())
        return AJI_ERR;
    if (!record_post(c, r, R.post.get(), csp, R.out_tensor.Get(),
                     R.pad_y[2].Get(), R.pad_uv[2].Get(), prow, io16))
        return AJI_ERR;
    if (!record_window(c, r, R.pad_y[2].Get(), doff_y, prow,
                       R.st_y[2].Get(), 0, R.st_pitch,
                       (uint32_t)(R.w * bpp), R.h) ||
        !record_window(c, r, R.pad_uv[2].Get(), doff_uv, prow,
                       R.st_uv[2].Get(), 0, R.st_pitch,
                       (uint32_t)(R.w * bpp), R.h / 2))
        return AJI_ERR;
    {
        D3D12_RESOURCE_BARRIER bs[2] = {};
        ID3D12Resource *st[2] = {R.st_y[2].Get(), R.st_uv[2].Get()};
        for (int i = 0; i < 2; i++) {
            bs[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bs[i].Transition.pResource = st[i];
            bs[i].Transition.StateBefore =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            bs[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }
        c->cl->ResourceBarrier(2, bs);
    }
    transition_planes(c, o_tex, o_sub, D3D12_RESOURCE_STATE_COMMON,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    copy_plane(c, false, o_tex, o_sub, 0, fmt, R.w, R.h, R.st_y[2].Get(),
               R.st_pitch);
    copy_plane(c, false, o_tex, o_sub, 1, fmt, R.w, R.h, R.st_uv[2].Get(),
               R.st_pitch);
    transition_planes(c, o_tex, o_sub, D3D12_RESOURCE_STATE_COPY_DEST,
                      D3D12_RESOURCE_STATE_COMMON);
    if (!r.exec())
        return AJI_ERR;

    // synchronous exception: the interpolated output is complete on return
    if (!wait_done_val(c, signal_done(c), 10000)) {
        c->set_error("GPU timeout in rife inference");
        diagnose_device(c, "rife fence wait");
        return AJI_ERR;
    }
    return AJI_OK;
}

extern "C" AJI_EXPORT const char *aji_last_error(aji_ctx *c)
{
    return c ? c->errbuf : "no context";
}

extern "C" AJI_EXPORT void aji_destroy(aji_ctx **pc)
{
    if (!pc || !*pc)
        return;
    aji_ctx *c = *pc;
    chain_teardown(c);
    if (c->mi_dml)
        g_ort->ReleaseMemoryInfo(c->mi_dml);
    if (c->env)
        g_ort->ReleaseEnv(c->env);
    if (c->fence_event)
        CloseHandle(c->fence_event);
    delete c;
    *pc = nullptr;
}
