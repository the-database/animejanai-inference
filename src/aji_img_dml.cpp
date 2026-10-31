/*
 * aji_img_dml — DirectML implementation of the image ABI (aji_image.h),
 * compiled into aji_dml alongside the video front-end (Windows only).
 *
 * Each context owns a D3D12 device + DirectML device + DIRECT queue on
 * the chosen adapter (by LUID), shared by every model's ORT session via
 * the _DML1 EP append, as the video path does. Tiles cross as host
 * tensors (ORT inserts the uploads/readbacks); the tile shape compiles
 * at load with two warm-up runs, which also discover the model's scale.
 * There is no engine cache, so models are READY as soon as they load.
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "aji_image.h"
#include "onnx_shape.h"
#include "core/fileio.h"
#include "core/half.h"
#include "core/ort_dml.h"
#include "core/tiling.h"

using Microsoft::WRL::ComPtr;
using namespace ajicore;
namespace fs = std::filesystem;

namespace {

std::string narrow(const wchar_t *w)
{
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? (size_t)n - 1 : 0, '\0');
    if (n > 0)
        WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

// D3DKMTQueryAdapterInfo for the adapter `luid`; false if anything fails.
// The D3DKMT structs mirror d3dkmthk.h (which needs the driver-kit NTSTATUS
// headers), and the calls resolve from gdi32 at runtime.
bool kmt_query(const LUID &luid, UINT type, void *data, UINT size)
{
    struct OpenFromLuid { LUID luid; UINT handle; };
    struct QueryInfo { UINT handle; UINT type; void *data; UINT size; };
    struct CloseAdapter { UINT handle; };
    using OpenFn = LONG(APIENTRY *)(OpenFromLuid *);
    using QueryFn = LONG(APIENTRY *)(const QueryInfo *);
    using CloseFn = LONG(APIENTRY *)(const CloseAdapter *);

    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    if (!gdi)
        gdi = LoadLibraryW(L"gdi32.dll");
    auto open = gdi ? (OpenFn)GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid") : nullptr;
    auto query = gdi ? (QueryFn)GetProcAddress(gdi, "D3DKMTQueryAdapterInfo") : nullptr;
    auto close = gdi ? (CloseFn)GetProcAddress(gdi, "D3DKMTCloseAdapter") : nullptr;
    if (!open || !query || !close)
        return false;
    OpenFromLuid o = {luid, 0};
    if (open(&o) != 0)
        return false;
    QueryInfo q = {o.handle, type, data, size};
    bool ok = query(&q) == 0;
    CloseAdapter c = {o.handle};
    close(&c);
    return ok;
}

// An indirect-display adapter: a virtual monitor (Parsec, Meta Quest Link, any
// IddCx driver). DXGI lists one per virtual display, each describing the GPU
// that renders for it (same name, VRAM and D3D12 support, its own LUID), so a
// single card would show up once per virtual display. The kernel adapter type
// tells them apart; a failed lookup keeps the adapter.
bool is_indirect_display(const LUID &luid)
{
    constexpr UINT KMTQAITYPE_ADAPTERTYPE = 15;
    constexpr UINT INDIRECT_DISPLAY_DEVICE = 1u << 6; // D3DKMT_ADAPTERTYPE bit
    UINT type = 0;
    return kmt_query(luid, KMTQAITYPE_ADAPTERTYPE, &type, sizeof(type)) &&
           (type & INDIRECT_DISPLAY_DEVICE);
}

// The device id: the adapter's PCI location, encoded as aji_img_trt's pci_id
// does, so a card has the same id on both backends and keeps it across
// reboots and driver restarts (a LUID is only unique until the next boot).
// D3DKMT reports no PCI domain; it is 0 on single-segment (desktop) systems.
// If Windows can't report an address, the LUID stands in (bit 63 clear, so
// it can't collide with a PCI id).
uint64_t device_id(const LUID &luid)
{
    constexpr UINT KMTQAITYPE_ADAPTERADDRESS = 6;
    struct { UINT bus, device, function; } addr = {}; // D3DKMT_ADAPTERADDRESS
    if (kmt_query(luid, KMTQAITYPE_ADAPTERADDRESS, &addr, sizeof(addr)))
        return (1ull << 63) | ((uint64_t)addr.bus << 8) | (uint64_t)(addr.device & 0xff);
    return ((uint64_t)(uint32_t)luid.HighPart << 32) | (uint64_t)luid.LowPart;
}

// Hardware adapters that can create a D3D12 device, in DXGI order, without
// the virtual-display duplicates (is_indirect_display).
std::vector<ComPtr<IDXGIAdapter1>> adapters()
{
    std::vector<ComPtr<IDXGIAdapter1>> out;
    ComPtr<IDXGIFactory1> f;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f))))
        return out;
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            !is_indirect_display(d.AdapterLuid) &&
            SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0,
                                        __uuidof(ID3D12Device), nullptr)))
            out.push_back(a);
        a.Reset();
    }
    return out;
}

} // namespace

struct aji_img_ctx {
    aji_img_log_fn log = nullptr;
    void *log_opaque = nullptr;
    ComPtr<ID3D12Device> dev;
    ComPtr<IDMLDevice> dml_dev;
    ComPtr<ID3D12CommandQueue> queue;
    OrtEnv *env = nullptr;
    OrtMemoryInfo *mi_cpu = nullptr;

    std::mutex mu;  // guards models and err
    std::set<aji_img_model *> models;
    std::string err;

    void set_error(const std::string &e) {
        err = e;
        if (log)
            log(log_opaque, 1, e.c_str());
    }
    // Returns false (with the error recorded) on a non-null status.
    bool ck(OrtStatus *st, const std::string &what, std::string *e) {
        if (!st)
            return true;
        *e = what + ": " + g_ort->GetErrorMessage(st);
        g_ort->ReleaseStatus(st);
        return false;
    }
};

struct aji_img_model : TileModel {
    aji_img_ctx *c = nullptr;
    std::string name;
    int overlap = DEFAULT_OVERLAP;
    std::mutex run_mu;
    OrtSession *session = nullptr;
    std::string in_name, out_name;
    bool fp16 = false;  // tensor element type (fp32 otherwise)
    std::vector<uint16_t> in16, out16;
    std::vector<float> in32, out32;

    // Run one tile from `in` (planar f32) into `out`. Host tensors on
    // both sides; ORT moves them to/from the DML device.
    int run_tile(const float *in, float *out, std::string *err) override
    {
        const size_t ni = (size_t)in_ch * tile_w * tile_h;
        const size_t no = (size_t)out_ch * tile_w * scale * tile_h * scale;
        const int64_t idims[4] = {1, in_ch, tile_h, tile_w};
        const int64_t odims[4] = {1, out_ch, (int64_t)tile_h * scale,
                                  (int64_t)tile_w * scale};
        void *ip, *op;
        size_t ib, ob;
        if (fp16) {
            for (size_t i = 0; i < ni; i++)
                in16[i] = f32_to_f16(in[i]);
            ip = in16.data(); op = out16.data();
            ib = ni * 2; ob = no * 2;
        } else {
            memcpy(in32.data(), in, ni * 4);
            ip = in32.data(); op = out32.data();
            ib = ni * 4; ob = no * 4;
        }
        const ONNXTensorElementDataType et = fp16
            ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        OrtValue *iv = nullptr, *ov = nullptr;
        int rc = AJI_IMG_ERR;
        do {
            if (!c->ck(g_ort->CreateTensorWithDataAsOrtValue(c->mi_cpu, ip, ib,
                                                             idims, 4, et, &iv),
                       "input tensor", err) ||
                !c->ck(g_ort->CreateTensorWithDataAsOrtValue(c->mi_cpu, op, ob,
                                                             odims, 4, et, &ov),
                       "output tensor", err))
                break;
            const char *in_n = in_name.c_str(), *out_n = out_name.c_str();
            if (!c->ck(g_ort->Run(session, nullptr, &in_n, &iv, 1, &out_n, 1, &ov),
                       "DirectML run of " + name, err))
                break;
            rc = AJI_IMG_OK;
        } while (0);
        if (iv)
            g_ort->ReleaseValue(iv);
        if (ov)
            g_ort->ReleaseValue(ov);
        if (rc != AJI_IMG_OK)
            return rc;
        if (fp16) {
            for (size_t i = 0; i < no; i++)
                out[i] = f16_to_f32(out16[i]);
        } else {
            memcpy(out, out32.data(), no * 4);
        }
        return AJI_IMG_OK;
    }
};

namespace {

// Run the zeroed tile through a fresh session twice (DML command-list
// warm-up) with ORT allocating the output, and read scale/out_ch from it.
bool discover(aji_img_model *m, std::string *e)
{
    aji_img_ctx *c = m->c;
    const size_t ni = (size_t)m->in_ch * m->tile_w * m->tile_h;
    std::vector<uint16_t> z16(m->fp16 ? ni : 0, 0);
    std::vector<float> z32(m->fp16 ? 0 : ni, 0.0f);
    const int64_t idims[4] = {1, m->in_ch, m->tile_h, m->tile_w};
    OrtValue *iv = nullptr, *ov = nullptr;
    bool ok = false;
    do {
        if (!c->ck(g_ort->CreateTensorWithDataAsOrtValue(
                       c->mi_cpu, m->fp16 ? (void *)z16.data() : (void *)z32.data(),
                       ni * (m->fp16 ? 2 : 4), idims, 4,
                       m->fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                               : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                       &iv),
                   "discovery tensor", e))
            break;
        const char *in_n = m->in_name.c_str(), *out_n = m->out_name.c_str();
        bool ran = true;
        for (int k = 0; k < 2 && ran; k++) {
            if (ov) {
                g_ort->ReleaseValue(ov);
                ov = nullptr;
            }
            ran = c->ck(g_ort->Run(m->session, nullptr, &in_n, &iv, 1, &out_n, 1, &ov),
                        m->name + " (warm-up run)", e);
        }
        if (!ran)
            break;
        OrtTensorTypeAndShapeInfo *si = nullptr;
        if (!c->ck(g_ort->GetTensorTypeAndShape(ov, &si), "output shape", e))
            break;
        int64_t d[4] = {0};
        size_t nd = 0;
        g_ort->GetDimensionsCount(si, &nd);
        if (nd == 4)
            g_ort->GetDimensions(si, d, 4);
        g_ort->ReleaseTensorTypeAndShapeInfo(si);
        if (nd != 4 || (d[1] != 1 && d[1] != 3) || d[3] % m->tile_w ||
            d[2] % m->tile_h || d[3] / m->tile_w != d[2] / m->tile_h) {
            *e = m->name + ": output is not an integer-scale 1/3-channel image "
                           "of the tile";
            break;
        }
        m->out_ch = (int)d[1];
        m->scale = (int)(d[3] / m->tile_w);
        ok = true;
    } while (0);
    if (iv)
        g_ort->ReleaseValue(iv);
    if (ov)
        g_ort->ReleaseValue(ov);
    return ok;
}

void copy_err(const std::string &e, char *err, size_t n)
{
    if (err && n)
        snprintf(err, n, "%s", e.c_str());
}

void release_model(aji_img_model *m)
{
    if (m->session)
        g_ort->ReleaseSession(m->session);
    delete m;
}

} // namespace

extern "C" AJI_IMG_EXPORT int aji_img_enum_devices(int backend,
                                                   aji_img_device *out, int max)
{
    if (backend != AJI_IMG_DIRECTML)
        return -1;
    std::vector<ComPtr<IDXGIAdapter1>> as = adapters();
    for (int i = 0; i < (int)as.size() && out && i < max; i++) {
        DXGI_ADAPTER_DESC1 d = {};
        as[i]->GetDesc1(&d);
        aji_img_device &o = out[i];
        memset(&o, 0, sizeof(o));
        o.id = device_id(d.AdapterLuid);
        snprintf(o.name, sizeof(o.name), "%s", narrow(d.Description).c_str());
        o.vram_bytes = d.DedicatedVideoMemory;
    }
    return (int)as.size();
}

extern "C" AJI_IMG_EXPORT aji_img_ctx *aji_img_create(const aji_img_create_params *p,
                                                      char *err, size_t err_size)
{
    if (!p || p->api_version != AJI_IMAGE_API_VERSION ||
        p->backend != AJI_IMG_DIRECTML) {
        copy_err("aji_img_create: bad params or API version mismatch", err, err_size);
        return nullptr;
    }
    std::string e;
    if (!load_ort(&e)) {
        copy_err(e, err, err_size);
        return nullptr;
    }
    ComPtr<IDXGIAdapter1> adapter;
    for (auto &a : adapters()) {
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        if (p->device_id == 0 || device_id(d.AdapterLuid) == p->device_id) {
            adapter = a;
            break;
        }
    }
    if (!adapter) {
        copy_err("no DirectX 12 adapter (or the requested one is gone)", err, err_size);
        return nullptr;
    }

    auto c = std::make_unique<aji_img_ctx>();
    c->log = p->log;
    c->log_opaque = p->log_opaque;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&c->dev))) ||
        FAILED(c->dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&c->queue))) ||
        FAILED(g_dml_create_device(c->dev.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                   IID_PPV_ARGS(&c->dml_dev)))) {
        copy_err("D3D12/DirectML device creation failed", err, err_size);
        return nullptr;
    }
    if (!c->ck(g_ort->CreateEnv(getenv("AJI_ORT_VERBOSE") ? ORT_LOGGING_LEVEL_VERBOSE
                                                          : ORT_LOGGING_LEVEL_WARNING,
                                "aji_img_dml", &c->env),
               "CreateEnv", &e) ||
        !c->ck(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                          &c->mi_cpu),
               "CreateCpuMemoryInfo", &e)) {
        copy_err(e, err, err_size);
        if (c->env)
            g_ort->ReleaseEnv(c->env);
        return nullptr;
    }
    return c.release();
}

extern "C" AJI_IMG_EXPORT aji_img_model *aji_img_load(aji_img_ctx *c,
                                                      const aji_img_model_params *p)
{
    if (!c)
        return nullptr;
    std::lock_guard<std::mutex> lock(c->mu);
    if (!p || !p->onnx_path) {
        c->set_error("aji_img_load: no model path");
        return nullptr;
    }
    aji_img_model *m = new aji_img_model;
    m->c = c;
    m->name = fs::u8path(p->onnx_path).stem().u8string();
    std::string e;
    bool ok = false;
    OrtSessionOptions *so = nullptr;
    do {
        if (!c->ck(g_ort->CreateSessionOptions(&so), "CreateSessionOptions", &e))
            break;
        g_ort->SetSessionExecutionMode(so, ORT_SEQUENTIAL);
        g_ort->DisableMemPattern(so);
        g_ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
        // Every run is one (1, 3, tile_h, tile_w) tile, so pin a dynamic
        // model's symbolic dims to it: with free dims the DML EP dispatches
        // the graph op by op instead of compiling it whole. A model with
        // static H/W fixes the tile size itself (below), so leave it alone.
        {
            std::vector<char> blob;
            AjiOnnxInput oin;
            if (ajicore::read_file_bytes(p->onnx_path, &blob) &&
                aji_onnx_input(blob.data(), blob.size(), &oin) &&
                oin.dims.size() == 4 && oin.dims[2] <= 0 && oin.dims[3] <= 0) {
                const int64_t tw = p->tile_w > 0 ? p->tile_w : DEFAULT_TILE;
                const int64_t th = p->tile_h > 0 ? p->tile_h : DEFAULT_TILE;
                bool pin_ok = true;
                for (const auto &pin : aji_onnx_pin_dims(oin, {1, 3, th, tw}))
                    if (!c->ck(g_ort->AddFreeDimensionOverrideByName(
                                   so, pin.first.c_str(), pin.second),
                               "AddFreeDimensionOverrideByName", &e)) {
                        pin_ok = false;
                        break;
                    }
                if (!pin_ok)
                    break;
            }
        }
        if (!c->ck(g_dml_api->SessionOptionsAppendExecutionProvider_DML1(
                       so, c->dml_dev.Get(), c->queue.Get()),
                   "DML EP append", &e))
            break;
        const std::wstring wpath = widen_utf8(p->onnx_path);
        if (!c->ck(g_ort->CreateSession(c->env, wpath.c_str(), so, &m->session),
                   "failed to load " + std::string(p->onnx_path), &e))
            break;

        size_t n_in = 0, n_out = 0;
        g_ort->SessionGetInputCount(m->session, &n_in);
        g_ort->SessionGetOutputCount(m->session, &n_out);
        if (n_in != 1 || n_out != 1) {
            e = m->name + ": expected 1 input and 1 output";
            break;
        }
        OrtAllocator *alloc = nullptr;
        g_ort->GetAllocatorWithDefaultOptions(&alloc);
        char *nm = nullptr;
        g_ort->SessionGetInputName(m->session, 0, alloc, &nm);
        m->in_name = nm;
        alloc->Free(alloc, nm);
        g_ort->SessionGetOutputName(m->session, 0, alloc, &nm);
        m->out_name = nm;
        alloc->Free(alloc, nm);

        OrtTypeInfo *ti = nullptr;
        const OrtTensorTypeAndShapeInfo *tti = nullptr;
        if (!c->ck(g_ort->SessionGetInputTypeInfo(m->session, 0, &ti),
                   "input type", &e))
            break;
        g_ort->CastTypeInfoToTensorInfo(ti, &tti);
        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        int64_t dims[4] = {0};
        size_t nd = 0;
        g_ort->GetTensorElementType(tti, &et);
        g_ort->GetDimensionsCount(tti, &nd);
        if (nd == 4)
            g_ort->GetDimensions(tti, dims, 4);
        g_ort->ReleaseTypeInfo(ti);
        if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            e = m->name + ": DirectML needs an fp32 or fp16 model input";
            break;
        }
        if (nd != 4 || (dims[1] > 0 && dims[1] != 1 && dims[1] != 3)) {
            e = m->name + ": expected one NCHW input with 1 or 3 channels";
            break;
        }
        m->fp16 = et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        m->in_ch = dims[1] > 0 ? (int)dims[1] : 3;
        if (dims[2] > 0 && dims[3] > 0) {
            m->tile_h = (int)dims[2];
            m->tile_w = (int)dims[3];
        } else {
            m->tile_w = p->tile_w > 0 ? p->tile_w : DEFAULT_TILE;
            m->tile_h = p->tile_h > 0 ? p->tile_h : DEFAULT_TILE;
        }
        m->overlap = std::min(p->overlap > 0 ? p->overlap : DEFAULT_OVERLAP,
                              std::min(m->tile_w, m->tile_h) / 2);
        if (!discover(m, &e))
            break;
        const size_t ni = (size_t)m->in_ch * m->tile_w * m->tile_h;
        const size_t no = (size_t)m->out_ch * m->tile_w * m->scale * m->tile_h * m->scale;
        try {
            if (m->fp16) {
                m->in16.resize(ni);
                m->out16.resize(no);
            } else {
                m->in32.resize(ni);
                m->out32.resize(no);
            }
        } catch (const std::bad_alloc &) {
            e = m->name + ": out of host memory for tile buffers";
            break;
        }
        ok = true;
    } while (0);
    if (so)
        g_ort->ReleaseSessionOptions(so);
    if (!ok) {
        c->set_error(e);
        release_model(m);
        return nullptr;
    }
    c->models.insert(m);
    return m;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_state(aji_img_model *m)
{
    return m ? AJI_IMG_STATE_READY : AJI_IMG_STATE_FAILED;
}

extern "C" AJI_IMG_EXPORT int aji_img_model_info_get(aji_img_model *m,
                                                     aji_img_model_info *info)
{
    if (!m || !info)
        return AJI_IMG_ERR_ARGS;
    info->scale = m->scale;
    info->in_channels = m->in_ch;
    info->out_channels = m->out_ch;
    info->tile_w = m->tile_w;
    info->tile_h = m->tile_h;
    info->overlap = m->overlap;
    return AJI_IMG_OK;
}

extern "C" AJI_IMG_EXPORT int aji_img_upscale(aji_img_model *m,
                                              const aji_img_buffer *in,
                                              const aji_img_buffer *out,
                                              aji_img_progress_fn progress,
                                              void *opaque)
{
    if (!m)
        return AJI_IMG_ERR_ARGS;
    std::lock_guard<std::mutex> run(m->run_mu);
    std::string e;
    const int rc = upscale_tiled(*m, m->overlap, in, out, progress, opaque, &e);
    if (rc != AJI_IMG_OK && rc != AJI_IMG_CANCELLED) {
        std::lock_guard<std::mutex> lock(m->c->mu);
        m->c->set_error(e);
    }
    return rc;
}

extern "C" AJI_IMG_EXPORT void aji_img_release(aji_img_model **pm)
{
    if (!pm || !*pm)
        return;
    aji_img_model *m = *pm;
    {
        std::lock_guard<std::mutex> lock(m->c->mu);
        m->c->models.erase(m);
    }
    {
        std::lock_guard<std::mutex> run(m->run_mu);  // let a running upscale finish
    }
    release_model(m);
    *pm = nullptr;
}

extern "C" AJI_IMG_EXPORT const char *aji_img_last_error(aji_img_ctx *c)
{
    return c ? c->err.c_str() : "no context";
}

extern "C" AJI_IMG_EXPORT void aji_img_destroy(aji_img_ctx **pc)
{
    if (!pc || !*pc)
        return;
    aji_img_ctx *c = *pc;
    std::vector<aji_img_model *> left(c->models.begin(), c->models.end());
    for (aji_img_model *m : left)
        aji_img_release(&m);
    if (c->mi_cpu)
        g_ort->ReleaseMemoryInfo(c->mi_cpu);
    if (c->env)
        g_ort->ReleaseEnv(c->env);
    delete c;
    *pc = nullptr;
}
