/*
 * dml_spike.cpp — Phase 3b spike: validate ONNX Runtime 1.24.4 + DirectML
 * 1.15.4 on AnimeJaNai models and measure steady-state throughput.
 *
 * CPU-tensor path, i.e. exactly the session recipe vs-mlrt's vsort ships
 * for ORT_DML (ORT_SEQUENTIAL + DisableMemPattern + OrtDmlApi EP append,
 * first inference run twice). The D3D12 zero-copy path comes later; this
 * answers "does the stack run our models and how fast".
 *
 * Usage:
 *   dml_spike <model.onnx> <width> <height> [iters] [--device N]
 *             [--graph-capture] [--fill16 v] [--alternate]
 *
 * --alternate builds a second input set with a different fill, swaps
 * inputs every run in the timed loop, then asserts the output sample
 * tracks whichever input was last run (catches stale outputs from a
 * replayed graph that ignores fresh bindings).
 *
 * Dynamic input dims resolve as: dim0 -> 1, last two -> H, W; others 1.
 * Inputs fill with a blocky gradient (fp16 or fp32 per model IO type).
 * Reports session-create / run1 / run2 / steady ms per frame + output
 * sanity stats (min/max/NaN over a sample).
 */

#include <windows.h>
#include <dxgi.h>

#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const OrtApi *ort;

static void ck(OrtStatus *st, const char *what)
{
    if (!st)
        return;
    fprintf(stderr, "FAIL %s: %s\n", what, ort->GetErrorMessage(st));
    ort->ReleaseStatus(st);
    exit(1);
}

static double qpc_ms(LARGE_INTEGER a, LARGE_INTEGER b, LARGE_INTEGER f)
{
    return (double)(b.QuadPart - a.QuadPart) * 1000.0 / (double)f.QuadPart;
}

static uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0)
        return (uint16_t)sign;            /* flush to zero, fine for fills */
    if (exp >= 31)
        return (uint16_t)(sign | 0x7c00); /* inf */
    return (uint16_t)(sign | (exp << 10) | (man >> 13));
}

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t x;
    if (exp == 0) {
        if (!man) {
            x = sign;
        } else {                          /* subnormal */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            x = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000 | (man << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static void print_adapter(int device_id)
{
    IDXGIFactory1 *fac = NULL;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&fac)))
        return;
    IDXGIAdapter1 *ad = NULL;
    if (SUCCEEDED(fac->EnumAdapters1(device_id, &ad))) {
        DXGI_ADAPTER_DESC1 d;
        ad->GetDesc1(&d);
        printf("adapter %d: %ls (%.1f GB dedicated)\n", device_id,
               d.Description, d.DedicatedVideoMemory / 1073741824.0);
        ad->Release();
    }
    fac->Release();
}

struct IoDesc {
    char *name;
    ONNXTensorElementDataType type;
    std::vector<int64_t> dims;
    size_t count;
};

static IoDesc describe(OrtSession *s, OrtAllocator *alloc, size_t i,
                       bool input, int W, int H)
{
    IoDesc d = {};
    OrtTypeInfo *ti;
    const OrtTensorTypeAndShapeInfo *tti;
    ck(input ? ort->SessionGetInputTypeInfo(s, i, &ti)
             : ort->SessionGetOutputTypeInfo(s, i, &ti), "GetTypeInfo");
    ck(ort->CastTypeInfoToTensorInfo(ti, &tti), "CastTypeInfo");
    ck(ort->GetTensorElementType(tti, &d.type), "GetTensorElementType");
    size_t nd;
    ck(ort->GetDimensionsCount(tti, &nd), "GetDimensionsCount");
    d.dims.resize(nd);
    ck(ort->GetDimensions(tti, d.dims.data(), nd), "GetDimensions");
    ck(input ? ort->SessionGetInputName(s, i, alloc, &d.name)
             : ort->SessionGetOutputName(s, i, alloc, &d.name), "GetName");

    d.count = 1;
    for (size_t j = 0; j < nd; j++) {
        if (d.dims[j] < 0) {
            if (j == 0)
                d.dims[j] = 1;
            else if (j == nd - 1)
                d.dims[j] = W;
            else if (j == nd - 2)
                d.dims[j] = H;
            else
                d.dims[j] = 1;
        }
        d.count *= (size_t)d.dims[j];
    }
    return d;
}

static const char *type_name(ONNXTensorElementDataType t)
{
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "fp32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "fp16";
    default: return "other";
    }
}

static void refill(OrtValue *v, const IoDesc &d, float fill)
{
    void *p;
    ck(ort->GetTensorMutableData(v, &p), "GetTensorMutableData");
    /* blocky gradient: 3 levels around the fill value */
    if (d.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        uint16_t lv[3] = { float_to_half(fill * 0.5f), float_to_half(fill),
                           float_to_half(fill * 1.5f) };
        uint16_t *q = (uint16_t *)p;
        for (size_t k = 0; k < d.count; k++)
            q[k] = lv[(k >> 6) % 3];
    } else if (d.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float lv[3] = { fill * 0.5f, fill, fill * 1.5f };
        float *q = (float *)p;
        for (size_t k = 0; k < d.count; k++)
            q[k] = lv[(k >> 6) % 3];
    } else {
        memset(p, 0, d.count);
    }
}

static OrtValue *make_filled(OrtAllocator *alloc, const IoDesc &d, float fill)
{
    OrtValue *v = nullptr;
    ck(ort->CreateTensorAsOrtValue(alloc, d.dims.data(), d.dims.size(),
                                   d.type, &v), "CreateTensor(in)");
    refill(v, d, fill);
    return v;
}

static void sample_stats(OrtValue *v, double *mn, double *mx, size_t *nan)
{
    OrtTensorTypeAndShapeInfo *info;
    ck(ort->GetTensorTypeAndShape(v, &info), "shape");
    ONNXTensorElementDataType t;
    size_t count;
    ck(ort->GetTensorElementType(info, &t), "type");
    ck(ort->GetTensorShapeElementCount(info, &count), "count");
    ort->ReleaseTensorTypeAndShapeInfo(info);
    void *p;
    ck(ort->GetTensorMutableData(v, &p), "data");
    *mn = 1e30; *mx = -1e30; *nan = 0;
    size_t step = count > 1000000 ? count / 1000000 : 1;
    for (size_t k = 0; k < count; k += step) {
        float x = t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                      ? half_to_float(((uint16_t *)p)[k])
                      : ((float *)p)[k];
        if (x != x) { (*nan)++; continue; }
        if (x < *mn) *mn = x;
        if (x > *mx) *mx = x;
    }
}

int wmain(int argc, wchar_t **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: dml_spike <model.onnx> <W> <H> [iters] "
                        "[--device N] [--graph-capture] [--fill16 v]\n");
        return 2;
    }
    const wchar_t *model = argv[1];
    int W = _wtoi(argv[2]), H = _wtoi(argv[3]);
    int iters = 100, device = 0, graph_capture = 0, alternate = 0;
    float fill = 0.5f;
    for (int i = 4; i < argc; i++) {
        if (!wcscmp(argv[i], L"--device") && i + 1 < argc)
            device = _wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--graph-capture"))
            graph_capture = 1;
        else if (!wcscmp(argv[i], L"--alternate"))
            alternate = 1;
        else if (!wcscmp(argv[i], L"--fill16") && i + 1 < argc)
            fill = (float)_wtof(argv[++i]);
        else if (iswdigit(argv[i][0]))
            iters = _wtoi(argv[i]);
    }

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    printf("onnxruntime %s, ORT_API_VERSION %d\n",
           OrtGetApiBase()->GetVersionString(), ORT_API_VERSION);
    print_adapter(device);
    printf("model: %ls\n", model);
    printf("resolve dynamic dims with W=%d H=%d, iters=%d, graph_capture=%d\n",
           W, H, iters, graph_capture);

    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);

    OrtEnv *env;
    ck(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dml_spike", &env), "CreateEnv");

    OrtSessionOptions *so;
    ck(ort->CreateSessionOptions(&so), "CreateSessionOptions");
    ck(ort->SetSessionExecutionMode(so, ORT_SEQUENTIAL), "SetExecutionMode");
    ck(ort->DisableMemPattern(so), "DisableMemPattern");
    ck(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL), "SetOptLevel");
    if (graph_capture)
        ck(ort->AddSessionConfigEntry(so, "ep.dml.enable_graph_capture", "1"),
           "enable_graph_capture");

    const OrtDmlApi *dml;
    ck(ort->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dml),
       "GetExecutionProviderApi(DML)");
    ck(dml->SessionOptionsAppendExecutionProvider_DML(so, device), "Append_DML");

    QueryPerformanceCounter(&t0);
    OrtSession *session;
    ck(ort->CreateSession(env, model, so, &session), "CreateSession");
    QueryPerformanceCounter(&t1);
    printf("session created in %.0f ms\n", qpc_ms(t0, t1, f));

    OrtAllocator *alloc;
    ck(ort->GetAllocatorWithDefaultOptions(&alloc), "GetAllocator");

    size_t n_in, n_out;
    ck(ort->SessionGetInputCount(session, &n_in), "GetInputCount");
    ck(ort->SessionGetOutputCount(session, &n_out), "GetOutputCount");

    std::vector<IoDesc> ins, outs;
    std::vector<const char *> in_names, out_names;
    std::vector<OrtValue *> in_vals(n_in, nullptr), in_vals_b(n_in, nullptr),
                            out_vals(n_out, nullptr);

    for (size_t i = 0; i < n_in; i++) {
        IoDesc d = describe(session, alloc, i, true, W, H);
        printf("input  %zu '%s' %s [", i, d.name, type_name(d.type));
        for (size_t j = 0; j < d.dims.size(); j++)
            printf("%s%lld", j ? "x" : "", (long long)d.dims[j]);
        printf("] (%zu elems)\n", d.count);

        in_vals[i] = make_filled(alloc, d, fill);
        if (alternate)
            in_vals_b[i] = make_filled(alloc, d, fill * 0.3f);
        in_names.push_back(d.name);
        ins.push_back(std::move(d));
    }
    for (size_t i = 0; i < n_out; i++) {
        IoDesc d = describe(session, alloc, i, false, W, H);
        out_names.push_back(d.name);
        outs.push_back(std::move(d));
    }

    /* run 1: let ORT allocate outputs (also resolves true output shapes) */
    QueryPerformanceCounter(&t0);
    ck(ort->Run(session, NULL, in_names.data(), in_vals.data(), n_in,
                out_names.data(), n_out, out_vals.data()), "Run #1");
    QueryPerformanceCounter(&t1);
    printf("run #1 (alloc+warmup) %.1f ms\n", qpc_ms(t0, t1, f));

    for (size_t i = 0; i < n_out; i++) {
        OrtTensorTypeAndShapeInfo *info;
        ck(ort->GetTensorTypeAndShape(out_vals[i], &info), "GetOutShape");
        ONNXTensorElementDataType t;
        size_t nd;
        ck(ort->GetTensorElementType(info, &t), "out type");
        ck(ort->GetDimensionsCount(info, &nd), "out ndims");
        std::vector<int64_t> dims(nd);
        ck(ort->GetDimensions(info, dims.data(), nd), "out dims");
        printf("output %zu '%s' %s [", i, out_names[i], type_name(t));
        for (size_t j = 0; j < nd; j++)
            printf("%s%lld", j ? "x" : "", (long long)dims[j]);
        printf("]\n");
        ort->ReleaseTensorTypeAndShapeInfo(info);
    }

    /* run 2: vsort's "replay the first dml execution" */
    QueryPerformanceCounter(&t0);
    ck(ort->Run(session, NULL, in_names.data(), in_vals.data(), n_in,
                out_names.data(), n_out, out_vals.data()), "Run #2");
    QueryPerformanceCounter(&t1);
    printf("run #2 (replay)       %.1f ms\n", qpc_ms(t0, t1, f));

    /* steady state with pre-allocated outputs (reused across runs);
     * --alternate swaps between two input tensor sets every run */
    QueryPerformanceCounter(&t0);
    for (int k = 0; k < iters; k++) {
        OrtValue **iv = (alternate && (k & 1)) ? in_vals_b.data()
                                               : in_vals.data();
        ck(ort->Run(session, NULL, in_names.data(), iv, n_in,
                    out_names.data(), n_out, out_vals.data()), "Run steady");
    }
    QueryPerformanceCounter(&t1);
    double ms = qpc_ms(t0, t1, f) / iters;
    printf("steady: %.2f ms/frame -> %.1f fps (%d iters%s)\n", ms, 1000.0 / ms,
           iters, alternate ? ", alternating inputs" : "");

    double mn, mx;
    size_t nan;
    sample_stats(out_vals[0], &mn, &mx, &nan);
    printf("output sample: min %.4f max %.4f nan %zu\n", mn, mx, nan);

    if (alternate) {
        /* output must track whichever input ran last */
        double mn_a, mx_a, mn_b, mx_b;
        ck(ort->Run(session, NULL, in_names.data(), in_vals.data(), n_in,
                    out_names.data(), n_out, out_vals.data()), "Run A");
        sample_stats(out_vals[0], &mn_a, &mx_a, &nan);
        ck(ort->Run(session, NULL, in_names.data(), in_vals_b.data(), n_in,
                    out_names.data(), n_out, out_vals.data()), "Run B");
        sample_stats(out_vals[0], &mn_b, &mx_b, &nan);
        printf("verify swap-tensor: A [%.4f, %.4f] vs B [%.4f, %.4f] -> %s\n",
               mn_a, mx_a, mn_b, mx_b,
               (mn_a != mn_b || mx_a != mx_b) ? "OUTPUT TRACKS INPUT (ok)"
                                              : "STALE OUTPUT (BAD)");

        /* same tensor objects, contents rewritten in place — our real
         * per-frame usage pattern */
        for (size_t i = 0; i < n_in; i++)
            refill(in_vals[i], ins[i], fill * 0.3f);
        double mn_c, mx_c;
        ck(ort->Run(session, NULL, in_names.data(), in_vals.data(), n_in,
                    out_names.data(), n_out, out_vals.data()), "Run inplace");
        sample_stats(out_vals[0], &mn_c, &mx_c, &nan);
        int inplace_ok = mn_c != mn_a || mx_c != mx_a;
        printf("verify in-place:    A [%.4f, %.4f] vs A' [%.4f, %.4f] -> %s\n",
               mn_a, mx_a, mn_c, mx_c,
               inplace_ok ? "OUTPUT TRACKS INPUT (ok)" : "STALE OUTPUT (BAD)");
        if ((mn_a == mn_b && mx_a == mx_b) || !inplace_ok)
            return 1;
    }

    printf("DML-SPIKE-OK\n");
    return 0;
}
