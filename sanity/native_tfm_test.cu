// aji_native_tfm_test: native/tfm.h kernels against CPU references (fp32 maths on the same fp16
// inputs): GEMM with every epilogue, LayerNorm, window attention (bias, image sizes that are not
// window multiples, a cyclic shift, 64- and 256-token windows), depthwise 3x3 with GELU, channel
// sums and channel-attention statistics, the fused MLP (with and without its LayerNorm). Skips
// (exit 0) without an sm_80+ GPU.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "native/pack.h"
#include "native/tfm.h"
#include "native/dat_kernels.h"

using namespace ajinative;

static std::mt19937 rng(7);
static float ur(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }
static uint16_t hb(float f) { return float_to_half_bits(f); }
static float hf(uint16_t b) { return half_bits_to_float(b); }
static float gelu(float v) { return 0.5f * v * (1.f + std::erf(v / 1.4140625f)); }
static int fails = 0, checks = 0;
static cudaError_t cuda_err = cudaSuccess;  // the first CUDA error: every later check fails with it

static void ck(cudaError_t e)
{
    if (e != cudaSuccess && cuda_err == cudaSuccess) cuda_err = e;
}
static void sync()
{
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());
}
// A running maximum error that a NaN or an infinity poisons (std::max(me, NaN) keeps me).
static double emax(double me, double e) { return std::isfinite(e) ? std::max(me, e) : INFINITY; }

static void report(const char *name, double maxe, double lim)
{
    checks++;
    if (cuda_err != cudaSuccess) {
        fails++;
        printf("FAIL %s: %s\n", name, cudaGetErrorString(cuda_err));
    } else if (!(maxe <= lim)) {
        fails++;
        printf("FAIL %s: max err %.5f (limit %.5f)\n", name, maxe, lim);
    }
}
template <typename T> T *dev(const std::vector<T> &h)
{
    T *d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)));
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> std::vector<T> host(const T *d, size_t n)
{
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static void test_gemm(int M, int K, int N, bool bias, bool gelu_, bool scale, bool add1, bool add2, bool lrelu_last = false)
{
    const int Kp = (K + 15) / 16 * 16, Np = (N + 15) / 16 * 16, sa = Kp + 8, so = Np + 16;
    std::vector<uint16_t> a((size_t)M * sa, 0), a1((size_t)M * so), a2((size_t)M * so);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) a[(size_t)m * sa + k] = hb(ur(1.f));
    for (auto &v : a1) v = hb(ur(1.f));
    for (auto &v : a2) v = hb(ur(1.f));
    std::vector<float> w((size_t)K * N), b(N), cs(Np, 0.f);
    for (auto &v : w) v = hf(hb(ur(0.1f)));
    for (auto &v : b) v = ur(0.3f);
    for (int n = 0; n < N; n++) cs[n] = ur(1.f);
    TMat tm;
    ck(tmat_upload(tm, w.data(), bias ? b.data() : nullptr, K, N));
    uint16_t *da = dev(a), *d1 = dev(a1), *d2 = dev(a2), *dout = dev(std::vector<uint16_t>((size_t)M * so, 0));
    float *dcs = dev(cs);
    TGemm g;
    g.a = da; g.sa = sa; g.M = M; g.out = dout; g.so = so; g.act = gelu_ ? TAct::Gelu : TAct::None;
    g.colscale = scale ? dcs : nullptr;
    if (add1) { g.add1 = d1; g.s1 = so; }
    if (add2) { g.add2 = d2; g.s2 = so; }
    if (lrelu_last) { g.act = TAct::LRelu; g.act_last = true; g.slope = 0.2f; }
    ck(tgemm(tm, g, nullptr));
    sync();
    auto o = host(dout, (size_t)M * so);
    double me = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double v = bias ? b[n] : 0;
            for (int k = 0; k < K; k++) v += hf(a[(size_t)m * sa + k]) * w[(size_t)k * N + n];
            if (gelu_) v = gelu((float)v);
            if (scale) v *= cs[n];
            if (add1) v += hf(a1[(size_t)m * so + n]);
            if (add2) v += hf(a2[(size_t)m * so + n]);
            if (lrelu_last && v < 0) v *= 0.2;
            me = emax(me, std::fabs(v - hf(o[(size_t)m * so + n])));
        }
    char nm[96];
    snprintf(nm, sizeof(nm), "gemm %dx%dx%d%s%s%s%s%s%s", M, K, N, lrelu_last ? " lrelu-last" : "", bias ? " bias" : "", gelu_ ? " gelu" : "", scale ? " scale" : "",
             add1 ? " add1" : "", add2 ? " add2" : "");
    report(nm, me, 0.02);
    tmat_release(tm);
    cudaFree(da); cudaFree(d1); cudaFree(d2); cudaFree(dout); cudaFree(dcs);
}

// The fused MLP against the two GEMMs it replaces (fc1 + GELU to an fp16 hidden, then fc2 with the
// residual): the same values bit for bit, fp32 stream in place or fp16 out; and against the CPU.
static void test_mlp(int M, int C, int hid, bool f32)
{
    const int Cp = (C + 15) / 16 * 16, Hp = (hid + 15) / 16 * 16;
    std::vector<uint16_t> a((size_t)M * Cp, 0);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < C; k++) a[(size_t)m * Cp + k] = hb(ur(2.f));
    std::vector<float> w1((size_t)C * hid), b1(hid), w2((size_t)hid * C), b2(C), res((size_t)M * Cp, 0.f);
    for (auto &v : w1) v = hf(hb(ur(0.15f)));
    for (auto &v : w2) v = hf(hb(ur(0.1f)));
    for (auto &v : b1) v = ur(0.5f);
    for (auto &v : b2) v = ur(0.3f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < C; n++) res[(size_t)m * Cp + n] = f32 ? ur(1000.f) : hf(hb(ur(4.f)));
    TMat m1, m2;
    ck(tmat_upload(m1, w1.data(), b1.data(), C, hid));
    ck(tmat_upload(m2, w2.data(), b2.data(), hid, C));
    std::vector<uint16_t> res16(res.size());
    for (size_t i = 0; i < res.size(); i++) res16[i] = hb(res[i]);
    const size_t es = f32 ? 4 : 2, n = (size_t)M * Cp;
    uint16_t *da = dev(a), *dh = dev(std::vector<uint16_t>((size_t)M * Hp, 0));
    void *dr = f32 ? (void *)dev(res) : (void *)dev(res16), *o1 = nullptr, *o2 = nullptr;
    ck(cudaMalloc(&o1, n * es));
    ck(cudaMalloc(&o2, n * es));
    ck(cudaMemcpy(o1, dr, n * es, cudaMemcpyDeviceToDevice));  // fp32: in place, as HAT's stream
    ck(cudaMemcpy(o2, dr, n * es, cudaMemcpyDeviceToDevice));
    TGemm g1;
    g1.a = da; g1.sa = Cp; g1.M = M; g1.out = dh; g1.so = Hp; g1.act = TAct::Gelu;
    TGemm g2;
    g2.a = dh; g2.sa = Hp; g2.M = M; g2.out = o1; g2.so = Cp; g2.out_f32 = f32;
    g2.add1 = f32 ? o1 : dr; g2.s1 = Cp; g2.add1_f32 = f32;
    ck(tgemm(m1, g1, nullptr));
    ck(tgemm(m2, g2, nullptr));
    TMlp t;
    t.a = da; t.sa = Cp; t.M = M; t.out = o2; t.so = Cp; t.out_f32 = f32;
    t.add1 = f32 ? o2 : dr; t.s1 = Cp; t.add1_f32 = f32;
    const cudaError_t e = tmlp(m1, m2, t, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "mlp %dx%dx%d%s", M, C, hid, f32 ? " f32" : "");
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", nm, cudaGetErrorString(e));
        fails++;
        checks++;
    } else {
        std::vector<uint8_t> h1(n * es), h2(n * es);
        ck(cudaMemcpy(h1.data(), o1, n * es, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(h2.data(), o2, n * es, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < h1.size(); i += es) diff += memcmp(&h1[i], &h2[i], es) != 0;
        char nb[120];
        snprintf(nb, sizeof(nb), "%s vs gemm+gemm (values differing)", nm);
        report(nb, (double)diff, 0);
        double me = 0;
        std::vector<float> hv(Hp);
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < hid; j++) {
                double v = b1[j];
                for (int k = 0; k < C; k++) v += hf(a[(size_t)m * Cp + k]) * w1[(size_t)k * hid + j];
                hv[j] = hf(hb(gelu((float)v)));
            }
            for (int c = 0; c < C; c++) {
                double v = b2[c] + res[(size_t)m * Cp + c];
                for (int j = 0; j < hid; j++) v += hv[j] * w2[(size_t)j * C + c];
                const size_t i = (size_t)m * Cp + c;
                const double got = f32 ? (double)reinterpret_cast<const float *>(h2.data())[i]
                                       : (double)hf(reinterpret_cast<const uint16_t *>(h2.data())[i]);
                me = emax(me, std::fabs(v - got) / (f32 ? 1.0 : std::max(1.0, std::fabs(v))));
            }
        }
        report(nm, me, f32 ? 2e-3 : 4e-3);
    }
    tmat_release(m1);
    tmat_release(m2);
    cudaFree(da); cudaFree(dh); cudaFree(dr); cudaFree(o1); cudaFree(o2);
}

// The fused MLP with its LayerNorm (fp32 stream in place, as HAT's) against tlayernorm2 then the
// fused MLP without it (the same maths in another order: a few hidden values may round the other way)
// and against the CPU.
static void test_mlp_ln(int M, int C, int hid)
{
    const int Cp = (C + 15) / 16 * 16;
    std::vector<float> w1((size_t)C * hid), b1(hid), w2((size_t)hid * C), b2(C), x((size_t)M * Cp, 0.f), gm(C), bt(C);
    for (auto &v : w1) v = hf(hb(ur(0.15f)));
    for (auto &v : w2) v = hf(hb(ur(0.1f)));
    for (auto &v : b1) v = ur(0.5f);
    for (auto &v : b2) v = ur(0.3f);
    for (auto &v : gm) v = 0.5f + ur(0.4f);
    for (auto &v : bt) v = ur(0.2f);
    for (int m = 0; m < M; m++) {
        const float off = ur(300.f), sc = 1.f + std::fabs(ur(200.f));  // a residual stream's spread
        for (int n = 0; n < C; n++) x[(size_t)m * Cp + n] = off + sc * ur(1.f);
    }
    TMat m1, m2;
    ck(tmat_upload(m1, w1.data(), b1.data(), C, hid));
    ck(tmat_upload(m2, w2.data(), b2.data(), hid, C));
    float *o1 = dev(x), *o2 = dev(x), *dg = dev(gm), *db = dev(bt);
    uint16_t *dn = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    ck(tlayernorm2(o1, true, Cp, dn, false, Cp, M, C, Cp, dg, db, 1e-5f, nullptr));
    TMlp t;
    t.a = dn; t.sa = Cp; t.M = M; t.out = o1; t.so = Cp; t.out_f32 = true; t.add1 = o1; t.s1 = Cp; t.add1_f32 = true;
    const cudaError_t e1 = tmlp(m1, m2, t, nullptr);
    t.a = o2; t.out = o2; t.add1 = o2; t.ln_gamma = dg; t.ln_beta = db; t.ln_eps = 1e-5f;
    uint16_t *dlo = dev(std::vector<uint16_t>((size_t)M * Cp, 0x7c00)), *dlr = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    t.lno = dlo; t.slno = Cp; t.lno_gamma = db; t.lno_beta = dg; t.lno_eps = 1e-3f;  // the next LayerNorm (other parameters)
    const cudaError_t e2 = tmlp(m1, m2, t, nullptr);
    ck(tlayernorm2(o2, true, Cp, dlr, false, Cp, M, C, Cp, db, dg, 1e-3f, nullptr));
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "mlp+ln %dx%dx%d", M, C, hid);
    checks++;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("FAIL %s: %s / %s\n", nm, cudaGetErrorString(e1), cudaGetErrorString(e2));
        fails++;
    } else {
        auto h1 = host(o1, x.size()), h2 = host(o2, x.size());
        double md = 0, me = 0;
        std::vector<double> y(C);
        std::vector<float> hv(hid);
        for (int m = 0; m < M; m++) {
            double mean = 0, var = 0;
            for (int n = 0; n < C; n++) mean += x[(size_t)m * Cp + n];
            mean /= C;
            for (int n = 0; n < C; n++) var += (x[(size_t)m * Cp + n] - mean) * (x[(size_t)m * Cp + n] - mean);
            const double rs = 1.0 / std::sqrt(var / C + 1e-5);
            for (int n = 0; n < C; n++) y[n] = hf(hb((float)((x[(size_t)m * Cp + n] - mean) * rs * gm[n] + bt[n])));
            for (int j = 0; j < hid; j++) {
                double v = b1[j];
                for (int k = 0; k < C; k++) v += y[k] * w1[(size_t)k * hid + j];
                hv[j] = hf(hb(gelu((float)v)));
            }
            for (int c = 0; c < C; c++) {
                double v = b2[c] + x[(size_t)m * Cp + c];
                for (int j = 0; j < hid; j++) v += hv[j] * w2[(size_t)j * C + c];
                const size_t i = (size_t)m * Cp + c;
                me = emax(me, std::fabs(v - h2[i]) / std::max(1.0, std::fabs(v) * 1e-3));
                md = emax(md, (double)std::fabs(h1[i] - h2[i]) / std::max(1.0, std::fabs((double)h1[i]) * 1e-3));
            }
        }
        report(nm, me, 0.1);
        char nb[120];
        snprintf(nb, sizeof(nb), "%s vs layernorm + mlp", nm);
        report(nb, md, 0.05);
        auto l1 = host(dlo, (size_t)M * Cp), l2 = host(dlr, (size_t)M * Cp);
        double le = 0;
        for (size_t i = 0; i < l1.size(); i++) le = emax(le, (double)std::fabs(hf(l1[i]) - hf(l2[i])));
        snprintf(nb, sizeof(nb), "%s next layernorm vs tlayernorm2", nm);
        report(nb, le, 0.01);
    }
    tmat_release(m1);
    tmat_release(m2);
    cudaFree(o1); cudaFree(o2); cudaFree(dg); cudaFree(db); cudaFree(dn); cudaFree(dlo); cudaFree(dlr);
}

// tgemm with its LayerNorm on the A load (fp16 rows, as FDAT's fc1) against tlayernorm then tgemm.
static void test_gemm_ln(int M, int K, int N)
{
    const int Kp = (K + 15) / 16 * 16, Np = (N + 15) / 16 * 16;
    std::vector<uint16_t> a((size_t)M * Kp, 0);
    for (int m = 0; m < M; m++) {
        const float off = ur(3.f), sc = 0.5f + std::fabs(ur(4.f));
        for (int k = 0; k < K; k++) a[(size_t)m * Kp + k] = hb(off + sc * ur(1.f));
    }
    std::vector<float> w((size_t)K * N), gm(K), bt(K);
    for (auto &v : w) v = hf(hb(ur(0.1f)));
    for (auto &v : gm) v = 0.5f + ur(0.4f);
    for (auto &v : bt) v = ur(0.2f);
    TMat tm;
    ck(tmat_upload(tm, w.data(), nullptr, K, N));
    uint16_t *da = dev(a), *dn = dev(std::vector<uint16_t>((size_t)M * Kp, 0)), *o1 = dev(std::vector<uint16_t>((size_t)M * Np, 0)),
             *o2 = dev(std::vector<uint16_t>((size_t)M * Np, 0));
    float *dg = dev(gm), *db = dev(bt);
    ck(tlayernorm(da, Kp, dn, Kp, M, K, Kp, dg, db, 1e-5f, nullptr));
    TGemm g;
    g.a = dn; g.sa = Kp; g.M = M; g.out = o1; g.so = Np; g.act = TAct::Gelu;
    ck(tgemm(tm, g, nullptr));
    g.a = da; g.out = o2; g.ln_gamma = dg; g.ln_beta = db; g.ln_eps = 1e-5f;
    const cudaError_t e = tgemm(tm, g, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "gemm+ln %dx%dx%d vs layernorm + gemm", M, K, N);
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", nm, cudaGetErrorString(e));
        fails++;
        checks++;
    } else {
        auto h1 = host(o1, (size_t)M * Np), h2 = host(o2, (size_t)M * Np);
        double me = 0;
        for (size_t i = 0; i < h1.size(); i++) me = emax(me, (double)std::fabs(hf(h1[i]) - hf(h2[i])));
        report(nm, me, 0.02);
    }
    tmat_release(tm);
    cudaFree(da); cudaFree(dn); cudaFree(o1); cudaFree(o2); cudaFree(dg); cudaFree(db);
}

// tdwgemm (FDAT's smix then fc2, in place on the stream) against tdwconv then tgemm: the same values.
static void test_dwgemm(int H, int W, int K, int N)
{
    const int Kp = (K + 15) / 16 * 16, Np = (N + 15) / 16 * 16, M = H * W;
    std::vector<uint16_t> h((size_t)M * Kp, 0), x((size_t)M * Np, 0);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) h[(size_t)m * Kp + k] = hb(ur(2.f));
        for (int n = 0; n < N; n++) x[(size_t)m * Np + n] = hb(ur(8.f));
    }
    std::vector<float> w((size_t)K * N), dw((size_t)K * 9);
    for (auto &v : w) v = hf(hb(ur(0.1f)));
    for (auto &v : dw) v = ur(0.4f);
    TMat tm;
    ck(tmat_upload(tm, w.data(), nullptr, K, N));
    uint16_t *dh = dev(h), *d2 = dev(std::vector<uint16_t>((size_t)M * Kp, 0)), *o1 = dev(x), *o2 = dev(x);
    float *ddw = dev(dw);
    ck(tdwconv(dh, Kp, d2, Kp, H, W, K, ddw, false, nullptr));
    TGemm g;
    g.a = d2; g.sa = Kp; g.M = M; g.out = o1; g.so = Np; g.add1 = o1; g.s1 = Np;
    ck(tgemm(tm, g, nullptr));
    TDwGemm dg;
    dg.in = dh; dg.sin = Kp; dg.H = H; dg.W = W; dg.dw = ddw; dg.out = o2; dg.so = Np; dg.add1 = o2; dg.s1 = Np;
    const cudaError_t e = tdwgemm(tm, dg, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "dwgemm %dx%d %d->%d vs dwconv + gemm (values differing)", H, W, K, N);
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", nm, cudaGetErrorString(e));
        fails++;
        checks++;
    } else {
        auto h1 = host(o1, (size_t)M * Np), h2 = host(o2, (size_t)M * Np);
        size_t diff = 0;
        for (size_t i = 0; i < h1.size(); i++) diff += h1[i] != h2[i];
        report(nm, (double)diff, 0);
    }
    tmat_release(tm);
    cudaFree(dh); cudaFree(d2); cudaFree(o1); cudaFree(o2); cudaFree(ddw);
}

// tdwgemm in DAT's SGFN form (depthwise + bias, times the gate, fp32 stream in place) against tdwconv2
// then tgemm: the same values.
static void test_dwgemm_gate(int H, int W, int C)
{
    const int Cp = (C + 15) / 16 * 16, C8 = (C + 7) / 8 * 8, M = H * W, FS = 2 * Cp;
    std::vector<uint16_t> n2((size_t)M * Cp, 0), h1((size_t)M * FS, 0);
    std::vector<float> x((size_t)M * Cp, 0.f);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < C; k++) n2[(size_t)m * Cp + k] = hb(ur(2.f));
        for (int k = 0; k < FS; k++) h1[(size_t)m * FS + k] = hb(ur(1.5f));
        for (int n = 0; n < C; n++) x[(size_t)m * Cp + n] = ur(500.f);
    }
    std::vector<float> w((size_t)C * C), dw((size_t)C8 * 9, 0.f), dwb(C8, 0.f), b(C);
    for (auto &v : w) v = hf(hb(ur(0.1f)));
    for (int k = 0; k < C * 9; k++) dw[k] = ur(0.4f);
    for (int k = 0; k < C; k++) dwb[k] = ur(0.3f);
    for (auto &v : b) v = ur(0.2f);
    TMat tm;
    ck(tmat_upload(tm, w.data(), b.data(), C, C));
    uint16_t *dn = dev(n2), *dh = dev(h1), *dp = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    float *o1 = dev(x), *o2 = dev(x), *ddw = dev(dw), *ddb = dev(dwb);
    ck(tdwconv2(dn, Cp, dp, Cp, H, W, C8, ddw, ddb, false, dh, FS, nullptr, nullptr));
    TGemm g;
    g.a = dp; g.sa = Cp; g.M = M; g.out = o1; g.so = Cp; g.out_f32 = true; g.add1 = o1; g.s1 = Cp; g.add1_f32 = true;
    ck(tgemm(tm, g, nullptr));
    TDwGemm dg;
    dg.in = dn; dg.sin = Cp; dg.H = H; dg.W = W; dg.dw = ddw; dg.dwb = ddb; dg.dwc = C8; dg.mul = dh; dg.smul = FS;
    dg.out = o2; dg.so = Cp; dg.out_f32 = true; dg.add1 = o2; dg.s1 = Cp; dg.add1_f32 = true;
    const cudaError_t e = tdwgemm(tm, dg, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "dwgemm gate %dx%d %d vs dwconv2 + gemm (values differing)", H, W, C);
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", nm, cudaGetErrorString(e));
        fails++;
        checks++;
    } else {
        auto h1o = host(o1, x.size()), h2o = host(o2, x.size());
        size_t diff = 0;
        for (size_t i = 0; i < h1o.size(); i++) diff += memcmp(&h1o[i], &h2o[i], 4) != 0;
        report(nm, (double)diff, 0);
    }
    tmat_release(tm);
    cudaFree(dn); cudaFree(dh); cudaFree(dp); cudaFree(o1); cudaFree(o2); cudaFree(ddw); cudaFree(ddb);
}

// DAT's SGFN on the fp32 stream with every LayerNorm fused: fc1 normalising its fp32 A rows on load and giving
// LN3's statistics of x2, then tdwgemm normalising x2 from them, against tlayernorm2 + tgemm, tlayernorm and
// tdwgemm (the same maths in another order: a few values round the other way).
static void test_sgfn_ln(int H, int W, int C)
{
    const int Cp = (C + 15) / 16 * 16, C8 = (C + 7) / 8 * 8, M = H * W, FS = 2 * Cp;
    std::vector<float> x((size_t)M * Cp, 0.f), w1((size_t)C * FS, 0.f), w2((size_t)C * C), b1(FS, 0.f), b2(C), g2(C), t2(C), g3(C), t3(C);
    for (int m = 0; m < M; m++) {
        const float off = ur(300.f), sc = 1.f + std::fabs(ur(200.f));  // a residual stream's spread
        for (int n = 0; n < C; n++) x[(size_t)m * Cp + n] = off + sc * ur(1.f);
    }
    for (int k = 0; k < C; k++)  // fc1 into x1 = columns [0, C), x2 = [Cp, Cp + C)
        for (int n = 0; n < C; n++) {
            w1[(size_t)k * FS + n] = hf(hb(ur(0.15f)));
            w1[(size_t)k * FS + Cp + n] = hf(hb(ur(0.15f)));
        }
    for (int n = 0; n < C; n++) {
        b1[n] = ur(0.5f);
        b1[Cp + n] = 0.4f + ur(0.5f);
    }
    for (auto &v : w2) v = hf(hb(ur(0.1f)));
    for (auto &v : b2) v = ur(0.2f);
    for (int k = 0; k < C; k++) {
        g2[k] = 0.5f + ur(0.4f); t2[k] = ur(0.2f); g3[k] = 0.5f + ur(0.4f); t3[k] = ur(0.2f);
    }
    std::vector<float> dw((size_t)C8 * 9, 0.f), dwb(C8, 0.f);
    for (int k = 0; k < C * 9; k++) dw[k] = ur(0.4f);
    for (int k = 0; k < C; k++) dwb[k] = ur(0.3f);
    TMat m1, m2;
    ck(tmat_upload(m1, w1.data(), b1.data(), C, FS));
    ck(tmat_upload(m2, w2.data(), b2.data(), C, C));
    float *s1 = dev(x), *s2 = dev(x), *dg2 = dev(g2), *dt2 = dev(t2), *dg3 = dev(g3), *dt3 = dev(t3), *ddw = dev(dw), *ddb = dev(dwb);
    float *dst = dev(std::vector<float>((size_t)M * 2, NAN));
    uint16_t *dn = dev(std::vector<uint16_t>((size_t)M * Cp, 0)), *h1 = dev(std::vector<uint16_t>((size_t)M * FS, 0)),
             *h2 = dev(std::vector<uint16_t>((size_t)M * FS, 0)), *n2 = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    // reference: the LayerNorms stored
    ck(tlayernorm2(s1, true, Cp, dn, false, Cp, M, C, Cp, dg2, dt2, 1e-5f, nullptr));
    TGemm g;
    g.a = dn; g.sa = Cp; g.M = M; g.out = h1; g.so = FS; g.act = TAct::Gelu;
    ck(tgemm(m1, g, nullptr));
    ck(tlayernorm(h1 + Cp, FS, n2, Cp, M, C, Cp, dg3, dt3, 1e-5f, nullptr));
    TDwGemm d;
    d.in = n2; d.sin = Cp; d.H = H; d.W = W; d.dw = ddw; d.dwb = ddb; d.dwc = C8; d.mul = h1; d.smul = FS;
    d.out = s1; d.so = Cp; d.out_f32 = true; d.add1 = s1; d.s1 = Cp; d.add1_f32 = true;
    ck(tdwgemm(m2, d, nullptr));
    // fused
    g.a = s2; g.out = h2; g.a_f32 = true; g.ln_gamma = dg2; g.ln_beta = dt2; g.ln_eps = 1e-5f;
    g.stats_out = dst; g.stats_c0 = Cp; g.stats_n = C; g.stats_eps = 1e-5f;
    const cudaError_t e1 = tgemm(m1, g, nullptr);
    d.in = h2 + Cp; d.sin = FS; d.mul = h2; d.out = s2; d.add1 = s2; d.ln_stats = dst; d.ln_gamma = dg3; d.ln_beta = dt3; d.ln_c = C;
    const cudaError_t e2 = tdwgemm(m2, d, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "sgfn+ln %dx%d %d vs layernorms stored", H, W, C);
    checks++;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("FAIL %s: %s / %s\n", nm, cudaGetErrorString(e1), cudaGetErrorString(e2));
        fails++;
    } else {
        auto a1 = host(h1, (size_t)M * FS), a2 = host(h2, (size_t)M * FS);
        double he = 0;
        for (size_t i = 0; i < a1.size(); i++) he = emax(he, (double)std::fabs(hf(a1[i]) - hf(a2[i])));
        char nb[120];
        snprintf(nb, sizeof(nb), "%s: fc1", nm);
        report(nb, he, 0.02);
        auto o1 = host(s1, x.size()), o2 = host(s2, x.size());
        double se = 0;
        for (size_t i = 0; i < o1.size(); i++) se = emax(se, (double)std::fabs(o1[i] - o2[i]) / std::max(1.0, std::fabs((double)o1[i]) * 1e-3));
        snprintf(nb, sizeof(nb), "%s: stream", nm);
        report(nb, se, 0.05);
    }
    tmat_release(m1);
    tmat_release(m2);
    for (void *q : {(void *)s1, (void *)s2, (void *)dg2, (void *)dt2, (void *)dg3, (void *)dt3, (void *)ddw, (void *)ddb, (void *)dst, (void *)dn,
                    (void *)h1, (void *)h2, (void *)n2})
        cudaFree(q);
}

// DAT's token-gate mix in the gate GEMM's epilogue against the GEMM storing its hidden layer then dat_mix_tg
// (the gate's dot product in another order: a few values may round the other way).
static void test_gate_mix(int M, int C, int r, int mode)
{
    const int Cp = (C + 15) / 16 * 16;
    std::vector<uint16_t> a((size_t)M * Cp, 0), b((size_t)M * Cp, 0);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < C; k++) {
            a[(size_t)m * Cp + k] = hb(ur(3.f));
            b[(size_t)m * Cp + k] = hb(ur(3.f));
        }
    std::vector<float> w((size_t)C * r), bias(r), cg(Cp, 0.f), w2(r);
    for (auto &v : w) v = hf(hb(ur(0.1f)));
    for (auto &v : bias) v = ur(0.3f);
    for (int k = 0; k < C; k++) cg[k] = ur(1.f);
    for (auto &v : w2) v = ur(0.5f);
    const float b2 = 0.1f;
    TMat tm;
    ck(tmat_upload(tm, w.data(), bias.data(), C, r));
    uint16_t *da = dev(a), *db = dev(b), *tg = dev(std::vector<uint16_t>((size_t)M * 16, 0)), *z1 = dev(std::vector<uint16_t>((size_t)M * Cp, 0x7c00)),
             *z2 = dev(std::vector<uint16_t>((size_t)M * Cp, 0x7c00));
    float *dcg = dev(cg), *dw2 = dev(w2);
    const void *src = mode ? db : da, *oth = mode ? da : db;
    TGemm g;
    g.a = src; g.sa = Cp; g.M = M; g.out = tg; g.so = 16; g.act = TAct::Gelu;
    ck(tgemm(tm, g, nullptr));
    ck(dat_mix_tg(da, db, Cp, dcg, tg, 16, r, dw2, b2, mode, z1, M, C, nullptr));
    g.mix_other = oth; g.mix_cg = dcg; g.mix_w2 = dw2; g.mix_b2 = b2; g.mix_r = r; g.mix_s = Cp; g.mix_z = z2;
    const cudaError_t e = tgemm(tm, g, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "gate mix %d x %d r %d mode %d vs gemm + mix", M, C, r, mode);
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", nm, cudaGetErrorString(e));
        fails++;
        checks++;
    } else {
        auto h1 = host(z1, (size_t)M * Cp), h2 = host(z2, (size_t)M * Cp);
        double me = 0;
        for (size_t i = 0; i < h1.size(); i++) me = emax(me, (double)std::fabs(hf(h1[i]) - hf(h2[i])));
        report(nm, me, 0.01);
    }
    tmat_release(tm);
    for (void *q : {(void *)da, (void *)db, (void *)tg, (void *)z1, (void *)z2, (void *)dcg, (void *)dw2}) cudaFree(q);
}

// FDAT's LN1 read by qkv (normalising on load and giving the rows' statistics) and by the depthwise conv
// (from those statistics) against tlayernorm, then tgemm and tdwconv of the stored rows.
static void test_ln1_fused(int H, int W, int C, int N)
{
    const int Cp = (C + 15) / 16 * 16, Np = (N + 15) / 16 * 16, M = H * W;
    std::vector<uint16_t> x((size_t)M * Cp, 0);
    for (int m = 0; m < M; m++) {
        const float off = ur(3.f), sc = 0.5f + std::fabs(ur(4.f));
        for (int k = 0; k < C; k++) x[(size_t)m * Cp + k] = hb(off + sc * ur(1.f));
    }
    std::vector<float> w((size_t)C * N), gm(C), bt(C), dw((size_t)C * 9);
    for (auto &v : w) v = hf(hb(ur(0.1f)));
    for (auto &v : gm) v = 0.5f + ur(0.4f);
    for (auto &v : bt) v = ur(0.2f);
    for (auto &v : dw) v = ur(0.4f);
    TMat tm;
    ck(tmat_upload(tm, w.data(), nullptr, C, N));
    uint16_t *dx = dev(x), *dn = dev(std::vector<uint16_t>((size_t)M * Cp, 0)), *q1 = dev(std::vector<uint16_t>((size_t)M * Np, 0)),
             *q2 = dev(std::vector<uint16_t>((size_t)M * Np, 0)), *c1 = dev(std::vector<uint16_t>((size_t)M * Cp, 0)),
             *c2 = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    float *dg = dev(gm), *db = dev(bt), *ddw = dev(dw), *st = dev(std::vector<float>((size_t)M * 2, NAN)), *s1 = dev(std::vector<float>(Cp, 0.f)),
          *s2 = dev(std::vector<float>(Cp, 0.f));
    ck(tlayernorm(dx, Cp, dn, Cp, M, C, Cp, dg, db, 1e-5f, nullptr));
    TGemm g;
    g.a = dn; g.sa = Cp; g.M = M; g.out = q1; g.so = Np;
    ck(tgemm(tm, g, nullptr));
    ck(tdwconv(dn, Cp, c1, Cp, H, W, C, ddw, true, s1, nullptr));
    g.a = dx; g.out = q2; g.ln_gamma = dg; g.ln_beta = db; g.ln_eps = 1e-5f; g.a_stats_out = st;
    const cudaError_t e1 = tgemm(tm, g, nullptr);
    const cudaError_t e2 = tdwconv_ln(dx, Cp, c2, Cp, H, W, C, ddw, true, s2, st, dg, db, nullptr);
    sync();
    char nm[96];
    snprintf(nm, sizeof(nm), "ln1 fused %dx%d %d->%d vs layernorm stored", H, W, C, N);
    checks++;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("FAIL %s: %s / %s\n", nm, cudaGetErrorString(e1), cudaGetErrorString(e2));
        fails++;
    } else {
        auto a1 = host(q1, (size_t)M * Np), a2 = host(q2, (size_t)M * Np), d1 = host(c1, (size_t)M * Cp), d2 = host(c2, (size_t)M * Cp);
        auto u1 = host(s1, (size_t)Cp), u2 = host(s2, (size_t)Cp);
        double qe = 0, de = 0, se = 0;
        for (size_t i = 0; i < a1.size(); i++) qe = emax(qe, (double)std::fabs(hf(a1[i]) - hf(a2[i])));
        for (size_t i = 0; i < d1.size(); i++) de = emax(de, (double)std::fabs(hf(d1[i]) - hf(d2[i])));
        for (int c = 0; c < C; c++) se = emax(se, std::fabs((double)u1[c] - u2[c]) / std::max(1.0, std::fabs((double)u1[c]) * 1e-3));
        char nb[120];
        snprintf(nb, sizeof(nb), "%s: qkv", nm);
        report(nb, qe, 0.02);
        snprintf(nb, sizeof(nb), "%s: dwconv", nm);
        report(nb, de, 0.02);
        snprintf(nb, sizeof(nb), "%s: sums", nm);
        report(nb, se, 0.05);
    }
    tmat_release(tm);
    for (void *q : {(void *)dx, (void *)dn, (void *)q1, (void *)q2, (void *)c1, (void *)c2, (void *)dg, (void *)db, (void *)ddw, (void *)st,
                    (void *)s1, (void *)s2})
        cudaFree(q);
}

static void test_ln(int M, int C)
{
    const int cp = (C + 15) / 16 * 16 + 8;
    std::vector<uint16_t> in((size_t)M * cp);
    for (auto &v : in) v = hb(ur(2.f) + 0.5f);
    std::vector<float> gm(C), bt(C);
    for (int c = 0; c < C; c++) { gm[c] = 1.f + ur(0.3f); bt[c] = ur(0.3f); }
    uint16_t *di = dev(in), *dout = dev(std::vector<uint16_t>((size_t)M * cp, 0x3c00));
    float *dg = dev(gm), *db = dev(bt);
    ck(tlayernorm(di, cp, dout, cp, M, C, (C + 15) / 16 * 16, dg, db, 1e-5f, nullptr));
    sync();
    auto o = host(dout, (size_t)M * cp);
    double me = 0;
    for (int m = 0; m < M; m++) {
        double s = 0, q = 0;
        for (int c = 0; c < C; c++) s += hf(in[(size_t)m * cp + c]);
        const double mean = s / C;
        for (int c = 0; c < C; c++) q += (hf(in[(size_t)m * cp + c]) - mean) * (hf(in[(size_t)m * cp + c]) - mean);
        const double rs = 1.0 / std::sqrt(q / C + 1e-5);
        for (int c = 0; c < C; c++) me = emax(me, std::fabs((hf(in[(size_t)m * cp + c]) - mean) * rs * gm[c] + bt[c] - hf(o[(size_t)m * cp + c])));
        for (int c = C; c < (C + 15) / 16 * 16; c++) me = emax(me, (double)std::fabs(hf(o[(size_t)m * cp + c])));
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "layernorm %dx%d", M, C);
    report(nm, me, 0.01);
    cudaFree(di); cudaFree(dout); cudaFree(dg); cudaFree(db);
}

static void test_win(int H, int W, int heads, int hd, int wh, int ww, int sy, int sx, bool bias)
{
    const int C = heads * hd, sq = 3 * C + 8, so = C + 8, T = wh * ww;
    std::vector<uint16_t> qkv((size_t)H * W * sq);
    for (auto &v : qkv) v = hb(ur(1.f));
    std::vector<float> bt((size_t)heads * T * T);
    for (auto &v : bt) v = ur(1.f);
    uint16_t *dq = dev(qkv), *dout = dev(std::vector<uint16_t>((size_t)H * W * so, 0));
    float *db = dev(bt);
    TWinAttn a;
    a.qkv = dq; a.sq = sq; a.qo = 0; a.ko = C; a.vo = 2 * C; a.H = H; a.W = W; a.heads = heads; a.hd = hd; a.wh = wh; a.ww = ww;
    a.shift_y = sy; a.shift_x = sx; a.scale = 1.f / std::sqrt((float)hd); a.bias = bias ? db : nullptr; a.out = dout; a.so = so;
    const cudaError_t e = twinattn(a, nullptr);
    sync();
    auto o = host(dout, (size_t)H * W * so);
    double me = 0;
    const int nwx = (W + ww - 1) / ww, nwy = (H + wh - 1) / wh;
    for (int wy = 0; wy < nwy; wy++)
        for (int wx = 0; wx < nwx; wx++)
            for (int h = 0; h < heads; h++) {
                std::vector<int> pix(T);
                for (int t = 0; t < T; t++) {
                    int y = wy * wh + t / ww, x = wx * ww + t % ww;
                    if (y >= H || x >= W) { pix[t] = -1; continue; }
                    pix[t] = ((y + sy) % H) * W + (x + sx) % W;
                }
                auto val = [&](int t, int off, int d) { return pix[t] < 0 ? 0.f : hf(qkv[(size_t)pix[t] * sq + off + h * hd + d]); };
                for (int i = 0; i < T; i++) {
                    if (pix[i] < 0) continue;
                    std::vector<double> s(T);
                    double mx = -1e30;
                    for (int j = 0; j < T; j++) {
                        double v = 0;
                        for (int d = 0; d < hd; d++) v += val(i, 0, d) * val(j, C, d);
                        v = v * a.scale + (bias ? bt[((size_t)h * T + i) * T + j] : 0);
                        s[j] = v;
                        mx = std::max(mx, v);
                    }
                    double l = 0;
                    for (int j = 0; j < T; j++) l += (s[j] = std::exp(s[j] - mx));
                    for (int d = 0; d < hd; d++) {
                        double v = 0;
                        for (int j = 0; j < T; j++) v += s[j] * val(j, 2 * C, d);
                        me = emax(me, std::fabs(v / l - hf(o[(size_t)pix[i] * so + h * hd + d])));
                    }
                }
            }
    char nm[96];
    snprintf(nm, sizeof(nm), "winattn %dx%d heads %d hd %d win %dx%d shift %d,%d%s", W, H, heads, hd, ww, wh, sy, sx, bias ? " bias" : "");
    report(nm, e == cudaSuccess ? me : 1e9, 0.01);
    cudaFree(dq); cudaFree(dout); cudaFree(db);
}

static void test_dw_sums(int H, int W, int C)
{
    const int st = (C + 15) / 16 * 16;
    std::vector<uint16_t> in((size_t)H * W * st, 0);
    for (int t = 0; t < H * W; t++)
        for (int c = 0; c < C; c++) in[(size_t)t * st + c] = hb(ur(1.f));
    std::vector<float> w((size_t)C * 9);
    for (auto &v : w) v = ur(0.5f);
    uint16_t *di = dev(in), *dout = dev(std::vector<uint16_t>((size_t)H * W * st, 0));
    float *dw = dev(w), *ds = dev(std::vector<float>(C, 0.f));
    float *dfs = dev(std::vector<float>(C, 0.f));
    uint16_t *dout2 = dev(std::vector<uint16_t>((size_t)H * W * st, 0));
    ck(tdwconv(di, st, dout, st, H, W, C, dw, true, nullptr));
    ck(tdwconv(di, st, dout2, st, H, W, C, dw, true, dfs, nullptr));  // fused channel sums of the output
    ck(tcolsum(di, st, H * W, C, ds, nullptr));
    sync();
    auto o = host(dout, (size_t)H * W * st);
    auto o2 = host(dout2, (size_t)H * W * st);
    auto sums = host(ds, C);
    auto fsums = host(dfs, C);
    double mf = 0;
    for (int c = 0; c < C; c++) {
        double s = 0;
        for (int t = 0; t < H * W; t++) {
            s += hf(o2[(size_t)t * st + c]);
            mf = emax(mf, (double)std::fabs(hf(o2[(size_t)t * st + c]) - hf(o[(size_t)t * st + c])));
        }
        mf = emax(mf, !(std::fabs(s - fsums[c]) / (1 + std::fabs(s)) <= 2e-3) ? 1.0 : 0.0);
    }
    double me = 0, ms = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++) {
                double v = 0;
                for (int k = 0; k < 9; k++) {
                    const int yy = y + k / 3 - 1, xx = x + k % 3 - 1;
                    if (yy >= 0 && yy < H && xx >= 0 && xx < W) v += hf(in[((size_t)yy * W + xx) * st + c]) * w[c * 9 + k];
                }
                me = emax(me, (double)std::fabs(gelu((float)v) - hf(o[((size_t)y * W + x) * st + c])));
            }
    for (int c = 0; c < C; c++) {
        double s = 0;
        for (int t = 0; t < H * W; t++) s += hf(in[(size_t)t * st + c]);
        ms = emax(ms, std::fabs(s - sums[c]) / (1 + std::fabs(s)));
    }
    report("dwconv gelu", me, 0.01);
    report("colsum", ms, 1e-4);
    report("dwconv + fused sums", mf, 1e-3);
    cudaFree(di); cudaFree(dout); cudaFree(dw); cudaFree(ds); cudaFree(dfs); cudaFree(dout2);
}

static void test_chanstats(int M, int heads, int hd)
{
    const int C = heads * hd, sq = 3 * C + 8;
    std::vector<uint16_t> qkv((size_t)M * sq);
    for (auto &v : qkv) v = hb(ur(1.f));
    const int n = heads * hd * hd + 2 * C;
    uint16_t *dq = dev(qkv);
    float *dst = dev(std::vector<float>(n, 0.f));
    ck(tchanstats(dq, sq, 0, C, M, heads, hd, dst, nullptr));
    sync();
    auto st = host(dst, n);
    double me = 0;
    for (int h = 0; h < heads; h++)
        for (int d = 0; d < hd; d++) {
            for (int e2 = 0; e2 < hd; e2++) {
                double s = 0;
                for (int t = 0; t < M; t++) s += hf(qkv[(size_t)t * sq + h * hd + d]) * hf(qkv[(size_t)t * sq + C + h * hd + e2]);
                me = emax(me, std::fabs(s - st[(h * hd + d) * hd + e2]) / (1 + std::fabs(s)));
            }
            double q = 0, k = 0;
            for (int t = 0; t < M; t++) {
                q += std::pow(hf(qkv[(size_t)t * sq + h * hd + d]), 2);
                k += std::pow(hf(qkv[(size_t)t * sq + C + h * hd + d]), 2);
            }
            me = emax(me, std::fabs(q - st[heads * hd * hd + h * hd + d]) / (1 + q));
            me = emax(me, std::fabs(k - st[heads * hd * hd + C + h * hd + d]) / (1 + k));
        }
    report("chanstats", me, 1e-4);
    cudaFree(dq); cudaFree(dst);
}

static void test_pack(int K, int N)
{
    std::vector<float> w((size_t)K * N);
    for (auto &v : w) v = hf(hb(ur(1.f)));
    TMat a, b;
    ck(tmat_upload(a, w.data(), nullptr, K, N));
    ck(tmat_alloc(b, K, N, nullptr));
    float *dw = dev(w);
    ck(tmat_pack(b, dw, nullptr));
    sync();
    const size_t n = (size_t)a.Kpad * a.Npad;
    auto ha = host((const uint16_t *)a.frag, n), hb2 = host((const uint16_t *)b.frag, n);
    double me = 0;
    for (size_t i = 0; i < n; i++) me = emax(me, (double)(ha[i] != hb2[i]));
    report("device pack == host pack", me, 0);
    tmat_release(a);
    tmat_release(b);
    cudaFree(dw);
}

// The comparison itself: a NaN result must fail its bound.
static void test_checker()
{
    checks++;
    double me = 0;
    me = emax(me, std::fabs(1.0 - std::nan("")));
    me = emax(me, 0.0);
    if (me <= 1.0) {
        fails++;
        printf("FAIL checker: a NaN result passes its error bound\n");
    }
}

int main()
{
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess || p.major < 8) {
        printf("native tfm: SKIP (no sm_80+ GPU)\n");
        return 0;
    }
    test_checker();
    test_gemm(200, 120, 360, false, false, false, false, false);
    test_gemm(130, 120, 120, true, false, true, true, true);
    test_gemm(77, 120, 240, false, true, false, false, false);
    test_gemm(64, 240, 120, false, false, false, true, false);
    test_gemm(100, 180, 540, true, true, false, false, false);
    test_gemm(333, 192, 544, true, false, true, false, false);  // K = 192 without adds: the row-stationary kernel
    test_gemm(77, 180, 360, false, true, false, false, false);
    test_gemm(200, 224, 672, true, false, false, false, false);  // K = 224 (SST-XL+)
    test_gemm(150, 120, 360, false, true, false, false, false);  // K = 128 (FDAT-M)
    test_gemm(90, 256, 128, true, false, false, false, false);
    test_gemm_ln(1000, 180, 360);  // FDAT-XL's LN2 + fc1
    test_gemm_ln(77, 120, 240);    // FDAT-M's
    test_dwgemm(40, 64, 360, 180);  // FDAT-XL's smix + fc2
    test_dwgemm(9, 128, 240, 120);  // FDAT-M's
    test_dwgemm_gate(20, 128, 180);  // DAT-2's SGFN
    test_sgfn_ln(20, 128, 180);      // the same with its LayerNorms fused (fp32 A of fc1, LN3 from fc1's statistics)
    test_sgfn_ln(7, 64, 180);
    test_gate_mix(1000, 180, 11, 0);  // DAT-2's token gates (from the attention / from the conv branch)
    test_gate_mix(77, 180, 11, 1);
    test_ln1_fused(23, 40, 120, 360);  // FDAT-M's LN1 -> qkv and the depthwise conv
    test_ln1_fused(16, 64, 180, 540);  // FDAT-XL's
    test_gemm(90, 120, 120, true, false, false, true, false, true);
    test_mlp(1000, 180, 360, true);  // HAT-L's MLP, a row tail
    test_mlp(77, 180, 360, false);
    test_mlp(4096, 180, 360, true);
    test_mlp_ln(1000, 180, 360);
    test_mlp_ln(70, 180, 360);
    test_ln(300, 120);
    test_ln(50, 180);
    test_win(16, 16, 4, 30, 8, 8, 0, 0, true);
    test_win(13, 21, 4, 30, 8, 8, 0, 0, true);
    test_win(32, 32, 6, 30, 16, 16, 8, 8, true);
    test_win(16, 40, 6, 30, 8, 32, 0, 0, false);
    test_dw_sums(23, 19, 120);
    test_dw_sums(16, 16, 240);
    test_dw_sums(9, 11, 30);  // scalar path
    test_chanstats(700, 4, 30);
    test_chanstats(300, 6, 30);
    test_chanstats(5000, 4, 30);
    test_chanstats(333, 8, 16);
    test_chanstats(300, 2, 40);  // CUDA-core path
    test_ln(77, 30);             // channels not a multiple of 8
    test_pack(120, 120);
    test_pack(180, 540);
    printf("native tfm: %s (%d checks)\n", fails ? "FAIL" : "OK", checks);
    return fails ? 1 : 0;
}
