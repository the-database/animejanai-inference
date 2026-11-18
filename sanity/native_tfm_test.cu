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

using namespace ajinative;

static std::mt19937 rng(7);
static float ur(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }
static uint16_t hb(float f) { return float_to_half_bits(f); }
static float hf(uint16_t b) { return half_bits_to_float(b); }
static float gelu(float v) { return 0.5f * v * (1.f + std::erf(v / 1.4140625f)); }
static int fails = 0, checks = 0;

static void report(const char *name, double maxe, double lim)
{
    checks++;
    if (!(maxe <= lim)) {
        fails++;
        printf("FAIL %s: max err %.5f (limit %.5f)\n", name, maxe, lim);
    }
}
template <typename T> T *dev(const std::vector<T> &h)
{
    T *d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <typename T> std::vector<T> host(const T *d, size_t n)
{
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
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
    tmat_upload(tm, w.data(), bias ? b.data() : nullptr, K, N);
    uint16_t *da = dev(a), *d1 = dev(a1), *d2 = dev(a2), *dout = dev(std::vector<uint16_t>((size_t)M * so, 0));
    float *dcs = dev(cs);
    TGemm g;
    g.a = da; g.sa = sa; g.M = M; g.out = dout; g.so = so; g.act = gelu_ ? TAct::Gelu : TAct::None;
    g.colscale = scale ? dcs : nullptr;
    if (add1) { g.add1 = d1; g.s1 = so; }
    if (add2) { g.add2 = d2; g.s2 = so; }
    if (lrelu_last) { g.act = TAct::LRelu; g.act_last = true; g.slope = 0.2f; }
    tgemm(tm, g, nullptr);
    cudaDeviceSynchronize();
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
            me = std::max(me, std::fabs(v - hf(o[(size_t)m * so + n])));
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
    tmat_upload(m1, w1.data(), b1.data(), C, hid);
    tmat_upload(m2, w2.data(), b2.data(), hid, C);
    std::vector<uint16_t> res16(res.size());
    for (size_t i = 0; i < res.size(); i++) res16[i] = hb(res[i]);
    const size_t es = f32 ? 4 : 2, n = (size_t)M * Cp;
    uint16_t *da = dev(a), *dh = dev(std::vector<uint16_t>((size_t)M * Hp, 0));
    void *dr = f32 ? (void *)dev(res) : (void *)dev(res16), *o1 = nullptr, *o2 = nullptr;
    cudaMalloc(&o1, n * es);
    cudaMalloc(&o2, n * es);
    cudaMemcpy(o1, dr, n * es, cudaMemcpyDeviceToDevice);  // fp32: in place, as HAT's stream
    cudaMemcpy(o2, dr, n * es, cudaMemcpyDeviceToDevice);
    TGemm g1;
    g1.a = da; g1.sa = Cp; g1.M = M; g1.out = dh; g1.so = Hp; g1.act = TAct::Gelu;
    TGemm g2;
    g2.a = dh; g2.sa = Hp; g2.M = M; g2.out = o1; g2.so = Cp; g2.out_f32 = f32;
    g2.add1 = f32 ? o1 : dr; g2.s1 = Cp; g2.add1_f32 = f32;
    tgemm(m1, g1, nullptr);
    tgemm(m2, g2, nullptr);
    TMlp t;
    t.a = da; t.sa = Cp; t.M = M; t.out = o2; t.so = Cp; t.out_f32 = f32;
    t.add1 = f32 ? o2 : dr; t.s1 = Cp; t.add1_f32 = f32;
    const cudaError_t e = tmlp(m1, m2, t, nullptr);
    cudaDeviceSynchronize();
    char nm[96];
    snprintf(nm, sizeof(nm), "mlp %dx%dx%d%s", M, C, hid, f32 ? " f32" : "");
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", nm, cudaGetErrorString(e));
        fails++;
        checks++;
    } else {
        std::vector<uint8_t> h1(n * es), h2(n * es);
        cudaMemcpy(h1.data(), o1, n * es, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2.data(), o2, n * es, cudaMemcpyDeviceToHost);
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
                me = std::max(me, std::fabs(v - got) / (f32 ? 1.0 : std::max(1.0, std::fabs(v))));
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
    const int Cp = (C + 15) / 16 * 16, Hp = (hid + 15) / 16 * 16;
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
    tmat_upload(m1, w1.data(), b1.data(), C, hid);
    tmat_upload(m2, w2.data(), b2.data(), hid, C);
    float *o1 = dev(x), *o2 = dev(x), *dg = dev(gm), *db = dev(bt);
    uint16_t *dn = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    tlayernorm2(o1, true, Cp, dn, false, Cp, M, C, Cp, dg, db, 1e-5f, nullptr);
    TMlp t;
    t.a = dn; t.sa = Cp; t.M = M; t.out = o1; t.so = Cp; t.out_f32 = true; t.add1 = o1; t.s1 = Cp; t.add1_f32 = true;
    const cudaError_t e1 = tmlp(m1, m2, t, nullptr);
    t.a = o2; t.out = o2; t.add1 = o2; t.ln_gamma = dg; t.ln_beta = db; t.ln_eps = 1e-5f;
    uint16_t *dlo = dev(std::vector<uint16_t>((size_t)M * Cp, 0x7c00)), *dlr = dev(std::vector<uint16_t>((size_t)M * Cp, 0));
    t.lno = dlo; t.slno = Cp; t.lno_gamma = db; t.lno_beta = dg; t.lno_eps = 1e-3f;  // the next LayerNorm (other parameters)
    const cudaError_t e2 = tmlp(m1, m2, t, nullptr);
    tlayernorm2(o2, true, Cp, dlr, false, Cp, M, C, Cp, db, dg, 1e-3f, nullptr);
    cudaDeviceSynchronize();
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
                me = std::max(me, std::fabs(v - h2[i]) / std::max(1.0, std::fabs(v) * 1e-3));
                md = std::max(md, (double)std::fabs(h1[i] - h2[i]) / std::max(1.0, std::fabs((double)h1[i]) * 1e-3));
            }
        }
        report(nm, me, 0.1);
        char nb[120];
        snprintf(nb, sizeof(nb), "%s vs layernorm + mlp", nm);
        report(nb, md, 0.05);
        auto l1 = host(dlo, (size_t)M * Cp), l2 = host(dlr, (size_t)M * Cp);
        double le = 0;
        for (size_t i = 0; i < l1.size(); i++) le = std::max(le, (double)std::fabs(hf(l1[i]) - hf(l2[i])));
        snprintf(nb, sizeof(nb), "%s next layernorm vs tlayernorm2", nm);
        report(nb, le, 0.01);
    }
    tmat_release(m1);
    tmat_release(m2);
    cudaFree(o1); cudaFree(o2); cudaFree(dg); cudaFree(db); cudaFree(dn); cudaFree(dlo); cudaFree(dlr);
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
    tlayernorm(di, cp, dout, cp, M, C, (C + 15) / 16 * 16, dg, db, 1e-5f, nullptr);
    cudaDeviceSynchronize();
    auto o = host(dout, (size_t)M * cp);
    double me = 0;
    for (int m = 0; m < M; m++) {
        double s = 0, q = 0;
        for (int c = 0; c < C; c++) s += hf(in[(size_t)m * cp + c]);
        const double mean = s / C;
        for (int c = 0; c < C; c++) q += (hf(in[(size_t)m * cp + c]) - mean) * (hf(in[(size_t)m * cp + c]) - mean);
        const double rs = 1.0 / std::sqrt(q / C + 1e-5);
        for (int c = 0; c < C; c++) me = std::max(me, std::fabs((hf(in[(size_t)m * cp + c]) - mean) * rs * gm[c] + bt[c] - hf(o[(size_t)m * cp + c])));
        for (int c = C; c < (C + 15) / 16 * 16; c++) me = std::max(me, (double)std::fabs(hf(o[(size_t)m * cp + c])));
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
    cudaDeviceSynchronize();
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
                        me = std::max(me, std::fabs(v / l - hf(o[(size_t)pix[i] * so + h * hd + d])));
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
    tdwconv(di, st, dout, st, H, W, C, dw, true, nullptr);
    tdwconv(di, st, dout2, st, H, W, C, dw, true, dfs, nullptr);  // fused channel sums of the output
    tcolsum(di, st, H * W, C, ds, nullptr);
    cudaDeviceSynchronize();
    auto o = host(dout, (size_t)H * W * st);
    auto o2 = host(dout2, (size_t)H * W * st);
    auto sums = host(ds, C);
    auto fsums = host(dfs, C);
    double mf = 0;
    for (int c = 0; c < C; c++) {
        double s = 0;
        for (int t = 0; t < H * W; t++) {
            s += hf(o2[(size_t)t * st + c]);
            mf = std::max(mf, (double)std::fabs(hf(o2[(size_t)t * st + c]) - hf(o[(size_t)t * st + c])));
        }
        mf = std::max(mf, std::fabs(s - fsums[c]) / (1 + std::fabs(s)) > 2e-3 ? 1.0 : 0.0);
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
                me = std::max(me, (double)std::fabs(gelu((float)v) - hf(o[((size_t)y * W + x) * st + c])));
            }
    for (int c = 0; c < C; c++) {
        double s = 0;
        for (int t = 0; t < H * W; t++) s += hf(in[(size_t)t * st + c]);
        ms = std::max(ms, std::fabs(s - sums[c]) / (1 + std::fabs(s)));
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
    tchanstats(dq, sq, 0, C, M, heads, hd, dst, nullptr);
    cudaDeviceSynchronize();
    auto st = host(dst, n);
    double me = 0;
    for (int h = 0; h < heads; h++)
        for (int d = 0; d < hd; d++) {
            for (int e2 = 0; e2 < hd; e2++) {
                double s = 0;
                for (int t = 0; t < M; t++) s += hf(qkv[(size_t)t * sq + h * hd + d]) * hf(qkv[(size_t)t * sq + C + h * hd + e2]);
                me = std::max(me, std::fabs(s - st[(h * hd + d) * hd + e2]) / (1 + std::fabs(s)));
            }
            double q = 0, k = 0;
            for (int t = 0; t < M; t++) {
                q += std::pow(hf(qkv[(size_t)t * sq + h * hd + d]), 2);
                k += std::pow(hf(qkv[(size_t)t * sq + C + h * hd + d]), 2);
            }
            me = std::max(me, std::fabs(q - st[heads * hd * hd + h * hd + d]) / (1 + q));
            me = std::max(me, std::fabs(k - st[heads * hd * hd + C + h * hd + d]) / (1 + k));
        }
    report("chanstats", me, 1e-4);
    cudaFree(dq); cudaFree(dst);
}

static void test_pack(int K, int N)
{
    std::vector<float> w((size_t)K * N);
    for (auto &v : w) v = hf(hb(ur(1.f)));
    TMat a, b;
    tmat_upload(a, w.data(), nullptr, K, N);
    tmat_alloc(b, K, N, nullptr);
    float *dw = dev(w);
    tmat_pack(b, dw, nullptr);
    cudaDeviceSynchronize();
    const size_t n = (size_t)a.Kpad * a.Npad;
    auto ha = host((const uint16_t *)a.frag, n), hb2 = host((const uint16_t *)b.frag, n);
    double me = 0;
    for (size_t i = 0; i < n; i++) me = std::max(me, (double)(ha[i] != hb2[i]));
    report("device pack == host pack", me, 0);
    tmat_release(a);
    tmat_release(b);
    cudaFree(dw);
}

int main()
{
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess || p.major < 8) {
        printf("native tfm: SKIP (no sm_80+ GPU)\n");
        return 0;
    }
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
