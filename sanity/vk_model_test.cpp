// aji_vk_model_test: one lowered model on aji_vk's executor (vk/executor.h), by hand.
//   aji_vk_model_test list
//   aji_vk_model_test info [N]       (everything device N, default all, offers the kernel families)
//   aji_vk_model_test <model.onnx> <w> <h> [--device N] [--iters N] [--ref WxH ...] [--cache DIR]
//                     [--seconds S]
// Runs the model at w x h on a synthetic frame and times it (GPU timestamps, model only, and the
// executor's create time; --cache keeps the variant and pipeline caches in DIR, as aji_vk keeps
// them in the model directory, so a second run shows a warm configure). Then, per --ref size (default 256x144 and a ragged 202x118), compares
// the Vulkan output with the fp32 CPU reference of the same plan (sanity/native_ref.h): PSNR and
// max error. With AJI_CUDA, also compares the w x h output with the CUDA native executor's.
// Exit 1 when a comparison is under its limits: --min-db (default 50) and --max-levels, the
// largest error in 8-bit levels (default 2), since PSNR alone can hide a few wrong pixels.
// --seconds S keeps the GPU on the model for S seconds and prints the model time per second, to
// show thermal throttling (phones and tablets).
// Plain C++ with the Vulkan loader opened at run time: it also builds for android-arm64 and runs
// from an adb shell.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "native/lower.h"
#include "native/pack.h"
#include "native_ref.h"
#include "vk/device.h"
#include "vk/executor.h"

#ifdef AJI_VK_TEST_CUDA
#include <cuda_runtime.h>
#include "native/executor.h"
#endif

using namespace ajinative;
using namespace ajivk;  // the Vulkan entry points (vk/vk.h)

namespace {

// A frame with edges, gradients and texture, in [0, 1]: NCHW [3][h][w].
std::vector<float> synth(int w, int h)
{
    std::vector<float> v((size_t)3 * w * h);
    uint32_t s = 12345u;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                s = s * 1664525u + 1013904223u;
                const float noise = ((s >> 9) & 0xffff) / 65535.f - 0.5f;
                float f = 0.5f + 0.3f * std::sin(0.07f * x + 1.3f * c) * std::cos(0.05f * y - 0.7f * c);
                if (((x / 24) + (y / 24)) % 2) f = 0.85f * f + 0.12f;  // checker edges
                if (x > w / 2 && y < h / 3) f = 0.1f + 0.8f * (float)x / w;  // a ramp
                f += 0.06f * noise;
                v[((size_t)c * h + y) * w + x] = std::fmin(1.f, std::fmax(0.f, f));
            }
    return v;
}

std::vector<uint16_t> to_half(const std::vector<float> &f)
{
    std::vector<uint16_t> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = float_to_half_bits(f[i]);
    return h;
}

struct Cmp {
    double db = 0, maxd = 0;
    size_t nan = 0;
};
Cmp compare(const std::vector<uint16_t> &a, const std::vector<float> &ref)
{
    Cmp c;
    double se = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const float x = half_bits_to_float(a[i]);
        if (!std::isfinite(x)) {
            c.nan++;
            continue;
        }
        const double d = std::fabs((double)x - ref[i]);
        se += d * d;
        if (d > c.maxd) c.maxd = d;
    }
    const double mse = se / ref.size();
    c.db = mse > 0 ? 10.0 * std::log10(1.0 / mse) : 200.0;
    return c;
}

bool run_vk(ajivk::Device &dev, const Plan &plan, int w, int h, const std::vector<float> &in, std::vector<uint16_t> *out,
            std::unique_ptr<ajivk::Executor> *keep, std::string *why, double *create_s = nullptr)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto ex = ajivk::Executor::create(dev, plan, w, h, why);
    if (create_s) *create_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ex) return false;
    const auto hin = to_half(in);
    if (!dev.upload(ex->input(), hin.data(), hin.size() * 2, why)) return false;
    if (!ex->run(w, h, why)) return false;
    out->resize((size_t)3 * ex->out_w() * ex->out_h());
    if (!dev.download(ex->output(), out->data(), out->size() * 2, why)) return false;
    if (keep) *keep = std::move(ex);
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    std::string why;
    if (argc >= 2 && !std::strcmp(argv[1], "list")) {
        const auto devs = ajivk::list_devices(&why);
        if (devs.empty()) std::printf("no Vulkan devices: %s\n", why.c_str());
        for (const auto &d : devs)
            std::printf("%u: %s vendor %04x device %04x, subgroup %u [%u..%u], shared %u B, family %s (cm16: %s; sg16: %s)\n",
                        d.index, d.name.c_str(), d.vendor, d.device_id, d.subgroup, d.min_subgroup, d.max_subgroup, d.max_shared,
                        ajivk::family_name(d.family), d.cm16 ? "yes" : d.cm16_why.c_str(), d.sg16 ? "yes" : d.sg16_why.c_str());
        return 0;
    }
    if (argc >= 2 && !std::strcmp(argv[1], "info")) {
        const auto devs = ajivk::list_devices(&why);
        if (devs.empty()) std::printf("no Vulkan devices: %s\n", why.c_str());
        for (const auto &d : devs) {
            if (argc >= 3 && (uint32_t)std::atoi(argv[2]) != d.index) continue;
            std::printf("%s", ajivk::capability_report(d.index, &why).c_str());
        }
        return 0;
    }
    if (argc < 4) {
        std::fprintf(stderr, "usage: aji_vk_model_test list | info [N] | <model.onnx> <w> <h> [--device N] [--iters N] [--ref WxH ...] "
                             "[--min-db D] [--max-levels L] [--cache DIR] [--seconds S]\n");
        return 2;
    }
    const char *model = argv[1];
    const int w = std::atoi(argv[2]), h = std::atoi(argv[3]);
    int device = -1, iters = 20;
    double seconds = 0;
    double min_db = 50.0, max_levels = 2.0;
    std::string cache;
    std::vector<std::pair<int, int>> refs;
    for (int i = 4; i < argc; i++) {
        if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--min-db") && i + 1 < argc) min_db = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-levels") && i + 1 < argc) max_levels = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--cache") && i + 1 < argc) cache = argv[++i];
        else if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) {
            int rw = 0, rh = 0;
            if (std::sscanf(argv[++i], "%dx%d", &rw, &rh) == 2) refs.push_back({rw, rh});
        }
    }
    if (refs.empty()) refs = {{256, 144}, {202, 118}};

    std::ifstream f(model, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    OnnxGraph g;
    Plan plan;
    if (bytes.empty() || !parse_onnx(bytes.data(), bytes.size(), &g, &why) || !lower(g, &plan, &why)) {
        std::fprintf(stderr, "%s: %s\n", model, why.empty() ? "unreadable" : why.c_str());
        return 1;
    }
    std::printf("%s: %s\n", model, describe(plan).c_str());

    auto dev = ajivk::Device::create(device, nullptr, &why);
    if (!dev) {
        std::fprintf(stderr, "device: %s\n", why.c_str());
        return 1;
    }
    std::printf("device %u: %s, family %s\n", dev->info().index, dev->info().name.c_str(), ajivk::family_name(dev->family()));
    if (!cache.empty()) dev->set_cache_dir(cache);

    // full size: create (twice: the second is warm, its pipelines cached), run, time
    const auto in = synth(w, h);
    std::vector<uint16_t> out;
    std::unique_ptr<ajivk::Executor> ex;
    double cold = 0;
    if (!run_vk(*dev, plan, w, h, in, &out, &ex, &why, &cold)) {
        std::fprintf(stderr, "vulkan %dx%d: %s\n", w, h, why.c_str());
        return 1;
    }
    ex.reset();
    const auto t0 = std::chrono::steady_clock::now();
    ex = ajivk::Executor::create(*dev, plan, w, h, &why);
    const double warm = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ex) {
        std::fprintf(stderr, "vulkan %dx%d: %s\n", w, h, why.c_str());
        return 1;
    }
    std::printf("%s", ex->describe().c_str());
    dev->save_caches();
    double best = -1;
    VkQueryPool qp = VK_NULL_HANDLE;
    // one batch of iters runs between two timestamps: ms per frame, or -1 on failure
    auto timed_batch = [&]() -> double {
        VkCommandBuffer cb = dev->begin();
        vkCmdResetQueryPool(cb, qp, 0, 2);
        ex->record(cb, w, h);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        for (int i = 0; i < iters; i++) ex->record(cb, w, h);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        if (!dev->submit_wait(cb, &why)) return -1;
        uint64_t t[2];
        vkGetQueryPoolResults(dev->device(), qp, 0, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        return (double)(t[1] - t[0]) * dev->info().timestamp_ns * 1e-6 / iters;
    };
    if (dev->info().timestamp_ns > 0) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        vkCreateQueryPool(dev->device(), &qpi, nullptr, &qp);
        for (int rep = 0; rep < 3; rep++) {
            const double ms = timed_batch();
            if (ms < 0) {
                std::fprintf(stderr, "timing: %s\n", why.c_str());
                return 1;
            }
            if (best < 0 || ms < best) best = ms;
        }
    }
    std::printf("vulkan %dx%d -> %dx%d: %.3f ms per frame (model only), create %.3f s (first in this process), %.3f s (again), %.1f MB on the device\n", w, h,
                ex->out_w(), ex->out_h(), best, cold, warm, ex->device_bytes() / 1048576.0);
    if (seconds > 0 && qp) {
        // sustained: batches back to back; per wall-clock second, the mean and the slowest batch
        const auto start = std::chrono::steady_clock::now();
        double sum = 0, worst = 0, first = -1, last = 0;
        int n = 0, sec = 1;
        for (;;) {
            const double ms = timed_batch();
            if (ms < 0) {
                std::fprintf(stderr, "sustained: %s\n", why.c_str());
                return 1;
            }
            sum += ms;
            worst = ms > worst ? ms : worst;
            n++;
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (el >= sec || el >= seconds) {
                last = sum / n;
                if (first < 0) first = last;
                std::printf("  sustained %4ds: %.3f ms per frame (slowest batch %.3f)\n", sec, last, worst);
                std::fflush(stdout);
                sum = worst = 0;
                n = 0;
                sec++;
            }
            if (el >= seconds) break;
        }
        std::printf("sustained %.0f s: first second %.3f ms, last second %.3f ms (%.0f%% of the first's speed)\n", seconds, first,
                    last, last > 0 ? 100.0 * first / last : 0.0);
    }
    if (qp) vkDestroyQueryPool(dev->device(), qp, nullptr);

    int bad = 0;
    for (const auto &r : refs) {
        const int rw = r.first, rh = r.second;
        const auto rin = synth(rw, rh);
        std::vector<uint16_t> rout;
        if (!run_vk(*dev, plan, rw, rh, rin, &rout, nullptr, &why)) {
            std::printf("ref %dx%d: vulkan: %s\n", rw, rh, why.c_str());
            bad++;
            continue;
        }
        const auto ref = ajiref::run_plan(plan, rin, rw, rh);
        const Cmp c = compare(rout, ref);
        const bool ok = c.nan == 0 && c.db >= min_db && c.maxd * 255 <= max_levels;
        std::printf("vs fp32 CPU reference at %dx%d: %.1f dB, max error %.4f (%.1f levels of 255), %zu non-finite: %s\n", rw, rh, c.db,
                    c.maxd, c.maxd * 255, c.nan, ok ? "ok" : "FAIL");
        bad += !ok;
    }

#ifdef AJI_VK_TEST_CUDA
    {
        cudaStream_t s;
        cudaStreamCreate(&s);
        auto cx = ajinative::Executor::create(plan, w, h, s, &why);
        if (!cx) {
            std::printf("vs CUDA native: not run (%s)\n", why.c_str());
        } else {
            const auto hin = to_half(in);
            void *din = nullptr, *dout = nullptr;
            cudaMalloc(&din, hin.size() * 2);
            cudaMalloc(&dout, out.size() * 2);
            cudaMemcpy(din, hin.data(), hin.size() * 2, cudaMemcpyHostToDevice);
            cx->run(din, dout, s);
            std::vector<uint16_t> cout(out.size());
            cudaMemcpy(cout.data(), dout, cout.size() * 2, cudaMemcpyDeviceToHost);
            size_t same = 0;
            std::vector<float> cf(cout.size());
            for (size_t i = 0; i < cout.size(); i++) {
                same += cout[i] == out[i];
                cf[i] = half_bits_to_float(cout[i]);
            }
            const Cmp c = compare(out, cf);
            std::printf("vs CUDA native at %dx%d: %.1f dB, max %.4f (%.1f levels), %.2f%% bit-identical\n", w, h, c.db, c.maxd, c.maxd * 255,
                        100.0 * same / cout.size());
            cudaFree(din);
            cudaFree(dout);
        }
        cudaStreamDestroy(s);
    }
#endif
    return bad ? 1 : 0;
}
