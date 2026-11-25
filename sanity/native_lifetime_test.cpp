#include "native/esrgan_lifetimes.h"
#include "native/hat_lifetimes.h"
#include "native/lifetime_check.h"
#include <cstdio>

using namespace ajinative;

static bool esrgan_2x_fixture(int width, int height, int strip_rows)
{
    // The shipped 2x graph has a 2x unshuffle, two upsamplers, an HR conv and an RGB conv.
    const int bw = width / 2, bh = height / 2;
    TailLayout tail;
    for (int i = 0; i < 2; ++i) tail.add(nullptr, 64, GAct::LRelu, 0.2f, true);
    tail.add(nullptr, 64, GAct::LRelu);
    tail.add(nullptr, 16);
    EsrganBuffers body;
    LifetimePlan plan;
    body.declare(plan, bw, bh, 16, 64, 192, strip_rows / 2, tail);
    const auto bytes = plan.arrange();
    // The CPU allocator returns virtual test addresses, never dereferenced.
    auto *base = reinterpret_cast<void *>(uintptr_t{0x10000});
    plan.bind(base);
    void *ws = reinterpret_cast<void *>(uintptr_t(base) + bytes);
    constexpr size_t ws_bytes = 12 << 20;
    LifetimeCheck check;
    check.capture(plan, base, bytes);
    body.ws_floats = ws_bytes / sizeof(float);
    if (!body.assign_workspace([&](size_t) { return ws; }, &check)) return false;
    if (!check.valid()) return false;
    constexpr size_t dense_ops = 23 * 3 * 5;
    for (size_t op = 0; op < 2 + dense_ops + tail.steps(); ++op)
        if (!check.workspace(body.step(op, dense_ops), body.ws, body.ws_floats * sizeof(float))) return false;
    // Negative controls for both kinds of overlap, checked against final slot addresses.
    if (check.workspace(4, body.D[0], 256)) return false;
    auto placement = [&](int k) -> LifetimePlan::Buffer & {
        return *std::find_if(plan.buffers.begin(), plan.buffers.end(), [&](const auto &b) { return b.slot == tail.out(k); });
    };
    placement(1).offset = placement(0).offset;
    plan.bind(base);
    LifetimeCheck aliased;
    aliased.capture(plan, base, bytes);
    return !aliased.valid();
}

static bool hat_l_fixture(int w, int h, size_t arena_limit, bool expect_external)
{
    constexpr int sms = 170, cp = 192;
    HatBuffers body;
    LifetimePlan plan;
    body.declare(plan, size_t(w) * h, cp, 64, 544, 368, 2);
    const size_t arena = plan.arrange();
    auto *base = reinterpret_cast<void *>(uintptr_t{0x10000});
    plan.bind(base);
    LifetimeCheck check;
    check.workspace_arena_limit = arena_limit;
    check.capture(plan, base, arena);
    StepWorkspaces<36> work;
    const size_t stream = size_t(w) * h * cp;
    for (int step : {2,11,12,28,31,32,33,34,35}) {
        const int chunk = step == 35 ? 16 : 64;
        work.need(step, step == 12 ? 0 : gconv_workspace_bound(sms, 16, chunk, step == 28 ? stream : 0, stream));
    }
    size_t fallback_bytes = 0;
    auto fallback = [&](size_t bytes) -> void * {
        if (fallback_bytes) return nullptr;
        fallback_bytes = bytes;
        return reinterpret_cast<void *>(uintptr_t(base) + arena);
    };
    if (!work.place(plan, fallback, &check) || !check.valid()) return false;
    int in_arena = 0, outside = 0;
    for (int step = 0; step < (int)work.slots.size(); ++step) {
        const auto &slot = work.slots[step];
        if (!check.workspace(step, slot.ptr, slot.floats * sizeof(float))) return false;
        if (slot.floats) {
            if (slot.ptr == reinterpret_cast<void *>(uintptr_t(base) + arena)) ++outside;
            else ++in_arena;
        }
    }
    if ((arena_limit != 0) != (in_arena > 0) || expect_external != (outside > 0)) return false;
    return work.slots[28].floats == stream && fallback_bytes == (expect_external ? stream * sizeof(float) : 0);
}

int main()
{
    for (int rows : {0, 256})
        for (const auto &shape : {std::pair<int,int>{1536,1536}, {2048,2048}, {896,640}})
            if (!esrgan_2x_fixture(shape.first, shape.second, rows)) {
                std::fprintf(stderr, "ESRGAN %dx%d, tail rows %d: placement check failed\n", shape.first, shape.second, rows);
                return 1;
            }
    for (auto [w,h] : {std::pair<int,int>{512,512}, {768,768}, {512,384}, {320,192}})
        if (!hat_l_fixture(w,h,size_t(-1),false) || !hat_l_fixture(w,h,0,true)) return 1;
    if (!hat_l_fixture(512,512,64u << 20,true)) return 1; // one shared fallback, other ops still in the arena
    // q = 2*SM-1, k = 2 attains the split bound. A smaller stock cutoff may
    // suppress an optional split, but must always cover the mandatory FP32 map.
    constexpr size_t q = 2 * 170 - 1, tile = 16 * 16 * 64;
    if (gconv_workspace_bound(170,16,64,0,size_t(-1)) != 2*q*tile ||
        gconv_workspace_bound(170,16,64,0,1024) != 1024 ||
        gconv_workspace_bound(170,16,64,2048,1024) != 2048) return 1;
    return 0;
}
