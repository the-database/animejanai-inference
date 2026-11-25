#include "native/esrgan_lifetimes.h"
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

int main()
{
    for (int rows : {0, 256})
        for (const auto &shape : {std::pair<int,int>{1536,1536}, {2048,2048}, {896,640}})
            if (!esrgan_2x_fixture(shape.first, shape.second, rows)) {
                std::fprintf(stderr, "ESRGAN %dx%d, tail rows %d: placement check failed\n", shape.first, shape.second, rows);
                return 1;
            }
    return 0;
}
