// Host-only checks for the capture key and per-plane argument rebinding.
#include "frame_graph.h"
#include <cstdio>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main()
{
    aji_frame in{}, out{};
    const auto original = ajigraph::frame_key(in, out);
    for (aji_frame *frame : {&in, &out}) {
        for (auto field : {&aji_frame::width, &aji_frame::height, &aji_frame::format}) {
            ++(frame->*field);
            CHECK(ajigraph::frame_key(in, out) != original);
            --(frame->*field);
        }
    }
    for (auto field : {&aji_frame::matrix, &aji_frame::range, &aji_frame::siting}) {
        ++(in.*field);
        CHECK(ajigraph::frame_key(in, out) != original);
        --(in.*field);
        ++(out.*field);
        CHECK(ajigraph::frame_key(in, out) == original);
        --(out.*field);
    }
    int pixels[4]{};
    out.plane[1] = &pixels[0]; out.stride[1] = 128;
    out.plane[2] = &pixels[1]; out.stride[2] = 192;
    CHECK(ajigraph::frame_key(in, out) == original);
    const ajigraph::PlaneKernel kernel{nullptr, 7, 2, 1, 2, true, true};
    ajigraph::KernelArgs args{};
    args.fill(&pixels[0]);
    kernel.bind(args.data(), out);
    CHECK(kernel.matches(args.data(), out));
    aji_frame next = out;
    next.stride[2] += 64;
    CHECK(!kernel.matches(args.data(), next));
    next = out;
    next.plane[1] = &pixels[3];
    CHECK(!kernel.matches(args.data(), next));
    next.stride[1] += 32;
    kernel.bind(args.data(), next);
    CHECK(kernel.matches(args.data(), next));
    CHECK(args[0] == &pixels[0] && args[1] == &pixels[0] && args[6] == &pixels[0]);
    CHECK(args[2] == &next.plane[1] && args[3] == &next.stride[1]);
    CHECK(args[4] == &next.plane[2] && args[5] == &next.stride[2]);
    return 0;
}
