// Host-side description of the caller-owned arguments in the CUDA frame conversion kernels.
#pragma once

#include <array>
#include <cstring>
#include <vector>

#include "aji.h"

struct aji_plan;

namespace ajigraph {

using FrameKey = std::array<int, 9>;

inline FrameKey frame_key(const aji_frame &in, const aji_frame &out)
{
    // run_post uses input colorimetry and fixed left siting for subsampled output.
    return {in.format, in.width, in.height, in.matrix, in.range, in.siting,
            out.format, out.width, out.height};
}

struct PlaneKernel {
    const void *symbol;
    int nargs, plane_arg, first_plane, planes;
    bool output, expected;

    bool matches(void *const *args, const aji_frame &frame) const
    {
        for (int j = 0; j < planes; ++j) {
            const int a = plane_arg + 2 * j, p = first_plane + j;
            if (std::memcmp(args[a], &frame.plane[p], sizeof(frame.plane[p])) ||
                std::memcmp(args[a + 1], &frame.stride[p], sizeof(frame.stride[p])))
                return false;
        }
        return true;
    }

    void bind(void **args, const aji_frame &frame) const
    {
        for (int j = 0; j < planes; ++j) {
            const int a = plane_arg + 2 * j, p = first_plane + j;
            args[a] = const_cast<void **>(&frame.plane[p]);
            args[a + 1] = const_cast<ptrdiff_t *>(&frame.stride[p]);
        }
    }
};

using KernelArgs = std::array<void *, 15>;
std::vector<PlaneKernel> conversion_kernels(int in_fmt, int out_fmt, const aji_plan *post);

}  // namespace ajigraph
