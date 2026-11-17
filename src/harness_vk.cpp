/*
 * aji_harness_vk — raw 4:2:0 frames through a model on Vulkan alone (vk/frame.h), no D3D: for
 * Linux, Android (adb shell) and checking the Vulkan frame kernels against the DirectML ones.
 *
 *   aji_harness_vk --model M.onnx --input in.raw --width W --height H [--frames N]
 *                  [--output out.raw] [--format nv12|p010] [--matrix 709|601|2020]
 *                  [--range limited|full] [--siting left|center|topleft] [--device N]
 *
 * Raw files: per frame the Y plane (W x H samples) then the interleaved CbCr plane (W x H/2),
 * 8-bit (NV12) or 16-bit little-endian MSB-aligned (P010), as aji_harness_dml reads and writes
 * them. Prints the time per frame (upload, model, download; frames after the first two) and the
 * GPU time of pre + model + post.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "aji.h"
#include "native/lower.h"
#include "native/onnx_graph.h"
#include "vk/device.h"
#include "vk/frame.h"

using namespace ajivk;

int main(int argc, char **argv)
{
    const char *model = nullptr, *input = nullptr, *output = nullptr;
    int w = 0, h = 0, frames = 1 << 30, device = -1;
    FrameFormat f;
    for (int i = 1; i + 1 < argc; i += 2) {
        const char *a = argv[i], *v = argv[i + 1];
        if (!std::strcmp(a, "--model")) model = v;
        else if (!std::strcmp(a, "--input")) input = v;
        else if (!std::strcmp(a, "--output")) output = v;
        else if (!std::strcmp(a, "--width")) w = std::atoi(v);
        else if (!std::strcmp(a, "--height")) h = std::atoi(v);
        else if (!std::strcmp(a, "--frames")) frames = std::atoi(v);
        else if (!std::strcmp(a, "--device")) device = std::atoi(v);
        else if (!std::strcmp(a, "--format")) f.format = std::strcmp(v, "p010") ? AJI_FMT_NV12 : AJI_FMT_P010;
        else if (!std::strcmp(a, "--matrix")) f.matrix = !std::strcmp(v, "601") ? AJI_MATRIX_BT601 : !std::strcmp(v, "2020") ? AJI_MATRIX_BT2020 : AJI_MATRIX_BT709;
        else if (!std::strcmp(a, "--range")) f.range = std::strcmp(v, "full") ? AJI_RANGE_LIMITED : AJI_RANGE_FULL;
        else if (!std::strcmp(a, "--siting")) f.siting = !std::strcmp(v, "center") ? AJI_SITING_CENTER : !std::strcmp(v, "topleft") ? AJI_SITING_TOPLEFT : AJI_SITING_LEFT;
        else {
            std::fprintf(stderr, "unknown option %s\n", a);
            return 2;
        }
    }
    if (!model || !input || w < 2 || h < 2) {
        std::fprintf(stderr, "usage: aji_harness_vk --model M.onnx --input in.raw --width W --height H [--frames N] [--output out.raw] "
                             "[--format nv12|p010] [--matrix 709|601|2020] [--range limited|full] [--siting left|center|topleft] [--device N]\n");
        return 2;
    }
    std::string why;
    std::ifstream mf(model, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(mf)), {});
    ajinative::OnnxGraph g;
    ajinative::Plan plan;
    if (bytes.empty() || !ajinative::parse_onnx(bytes.data(), bytes.size(), &g, &why) || !ajinative::lower(g, &plan, &why)) {
        std::fprintf(stderr, "%s: %s\n", model, why.empty() ? "unreadable" : why.c_str());
        return 1;
    }
    auto dev = Device::create(device, nullptr, &why);
    if (!dev) {
        std::fprintf(stderr, "device: %s\n", why.c_str());
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto fp = FramePipeline::create(*dev, plan, w, h, f, &why);
    if (!fp) {
        std::fprintf(stderr, "frame pipeline: %s\n", why.c_str());
        return 1;
    }
    std::printf("device %s, family %s; %s; %dx%d -> %dx%d %s, configured in %.3f s\n", dev->info().name.c_str(),
                family_name(dev->family()), ajinative::describe(plan).c_str(), w, h, fp->out_w(), fp->out_h(),
                f.format == AJI_FMT_P010 ? "P010" : "NV12",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    const int bps = f.format == AJI_FMT_P010 ? 2 : 1, ow = fp->out_w(), oh = fp->out_h();
    const size_t in_frame = (size_t)w * h * 3 / 2 * bps, out_frame = (size_t)ow * oh * 3 / 2 * bps;
    std::vector<char> raw(in_frame), out(out_frame);
    std::vector<uint8_t> py((size_t)fp->in_pitch() * h, 0), puv((size_t)fp->in_pitch() * (h / 2), 0);
    std::vector<uint8_t> qy((size_t)fp->out_pitch() * oh), quv((size_t)fp->out_pitch() * (oh / 2));
    std::FILE *fi = std::fopen(input, "rb");
    std::FILE *fo = output ? std::fopen(output, "wb") : nullptr;
    if (!fi || (output && !fo)) {
        std::fprintf(stderr, "cannot open %s\n", !fi ? input : output);
        return 1;
    }
    VkQueryPool qp = VK_NULL_HANDLE;
    if (dev->info().timestamp_ns > 0) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        vkCreateQueryPool(dev->device(), &qpi, nullptr, &qp);
    }
    double total_ms = 0, gpu_ms = 0;
    int n = 0, timed = 0;
    while (n < frames && std::fread(raw.data(), 1, in_frame, fi) == in_frame) {
        const auto a = std::chrono::steady_clock::now();
        const size_t row = (size_t)w * bps;
        for (int y = 0; y < h; y++) std::memcpy(py.data() + (size_t)y * fp->in_pitch(), raw.data() + y * row, row);
        for (int y = 0; y < h / 2; y++) std::memcpy(puv.data() + (size_t)y * fp->in_pitch(), raw.data() + (size_t)h * row + y * row, row);
        if (!dev->upload(fp->in_y(), py.data(), py.size(), &why) || !dev->upload(fp->in_uv(), puv.data(), puv.size(), &why)) {
            std::fprintf(stderr, "upload: %s\n", why.c_str());
            return 1;
        }
        VkCommandBuffer cb = dev->begin();
        if (qp) {
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        }
        fp->record(cb);
        if (qp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        if (!dev->submit_wait(cb, &why)) {
            std::fprintf(stderr, "run: %s\n", why.c_str());
            return 1;
        }
        if (!dev->download(fp->out_y(), qy.data(), qy.size(), &why) || !dev->download(fp->out_uv(), quv.data(), quv.size(), &why)) {
            std::fprintf(stderr, "download: %s\n", why.c_str());
            return 1;
        }
        const size_t orow = (size_t)ow * bps;
        for (int y = 0; y < oh; y++) std::memcpy(out.data() + y * orow, qy.data() + (size_t)y * fp->out_pitch(), orow);
        for (int y = 0; y < oh / 2; y++)
            std::memcpy(out.data() + (size_t)oh * orow + y * orow, quv.data() + (size_t)y * fp->out_pitch(), orow);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        if (n >= 2) {
            total_ms += ms;
            timed++;
            if (qp) {
                uint64_t t[2];
                vkGetQueryPoolResults(dev->device(), qp, 0, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                gpu_ms += (double)(t[1] - t[0]) * dev->info().timestamp_ns * 1e-6;
            }
        }
        if (fo) std::fwrite(out.data(), 1, out_frame, fo);
        n++;
    }
    if (qp) vkDestroyQueryPool(dev->device(), qp, nullptr);
    std::printf("frames: %d, %.3f ms/frame with transfers, %.3f ms/frame on the GPU (pre + model + post) (%d timed)\n", n,
                timed ? total_ms / timed : 0.0, timed ? gpu_ms / timed : 0.0, timed);
    std::fclose(fi);
    if (fo) std::fclose(fo);
    return n ? 0 : 1;
}
