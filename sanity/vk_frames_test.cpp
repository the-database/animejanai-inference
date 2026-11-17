/*
 * aji_vk_frames_test — the Vulkan-frames ABI (aji_create_vk) as a player uses it: a device of its
 * own playing the caller (vk/device.h's Device, standing in for mpv's libplacebo device), raw
 * NV12 / P010 frames uploaded into multi-planar VkImages with a timeline semaphore each (FFmpeg's
 * AVVkFrame), through aji.dll / libaji.so's aji_create_vk, aji_configure, aji_infer and the
 * tickets, the output images read back into a raw file.
 *
 *   aji_vk_frames_test --conf C --model-dir D --slot N --input in.raw --width W --height H
 *                      [--frames N] [--output out.raw] [--format nv12|p010] [--out-format nv12|p010]
 *                      [--no-coopmat] [--rife --rife-model-dir R]
 *
 * --no-coopmat tells libaji the device was created without cooperative matrices (they are left
 * out of the features chain and the extension list): the family it then picks is the one a
 * player that didn't enable them gets. Raw files as aji_harness_dml's: per frame the Y plane, then
 * the interleaved CbCr plane; 16-bit little-endian MSB-aligned for P010.
 *
 * A temporal chain (aji_temporal_radius > 0) runs as aji_harness_dml's: frames ingested in order,
 * r ahead of the one upscaled (aji_ingest, aji_infer_seq), --reset-at K resetting before frame K.
 *
 * --diff: aji_frame_diff of each frame against the one before, one line each: the pair, max_tile
 * and the 16 region means (%.9g), for an exact comparison with a reference.
 *
 * --rife: aji_harness_dml --rife's interpolation-only loop (a scale-free 2/1 RIFE chain): writes
 * f0, i01, f1, i12, ..., the left frame in place of an interpolated one on a scene change.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "aji.h"
#include "vk/device.h"

using namespace ajivk;

namespace {

void log_cb(void *, int level, const char *msg) { std::fprintf(stderr, "[aji:%d] %s\n", level, msg); }

struct Image {
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkSemaphore sem = VK_NULL_HANDLE;
    aji_vk_image d{};
};

bool make_image(Device &dev, int w, int h, bool p010, Image *im, std::string *why)
{
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = p010 ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    ci.extent = {(uint32_t)w, (uint32_t)h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev.device(), &ci, nullptr, &im->img) != VK_SUCCESS) return *why = "vkCreateImage", false;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev.device(), im->img, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(dev.physical(), &mp);
    uint32_t type = 0;
    while (type < mp.memoryTypeCount &&
           !((mr.memoryTypeBits & (1u << type)) && (mp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
        type++;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if (type == mp.memoryTypeCount || vkAllocateMemory(dev.device(), &ai, nullptr, &im->mem) != VK_SUCCESS ||
        vkBindImageMemory(dev.device(), im->img, im->mem, 0) != VK_SUCCESS)
        return *why = "image memory", false;
    VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st};
    if (vkCreateSemaphore(dev.device(), &si, nullptr, &im->sem) != VK_SUCCESS) return *why = "semaphore", false;
    im->d.image = im->img;
    im->d.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    im->d.queue_family = VK_QUEUE_FAMILY_IGNORED;
    im->d.semaphore = im->sem;
    im->d.value = 0;
    return true;
}

// The caller's own transfer between a host-visible staging buffer and an image's two planes, on
// its queue, synchronised the AVVkFrame way (wait value, signal value + 1).
bool transfer(Device &dev, Image &im, const Buffer &stage, int w, int h, int bps, bool up, std::string *why)
{
    const int frame_y = w * h * bps;
    int slot = 0;
    VkCommandBuffer cb = dev.frame_begin(&slot);
    if (!cb) return *why = "frame_begin", false;
    const VkImageLayout lay = up ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = up ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = up ? VK_IMAGE_LAYOUT_UNDEFINED : static_cast<VkImageLayout>(im.d.layout);
    b.newLayout = lay;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy r[2] = {};
    r[0].bufferRowLength = (uint32_t)w;
    r[0].imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
    r[0].imageExtent = {(uint32_t)w, (uint32_t)h, 1};
    r[1].bufferOffset = (VkDeviceSize)frame_y;
    r[1].bufferRowLength = (uint32_t)(w / 2);
    r[1].imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
    r[1].imageExtent = {(uint32_t)(w / 2), (uint32_t)(h / 2), 1};
    if (up) vkCmdCopyBufferToImage(cb, stage.buf, im.img, lay, 2, r);
    else vkCmdCopyImageToBuffer(cb, im.img, lay, stage.buf, 2, r);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const Device::SemValue wait{im.sem, im.d.value}, sig{im.sem, im.d.value + 1};
    if (!dev.frame_submit(slot, &wait, 1, &sig, 1, why)) return false;
    im.d.value += 1;
    im.d.layout = lay;
    // the caller waits here for the download only
    if (!up) {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &im.sem;
        wi.pValues = &im.d.value;
        if (vkWaitSemaphores(dev.device(), &wi, 10ull * 1000 * 1000 * 1000) != VK_SUCCESS) return *why = "download wait", false;
    }
    return true;
}

// aji_harness_dml --rife's loop on Vulkan frames
bool run_rife(Device &dev, aji_ctx *c, const char *input, const char *output, int w, int h, int frames, bool p010)
{
    int rn = 0, rd = 0;
    if (!aji_rife_factor(c, &rn, &rd) || rn != 2 * rd) {
        std::fprintf(stderr, "rife test needs an active scale-free 2/1 rife chain\n");
        return false;
    }
    std::string why;
    const int bps = p010 ? 2 : 1;
    const size_t fb = (size_t)w * h * 3 / 2 * bps;
    Image fr[3];
    Buffer st;
    for (Image &im : fr)
        if (!make_image(dev, w, h, p010, &im, &why)) {
            std::fprintf(stderr, "rife images: %s\n", why.c_str());
            return false;
        }
    if (!dev.make_buffer(&st, fb, true, &why)) return false;
    aji_frame fd[3];
    for (int i = 0; i < 3; i++) {
        fd[i] = aji_frame{};
        fd[i].width = w;
        fd[i].height = h;
        fd[i].format = p010 ? AJI_FMT_P010 : AJI_FMT_NV12;
        fd[i].matrix = AJI_MATRIX_BT709;
        fd[i].range = AJI_RANGE_LIMITED;
        fd[i].siting = AJI_SITING_LEFT;
        fd[i].plane[0] = &fr[i].d;
    }
    std::ifstream fi(input, std::ios::binary);
    std::ofstream fo;
    if (output) fo.open(output, std::ios::binary);
    std::vector<char> raw(fb);
    int prev = -1, cur = 0, n = 0, interp = 0, scenes = 0, timed = 0;
    double ms = 0;
    while (n < frames && fi.read(raw.data(), (std::streamsize)fb)) {
        std::memcpy(st.map, raw.data(), fb);
        if (!transfer(dev, fr[cur], st, w, h, bps, true, &why)) return false;
        if (prev >= 0) {
            const auto t0 = std::chrono::steady_clock::now();
            const int ret = aji_infer_rife_with_scene(c, &fd[prev], &fd[cur], 0.5, &fd[2], nullptr, -1);
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (interp + scenes >= 1) ms += t, timed++;
            int emit;
            if (ret == AJI_OK) emit = 2, interp++;
            else if (ret == AJI_SCENE) emit = prev, scenes++;
            else {
                std::fprintf(stderr, "aji_infer_rife: %d: %s\n", ret, aji_last_error(c));
                return false;
            }
            if (!transfer(dev, fr[emit], st, w, h, bps, false, &why)) return false;
            if (fo) fo.write(static_cast<const char *>(st.map), (std::streamsize)fb);
        }
        if (fo) fo.write(raw.data(), (std::streamsize)fb);
        if (prev < 0) prev = cur, cur = 1;
        else std::swap(prev, cur);
        n++;
    }
    std::printf("rife: %d source frames, %d interpolated, %d scene skips, %.1f ms/interp avg (%d timed)\n", n, interp, scenes,
                timed ? ms / timed : 0.0, timed);
    aji_destroy(&c);
    dev.destroy(st);
    for (Image &im : fr) {
        vkDestroySemaphore(dev.device(), im.sem, nullptr);
        vkDestroyImage(dev.device(), im.img, nullptr);
        vkFreeMemory(dev.device(), im.mem, nullptr);
    }
    return true;
}

// aji_harness_dml's temporal loop on Vulkan frames
bool run_temporal(Device &dev, aji_ctx *c, Image &in, Image &out, Buffer &sin, Buffer &sout, const char *input, const char *output,
                  int w, int h, int ow, int oh, int frames, bool p010, bool out_p010, int reset_at)
{
    const int r = aji_temporal_radius(c), ibps = p010 ? 2 : 1, obps = out_p010 ? 2 : 1;
    const size_t in_bytes = (size_t)w * h * 3 / 2 * ibps, out_bytes = (size_t)ow * oh * 3 / 2 * obps;
    std::printf("temporal: %d-frame window, reset at %d\n", 2 * r + 1, reset_at);
    std::ifstream fi(input, std::ios::binary);
    std::vector<char> all;
    std::vector<char> buf(in_bytes);
    size_t nfr = 0;
    while ((int)nfr < frames && fi.read(buf.data(), (std::streamsize)in_bytes)) {
        all.insert(all.end(), buf.begin(), buf.end());
        nfr++;
    }
    std::ofstream fo;
    if (output) fo.open(output, std::ios::binary);
    aji_frame fin{}, fout{};
    fin.width = w;
    fin.height = h;
    fin.format = p010 ? AJI_FMT_P010 : AJI_FMT_NV12;
    fin.matrix = AJI_MATRIX_BT709;
    fin.range = AJI_RANGE_LIMITED;
    fin.siting = AJI_SITING_LEFT;
    fin.plane[0] = &in.d;
    fout = fin;
    fout.width = ow;
    fout.height = oh;
    fout.format = out_p010 ? AJI_FMT_P010 : AJI_FMT_NV12;
    fout.plane[0] = &out.d;
    std::string why;
    size_t base = 0, next = 0;
    int n = 0;
    double total = 0;
    for (size_t k = 0; k < nfr; k++) {
        if ((int)k == reset_at) {
            aji_temporal_reset(c);
            base = next = k;
        }
        const auto t0 = std::chrono::steady_clock::now();
        while (next < nfr && next <= k + r) {
            std::memcpy(sin.map, all.data() + next * in_bytes, in_bytes);
            if (!transfer(dev, in, sin, w, h, ibps, true, &why)) return false;
            if (aji_ingest(c, &fin, (uint64_t)(next - base), nullptr) != AJI_OK || aji_wait(c, aji_flush(c, nullptr)) != AJI_OK) {
                std::fprintf(stderr, "aji_ingest: %s\n", aji_last_error(c));
                return false;
            }
            next++;
        }
        if (aji_infer_seq(c, (uint64_t)(k - base), &fout, nullptr) != AJI_OK || aji_wait(c, aji_flush(c, nullptr)) != AJI_OK) {
            std::fprintf(stderr, "aji_infer_seq: %s\n", aji_last_error(c));
            return false;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (k >= 2) total += ms;
        if (!transfer(dev, out, sout, ow, oh, obps, false, &why)) return false;
        if (fo) fo.write(static_cast<const char *>(sout.map), (std::streamsize)out_bytes);
        n++;
    }
    std::printf("frames: %d, %dx%d -> %dx%d, %.3f ms a frame (ingest + infer_seq to the ticket, after the first two)\n", n, w, h, ow,
                oh, n > 2 ? total / (n - 2) : 0.0);
    return true;
}

// aji_frame_diff of consecutive frames
bool run_diff(Device &dev, aji_ctx *c, const char *input, int w, int h, int frames, bool p010)
{
    std::string why;
    const int bps = p010 ? 2 : 1;
    const size_t fb = (size_t)w * h * 3 / 2 * bps;
    Image im[2];
    Buffer st;
    for (Image &i : im)
        if (!make_image(dev, w, h, p010, &i, &why)) return false;
    if (!dev.make_buffer(&st, fb, true, &why)) return false;
    std::ifstream fi(input, std::ios::binary);
    std::vector<char> raw(fb);
    aji_frame f[2];
    for (int i = 0; i < 2; i++) {
        f[i] = aji_frame{};
        f[i].width = w;
        f[i].height = h;
        f[i].format = p010 ? AJI_FMT_P010 : AJI_FMT_NV12;
        f[i].matrix = AJI_MATRIX_BT709;
        f[i].range = AJI_RANGE_LIMITED;
        f[i].siting = AJI_SITING_LEFT;
        f[i].plane[0] = &im[i].d;
    }
    int n = 0;
    double ms = 0;
    while (n < frames && fi.read(raw.data(), (std::streamsize)fb)) {
        std::memcpy(st.map, raw.data(), fb);
        if (!transfer(dev, im[n % 2], st, w, h, bps, true, &why)) return false;
        // the upload's semaphore value orders the comparison; the staging buffer is reused next
        // frame, so wait for the upload here
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &im[n % 2].sem;
        wi.pValues = &im[n % 2].d.value;
        vkWaitSemaphores(dev.device(), &wi, 10ull * 1000 * 1000 * 1000);
        if (n > 0) {
            aji_frame_diff_result r{};
            const auto t0 = std::chrono::steady_clock::now();
            const int ret = aji_frame_diff(c, &f[(n - 1) % 2], &f[n % 2], &r, nullptr);
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ret != AJI_OK) {
                std::fprintf(stderr, "aji_frame_diff: %d: %s\n", ret, aji_last_error(c));
                return false;
            }
            std::printf("diff %d %d %.9g", n - 1, n, r.max_tile);
            for (float v : r.region_mean) std::printf(" %.9g", v);
            std::printf("\n");
        }
        n++;
    }
    std::printf("frame diff: %d pairs, %.3f ms a pair\n", n - 1, n > 1 ? ms / (n - 1) : 0.0);
    dev.destroy(st);
    for (Image &i : im) {
        vkDestroySemaphore(dev.device(), i.sem, nullptr);
        vkDestroyImage(dev.device(), i.img, nullptr);
        vkFreeMemory(dev.device(), i.mem, nullptr);
    }
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    const char *conf = nullptr, *model_dir = nullptr, *input = nullptr, *output = nullptr, *rife_dir = nullptr;
    int w = 0, h = 0, frames = 1 << 30, slot = 1, reset_at = -1;
    bool p010 = false, out_p010 = false, no_cm = false, rife = false, diff = false;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!std::strcmp(a, "--no-coopmat")) {
            no_cm = true;
            continue;
        }
        if (!std::strcmp(a, "--rife")) {
            rife = true;
            continue;
        }
        if (!std::strcmp(a, "--diff")) {
            diff = true;
            continue;
        }
        i++;
        if (!std::strcmp(a, "--conf")) conf = v;
        else if (!std::strcmp(a, "--model-dir")) model_dir = v;
        else if (!std::strcmp(a, "--rife-model-dir")) rife_dir = v;
        else if (!std::strcmp(a, "--input")) input = v;
        else if (!std::strcmp(a, "--output")) output = v;
        else if (!std::strcmp(a, "--width")) w = std::atoi(v);
        else if (!std::strcmp(a, "--height")) h = std::atoi(v);
        else if (!std::strcmp(a, "--frames")) frames = std::atoi(v);
        else if (!std::strcmp(a, "--slot")) slot = std::atoi(v);
        else if (!std::strcmp(a, "--reset-at")) reset_at = std::atoi(v);
        else if (!std::strcmp(a, "--format")) p010 = !std::strcmp(v, "p010");
        else if (!std::strcmp(a, "--out-format")) out_p010 = !std::strcmp(v, "p010");
    }
    if (!conf || !model_dir || !input || w < 2 || h < 2) {
        std::fprintf(stderr, "usage: aji_vk_frames_test --conf C --model-dir D --slot N --input in.raw --width W --height H "
                             "[--frames N] [--output out.raw] [--format nv12|p010] [--out-format nv12|p010] [--no-coopmat]\n");
        return 2;
    }
    std::string why;
    // the caller's device
    auto dev = Device::create(-1, nullptr, &why);
    if (!dev) {
        std::fprintf(stderr, "caller device: %s\n", why.c_str());
        return 1;
    }
    const bool cm = dev->family() == Family::CM16 && !no_cm;
    const bool sg = dev->family() != Family::PLAIN;
    // what Device::create enabled (vk/device.cpp), as the player would report its own device's
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, cm ? &fcm : nullptr};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f11.storageBuffer16BitAccess = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.vulkanMemoryModel = f12.vulkanMemoryModelDeviceScope = dev->family() == Family::CM16 ? VK_TRUE : VK_FALSE;
    f13.synchronization2 = VK_TRUE;
    f13.subgroupSizeControl = f13.computeFullSubgroups = sg ? VK_TRUE : VK_FALSE;
    fcm.cooperativeMatrix = VK_TRUE;
    const char *exts[1] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    aji_vk_device vd{};
    vd.struct_size = sizeof vd;
    vd.physical_device = dev->physical();
    vd.device = dev->device();
    vd.queue = dev->queue();
    vd.queue_family = dev->queue_family();
    vd.enabled_extensions = cm ? exts : nullptr;
    vd.num_enabled_extensions = cm ? 1 : 0;
    vd.enabled_features = &f2;
    std::printf("caller device: %s (family %s; reported: cooperative matrices %s)\n", dev->info().name.c_str(), family_name(dev->family()),
                cm ? "on" : "off");

    aji_create_params p{};
    p.api_version = AJI_API_VERSION;
    p.conf_path = conf;
    p.model_dir = model_dir;
    p.slot = slot;
    p.rife_model_dir = rife_dir;
    p.log = log_cb;
    aji_ctx *c = aji_create_vk(&p, &vd);
    if (!c) {
        std::fprintf(stderr, "aji_create_vk failed\n");
        return 1;
    }
    int ow = 0, oh = 0;
    const int act = aji_configure(c, w, h, 23.976, &ow, &oh);
    std::printf("%s", aji_current_log(c));
    if (rife) return run_rife(*dev, c, input, output, w, h, frames, p010) ? 0 : 1;
    if (diff) {
        const bool ok = run_diff(*dev, c, input, w, h, frames, p010);
        aji_destroy(&c);
        return ok ? 0 : 1;
    }
    if (act != 1) {
        std::fprintf(stderr, "aji_configure: %d (%s)\n", act, aji_last_error(c));
        aji_destroy(&c);
        return act == 0 ? 0 : 1;
    }
    const int ibps = p010 ? 2 : 1, obps = out_p010 ? 2 : 1;
    const size_t in_bytes = (size_t)w * h * 3 / 2 * ibps, out_bytes = (size_t)ow * oh * 3 / 2 * obps;
    Image in, out;
    Buffer sin, sout;
    if (!make_image(*dev, w, h, p010, &in, &why) || !make_image(*dev, ow, oh, out_p010, &out, &why) ||
        !dev->make_buffer(&sin, in_bytes, true, &why) || !dev->make_buffer(&sout, out_bytes, true, &why)) {
        std::fprintf(stderr, "caller resources: %s\n", why.c_str());
        return 1;
    }
    if (aji_temporal_radius(c) > 0) {
        const bool ok = run_temporal(*dev, c, in, out, sin, sout, input, output, w, h, ow, oh, frames, p010, out_p010, reset_at);
        aji_destroy(&c);
        return ok ? 0 : 1;
    }
    std::ifstream fi(input, std::ios::binary);
    std::ofstream fo;
    if (output) fo.open(output, std::ios::binary);
    std::vector<char> buf(in_bytes);
    int n = 0;
    double total = 0;
    while (n < frames && fi.read(buf.data(), (std::streamsize)in_bytes)) {
        std::memcpy(sin.map, buf.data(), in_bytes);
        if (!transfer(*dev, in, sin, w, h, ibps, true, &why)) {
            std::fprintf(stderr, "upload: %s\n", why.c_str());
            return 1;
        }
        aji_frame fin{}, fout{};
        fin.width = w;
        fin.height = h;
        fin.format = p010 ? AJI_FMT_P010 : AJI_FMT_NV12;
        fin.matrix = AJI_MATRIX_BT709;
        fin.range = AJI_RANGE_LIMITED;
        fin.siting = AJI_SITING_LEFT;
        fin.plane[0] = &in.d;
        fout = fin;
        fout.width = ow;
        fout.height = oh;
        fout.format = out_p010 ? AJI_FMT_P010 : AJI_FMT_NV12;
        fout.plane[0] = &out.d;
        const auto t0 = std::chrono::steady_clock::now();
        if (aji_infer(c, &fin, &fout, nullptr) != AJI_OK || aji_wait(c, aji_flush(c, nullptr)) != AJI_OK) {
            std::fprintf(stderr, "aji_infer: %s\n", aji_last_error(c));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (n >= 2) total += ms;
        if (!transfer(*dev, out, sout, ow, oh, obps, false, &why)) {
            std::fprintf(stderr, "download: %s\n", why.c_str());
            return 1;
        }
        if (fo) fo.write(static_cast<const char *>(sout.map), (std::streamsize)out_bytes);
        n++;
    }
    std::printf("frames: %d, %dx%d -> %dx%d, %.3f ms a frame (infer to its ticket, frames after the first two); image values %llu / %llu\n",
                n, w, h, ow, oh, n > 2 ? total / (n - 2) : 0.0, (unsigned long long)in.d.value, (unsigned long long)out.d.value);
    aji_destroy(&c);
    dev->destroy(sin);
    dev->destroy(sout);
    for (Image *im : {&in, &out}) {
        vkDestroySemaphore(dev->device(), im->sem, nullptr);
        vkDestroyImage(dev->device(), im->img, nullptr);
        vkFreeMemory(dev->device(), im->mem, nullptr);
    }
    return 0;
}
