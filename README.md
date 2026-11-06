# animejanai-inference

Optional addon interfaces are documented in [docs/ADDON-API.md](docs/ADDON-API.md), including build dependencies, inactive behavior and qualification limits.

A GPU inference engine for upscaling. `libaji` runs ONNX upscaling models, plus RIFE
interpolation and temporal models, entirely on the GPU, with three backends: TensorRT and
TensorRT-RTX (NVIDIA, through CUDA) and DirectML (Windows, any DirectX 12 GPU). Callers load it at
runtime through one of two C ABIs, so they never link TensorRT:

- **Video** (`include/aji.h`): GPU-resident YUV frames in, upscaled frames out, with chains,
  RIFE and temporal models configured by `animejanai.conf`. `vf_animejanai` in the
  [the-database/mpv](https://github.com/the-database/mpv) fork uses it for real-time playback, and
  `aji_encode` uses it to upscale video files.
- **Still images** (`include/aji_image.h`): 8-bit images of any size, tiled on the GPU.

Because the player loads it at runtime, the backend and TensorRT version can change without
rebuilding mpv, and on Windows the MSVC-built library coexists with a mingw-built mpv.

Used by [mpv-AnimeJaNai](https://github.com/the-database/mpv-AnimeJaNai) (real-time playback),
[VideoJaNai](https://github.com/the-database/VideoJaNai) (video files, through `aji_encode`),
[MangaJaNai GUI](https://github.com/the-database/MangaJaNaiGUI) and
[MangaJaNaiCLI](https://github.com/the-database/MangaJaNaiCLI) (manga and images), and
[yosh](https://github.com/the-database/yosh)'s AI upscaling add-on.

## Build (Linux)

Needs CUDA toolkit 13.x and TensorRT 11.x (default path:
`~/sdk/tensorrt/usr`, override with `-DAJI_TRT_ROOT=`; on a system TensorRT
install from NVIDIA's apt repo, pass `-DAJI_TRT_ROOT=/usr`).

```sh
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S .
cmake --build build -j
```

## Engine

```sh
trtexec --onnx=models/<model>.onnx --fp16 \
        --minShapes=input:1x3x64x64 --optShapes=input:1x3x1080x1920 \
        --maxShapes=input:1x3x1088x1920 --saveEngine=models/<model>.engine
```

## Harness (no player needed)

```sh
ffmpeg -i clip.mkv -frames:v 12 -pix_fmt p010le -f rawvideo in.raw
./build/aji_harness --engine models/<model>.engine --input in.raw \
    --width 1400 --height 1080 --format p010 --output out.raw
```

Reports device-side ms/frame for the full pre+infer+post chain
(RTX 5090 reference: ~3.8 ms at 1920x1080→4K fp16).

## Still images (`include/aji_image.h`)

A separate C ABI for upscaling single images with any ONNX super-resolution
model — no conf, slots or YUV. 8-bit gray or RGB pixels go in from host
memory, the upscaled image comes back; tiling, overlap blending, padding and
dtype conversion happen inside.

```c
#include "aji_image.h"

aji_img_create_params cp = {0};
cp.api_version = AJI_IMAGE_API_VERSION;
cp.backend = AJI_IMG_DIRECTML;          /* or AJI_IMG_TENSORRT */
char err[256];
aji_img_ctx *c = aji_img_create(&cp, err, sizeof err);

aji_img_model_params mp = {0};
mp.onnx_path = "4x_model.onnx";
aji_img_model *m = aji_img_load(c, &mp);  /* NULL: aji_img_last_error(c) */

aji_img_model_info info;
aji_img_model_info_get(m, &info);         /* info.scale sizes the output */

aji_img_buffer in  = {w, h, 3, w * 3, pixels};
aji_img_buffer out = {w * info.scale, h * info.scale, 3,
                      w * info.scale * 3, out_pixels};
aji_img_upscale(m, &in, &out, NULL, NULL);

aji_img_release(&m);
aji_img_destroy(&c);
```

Load once and reuse: creating the context and loading a model are the slow
parts (DirectML compiles the model; TensorRT deserializes or builds an engine),
upscaling is per image. `src/aji_image_cli.cpp` (the `aji_image` tool, PNM in
and out) is the complete working example.

Things the header can't tell you:

- **Deployment.** Load `aji.dll` (`libaji.so`); it loads `aji_dml` / `aji_trt`
  from its own directory. Their runtimes must sit beside them: `onnxruntime.dll`
  + `DirectML.dll` for DirectML; `nvinfer_11`, `nvinfer_plugin_11`,
  `nvonnxparser_11`, `cudart64_13`, the builder resource for the GPU's SM and
  `trtexec` for TensorRT (`trtexec` can be passed explicitly instead).
- **First TensorRT load builds an engine** — minutes, once per model, tile
  size, GPU and TensorRT version; later loads hit the cache (`cache_dir`, or
  the model's directory). `aji_img_load` blocks through the build by default;
  interactive apps pass `AJI_IMG_LOAD_ASYNC_BUILD` and poll
  `aji_img_model_state` instead.
- **Models**: one NCHW input and one NCHW output, 1 or 3 channels, an integer
  scale, fp32/fp16 (TensorRT also bf16), dynamic or fixed spatial shape (a fixed
  shape becomes the tile). For DirectML, export at **opset 19 or lower** — the
  DirectML EP lacks newer ops such as opset-20 `Gelu`, which then fall back to
  the CPU and run tens of times slower.
- **Channels**: a gray image fed to a 3-channel model is replicated in and
  channel 0 is kept. Alpha is not handled: composite onto a background, or
  upscale the alpha yourself, before and after calling.
- **Threading**: a context works from any thread; calls on one model are
  serialized, different models may run concurrently.

## mpv usage (single engine, for testing)

```sh
mpv --hwdec=nvdec \
    --vf=animejanai=engine=models/<model>.engine:lib=build/libaji.so video.mkv
```
