# animejanai-inference

Inference shim for the native mpv AnimeJaNai pipeline. `libaji` exposes a
strict C ABI (`include/aji.h`) that `vf_animejanai` in the
[the-database/mpv](https://github.com/the-database/mpv) fork loads at
runtime — mpv never links TensorRT, so the backend and TRT version can be
swapped without rebuilding the player, and on Windows the MSVC-built shim
coexists with a mingw-built mpv.

Current backend: TensorRT (`src/aji_trt.cpp`) with CUDA pre/post kernels
(`src/kernels.cu`): NV12/P010 → fp16 NCHW RGB (BT.709 limited) → engine →
NV12/P010, all device-side. Phase 0 spike scope; chain selection,
engine building, colorspace metadata and CUDA graphs land here in phase 1.

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

## mpv usage (spike)

```sh
mpv --hwdec=nvdec \
    --vf=animejanai=engine=models/<model>.engine:lib=build/libaji.so video.mkv
```
