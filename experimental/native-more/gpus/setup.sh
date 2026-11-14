#!/bin/bash
# Prepares a Linux GPU box (Ubuntu 24.04, CUDA toolkit at /usr/local/cuda, e.g. Runpod's PyTorch
# template) for bench.sh: TensorRT 11.4 and cudart from NVIDIA's repos, this branch built, the
# MangaJaNai / IllustrationJaNai image models from their public releases, and random-weight TFDAT
# v2 exports (needs torch). The video models, clips and confs come from up.tar (bundle.sh).
# usage: R=/workspace setup.sh
set -e
R=${R:-/workspace}; mkdir -p $R/sdk $R/models/img $R/tfv2; cd $R
(apt-get -qq update && apt-get -qq install -y ninja-build p7zip-full >/dev/null) 2>&1 | grep -v debconf | tail -2 || true
cd $R/sdk
repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64; ver=11.4.0.106-1+cuda13.4; mkdir -p trt
for p in libnvinfer-headers-dev libnvinfer-dev libnvinfer11 libnvinfer-plugin11 libnvonnxparsers11 libnvinfer-bin \
         libnvinfer-headers-plugin-dev libnvinfer-plugin-dev libnvonnxparsers-dev; do
  [ -f trt/.$p ] || { curl -fsSL -o $p.deb "$repo/${p}_${ver}_amd64.deb" && dpkg-deb -x $p.deb trt && rm $p.deb && touch trt/.$p; }
done
[ -d cudart ] || { curl -fsSL -o cr.tar.xz https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.4.49-archive.tar.xz \
  && mkdir -p cudart && tar -xf cr.tar.xz -C cudart --strip-components=1 && rm cr.tar.xz; }
cd $R
[ -d src ] || git clone -q --branch experimental/native-more --single-branch https://github.com/the-database/animejanai-inference.git src
git -C src log --oneline -1
[ -f build/build.ninja ] || cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DAJI_TRT_ROOT=$R/sdk/trt/usr > configure.log 2>&1 || { tail -20 configure.log; exit 1; }
cmake --build build -j $(nproc) --target aji_trt aji aji_harness aji_image aji_native_pack_test aji_native_lower_test \
  aji_native_conv_test aji_native_gconv_test aji_native_tfm_test aji_native_vid_run > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "build ok"
# image models
cd $R/models/img
for a in "1.0.0 MangaJaNai_V1_ONNX_fp16.7z" "3.0.0 IllustrationJaNai_V3_ONNX_fp16.7z" "3.0.0 IllustrationJaNai_V3_ONNX_bf16.7z"; do
  set -- $a
  [ -f .$2 ] || { curl -fsSL -o $2 "https://github.com/the-database/MangaJaNai/releases/download/$1/$2" && 7z e -y -bd $2 '*.onnx' -r >/dev/null && rm $2 && touch .$2; }
done
ls | wc -l
# TFDAT v2 random exports (1x, 4x)
cd $R/tfv2
if [ -f export.py ] && python3 -c "import torch" 2>/dev/null; then
  python3 -c "import onnx" 2>/dev/null || pip -q install onnx >/dev/null 2>&1
  for sc in 1 4; do [ -f ${sc}x_tfdat2_rand_c5_fp16.onnx ] || python3 export.py $sc 5 . 2>&1 | grep -E "wrote|Error"; done
fi
echo "setup done"
