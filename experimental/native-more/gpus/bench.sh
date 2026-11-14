#!/bin/bash
# Native kernels vs TensorRT on a Linux GPU box: every shipped model the native kernels cover, at
# libaji level (aji_harness for video, aji_image for images), each case native=off then on with
# the same build, outputs compared. Prints one line per case; results.md collects them.
# Layout under R: build/ (this branch), sdk/{trt,cudart}, models/ (onnx; rife/ below it; img/ the
# image packs' onnx), tfv2/ (random TFDAT v2 exports), in/ (clips, page.ppm), conf/ (the confs).
# usage: bench.sh [--cached]   (--cached skips the pass that builds the TensorRT engines)
R=${R:-/workspace}; B=$R/build; SDK=$R/sdk; I=$R/in; C=$R/conf; O=$R/bench; mkdir -p $O
export LD_LIBRARY_PATH=$SDK/trt/usr/lib/x86_64-linux-gnu:$SDK/cudart/lib
export CUDA_VISIBLE_DEVICES=${GPU:-0}
TRTEXEC=$SDK/trt/usr/bin/trtexec
PY=${PY:-python3}  # with numpy, for the output comparisons

vid() {  # vid NAME CLIP W H FRAMES CONF SLOT MODELDIR [harness args]
  local name=$1 clip=$2 w=$3 h=$4 n=$5 conf=$6 slot=$7 md=$8; shift 8
  run() { (cd $B && AJI_TEMPORAL_STATS=1 ./aji_harness --input $I/$clip --width $w --height $h --format nv12 --frames $n \
          --conf $C/${conf}_$1.conf --model-dir $md --rife-model-dir $R/models/rife --trtexec $TRTEXEC --slot $slot \
          --output $O/o_$1.raw "$@" 2>&1); }
  local off on dims t_off t_on nat err r
  off=$(run off "$@"); on=$(run on "$@")
  echo "$off" > $O/$name.off.log; echo "$on" > $O/$name.on.log
  dims=$(echo "$on" | grep -oE 'configured: [0-9]+x[0-9]+ [a-z0-9]+ -> [0-9]+x[0-9]+' | grep -oE '[0-9]+x[0-9]+$')
  if [ "$1" = "--rife" ]; then dims=${w}x$h; fi
  t_off=$(echo "$off" | grep -oE 'device chain time: [0-9.]+|rife time: [0-9.]+' | head -1 | grep -oE '[0-9.]+' | head -1)
  t_on=$(echo "$on" | grep -oE 'device chain time: [0-9.]+|rife time: [0-9.]+' | head -1 | grep -oE '[0-9.]+' | head -1)
  nat=$(echo "$on" | grep -cE 'Native kernels: ')
  err=$(echo "$on$off" | grep -iE 'error|failed|not used' | head -1)
  r=$( [ -n "$dims" ] && $PY -I $C/cmp.py $O/o_off.raw $O/o_on.raw ${dims%x*} ${dims#*x} nv12 )
  echo "$name | ${w}x$h -> $dims | native steps $nat | TensorRT $t_off ms | native $t_on ms | $(python3 -c "print('%.2fx' % ($t_off / $t_on))" 2>/dev/null) | $r ${err:+| NOTE: $err}"
  rm -f $O/o_off.raw $O/o_on.raw
}

img() {  # img NAME MODEL.onnx [tile]
  local name=$1 model=$2 tile=${3:-512}
  run() { (cd $B && AJI_IMAGE_VERBOSE=1 AJI_NATIVE=$1 ./aji_image --model $model --tile $tile --cache $R/imgcache --trtexec $TRTEXEC --repeat 4 \
          $I/page.ppm $O/i_$1.ppm 2>&1); }
  local off on t_off t_on nat err
  off=$(run off); on=$(run on)
  echo "$off" > $O/$name.off.log; echo "$on" > $O/$name.on.log
  t_off=$(echo "$off" | grep -E '^upscale' | tail -1 | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+')
  t_on=$(echo "$on" | grep -E '^upscale' | tail -1 | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+')
  nat=$(echo "$on" | grep -ciE 'native kernels: ')
  err=$(echo "$on$off" | grep -iE 'error|failed|not used' | head -1)
  r=$($PY -I $C/ppmcmp.py $O/i_off.ppm $O/i_on.ppm)
  echo "$name | page 1200x1000, tile $tile | native $nat | TensorRT $t_off ms | native $t_on ms | $(python3 -c "print('%.2fx' % ($t_off / $t_on))" 2>/dev/null) | $r ${err:+| NOTE: $err}"
  rm -f $O/i_off.ppm $O/i_on.ppm
}

M=$R/models; G=$R/models/img
{
echo "# $(nvidia-smi -i ${GPU:-0} --query-gpu=name,compute_cap,power.limit,clocks.max.sm,memory.total,driver_version --format=csv,noheader)"
echo "# $(date -u +%FT%TZ), branch $(git -C $R/src log --oneline -1)"
echo "== native tests"
(cd $B && for t in aji_native_pack_test aji_native_lower_test aji_native_conv_test aji_native_gconv_test aji_native_tfm_test; do
   [ -x ./$t ] && echo "$t: $(./$t 2>&1 | tail -1)"; done)
echo "== video (aji_harness, device chain time per output frame; RIFE: per interpolated frame)"
vid span-perf-1080 ninja1080_nv12.raw 1920 1080 12 span 2 $M
vid span-bal-1080 ninja1080_nv12.raw 1920 1080 12 span 3 $M
vid span-compact-854 ninja854_nv12.raw 854 480 12 span 1 $M
vid rife-4.26-1080 ninja1080_nv12.raw 1920 1080 12 rife 9 $M --rife
vid rife-4.22lite-1080 ninja1080_nv12.raw 1920 1080 12 rife 7 $M --rife
vid rife-4.14-1080 ninja1080_nv12.raw 1920 1080 12 rife 3 $M --rife
vid tfix-dvd dvd768_nv12.raw 768 576 12 tx 1 $M
vid tfix-720 event720_nv12.raw 1280 720 12 tx 1 $M
vid tfix-1080 live1080_nv12.raw 1920 1080 12 tx 1 $M
vid tfdat-bndl-720 event720_nv12.raw 1280 720 12 tf 1 $M
vid tfdat-bndl-1080 ninja1080_nv12.raw 1920 1080 12 tf 1 $M
[ -f $R/tfv2/1x_tfdat2_rand_c5_fp16.onnx ] && vid tfdat-v2-1x-720 event720_nv12.raw 1280 720 12 h1 1 $R/tfv2
[ -f $R/tfv2/4x_tfdat2_rand_c5_fp16.onnx ] && vid tfdat-v2-4x-dvd dvd768_nv12.raw 768 576 8 h4 1 $R/tfv2
echo "== images (aji_image, one 1200x1000 page, warm)"
img esrgan-2x $G/2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx
img esrgan-4x $G/4x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx
img fdat-2x $G/2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.onnx
img fdat-4x $G/4x_IllustrationJaNai_V3detail_FDAT_M_40k_fp16.onnx
img hat-l-4x $G/4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx
img dat2-4x $G/4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx
} 2>&1 | tee $O/results.txt
