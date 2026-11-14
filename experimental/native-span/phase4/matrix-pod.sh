#!/bin/bash
# Phase 3 matrix on Linux: each case through aji_harness with native=off and native=on (same build), all frames compared.
R=/workspace; D=$R/p3; B=$R/build; SDK=$R/sdk
export LD_LIBRARY_PATH=$SDK/trt/usr/lib/x86_64-linux-gnu:$SDK/trtrtx/lib:$SDK/cudart/lib
export CUDA_VISIBLE_DEVICES=${GPU:-0}
h() { (cd $B && ./aji_harness --input $D/in/$1.raw --width $2 --height $3 --format $4 --frames $5 \
      --conf $D/$6.conf --model-dir $R/models --rife-model-dir $R/models/rife \
      --trtexec $SDK/trt/usr/bin/trtexec --slot $7 "${@:8}" 2>&1); }
cases=(
 "perf-1080 ninja1080_nv12 1920 1080 nv12 2"
 "bal-1080 ninja1080_nv12 1920 1080 nv12 3"
 "perf-p010 ninja1080_p010 1920 1080 p010 2"
 "perf-1918x1078 ninja1918_nv12 1918 1078 nv12 2"
 "bal-1918x1078 ninja1918_nv12 1918 1078 nv12 3"
 "bal-1440x1080 ninja1440_nv12 1440 1080 nv12 3"
 "compact-854x480 ninja854_nv12 854 480 nv12 1"
 "perf-854x480 ninja854_nv12 854 480 nv12 2"
 "compact-dvd768x576 dvd768_nv12 768 576 nv12 1"
 "perf-live1080 live1080_nv12 1920 1080 nv12 2"
 "compact-live1080 live1080_nv12 1920 1080 nv12 1"
 "perf-event1280x642 oso1280x642_nv12 1280 642 nv12 2"
 "chain-bal+perf-854 ninja854_nv12 854 480 nv12 6"
 "resize50+perf-1080 ninja1080_nv12 1920 1080 nv12 7"
 "temporal+perf-720 ninja720_nv12 1280 720 nv12 9"
)
# pass 1 builds every TensorRT engine (results discarded), so pass 2 times cached engines only
if [ "$1" != "--cached" ]; then
  for c in "${cases[@]}"; do set -- $c; h $2 $3 $4 $5 2 off $6 > /dev/null; done
  h ninja1080_nv12 1920 1080 nv12 2 off 8 --rife-chain > /dev/null
fi
for c in "${cases[@]}"; do set -- $c
  off=$(h $2 $3 $4 $5 12 off $6 --output $D/o_off.raw)
  on=$(h $2 $3 $4 $5 12 on $6 --output $D/o_on.raw)
  dims=$(echo "$on" | grep -oE 'configured: [0-9]+x[0-9]+ [a-z0-9]+ -> [0-9]+x[0-9]+' | grep -oE '[0-9]+x[0-9]+$')
  nat=$(echo "$on" | grep -cE 'Native kernels: ')
  not=$(echo "$on" | grep -oE 'Native kernels not used[^;]*' | head -1)
  err=$(echo "$on$off" | grep -iE 'error|failed' | head -1)
  r=$( [ -n "$dims" ] && python3 -I $D/cmp.py $D/o_off.raw $D/o_on.raw ${dims%x*} ${dims#*x} $5 )
  t_off=$(echo "$off" | grep -oE 'device chain time: [0-9.]+' | grep -oE '[0-9.]+'); t_on=$(echo "$on" | grep -oE 'device chain time: [0-9.]+' | grep -oE '[0-9.]+')
  echo "$1 -> $dims | native steps $nat ${not:+($not)} | $r | ms off $t_off on $t_on ${err:+| ERR: $err}"
done
rm -f $D/o_off.raw $D/o_on.raw
echo "== RIFE chain (upscale then RIFE 2x), timing only"
for cf in off on; do echo "$cf: $(h ninja1080_nv12 1920 1080 nv12 12 $cf 8 --rife-chain | grep -E 'Native kernels:|rife-chain:|device chain time|rror' | tr '\n' ' ')"; done
echo "== async build mode"
echo "on: $(h ninja1080_nv12 1920 1080 nv12 12 on 2 --async-build | grep -E 'Native|timing: async|configured|chain time|rror' | tr '\n' ' ')"
