#!/bin/bash
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad; D=$S/p3; B=$S/build-native-all
P="$LOCALAPPDATA/Programs/mpv-AnimeJaNai/animejanai"; R="$LOCALAPPDATA/the-database/aji/packs"
others() { nvidia-smi pmon -c 1 | awk '$4 ~ /^[0-9]+$/ && $4 > 3' | grep -v aji_harness | wc -l; }
wait_idle() { q=0; for i in $(seq 1 1080); do [ "$(others)" -eq 0 ] && q=$((q+1)) || q=0; [ $q -ge 3 ] && return 0; sleep 10; done; }
h() { (cd $B && ./aji_harness.exe --input "$(cygpath -w $D/in/$1.raw)" --width $2 --height $3 --format $4 --frames 1000 \
      --conf "$(cygpath -w $D/$5.conf)" --model-dir "$(cygpath -w "$P/onnx")" --rife-model-dir "$(cygpath -w "$P/rife")" \
      --trtexec "$(cygpath -w "$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe")" --slot $6 2>&1 | grep -oE 'device chain time: [0-9.]+' | grep -oE '[0-9.]+'); }
cases=("perf-854x480 ninja854_nv12 854 480 nv12 2" "compact-854x480 ninja854_nv12 854 480 nv12 1" "compact-dvd768x576 dvd768_nv12 768 576 nv12 1"
       "perf-event1280x642 oso1280x642_nv12 1280 642 nv12 2" "resize50+perf-1080 ninja1080_nv12 1920 1080 nv12 7"
       "bal-1440x1080 ninja1440_nv12 1440 1080 nv12 3" "perf-1918x1078 ninja1918_nv12 1918 1078 nv12 2" "bal-1918x1078 ninja1918_nv12 1918 1078 nv12 3"
       "compact-live1080 live1080_nv12 1920 1080 nv12 1" "chain-bal+perf-854 ninja854_nv12 854 480 nv12 6")
for round in 1 2; do
  wait_idle
  for c in "${cases[@]}"; do set -- $c
    echo "round $round $1: TensorRT $(h $2 $3 $4 $5 off $6) | native $(h $2 $3 $4 $5 on $6)  (other GPU procs: $(others))"
  done
done
