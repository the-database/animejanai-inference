#!/bin/bash
# libaji end to end (aji_harness: NV12 in, pre-kernel, model, post-kernel, NV12 out, CUDA graph),
# the same build with native=off (TensorRT) and native=on.
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad
L=$S/libaji-base; B=$S/build-native-dev; P="$LOCALAPPDATA/Programs/mpv-AnimeJaNai/animejanai"; R="$LOCALAPPDATA/the-database/aji/packs"
others() { nvidia-smi pmon -c 1 | awk '$4 ~ /^[0-9]+$/ && $4 > 3' | grep -v aji_harness | wc -l; }
wait_idle() { q=0; for i in $(seq 1 1080); do [ "$(others)" -eq 0 ] && q=$((q+1)) || q=0; [ $q -ge 3 ] && return 0; sleep 10; done; }
h() { (cd $B && ./aji_harness.exe --input "$(cygpath -w $L/$1.nv12)" --width $2 --height $3 --format nv12 --frames $4 --conf "$(cygpath -w $L/$5.conf)" --model-dir "$(cygpath -w "$P/onnx")" --trtexec "$(cygpath -w "$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe")" --slot $6 $7 2>&1); }
cases=("compact sd480 720 480 1" "compact ninja720 1280 720 1" "balanced ninja720 1280 720 3" "balanced ninja1080 1920 1080 3" "performance ninja1080 1920 1080 2" "v2-ultracompact ninja720 1280 720 4")
echo "== parity (first output frame, NV12)"
for c in "${cases[@]}"; do set -- $c
  h $2 $3 $4 2 bench $5 "--output $(cygpath -w $L/p_trt.raw)" > /dev/null
  h $2 $3 $4 2 bench-native $5 "--output $(cygpath -w $L/p_nat.raw)" > /dev/null
  echo "$1 ${3}x$4: $(py -3.13 -I $S/nv12cmp.py "$(cygpath -w $L/p_trt.raw)" "$(cygpath -w $L/p_nat.raw)" $((2*$3)) $((2*$4)) | tr '\n' ' ')"
done
rm -f $L/p_trt.raw $L/p_nat.raw
for round in 1 2 3; do
  wait_idle
  for c in "${cases[@]}"; do set -- $c
    b=$(others)
    t=$(h $2 $3 $4 1000 bench $5 | grep -oE 'device chain time: [0-9.]+' | grep -oE '[0-9.]+')
    n=$(h $2 $3 $4 1000 bench-native $5 | grep -oE 'device chain time: [0-9.]+' | grep -oE '[0-9.]+')
    echo "round $round $1 ${3}x$4: TensorRT $t | native $n  (other GPU procs: $b/$(others))"
  done
done
