#!/bin/bash
# libaji TensorRT (aji_harness, mpv-AnimeJaNai 3.10.0 install, full chain incl. NV12<->RGB) vs native library (model only)
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad
P="$LOCALAPPDATA/Programs/mpv-AnimeJaNai/animejanai"; R="$LOCALAPPDATA/the-database/aji/packs"
export PATH="$R/trt-runtime/11.4.0.106-cuda13.4:$R/trt-sm120/11.4.0.106:/c/Users/jsoos/aji-win/cuda134/bin:$PATH"
TX="$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe"; D=$S/wt-native/experimental/native-span/v2/data; B=$S/build-native-dev
others() { nvidia-smi pmon -c 1 | awk '$4 ~ /^[0-9]+$/ && $4 > 3' | grep -vE 'aji_harness|aji_native|trtexec' | wc -l; }
wait_idle() { q=0; for i in $(seq 1 1080); do [ "$(others)" -eq 0 ] && q=$((q+1)) || q=0; [ $q -ge 3 ] && return 0; sleep 10; done; }
harness() { (cd "$P/inference" && ./aji_harness.exe --input "$(cygpath -w $S/libaji-base/$1.nv12)" --width $2 --height $3 --format nv12 --frames $4 --conf "$(cygpath -w $S/libaji-base/bench.conf)" --model-dir ../onnx --trtexec "$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe" --slot $5 2>&1); }
cases=("sd480 720 480 1 compact compact480" "ninja720 1280 720 1 compact compact720" "ninja720 1280 720 3 spanf3 bal720" "ninja1080 1920 1080 3 spanf3 bal1080" "ninja1080 1920 1080 2 spanf3 perf1080")
echo "warm-up (engine builds):"; for c in "${cases[@]}"; do set -- $c; harness $1 $2 $3 20 $4 | grep -E 'configured|chain time|rror' | tail -1; done
for round in 1 2 3; do
  wait_idle
  for c in "${cases[@]}"; do set -- $c
    b=$(others)
    t=$(harness $1 $2 $3 1000 $4 | grep -oE 'device chain time: [0-9.]+' | grep -oE '[0-9.]+')
    nl=$($B/aji_native_bench.exe $5 "$(cygpath -w $D/$6)" $2 $3 1000); n=$(echo "$nl" | grep -oE "whole model [0-9.]+" | grep -oE "[0-9.]+"); q=$(echo "$nl" | grep -oE "PSNR [0-9.]+ dB, max [0-9]+ levels")
    echo "round $round $6 ${2}x$3: libaji TRT $t | native $n [$q]  (other GPU procs: $b/$(others))"
  done
done
