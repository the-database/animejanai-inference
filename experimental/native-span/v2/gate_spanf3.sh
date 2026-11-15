#!/bin/bash
# Interleaved: TensorRT 11.4 vs NeoRTX vs native, SPANF3 Performance 1080p -> 4K (NCHW fp16 in/out)
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad; W=$(cygpath -w $S)
V=$S/wt-native/experimental/native-span/v2
R="$LOCALAPPDATA/the-database/aji/packs"; export PATH="$R/trt-runtime/11.4.0.106-cuda13.4:$R/trt-sm120/11.4.0.106:$PATH"; TX="$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe"
MT="$W\neo-dl\app\cHiDeScaler-Neo\models\AnimeJaNai\2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_fp16.onnx"
others() { nvidia-smi pmon -c 1 | awk '$4 ~ /^[0-9]+$/ && $4 > 3' | grep -vE 'trtexec|python|spanf3_native|compact_native' | wc -l; }
wait_idle() { q=0; for i in $(seq 1 1080); do [ "$(others)" -eq 0 ] && q=$((q+1)) || q=0; [ $q -ge 3 ] && return 0; sleep 10; done; }
cd $S
for round in 1 2 3; do
  wait_idle
  b=$(others)
  t=$("$TX" --loadEngine=theirs1080.engine --useCudaGraph --iterations=1000 --warmUp=1000 2>&1 | grep -aoE 'median = [0-9.]+' | head -1 | grep -oE '[0-9.]+')
  n=$($S/ortenv/Scripts/python.exe -I neortx_bench.py "$W/neo-dl/pack" "$MT" "$W/frames/ninja1080.rgba" 1920 1080 "$W/frames/tmp_out.rgba" 1000 2>&1 | grep -a 'shared-io' | grep -aoE 'image_gpu [0-9.]+' | grep -oE '[0-9.]+')
  x=$($V/spanf3_native.exe "$(cygpath -w $V/data/perf1080)" 1920 1080 1000 2>&1 | grep 'warps8 rows2' | grep -oE 'model [0-9.]+' | grep -oE '[0-9.]+')
  echo "round $round: TRT $t | NeoRTX $n | native $x  (other GPU procs: $b/$(others))"
done
