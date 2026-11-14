#!/bin/bash
D=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad/spanproto/sweep
R="$LOCALAPPDATA/the-database/aji/packs"; export PATH="$R/trt-runtime/11.4.0.106-cuda13.4:$R/trt-sm120/11.4.0.106:$PATH"; TX="$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe"
cd $D
others() { nvidia-smi pmon -c 1 | awk '$4 ~ /^[0-9]+$/ && $4 > 3' | grep -vE 'trtexec|span_sweep|python' | wc -l; }
wait_idle() { q=0; for i in $(seq 1 1080); do [ "$(others)" -eq 0 ] && q=$((q+1)) || q=0; [ $q -ge 3 ] && return 0; sleep 10; done; return 1; }
for round in 1 2 3; do
  wait_idle; echo "== round $round start $(date +%T)"
  while read name c nb act att; do
    b=$(others)
    tm=$("$TX" --loadEngine=$name/model.engine --useCudaGraph --iterations=500 --warmUp=1000 2>&1 | grep -aoE 'median = [0-9.]+' | head -1 | grep -oE '[0-9.]+')
    tb=$("$TX" --loadEngine=$name/body.engine --useCudaGraph --iterations=500 --warmUp=1000 2>&1 | grep -aoE 'median = [0-9.]+' | head -1 | grep -oE '[0-9.]+')
    fb=$(../span_sweep.exe "$(cygpath -w $D/$name)" $c $nb $act $att 540 960 500 2>&1 | grep -oE 'fused body: [0-9.]+' | grep -oE '[0-9.]+')
    a=$(others)
    echo "$round $name trt_model=$tm trt_body=$tb fused_body=$fb busy=$b/$a"
  done < variants.txt
done
