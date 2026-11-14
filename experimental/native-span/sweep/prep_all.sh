#!/bin/bash
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad; D=$S/spanproto/sweep
R="$LOCALAPPDATA/the-database/aji/packs"; export PATH="$R/trt-runtime/11.4.0.106-cuda13.4:$R/trt-sm120/11.4.0.106:$PATH"; TX="$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe"
cd $D
while read name c nb act att; do
  [ -f $name/model.engine ] && [ -f $name/body.engine ] && continue
  $S/ortenv/Scripts/python.exe sweep_prep.py "$(cygpath -w $D)" $name $c $nb $act $att "$(cygpath -w $S/frames/ninja1080.rgba)" > $name.prep.log 2>&1 || { echo "$name PREP FAILED"; tail -5 $name.prep.log; continue; }
  feat=$(head -1 $name/body_io.txt)
  "$TX" --onnx=$name/model.onnx --shapes=input:1x3x1080x1920 --saveEngine=$name/model.engine --skipInference > $name/trt_model.log 2>&1 || echo "$name model engine FAILED"
  "$TX" --onnx=$name/body.onnx --shapes="$feat":1x${c}x540x960 --saveEngine=$name/body.engine --skipInference > $name/trt_body.log 2>&1 || echo "$name body engine FAILED"
  ../span_sweep.exe "$(cygpath -w $D/$name)" $c $nb $act $att 540 960 20 > $name/proto_check.log 2>&1
  echo "$name: $(py -3.13 -I cmp.py "$(cygpath -w $D/$name)" 2>&1)"
done < variants.txt
