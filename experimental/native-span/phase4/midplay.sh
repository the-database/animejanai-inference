#!/bin/bash
# usage: midplay.sh conf slot seq clip start frames tag   (seq: comma-separated slots, one per 200 frames)
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad
D=$S/p4; P="$LOCALAPPDATA/Programs/mpv-AnimeJaNai"; R="$LOCALAPPDATA/the-database/aji/packs"
q() { local w; w=$(cygpath -w "$1"); printf '%%%d%%%s' "$(printf '%s' "$w" | wc -c)" "$w"; }
conf=$1 slot=$2 seq=$3 clip=$4 start=$5 frames=$6 tag=$7
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $D/$tag.mem &
mp=$!
"$P/mpv.com" --no-config --hwdec=nvdec --vo=null --untimed --no-audio --start=$start --frames=$frames --msg-time \
  --log-file="$(cygpath -w $D/$tag.log)" --script="$(cygpath -w $D/switch.lua)" --script-opts=aji-seq=$seq \
  "--vf=@aji:animejanai:lib=$(q $S/build-native-all/aji.dll):conf=$(q $S/p3/$conf.conf):model-dir=$(q "$P/animejanai/onnx"):rife-model-dir=$(q "$P/animejanai/rife"):trtexec=$(q "$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe"):stats=$(q $D/$tag.stats):slot=$slot" \
  "$clip" > $D/$tag.out 2>&1
echo "mpv exit $?"
kill $mp
