#!/bin/bash
S=/c/Users/jsoos/AppData/Local/Temp/claude/79c4ac1d5f6e4bd2/adb5bb4c-2911-4880-8b6b-57ba589c82df/scratchpad; D=$S/p3; B=$S/build-native-all
P="$LOCALAPPDATA/Programs/mpv-AnimeJaNai/animejanai"; R="$LOCALAPPDATA/the-database/aji/packs"
export PATH="/c/Users/jsoos/aji-win/ffmpeg-shared/ffmpeg-9.0.2-full_build-shared/bin:$PATH"
enc() { # cf slot clip pixfmt out
  local t0=$(date +%s.%N)
  (cd $B && ./aji_encode.exe --input "$(cygpath -w $D/enc/$3.mkv)" --output "$(cygpath -w $D/enc/$5)" --conf "$(cygpath -w $D/$1.conf)" --slot $2 \
     --model-dir "$(cygpath -w "$P/onnx")" --rife-model-dir "$(cygpath -w "$P/rife")" --trtexec "$(cygpath -w "$R/trt-runtime/11.4.0.106-cuda13.4/trtexec.exe")" \
     --vcodec ffv1 --pix-fmt $4 --overwrite --no-audio --progress none --log "$(cygpath -w $D/enc/$5.log)" > $D/enc/$5.out 2>&1); local rc=$?
  echo "$rc $(py -3.13 -c "print(f'{$(date +%s.%N)-$t0:.1f}')")"; }
for c in "ninja3s 2 yuv420p10" "ninja3s 3 yuv444p10" "live3s 1 yuv420p10" "live3s 2 yuv420p10"; do set -- $c
  a=$(enc off $2 $1 $3 off.mkv); b=$(enc on $2 $1 $3 on.mkv)
  nat=$(grep -aoE 'Native kernels: [^ ]+' $D/enc/on.mkv.log $D/enc/on.mkv.out 2>/dev/null | head -1)
  psnr=$(ffmpeg -v info -i $D/enc/off.mkv -i $D/enc/on.mkv -lavfi "[0][1]psnr" -f null - 2>&1 | grep -oE 'PSNR y:[^ ]+ u:[^ ]+ v:[^ ]+ average:[^ ]+ min:[^ ]+')
  frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,width,height,pix_fmt -of csv=p=0 $D/enc/on.mkv)
  echo "$1 slot $2 $3: rc/s off ($a) on ($b) | ${nat:-NO NATIVE LINE} | $frames | $psnr"
done
rm -f $D/enc/off.mkv $D/enc/on.mkv
