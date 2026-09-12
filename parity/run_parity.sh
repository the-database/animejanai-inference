#!/bin/bash
# Full parity sweep: VS/zimg goldens (shipped full-package VSPipe) vs aji
# kernels. WSL-side paths for aji tools, Windows paths for VSPipe.
set -e

# The shared parity working dir, spelled both ways: WROOT is what VSPipe.exe
# (a Windows binary) sees, LROOT is the same directory from WSL. Point
# AJI_PARITY_WIN / AJI_PARITY_WSL at your own location.
WROOT=${AJI_PARITY_WIN:-'C:\parity'}
LROOT=${AJI_PARITY_WSL:-/mnt/c/parity}
# the current shipped reference (VS R73 / vsmlrt 3.22.38 / TRT 10.16.0);
# override with AJI_VSPIPE to golden against another package version
VSPIPE=${AJI_VSPIPE:-/mnt/c/mpv-upscale-2x_animejanai-v3.3.0/VSPipe.exe}
KT=~/src/animejanai-inference/build/aji_kernel_test
PY=~/src/vsenv/bin/python
CMP=~/src/animejanai-inference/parity/compare.py

vsp() {
    # </dev/null: interop exes consume the caller's stdin (kills while-read loops)
    "$VSPIPE" "${@:1:$#-1}" "$WROOT\\golden.vpy" "$WROOT\\${@: -1}" \
        </dev/null >/dev/null 2>>/tmp/vsp-golden.log || {
        echo "vsp failed: ${*: -1}" >&2
        tail -8 /tmp/vsp-golden.log >&2
        return 1
    }
}

# vspipe dumps RGB planes in G,B,R order; normalize golden files to R,G,B
fixrgb() {
    $PY - "$1" "$2" "$3" <<'P'
import sys
import numpy as np
p, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = np.fromfile(p, dtype=np.uint16).reshape(-1, 3, w * h)
a[:, [0, 1, 2]] = a[:, [2, 0, 1]]
a.tofile(p)
P
}
# fixture metadata: name w h format matrix(vs/aji) range siting
FIX="f1 1920 1080 nv12 709 709 limited left
f2 1400 1080 p010 709 709 limited left
f3 640 480 nv12 170m 601 limited center
f4 1920 1080 p010 709 709 limited topleft
f5 1280 720 nv12 709 709 full left"

cp ~/src/animejanai-inference/parity/golden.vpy "$LROOT/"

echo "== PRE: YUV420 -> RGBH (chroma upsample + matrix) =="
while read -r n w h fmt vmat amat rng sit; do
    pf=$([ "$fmt" = p010 ] && echo p10 || echo p8)
    ob=$([ "$fmt" = p010 ] && echo yuv10 || echo yuv8)
    vsp --arg mode=pre --arg "input=$WROOT\\fixtures\\${n}_${w}x${h}.$pf" \
        --arg w=$w --arg h=$h --arg fmt=$ob --arg matrix=$vmat \
        --arg vrange=$rng --arg siting=$sit "goldens\\pre_$n.rgbh"
    fixrgb "$LROOT/goldens/pre_$n.rgbh" $w $h
    $KT pre --input "$LROOT/fixtures/${n}_${w}x${h}.$fmt" \
        --output "$LROOT/ours/pre_$n.rgbh" --width $w --height $h \
        --format $fmt --matrix $amat --range $rng --siting $sit \
        </dev/null 2>/dev/null
    $PY $CMP rgbf16 "$LROOT/ours/pre_$n.rgbh" "$LROOT/goldens/pre_$n.rgbh" \
        $w $h --label "pre $n ${w}x${h} $fmt $amat $rng $sit"
done <<< "$FIX"

echo "== POST: RGBH -> YUV420 (matrix + chroma downsample), input = golden pre RGB =="
while read -r n w h fmt vmat amat rng sit; do
    ob=$([ "$fmt" = p010 ] && echo yuv10 || echo yuv8)
    vsp --arg mode=post --arg "input=$WROOT\\goldens\\pre_$n.rgbh" \
        --arg w=$w --arg h=$h --arg fmt=rgbh --arg outfmt=$ob \
        --arg matrix=$vmat --arg vrange=$rng --arg siting=$sit "goldens\\post_$n.yuv"
    # ours sites LEFT on the way down regardless of $sit: VS/zimg ignores the
    # chroma-location prop for RGB->YUV (no propagation from unsubsampled
    # sources), so the reference output is always left-sited.
    $KT post --input "$LROOT/goldens/pre_$n.rgbh" \
        --output "$LROOT/ours/post_$n.yuv" --width $w --height $h \
        --format $fmt --matrix $amat --range $rng --siting left \
        </dev/null 2>/dev/null
    $PY $CMP $ob "$LROOT/ours/post_$n.yuv" "$LROOT/goldens/post_$n.yuv" \
        $w $h --label "post $n ${w}x${h} $fmt $amat $rng $sit"
done <<< "$FIX"

echo "== RESIZE: RGBH Spline36, input = golden pre_f1 (1920x1080) =="
for geo in "1280 720" "960 540" "1728 972" "3840 2160"; do
    read -r dw dh <<< "$geo"
    vsp --arg mode=resize --arg "input=$WROOT\\goldens\\pre_f1.rgbh" \
        --arg w=1920 --arg h=1080 --arg fmt=rgbh --arg dw=$dw --arg dh=$dh \
        "goldens\\rs_${dw}x${dh}.rgbh"
    fixrgb "$LROOT/goldens/rs_${dw}x${dh}.rgbh" $dw $dh
    $KT resize --input "$LROOT/goldens/pre_f1.rgbh" \
        --output "$LROOT/ours/rs_${dw}x${dh}.rgbh" \
        --width 1920 --height 1080 --out-width $dw --out-height $dh \
        </dev/null 2>/dev/null
    $PY $CMP rgbf16 "$LROOT/ours/rs_${dw}x${dh}.rgbh" \
        "$LROOT/goldens/rs_${dw}x${dh}.rgbh" $dw $dh \
        --label "resize 1920x1080 -> ${dw}x${dh}"
done
