#!/usr/bin/env bash
# Regression matrix for the temporal (multi-frame) path, comparing a
# reference build (BASE) with a candidate (DEV):
#
#   exact  chains without stacked temporal models must produce bit-identical
#          output (harness raw frames and aji_encode framemd5), and BASE must
#          be deterministic run to run
#   perf   the same chains must be as fast: median of alternating harness
#          runs, |delta| <= max(PERF_TOL %, the BASE-vs-BASE noise)
#   stack  stacked temporal models (DEV only): equivalence with a two-pass
#          reference, frame alignment, each stage run once per frame, short
#          clips, ingest-ahead and reset scheduling
#
# Usage: regress/temporal_regress.sh [exact] [perf] [stack]   (default: all)
#
# Environment (paths as the Windows tools see them):
#   BASE, DEV       build dirs; the runtime DLLs (TensorRT, cudart, ORT,
#                   DirectML, FFmpeg) must sit next to the binaries
#   MODELS          onnx dir holding $SD $TM $TF $TF2 $HD (engines are built
#                   there once and shared by both builds, so TensorRT tactic
#                   choices cannot differ between them)
#   RIFE            RIFE model dir;  TRTEXEC  trtexec.exe
#   IN480           640x480 P010 raw, >= 360 frames, with real motion and cuts
#   IN1080          1920x1080 P010 raw, >= 60 frames
#   CLIP480/1080    the same sources as video files, for aji_encode
#   WORK            scratch dir
#   BACKENDS        default "TensorRT DirectML";  PERF_RUNS default 5;
#   PERF_TOL        percent, default 1
set -u
: "${BASE:?}" "${DEV:?}" "${MODELS:?}" "${RIFE:?}" "${TRTEXEC:?}" \
  "${IN480:?}" "${IN1080:?}" "${CLIP480:?}" "${CLIP1080:?}" "${WORK:?}"
BACKENDS=${BACKENDS:-"TensorRT DirectML"}
PERF_RUNS=${PERF_RUNS:-5}
PERF_TOL=${PERF_TOL:-1}
SD=${SD:-2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo}
TM=${TM:-2x_bndl_animefilm_v3_1x27xHxW_dyn-HW_fp16_op20}
TF=${TF:-temporalfix_s3_v1_op18_fp16}
TF2=${TF2:-temporalfix_s1_v1.1_op18_fp16}
HD=${HD:-2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16}
SECTIONS=${*:-exact perf stack}
mkdir -p "$WORK"
FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }
pass() { echo "ok:   $*"; }

# conf <file> <backend> <rife: no|after|before> <model>...
conf() {
    local f=$1 be=$2 rife=$3 i=1 m
    shift 3
    {
        echo "[global]"; echo "config_version=2"; echo "logging=yes"
        echo "backend=$be"; echo "[slot_1]"; echo "profile_name=regress"
        echo "chain_1_min_resolution=0x0"; echo "chain_1_max_resolution=0x0"
        echo "chain_1_min_fps=0"; echo "chain_1_max_fps=0"
        for m in "$@"; do
            echo "chain_1_model_${i}_resize_height_before_upscale=0"
            echo "chain_1_model_${i}_resize_factor_before_upscale=100"
            echo "chain_1_model_${i}_name=$m"
            i=$((i + 1))
        done
        if [ "$rife" = no ]; then
            echo "chain_1_rife=no"
        else
            echo "chain_1_rife=yes"; echo "chain_1_rife_model=413"
            echo "chain_1_rife_factor_numerator=2"
            echo "chain_1_rife_factor_denominator=1"
            echo "chain_1_rife_scene_detect_threshold=0.15"
            echo "chain_1_rife_ensemble=no"
            [ "$rife" = before ] && echo "chain_1_rife_before_upscale=yes" \
                                 || echo "chain_1_rife_before_upscale=no"
        fi
    } > "$f"
}

# harness <build> <backend> <conf> <input> <w> <h> <out|-> [args...]
harness() {
    local b=$1 be=$2 c=$3 in=$4 w=$5 h=$6 o=$7 exe
    shift 7
    if [ "$be" = TensorRT ]; then
        exe=("$b/aji_harness.exe" --trtexec "$TRTEXEC")
    else
        exe=("$b/aji_harness_dml.exe" --frames 100000)
    fi
    [ "$o" = - ] || set -- --output "$o" "$@"
    "${exe[@]}" --input "$in" --width "$w" --height "$h" --format p010 \
        --conf "$c" --model-dir "$MODELS" --slot 1 "$@" 2>&1
}

# encode <build> <conf> <clip> <out.mkv> <backend>: prints the framemd5 path
encode() {
    local b=$1 c=$2 clip=$3 o=$4 be=$5 pf=yuv444p10
    [ "$be" = DirectML ] && pf=yuv420p10  # DirectML outputs 4:2:0 only
    "$b/aji_encode.exe" --input "$clip" --output "$o" --conf "$c" --slot 1 \
        --model-dir "$MODELS" --rife-model-dir "$RIFE" --trtexec "$TRTEXEC" \
        --backend "$(echo "$be" | tr 'A-Z' 'a-z')" --vcodec ffv1 \
        --pix-fmt $pf --overwrite --no-audio --progress none \
        > "$o.log" 2>&1 || return 1
    ffmpeg -v error -y -i "$o" -map 0:v -f framemd5 "$o.md5" || return 1
}

same() { cmp -s "$1" "$2"; }
frames_of() { echo $(( $(stat -c %s "$1") / $2 )); }
IFS480=$((640 * 480 * 3))       # 640x480 P010 input frame bytes
FS480=$((1280 * 960 * 3))       # its 2x output

if [[ " $SECTIONS " == *" exact "* ]]; then
    echo "== exact: chains without stacking, BASE vs DEV (and BASE vs BASE)"
    for be in $BACKENDS; do
        W="$WORK/exact_$be"; mkdir -p "$W"
        conf "$W/hd.conf" "$be" no "$HD"
        conf "$W/sd.conf" "$be" no "$SD"
        conf "$W/sdhd.conf" "$be" no "$SD" "$HD"
        conf "$W/tm.conf" "$be" no "$TM"
        conf "$W/sdtf.conf" "$be" no "$SD" "$TF"
        conf "$W/tmhd.conf" "$be" no "$TM" "$HD"
        conf "$W/rafter.conf" "$be" after "$SD"
        conf "$W/rbefore.conf" "$be" before "$SD"
        conf "$W/tmrife.conf" "$be" after "$TM"
        # harness raw frames (the 480p cases also run 1-, 2- and 9-frame
        # clips through the temporal edge handling)
        for case in "hd $IN1080 1920 1080 40" "sd $IN480 640 480 360" \
                    "sdhd $IN480 640 480 60" "tm $IN480 640 480 360" \
                    "sdtf $IN480 640 480 360" "tmhd $IN480 640 480 60" \
                    "tm $IN480 640 480 1" "tm $IN480 640 480 2" \
                    "tm $IN480 640 480 9" "sdtf $IN480 640 480 4"; do
            set -- $case
            c=$1 in=$2 w=$3 h=$4 n=$5
            for run in base1 base2 dev; do
                b=$BASE; [ $run = dev ] && b=$DEV
                harness "$b" "$be" "$W/$c.conf" "$in" "$w" "$h" \
                    "$W/$c.$n.$run.raw" --frames "$n" > "$W/$c.$n.$run.log"
            done
            if [ ! -s "$W/$c.$n.base1.raw" ]; then
                fail "$be harness $c/$n: no output ($(tail -1 "$W/$c.$n.base1.log"))"
            elif ! same "$W/$c.$n.base1.raw" "$W/$c.$n.base2.raw"; then
                fail "$be harness $c/$n: BASE is not deterministic"
            elif same "$W/$c.$n.base1.raw" "$W/$c.$n.dev.raw"; then
                pass "$be harness $c/$n bit-exact"
            else
                fail "$be harness $c/$n: DEV output differs"
            fi
            rm -f "$W/$c.$n."*.raw
        done
        # aji_encode end to end (decode, RIFE scheduling, encode)
        for case in "sd $CLIP480" "tm $CLIP480" "sdtf $CLIP480" \
                    "rafter $CLIP480" "rbefore $CLIP480" "tmrife $CLIP480" \
                    "hd $CLIP1080"; do
            set -- $case
            c=$1 clip=$2
            ok=1
            for run in base1 base2 dev; do
                b=$BASE; [ $run = dev ] && b=$DEV
                encode "$b" "$W/$c.conf" "$clip" "$W/$c.$run.mkv" "$be" || {
                    ok=0
                    fail "$be encode $c ($run) failed: $(tail -1 "$W/$c.$run.mkv.log")"
                }
            done
            if [ $ok = 0 ]; then
                :
            elif ! same "$W/$c.base1.mkv.md5" "$W/$c.base2.mkv.md5"; then
                fail "$be encode $c: BASE is not deterministic"
            elif same "$W/$c.base1.mkv.md5" "$W/$c.dev.mkv.md5"; then
                pass "$be encode $c bit-exact ($(grep -vc '^#' "$W/$c.dev.mkv.md5") frames)"
            else
                fail "$be encode $c: DEV output differs"
            fi
            rm -f "$W/$c."*.mkv
        done
    done
fi

# median ms/frame of a harness log list
median() { sort -n | awk '{a[NR]=$1} END{print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }
msf() { grep -o 'time: [0-9.]*' | head -1 | cut -d' ' -f2; }

if [[ " $SECTIONS " == *" perf "* ]]; then
    echo "== perf: BASE vs DEV, $PERF_RUNS alternating runs each, median"
    for be in $BACKENDS; do
        W="$WORK/perf_$be"; mkdir -p "$W"
        conf "$W/hd.conf" "$be" no "$HD"
        conf "$W/sd.conf" "$be" no "$SD"
        conf "$W/tm.conf" "$be" no "$TM"
        for case in "hd $IN1080 1920 1080 60" "sd $IN480 640 480 360" \
                    "tm $IN480 640 480 360"; do
            set -- $case
            c=$1 in=$2 w=$3 h=$4 n=$5
            : > "$W/$c.a"; : > "$W/$c.b"; : > "$W/$c.d"
            # warm both (engine load, clocks), then ABDABD...: A and B are
            # BASE, so A-vs-B is the noise floor
            harness "$BASE" "$be" "$W/$c.conf" "$in" "$w" "$h" - --frames "$n" > /dev/null
            harness "$DEV" "$be" "$W/$c.conf" "$in" "$w" "$h" - --frames "$n" > /dev/null
            for i in $(seq "$PERF_RUNS"); do
                harness "$BASE" "$be" "$W/$c.conf" "$in" "$w" "$h" - --frames "$n" | msf >> "$W/$c.a"
                harness "$BASE" "$be" "$W/$c.conf" "$in" "$w" "$h" - --frames "$n" | msf >> "$W/$c.b"
                harness "$DEV" "$be" "$W/$c.conf" "$in" "$w" "$h" - --frames "$n" | msf >> "$W/$c.d"
            done
            a=$(median < "$W/$c.a"); b=$(median < "$W/$c.b"); d=$(median < "$W/$c.d")
            awk -v a="$a" -v b="$b" -v d="$d" -v tol="$PERF_TOL" -v name="$be $c" 'BEGIN{
                base = (a + b) / 2
                noise = 100 * (a > b ? a - b : b - a) / base
                delta = 100 * (d - base) / base
                lim = noise > tol ? noise : tol
                printf "%s %s: base %.3f/%.3f ms, dev %.3f ms, delta %+.2f%% (noise %.2f%%, limit %.2f%%)\n",
                    (delta <= lim ? "ok:  " : "FAIL:"), name, a, b, d, delta, noise, lim
                exit !(delta <= lim)
            }' || FAIL=1
        done
    done
fi

if [[ " $SECTIONS " == *" stack "* ]]; then
    echo "== stack: stacked temporal models (DEV)"
    for be in $BACKENDS; do
        W="$WORK/stack_$be"; mkdir -p "$W"
        conf "$W/tm.conf" "$be" no "$TM"
        conf "$W/tf.conf" "$be" no "$TF"
        conf "$W/stack.conf" "$be" no "$TM" "$TF"
        conf "$W/stack3.conf" "$be" no "$TM" "$TF" "$TF2"
        R="-f rawvideo -pix_fmt p010le -s 1280x960"

        # equivalence with the two-pass reference (the intermediate goes
        # through 10-bit 4:2:0 there, so not bit-exact) and frame alignment
        harness "$DEV" "$be" "$W/tm.conf" "$IN480" 640 480 "$W/tm.raw" --frames 360 > /dev/null
        harness "$DEV" "$be" "$W/tf.conf" "$W/tm.raw" 1280 960 "$W/two.raw" --frames 360 > /dev/null
        AJI_TEMPORAL_STATS=1 harness "$DEV" "$be" "$W/stack.conf" "$IN480" 640 480 \
            "$W/one.raw" --frames 360 > "$W/one.log"
        # (a relative stats path: a drive colon breaks the filter parser)
        (cd "$W" && ffmpeg -v error $R -i one.raw $R -i two.raw \
            -lavfi "psnr=stats_file=ps.txt" -f null -)
        awk -v name="$be" '{for (i = 1; i <= NF; i++) if ($i ~ /^psnr_y:/) {
                split($i, a, ":"); y = a[2] == "inf" ? 99 : a[2]
                if (min == "" || y < min) min = y; n++ } }
            END { ok = n == 360 && min >= 50
                  printf "%s %s stacked vs two-pass: %d frames, min Y PSNR %.2f dB (>= 50)\n",
                      ok ? "ok:  " : "FAIL:", name, n, min; exit !ok }' "$W/ps.txt" || FAIL=1
        sh=$(ffmpeg $R -i "$W/one.raw" $R -i "$W/two.raw" -lavfi \
            "[0]trim=start_frame=1,setpts=PTS-STARTPTS[a];[a][1]psnr" -f null - 2>&1 |
            grep -o 'y:[0-9.]*' | head -1 | cut -d: -f2)
        awk -v s="$sh" -v name="$be" 'BEGIN{ ok = s < 40
            printf "%s %s off-by-one check: shifted PSNR %.2f dB (< 40)\n",
                ok ? "ok:  " : "FAIL:", name, s; exit !ok }' || FAIL=1
        grep -q "360 ingested; stage 0 (9 frames): 360 runs; stage 1 (7 frames): 360 runs" "$W/one.log" &&
            pass "$be each stage ran once per frame" ||
            fail "$be stage run counts: $(grep stats "$W/one.log")"

        for c in stack stack3; do
            # short clips: frame count and run counts
            for n in 1 2 7 8 15 21; do
                AJI_TEMPORAL_STATS=1 harness "$DEV" "$be" "$W/$c.conf" "$IN480" 640 480 \
                    "$W/short.raw" --frames $n > "$W/short.log"
                got=$(frames_of "$W/short.raw" $FS480)
                runs=$(grep -o '[0-9]* runs' "$W/short.log" | sort -u | tr '\n' ' ')
                [ "$got" = $n ] && [ "$runs" = "$n runs " ] &&
                    pass "$be $c $n-frame clip" ||
                    fail "$be $c $n-frame clip: $got frames, runs '$runs' $(grep -m1 -i 'error\|fail' "$W/short.log")"
            done
            # ingest-ahead (the contract allows 4) must not change the output
            harness "$DEV" "$be" "$W/$c.conf" "$IN480" 640 480 "$W/lock.raw" --frames 120 --ingest-ahead 0 > /dev/null
            for a in 1 4; do
                harness "$DEV" "$be" "$W/$c.conf" "$IN480" 640 480 "$W/ahead.raw" --frames 120 --ingest-ahead $a > "$W/ahead.log"
                same "$W/lock.raw" "$W/ahead.raw" && pass "$be $c ingest $a ahead" ||
                    fail "$be $c ingest $a ahead differs $(grep -m1 -i fail "$W/ahead.log")"
            done
            # a reset mid-stream must match a fresh start at that frame
            tail -c +$((IFS480 * 100 + 1)) "$IN480" | head -c $((IFS480 * 60)) > "$W/from100.raw"
            harness "$DEV" "$be" "$W/$c.conf" "$IN480" 640 480 "$W/rst.raw" --frames 160 --reset-at 100 > /dev/null
            harness "$DEV" "$be" "$W/$c.conf" "$W/from100.raw" 640 480 "$W/fresh.raw" --frames 60 --ingest-ahead 0 > /dev/null
            tail -c +$((FS480 * 100 + 1)) "$W/rst.raw" > "$W/rst_tail.raw"
            same "$W/rst_tail.raw" "$W/fresh.raw" && [ -s "$W/fresh.raw" ] &&
                pass "$be $c reset matches a fresh start" || fail "$be $c reset differs"
        done
        mv "$W/one.raw" "$WORK/stacked_$be.raw"
        rm -f "$W"/*.raw
    done
    # the two backends' stacked output should agree closely
    if [ -s "$WORK/stacked_TensorRT.raw" ] && [ -s "$WORK/stacked_DirectML.raw" ]; then
        R="-f rawvideo -pix_fmt p010le -s 1280x960"
        x=$(ffmpeg $R -i "$WORK/stacked_TensorRT.raw" $R -i "$WORK/stacked_DirectML.raw" \
            -lavfi psnr -f null - 2>&1 | grep -o 'y:[0-9.inf]*' | head -1 | cut -d: -f2)
        [ "$x" = inf ] && x=99
        awk -v x="$x" 'BEGIN{ ok = x >= 45
            printf "%s stacked TensorRT vs DirectML: Y PSNR %.2f dB (>= 45)\n",
                ok ? "ok:  " : "FAIL:", x; exit !ok }' || FAIL=1
    fi
    rm -f "$WORK"/stacked_*.raw
fi

[ $FAIL = 0 ] && echo "ALL PASSED" || echo "FAILURES"
exit $FAIL
