#!/bin/bash
# Parity fixtures. Each fixture is generated ONCE in a canonical planar
# format; the packed twin (NV12/P010) is derived by lossless repack so both
# layouts carry identical pixel values. (Generating the two layouts with
# separate ffmpeg invocations takes different swscale paths and yields
# different bytes — that burned us once.)
set -e
FX="$1"          # output dir
TONGARI="$2"     # 1080p BT.709 limited left AVC episode
EVA_P010="$3"    # existing packed P010 raw, 1400x1080
mkdir -p "$FX"
F="ffmpeg -nostdin -v error -y"

repack() { # in_file in_fmt out_file out_fmt WxH
    $F -f rawvideo -pix_fmt "$2" -s "$5" -i "$1" -pix_fmt "$4" -f rawvideo "$3"
}

# f1: real anime 1080p 8-bit (709/limited/left), 5 frames mid-episode
$F -ss 480 -i "$TONGARI" -frames:v 5 -pix_fmt yuv420p -f rawvideo "$FX/f1_1920x1080.p8"
repack "$FX/f1_1920x1080.p8" yuv420p "$FX/f1_1920x1080.nv12" nv12 1920x1080

# f2: real P010 (709/limited/left) from the phase-0 capture
cp "$EVA_P010" "$FX/f2_1400x1080.p010"
repack "$EVA_P010" p010le "$FX/f2_1400x1080.p10" yuv420p10le 1400x1080

# f3: SD content for the 601(170m)/center path
$F -ss 480 -i "$TONGARI" -frames:v 5 -vf scale=640:480 -pix_fmt yuv420p -f rawvideo "$FX/f3_640x480.p8"
repack "$FX/f3_640x480.p8" yuv420p "$FX/f3_640x480.nv12" nv12 640x480

# f4: synthetic 10-bit gradients (testsrc2 is deterministic), topleft-tagged
$F -f lavfi -i testsrc2=size=1920x1080:rate=24 -frames:v 3 -pix_fmt yuv420p10le -f rawvideo "$FX/f4_1920x1080.p10"
repack "$FX/f4_1920x1080.p10" yuv420p10le "$FX/f4_1920x1080.p010" p010le 1920x1080

# f5: full-range 8-bit synthetic
$F -f lavfi -i testsrc2=size=1280x720:rate=24 -frames:v 3 -vf scale=out_range=full -pix_fmt yuv420p -f rawvideo "$FX/f5_1280x720.p8"
repack "$FX/f5_1280x720.p8" yuv420p "$FX/f5_1280x720.nv12" nv12 1280x720

# repack must be bit-lossless: round-trip check on the 10-bit pair
repack "$FX/f4_1920x1080.p010" p010le /tmp/f4_rt.p10 yuv420p10le 1920x1080
cmp -s /tmp/f4_rt.p10 "$FX/f4_1920x1080.p10" || { echo "repack not lossless!"; exit 1; }

ls -la "$FX"
