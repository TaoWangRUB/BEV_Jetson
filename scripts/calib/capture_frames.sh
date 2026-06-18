#!/usr/bin/env bash
# Capture checkerboard frames from one IMX219 (Argus sensor-id) for fisheye
# intrinsic calibration. Headless — grabs N JPEG frames at ~2 fps while you move
# a checkerboard across the WHOLE field of view. For a 160° fisheye, COVER THE
# EDGES AND CORNERS (tilts too) — that's where the distortion model is fit.
# Run on the TX2.
#
#   ./capture_frames.sh <sensor-id> [count] [out-dir] [width] [height]
set -euo pipefail
SID="${1:?usage: capture_frames.sh <sensor-id> [count] [out-dir] [w] [h]}"
COUNT="${2:-60}"
OUT="${3:-/media/nvidia/workspace/BEV/calib_frames/cam${SID}}"
W="${4:-1280}"; H="${5:-720}"

mkdir -p "$OUT"; rm -f "$OUT"/frame_*.jpg
echo "Capturing $COUNT frames from sensor-id $SID at ${W}x${H} -> $OUT"
echo ">> Move the checkerboard slowly across the FULL view (corners/edges + tilts)."
echo ">> ~${COUNT} frames over ~$((COUNT/2)) s. Starting in 3s..."
sleep 3
gst-launch-1.0 -e nvarguscamerasrc sensor-id="$SID" num-buffers="$COUNT" \
  ! "video/x-raw(memory:NVMM),width=${W},height=${H},framerate=2/1" \
  ! nvjpegenc ! multifilesink location="$OUT/frame_%03d.jpg"
echo "Done: $(ls "$OUT"/frame_*.jpg 2>/dev/null | wc -l) frames in $OUT"
