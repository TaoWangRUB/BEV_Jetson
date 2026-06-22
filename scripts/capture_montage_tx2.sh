#!/usr/bin/env bash
# Capture the 4 individual fisheye views + the stitched panorama and montage them into one image.
# Run on the TX2 host from the repo root. The 4-cam capture and the panorama can't hold Argus at
# the same time, so this does two short phases then montages (via scripts/port/grab_views.py).
#
#   ./scripts/capture_montage_tx2.sh [out.png]      # default out: bev_views.png
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-bev_views.png}"
IMG="${IMG:-cuvslam-foxy:tx2}"
VDIR="$(mktemp -d /tmp/bev_views.XXXX)"

OPTS=(--runtime nvidia --network host
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
  -v /usr/local/cuda:/usr/local/cuda:ro
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro
  -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev
  -v "$VDIR":/views -v "$PWD":/workspace -w /workspace "$IMG")

# Restart nvargus-daemon to clear any leaked Argus session. Non-blocking: if passwordless
# sudo isn't set up, it's skipped — restart it manually if a capture phase fails.
restart_argus() { sudo -n systemctl restart nvargus-daemon 2>/dev/null || true; sleep 3; }
src='source /opt/ros/foxy/setup.bash && source install/setup.bash'

# --- phase 1: capture node -> the 4 raw views (rotate 180: modules are mounted upside-down) ---
restart_argus
docker rm -f bev_grab >/dev/null 2>&1 || true
docker run -d --name bev_grab "${OPTS[@]}" bash -lc "$src && ros2 run bev_camera argus_capture_node" >/dev/null
sleep 9
docker exec bev_grab bash -lc "$src && python3 scripts/port/grab_views.py grab --out-dir /views --rotate180 \
  --topics /cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw"
docker rm -f bev_grab >/dev/null 2>&1

# --- phase 2: panorama node -> the stitched panorama (already upright via flip_180) ---
restart_argus
docker run -d --name bev_grab "${OPTS[@]}" bash -lc "$src && ros2 launch bev_cuvslam bev_panorama.launch.py" >/dev/null
sleep 18
docker exec bev_grab bash -lc "$src && python3 scripts/port/grab_views.py grab --out-dir /views --topics /bev/panorama"
docker rm -f bev_grab >/dev/null 2>&1
restart_argus

# --- phase 3: montage (4 cams on top, panorama below) ---
docker run --rm "${OPTS[@]}" python3 scripts/port/grab_views.py montage --dir /views --out /views/montage.png
cp "$VDIR/montage.png" "$OUT"
rm -rf "$VDIR"
echo "wrote $OUT"
