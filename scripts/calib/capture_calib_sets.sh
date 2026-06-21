#!/usr/bin/env bash
# Capture N sets of raw 4-camera frames for extrinsic calibration (scripts/calib/extrinsic_calib.py).
#
# Run on the TX2 host from the repo root. The 4 cameras are grabbed every INTERVAL seconds into
# scripts/calib/capture/setNN/camX_image_raw.png. SLOWLY PAN/ROTATE the whole rig during capture so
# each set sees a different scene in every overlap band -> the rotation becomes over-determined.
#
# For a reliable result aim at a DISTANT, TEXTURED scene (objects > ~5 m, e.g. across the room / out a
# door) so the ~3 cm rig baseline's parallax is negligible (the calibrator assumes pure rotation).
#
#   ./scripts/calib/capture_calib_sets.sh [N_SETS] [INTERVAL_S]
# then pull the sets to your dev box and:
#   python3 scripts/calib/extrinsic_calib.py --images scripts/calib/capture/set*
set -e
N=${1:-8}
INTERVAL=${2:-4}
ROOT=/media/nvidia/workspace/BEV_Jetson
OUT=$ROOT/scripts/calib/capture
cd "$ROOT"

echo nvidia | sudo -S systemctl restart nvargus-daemon
sleep 4
docker ps -aq --filter "ancestor=cuvslam-foxy:tx2" | xargs -r docker rm -f >/dev/null 2>&1
sudo rm -rf "$OUT"/set*; mkdir -p "$OUT"   # sudo: prior sets are written root-owned by the container

OPTS="--rm --runtime nvidia --network host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/local/cuda:/usr/local/cuda:ro -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev -v $ROOT:/workspace -w /workspace cuvslam-foxy:tx2"

docker run -d --name calibcap $OPTS bash -lc \
  "source /opt/ros/foxy/setup.bash && source install/setup.bash && \
   ros2 run bev_camera argus_capture_node --ros-args -p sensor_ids:=[1,2,3,4] \
   -p width:=1640 -p height:=1232 -p fps:=20" >/dev/null
sleep 9

for i in $(seq 0 $((N - 1))); do
  d=$(printf "set%02d" "$i")
  echo ">>> grabbing $d ($((i + 1))/$N) -- keep panning the rig slowly..."
  docker exec calibcap bash -lc \
    "source /opt/ros/foxy/setup.bash && python3 scripts/port/grab_views.py grab --out-dir /workspace/scripts/calib/capture/$d \
     --topics /cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw" || true
  sleep "$INTERVAL"
done

docker stop -t 3 calibcap >/dev/null 2>&1 || true
echo nvidia | sudo -S systemctl restart nvargus-daemon
echo "done: $(ls -d "$OUT"/set* | wc -l) sets in $OUT"
