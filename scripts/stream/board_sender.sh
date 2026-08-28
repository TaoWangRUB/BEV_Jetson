#!/usr/bin/env bash
# TX2 side: capture and publish, and NOTHING else.
#
# The board is a 6-core TX2 that also runs the ISP and Argus. Recording, tag detection
# and preview encoding on top of capture pushed load to 8 and starved everything —
# frames still captured, but the preview died and the bag crawled. So those all move to
# the host, and this side does only what has to happen here: Argus capture, per-frame
# timing (written locally as CSV, tiny), and publishing over DDS.
#
#   ./board_sender.sh [HOST_IP] [EVERY_N]
set -euo pipefail
HOST_IP="${1:-10.42.0.1}"
EVERY_N="${2:-8}"                    # 30 Hz / 8 = 3.75 Hz; 1 GbE carries 4 cams at ~19 MB/s
OUT="${OUT:-/media/nvidia/workspace/calib_$(date +%Y%m%d_%H%M)}"
IFACE="${IFACE:-eth0}"
DOMAIN="${ROS_DOMAIN_ID:-42}"

mkdir -p "$OUT"
sed -e "s/\${NET_IFACE}/$IFACE/" -e "s/\${PEER}/$HOST_IP/" \
    "$(dirname "$0")/cyclonedds_lowlat.xml" > "$OUT/cyclonedds.xml"

# Preflight: free-running cameras are the silent failure (see README 4.7 / rig_layout).
MODE=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo missing)
[ "$MODE" = "1" ] || { echo "trigger_mode=$MODE — set it to 1 first"; exit 1; }

echo nvidia | sudo -S sysctl -q -w net.core.rmem_max=33554432 net.core.wmem_max=33554432 2>/dev/null || true

docker rm -f calibcap >/dev/null 2>&1 || true
docker run -d --name calibcap --runtime nvidia --network host --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_DOMAIN_ID="$DOMAIN" \
  -e CYCLONEDDS_URI=file:///out/cyclonedds.xml \
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev \
  -v /media/nvidia/workspace/bev_build_test:/ws -v "$OUT":/out -w /ws \
  cuvslam-foxy:tx2 bash -lc "
    source install/setup.bash
    mkdir -p /root/.ros/log
    ros2 run bev_camera argus_capture_node --ros-args \
      -p exposure_us:=4986 -p publish_every_n:=$EVERY_N -p frame_log_dir:=/out \
      > /out/capture.log 2>&1 &
    sleep 6
    exec ros2 run bev_imu imu_node --ros-args -p csv:=/out/imu.csv > /out/imu.log 2>&1
  " >/dev/null
sleep 15
docker ps --filter name=calibcap --format '{{.Status}}'
grep -E "capture up" "$OUT/capture.log" || { tail -20 "$OUT/capture.log"; exit 1; }
echo "publishing on domain $DOMAIN via $IFACE -> $HOST_IP; data dir $OUT"
