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
#   PORTS="c"   ./board_sender.sh          # stage 1-4: ONE camera
#   PORTS="c d" ./board_sender.sh 10.42.0.1 4   # a pair stage, at 7.5 Hz
#
# PUBLISH ONLY THE STAGE'S CAMERAS. A calibration stage uses one camera or two, never
# four, and every camera opened is 1.58 MB per published frame on the wire whether or not
# anything looks at it. Four cameras at EVERY_N=8 measured ~50% sample loss host-side;
# the cameras a stage does not use are pure cost, and dropping them buys the rate back.
# Argus also opens faster with fewer sessions (see the stagger below).
set -euo pipefail
HOST_IP="${1:-10.42.0.1}"
EVERY_N="${2:-8}"                    # 30 Hz / 8 = 3.75 Hz; 1 GbE carries 4 cams at ~19 MB/s
PORTS="${PORTS:-c d e f}"            # which ports this stage needs
# Gyro DLPF index -> bandwidth and GROUP DELAY, from the MPU-9250 datasheet table that
# imu_node.cpp carries: 0=250Hz/0.97ms  1=184Hz/2.9ms  2=92Hz/3.9ms  3=41Hz/5.9ms
# 4=20Hz/9.9ms.  The delay is REPORTED, not applied, so it lands inside Delta - which
# makes it measurable: change the index and Delta must move by the change in group delay.
GYRO_DLPF="${GYRO_DLPF:-1}"
# The stamp is SOF - exposure/2, so this MUST match the generator's asserted pulse width.
# Default reads it back rather than assuming: the rig has run at 4986 us and at 29986 us,
# which differ by 12.5 ms of stamp - 1550x the Delta we are trying to measure.
if [ -z "${EXPOSURE_US:-}" ]; then
  PULSE=$(python3 "${TRIGCTL:-/home/nvidia/tools/j106-trigctl.py}" --port "${TRIG_PORT:-/dev/ttyACM0}" status 2>/dev/null \
          | grep -oP 'ch1_exposure_us=\d+ pulse_ns=\K\d+')
  [ -n "$PULSE" ] || { echo "cannot read the trigger generator - set EXPOSURE_US explicitly"; exit 1; }
  EXPOSURE_US=$(( (PULSE + 500) / 1000 ))
fi
echo "exposure_us=$EXPOSURE_US (from the generator's measured pulse width)"
OUT="${OUT:-/media/nvidia/workspace/calib_$(date +%Y%m%d_%H%M)}"
IFACE="${IFACE:-eth0}"
DOMAIN="${ROS_DOMAIN_ID:-42}"

mkdir -p "$OUT"
sed -e "s/\${NET_IFACE}/$IFACE/" -e "s/\${PEER}/$HOST_IP/" \
    "$(dirname "$0")/cyclonedds_lowlat.xml" > "$OUT/cyclonedds.xml"

# Preflight: free-running cameras are the silent failure (see README 4.7 / rig_layout).
MODE=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo missing)
[ "$MODE" = "1" ] || { echo "trigger_mode=$MODE — set it to 1 first"; exit 1; }

# POLARITY. The F401 boots at its compiled-in default `pol 1` (active_high) after ANY power
# cycle, and this rig's optocouplers need `pol 0`. At the wrong polarity the sensor still
# triggers, frames still flow, nothing logs an error - but the asserted window is the
# COMPLEMENT of the commanded exposure: command 5000 us and the sensors expose for 28333.
# Measured 2026-09-01: at pol 1 the image was 3-4x darker at the periphery, which read as
# "the room is dark" until the polarity was checked. See hw-trigger/WIRING.md 9a.
POL=$(python3 "${TRIGCTL:-/home/nvidia/tools/j106-trigctl.py}" --port "${TRIG_PORT:-/dev/ttyACM0}" status 2>/dev/null \
      | grep -oP 'polarity=\K\w+')
if [ "$POL" != "active_low" ]; then
  echo "trigger polarity is '$POL', this rig needs active_low (pol 0)."
  echo "  the exposure would be the COMPLEMENT of what you command. Fix with:"
  echo "    python3 j106-trigctl.py --port ${TRIG_PORT:-/dev/ttyACM0} raw 'pol 0'"
  exit 1
fi

echo nvidia | sudo -S sysctl -q -w net.core.rmem_max=33554432 net.core.wmem_max=33554432 2>/dev/null || true

# ports -> camN is fixed by the carrier wiring (config/rig/rig_layout.yaml):
#   c = front-left = cam1    d = front-right = cam2
#   e = back-left  = cam3    f = back-right  = cam4
# The node takes ports/topics/frame_ids as PARALLEL vectors and only checks the lengths,
# so a subset must carry its own topic names - otherwise PORTS="d" would publish port d
# on /cam1/image_raw and every downstream file would name the wrong camera.
declare -A CAMOF=( [c]=cam1 [d]=cam2 [e]=cam3 [f]=cam4 )
P_ARG="["; T_ARG="["; F_ARG="["
for p in $PORTS; do
  cam="${CAMOF[$p]:-}"
  [ -n "$cam" ] || { echo "unknown port '$p' (expected c, d, e or f)"; exit 1; }
  P_ARG="$P_ARG$p,"; T_ARG="$T_ARG/$cam/image_raw,"; F_ARG="$F_ARG$cam,"
done
P_ARG="${P_ARG%,}]"; T_ARG="${T_ARG%,}]"; F_ARG="${F_ARG%,}]"
echo "publishing ports $PORTS -> $T_ARG at 1/$EVERY_N of 30 Hz"

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
      -p ports:=\"$P_ARG\" -p topics:=\"$T_ARG\" -p frame_ids:=\"$F_ARG\" \
      -p exposure_us:=$EXPOSURE_US -p publish_every_n:=$EVERY_N -p frame_log_dir:=/out \
      > /out/capture.log 2>&1 &
    sleep 6
    exec ros2 run bev_imu imu_node --ros-args -p csv:=/out/imu.csv -p gyro_dlpf:=$GYRO_DLPF > /out/imu.log 2>&1
  " >/dev/null
sleep 15
docker ps --filter name=calibcap --format '{{.Status}}'
grep -E "capture up" "$OUT/capture.log" || { tail -20 "$OUT/capture.log"; exit 1; }
echo "publishing on domain $DOMAIN via $IFACE -> $HOST_IP; data dir $OUT"
