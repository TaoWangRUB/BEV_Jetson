#!/usr/bin/env bash
# HOST side of the ROS calibration path: view and record the TX2's topics over DDS.
#
# The counterpart to board_sender.sh, which runs Argus capture + IMU on the board and
# NOTHING else. Everything that costs CPU lives here, because the host has 16 idle cores
# and the TX2 has 6 already running the ISP:
#
#   TX2  argus_capture_node + imu_node ──CycloneDDS──▶  host  preview  (this script)
#        one Argus consumer, timing to CSV     domain 42        record   (this script)
#
# The preview and the recording are BOTH on this end, off the same DDS stream, so they
# do not contend: Argus serves one consumer at a time, and that consumer is the capture
# node. Any second camera opener (csi_sender.sh, calib_sender.sh, a docker camera run)
# must be stopped first or the capture node gets nothing.
#
#   ./host_receiver.sh preview [--detect cam1] [--port 8080]
#   ./host_receiver.sh record  datasets/calib_20260831/stage1_cam1
#   ./host_receiver.sh topics                  # what is actually arriving
#   ./host_receiver.sh tune --jumbo            # link tuning only, then exit
#
# preview and record are separate containers on the same domain, so recording can be
# started and stopped without disturbing the preview.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BOARD="${BOARD:-10.42.0.157}"
IFACE="${IFACE:-$(ip -4 -o addr show | awk '/10\.42\.0\.1\//{print $2; exit}')}"
DOMAIN="${ROS_DOMAIN_ID:-42}"
IMAGE="${IMAGE:-bev-host-foxy:latest}"
RUN="/tmp/bev-host-foxy"

[ -n "$IFACE" ] || { echo "no interface on the 10.42.0.0/24 link — is the cable in?" >&2; exit 1; }

# ---- link tuning ----------------------------------------------------------------
# The XML asks for a 16 MB DDS receive buffer; the kernel silently clamps that to
# net.core.rmem_max, and a clamped buffer drops fragments under a 1.58 MB burst. One
# lost fragment discards the WHOLE sample, which is why this reads as "images are laggy"
# rather than as an error. board_sender.sh does the same on its end.
tune() {
  sudo -n sysctl -q -w net.core.rmem_max=33554432 net.core.wmem_max=33554432 \
                       net.core.rmem_default=16777216 net.core.wmem_default=16777216 \
                       net.core.netdev_max_backlog=30000 2>/dev/null \
    || echo "  (could not set sysctls — need passwordless sudo; DDS will still work, slower)"
  echo "  host socket buffers: rmem_max=$(cat /proc/sys/net/core/rmem_max)"

  # Jumbo frames, both ends or neither. At MTU 1500 a 64 kB DDS fragment is ~45 IP
  # fragments and losing any one of them costs the whole 1.58 MB sample; at 9000 it is
  # ~8. Both this NIC and the board's eth0 accept 9000 (verified 2026-08-31).
  if [ "${1:-}" = "--jumbo" ]; then
    ssh "${SSH_HOST:-tx2-eth}" "echo nvidia | sudo -S ip link set dev eth0 mtu 9000" 2>/dev/null || {
      echo "  board MTU change FAILED — leaving host at 1500 so the two ends match" >&2; return; }
    sudo -n ip link set dev "$IFACE" mtu 9000 || {
      echo "  host MTU change failed; reverting the board so the ends match" >&2
      ssh "${SSH_HOST:-tx2-eth}" "echo nvidia | sudo -S ip link set dev eth0 mtu 1500" 2>/dev/null; return; }
    echo "  MTU 9000 on both ends (host $IFACE, board eth0)"
  else
    echo "  MTU $(cat /sys/class/net/$IFACE/mtu) (pass --jumbo for 9000 on both ends)"
  fi
}

# ---- config + image -------------------------------------------------------------
prepare() {
  mkdir -p "$RUN"
  sed -e "s/\${NET_IFACE}/$IFACE/" -e "s/\${PEER}/$BOARD/" \
      scripts/stream/cyclonedds_lowlat.xml > "$RUN/cyclonedds.xml"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "building $IMAGE (osrf/ros:foxy-desktop ships fastrtps only; this link is CycloneDDS)"
    docker build -q -f docker/Dockerfile.host-foxy -t "$IMAGE" . >/dev/null
  fi
}

# Foxy is Ubuntu 20.04's ROS and the host's python3 is 3.12, so the native /opt/ros/foxy
# cannot be used from here (missing _rclpy.cpython-312). Hence the container.
#
# Runs as the invoking uid so recorded bags are not root-owned (the 2026-08-28 solve
# outputs were, and cleaning them up needed sudo). That leaves HOME unset inside, and
# rcl then tries to create "//.ros/log" and aborts before any node starts - hence HOME.
dock() {
  local name="$1"; shift
  local tty=""; [ -t 0 ] && tty="-it"          # -t without a terminal aborts the run
  docker run --rm $tty --name "$name" --network host \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_DOMAIN_ID="$DOMAIN" \
    -e HOME=/tmp \
    -e CYCLONEDDS_URI=file:///cfg/cyclonedds.xml \
    -v "$RUN":/cfg -v "$PWD":/repo -w /repo \
    -u "$(id -u):$(id -g)" \
    "$IMAGE" bash -lc "$*"
}

CMD="${1:-preview}"; shift || true
case "$CMD" in
  tune)
    tune "${1:-}" ;;

  topics)
    prepare
    # `topic list` only proves DISCOVERY. Data flowing is a separate question, and
    # `ros2 topic hz` CANNOT answer it here: it subscribes RELIABLE, the capture node
    # publishes BEST_EFFORT (deliberately - a reliable subscriber can back-pressure the
    # Argus thread), the two never match, and hz then prints nothing at all. That silence
    # is indistinguishable from a dead link and cost an hour chasing MTU. Foxy's hz has no
    # --qos-profile, so count with a sensor-data subscription instead.
    dock bev-topics "ros2 topic list && python3 - <<'EOF'
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import time
rclpy.init()
n = Node('bev_flowcheck')
seen = {}
for cam in ('cam1', 'cam2', 'cam3', 'cam4'):
    n.create_subscription(Image, '/%s/image_raw' % cam,
                          lambda m, c=cam: seen.__setitem__(c, seen.get(c, 0) + 1),
                          qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.2)
dt = time.time() - t0
print('--- frames in %.0f s (best_effort/sensor_data QoS) ---' % dt)
for cam in ('cam1', 'cam2', 'cam3', 'cam4'):
    print('  /%s/image_raw  %3d  (%.2f Hz)' % (cam, seen.get(cam, 0), seen.get(cam, 0) / dt))
EOF" ;;

  preview)
    DETECT="cam1"; PORT=8080          # DETECT may be several cams: --detect cam3 cam1
    # Detecting on EVERY frame of TWO full-res streams starves the DDS subscriber on this
    # host: measured 7.48/7.58 Hz with the preview off against 6.96/6.66 with it on, at the
    # same bitrate that one camera carried losslessly. It is CPU, not the link. Detect on
    # every 3rd frame for a pair - coverage still fills, at a third of the cost.
    EVERY="${DETECT_EVERY:-1}"
    # This rig's modules are mounted inverted and the capture node publishes raw sensor
    # orientation, so the preview is upside-down without this. Display only - the bag is
    # recorded off DDS and is never touched by it. ROTATE=0 to disable.
    ROT="--rotate180"; [ "${ROTATE:-1}" = "0" ] && ROT=""
    # --detect takes a LIST: a pair stage needs both cameras detecting at once, because a
    # pair frame only counts when both saw the board at the same instant. Collect every
    # camN following --detect rather than just the first.
    while [ $# -gt 0 ]; do
      case "$1" in
        --detect) DETECT=""; shift
                  while [ $# -gt 0 ] && [ "${1#cam}" != "$1" ]; do DETECT="$DETECT $1"; shift; done ;;
        --port)   PORT="$2"; shift 2 ;;
        *)        shift ;;
      esac
    done
    DETECT="${DETECT# }"
    tune; prepare
    echo "$PORT" > "$RUN/preview_port"        # so `record` can clear the coverage grid
    echo "preview on http://localhost:$PORT/   (detection + coverage grid on: $DETECT)"
    SEEDARG=""; [ -n "${SEED:-}" ] && SEEDARG="--seed $SEED"
    dock bev-preview "python3 scripts/stream/preview_server.py --port $PORT --detect $DETECT --detect-every $EVERY $ROT $SEEDARG" ;;

  record)
    OUT="${1:?usage: host_receiver.sh record <output-dir>}"
    tune; prepare
    mkdir -p "$(dirname "$OUT")"
    # Clear the coverage grid at the start of every take. Coverage accumulated while
    # AIMING is not coverage of the recording: the grid would show cells filled by frames
    # that no bag contains, and the operator stops sweeping believing a region is covered.
    # The grid must answer "is THIS bag covered?" and nothing else.
    if [ -f "$RUN/preview_port" ]; then
      if curl -fs "http://localhost:$(cat "$RUN/preview_port")/reset" >/dev/null; then
        echo "  coverage grid cleared"
      else
        echo "  WARNING: could not clear the coverage grid (preview not running?) —" \
             "what it shows includes detections from BEFORE this recording" >&2
      fi
    else
      echo "  WARNING: no preview running, so no coverage feedback for this take" >&2
    fi
    # ONLY topics whose types exist in this image. /camN/frame_meta is
    # bev_camera/msg/FrameMeta, a package built on the BOARD, and rosbag2 aborts the
    # WHOLE recording on an unknown type rather than skipping that one topic.
    #
    # Nothing is lost by omitting it: frame_log_dir makes the capture node write
    # camN.csv on the board at FULL rate (not the decimated publish rate) with sof_ns,
    # exposure and both sequence counters, and that CSV is the authoritative timing
    # record - the bag's FrameMeta was always the convenience copy. Collect it with the
    # bag (`scp` the board's data dir) and the session is complete.
    #
    # Missing topics are fine here: rosbag2 records those that appear, so listing all
    # four cameras works whatever PORTS the board is publishing.
    dock bev-record "ros2 bag record -o /repo/$OUT \
        /cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw /imu0" ;;

  *) sed -n '1,25p' "$0"; exit 1 ;;
esac
