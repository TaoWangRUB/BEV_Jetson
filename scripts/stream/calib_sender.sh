#!/usr/bin/env bash
# TX2 side of the calibration capture: Argus -> hardware JPEG -> MJPEG over TCP.
#
# The board does capture and nothing else. Everything that thinks - tag detection,
# preview, coverage, recording - runs on the host, which has 16 idle cores while the
# TX2 has 6 that are already busy with the ISP. Running detection and a bag recorder
# here pushed load to 8 and starved the stream; this pipeline barely touches the CPU
# because nvjpegenc is the hardware encoder.
#
# MJPEG, not H.264, on purpose: every frame is independent and there are no inter-frame
# artifacts to smear a tag corner, which is the one thing a calibration cannot tolerate.
# At quality 92 on 1 GbE this is ~5 MB/s per camera at 4 Hz - nowhere near the link.
#
# ⚠ The RTP/TCP path carries NO capture timestamp. That is fine for intrinsics and for
# the pairwise extrinsics, which need no time at all. It is NOT fine for the camera-IMU
# stage, where the offset is the thing being measured - that stage needs the ROS path
# (docs/timestamps.md) or the J106 frame-time tooling.
#
#   ./calib_sender.sh <port-letter|all> [fps] [quality]
#   e.g. ./calib_sender.sh c 4        -> front-left camera on tcp 5000
set -euo pipefail
WHICH="${1:-all}"; FPS="${2:-4}"; Q="${3:-92}"
W=1456; H=1088

declare -A DEVOF296=( [c]=2-001a [d]=2-0018 [e]=7-001a [f]=7-0018 )
declare -A PORT=( [c]=5000 [d]=5001 [e]=5002 [f]=5003 )
declare -A NAME=( [c]="cam1 front-left" [d]="cam2 front-right" [e]="cam3 back-left" [f]="cam4 back-right" )

# Resolve port -> Argus sensor-id at runtime: Argus numbers cameras in /dev/video bind
# order, which is not port order and is not stable across boots.
declare -A SID; sid=0
for V in $(ls -v /dev/video* 2>/dev/null); do
  dev=$(cat "/sys/class/video4linux/$(basename "$V")/name" 2>/dev/null); dev=${dev##* }
  for p in c d e f; do [ "${DEVOF296[$p]}" = "$dev" ] && SID[$p]=$sid; done
  sid=$((sid+1))
done

MODE=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo 0)
if [ "$MODE" = "1" ]; then
  # Under external trigger the exposure IS the pulse width, so AE cannot move its main
  # actuator and hunts on gain instead - a 3.5 Hz limit cycle at 171% of mean luma.
  AE='aelock=true gainrange="16 16" ispdigitalgainrange="4 4"'
  echo "external trigger active -> AE locked"
else
  AE=''
  echo "⚠ trigger_mode=0: cameras are FREE-RUNNING and unsynchronised"
fi

[ "$WHICH" = "all" ] && PORTS="c d e f" || PORTS="$WHICH"

echo nvidia | sudo -S systemctl restart nvargus-daemon >/dev/null 2>&1
sleep 3
killall -9 gst-launch-1.0 2>/dev/null || true
trap 'echo; echo stopping; pkill -P $$ 2>/dev/null; exit 0' INT TERM

for p in $PORTS; do
  [ -n "${SID[$p]:-}" ] || { echo "no camera on port $p"; continue; }
  echo "port $p (${NAME[$p]}) sensor-id ${SID[$p]} -> tcp://0.0.0.0:${PORT[$p]}  ${FPS} fps q${Q}"
  eval gst-launch-1.0 -q nvarguscamerasrc sensor-id="${SID[$p]}" $AE \
    ! "'video/x-raw(memory:NVMM),width=$W,height=$H,framerate=30/1,format=NV12'" \
    ! nvvidconv ! "'video/x-raw,format=I420'" \
    ! videorate ! "'video/x-raw,framerate=$FPS/1'" \
    ! nvjpegenc quality=$Q ! multipartmux ! tcpserversink host=0.0.0.0 port="${PORT[$p]}" \
      sync=false async=false recover-policy=keyframe &
  sleep 2
done
echo; echo "streaming. Ctrl-C to stop."
wait
