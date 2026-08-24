#!/usr/bin/env bash
# CSI -> H.264 / RTP / UDP sender.  RUN ON THE TX2 (board host, not in a container).
#
# Streams every present camera over UDP so you can preview them live from the host with
# scripts/stream/csi_receiver.sh (a 2x3 port grid) -- a quick "which camera is alive?" check
# without ROS or docker.
#
# Port -> Argus sensor-id is resolved AT RUNTIME from each /dev/videoN's i2c name. Do NOT hard-code
# it: Argus numbers cameras in /dev/video order, which is bind order, not port order. (Seen live:
# binding F before E gives video4=7-0012=port f and video5=7-0010=port e.) Carrier wiring is fixed:
#   port a = imx219 1-0010   port b = 1-0012   port c = 2-0010
#   port d = 2-0012          port e = 7-0010   port f = 7-0012
# Each port -> its own UDP port (a=5000, b=5001, c=5002, d=5003, e=5004, f=5005).
#
#   ./csi_sender.sh [HOST_IP]        # HOST_IP = receiving host (default 10.42.0.1 = eth direct link;
#                                    #           over wifi pass the host's wlan IP, e.g. 192.168.0.225)
#
# Uses nvarguscamerasrc (L4T R32.7) + nvv4l2h264enc. ~4 Mbit/s/cam (5 cams ~20 Mbit/s, trivial on
# 1G eth). Argus serves ONE consumer at a time -> stop the ROS capture / docker camera runs first.
# Ctrl-C stops all. If a camera is black, reset it (README 4.3): nvargus-daemon -> j106_reset_recover
# -> cold power-cycle.

HOST_IP="${1:-10.42.0.1}"
W=640; H=480; FPS=30; BR=4000000
declare -A DEVOF=( [a]=1-0010 [b]=1-0012 [c]=2-0010 [d]=2-0012 [e]=7-0010 [f]=7-0012 )
declare -A UDP=( [a]=5000 [b]=5001 [c]=5002 [d]=5003 [e]=5004 [f]=5005 )

# resolve port -> Argus sensor-id: the Nth /dev/video (numeric order) is sensor-id N
declare -A SID=(); sid=0
for V in $(ls -v /dev/video* 2>/dev/null); do
  dev=$(cat "/sys/class/video4linux/$(basename "$V")/name" 2>/dev/null)   # "vi-output, imx219 2-0010"
  dev=${dev##* }                                                          # -> 2-0010
  for p in a b c d e f; do [ "${DEVOF[$p]}" = "$dev" ] && SID[$p]=$sid; done
  sid=$((sid+1))
done
PORTS=()
for p in a b c d e f; do [ -n "${SID[$p]}" ] && PORTS+=("$p"); done
echo "cameras present: ${PORTS[*]:-none} (${#PORTS[@]}/6)"

# Argus admits at most 5 concurrent capture sessions here: the DT budget
# tegra-camera-platform/max_pixel_rate = 240000 kpix/s, and 6 x mode4 (1280x720@44)
# needs 243302. The 6th session fails "Failed to create CaptureSession" AND wedges the
# daemon, so the other five stall too -- never launch more than 5.
# Override which ports to stream:  PORTS_ONLY="a b c d f" ./csi_sender.sh
if [ -n "${PORTS_ONLY:-}" ]; then
  sel=(); for p in $PORTS_ONLY; do [ -n "${SID[$p]}" ] && sel+=("$p"); done
  PORTS=("${sel[@]}"); echo "streaming subset: ${PORTS[*]}"
elif [ ${#PORTS[@]} -gt 5 ]; then
  PORTS=("${PORTS[@]:0:5}"); echo "capped to 5 (Argus limit): ${PORTS[*]}"
fi

# clean any previous streamers (by process NAME, so this never matches its own command line)
killall -9 gst-launch-1.0 2>/dev/null || true
trap 'echo; echo "stopping sender..."; pkill -P $$ 2>/dev/null; exit 0' INT TERM

for p in "${PORTS[@]}"; do
  gst-launch-1.0 -q \
    nvarguscamerasrc sensor-id="${SID[$p]}" ! \
    "video/x-raw(memory:NVMM),width=$W,height=$H,framerate=$FPS/1,format=NV12" ! \
    nvv4l2h264enc bitrate=$BR insert-sps-pps=1 iframeinterval=15 idrinterval=15 maxperf-enable=1 ! \
    h264parse ! rtph264pay config-interval=1 pt=96 ! \
    udpsink host="$HOST_IP" port="${UDP[$p]}" sync=false async=false &
  echo "  port $p (sensor-id ${SID[$p]}) -> udp $HOST_IP:${UDP[$p]}"
  sleep 3   # stagger Argus session starts; simultaneous opens lose the race (a camera fails to stream,
            # and at 6 sessions the failure cascades: CANCELLED -> DISCONNECTED -> daemon socket reset,
            # taking every stream down). 1 s was enough for 5 cameras, 6 needs more.
done

echo "streaming ${#PORTS[@]} cameras to $HOST_IP -- run csi_receiver.sh on the host. Ctrl-C to stop."
wait
