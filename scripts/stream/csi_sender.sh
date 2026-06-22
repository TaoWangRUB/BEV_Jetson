#!/usr/bin/env bash
# CSI -> H.264 / RTP / UDP sender.  RUN ON THE TX2 (board host, not in a container).
#
# Streams every present camera over UDP so you can preview them live from the host with
# scripts/stream/csi_receiver.sh (a 2x3 port grid) -- a quick "which camera is alive?" check
# without ROS or docker. Port b has no sensor (skipped). Mapping (this rig):
#   port a = sensor 0   (cam0, not in the calibrated BEV set)
#   port c = sensor 1 = cam1     port d = sensor 2 = cam2
#   port e = sensor 3 = cam3     port f = sensor 4 = cam4
# Each port -> its own UDP port (a=5000, c=5001, d=5002, e=5003, f=5004).
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
PORTS=(a c d e f)                              # port b absent (no sensor)
declare -A SID=( [a]=0 [c]=1 [d]=2 [e]=3 [f]=4 )
declare -A UDP=( [a]=5000 [c]=5001 [d]=5002 [e]=5003 [f]=5004 )

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
  sleep 1   # stagger Argus session starts; simultaneous opens can lose the race (a camera fails to stream)
done

echo "streaming ${#PORTS[@]} cameras to $HOST_IP -- run csi_receiver.sh on the host. Ctrl-C to stop."
wait
