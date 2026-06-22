#!/usr/bin/env bash
# CSI -> H.264 / RTP / UDP sender.  RUN ON THE TX2 (board host, not in a container).
#
# Streams the 4 BEV fisheye cameras over UDP so you can preview them live from the host with
# scripts/stream/csi_receiver.sh -- a quick "are all cameras alive / which one is dark?" check
# without ROS or docker. Mapping (this rig): Argus sensor-id 1=port c=cam1, 2=d=cam2, 3=e=cam3,
# 4=f=cam4 (sensor 0 = port a, unused). Each camera -> its own UDP port 5001..5004.
#
#   ./csi_sender.sh [HOST_IP]        # HOST_IP = receiving host (default 10.42.0.1 = eth direct link;
#                                    #           use the host's wifi IP, e.g. 192.168.0.x, over wlan)
#
# Uses nvarguscamerasrc (L4T R32.7) + nvv4l2h264enc. Argus serves ONE consumer at a time, so stop the
# ROS capture / any docker camera run first (this owns the cameras while it runs). Ctrl-C to stop all.
#
# If a camera is black, reset it (see README 4.3): restart nvargus-daemon, then j106_reset_recover,
# then a cold power-cycle.

HOST_IP="${1:-10.42.0.1}"
W=640; H=480; FPS=30; BR=4000000              # 640x480@30, ~4 Mbit/s each (~16 Mbit/s total)
SIDS=(1 2 3 4)                                # Argus sensor-ids = ports c,d,e,f = cam1..4
declare -A UDP=( [1]=5001 [2]=5002 [3]=5003 [4]=5004 )
declare -A NAME=( [1]="cam1 port c" [2]="cam2 port d" [3]="cam3 port e" [4]="cam4 port f" )

# drop any previous sender, and kill our children on exit
pkill -f "gst-launch-1.0.*nvarguscamerasrc" 2>/dev/null || true
trap 'echo; echo "stopping sender..."; pkill -P $$ 2>/dev/null; exit 0' INT TERM

for SID in "${SIDS[@]}"; do
  gst-launch-1.0 -q \
    nvarguscamerasrc sensor-id="$SID" ! \
    "video/x-raw(memory:NVMM),width=$W,height=$H,framerate=$FPS/1,format=NV12" ! \
    nvv4l2h264enc bitrate=$BR insert-sps-pps=1 iframeinterval=15 idrinterval=15 maxperf-enable=1 ! \
    h264parse ! rtph264pay config-interval=1 pt=96 ! \
    udpsink host="$HOST_IP" port="${UDP[$SID]}" sync=false async=false &
  echo "  ${NAME[$SID]} (sensor-id $SID) -> udp $HOST_IP:${UDP[$SID]}"
done

echo "streaming 4 cameras to $HOST_IP -- run csi_receiver.sh on the host. Ctrl-C to stop."
wait
