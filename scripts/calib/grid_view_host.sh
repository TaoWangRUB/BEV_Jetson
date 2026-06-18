#!/usr/bin/env bash
# Display the 2x3 camera grid streamed from the TX2 (start grid_view_tx2.sh there).
# Run on the HOST (needs a display + GStreamer with avdec_h264). Open this first.
#
#   ./grid_view_host.sh [port]
#
# Grid layout (sensor-id, reading order):   0 1 2
#                                           3 4 (5)
PORT="${1:-5005}"
echo "Listening for the camera grid on udp://0.0.0.0:${PORT} ..."
exec gst-launch-1.0 -e \
  udpsrc port="${PORT}" caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtpjitterbuffer latency=80 ! rtph264depay ! h264parse ! avdec_h264 \
  ! videoconvert ! textoverlay text="grid: ids 0 1 2 / 3 4 (5)" valignment=bottom halignment=left font-desc="Sans 18" \
  ! autovideosink sync=false
