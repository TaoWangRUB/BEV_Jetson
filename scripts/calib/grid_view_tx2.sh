#!/usr/bin/env bash
# Composite the working IMX219 cameras into a 2x3 grid and H264-stream it to the
# host, so you can identify/label which Argus sensor-id is which physical camera.
# Run on the TX2 (headless); view it on the host with grid_view_host.sh.
#
#   ./grid_view_tx2.sh [host-ip] [port] ["0 1 2 3 4"]
#
# Grid cells fill in reading order from the camera list:
#     pos0 pos1 pos2        (top row)
#     pos3 pos4 pos5        (bottom row)
# e.g. cameras "0 1 2 3 4" -> ids 0,1,2 on top; 3,4 on bottom; last cell empty.
# Tip: wave a hand in front of each physical camera and see which cell moves.
set -uo pipefail
HOST="${1:-10.42.0.1}"
PORT="${2:-5005}"
CAMS="${3:-0 1 2 3 4}"
CW=640; CH=360; FR=15

xs=(0 640 1280 0 640 1280)
ys=(0 0 0 360 360 360)

ARGS=(nvcompositor name=comp)
i=0
for _ in $CAMS; do
  ARGS+=("sink_${i}::xpos=${xs[$i]}" "sink_${i}::ypos=${ys[$i]}"
         "sink_${i}::width=${CW}" "sink_${i}::height=${CH}")
  i=$((i + 1))
done
ARGS+=("!" "video/x-raw(memory:NVMM),width=1920,height=720"
       "!" nvv4l2h264enc insert-sps-pps=1 idrinterval=15
       "!" h264parse "!" rtph264pay config-interval=1 pt=96
       "!" udpsink "host=${HOST}" "port=${PORT}" sync=false)
i=0
for sid in $CAMS; do
  ARGS+=(nvarguscamerasrc "sensor-id=${sid}"
         "!" "video/x-raw(memory:NVMM),width=${CW},height=${CH},framerate=${FR}/1"
         "!" "comp.sink_${i}")
  i=$((i + 1))
done

echo "Grid (reading order) = sensor-ids: ${CAMS}  ->  streaming to ${HOST}:${PORT}"
echo "Start the viewer on the host:  ./scripts/calib/grid_view_host.sh ${PORT}"
exec gst-launch-1.0 -e "${ARGS[@]}"
