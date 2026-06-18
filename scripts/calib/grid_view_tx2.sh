#!/usr/bin/env bash
# Live 2x3 IMX219 camera grid on the TX2's own HDMI display, to identify/label
# which Argus sensor-id is which physical camera. Run on the TX2.
#
#   ./grid_view_tx2.sh ["0 1 2 3 4"]
#
# Layout (sensor-id):  [0][1][2]
#                      [3][4][5]      (empty/missing cells show "CAM N - none")
# Tip: wave a hand at each physical camera and see which cell moves.
#
# Baked-in from the j106 bring-up (tools/grid-display-x.sh):
#   * nvcompositor only composites cleanly to nv3dsink (a local X window) on this
#     board — it fails ("nvbuffer_composite Failed") to encoder/file sinks, so the
#     grid is shown on the TX2 display, not streamed.
#   * a `queue` before each sink fixes live-source latency negotiation.
#   * 5 simultaneous Argus sessions race — restart nvargus-daemon first + retry.
set -u
export PATH=/usr/bin:/bin:/usr/local/bin:/usr/sbin:/sbin
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
CAMS="${1:-0 1 2 3 4}"
CW=640; CH=360; FR=30
xs=(0 640 1280 0 640 1280)
ys=(0 0 0 360 360 360)

build_args() {
  A=(nvcompositor name=comp)
  local n
  for n in 0 1 2 3 4 5; do
    A+=("sink_${n}::xpos=${xs[$n]}" "sink_${n}::ypos=${ys[$n]}"
        "sink_${n}::width=${CW}" "sink_${n}::height=${CH}")
  done
  A+=("!" "video/x-raw(memory:NVMM),width=1920,height=720" "!" nv3dsink sync=false)
  local i=0 sid
  declare -ga _have=()
  for sid in $CAMS; do _have[$i]="$sid"; i=$((i + 1)); done
  for n in 0 1 2 3 4 5; do
    if [ -n "${_have[$n]:-}" ]; then
      A+=(nvarguscamerasrc "sensor-id=${_have[$n]}"
          "!" "video/x-raw(memory:NVMM),width=${CW},height=${CH},framerate=${FR}/1"
          "!" nvvidconv flip-method=2 "!" queue "!" "comp.sink_${n}")
    else
      A+=(videotestsrc is-live=true pattern=2
          "!" "video/x-raw,width=${CW},height=${CH},framerate=${FR}/1"
          "!" textoverlay "text=cell ${n} - none" valignment=center halignment=center font-desc="Sans Bold 26"
          "!" nvvidconv "!" "video/x-raw(memory:NVMM),width=${CW},height=${CH}" "!" queue "!" "comp.sink_${n}")
    fi
  done
}

echo "Restarting nvargus-daemon (needs sudo) to clear the multi-session race..."
sudo systemctl restart nvargus-daemon 2>/dev/null || echo "(could not restart daemon; continuing)"
sleep 2
build_args
echo "Grid on TX2 HDMI display.  Cell layout (sensor-id, reading order): ${CAMS}"
for attempt in 1 2 3 4 5; do
  gst-launch-1.0 -e "${A[@]}" 2>/tmp/grid_err.log &
  GPID=$!
  sleep 9
  if kill -0 "$GPID" 2>/dev/null && ! grep -qiE "error|not-negotiated|data stream|composite Failed" /tmp/grid_err.log; then
    echo "grid live on the TX2 monitor (attempt ${attempt}, pid ${GPID}). Ctrl-C to stop."
    wait "$GPID"; exit 0
  fi
  echo "attempt ${attempt} raced/failed; restarting daemon + retrying..."
  kill "$GPID" 2>/dev/null || true
  sudo systemctl restart nvargus-daemon 2>/dev/null || true
  sleep 2
done
echo "grid failed after retries — see /tmp/grid_err.log" >&2
exit 1
