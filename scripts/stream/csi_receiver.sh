#!/usr/bin/env bash
# Receive the CSI H.264 / RTP / UDP streams from the TX2 (scripts/stream/csi_sender.sh) and show them
# in a 2x3 grid laid out BY PORT.  RUN ON THE HOST.
#
#   ./csi_receiver.sh
#
# Layout (J106 ports):   [a][b][c]      a=5000  b=5001  c=5002
#                        [d][e][f]      d=5003  e=5004  f=5005
# Every port has its own UDP port. Any camera that isn't streaming shows its
# "port X (no signal)" placeholder instead of stalling the grid (compositor ignore-inactive-pads).
# Ctrl-C to stop. Needs gstreamer1.0 + gst-libav (avdec_h264).

CW=480; CH=360                                 # per-cell size -> 1440x720 canvas
CANVAS_W=$((CW*3)); CANVAS_H=$((CH*2))
CAPS="application/x-rtp,media=video,encoding-name=H264,payload=96"
PORTS_ALL=(a b c d e f)                         # all 6 grid cells
CAMS=(a b c d e f)                              # all 6 ports; a dead one just shows "no signal"
declare -A UDP=( [a]=5000 [b]=5001 [c]=5002 [d]=5003 [e]=5004 [f]=5005 )
declare -A X=( [a]=0 [b]=$CW [c]=$((CW*2)) [d]=0 [e]=$CW [f]=$((CW*2)) )
declare -A Y=( [a]=0 [b]=0 [c]=0 [d]=$CH [e]=$CH [f]=$CH )

PROPS=""; BRANCHES=""; i=0
# bottom layer: a labelled placeholder tile per cell (so empty/dead cells are clearly marked)
for p in "${PORTS_ALL[@]}"; do
  PROPS="$PROPS sink_${i}::xpos=${X[$p]} sink_${i}::ypos=${Y[$p]} sink_${i}::width=$CW sink_${i}::height=$CH"
  BRANCHES="$BRANCHES videotestsrc is-live=true pattern=2 ! video/x-raw,width=$CW,height=$CH,framerate=30/1 !"
  BRANCHES="$BRANCHES textoverlay text=\"port $p (no signal)\" valignment=center halignment=center font-desc=\"Sans 13\" ! comp.sink_${i}"
  i=$((i+1))
done
# top layer: the live cameras, overlaid on their cell (covers the placeholder when streaming)
for p in "${CAMS[@]}"; do
  PROPS="$PROPS sink_${i}::xpos=${X[$p]} sink_${i}::ypos=${Y[$p]} sink_${i}::width=$CW sink_${i}::height=$CH"
  BRANCHES="$BRANCHES udpsrc port=${UDP[$p]} caps=$CAPS ! rtpjitterbuffer latency=100 ! rtph264depay ! avdec_h264 !"
  BRANCHES="$BRANCHES videoflip method=rotate-180 ! videoconvert ! videoscale ! video/x-raw,width=$CW,height=$CH !"
  BRANCHES="$BRANCHES textoverlay text=\"port $p\" valignment=top halignment=left font-desc=\"Sans Bold 16\" shaded-background=true ! comp.sink_${i}"
  i=$((i+1))
done

SINK="${SINK:-autovideosink sync=false}"          # SINK=fakesink for a headless link/decode check
CMD="gst-launch-1.0 -e compositor name=comp ignore-inactive-pads=true $PROPS ! \
  video/x-raw,width=$CANVAS_W,height=$CANVAS_H ! videoconvert ! $SINK \
  $BRANCHES"

echo "2x3 port grid (a..f); port b + any dead camera show 'no signal'. Ctrl-C to stop."
eval exec "$CMD"
