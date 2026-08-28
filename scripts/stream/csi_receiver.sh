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

CW=480; CH=360                                 # per-cell size
CAPS="application/x-rtp,media=video,encoding-name=H264,payload=96"
declare -A UDP=( [a]=5000 [b]=5001 [c]=5002 [d]=5003 [e]=5004 [f]=5005 )

# Which cells to show. Default is all six (3x2, the original layout). Set PORTS
# to just the populated ports to drop the empty cells, e.g. on the 4x IMX296 rig:
#   PORTS="c d e f" ./csi_receiver.sh      -> 2x2, no wasted space
PORTS_ALL=( ${PORTS:-a b c d e f} )
CAMS=( "${PORTS_ALL[@]}" )
N=${#PORTS_ALL[@]}
if   [ "$N" -le 2 ]; then COLS=$N
elif [ "$N" -le 4 ]; then COLS=2
else                     COLS=3
fi
ROWS=$(( (N + COLS - 1) / COLS ))
CANVAS_W=$((CW*COLS)); CANVAS_H=$((CH*ROWS))

declare -A X Y
_i=0
for p in "${PORTS_ALL[@]}"; do
  X[$p]=$(( (_i % COLS) * CW ))
  Y[$p]=$(( (_i / COLS) * CH ))
  _i=$((_i+1))
done

# csi_sender.sh already rotates 180 in HARDWARE on the board's ISP (its FLIP=2),
# so rotating again here would put the image back upside down - and cost host CPU
# per camera doing it. Override only if the sender runs with FLIP=0.
RXFLIP="${RXFLIP:-none}"

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
  BRANCHES="$BRANCHES videoflip method=$RXFLIP ! videoconvert ! videoscale ! video/x-raw,width=$CW,height=$CH !"
  BRANCHES="$BRANCHES textoverlay text=\"port $p\" valignment=top halignment=left font-desc=\"Sans Bold 16\" shaded-background=true ! comp.sink_${i}"
  i=$((i+1))
done

SINK="${SINK:-autovideosink sync=false}"          # SINK=fakesink for a headless link/decode check
CMD="gst-launch-1.0 -e compositor name=comp ignore-inactive-pads=true $PROPS ! \
  video/x-raw,width=$CANVAS_W,height=$CANVAS_H ! videoconvert ! $SINK \
  $BRANCHES"

echo "${COLS}x${ROWS} port grid (${PORTS_ALL[*]}); any dead camera shows 'no signal'. Ctrl-C to stop."
eval exec "$CMD"
