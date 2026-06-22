#!/usr/bin/env bash
# Receive the 4 CSI H.264 / RTP / UDP streams from the TX2 (scripts/stream/csi_sender.sh) and show
# them in a labelled 2x2 mosaic.  RUN ON THE HOST.
#
#   ./csi_receiver.sh
#
# Ports/labels match csi_sender.sh:
#   5001 = cam1 (port c)   5002 = cam2 (port d)
#   5003 = cam3 (port e)   5004 = cam4 (port f)
# Nothing shows for a stream whose camera is dark/dead on the TX2 -> that's your "which camera is
# broken" check. Ctrl-C to stop. (Needs gstreamer1.0 + gst-libav for avdec_h264.)

W=640; H=480
CAPS="application/x-rtp,media=video,encoding-name=H264,payload=96"

rx() {  # $1=udp port  $2=label  $3=compositor sink index
  echo "udpsrc port=$1 caps=$CAPS ! rtpjitterbuffer latency=100 ! rtph264depay ! avdec_h264 !"\
       "videoconvert ! videoscale ! video/x-raw,width=$W,height=$H !"\
       "textoverlay text=\"$2\" valignment=top halignment=left font-desc=\"Sans Bold 18\" shaded-background=true !"\
       "comp.sink_$3"
}

CMD="gst-launch-1.0 -e compositor name=comp \
  sink_0::xpos=0   sink_0::ypos=0 \
  sink_1::xpos=$W  sink_1::ypos=0 \
  sink_2::xpos=0   sink_2::ypos=$H \
  sink_3::xpos=$W  sink_3::ypos=$H ! \
  video/x-raw,width=$((W*2)),height=$((H*2)) ! videoconvert ! autovideosink sync=false \
  $(rx 5001 'cam1 port c' 0) \
  $(rx 5002 'cam2 port d' 1) \
  $(rx 5003 'cam3 port e' 2) \
  $(rx 5004 'cam4 port f' 3)"

echo "receiving udp 5001-5004 -> 2x2 mosaic (Ctrl-C to stop)"
eval exec "$CMD"
