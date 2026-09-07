#!/usr/bin/env bash
# Bracket the rig-wide gain and report what each setting does to the histogram.
#
#   ./scripts/calib/exposure_bracket.sh                       # 16 8 4 2, 4 s each
#   GAINS="16 8 4" SECONDS_EACH=6 ./scripts/calib/exposure_bracket.sh
#
# Run it TWICE, standing still: once in the BRIGHTEST part of the route and once in the
# DIMMEST. The answer is the highest gain whose bright-room saturation is near zero and
# whose dim-room local contrast is still there - one number for the whole route, because
# the exposure is one trigger pulse for all four cameras and the gain must match it.
#
# WHY GAIN AND NOT THE PULSE WIDTH. Under external trigger the exposure IS the pulse
# width, so Argus AE cannot reach its main actuator and hunts on gain; that is why gain
# was clamped to 16x analog x 4x digital and left there (4.7). 64x is most of why a sunlit
# room clips. Gain needs no MCU command and - unlike the pulse width - does not move the
# exposure-midpoint timestamp, so nothing downstream has to re-read exposure_us.
#
# BIAS DARK. cuVSLAM tracks gradients: an underexposed frame with 20 levels of local
# contrast still tracks, a clipped one has exactly zero. On run1 the dim room gave the MOST
# features (2500-3400/frame) and 68% saturation still gave 1887, while 89% gave none. The
# dark floor is UNMEASURED though - that log never went below mean ~100 - so this reports
# local contrast at each step, which is where the floor will show itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."/..
GAINS="${GAINS:-16 8 4 2}"
SECS="${SECONDS_EACH:-4}"
DGAIN="${AE_DGAIN:-1.0}"        # digital gain adds noise, not signal: pin it low and move analog
OUT="${OUT:-/tmp/exposure_bracket_$(date +%H%M%S)}"
TX2="${TX2:-tx2-eth}"
BEVDIR="${BEVDIR:-/media/nvidia/workspace/BEV_Jetson}"

echo "bracket: gains [$GAINS] x ${SECS}s, dgain ${DGAIN}, on $TX2"
echo "STAND STILL and keep the rig pointed at the scene you are characterising."
EXPO=$(ssh "$TX2" "python3 /home/nvidia/j106-trigctl.py --port /dev/ttyTHS1 status" \
       | sed -n 's/.*ch1_exposure_us=\([0-9]*\).*/\1/p')
[ -n "$EXPO" ] || { echo "REFUSING: could not read the pulse width from the generator" >&2; exit 1; }
echo "trigger pulse width: ${EXPO} us"

for g in $GAINS; do
  echo; echo "=== analog gain ${g}x (total $(echo "$g*$DGAIN" | bc)x) ==="
  # Restart the Argus daemon between captures. A session leaked by the previous run fails
  # the next one with "no session for 0" - the capture starts, announces its log directory,
  # and produces nothing, so it reads as a hardware fault (5.7). log_rig.sh does this in
  # preflight for the same reason; a bracket runs back-to-back captures and needs it more.
  ssh "$TX2" "sudo systemctl restart nvargus-daemon" >/dev/null 2>&1 || true
  sleep 3
  # LOG_DIR is a path INSIDE the container: /logs is the bind mount for the host's
  # /home/nvidia/logs. Passing the host path instead gives a container-local directory
  # that vanishes with the container - the capture runs, reports success, streams all four
  # cameras, and leaves nothing behind.
  ssh "$TX2" "cd $BEVDIR && AE_GAIN=$g AE_DGAIN=$DGAIN EXPOSURE_US=$EXPO \
      LOG_DIR=/logs LOG_LABEL=gain${g} MOTION_SECONDS=$SECS \
      docker compose run --rm logonly" >/dev/null 2>&1 || {
    echo "  capture FAILED at gain ${g}"; continue; }
  d=$(ssh "$TX2" "ls -1dt /home/nvidia/logs/imglog_gain${g}_* 2>/dev/null | head -1")
  [ -n "$d" ] || { echo "  no log directory for gain ${g}"; continue; }
  mkdir -p "$OUT"; rsync -a --info=none "$TX2:$d" "$OUT/" 2>/dev/null || scp -qr "$TX2:$d" "$OUT/"
  python3 scripts/calib/exposure_report.py "$OUT/$(basename "$d")" --label "gain ${g}x"
done
echo; echo "logs under $OUT"
