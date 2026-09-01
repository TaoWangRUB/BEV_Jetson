#!/bin/bash
# Section 5 motion test, board side. Run ON the TX2, from the repo root.
#
#   ./scripts/vo/run_motion_test.sh <label> <tape_measure_metres> [--record-images]
#
# PREFLIGHT IS NOT OPTIONAL, AND IT GATES RATHER THAN WARNS. trigger_mode does not persist
# across a reboot and forgetting it is SILENT: the STM32 keeps pulsing, the cameras ignore
# it and free-run, nothing logs an error, and the only symptoms are AE gain-hunting and
# sets that are not sets. That cost a full run on 2026-09-01. The same day, the VO rejected
# every set at one frame period twice over while the trigger was perfect - so a run that
# starts wrong is expensive to diagnose afterwards. Refuse instead.
#
# --record-images ALSO bags the four camera streams, which is what makes a run replayable:
# move the rig once, then re-run the VO against it as often as needed without being at the
# rig. Cost is bandwidth (~95 MB/s at 15 Hz), so it decimates and expects short runs.
#
# DO TWO PASSES, and do not merge them:
#   1. --record-images, for a replayable bag
#   2. without it, for the live 5.1/5.2 numbers
# The recorder competes for CPU and I/O and can induce drops of its own; a tape-measure
# number taken with it running cannot be told apart from the rig misbehaving.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # repo root = compose file location

LABEL=${1:?usage: run_motion_test.sh <label> <tape_metres> [--record-images]}
TAPE=${2:?give the tape-measured distance in metres, even for a return-to-origin run}
RECORD_IMAGES=0
[ "${3:-}" = "--record-images" ] && RECORD_IMAGES=1

TRIG_PORT="${TRIG_PORT:-/dev/ttyACM0}"
# F401: reachable on BOTH /dev/ttyACM0 (USB CDC) and /dev/ttyTHS1 (UART) - verified 2026-09-01, same generator on both. ACM0 is the default only for consistency.
TRIGCTL="${TRIGCTL:-/home/nvidia/tools/j106-trigctl.py}"
fail() { echo "REFUSING: $*" >&2; exit 1; }

# 1. the driver must be following the trigger
trig=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo missing)
[ "$trig" = "1" ] || fail "trigger_mode=$trig, expected 1.
  sudo sh -c 'echo 1 > /sys/module/imx296/parameters/trigger_mode'"

# 2. the generator must be running, ACTIVE_LOW, and tell us its real pulse width. Polarity
#    resets to the compiled-in default on a POWER CYCLE (it survives a warm reboot), and the
#    wrong polarity does not fail - it just makes every image 3-4x darker.
status=$(python3 "$TRIGCTL" --port "$TRIG_PORT" status 2>/dev/null) \
  || fail "cannot read the trigger generator on $TRIG_PORT. exposure_us would be a guess,
  and the stamp is SOF - exposure/2, so a wrong value silently biases every timestamp."
echo "$status" | grep -q "running=1" || fail "the trigger generator is not running."
echo "$status" | grep -q "polarity=active_low" || fail "polarity is not active_low.
  python3 $TRIGCTL --port $TRIG_PORT raw 'pol 0'"
PULSE_NS=$(echo "$status" | sed -n 's/^ch1_exposure_us=[0-9]* pulse_ns=\([0-9]*\).*/\1/p')
[ -n "$PULSE_NS" ] || fail "could not parse pulse_ns from the generator status"
EXPOSURE_US=$(( (PULSE_NS + 500) / 1000 ))

# 3. clocks, and a clean Argus - a SIGKILLed previous run leaks a session and the next
#    start dies with "Argus setup failed"
grep -q 2035200 /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null \
  || { echo "note: jetson_clocks not applied, re-applying"; sudo jetson_clocks; }
sudo systemctl restart nvargus-daemon; sleep 8   # 4 s was not enough: "no session for 0"

OUT=/media/nvidia/workspace/motion_${LABEL}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"; echo "$TAPE" > "$OUT/tape_metres.txt"
echo "$status" > "$OUT/trigger_status.txt"
{ echo "label=$LABEL"; echo "tape_m=$TAPE"; echo "exposure_us=$EXPOSURE_US";
  echo "record_images=$RECORD_IMAGES"; echo "publish_every_n=${PUBLISH_EVERY_N:-1}";
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)";
  echo "calib_session=$(sed -n 's/^calib_session: //p' config/rig/rig_extrinsics_imx296.yaml)"; } > "$OUT/run_meta.txt"

echo "preflight OK  trigger_mode=1  active_low  exposure_us=$EXPOSURE_US"
echo "output: $OUT"
echo
echo "  5.1 straight line: move exactly $TAPE m and STOP"
echo "  5.2 return-to-origin: go out and come back to the SAME pose"
[ "$RECORD_IMAGES" = 1 ] && echo "  recording IMAGES too - keep this run short (~30 s)"
echo

MOTION_LABEL="$LABEL" RECORD_IMAGES="$RECORD_IMAGES" EXPOSURE_US="$EXPOSURE_US" \
  PUBLISH_EVERY_N="${PUBLISH_EVERY_N:-1}" \
  docker compose run --rm motion 2>&1 | tee "$OUT/vo.log" || true

echo; echo "=== sets / skew / drops, from the node's own reporting ==="
grep -E "sets |dropped|remap|tracking lost" "$OUT/vo.log" | tail -12
mv bags/motion_${LABEL}_* "$OUT/" 2>/dev/null || true
echo
echo "bag under $OUT - copy the whole directory to the host and run:"
echo "  python3 scripts/vo/analyze_motion.py $OUT"
