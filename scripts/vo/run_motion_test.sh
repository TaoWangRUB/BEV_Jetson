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
# DEFAULT IS THE FUSED ZERO-COPY NODE, and that is the one to use for section 5.
#
# 5.1, 5.2 and 5.4 all measure ODOMETRY - translation against a tape measure, drift on
# return to origin, rate. None of them needs images. The fused node reads Argus straight
# into CUDA and publishes no images at all, so recording it costs 4.5 MB a minute instead
# of ~95 MB/s, and - the real point - the pipeline being MEASURED is the pipeline that
# would be DEPLOYED. Recording the modular path instead would validate something we do not
# intend to ship.
#
# --record-images switches to the MODULAR node and additionally bags the four camera
# streams. That is a convenience for offline tuning (move the rig once, re-run the VO
# against it as often as needed), NOT the measurement:
#   - it is a different pipeline: images over DDS, CPU remap, ~95 MB/s, so decimate and
#     keep runs short. The SD cannot absorb four cameras at 30 Hz.
#   - a bag can only ever be replayed through the modular node. The fused node consumes
#     Argus directly and cannot replay one, by construction.
#   - the recorder competes for CPU and I/O and can induce drops of its own, which on
#     replay are indistinguishable from the rig misbehaving.
# So if you want both, do two separate passes and do not merge them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # repo root = compose file location

# MOTION_SECONDS=<n> records for exactly n seconds and stops itself cleanly, which is what
# you want for a repeatable run and what makes the pipeline testable without a terminal.
# Leave it unset for an open-ended run stopped with Ctrl-C.
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

# STOP THIS RUN WITH Ctrl-C, AND ONLY Ctrl-C.
#
# rosbag2 writes metadata.yaml when it shuts down cleanly, and WITHOUT that file the bag is
# unreadable - rosbags refuses it outright, and the .db3 itself comes back with 0 messages
# because the last transaction never committed. `docker stop` (SIGTERM) is not enough; the
# signal has to reach rosbag2 as SIGINT. Verified the hard way on 2026-09-01: a run stopped
# with docker stop produced a directory that looked complete and contained nothing.
#
# Forward SIGINT to the compose child so Ctrl-C here reaches the recorder in the container.
# Deliver the signal to the CONTAINER by name, not to `docker compose run`. Compose only
# forwards SIGINT when it has a TTY, so a script-driven stop (or any non-interactive run)
# leaves the container going with the recorder holding an open bag. Naming it and using
# `docker kill -s INT` works the same way whether or not anyone is at a terminal.
CNAME="bev_motion_$$"
forward_int() {
  docker kill -s INT "$CNAME" >/dev/null 2>&1 || true
  # AND WAIT FOR IT TO ACTUALLY EXIT. bash's `wait` returns as soon as a trapped signal
  # arrives, even though the child is still running - so without this the script raced on
  # to move the bag while rosbag2 was still writing it, producing exactly the unreadable
  # -wal/-shm directory this whole trap exists to prevent.
  docker wait "$CNAME" >/dev/null 2>&1 || true
}
trap forward_int INT TERM

echo "preflight OK  trigger_mode=1  active_low  exposure_us=$EXPOSURE_US"
echo "output: $OUT"
echo
[ -n "${MOTION_SECONDS:-}" ] && echo "  recording for ${MOTION_SECONDS}s, then stopping itself" || echo "  stop with Ctrl-C when the move is complete"
echo "  5.1 straight line: move exactly $TAPE m and STOP"
echo "  5.2 return-to-origin: go out and come back to the SAME pose"
[ "$RECORD_IMAGES" = 1 ] && echo "  recording IMAGES too - keep this run short (~30 s)"
echo

# NOT `docker compose ... | tee ... &`: in a pipeline $! is the PID of the LAST element
# (tee), so the SIGINT trap would signal tee and leave the container running with the
# recorder holding an open bag. Redirect to the log and tail it for the live view, so
# $! is the compose process the trap needs to reach.
if [ "$RECORD_IMAGES" = 1 ]; then
  export MOTION_LABEL="$LABEL" RECORD_IMAGES=1 EXPOSURE_US MOTION_SECONDS PUBLISH_EVERY_N="${PUBLISH_EVERY_N:-1}"
  docker compose run --rm --name "$CNAME" motion > "$OUT/vo.log" 2>&1 &
else
  # Fused zero-copy + RECORD=1: bags /cuvslam/odometry and /tf from inside the same
  # container. Both are RELIABLE publishers, so no QoS override is needed here - unlike the
  # image topics, which are best_effort and silently record nothing without one.
  export RECORD=1 EXPOSURE_US MOTION_SECONDS
  docker compose run --rm --name "$CNAME" fused > "$OUT/vo.log" 2>&1 &
fi
COMPOSE_PID=$!
tail -f "$OUT/vo.log" & TAIL_PID=$!
wait "$COMPOSE_PID" || true
kill "$TAIL_PID" 2>/dev/null || true
# Belt and braces: if the container is somehow still up, do not move an open bag.
docker wait "$CNAME" >/dev/null 2>&1 || true
sudo mv bags/fused_* bags/motion_${LABEL}_* "$OUT/" 2>/dev/null || true

echo; echo "=== rate / timing / drops, from the node's own reporting ==="
grep -E "avg/|sets |dropped|tracking lost" "$OUT/vo.log" | tail -8
sudo chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true   # the container writes as root
echo
if ! ls "$OUT"/*/metadata.yaml >/dev/null 2>&1; then
  echo "WARNING: no metadata.yaml in the bag - it was not closed cleanly and is UNREADABLE."
  echo "         Stop the run with Ctrl-C, not by killing the container."
fi
echo "bag under $OUT - copy the whole directory to the host and run:"
echo "  python3 scripts/vo/analyze_motion.py $OUT"
