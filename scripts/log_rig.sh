#!/usr/bin/env bash
# ONE RIG RECORDING: the 4 synchronised cameras AND the IMU, both straight to disk, into
# one directory. Runs ON THE TX2 HOST (it drives docker), not inside a container.
#
#   MOTION_SECONDS=90 LOG_LABEL=run1 ./scripts/log_rig.sh
#   LOG_DIRS="/logs,/logs,/sdlog,/sdlog" MOTION_SECONDS=180 ./scripts/log_rig.sh
#
# TWO CONTAINERS, AND NOTHING PASSES BETWEEN THEM. The IMU needs root + privileged for
# /dev/spidev1.0, /dev/gpiochip* and SCHED_FIFO; the image logger must not hold those. They
# cannot share a container, and they do not need to: each writes its own file. Both stamp
# CLOCK_MONOTONIC, so they are joined offline on the timestamp - no DDS, no discovery.
#
# THE IMU BRACKETS THE CAMERAS, deliberately: started first and stopped last, so every
# camera frame has IMU samples on both sides of it to interpolate between. A frame outside
# the IMU span cannot be used by a VIO at all.
#
# delta_camera_imu IS STILL UNMEASURED (docs/timestamps.md). Both streams are on one clock, so they
# are comparable, but the fixed offset between the exposure midpoint and the IMU data-ready
# edge has never been solved for on this rig. Until a Kalibr cam-IMU run says otherwise,
# treat this as data to CALIBRATE from, not as a calibrated dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

TRIG_PORT="${TRIG_PORT:-/dev/ttyTHS1}"
TRIGCTL="${TRIGCTL:-/home/nvidia/j106-trigctl.py}"
SECS="${MOTION_SECONDS:-90}"
LABEL="${LOG_LABEL:-rig}"
HOST_LOGS="${HOST_LOGS:-/home/nvidia/logs}"     # what /logs is bound to on the host
fail() { echo "REFUSING: $*" >&2; exit 1; }

# PREFLIGHT GATES RATHER THAN WARNS, because both of these reset on a POWER CYCLE and
# neither announces itself. With trigger_mode=0 the cameras free-run while the generator
# keeps pulsing happily; with the wrong polarity every image is 3-4x darker. Both were wrong
# after the 2026-09-03 battery repower, and a run that starts wrong is expensive to diagnose
# afterwards - the images look plausible either way.
trig=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo missing)
[ "$trig" = "1" ] || fail "trigger_mode=$trig, expected 1.
  echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode"

status=$(timeout 20 python3 "$TRIGCTL" --port "$TRIG_PORT" status 2>/dev/null) \
  || fail "cannot read the trigger generator on $TRIG_PORT. exposure_us would be a guess,
  and the stamp is SOF - exposure/2, so a wrong value silently biases every timestamp."
echo "$status" | grep -q "running=1" || fail "the trigger generator is not running."
echo "$status" | grep -q "polarity=active_low" || fail "polarity is not active_low.
  python3 $TRIGCTL --port $TRIG_PORT raw 'pol 0'"

PULSE_NS=$(echo "$status" | sed -n 's/^ch1_exposure_us=[0-9]* pulse_ns=\([0-9]*\).*/\1/p')
[ -n "$PULSE_NS" ] || fail "could not parse pulse_ns from the generator status"
EXPOSURE_US=$(( (PULSE_NS + 500) / 1000 ))
FPS_MILLI=$(echo "$status" | sed -n 's/^fps_milli=\([0-9]*\)$/\1/p')
[ -n "$FPS_MILLI" ] || fail "could not parse fps_milli from the generator status"
TRIGGER_FPS=$(( FPS_MILLI / 1000 ))
[ "$TRIGGER_FPS" -gt 0 ] || fail "generator reports ${TRIGGER_FPS} fps"

STAMP=$(date +%Y%m%d_%H%M%S)
DIR="imglog_${LABEL}_${STAMP}"
IMU_NAME="bev_imulog_${STAMP}"
RANGE_NAME="bev_rangelog_${STAMP}"

echo "preflight OK  trigger_mode=1  active_low  ${TRIGGER_FPS} fps  exposure_us=${EXPOSURE_US}"
echo "output ${HOST_LOGS}/${DIR}  (camN.raw + camN.csv + imu0.csv + range0.csv)"
mkdir -p "${HOST_LOGS}/${DIR}"

# The IMU container is started detached and stopped by NAME. `docker stop` sends the
# compose stop_signal (SIGINT), which is what makes imu_node close its CSV - a killed node
# leaves the last line truncated, which a parser reports as corruption.
cleanup() { docker stop -t 20 "$RANGE_NAME" >/dev/null 2>&1 || true
            docker rm -f "$RANGE_NAME"      >/dev/null 2>&1 || true
            docker stop -t 20 "$IMU_NAME"   >/dev/null 2>&1 || true
            docker rm -f "$IMU_NAME"        >/dev/null 2>&1 || true; }
trap cleanup EXIT

# THE RANGEFINDER IS OPTIONAL AND MUST NEVER FAIL THE RUN, so this does not gate on it the
# way the IMU does below. It also has to start HERE and not earlier: it takes sole ownership
# of /dev/ttyTHS1, which the preflight above was still using for the generator status. From
# this point until cleanup nothing else may open that port - the range stream and the
# trigger console share it, and two readers steal each other's bytes.
echo "starting the range logger (sole owner of $TRIG_PORT from here)"
RANGE_CSV="/logs/${DIR}/range0.csv" TRIG_PORT="$TRIG_PORT" RANGE_DIV="${RANGE_DIV:-15}" \
  docker compose run -d --name "$RANGE_NAME" rangelog >/dev/null 2>&1 || true
sleep 3
if docker ps --format '{{.Names}}' | grep -qx "$RANGE_NAME"; then
  echo "range logger up (1 reading / ${RANGE_DIV:-15} pulses)"
else
  echo "NOTE: the range logger did not start - recording without range." >&2
  docker logs "$RANGE_NAME" 2>&1 | tail -5 >&2 || true
fi

echo "starting the IMU first, so it brackets the cameras"
EXPOSURE_US="$EXPOSURE_US" IMU_CSV="/logs/${DIR}/imu0.csv" IMU_RATE="${IMU_RATE:-200}" \
  docker compose run -d --name "$IMU_NAME" imulog >/dev/null
sleep 4
docker ps --format '{{.Names}}' | grep -qx "$IMU_NAME" \
  || fail "the IMU container exited immediately. Its log:
$(docker logs "$IMU_NAME" 2>&1 | tail -20)"
echo "IMU up"

EXPOSURE_US="$EXPOSURE_US" TRIGGER_FPS="$TRIGGER_FPS" \
  LOG_STAMP="$STAMP" LOG_LABEL="$LABEL" \
  LOG_DIR="${LOG_DIR:-/logs}" LOG_DIRS="${LOG_DIRS:-}" \
  IMAGE_LOG_DIRECT="${IMAGE_LOG_DIRECT:-false}" \
  MOTION_SECONDS="$SECS" \
  docker compose run --rm -T logonly

echo "cameras done - stopping the IMU and range logger last"
cleanup
trap - EXIT

echo
echo "=== ${HOST_LOGS}/${DIR} ==="
ls -la "${HOST_LOGS}/${DIR}"
if [ -s "${HOST_LOGS}/${DIR}/imu0.csv" ]; then
  echo "imu0.csv: $(grep -vc '^#' "${HOST_LOGS}/${DIR}/imu0.csv") samples"
else
  echo "WARNING: imu0.csv is empty - the IMU produced nothing this run." >&2
fi
# A range channel that produced nothing is a NOTE, not a WARNING: it is the one OPTIONAL
# stream here and the recording is complete without it. Join it to the frames on
# `pulses` <-> `seq`, never on the timestamp - see the header inside the file.
if [ -f "${HOST_LOGS}/${DIR}/range0.csv" ]; then
  nrange=$(grep -vc '^#' "${HOST_LOGS}/${DIR}/range0.csv" 2>/dev/null || echo 0)
  if [ "${nrange:-0}" -gt 0 ]; then
    echo "range0.csv: ${nrange} readings"
  else
    echo "NOTE: range0.csv has no readings - rangefinder absent this run."
  fi
else
  echo "NOTE: no range0.csv - the range logger did not run."
fi
