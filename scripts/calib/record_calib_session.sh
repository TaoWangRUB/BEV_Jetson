#!/usr/bin/env bash
# Record ONE calibration session for the quarterKalibr pipeline. Runs on the TX2.
#
# The session is a performance: eight stages, in ring order, recovered afterwards from
# which cameras could see the AprilGrid (scripts/calib/extract_quarterkalibr_bags.py).
# This script prompts them so they are not tracked in your head, and refuses to start
# if the rig is in a state that would silently produce an unusable recording.
#
# THE PREFLIGHT IS THE POINT. A board reboot silently resets trigger_mode to 0: the
# STM32 keeps pulsing, the cameras ignore it and free-run, nothing logs an error, and
# the only symptoms are AE gain-hunting and frames that are not a set. That has already
# happened once mid-session here, so it is checked rather than assumed.
#
#   ./scripts/calib/record_calib_session.sh [output_dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT="${1:-bags/calib_$(date +%Y%m%d_%H%M%S)}"
EXPOSURE_US="${EXPOSURE_US:-}"        # default: read from the trigger generator
EVERY_N="${EVERY_N:-8}"               # 30 Hz / 8 = 3.75 Hz images; solvers want ~4 Hz
TRIGCTL="${TRIGCTL:-/home/nvidia/tools/j106-trigctl.py}"
TRIG_PORT="${TRIG_PORT:-/dev/ttyACM0}"
# F401: reachable on BOTH /dev/ttyACM0 (USB CDC) and /dev/ttyTHS1 (UART) - verified 2026-09-01, same generator on both. ACM0 is the default only for consistency.
SUDO="echo nvidia | sudo -S"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------
say "Preflight"

MODE=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo missing)
if [ "$MODE" != "1" ]; then
  echo "  trigger_mode is '$MODE' — the cameras are NOT following the trigger."
  read -r -p "  set it to 1 now? [Y/n] " a
  [ "${a:-y}" = "n" ] && fail "cannot record a synchronised session with the trigger off"
  eval "$SUDO bash -c 'echo 1 > /sys/module/imx296/parameters/trigger_mode'" 2>/dev/null
  [ "$(cat /sys/module/imx296/parameters/trigger_mode)" = "1" ] || fail "could not set trigger_mode"
fi
echo "  trigger_mode = 1"

# The generator is on the board's UART, not USB CDC. Its settings do NOT survive an MCU
# reset (it boots at compiled-in defaults), so read them rather than assuming.
STATUS=$(eval "$SUDO python3 $TRIGCTL --port $TRIG_PORT status" 2>/dev/null | grep -v '^\[sudo') \
  || fail "cannot talk to the trigger generator on $TRIG_PORT"
grep -q "running=1" <<<"$STATUS" || fail "trigger generator is not running (start it with j106-trigctl.py start)"
PERIOD_US=$(grep -oP 'period_us=\K[0-9]+' <<<"$STATUS")
PULSE_NS=$(grep -oP 'ch1_exposure_us=[0-9]+ pulse_ns=\K[0-9]+' <<<"$STATUS")
[ -n "$EXPOSURE_US" ] || EXPOSURE_US=$(( (PULSE_NS + 500) / 1000 ))
echo "  trigger running: period ${PERIOD_US} us, pulse ${PULSE_NS} ns -> exposure_us=${EXPOSURE_US}"
# Per-channel exposures would break the one-exposure-per-rig assumption in the capture node.
if [ "$(grep -oP 'ch._exposure_us=\K[0-9]+' <<<"$STATUS" | sort -u | wc -l)" != "1" ]; then
  fail "the four trigger channels have DIFFERENT exposures; the rig-wide exposure assumption no longer holds"
fi

MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
if [ "$MIN" != "$MAX" ]; then
  echo "  jetson_clocks not applied (cpu0 ${MIN}..${MAX}) — three concurrent streams collapse without it"
  eval "$SUDO /usr/bin/jetson_clocks" 2>/dev/null && echo "  applied"
fi

AVAIL=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$AVAIL" -ge 10 ] || fail "only ${AVAIL}G free; a session is several GB"
echo "  ${AVAIL}G free"

# NTP slews CLOCK_MONOTONIC and its servo makes the frame-time fit residual wander
# (30.9 us with timesyncd running vs 8.4 us without). Stop it for the recording only.
TIMESYNC_WAS=$(systemctl is-active systemd-timesyncd 2>/dev/null || echo inactive)
if [ "$TIMESYNC_WAS" = "active" ]; then
  eval "$SUDO systemctl stop systemd-timesyncd" 2>/dev/null
  echo "  stopped systemd-timesyncd (restored on exit)"
fi
restore() {
  [ "$TIMESYNC_WAS" = "active" ] && eval "$SUDO systemctl start systemd-timesyncd" 2>/dev/null || true
  docker rm -f calibrec >/dev/null 2>&1 || true
}
trap restore EXIT

mkdir -p "$OUT"

# ---- provenance ------------------------------------------------------------
# A recording that does not state the clock, the trigger and the offsets in force
# cannot be interpreted later — and "later" is when someone asks why the scale is off.
cat > "$OUT/meta.json" <<JSON
{
  "recorded": "$(date -Is)",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "clock": "CLOCK_MONOTONIC",
  "camera_stamp": "exposure midpoint (SOF - exposure/2), see docs/timestamps.md",
  "imu_stamp": "data-ready edge, CLOCK_MONOTONIC",
  "trigger": {"period_us": $PERIOD_US, "pulse_ns": $PULSE_NS, "exposure_us": $EXPOSURE_US,
              "source": "j106-trigctl.py status on $TRIG_PORT"},
  "images": {"resolution": "1456x1088", "published_every_n": $EVERY_N,
             "rate_hz": $(awk "BEGIN{printf \"%.2f\", 1000000/$PERIOD_US/$EVERY_N}")},
  "rig_layout": "config/rig/rig_layout.yaml",
  "target": "config/calib/april_6x6.yaml",
  "imu_noise": "config/calib/imu_mpu9250.yaml",
  "delta_camera_imu": "UNMEASURED",
  "timesyncd_stopped": $( [ "$TIMESYNC_WAS" = "active" ] && echo true || echo false )
}
JSON

# ---- start capture + imu + recorder ----------------------------------------
say "Starting capture, IMU and recorder"
TOPICS="/cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw"
TOPICS="$TOPICS /cam1/frame_meta /cam2/frame_meta /cam3/frame_meta /cam4/frame_meta /imu0"

docker run -d --name calibrec --runtime nvidia --network host --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev \
  -v "$PWD":/workspace -w /workspace cuvslam-foxy:tx2 bash -lc "
    source install/setup.bash
    ros2 run bev_camera argus_capture_node --ros-args \
      -p exposure_us:=$EXPOSURE_US -p publish_every_n:=$EVERY_N \
      -p frame_log_dir:=/workspace/$OUT &
    ros2 run bev_imu imu_node --ros-args -p csv:=/workspace/$OUT/imu0.csv &
    sleep 8
    exec ros2 bag record -o /workspace/$OUT/bag $TOPICS
  " >/dev/null
sleep 12
docker ps --filter name=calibrec --format '{{.Status}}' | grep -q Up || {
  docker logs calibrec 2>&1 | tail -20; fail "the recorder did not come up"; }
echo "  recording to $OUT"

# ---- the eight stages ------------------------------------------------------
# Ring order (config/rig/rig_layout.yaml): c=front-left, d=front-right, e=back-left,
# f=back-right, so neighbours are c -> d -> f -> e. cam2 and cam3 are DIAGONAL.
stage() {
  printf '\n  \033[1mStage %s/9: show the target to %s\033[0m\n' "$1" "$2"
  printf '    %s\n' "$3"
  read -r -p "    press ENTER when that stage is done "
}
say "Nine stages. Fill the frame, move the target around within it, and keep it OUT of
the other cameras' view during the single-camera stages.

WORK THE PERIPHERY. The 2026-08-28 session left 13-21 of 64 cells per camera short of
quota, all peripheral - which is exactly where a >180 degree lens is unconstrained and
where a calibration that never saw the target is confidently wrong. You do NOT need the
whole board in frame: Kalibr accepts an observation at 7 tags, so push the grid into the
corners where only a third of it is visible. Tilt 30-45 degrees in BOTH axes, and pause
at each pose."
stage 1 "cam1 ONLY (port c, front-left)"   "intrinsics for the front-left camera"
stage 2 "cam2 ONLY (port d, front-right)"  "intrinsics for the front-right camera"
stage 3 "cam4 ONLY (port f, back-right)"   "intrinsics for the back-right camera"
stage 4 "cam3 ONLY (port e, back-left)"    "intrinsics for the back-left camera"
stage 5 "cam3 AND cam1 (back-left + front-left)"   "left-side overlap"
stage 6 "cam1 AND cam2 (front-left + front-right)" "front overlap"
stage 7 "cam2 AND cam4 (front-right + back-right)" "right-side overlap"
stage 8 "cam4 AND cam3 (back-right + back-left)"   "rear overlap"
stage 9 "THREE OR MORE cameras at once"           "board at a rig corner, ~0.8-1.2 m, so two adjacent pairs and their shared camera all see it"

# Stage 9 exists because the four pairwise stages cannot see their own error. They closed
# the ring to 3.63 deg / 9.2 mm, and a Monte-Carlo over each pair's reported spread says
# random error would leave only ~0.5 deg - so most of it is SYSTEMATIC bias inside each
# recording, which is invisible while every constraint is pairwise and gets absorbed
# silently by the ring closure. Three cameras on one board at one instant makes it
# observable instead.

say "IMU excitation — hold the target in view of any camera and rotate the rig about
all three axes, then translate along all three. Kalibr needs the target visible while
the IMU is excited; this is what makes Delta observable."
read -r -p "  press ENTER when done "

say "Stopping"
docker stop --signal SIGINT --time 20 calibrec >/dev/null
sleep 2
du -sh "$OUT" 2>/dev/null
say "Recorded to $OUT
Next: convert to ROS1 and split into stages —
  rosbags-convert $OUT/bag --dst $OUT/session.bag
  python3 scripts/calib/extract_quarterkalibr_bags.py --bag $OUT/session.bag --out $OUT/stages"
