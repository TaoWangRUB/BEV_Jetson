#!/usr/bin/env bash
# Raw 4-camera image log. RUNS INSIDE the container (docker compose run --rm logonly).
#
# WHY RAW, AND WHY SPLIT ACROSS TARGETS. Measured on this board 2026-09-02:
#   ros2 bag record            6-7 fps   38-46 MB/s   <- the writer, not DDS (>=142 MB/s)
#   raw direct write           29.7 fps  190 MB/s     <- storage-bound, which is where it belongs
# and no single target absorbs 190 MB/s: eMMC 136, SD 62.6, RAM holds ~21 s. So for a full
# minute the cameras are SPLIT - 2 to eMMC, 1 to the SD, 1 to RAM - and each target stays
# inside its own measured limit.
#
# BUT LOWER THE TRIGGER FIRST IF 30 fps IS NOT ITSELF THE REQUIREMENT. Four cameras at
# 20 fps is ~127 MB/s, which fits eMMC's measured 136 MB/s with ALL FOUR on one target -
# no split, no SD, no RAM staging, no copy-off step. Measured over 60 s, 2026-09-03:
#   30 fps, LOG_DIRS=/logs,/logs,/sdlog,/ramlog   97.2% complete sets  28.74 Hz
#   20 fps, LOG_DIR=/logs                         99.9% complete sets  19.93 Hz
# Neither drops anything in the writer queues (max depth 1 of 64). At 20 fps what remains is
# ONE brief global stall roughly every 80 s - all four cameras miss the same 1-3 edges at
# once - plus a startup transient (see below). Measured across 5.5 min of 20 fps runs:
# 60 s had none, 90 s had one, 180 s had two. A single short run showing 100% is luck, not
# proof; quote the rate, not one run.
#
# THAT RESIDUAL IS STORAGE BANDWIDTH, and it is worth knowing before chasing it elsewhere.
# Decimating the IMAGES 10x (PUBLISH_EVERY_N=10, ~12.7 MB/s instead of 126.7) while the frame
# CSV still records every trigger edge gave ZERO stalls in 118 s. So it scales with write
# rate, not with Argus, the driver or the trigger.
#
# DO NOT REACH FOR IMAGE_LOG_DIRECT HERE. O_DIRECT removes the page cache that is smoothing
# these writes, and eMMC cannot take 126.7 MB/s without it: all four direct on eMMC pinned
# the writer queue at 64/64 and dropped 661 frames in 90 s, and on the 2-eMMC/2-SD split the
# SD pair dropped 45. The 136 MB/s eMMC figure is a BUFFERED number. Direct I/O helps at
# 30 fps where the split spreads the load; at 20 fps on one target it is strictly worse.
#
# The 30 fps figure is NOT the same kind of number, and lowering the trigger is the honest
# fix rather than a preference. Its losses are absent for the first 30 s and then continuous
# to the end, spread uniformly rather than on any period - the signature of the split
# saturating, not of a stall: by then tmpfs is holding well over a gigabyte of cam4 and the
# eMMC pair has a minute of writeback behind it. 30 fps of four cameras is 190 MB/s against
# 136+62.6 of real device bandwidth plus RAM that has to be given back afterwards. It works,
# and it degrades over the length of the run.
#
# Set the trigger with `j106-trigctl.py --port /dev/ttyTHS1 fps 20` and tell the space check
# about it with TRIGGER_FPS=20 - it cannot read the generator from inside the container.
#
#   LOG_DIRS="/logs,/logs,/sdlog,/ramlog" MOTION_SECONDS=60 log_raw.sh
#   TRIGGER_FPS=20 LOG_DIR=/logs MOTION_SECONDS=60 log_raw.sh   # all four, one target
#   LOG_DIR=/ramlog MOTION_SECONDS=10 log_raw.sh          # single target, all four
set -euo pipefail

: "${EXPOSURE_US:?set EXPOSURE_US to the measured trigger pulse width in us (j106-trigctl.py status)}"
SECS="${MOTION_SECONDS:-60}"
LABEL="${LOG_LABEL:-run}"
EVERY_N="${PUBLISH_EVERY_N:-1}"
STAMP=$(date +%Y%m%d_%H%M%S)
BYTES_PER_FRAME=$((1456 * 1088))
# The frame rate is the TRIGGER's, and this script cannot see the generator - the serial
# port is on the host, not in this container. So it has to be told. Getting it wrong only
# mis-sizes the space check, but in the direction that refuses a run that would have fit:
# at a 20 fps trigger the 30 fps default asked for 15228 MB of a 15809 MB disk to write 7.3 GB.
TRIG_FPS="${TRIGGER_FPS:-30}"
EFF_FPS=$((TRIG_FPS / EVERY_N))

# One base path per camera; a single LOG_DIR fans out to all four.
IFS=',' read -r -a BASES <<< "${LOG_DIRS:-${LOG_DIR:-/logs}}"
[ "${#BASES[@]}" -eq 1 ] && BASES=("${BASES[0]}" "${BASES[0]}" "${BASES[0]}" "${BASES[0]}")
[ "${#BASES[@]}" -eq 4 ] || { echo "give 1 or 4 paths in LOG_DIRS; got ${#BASES[@]}" >&2; exit 1; }

# Space check PER TARGET, counting how many cameras land on each. +40% because startup and
# shutdown make a run longer than MOTION_SECONDS - asking for 15 s once produced 21.5 s and
# overran a tmpfs mid-write.
DIRS=""
declare -A SEEN
for i in 0 1 2 3; do
  D="${BASES[$i]}/imglog_${LABEL}_${STAMP}"
  mkdir -p "$D"
  DIRS="${DIRS:+$DIRS,}$D"
  SEEN[${BASES[$i]}]=$(( ${SEEN[${BASES[$i]}]:-0} + 1 ))
done
for base in "${!SEEN[@]}"; do
  n=${SEEN[$base]}
  need=$(( SECS * EFF_FPS * n * BYTES_PER_FRAME * 14 / 10 / 1048576 ))
  free=$(df -Pm "$base" | awk 'NR==2{print $4}')
  rate=$(( EFF_FPS * n * BYTES_PER_FRAME / 1048576 ))
  printf "  %-28s %d cam  ~%d MB/s  needs %d MB, %d MB free\n" "$base" "$n" "$rate" "$need" "$free"
  [ "$free" -ge "$need" ] || { echo "REFUSING: $base has $free MB free, needs ~$need MB." >&2; exit 1; }
done

# Writeback settings matter for runs over ~30 s and are NOT the container's to set - they
# live on the host, in /etc/sysctl.d/60-bev-writeback.conf. Measured over 60 s: 97.4%
# complete 4-camera sets with continuous writeback against 95.8% on the kernel defaults,
# because the default 785 MB dirty threshold lets ~4 s of log accumulate and then flushes it
# in a burst that stalls the capture thread. Warn rather than fail: a short run is unaffected.
if [ "$SECS" -gt 30 ] && [ "$(cat /proc/sys/vm/dirty_bytes 2>/dev/null || echo 0)" = "0" ]; then
  echo "NOTE: host writeback is at kernel defaults; runs over ~30 s lose noticeably more"
  echo "      complete sets. See /etc/sysctl.d/60-bev-writeback.conf on the board."
fi

# CAP THE QUEUE BY MEMORY, NOT BY FRAME COUNT.
#
# The depth is per camera, so the real cost is depth x 4 x 1.584 MB. WRITE_QUEUE_DEPTH=600
# asked for 3.8 GB of buffers on a 7.7 GB board and took it down hard - unreachable on both
# interfaces, needing a power cycle. Under O_DIRECT the queues actually fill (there is no
# page cache absorbing the writes), so the worst case is not hypothetical.
#
# Leave RESERVE_MB for the kernel, Argus NVMM buffers and the container, and clamp.
RESERVE_MB="${RESERVE_MB:-2560}"
AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')
FRAME_MB=$(( BYTES_PER_FRAME / 1048576 + 1 ))
MAX_DEPTH=$(( (AVAIL_MB - RESERVE_MB) / (4 * FRAME_MB) ))
[ "$MAX_DEPTH" -lt 8 ] && MAX_DEPTH=8
if [ "${WRITE_QUEUE_DEPTH:-64}" -gt "$MAX_DEPTH" ]; then
  echo "queue depth ${WRITE_QUEUE_DEPTH} would need ~$(( WRITE_QUEUE_DEPTH * 4 * FRAME_MB )) MB;"
  echo "  only ${AVAIL_MB} MB available and ${RESERVE_MB} MB reserved - clamping to ${MAX_DEPTH}"
  WRITE_QUEUE_DEPTH="$MAX_DEPTH"
fi
echo "queue depth ${WRITE_QUEUE_DEPTH:-64} (~$(( ${WRITE_QUEUE_DEPTH:-64} * 4 * FRAME_MB )) MB of buffers)"

echo "recording ${SECS}s at ${EFF_FPS} fps -> $DIRS"

# NO COMMENTS INSIDE THE CONTINUATION BELOW. A `#` line between two backslash-continued
# lines does not comment "just that line" - bash joins the continuation first, so the `#`
# ends the whole command there and the remaining lines become a separate command. That is
# not theoretical: a comment sat above -p image_log_direct from 2026-09-02 to 2026-09-03 and
# silently dropped all three log-dir parameters, leaving `-p image_log_dir:=... &` as its own
# (failing, backgrounded) command. The node then ran in the FOREGROUND with no image log at
# all, so the script never reached its own `sleep`/`stop` - a 20 s run captured 9308 sets
# over five minutes and wrote zero bytes, while every rate and skew line looked perfect.
#
# On image_log_direct: the inner quotes force a STRING. ros2 parses a bare true/false as a
# bool, and this parameter takes a per-camera list ("true,true,false,false"), so a bool kills
# startup with InvalidParameterTypeException.
#
# On fps:=30 being hardcoded while TRIGGER_FPS may be 20: this parameter selects the ARGUS
# SENSOR MODE, and under an external trigger the sensor emits one frame per pulse regardless.
# Verified 2026-09-03 - a 20 fps trigger against fps:=30 logged 1158 frames per camera in
# 58.7 s (19.61 Hz) with 99.5% complete sets. TRIGGER_FPS is for the space check only.
ros2 run bev_camera argus_capture_node --ros-args \
  -p width:=1456 -p height:=1088 -p fps:=30 \
  -p publish_every_n:="$EVERY_N" -p exposure_us:="$EXPOSURE_US" \
  -p write_queue_depth:="${WRITE_QUEUE_DEPTH:-64}" \
  -p image_log_direct:="\"${IMAGE_LOG_DIRECT:-false}\"" \
  -p image_log_dir:="$DIRS" -p frame_log_dir:="${BASES[0]}/imglog_${LABEL}_${STAMP}" &
CAP=$!

# LET THE WRITE QUEUES DRAIN BEFORE KILLING ANYTHING.
#
# The node's destructor joins the writer threads, so a clean SIGINT flushes whatever is
# still queued. Killing on a fixed short timer instead would discard up to
# write_queue_depth frames per camera AND leave the index disagreeing with the .raw - the
# same class of silent truncation this logger already had once.
#
# The wait is generous but still bounded: the worst case is a full queue draining at the
# slowest device's rate, which for depth 400 on eMMC is under 20 s. If it has not exited by
# then something is genuinely stuck and a kill is the right answer.
DRAIN_MAX="${DRAIN_MAX:-120}"
stop() {
  # Signal the NODE, not the `ros2 run` wrapper. SIGINT to the wrapper does not reach the
  # node: with the old 8 s kill that went unnoticed, but once the drain window was widened
  # to 120 s the node simply kept capturing for two minutes, overflowed every queue and
  # dropped 2805 frames. -x matches the process NAME exactly, so it cannot match this
  # script's own command line the way `pkill -f` would.
  pkill -INT -x argus_capture_node 2>/dev/null || true
  kill -INT "$CAP" 2>/dev/null || true
  for i in $(seq 1 "$DRAIN_MAX"); do
    pgrep -x argus_capture_node >/dev/null 2>&1 || { echo "writers drained, node exited cleanly after ${i}s"; exit 0; }
    [ "$i" = 5 ] && echo "draining write queues..."
    sleep 1
  done
  echo "WARNING: still running after ${DRAIN_MAX}s - killing. The log may be truncated." >&2
  pkill -KILL -x argus_capture_node 2>/dev/null || true
  kill -KILL "$CAP" 2>/dev/null || true
  exit 0
}
trap stop INT TERM
sleep "$SECS"
echo "elapsed - stopping"
stop
