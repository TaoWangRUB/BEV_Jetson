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
#   LOG_DIRS="/logs,/logs,/sdlog,/ramlog" MOTION_SECONDS=60 log_raw.sh
#   LOG_DIR=/ramlog MOTION_SECONDS=10 log_raw.sh          # single target, all four
set -euo pipefail

: "${EXPOSURE_US:?set EXPOSURE_US to the measured trigger pulse width in us (j106-trigctl.py status)}"
SECS="${MOTION_SECONDS:-60}"
LABEL="${LOG_LABEL:-run}"
EVERY_N="${PUBLISH_EVERY_N:-1}"
STAMP=$(date +%Y%m%d_%H%M%S)
BYTES_PER_FRAME=$((1456 * 1088))
EFF_FPS=$((30 / EVERY_N))

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
ros2 run bev_camera argus_capture_node --ros-args \
  -p width:=1456 -p height:=1088 -p fps:=30 \
  -p publish_every_n:="$EVERY_N" -p exposure_us:="$EXPOSURE_US" \
  -p write_queue_depth:="${WRITE_QUEUE_DEPTH:-64}" \
  # The inner quotes force a STRING: ros2 parses a bare true/false as a bool, and this
  # parameter takes a per-camera list ("true,true,false,false"), so a bool kills startup
  # with InvalidParameterTypeException.
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
