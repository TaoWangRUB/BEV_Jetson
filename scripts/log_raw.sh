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

echo "recording ${SECS}s at ${EFF_FPS} fps -> $DIRS"
ros2 run bev_camera argus_capture_node --ros-args \
  -p width:=1456 -p height:=1088 -p fps:=30 \
  -p publish_every_n:="$EVERY_N" -p exposure_us:="$EXPOSURE_US" \
  -p image_log_dir:="$DIRS" -p frame_log_dir:="${BASES[0]}/imglog_${LABEL}_${STAMP}" &
CAP=$!

# Bounded stop: the node has Argus sessions and file handles to close, but it must not be
# able to outlast the requested run length.
stop() {
  kill -INT "$CAP" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8; do kill -0 "$CAP" 2>/dev/null || break; sleep 1; done
  kill -KILL "$CAP" 2>/dev/null || true
  exit 0
}
trap stop INT TERM
sleep "$SECS"
echo "elapsed - stopping"
stop
