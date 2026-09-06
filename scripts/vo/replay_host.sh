#!/usr/bin/env bash
# Replay a camera bag through modular cuVSLAM on the HOST (x86_64).
#
# Same idea as the TX2 replay_bags/replay.sh, but in bev-host-cuvslam with the
# host-built libcuvslam.so — avoids TX2 OOM on large bags.
#
#   ./scripts/vo/replay_host.sh /tmp/run1_motion.bag
#   ./scripts/vo/replay_host.sh /tmp/run1_motion.bag 0.5
#   RATE=0.25 OUT=datasets/replay_out/odom_run1 ./scripts/vo/replay_host.sh /tmp/run1_motion.bag
#
# Foxy's ros2 bag play on this stack has no --clock; VO matches on image header
# stamps (same as the successful 2026-09-03 TX2 replay).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BAG="${1:?usage: $0 <camera_bag_dir> [rate]}"
RATE="${2:-${RATE:-0.5}}"
OUT="${OUT:-${REPO_ROOT}/datasets/replay_out/odom_$(date +%Y%m%d_%H%M%S)}"
COMPOSE=(docker compose -f docker-compose.host.yml)

[[ -d "$BAG" || -f "$BAG" ]] || { echo "REFUSING: bag not found: $BAG" >&2; exit 1; }
[[ -f third_party/cuVSLAM/build_host/bin/libcuvslam.so ]] || {
  echo "REFUSING: host libcuvslam.so missing. Run:" >&2
  echo "  ${COMPOSE[*]} run --rm build-cuvslam-host" >&2
  exit 1
}
[[ -x install_host/bev_cuvslam/lib/bev_cuvslam/cuvslam_multicam_node ]] || {
  echo "REFUSING: host VO node missing. Run:" >&2
  echo "  ${COMPOSE[*]} run --rm build-ws-host" >&2
  exit 1
}

# Make the bag path visible inside the container (/workspace or /tmp).
if [[ "$BAG" != /* ]]; then BAG="${REPO_ROOT}/${BAG}"; fi
case "$BAG" in
  "${REPO_ROOT}"/*) BAG_IN="/workspace/${BAG#${REPO_ROOT}/}" ;;
  /tmp/*)           BAG_IN="$BAG" ;;
  *) echo "REFUSING: bag must live under the repo or /tmp (got $BAG)" >&2; exit 1 ;;
esac
case "$OUT" in
  "${REPO_ROOT}"/*) OUT_IN="/workspace/${OUT#${REPO_ROOT}/}" ;;
  /tmp/*)           OUT_IN="$OUT" ;;
  *) OUT="${REPO_ROOT}/datasets/replay_out/$(basename "$OUT")"; OUT_IN="/workspace/${OUT#${REPO_ROOT}/}" ;;
esac
mkdir -p "$(dirname "$OUT")"

echo "replay BAG=$BAG_IN RATE=$RATE OUT=$OUT_IN"
"${COMPOSE[@]}" run --rm shell bash -lc "
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /workspace/install_host/setup.bash
set -u
export LD_LIBRARY_PATH=/workspace/third_party/cuVSLAM/build_host/bin:\${LD_LIBRARY_PATH:-}
rm -rf '${OUT_IN}'
ros2 launch bev_cuvslam bev_cuvslam.launch.py > /tmp/vo_host.log 2>&1 &
VO=\$!
for i in \$(seq 1 60); do
  ros2 node list 2>/dev/null | grep -q cuvslam_multicam && break
  sleep 1
done
ros2 node list | grep cuvslam || { echo 'VO node failed to start'; tail -40 /tmp/vo_host.log; exit 1; }
ros2 bag record -o '${OUT_IN}' /cuvslam/odometry /tf > /tmp/odom_rec_host.log 2>&1 &
REC=\$!
sleep 2
echo '=== playing ==='
ros2 bag play '${BAG_IN}' -r '${RATE}' --read-ahead-queue-size 10
sleep 2
kill -INT \$REC 2>/dev/null || true
wait \$REC 2>/dev/null || true
kill -INT \$VO 2>/dev/null || true
sleep 3
kill -KILL \$VO 2>/dev/null || true
echo '=== VO log tail ==='
tail -40 /tmp/vo_host.log
echo '=== odom bag ==='
ros2 bag info '${OUT_IN}' 2>/dev/null || ls -la '${OUT_IN}'
echo OUTDIR=${OUT}
"
echo "host path: $OUT"
