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
# OBS=1 also publishes and records /cuvslam/landmarks + /cuvslam/observations, which is
# what scripts/vo/rerun_multicam.py needs to draw the tracked features. It costs frame
# rate (the landmark export is on the Track() thread), so leave it OFF for any run whose
# rate or trajectory is the measurement, and name the output obs_* rather than odom_*.
#
# Foxy's ros2 bag play on this stack has no --clock; VO matches on image header
# stamps (same as the successful 2026-09-03 TX2 replay).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BAG="${1:?usage: $0 <camera_bag_dir> [rate]}"
OBS="${OBS:-0}"
# SLAM triples Track(): 9-12 ms without it, 16-33 ms with it and spikes to 133 ms, because
# enabling SLAM turns on the observation/landmark exports INSIDE the Odometry config, so the
# cost lands in Track() itself. At 1.0x the budget is 50 ms/set, so it overruns, sets are
# dropped, and the gaps make the TRACKER fail - a 3.42 m step appears that is simply not
# there in a clean run. Slowing the replay is the fix, and it is a real fix rather than a
# workaround: 0.4x gives 125 ms/set and the VO comes back identical to the no-SLAM baseline
# (859 vs 861 poses, 22.21 vs 22.24 m, worst step 0.39 vs 0.47 m).
if [[ "${SLAM:-0}" == "1" ]]; then _default_rate=0.4; else _default_rate=1.0; fi
RATE="${2:-${RATE:-$_default_rate}}"
_pfx=odom; [[ "$OBS" == "1" ]] && _pfx=obs
OUT="${OUT:-${REPO_ROOT}/datasets/replay_out/${_pfx}_$(date +%Y%m%d_%H%M%S)}"
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

# Replay wants RELIABLE image QoS: the default best-effort loses ~1 set in 6 to dropped
# UDP fragments at every replay rate (5.10). Override with IMAGE_QOS=sensor_data to
# reproduce the old behaviour.
QOS="${IMAGE_QOS:-reliable}"
QOS_DEPTH="${IMAGE_QOS_DEPTH:-100}"
if [[ "$OBS" == "1" ]]; then
  LAUNCH_ARGS="publish_landmarks:=true publish_observations:=true"
  REC_TOPICS="/cuvslam/odometry /tf /cuvslam/landmarks /cuvslam/observations"
else
  LAUNCH_ARGS=""
  REC_TOPICS="/cuvslam/odometry /tf"
fi
LAUNCH_ARGS="${LAUNCH_ARGS} image_qos:=${QOS} image_qos_depth:=${QOS_DEPTH}"
# SLAM=1 adds the pose graph and loop closure, publishing /cuvslam/slam_odometry BESIDE the
# pure-VO /cuvslam/odometry. Both are recorded so the two trajectories can be compared.
if [[ "${SLAM:-0}" == "1" ]]; then
  # Unlimited pose graph by default offline: the 300-node cap ends the optimised
  # trajectory mid-run (1.7h). SLAM_MAX_MAP_SIZE=300 restores the real-time figure.
  LAUNCH_ARGS="${LAUNCH_ARGS} enable_slam:=true slam_max_map_size:=${SLAM_MAX_MAP_SIZE:-0}"
  REC_TOPICS="${REC_TOPICS} /cuvslam/slam_odometry /cuvslam/loop_closures"
  REC_TOPICS="${REC_TOPICS} /cuvslam/slam_path /cuvslam/loop_closure_edges"
fi

echo "replay BAG=$BAG_IN RATE=$RATE OUT=$OUT_IN OBS=$OBS QOS=$QOS/$QOS_DEPTH SLAM=${SLAM:-0}"
if [[ "${SLAM:-0}" == "1" ]] && awk "BEGIN{exit !($RATE > 0.6)}"; then
  echo "WARNING: SLAM at ${RATE}x will overrun the per-set budget and DROP frames, which"
  echo "  makes the odometry itself worse - not just the SLAM layer. Use 0.4 or slower."
fi
"${COMPOSE[@]}" run --rm shell bash -lc "
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /workspace/install_host/setup.bash
set -u
export LD_LIBRARY_PATH=/workspace/third_party/cuVSLAM/build_host/bin:\${LD_LIBRARY_PATH:-}
rm -rf '${OUT_IN}'
ros2 launch bev_cuvslam bev_cuvslam.launch.py ${LAUNCH_ARGS} > /tmp/vo_host.log 2>&1 &
VO=\$!
for i in \$(seq 1 60); do
  ros2 node list 2>/dev/null | grep -q cuvslam_multicam && break
  sleep 1
done
ros2 node list | grep cuvslam || { echo 'VO node failed to start'; tail -40 /tmp/vo_host.log; exit 1; }
ros2 bag record -o '${OUT_IN}' ${REC_TOPICS} > /tmp/odom_rec_host.log 2>&1 &
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
