#!/bin/bash
# Section 5 motion test, board side. Run ON the TX2.
#
#   ./run_motion_test.sh <label> <tape_measure_metres>
#
# PREFLIGHT IS NOT OPTIONAL. trigger_mode does not persist across a reboot and forgetting
# it is SILENT: the STM32 keeps pulsing, the cameras ignore it and free-run, nothing logs
# an error, and the only symptoms are AE gain-hunting and sets that are not sets. That
# already cost one mid-session recording. Same for jetson_clocks.
set -euo pipefail
LABEL=${1:?usage: run_motion_test.sh <label> <tape_metres>}
TAPE=${2:?give the tape-measured distance in metres, even for a return-to-origin run}
OUT=/media/nvidia/workspace/motion_${LABEL}_$(date +%H%M%S)

trig=$(cat /sys/module/imx296/parameters/trigger_mode 2>/dev/null || echo missing)
[ "$trig" = "1" ] || { echo "REFUSING: trigger_mode=$trig, expected 1."; \
                       echo "  sudo sh -c 'echo 1 > /sys/module/imx296/parameters/trigger_mode'"; exit 1; }
grep -q 2035200 /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null \
  || { echo "note: jetson_clocks not applied, re-applying"; sudo jetson_clocks; }
python3 /home/nvidia/tools/j106-trigctl.py --port /dev/ttyTHS1 status || \
  echo "WARNING: could not read the trigger generator - exposure_us may be stale"

mkdir -p "$OUT"; echo "$TAPE" > "$OUT/tape_metres.txt"
ros2 launch bev_cuvslam bev_cuvslam.launch.py > "$OUT/vo.log" 2>&1 &
VO=$!; trap 'kill $VO 2>/dev/null || true' EXIT
sleep 8   # cuVSLAM warms up the GPU and builds 8 remap tables before it tracks

echo
echo "recording to $OUT - move the rig now, Ctrl-C when done"
echo "  5.1 straight line: move exactly $TAPE m and STOP"
echo "  5.2 return-to-origin: go out and come back to the SAME pose"
ros2 bag record -o "$OUT/bag" /cuvslam/odometry /tf /cam1/frame_meta || true

echo; echo "=== set / skew / drop / remap, from the node's own reporting ==="
grep -E "sets |dropped|remap|tracking lost|frustum" "$OUT/vo.log" | tail -20
echo "bag: $OUT/bag   copy the whole $OUT to the host and run analyze_motion.py"
