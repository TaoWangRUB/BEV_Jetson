#!/usr/bin/env bash
# cuVSLAM straight from a raw rig log (retarget-vo-to-imx296-rig 5.10c): no ROS, no bag, no
# wall clock - every set, in order, and the same output twice. See cuvslam_raw_replay.cpp.
#
#   scripts/vo/replay_raw.sh <imglog dir> <out dir> [cuvslam_raw_replay options]
#
#   # VO, the reference configuration (precision, all 8 virtual cameras primary)
#   scripts/vo/replay_raw.sh datasets/imglog_run6_20260907_143036 datasets/replay_out/raw_run6_precision
#   # SLAM, the configuration README section 6 settled on
#   scripts/vo/replay_raw.sh datasets/imglog_run6_20260907_143036 datasets/replay_out/raw_run6_slam0125_performance \
#       --multicam-mode performance --slam --slam-primaries 0,1,2,5
#
# Built with the host's g++, OpenCV and yaml-cpp against third_party/cuVSLAM/build_host's
# libcuvslam.so (patch 0003 applied), as verify_rig_build.sh builds its test - no container.
# That library links CUDA 12 (libcusolver.so.11), which the host loader finds under
# /usr/local/cuda-12; /usr/local/cuda itself now points at 13.2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LIB=third_party/cuVSLAM/build_host/bin
[[ -f $LIB/libcuvslam.so ]] || {
  echo "REFUSING: no $LIB/libcuvslam.so. Run: docker compose -f docker-compose.host.yml run --rm build-cuvslam-host" >&2
  exit 1
}
SRC=scripts/vo/cuvslam_raw_replay.cpp
INC=ros2/bev_cuvslam/include/bev_cuvslam
BIN_DIR=${BIN_DIR:-$HOME/.cache/bev_cuvslam_raw_replay}
BIN=$BIN_DIR/cuvslam_raw_replay
mkdir -p "$BIN_DIR"
if [[ ! -x $BIN || $SRC -nt $BIN || $INC/rig_build.hpp -nt $BIN || $INC/virtual_pinhole.hpp -nt $BIN \
      || $LIB/libcuvslam.so -nt $BIN ]]; then
  echo "building $BIN"
  g++ -O2 -std=c++17 -I ros2/bev_cuvslam/include -I "${OPENCV_INC:-/usr/include/opencv4}" \
      -I third_party/cuVSLAM/libs "$SRC" -o "$BIN" \
      -L "$LIB" -lcuvslam -Wl,-rpath,"$(pwd)/$LIB" \
      -lopencv_core -lopencv_imgproc -lopencv_calib3d -lyaml-cpp
fi
exec "$BIN" "$@"
