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
#   # Multisensor (cuNLS) + the MPU-9250, against the USE_CUNLS=ON library (build_cuvslam_host.sh)
#   CUVSLAM_LIB_DIR=third_party/cuVSLAM/build_host_cunls/bin scripts/vo/replay_raw.sh \
#       datasets/imglog_run6_20260907_143036 datasets/replay_out/raw_run6_multisensor_imu --multisensor --imu
#
# Built with the host's g++, OpenCV and yaml-cpp against third_party/cuVSLAM/build_host's
# libcuvslam.so (patch 0003 applied), as verify_rig_build.sh builds its test - no container.
# That library links CUDA 12 (libcusolver.so.11), which the host loader finds under
# /usr/local/cuda-12; /usr/local/cuda itself now points at 13.2.
# CUVSLAM_LIB_DIR picks another libcuvslam.so; each library gets its own binary, linked with
# that directory as its runpath and run with it first on LD_LIBRARY_PATH, so the two can never
# be mixed up. CUDA 12's library directory goes right behind it (CUDA12_LIB, default
# /usr/local/cuda-12/targets/x86_64-linux/lib): the cuNLS build also needs libcusparse.so.12,
# and CUDA 13.2 ships a file of that name too, which /usr/local/cuda (the library's own runpath,
# set inside the build container where it was 12.8) and the ld.so.conf order would pick first -
# 13.2's cuSPARSE under a 12.8 runtime on a driver that reports CUDA 13.0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DEFAULT_LIB=third_party/cuVSLAM/build_host/bin
LIB=${CUVSLAM_LIB_DIR:-$DEFAULT_LIB}
[[ -f $LIB/libcuvslam.so ]] || {
  echo "REFUSING: no $LIB/libcuvslam.so. Run: docker compose -f docker-compose.host.yml run --rm build-cuvslam-host" >&2
  exit 1
}
case $LIB in
  /*) LIB_ABS=$LIB ;;
  *) LIB_ABS=$(pwd)/$LIB ;;
esac
TAG=
[[ $LIB != "$DEFAULT_LIB" ]] && TAG=_$(basename "$(dirname "$LIB_ABS")")
SRC=scripts/vo/cuvslam_raw_replay.cpp
INC=ros2/bev_cuvslam/include/bev_cuvslam
BIN_DIR=${BIN_DIR:-$HOME/.cache/bev_cuvslam_raw_replay}
BIN=$BIN_DIR/cuvslam_raw_replay$TAG
mkdir -p "$BIN_DIR"
if [[ ! -x $BIN || $SRC -nt $BIN || $INC/rig_build.hpp -nt $BIN || $INC/virtual_pinhole.hpp -nt $BIN \
      || $LIB/libcuvslam.so -nt $BIN ]]; then
  echo "building $BIN"
  g++ -O2 -std=c++17 -I ros2/bev_cuvslam/include -I "${OPENCV_INC:-/usr/include/opencv4}" \
      -I third_party/cuVSLAM/libs "$SRC" -o "$BIN" \
      -L "$LIB" -lcuvslam -Wl,-rpath,"$LIB_ABS" \
      -lopencv_core -lopencv_imgproc -lopencv_calib3d -lyaml-cpp
fi
if [[ -n $TAG ]]; then
  CUDA12_LIB=${CUDA12_LIB-/usr/local/cuda-12/targets/x86_64-linux/lib}
  [[ -d $CUDA12_LIB ]] || CUDA12_LIB=
  echo "libcuvslam: $LIB_ABS${CUDA12_LIB:+ (CUDA libraries from $CUDA12_LIB first)}"
  LD_LIBRARY_PATH=$LIB_ABS${CUDA12_LIB:+:$CUDA12_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec "$BIN" "$@"
fi
exec "$BIN" "$@"
