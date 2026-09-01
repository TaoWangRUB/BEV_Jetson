#!/bin/bash
# Verify the node's rig construction WITHOUT a board, a ROS 2 install, or cuVSLAM's .so.
#
# Three things can silently break the virtual-stereo path, and none of them announce
# themselves at runtime - cuVSLAM would simply track badly, or not at all:
#   1. the hand-written Mei projection drifting from the model the calibration was solved with
#   2. the rotation-matrix -> quaternion conversion (four trace branches, easy to get wrong)
#   3. the rig_from_fisheye * Ry composition, where a sign error still produces a valid rig
#
# The third is caught by re-running cuVSLAM's own frustum-intersection test on the poses
# the C++ actually emits: a wrong composition drops the pairing from ~0.94 to ~0.03.
set -e
cd "$(dirname "$0")/../.."
OPENCV_INC=${OPENCV_INC:-/usr/include/opencv4}
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

sed -n '/^cv::Matx44d load_matrix4/,/^}$/p;/^cuvslam::Pose pose_from_matrix/,/^}$/p' \
    ros2/bev_cuvslam/src/cuvslam_multicam_node.cpp > "$OUT/helpers.inc"
cp scripts/vo/rig_build_test.cpp "$OUT/"
g++ -O2 -I "$OUT" -I ros2/bev_cuvslam/include -I "$OPENCV_INC" -I third_party/cuVSLAM/libs \
    "$OUT/rig_build_test.cpp" -o "$OUT/rig_build_test" \
    -lopencv_core -lopencv_imgproc -lopencv_calib3d -lyaml-cpp
# Defaults are the tracked config - the rig the node will actually load. Override all
# three to verify a CANDIDATE solve before it is promoted (3R.16): a rig that fails the
# 0.5 frustum gate should be found here, not diagnosed as ROS 2 wiring on the TX2.
#   verify_rig_build.sh <extrinsics.yaml> <virtual_stereo.yaml> <calib_dir>
EXT=${1:-config/rig/rig_extrinsics_imx296.yaml}
VS=${2:-config/rig/virtual_stereo_imx296.yaml}
CALIB=${3:-config/calib/imx296_1456x1088}
echo "extrinsics: $EXT"
echo "vstereo:    $VS"
echo "calib:      $CALIB"
"$OUT/rig_build_test" "$EXT" "$VS" "$CALIB" > "$OUT/poses.txt"
python3 scripts/vo/check_rig_poses.py "$OUT/poses.txt" "$EXT" config/rig/rig_layout.yaml
