#!/usr/bin/env bash
# Does cuVSLAM actually TRACK all eight virtual cameras?
#
# WHY THIS EXISTS. On 2026-09-08 cam2+45 was found to produce 27 features per frame against
# 300-457 for the other seven - in EVERY run ever recorded. Nothing in the project caught it:
# the carve is valid, the image is textured and correctly exposed, the frustum overlap is
# 0.936, the calibration is not the weakest of the four, and the pair's epipolar residual is
# better than a working pair's. verify_rig_build.sh passed it, because a camera can have a
# perfectly good stereo partner and still be dropped AFTER the graph is built.
#
# cuvslam2.cpp declares camera 0 a "depth camera" for every non-RGBD/non-Multisensor run
# (depth_ids={0}, allow_stereo_track_for_depth=false). The FIG then erases every secondary
# edge landing on camera 0, deletes any primary left with no secondaries, and re-adds camera 0
# but not its orphan. MultiSOFGPU builds mono_sof_ from primary_cameras(), so the orphan never
# gets a tracker. On a ring of DISJOINT pairs - every vcam has exactly one partner - camera 0's
# partner dies every time. Reordering the camera list only moves which one.
#
# This runs cuVSLAM's own selection logic on our topology. No GPU, no images, ~1 second.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO}/third_party/cuVSLAM"
[[ -f "${SRC}/libs/camera/frustum_intersection_graph.cpp" ]] || {
  echo "REFUSING: cuVSLAM source not at ${SRC}" >&2; exit 1; }
EIGEN="$(dirname "$(dirname "$(find /usr/include -name Geometry -path '*Eigen*' 2>/dev/null | head -1)")")"
[[ -n "$EIGEN" ]] || { echo "REFUSING: Eigen headers not found" >&2; exit 1; }
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
# NOT -I libs/common: it puts cuVSLAM's own time.h ahead of the system <time.h>.
g++ -std=c++17 -I "${SRC}/libs" -I "$EIGEN" \
  "${REPO}/scripts/vo/fig_primary_test.cpp" \
  "${REPO}/scripts/vo/fig_primary_stubs.cpp" \
  "${SRC}/libs/camera/frustum_intersection_graph.cpp" \
  -o "${OUT}/fig_primary_test"
# Does this cuVSLAM tree carry the fix? Gate on what it will actually build.
PATCHED=0
grep -q 'enable_depth_stereo_tracking_fig = true;' "${SRC}/libs/cuvslam/cuvslam2.cpp" && PATCHED=1
"${OUT}/fig_primary_test" "$PATCHED"
