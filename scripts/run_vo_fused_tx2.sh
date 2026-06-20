#!/usr/bin/env bash
# Run the FUSED zero-copy Argus->cuVSLAM VO node on the TX2 — one process captures and
# tracks, feeding NVMM frames to cuVSLAM as GPU memory (no CPU round-trip, no DDS images).
# ~3x less CPU than the modular capture+VO pipeline (see scripts/run_vo_tx2.sh).
#
#   ./scripts/run_vo_fused_tx2.sh            # run fused VO
#   RECORD=1 ./scripts/run_vo_fused_tx2.sh  # also rosbag /cuvslam/odometry + /tf into bags/
#
# Publishes /cuvslam/odometry + odom->base_link TF. Stop with Ctrl-C (SIGINT) or
# `docker stop` (SIGTERM) — both shut the node down cleanly so Argus isn't left wedged.
set -euo pipefail
BEV="${BEV:-/media/nvidia/workspace/BEV_Jetson}"
IMG="${IMG:-cuvslam-foxy:tx2}"

exec docker run --rm -it --runtime nvidia --network host --stop-signal=SIGTERM \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RECORD="${RECORD:-0}" \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev \
  -v "$BEV":/workspace -w /workspace "$IMG" bash -lc '
    set -e
    source /opt/ros/foxy/setup.bash && source install/setup.bash
    export LD_LIBRARY_PATH=/workspace/third_party/cuVSLAM/build_tx2gpu/bin:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
    if [ "${RECORD:-0}" = 1 ]; then
      mkdir -p /workspace/bags
      ros2 bag record -o /workspace/bags/fused_$(date +%Y%m%d_%H%M%S) /cuvslam/odometry /tf &
    fi
    exec ros2 run bev_cuvslam bev_cuvslam_fused_node --ros-args \
      -p calib_dir:=scripts/config/832x624 \
      -p width:=832 -p height:=624 -p sensor_width:=1640 -p sensor_height:=1232 -p fps:=60 \
      -p rig_extrinsics:=config/rig/rig_extrinsics.yaml \
      -p cameras:=[cam1,cam2,cam3,cam4]
  '
