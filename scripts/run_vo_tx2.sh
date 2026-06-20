#!/usr/bin/env bash
# Run the 4-camera capture + cuVSLAM multicam VO together in ONE container on the TX2
# (single-container topology — sidesteps cross-container DDS discovery). Capture starts
# in the background, the VO node runs in the foreground; Ctrl-C stops both.
#
#   ./scripts/run_vo_tx2.sh            # run capture + VO
#   RECORD=1 ./scripts/run_vo_tx2.sh  # also rosbag /cuvslam/odometry + /tf into bags/
# (Recording the camera streams too would add ~20 MB/s of SD writes and throttle the
#  live pipeline, so RECORD captures only the lightweight VO output. Stop with Ctrl-C —
#  SIGINT lets ros2 bag close the sqlite WAL cleanly.)
#
# Outputs: /cuvslam/odometry (nav_msgs/Odometry) + odom->base_link TF. Move the rig to
# see it track. Calibration/extrinsics default to scripts/config/calib + config/rig.
set -euo pipefail
BEV="${BEV:-/media/nvidia/workspace/BEV_Jetson}"
IMG="${IMG:-cuvslam-foxy:tx2}"

exec docker run --rm -it --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RECORD="${RECORD:-0}" \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev \
  -v "$BEV":/workspace -w /workspace "$IMG" bash -lc '
    set -e
    source /opt/ros/foxy/setup.bash && source install/setup.bash
    export LD_LIBRARY_PATH=/workspace/third_party/cuVSLAM/build_tx2gpu/bin:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
    ros2 run bev_camera argus_capture_node --ros-args \
      -p sensor_ids:=[0,1,2,3] -p width:=1640 -p height:=1232 -p fps:=20 &
    trap "kill %1 2>/dev/null" EXIT
    sleep 8
    if [ "${RECORD:-0}" = 1 ]; then
      mkdir -p /workspace/bags
      ros2 bag record -o /workspace/bags/vo_$(date +%Y%m%d_%H%M%S) \
        /cuvslam/odometry /tf &
    fi
    exec ros2 run bev_cuvslam cuvslam_multicam_node --ros-args \
      -p calib_dir:=scripts/config/calib \
      -p rig_extrinsics:=config/rig/rig_extrinsics.yaml \
      -p cameras:=[cam1,cam2,cam3,cam4]
  '
