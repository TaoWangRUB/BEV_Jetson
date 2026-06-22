## Why

The 4-camera cuVSLAM stack builds and each piece works in isolation — `libcuvslam.so`
is built + WarmUpGPU-validated in the consolidated `cuvslam-foxy:tx2` image, and the
capture node brings Argus up and creates `/camN/image_raw` topics — but capture and
VO have **never run as a pair**. cuVSLAM has never received real frames, so whether the
fisheye (KANNALA_BRANDT) intrinsics + rig extrinsics produce stable tracking is the
single largest unverified assumption. Everything downstream (IMU fusion, BEV, occupancy)
depends on the VO actually working, so this is the critical path.

## What Changes

- Verify the capture node **sustains** frame flow on all 4 cameras (`ros2 topic hz
  /camN/image_raw` at the expected ~20 Hz over a sustained run), not just "capture up".
- Fix the runtime `calib_dir` resolution: the node default was `config/calib/1640x1232`
  (a path that does not exist); the intrinsics actually live, tracked, at
  `scripts/config/calib/camN.yaml`. Point the default there.
- Run capture → VO **together** (single container, or two containers over
  `--network host` DDS) and confirm `cuvslam_multicam_node` publishes a tracking
  `nav_msgs/Odometry` on `/odom` plus the `odom→base_link` TF.
- Validate cross-camera timestamp sync feeds cuVSLAM a coherent synchronized 4-image set.

## Capabilities

### New Capabilities
- `visual-odometry`: 4-camera cuVSLAM multicam VO running end-to-end on the TX2 — fed by
  sustained Argus capture, using fisheye intrinsics + rig extrinsics, producing odometry
  that tracks real motion.

### Modified Capabilities
<!-- None: no existing specs in openspec/specs/ yet. -->

## Impact

- Nodes: `ros2/bev_camera` (capture rate), `ros2/bev_cuvslam` (`calib_dir` default,
  Track() over real frames), `ros2/bev_cuvslam/launch/bev_cuvslam.launch.py`.
- Config: `scripts/config/calib/camN.yaml` (intrinsics), `config/rig/rig_extrinsics.yaml`.
- Runtime: single `cuvslam-foxy:tx2` container or two-container DDS over `--network host`.
- Out of scope (future, see status.md / design Non-Goals): cam0 + 6th camera, IMU/EKF
  fusion, fused single-process GPU-memory node, depth/occupancy/BEV.
