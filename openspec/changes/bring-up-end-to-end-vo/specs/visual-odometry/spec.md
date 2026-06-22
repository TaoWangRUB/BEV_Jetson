## ADDED Requirements

### Requirement: Sustained multi-camera capture

The capture node SHALL sustain a steady frame rate on all four cameras for the duration
of a run, not merely initialize Argus. Each `/camN/image_raw` topic (cam1–cam4) SHALL
publish mono8 frames at the configured rate (~20 Hz) without stalls or session drops.

#### Scenario: All four camera topics sustain the configured rate

- **WHEN** `argus_capture_node` runs with `sensor_ids:=[0,1,2,3]` at `width:=1640 height:=1232 fps:=20`
- **THEN** `ros2 topic hz /cam1/image_raw` … `/cam4/image_raw` each report ≈20 Hz over a sustained (≥30 s) window
- **AND** no Argus session failures or topic stalls occur during the window

### Requirement: Calibration directory resolves at runtime

The VO node SHALL locate the fisheye intrinsics (`camN.yaml`, KANNALA_BRANDT) and rig
extrinsics at runtime. The default `calib_dir` SHALL point at the actual data location,
or the required override SHALL be documented so the node loads calibration without error.

#### Scenario: Node loads intrinsics and extrinsics on startup

- **WHEN** `cuvslam_multicam_node` starts with `calib_dir` pointing at `scripts/config/calib` (the default) and `rig_extrinsics:=config/rig/rig_extrinsics.yaml`
- **THEN** it loads `cam1.yaml`…`cam4.yaml` and the rig extrinsics without a file-not-found error
- **AND** it builds a 4-camera cuVSLAM `Rig`

### Requirement: End-to-end visual odometry publishes tracking

With capture and VO running together, the system SHALL feed cuVSLAM synchronized
4-image sets and publish odometry that tracks real motion. `cuvslam_multicam_node`
SHALL call `Odometry::Track()` on each synchronized set and publish `nav_msgs/Odometry`
on `/odom` plus the `odom→base_link` TF.

#### Scenario: Odometry tracks under motion

- **WHEN** `argus_capture_node` and `cuvslam_multicam_node` run together (single container or two containers over `--network host` DDS) and the rig is moved
- **THEN** `/odom` publishes a continuously updating `nav_msgs/Odometry` pose that changes consistently with the rig's motion
- **AND** the `odom→base_link` TF is broadcast

#### Scenario: Synchronized 4-image sets reach the tracker

- **WHEN** the four camera streams are time-synchronized into a frame set
- **THEN** each `Track()` call receives a coherent set of four images with matched timestamps within the sync tolerance

#### Scenario: Cross-camera frustum overlap yields metric scale

- **WHEN** the rig runs in `OdometryMode::Multicamera` with the four 160° fisheye cameras at 90° spacing
- **THEN** cuVSLAM auto-forms stereo connections across each adjacent overlapping pair (the connected ring), so tracking is metric (not up-to-scale) rather than degrading to isolated monocular tracking
