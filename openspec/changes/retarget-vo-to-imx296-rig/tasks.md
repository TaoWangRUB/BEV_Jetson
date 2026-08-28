## 0. Board prerequisites (verify before writing code)

- [ ] 0.1 Find the board and confirm the population: SSH in, confirm it booted `LABEL j106imx296`, and list `/dev/video*` with each device's i2c name — expect `2-001a`, `2-0018`, `7-001a`, `7-0018` (ports C–F). Record the bind order actually observed.
- [ ] 0.2 Confirm trigger state: `cat /sys/module/imx296/parameters/trigger_mode` and that the STM32 is emitting (`j106-trigctl.py` in the J106 repo). Record the trigger rate and pulse width in use.
- [ ] 0.3 Confirm `jetson-clocks` is applied, then capture a 4-camera baseline with `scripts/stream/csi_sender.sh` + `csi_receiver.sh` (`PORTS="c d e f"`) — all four live, stable brightness, no flashing.
- [ ] 0.4 Measure the live inter-camera skew and frame rate on the board (the J106 repo's `j106-sync-check.py`) and record the numbers as this change's baseline.

## 1. Capture node → IMX296

- [ ] 1.1 Add runtime port→sensor-id resolution to `argus_capture_node.cpp` (D1): read each `/dev/videoN`'s i2c name from sysfs, map through the port table incl. the IMX219 aliases, log the resolved mapping at startup. Keep `sensor_ids` as an explicit override.
- [ ] 1.2 Fail startup with a named message when a configured port has no live device (spec: *A missing camera is reported, not silently dropped*).
- [ ] 1.3 Default capture geometry to 1456×1088; parameterise the rate from the trigger rate found in 0.2.
- [ ] 1.4 Detect trigger mode and lock AE (gain + digital gain clamp) when it is active; leave AE free otherwise; log which branch was taken (spec: *Exposure is stable under external trigger*).
- [ ] 1.5 Add the calibration/geometry cross-check: refuse to publish when the loaded calibration's image size or sensor family disagrees with the live capture configuration.
- [ ] 1.6 Publish per-frame Argus timestamps (already the case) and add a set-skew/drop counter suitable for the health requirement — expose it on a diagnostics topic or as throttled logging with the measured spread.
- [ ] 1.7 Build on the board in the `cuvslam-foxy:tx2` container and verify: all 4 topics publish at the trigger rate, mapping log matches 0.1, brightness p2p < 5 luma levels over 30 s (spec: *Brightness holds steady*).

## 2. Intrinsics for the IMX296 modules

- [ ] 2.1 Move `scripts/config/calib/cam{1..4}.yaml` (+ their `.npz`/preview artefacts) to `scripts/config/calib/imx219-1640x1232/`, labelled with the rig they belong to (D6).
- [ ] 2.2 Capture calibration sets at 1456×1088 for all four cameras with `scripts/calib/capture_calib_sets.sh` (adjust for the new size/ports).
- [ ] 2.3 Solve KANNALA_BRANDT intrinsics per camera with `intrinsic_calib.py`; reject and re-shoot any camera whose RMS reprojection error is ≥ 1.0 px.
- [ ] 2.4 Write the new `cam{1..4}.yaml` with `sensor: imx296` and the true image size; commit with `git add -f` (`scripts/config` is gitignored — see design D6).
- [ ] 2.5 Verify undistortion previews look sane per camera (straight lines straight near the centre, no gross fisheye residual).

## 3. Extrinsics and camera↔IMU Δ via Kalibr

- [ ] 3.1 Stand up Kalibr in Docker on the host (Noetic image); verify it runs on one of the existing `datasets/` bags before trusting it on ours.
- [ ] 3.2 Print/obtain an AprilGrid target and record its geometry (tag size, spacing) in the repo alongside the calibration.
- [ ] 3.3 Record the four-camera calibration bag on the board (`ros2 bag`, ~4 Hz images, target moving through all four fields of view incl. the adjacent overlaps); convert to ROS1 with `rosbags-convert` on the host.
- [ ] 3.4 Record the camera+IMU bag: one camera at full rate + MPU-9250 at full rate, with the excitation sequence Kalibr wants (rotation about all three axes, then translation).
- [ ] 3.5 Solve rig extrinsics with `kalibr_calibrate_cameras` (pinhole-equi). If the four-camera chain will not converge, fall back per design D4 (pairwise + compose, or keep the feature-based extrinsics) and record which route was taken.
- [ ] 3.6 Solve Δ with `kalibr_calibrate_imu_camera` on the single camera + IMU; record the value, the method, and its uncertainty.
- [ ] 3.7 Write the results: extrinsics into `config/rig/rig_extrinsics_vo.yaml` (stating frame convention, source recording, date, per-camera residuals) and Δ as a stated constant with provenance. Check the ring loop-closure residual (spec: *Extrinsics are consistent around the rig*).
- [ ] 3.8 Confirm the mounting orientation question for the IMX296 modules — are they inverted like the IMX219s were? Whatever the answer, it must be *in* the extrinsics, not applied as a separate hidden roll.

## 4. Remove the sync workaround

- [ ] 4.1 `cuvslam_multicam_node.cpp`: delete the latest-frame bundler and `sync_slop_ms`; form sets from frames whose timestamps span < `max_skew_us` (default 1000), pass each image its own `timestamp_ns`, drop and count what fails (D2).
- [ ] 4.2 `bev_cuvslam_fused_node.cpp`: stamp each image with its own `iframe->getTime()` instead of `ts0`, and apply the same skew gate to the lock-step set.
- [ ] 4.3 Report the drop counter and recent worst-case skew from both nodes; make a stopped trigger diagnosable as a trigger fault, not a camera failure (spec: *A stopped trigger is diagnosable*).
- [ ] 4.4 Update `fused_vo_params.yaml`, `run_vo_tx2.sh`, `run_vo_fused_tx2.sh` for the new parameters and resolution; remove references to the bundler from comments and docs.
- [ ] 4.5 Run both nodes on the board: confirm zero dropped sets with the trigger live, worst-case skew < 1 ms, and `/cuvslam/odometry` tracking with no "tracking lost".

## 5. Motion test (closes bring-up-end-to-end-vo 3.4 / 3.6)

- [ ] 5.1 Move the rig a measured straight-line distance; record `/cuvslam/odometry` + `/tf` and compare reported translation against the tape measure (spec: *Translation is recovered at true scale*, 5 %).
- [ ] 5.2 Return the rig to its starting pose and check the trajectory returns near the origin; record the drift.
- [ ] 5.3 Determine whether cross-camera stereo links actually form (cuVSLAM's 0.5 frustum-overlap gate). If none do, make the node report that it is running unscaled, and record the evidence — it is the input to the OpenMAVIS/D2SLAM decision.
- [ ] 5.4 Compare against the old rig's ~8.5 Hz bundled odometry: rate, drift, and whether tracking survives motion that previously broke it.

## 6. Wrap-up

- [ ] 6.1 Tick `bring-up-end-to-end-vo` tasks 3.4/3.6 with the evidence from §5, or state precisely why they remain open.
- [ ] 6.2 Update `README.md` and `docs/` for the IMX296 rig: population, trigger prerequisite, `jetson-clocks`, new resolution, calibration layout.
- [ ] 6.3 Update the project memory notes with the measured outcome (skew, rate, whether tracking is metric, Δ).
- [ ] 6.4 Archive this change once §5 has a verdict.
