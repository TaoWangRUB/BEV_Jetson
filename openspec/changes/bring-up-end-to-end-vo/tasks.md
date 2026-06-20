## 0. Environment / build (board: TX2 `/media/nvidia/workspace/BEV_Jetson`)

- [x] 0.1 Sync via git (push from dev → pull on TX2 `BEV_Jetson`); submodule git error fixed via patch refactor + `ignore = all`
- [x] 0.2 Build the ROS 2 workspace (`colcon build bev_camera bev_cuvslam`) in the `cuvslam-foxy:tx2` container — both packages link
- [~] 0.3 `build_and_validate.sh` on the patch-based build (lib + WarmUpGPU) — running; patch step confirmed "already applied"

## 1. Capture frame-flow validation

- [x] 1.1 Run `argus_capture_node` with `sensor_ids:=[0,1,2,3]` @ 1640×1232 / 20 fps (nvargus-daemon already active; no host restart needed) → "Argus capture up: 4 cameras", first frame published on all 4 cams
- [~] 1.2 Measure `ros2 topic hz /cam1..4/image_raw` over a sustained (≥30 s) window; confirm ≈20 Hz, no stalls — **in progress**: `ros2 topic hz` stdout buffering defeated naive measurement; re-measuring with a fixed-window subscriber. Diagnostics show no error path firing (no acquireFrame timeout / NvBuffer-map failure), so frames likely flow — needs a clean count
- [ ] 1.3 Record the actual sustained rate (and any drops) in this change / change log

## 2. Calibration path fix

- [x] 2.1 Locate the real intrinsics: `scripts/config/calib/cam{1..4}.yaml` (KANNALA_BRANDT, 1640×1232, tracked) — the old default `config/calib/1640x1232` did not exist
- [x] 2.2 Point the node + launch default `calib_dir` at `scripts/config/calib`; update docs
- [ ] 2.3 Verify the 4-camera cuVSLAM `Rig` builds without file-not-found errors (needs VO node board run)

## 3. End-to-end capture → VO

- [ ] 3.1 Choose topology: run both nodes in one container (capture backgrounded, VO foreground) — single-container chosen for bring-up (cross-container DDS discovery failed: topics invisible from a 2nd container even with `--network host`)
- [ ] 3.2 Launch capture + `cuvslam_multicam_node` together; confirm `Track()` receives synchronized 4-image sets
- [ ] 3.3 `ros2 topic echo /odom --no-arr` — confirm a continuously updating pose
- [ ] 3.4 Move the rig; confirm `/odom` tracks consistently with motion and `odom→base_link` TF is broadcast
- [ ] 3.5 Inspect cross-camera timestamp sync; tighten the sync tolerance if frame sets are mismatched
- [ ] 3.6 Confirm the frustum-overlap auto-pairing connects the ring — verify cuVSLAM forms stereo links across each adjacent fisheye pair (not just isolated monos) so tracking has scale

## 4. Wrap-up

- [ ] 4.1 Capture a short bag of `/camN/image_raw` + `/odom` for regression/inspection
- [ ] 4.2 Record final status: mark #1 capture-rate + #2 end-to-end VO done (or log the failure mode)
