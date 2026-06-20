## 0. Environment / build (board: TX2 `/media/nvidia/workspace/BEV_Jetson`)

- [x] 0.1 Sync via git (push from dev → pull on TX2 `BEV_Jetson`); submodule git error fixed via patch refactor + `ignore = all`
- [x] 0.2 Build the ROS 2 workspace (`colcon build bev_camera bev_cuvslam`) in the `cuvslam-foxy:tx2` container — both packages link
- [~] 0.3 `build_and_validate.sh` on the patch-based build (lib + WarmUpGPU) — running; patch step confirmed "already applied"

## 1. Capture frame-flow validation

- [x] 1.1 Run `argus_capture_node` @ 1640×1232 / 20 fps → "Argus capture up: 4 cameras", all 4 publish
- [x] 1.2 Measured sustained rate (fixed-window `topic_rate.py`): ~20 Hz right after launch (238 msgs/12 s), but found a **stall** under a reliable subscriber → fixed with best-effort `SensorDataQoS` (commit a66cc81). Post-fix: sustains, ~12–20 Hz (uneven per camera, CPU-bound single-thread)
- [x] 1.3 Recorded: capture is healthy; per-camera rates drift (cam1 ~12.6 vs cam4 ~15.2 Hz), which matters for sync (below)

## 2. Calibration path fix

- [x] 2.1 Locate the real intrinsics: `scripts/config/calib/cam{1..4}.yaml` (KANNALA_BRANDT, 1640×1232, tracked) — the old default `config/calib/1640x1232` did not exist
- [x] 2.2 Point the node + launch default `calib_dir` at `scripts/config/calib`; update docs
- [ ] 2.3 Verify the 4-camera cuVSLAM `Rig` builds without file-not-found errors (needs VO node board run)

## 3. End-to-end capture → VO  ⛔ BLOCKED on camera sync (see 3.5)

- [x] 3.1 Topology: single container chosen (cross-container DDS discovery failed — topics invisible from a 2nd container even with `--network host`)
- [x] 3.2 Brought VO + capture up together. VO loads calib, builds the 4-cam Rig, `Multicamera` mode inits. `Track()` reached only with `sync_slop_ms` ≥ ~80 ms
- [ ] 3.3 `/cuvslam/odometry` (note: topic is `cuvslam/odometry`, not `/odom`) — **0 messages**: see blocker
- [ ] 3.4 Rig-motion tracking — blocked
- [ ] **3.5 BLOCKER — no hardware camera sync.** cuVSLAM `Multicamera` **hard-rejects** sets whose per-camera timestamps differ by >1 ms (`Track() failed: Timestamps differ by more than 1.000000 ms`). The IMX219 rig free-runs with **no HW trigger**; measured 4-cam spread **~30–66 ms** (one frame period), with cameras drifting at *different* rates. Workarounds tried: unified per-set timestamp (commit 4061737) clears the 1 ms check, and `sync_slop_ms` 80→150. Even so, ApproximateTime can't reliably form 4-way sets from 4 unsynchronized, drifting, best-effort streams → `Track()` rarely/never called → no odometry. **Needs a decision: hardware frame sync, or switch to an async-camera VIO (e.g. OpenVINS).**
- [ ] 3.6 Frustum-overlap auto-pairing — untestable until tracking runs

## 4. Wrap-up

- [ ] 4.1 Capture a short bag of `/camN/image_raw` + `/odom` for regression/inspection
- [ ] 4.2 Record final status: mark #1 capture-rate + #2 end-to-end VO done (or log the failure mode)
