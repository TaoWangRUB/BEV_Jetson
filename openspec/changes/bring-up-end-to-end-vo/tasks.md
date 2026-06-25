## 0. Environment / build (board: TX2 `/media/nvidia/workspace/BEV_Jetson`)

- [x] 0.1 Sync via git (push from dev → pull on TX2 `BEV_Jetson`); submodule git error fixed via patch refactor + `ignore = all`
- [x] 0.2 Build the ROS 2 workspace (`colcon build bev_camera bev_cuvslam`) in the `cuvslam-foxy:tx2` container — both packages link
- [x] 0.3 `build_and_validate.sh` on the patch-based build (lib + WarmUpGPU) — validated; patch step idempotent ("already applied"); workspace rebuilds clean

## 1. Capture frame-flow validation

- [x] 1.1 Run `argus_capture_node` @ 1640×1232 / 20 fps → "Argus capture up: 4 cameras", all 4 publish
- [x] 1.2 Measured sustained rate (fixed-window `topic_rate.py`): ~20 Hz right after launch (238 msgs/12 s), but found a **stall** under a reliable subscriber → fixed with best-effort `SensorDataQoS` (commit a66cc81). Post-fix: sustains, ~12–20 Hz (uneven per camera, CPU-bound single-thread)
- [x] 1.3 Recorded: capture is healthy; per-camera rates drift (cam1 ~12.6 vs cam4 ~15.2 Hz), which matters for sync (below)

## 2. Calibration path fix

- [x] 2.1 Locate the real intrinsics: `scripts/config/calib/cam{1..4}.yaml` (KANNALA_BRANDT, 1640×1232, tracked) — the old default `config/calib/1640x1232` did not exist
- [x] 2.2 Point the node + launch default `calib_dir` at `scripts/config/calib`; update docs
- [x] 2.3 Verify the 4-camera cuVSLAM `Rig` builds without file-not-found errors — confirmed on the board (2026-06-24): fused + modular nodes load calib + extrinsics, build the 4-cam Rig, `Multicamera` inits, no file-not-found

## 3. End-to-end capture → VO  ✅ WORKING (with the sync workaround)

- [x] 3.1 Topology: single container chosen (cross-container DDS discovery failed — topics invisible from a 2nd container even with `--network host`)
- [x] 3.2 VO + capture up together; VO loads calib, builds the 4-cam Rig, `Multicamera` inits, and `Track()` fires reliably via the **latest-frame bundler**
- [x] 3.3 **Output contract verified**: `/cuvslam/odometry` (~8.5 Hz, no "tracking lost") with pose **and covariance** (cuVSLAM cov now propagated, reordered to ROS layout — was all zeros), plus **`/tf` `odom→base_link`** confirmed (39 msgs/8 s). (Topic is `cuvslam/odometry`, not `/odom`.)
- [~] 3.4 Rig-motion tracking — **tracking confirmed live** (bench-stationary pose drifts ~1 cm, not frozen identity → cuVSLAM is processing). Full motion check still **needs physically moving the rig**: `./scripts/run_vo_tx2.sh` (capture+VO one container), then watch `/cuvslam/odometry` + TF
- [x] **3.5 Sync resolved (worked around).** Root cause: cuVSLAM `Multicamera` rejects sets >1 ms apart; the unsynced IMX219 rig is ~30–86 ms apart and drifts. Fix: (a) replaced ApproximateTime with a **latest-frame bundler** (driver cam triggers Track on newest others within `sync_slop_ms`, default 120 ms); (b) **unified per-set timestamp** so cuVSLAM accepts it. Residual: up to ~120 ms inter-camera skew = the rig's main accuracy limiter under motion. True fix remains hardware frame sync (or an async VIO).
- [ ] 3.6 Frustum-overlap auto-pairing — now testable; verify under motion that tracking is metric (stereo links form), not drifting mono

## 4. Wrap-up

- [x] 4.1 Bag path validated (`ros2 bag record` → 80 msgs/6.3 s, odom+tf+cam1; close with SIGINT so the sqlite WAL flushes). `scripts/run_vo_tx2.sh` with `RECORD=1` captures the lightweight VO output (`/cuvslam/odometry` + `/tf`); recording camera streams too throttles the pipeline (~20 MB/s SD writes → odom dropped to ~1.3 Hz)
- [x] 4.2 Final status recorded: capture-rate + end-to-end VO **done** (~8.5 Hz odom, tracking live); calibrated extrinsics applied to VO (roll folded). **Remaining (this change stays open): physical-motion confirmation — 3.4 / 3.6** (move the rig, record `/cuvslam/odometry`+`/tf`, check drift/metric tracking)
