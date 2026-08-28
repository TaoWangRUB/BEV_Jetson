## 0. Board prerequisites (verify before writing code)

- [x] 0.1 **Confirmed live 2026-08-28** (`ssh tx2-eth` = `nvidia@10.42.0.157`, 4.9.337-tegra). Boot label is `j106fix` (newer than `j106imx296`). `/dev/video0..3` = `imx296 2-001a`, `2-0018`, `7-001a`, `7-0018` → bind order **equals** port order (c,d,e,f) *on this boot* — which is exactly what must not be assumed (0.1 of a different boot shifted it), so 1.1 still resolves at runtime.
- [x] 0.2 `trigger_mode=1` and frames flow (30 raw V4L2 frames off `/dev/video0`), so the STM32 **is** pulsing at **30.000 Hz**. Note: no `/dev/ttyACM*` on the board *or* the host, so the CDC control link is not currently connected — the trigger free-runs and its pulse width (= the exposure) cannot be queried or changed right now. Reconnect the CDC link before any exposure tuning.
- [~] 0.3 `jetson_clocks` **is** applied (all 6 CPUs pinned at 2035200 kHz) and `nvpmodel` is **MAXN**. The visual 4-up check via `csi_sender.sh`/`csi_receiver.sh` still wants a human at the host display — the numbers in 0.4 already cover liveness and rate.
- [x] 0.4 **Baseline measured** (`j106-sync-check.py -n 300`, 10 s, board copy at `/home/nvidia/tools/`):

  | node | frames | dropped | interval | jitter | rate | median skew | max skew | drift |
  |---|---|---|---|---|---|---|---|---|
  | video0 (c) | 300 | 0 | 33333.0 µs | 0.7 µs | 30.00 fps | — | — | — |
  | video1 (d) | 300 | 0 | 33333.0 µs | 0.7 µs | 30.00 fps | 0.0 µs | 1.0 µs | 0.01 µs/s |
  | video2 (e) | 300 | 0 | 33333.0 µs | 0.7 µs | 30.00 fps | 0.0 µs | 1.0 µs | 0.00 µs/s |
  | video3 (f) | 300 | 0 | 33333.0 µs | 0.7 µs | 30.00 fps | 0.0 µs | 1.0 µs | 0.00 µs/s |

  Verdict **SYNCHRONISED** — worst skew 1.0 µs over 10 s, worst drift 0.01 µs/s. That is 1/1000 of cuVSLAM's 1 ms gate, and ~30–86 ms better than the IMX219 rig the bundler was written for.

## 1. Capture node → IMX296

- [x] 1.1 Done. Verified live: `port c (imx296 2-001a) -> sensor-id 0 -> cam1` … `port f -> sensor-id 3 -> cam4`. `sensor_ids` override kept, and it warns that the mapping is then unverified.
- [x] 1.2 Done — throws `no camera on port(s): <list> — refusing to start a partially populated rig`.
- [x] 1.3 Done — defaults 1456×1088 @ 30 fps (the trigger's rate); running live at that geometry.
- [x] 1.4 Done — reads `/sys/module/imx296/parameters/trigger_mode`; logs `external trigger active -> locking AE (gain 16.0-16.0, dgain 4.0-4.0)`. `ae_lock=auto|on|off` overrides.
- [~] 1.5 Implemented (optional `calib_dir` param; compares `image_width/height` and a new `sensor:` key against the live rig, warns when the file does not state a sensor). **Not yet exercised** — it needs the IMX296 calibration from §2 to test against.
- [x] 1.6 Done — **and it found two real bugs, both invisible on the old rig:**

  1. *The published timestamp was not the sensor's.* The node stamped frames with `EGLStream::IFrame::getTime()`, which is the consumer-side frame time: measured live it put the cameras ~7 ms apart **in the order the capture loop visits them** (cam4 −7.0, cam1 0, cam2 +6.8, cam3 +13.8 ms) — it was reporting the loop's own phase. Fixed by taking the kernel SOF time from `ICaptureMetadata::getSensorTimestamp()`. Under 30–86 ms of free-running IMX219 skew this was undetectable; on a 1 µs rig it is the whole measurement. **The bundler and the fused node's `ts0` have been consuming these timestamps all along.**
  2. *Pairing frames by loop position is not a set.* Each camera's EGLStream queue advances independently, so one sweep can return frame k from one camera and k+1 from the next — reported as ~35 ms skew. Replaced with a nearest-frame matcher over an 8-deep per-camera history (same matching the VO needs in §4).

  With both fixed: **worst skew 1 µs per 5 s window, per-camera offsets 0 µs, 2 over-limit sets total (both at startup)** — the node's own measurement now agrees with the V4L2 baseline in 0.4.
- [~] 1.7 Built and run on the board (scratch workspace `/media/nvidia/workspace/bev_build_test`, `cuvslam-foxy:tx2`). Mapping and AE log as expected; all four topics publish. 30 s measurement (`scripts/port/luma_stability.py`):

  | topic | frames | rate | gaps | luma mean | p2p | sd |
  |---|---|---|---|---|---|---|
  | /cam1 | 593 | 30.00/s | 183 | 99.2 | 4.1 | 1.10 |
  | /cam2 | 576 | 30.00/s | 211 | 128.0 | 6.9 | 1.51 |
  | /cam3 | 625 | 30.00/s | 172 | 139.7 | 4.8 | 1.23 |
  | /cam4 | 821 | 30.00/s | 55 | 141.4 | 5.3 | 1.08 |

  Two things left open:
  - **Brightness**: the AE limit cycle is gone (171 % of mean → 3–5 %), but p2p 4.1–6.9 does not clear the spec's "< 5 levels" on all four. The residual is periodic and looks like **mains flicker** (50 Hz lighting beating with the 30 Hz trigger), which is a scene property, not an AE fault — needs confirming under daylight or DC light before either passing it or changing the threshold. Do **not** relax the spec until that is measured.
  - **Dropped frames**: 55–211 gaps per 30 s here, i.e. ~20 Hz effective — but **1.9 showed this is the measurement, not the capture.** With no subscriber the same node loses 0–3 frames in 30 s at a full 30 Hz; the loss is in delivery to a Python subscriber decoding four 1.58 MB streams. The modular path's memcpy+DDS cost is still real and is still why the fused node exists, but capture itself is not the bottleneck.

- [x] 1.8 Stamp the **exposure midpoint** (`SOF − exposure/2`), one rig-wide exposure for all four cameras, per README 4.7. Argus reports 0.521 ms here; under the trigger the true exposure is the pulse width, so `exposure_us` overrides it and the node warns until it is set.
- [x] 1.9 **Done — timing is recorded, with both sequence counters.** ROS 2 has no `header.seq`, so timing travels as its own message: `bev_camera/msg/FrameMeta` on `/camN/frame_meta`, same stamp as the image, carrying `sof_ns` (so the midpoint correction stays undoable), `exposure_ns`, and both `capture_id` (Argus session-side) and `frame_number` (consumer-side) — a gap in one but not the other localises the loss. `frame_log_dir` additionally writes `camN.csv` in the shape `j106-frametime.py` fits, under a provenance header (clock, convention, port, sensor, resolution, trigger state/rate, exposure source, Δ marked UNMEASURED).

  Verified live, and it immediately corrected 1.7: over 30 s with **no subscribers**, 859 frames per camera at 29.86–30.03/s with **0–3 lost, all in one startup gap** — the capture path is essentially lossless. In steady state `seq` advances exactly 1 per trigger edge (351/353 intervals), so it is a valid fit index once the startup transient is dropped. (Also fixed: the CSV now flushes per row — killing the node truncated the last line, which reads as corruption rather than as the loss it resembles.)
- [x] 1.10 **Trigger pulse width read — Argus was wrong by ~10×.** The MCU is not on USB CDC at all; it is on the M110 UART, `/dev/ttyTHS1`. `j106-trigctl.py --port /dev/ttyTHS1 status`: `period_us=33333`, `polarity=active_high`, `opto_skew_ns=0`, and **all four channels at `ch_exposure_us=5000`, `pulse_ns=4985740`**. Argus reported **0.521 ms** for the same frames, so stamping from Argus's exposure put every frame **2.2 ms** off the true midpoint — squarely in the range Kalibr resolves Δ to. `exposure_us:=4986` is now set in the compose `capture`/`modular` services. Re-read after any MCU reset: the firmware boots at compiled-in defaults silently.

## 2. Calibration prerequisites (host-side, before any rig time)

- [x] 2.1 Target recorded: `config/calib/april_6x6.yaml` — AprilGrid 6×6, **tagSize 0.055 m, tagSpacing 0.3** (5.5 cm tags, 1.65 cm gaps → 41.2 cm board). `tagSpacing` is a ratio, not a length.
- [ ] 2.2 **Measure the printed board** and correct `tagSize` if the print rescaled. This number sets the metric scale of every extrinsic and of the VIO; mount it flat (foam board), since a curl is a systematic error nothing downstream recovers.
- [~] 2.3 **tartancalib container**: building from `castacks/tartancalib` (`Dockerfile_ros1_20_04`, noetic-desktop-full + catkin, 16 cores). In progress.

  The empty dockerfiles in quarterKalibr turned out not to matter: it never builds anything, it *pulls* `mortyl0834/omnitartancalib:quad_cam` and `:kalibr` from Docker Hub (1.5 GB each, amd64, last pushed 2024-03). We are **not** using those — an unvetted personal account, mounted over our data — because upstream `castacks/tartancalib` ships the exact commands quarterKalibr invokes: `tartan_calibrate` with `--models omni-radtan` (`cameraModels` line 52) *and* `kalibr_calibrate_imu_camera`. Our own build covers both roles.
- [x] 2.4 **Not needed as a separate image.** tartancalib is a Kalibr fork and carries the full `kalibr_*` tool set, so one container serves the intrinsics, the pairwise stereo and the camera-IMU stages. (quarterKalibr uses two images only because its two published tags were built separately.)
- [ ] 2.8 **Decide what cuVSLAM is fed** — quarterKalibr calibrates `omni-radtan` (Mei), not KANNALA_BRANDT, and its stage 4 generates *virtual stereo* pairs. In the OmniNxt architecture cuVSLAM never sees a fisheye model at all: it consumes rectified virtual-pinhole stereo. That is the locked plan, but it changes §4/§5 from "4 fisheye cameras with KB intrinsics" to "N virtual stereo pairs", including the calibration files the VO loads. Settle this before the recording session, since it decides which outputs matter.
- [x] 2.5 **Extractor written: `scripts/calib/extract_quarterkalibr_bags.py`.** Reads our four separate topics (quarterKalibr's assumes one assembled image it splits into four), groups frames into sets by their own timestamps rather than arrival order, detects the AprilGrid per camera, and writes the eight staged bags under the names its notebook expects, plus `imu.bag` and the target yaml.

  It also recovers **the recording protocol, which is nowhere in quarterKalibr's README** — it is encoded in its `step_dict`. In one continuous recording, show the target to: cam1 alone, cam2 alone, cam3 alone, cam4 alone, then the four *adjacent* overlaps in order — cam4+cam1, cam1+cam2, cam2+cam3, cam3+cam4. Stages 1–4 give intrinsics, 5–8 give the pairwise extrinsics that compose around the ring. A stage is entered only when the set of cameras seeing tags matches the next expected pattern, so skipping one stalls everything after it. Our version also ignores unexpected patterns (three cameras seeing the target at once) instead of raising, which is what `step_dict.get(status) - current_step` does mid-recording.
- [x] 2.6 `config/calib/imu_mpu9250.yaml`: `/imu0` at 200 Hz, noise densities derived from the datasheet (gyro 0.01 °/s/√Hz → 1.75e-4 rad/s/√Hz; accel 300 µg/√Hz → 2.94e-3 m/s²/√Hz) and marked as datasheet-not-measured. The random walks are typical consumer-MEMS values and are flagged as the least trustworthy numbers in the file — an Allan-variance run (hours of static recording) only if Δ or the VIO proves sensitive.
- [ ] 2.7 Move the IMX219 calibration aside to `scripts/config/calib/imx219-1640x1232/`, labelled with the rig it belongs to, so a stale file cannot be picked up (D6). Note `scripts/config` is gitignored — new files need `git add -f`.

## 3. Calibrate, in stages (the quarterKalibr method — never a joint 4-camera solve)

- [ ] 3.1 Record on the board: AprilGrid through all four fields of view **including the adjacent overlaps**, then an IMU excitation sequence (rotation about all three axes, then translation). `ros2 bag` at ~4 Hz images + full-rate `/imu0`; stop `systemd-timesyncd` first (it cuts the frame-time fit residual 30.9 µs → 8.4 µs).
- [ ] 3.2 Write the recording's provenance beside the bag — clock, trigger rate and pulse width, exposure source, Δ and its source — the way `j106-record-sync.py` writes `meta.json`. A recording without it cannot be re-interpreted later.
- [ ] 3.3 Convert ROS2 → ROS1 on the host (`rosbags-convert`) and extract per-camera bags.
- [ ] 3.4 **Stage 1 — intrinsics per camera with tartancalib** (pinhole-equi / KB). Reject and re-shoot any camera over 1.0 px RMS reprojection.
- [ ] 3.5 **Stage 2 — extrinsics for adjacent PAIRS**, sequentially, then compose around the ring and check the loop-closure residual (spec: *Extrinsics are consistent around the rig*). A pair that will not solve is evidence about overlap, not just a failed run — it is the same question §5.3 asks of cuVSLAM's frustum gate.
- [ ] 3.6 **Stage 3 — camera↔IMU** for Δ. Record the value, the method, and its uncertainty; one constant for the whole rig, since a shared trigger edge leaves no per-camera component.
- [ ] 3.7 **Stage 4 — virtual stereo configs.** Not a bonus: this is the artifact the OmniNxt-frontend→cuVSLAM plan needs next.
- [ ] 3.8 Write the results: extrinsics into `config/rig/rig_extrinsics_vo.yaml` (frame convention, source recording, date, per-camera residuals) and Δ as a stated constant with provenance. Convert tartancalib's camchain into our `camN.yaml` (`fu,fv,pu,pv` + `k1..k4` → `mu,mv,u0,v0` + `k2..k5`), with `sensor: imx296` and the true image size.
- [ ] 3.9 **Settle the mounting orientation.** A live frame off port C shows the desk and cables at the *bottom* of the image, i.e. this rig reads as mounted **upright** — unlike the IMX219 rig, whose 180° roll is baked into `rig_extrinsics_vo.yaml` and the panorama's `flip_180`. Confirm before anything consumes the extrinsics; whatever the answer, it belongs *in* the extrinsics, not as a separate hidden roll.
- [ ] 3.10 Optional cross-check: solve one camera's intrinsics with the existing OpenCV checkerboard path (`intrinsic_calib.py`) and compare. Independent tooling on independent data is worth having if a tartancalib result looks odd — it is not on the critical path.

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
