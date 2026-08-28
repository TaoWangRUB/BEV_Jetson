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
- [x] 3.3b **IMU node built: `bev_imu/imu_node` (C++).** Talks to `/dev/spidev1.0` and the GPIO character device directly — the sample loop must not do anything that can stall between the data-ready edge and the timestamp, and the timestamp is taken before the queue is drained or the SPI burst runs. Publishes `sensor_msgs/Imu` on `/imu0` stamped on `CLOCK_MONOTONIC`, the same clock as the cameras, so images and IMU land in one bag on one timebase.

  Verified live: WHO_AM_I 0x71, `/dev/gpiochip1` offset 42, falling edge (the J106 inverts the line), 200.24 Hz, **interval sd 7.6 µs, max 5184 µs, 0 dropped, 0 late reads over 4279 samples** — better than the reference `j106-imu-read.py` under `chrt -f 80` (sd 15.7 µs). DLPF group delays are logged and deliberately **not** applied (gyro lags accel by 1.02 ms), and Δ is logged UNMEASURED.

  Board specifics that are load-bearing in that file: INT is inverted with no pull-up → push-pull config and a *falling* edge; the chardev's own event stamp is `CLOCK_REALTIME` and is used only to wait; kernel 4.9 means the **v1** GPIO event ABI only. Needs `privileged` (device cgroup for `/dev/spidev1.0` + `/dev/gpiochip*`, `CAP_SYS_NICE` for `SCHED_FIFO`) — the compose `imu` service sets it.

- [ ] 3.3c Recording hygiene: stop `systemd-timesyncd` for the run (it cuts the frame-time fit residual 30.9 µs → 8.4 µs), and write the provenance alongside the bag — clock, trigger rate, exposure in force, Δ and its source — the way `j106-record-sync.py` writes `meta.json`. A recording with no provenance cannot be re-interpreted later.
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
