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
- [x] 2.2 **Board measured and confirmed as printed** — 5.5 cm tags, 1.65 cm gaps, so `tagSize: 0.055`, `tagSpacing: 0.3` stand. Still mount it flat: a curl is a systematic error nothing downstream recovers.
- [x] 2.3 **tartancalib container built and verified** — `tartancalib:latest`, 5.89 GB, from `castacks/tartancalib` `Dockerfile_ros1_20_04` (noetic-desktop-full + catkin, ~10 min on 16 cores). `rosrun kalibr tartan_calibrate --help` runs and offers `pinhole-radtan, pinhole-equi, pinhole-fov, omni-none, omni-radtan, eucm-none, ds-none`; `kalibr_calibrate_imu_camera` is in the same image. Known gap: `kalibr_create_target_pdf` needs `pyx`, absent from the image — only needed to print a target, which we already have.

  The empty dockerfiles in quarterKalibr turned out not to matter: it never builds anything, it *pulls* `mortyl0834/omnitartancalib:quad_cam` and `:kalibr` from Docker Hub (1.5 GB each, amd64, 2024-03). We are **not** using those — an unvetted personal account, mounted over our data — because upstream ships the exact commands it invokes.
- [x] 2.4 **Not needed as a separate image.** tartancalib is a Kalibr fork and carries the full `kalibr_*` tool set, so one container serves the intrinsics, the pairwise stereo and the camera-IMU stages. (quarterKalibr uses two images only because its two published tags were built separately.)
- [x] 2.8 **Resolved by reading the cuVSLAM source — no decision needed before recording.** cuVSLAM accepts `Pinhole`, `Fisheye` (equidistant; *"compatible with ethz-asl/kalibr (pinhole-equi) and OpenCV::fisheye"*), `Brown`, `Polynomial` — and **no omni/EUCM/double-sphere**, so quarterKalibr's `omni-radtan` output is not loadable by our VO as it stands. Since `tartan_calibrate --models` takes one model per camera, both routes come from the same bags in two solver runs: `pinhole-equi` for direct `Fisheye`, `omni-radtan` for the virtual-stereo route. Decide from what tracks.
- [ ] 2.9 **Take the FOV number from the calibration.** cuVSLAM's `Fisheye` path is capped at **FOV < 180°** — it parameterises through `x/z` with θ = arctan(r), which cannot represent rays at or past 90° incidence (README 4.8). Under 180°, the direct route is legal; over it, virtual stereo is mandatory, not preferable. Ours are believed ~130°, but that is an assumption inherited from the IMX219 panorama config.

## 3. Calibrate, in stages (the quarterKalibr method — never a joint 4-camera solve)

- [~] 3.1 **Two capture paths, one per purpose.** First attempt ran everything on the board — capture, IMU, bag recorder, live preview with tag detection — and it does not fit: load hit 8 on six cores, the preview died, and the recorder crawled. Frames were never lost, but the operator was flying blind and the first stage came back 23 % usable with 22 of 36 image cells untouched, all at the periphery.

  So the work is split by what each machine is for:
  - **`scripts/stream/calib_sender.sh` + `calib_receiver.py`** (stages 1–8). Board: Argus → **hardware** JPEG (`nvjpegenc`) → MJPEG over TCP, and nothing else — load fell to ~0.3. Host: decode, detect, coverage grid, record. MJPEG rather than H.264 so no inter-frame artifact can smear a tag corner. **Carries no capture timestamp**, which is fine here: intrinsics and pairwise extrinsics use no time at all.
  - **`scripts/calib/record_calib_session.sh`** (ROS path) for the **camera↔IMU stage only**, where the offset *is* the measurement and the exposure-midpoint stamps of README 4.7 are the whole point.

- [x] 3.2 **Provenance written by the record script** — `meta.json` beside the bag: clock, both stamp conventions, trigger period and measured pulse width with its source, image resolution and decimation, the layout/target/noise files in force, `delta_camera_imu: UNMEASURED`, whether timesyncd was stopped, and the git commit.
- [x] 3.2d **Record everything at full rate; select frames offline.** Now that recording is host-side on NVMe, live decimation has no reason to exist: 4 cameras × 30 Hz × ~0.3 MB hardware JPEG ≈ 36 MB/s (288 Mbit/s), which 1 GbE carries and NVMe absorbs — a 10-minute session is ~21 GB. `select_frames.py` then picks the ~150 frames that earn their place, greedily by the coverage each ADDS (fisheye calibration lives or dies at the periphery), breaking ties on sharpness measured **on the target's bounding box**, not the whole frame. A selection step is required regardless — tartancalib on 9000 frames is an abandoned run, not a long one — so it is better to select from everything than from an arbitrary eighth. The 4 Hz figure was always a solver preference, never a transport limit.
- [x] 3.2b **Image decimation added** (`publish_every_n`, capture node). Four cameras at 30 Hz of 1456×1088 is ~190 MB/s, which the SD cannot absorb, and a bag left to drop frames drops them unevenly and silently. Decimating on the Argus frame *number* keeps whole synchronised sets (every camera skips the same trigger edges), and **timing metadata is still published for every frame**, so the frame-time fit and drop accounting see the full sequence. Verified live at 1/8 → 3.75 Hz.
- [x] 3.2c **`trigger_mode` not persisting is INTENDED — do not "fix" it.** The cameras free-run by default and external trigger is opt-in per session, so it must be set explicitly after every boot (`echo 1 > /sys/module/imx296/parameters/trigger_mode`); `jetson_clocks` is the same. No kernel cmdline, no boot service.

  What makes it dangerous is that forgetting is **silent**: the STM32 keeps pulsing, the cameras ignore it, nothing logs an error, and the only symptoms are AE gain-hunting (luma p2p 60–97 against 0.5 locked) and sets that are not sets. Hit exactly that mid-session after a reboot here. `record_calib_session.sh`'s preflight is the guard: it refuses to record unless `trigger_mode` is 1 and the generator reports running, and re-applies `jetson_clocks` if a reboot dropped it.
- [ ] 3.3 Convert ROS2 → ROS1 on the host (`rosbags-convert`) and extract per-camera bags.
- [x] 3.4 **Stage 1 done — intrinsics for all four, omni-radtan (Mei), 0.28–0.40 px.** `pinhole-equi` was attempted first and diverged on every camera: the lens is 1.78 mm D190/H160 on 1/3", i.e. a genuine >180° fisheye (vendor spec, our ~192° fits, and OmniNxt's hard-coded fov=190 all agree), and the equidistant model cannot represent rays past 90° incidence. Results in `config/calib/imx296_1456x1088/`.
- [x] 3.5 **Stage 2 done — all four pairs solved.** Baselines 147.7/148.7/149.2/149.2 mm agree to **1.5 mm** across independent recordings; every hop within ~2° of 90°. Ring closes at **4.75° rotation, 4.9 mm translation** over four hops — translation excellent, rotation the weak part (right pair 92.2°, fewest poses). Solved with Kalibr's own detector and `T_t_c`; a hand-rolled board model plateaued at 11 px residual and a metre of baseline error.
- [x] 3.6 **Δ MEASURED: −8.06 ms** (`timeshift_cam_imu`), with T_cam_imu, from 911 images at 9.4 Hz + 19 717 IMU samples at 204 Hz. Residuals 0.366 px / 0.00156 rad s⁻¹ / 0.0424 m s⁻². The IMU had to be rebuilt from the node's CSV — the bag held only 43 Hz because rosbag2's single writer could not carry full-rate images and 200 Hz IMU together. ⚠ −8.06 ms is larger than our stamping discipline should leave; the DLPF group delays (2.9/1.88 ms, unapplied) and half the 16.1 ms readout (8.05 ms) are both candidates and should be resolved before trusting VIO scale.
- [ ] 3.7 **Stage 4 — virtual stereo configs.** Not a bonus: this is the artifact the OmniNxt-frontend→cuVSLAM plan needs next.
- [ ] 3.8 Write the results: extrinsics into `config/rig/rig_extrinsics_vo.yaml` (frame convention, source recording, date, per-camera residuals) and Δ as a stated constant with provenance. Convert tartancalib's camchain into our `camN.yaml` (`fu,fv,pu,pv` + `k1..k4` → `mu,mv,u0,v0` + `k2..k5`), with `sensor: imx296` and the true image size.
- [x] 3.9 **Orientation settled: the 180° roll IS needed** — confirmed on the hardware. My inference from a live frame (desk and cables at the bottom ⇒ upright) was **wrong**; the modules are mounted inverted, as the IMX219 rig was. So `flip_180` and the roll folded into `rig_extrinsics_vo.yaml` stay. Recorded in `config/rig/rig_layout.yaml` together with the camera positions and the measured IMU axes.
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
