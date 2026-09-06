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
- [x] 2.9 **Answered by 3.4: the lenses are over 180°, so virtual stereo is mandatory.** cuVSLAM's `Fisheye` path is capped at **FOV < 180°** — it parameterises through `x/z` with θ = arctan(r), which cannot represent rays at or past 90° incidence (README 4.8). Under 180°, the direct route is legal; over it, virtual stereo is mandatory, not preferable. The ~130° inherited from the IMX219 panorama config was wrong: the fits give **~192° diagonal**,
  and the vendor spec (D190/H160 on 1/3") agrees. Over 180°, so the direct `Fisheye` route is not
  legal and §3.7's virtual stereo is required, not preferred.

## 3. Calibrate, in stages (the quarterKalibr method — never a joint 4-camera solve)

> **Superseded 2026-08-31 for the hardware now on the rig.** The cameras were reconnected, the
> lenses refocused and the trigger source changed to an F401, so 3.4/3.5/3.5b/3.6/3.7 describe a
> rig that no longer exists. They are kept as the method of record and as the prior every new
> number is cross-checked against — see **§3R**, which re-runs them. Nothing downstream may be
> run against these results.

- [~] 3.1 **Two capture paths, one per purpose.** First attempt ran everything on the board — capture, IMU, bag recorder, live preview with tag detection — and it does not fit: load hit 8 on six cores, the preview died, and the recorder crawled. Frames were never lost, but the operator was flying blind and the first stage came back 23 % usable with 22 of 36 image cells untouched, all at the periphery.

  So the work is split by what each machine is for:
  - **`scripts/stream/calib_sender.sh` + `calib_receiver.py`** (stages 1–8). Board: Argus → **hardware** JPEG (`nvjpegenc`) → MJPEG over TCP, and nothing else — load fell to ~0.3. Host: decode, detect, coverage grid, record. MJPEG rather than H.264 so no inter-frame artifact can smear a tag corner. **Carries no capture timestamp**, which is fine here: intrinsics and pairwise extrinsics use no time at all.
  - **`scripts/calib/record_calib_session.sh`** (ROS path) for the **camera↔IMU stage only**, where the offset *is* the measurement and the exposure-midpoint stamps of README 4.7 are the whole point.

- [x] 3.2 **Provenance written by the record script** — `meta.json` beside the bag: clock, both stamp conventions, trigger period and measured pulse width with its source, image resolution and decimation, the layout/target/noise files in force, `delta_camera_imu: UNMEASURED`, whether timesyncd was stopped, and the git commit.
- [x] 3.2d **Record everything at full rate; select frames offline.** Now that recording is host-side on NVMe, live decimation has no reason to exist: 4 cameras × 30 Hz × ~0.3 MB hardware JPEG ≈ 36 MB/s (288 Mbit/s), which 1 GbE carries and NVMe absorbs — a 10-minute session is ~21 GB. `select_frames.py` then picks the ~150 frames that earn their place, greedily by the coverage each ADDS (fisheye calibration lives or dies at the periphery), breaking ties on sharpness measured **on the target's bounding box**, not the whole frame. A selection step is required regardless — tartancalib on 9000 frames is an abandoned run, not a long one — so it is better to select from everything than from an arbitrary eighth. The 4 Hz figure was always a solver preference, never a transport limit.
- [x] 3.2b **Image decimation added** (`publish_every_n`, capture node). Four cameras at 30 Hz of 1456×1088 is ~190 MB/s, which the SD cannot absorb, and a bag left to drop frames drops them unevenly and silently. Decimating on the Argus frame *number* keeps whole synchronised sets (every camera skips the same trigger edges), and **timing metadata is still published for every frame**, so the frame-time fit and drop accounting see the full sequence. Verified live at 1/8 → 3.75 Hz.
- [x] 3.2c **`trigger_mode` not persisting is INTENDED — do not "fix" it.** The cameras free-run by default and external trigger is opt-in per session, so it must be set explicitly after every boot (`echo 1 > /sys/module/imx296/parameters/trigger_mode`); `jetson_clocks` is the same. No kernel cmdline, no boot service.

  What makes it dangerous is that forgetting is **silent**: the STM32 keeps pulsing, the cameras ignore it, nothing logs an error, and the only symptoms are AE gain-hunting (luma p2p 60–97 against 0.5 locked) and sets that are not sets. Hit exactly that mid-session after a reboot here. `record_calib_session.sh`'s preflight is the guard: it refuses to record unless `trigger_mode` is 1 and the generator reports running, and re-applies `jetson_clocks` if a reboot dropped it.
- [x] 3.3 **Done** — `rosbags-convert`; per-stage bags in `datasets/calib_20260828/ros1/`. Convert ROS2 → ROS1 on the host (`rosbags-convert`) and extract per-camera bags.
- [x] 3.4 **Stage 1 done — intrinsics for all four, omni-radtan (Mei), 0.28–0.40 px.** `pinhole-equi` was attempted first and diverged on every camera: the lens is 1.78 mm D190/H160 on 1/3", i.e. a genuine >180° fisheye (vendor spec, our ~192° fits, and OmniNxt's hard-coded fov=190 all agree), and the equidistant model cannot represent rays past 90° incidence. Results in `config/calib/imx296_1456x1088/`.
- [x] 3.5 **Stage 2 done — all four pairs solved.** Baselines 147.8/148.7/149.1/149.1 mm agree to **1.3 mm** across independent recordings; every hop within ~2° of 90°. Solved with Kalibr's own detector and `T_t_c`; a hand-rolled board model plateaued at 11 px residual and a metre of baseline error. Ring closure was **3.63° rotation, 9.2 mm translation** (an earlier 4.75°/4.9 mm was a different composition base point; the rotation is base-point invariant, the translation is not — which is itself the evidence in 3.5b).
- [x] 3.5b **Ring closed — the four pairs now describe one rigid body.** `scripts/calib/close_rig_ring.py`: three camera poses in cam1's frame (18 dof) instead of four free edges (24), Levenberg-Marquardt. Residual **3.63° / 9.2 mm → 0.0000° / 0.043 mm**; per-edge corrections 0.57–1.31°.

  **It bought consistency, not accuracy, and that was established before running it.** A Monte-Carlo over each pair's own reported spread says random per-pose error would leave only ~0.5° of loop residual — so the 3.63° is systematic bias inside each recording, averaging is already saturated, and more frames in these recordings cannot help. Measured cost, per pair, epipolar median before → after: left 0.81 → 0.45, front 1.42 → 1.79, right 0.62 → 2.55, rear 0.83 → 0.66; mean rms 2.94 → 3.06 px. Rigidity costs ~4 % in rms and moves error between pairs — worth it against 3.63° (35 px at the image edge), but a trade.

  Weighting by measured epipolar residual instead of internal angular spread was tried and is worse (mean rms 3.45). Kept as an option, not the default.

  Independent confirmation: the angle between facing virtual optical axes was 1.1/1.3/**2.1**/1.3° and is now 1.1/1.4/**1.1**/1.0°. The right pair's outlier was the inconsistency, not the geometry — and nothing in the closure targeted that number.

  **Still open:** the ~1° of systematic per-recording bias. It needs a recording where **three or more cameras see the board simultaneously**, which makes the bias observable instead of absorbed. Also worth calipers on the printed board — a print-scale error is common-mode across all four recordings, invisible to every consistency check run so far, and scales VO output directly.
- [x] 3.6 **Δ MEASURED: −8.06 ms**, and verified as a genuine timestamp offset — shifting every camera stamp by a known +10.000 ms moved the estimate by −10.000 ms (residual −1.2 µs). Kalibr fits a continuous spline, so the 200 Hz IMU rate does not quantise it. Solved from 911 images at 9.40 Hz and 19 717 IMU samples at **199.50 Hz** (an earlier "204 Hz" was my arithmetic — IMU count over the *image* span). Residuals 0.366 px / 0.00156 rad s⁻¹ / 0.0424 m s⁻².
- [x] 3.6b **ANSWERED 2026-09-01 — the offset is the IMU's DLPF group delay, and no GPIO echo was
  needed for the conclusion.** Two independent measurements, each with the prediction stated first:

  1. **Delta vs exposure** (3R.14b): a mis-modelled exposure term predicted a 2.5 ms shift over a
     5 ms exposure change. Measured **0.13 ms**. The camera-side stamping is correct.
  2. **Delta vs gyro DLPF bandwidth**: the MPU-9250's group delay is *reported and deliberately not
     applied*, so it should land inside Delta and move with the bandwidth.

     | gyro DLPF | datasheet group delay | Delta measured |
     |---|---|---|
     | 184 Hz (index 1) | 2.90 ms | **+3.7556 ms** |
     | 41 Hz (index 3) | 5.90 ms | **+6.5479 ms** |
     | change | **+3.00 ms** | **+2.792 ms** |

     Delta tracked the group delay to **93 %** of the predicted slope. Mechanism confirmed.

  **The trigger echo was wired anyway (route A, open-drain), and it is what makes the accounting
  complete** rather than merely consistent. `PA5` on the F401 drives `TIM2_CH1` open-drain into
  `gpio-389` (`GPIO_PQ5_PI5`, M110 J21 pin 8, 1.8 V unbuffered), with the pull-up supplied by a DTB
  change (`LABEL j106echo`) so nothing drives that pad above 1.8 V — no divider, no resistor.
  `scripts/trig-echo-stamp.py` timestamps the falling edge, `trig-echo-delta.py` compares it with
  the frame log:

      e = t_sof - t_edge - exposure     (0 if SOF marks the end of exposure)
      5 ms  ->  +0.3112 ms
      10 ms ->  +0.3097 ms
      20 ms ->  +0.3109 ms

  **A constant +0.31 ms, varying by 1.5 us over a 4x exposure change.** So SOF sits a fixed 0.31 ms
  after end-of-exposure and the camera contributes only `w - e` = **-0.26 ms** to Delta. Transport
  was excluded separately: the camera's messages arrive ~55 ms later relative to their stamps than
  the IMU's, but both are stamped at source, so DDS latency cannot reach Delta — and if it could,
  the signature would be 55 ms, not 4.

  Residual worth stating: 0.05 - 0.31 + 2.90 = **2.64 ms** predicted against **3.76 ms** measured,
  so ~1.1 ms is still unaccounted for. The *slope* is what confirms the mechanism; the remaining
  offset is plausibly the accel path (1.88 ms, unchanged between the two runs) diluting a fit that
  uses both. Not chased further - it is inside Delta, which is applied as one measured constant.
- [x] 3.7 **Stage 4 done — four virtual stereo pairs generated, measured, and rebuilt on the closed rig.** Each fisheye carved into two virtual pinholes at ±45°; the facing pair is *derived* from the extrinsic (optical axes 1.0–1.4° apart) rather than assumed. Config in `config/rig/virtual_stereo_imx296.yaml`, generator + checker + disparity in `scripts/calib/`.

  **768×576, fov 70° (from the horizontal 160°), focal 548.4 px.** The first attempt carved by the *diagonal* 190°, which asks each pinhole to reach 95° off-axis where the lens delivers 80 — every rectified view came back with a black wedge (90 % non-black vs 100 % now). The lens is D190/**H160** and the split is by yaw, so the horizontal field governs.

  | pair | median \|dy\| | rms | dense valid | depth |
  |---|---|---|---|---|
  | left | **0.45 px** | 3.15 | 14 % | 0.44 m |
  | front | **1.79 px** | 3.15 | 19 % | 0.32 m |
  | right | **2.55 px** | 4.07 | 14 % | 0.39 m |
  | rear | **0.66 px** | 1.88 | 22 % | 0.40 m |

  Usable, not excellent — a good physical stereo rig reaches under 0.5 px, so disparity search needs a few pixels of vertical tolerance. Disparity sign is 100 % consistent on every pair.

  **Dense disparity here is not evidence either way.** These are calibration sweeps: a repetitive tag grid at 0.3–0.6 m against blank wall and floor, close to the worst case for block matching. Widening the search range made it worse, so texture is the limit, not geometry. Ordering and depth were correct where it matched. A real verdict needs a textured scene at 1–3 m, which has not been recorded.

  Three measurement lessons: **resolution is not free** — at 480×360 the virtual focal is 0.38× the fisheye's near-axis scale and tag detection collapsed from ~10/frame to 0.1. **ORB-based epipolar measurement is worthless here** — 4.15 px median / 74 px p90 on a repetitive scene, measuring the matcher rather than the rig; tag identity removes the ambiguity. And **a closed loop is not evidence of a correct answer** — plain Gauss-Newton drove the ring residual to zero while walking the rig to 70–115° per edge, which is why 3.5b prints per-edge corrections.

- [ ] 3.8 Write the results: extrinsics into `config/rig/rig_extrinsics_vo.yaml` (frame convention, source recording, date, per-camera residuals) and Δ as a stated constant with provenance. Convert tartancalib's camchain into our `camN.yaml` (`fu,fv,pu,pv` + `k1..k4` → `mu,mv,u0,v0` + `k2..k5`), with `sensor: imx296` and the true image size.
- [x] 3.9 **Orientation settled: the 180° roll IS needed** — confirmed on the hardware. My inference from a live frame (desk and cables at the bottom ⇒ upright) was **wrong**; the modules are mounted inverted, as the IMX219 rig was. So `flip_180` and the roll folded into `rig_extrinsics_vo.yaml` stay. Recorded in `config/rig/rig_layout.yaml` together with the camera positions and the measured IMU axes.
- [ ] 3.10 Optional cross-check: solve one camera's intrinsics with the existing OpenCV checkerboard path (`intrinsic_calib.py`) and compare. Independent tooling on independent data is worth having if a tartancalib result looks odd — it is not on the critical path.

## 3R. Recalibration after the 2026-08-31 hardware change

The rig changed under the calibration: **cameras were reconnected, the lenses were refocused, and
the trigger source is now an F401.** Every number in §3 was measured on the previous state, so §3
above is retained as the *method of record and the prior* — not as the calibration in force. Nothing
in §4.5 or §5 may be run against it.

What each change invalidates, and why:

| change | invalidates | why |
|---|---|---|
| lenses refocused | **all four intrinsics** (3.4), and through them the pairwise extrinsics (3.5), the closure (3.5b) and the virtual-stereo carve (3.7) | focus moves focal length, principal point and the distortion curve together; on an M12 barrel it also rotates the lens, so the distortion centre and the optical axis both shift |
| cameras reconnected | the **port ↔ module ↔ camN** identity, and the mount pose if a module was disturbed | intrinsics belong to a *module*, extrinsics to a *mounting position*. A module that moved ports carries its intrinsics with it; a mount that was nudged does not |
| trigger source → F401 | the exposure constant (1.10), the frame-time provenance (1.9), and **Δ** (3.6) | the stamp is `SOF − exposure/2` and the exposure *is* the trigger pulse width. A different pulse width moves every stamp by half the difference, and Δ absorbs exactly that |

Sync itself (0.4) and the capture node (§1) are **not** assumed broken — but 3R.2 re-measures rather
than assumes, because a trigger swap is precisely the event that can degrade skew silently.

### A. Re-establish identity and the trigger (board, ~1 h)

- [x] 3R.1 **Port → position → camN re-verified 2026-08-31, from the images themselves.**
  `c = front-left (cam1), d = front-right (cam2), e = back-left (cam3), f = back-right (cam4)`,
  as the operator states and as `config/rig/rig_layout.yaml` already had it. Kernel bind order also
  unchanged (`/dev/video0..3` = `imx296 2-001a, 2-0018, 7-001a, 7-0018`), though 1.1 still resolves
  it at runtime rather than trusting that.

  **Verified by overlap, not by boot order**, which is what a reconnect can silently break: one raw
  frame from each port, debayered and rotated 180°. The nappy box sits at cam1's **right** edge and
  cam2's **left** edge (a front-left/front-right pair shares exactly that region); the rover wheel and
  can sit at cam1's **left** and cam3's **right** (front-left/back-left share that one). So the ring
  order `c → d → f → e` holds and no ribbon pair was swapped — which is the failure that matters,
  because a swap would have the pairwise stage solving a **diagonal** pair as a stereo baseline.

  **The 180° roll survives the reconnect**, and it was checked the unambiguous way — printed text.
  As captured, the nappy box reads upside-down and the floor sits at the *top* of the frame; both
  come right only after `ROTATE_180`. So 3.9's `camera_roll_deg: 180` and `flip_180` stay. (Rendered
  both ways deliberately: 3.9 records that this same inference was made from a live frame once
  before and was **wrong**, so a judgement about "which way up does this look" is not good enough —
  text is.)

  **Module identity is NOT established, and per the operator the module on port c is probably not
  the one that was there before** — only the port sequence c/d/f/e is unchanged. That is fine and
  changes nothing about the plan: every intrinsic is re-measured per port, so whatever sits on port c
  simply becomes cam1. The one consequence is that **the old `camN.yaml` cannot be attributed to the
  new `camN`**, which guts 3R.7.
- [x] 3R.2 **Sync baseline re-measured on the F401 — PASS, and identical to the H7.**
  `j106-sync-check.py -n 300`, 10 s, 2026-08-31:

  | node | frames | dropped | interval | jitter (sd) | rate | median skew | max skew | drift |
  |---|---|---|---|---|---|---|---|---|
  | video0 (c) | 300 | 0 | 33330.7 µs | 5.7 µs | 30.00 fps | — | — | — |
  | video1 (d) | 300 | 0 | 33330.7 µs | 5.7 µs | 30.00 fps | 0.0 µs | 1.0 µs | −0.01 µs/s |
  | video2 (e) | 300 | 0 | 33330.7 µs | 5.7 µs | 30.00 fps | 0.0 µs | 1.0 µs | −0.01 µs/s |
  | video3 (f) | 300 | 0 | 33330.7 µs | 5.7 µs | 30.00 fps | 0.0 µs | 1.0 µs | −0.00 µs/s |

  Verdict **SYNCHRONISED** — worst skew 1.0 µs, worst drift 0.01 µs/s, exactly 0.4's numbers. All
  four channels are compare outputs on one 32-bit counter (TIM2 here, TIM5 on the H7), so the edges
  are the same hardware event and the skew is a property of that sharing, not of the part.

  Two differences from the H7, both benign and both worth having on record:
  - **Jitter 5.7 µs sd against the H7's 0.7 µs.** Still 1/175 of the 1 ms gate. Unexplained; the
    F401's timer tick is 11.9 ns against 8.33 ns, which does not account for it.
  - **Mean interval 33330.7 µs against a commanded 33333.3**, i.e. the frames arrive ~78 ppm fast
    *in TX2 CLOCK_MONOTONIC*. That is the difference between the F401's 25 MHz crystal and the TX2's
    oscillator, and it is harmless here because the stamps are taken by the TX2 kernel at SOF, not
    by the MCU. Do not "correct" it.
- [x] 3R.3 **F401 read out, and it is not a like-for-like replacement.** It enumerates as USB CDC
  (`0483:5740`, descriptor string `STM32F407`) on **`/dev/ttyACM0`** — the H7 was reachable only on
  the M110 UART `/dev/ttyTHS1`, so `TRIGCTL`/`TRIG_PORT` must change wherever they are hard-coded
  (`scripts/calib/record_calib_session.sh` defaults to `/dev/ttyTHS1`). `j106-trigctl.py` speaks to
  both unchanged. Live status 2026-08-31:

  ```
  clock=hse25-pll84  timer_hz=84000000  running=1
  period_us=33333    fps_milli=30000    polarity=active_low (idle LED ON)
  ch1..4_exposure_us=30000  pulse_ns=29985740  ccr=2518802   opto_skew_ns=0
  ```

  `clock=hse25-pll84` confirms the crystal started rather than the ~1 %-accurate HSI fallback, which
  would land directly on the frame rate. **Two settings differ from the H7 and both matter:**

  1. **Polarity is now `active_low`**, where the H7 reported `active_high`. The label is not evidence
     of what the sensor sees, so it was measured: at a fixed scene and gain, cam1's mean raw level
     went **1001.2 → 1108.1 → 1270.7** for commanded **5000 → 15000 → 30000 µs** — incremental slope
     **0.0107 and 0.0108 counts/µs**, agreeing to 1 %, on a ~736-count black level. So **the
     commanded exposure is the asserted exposure**; `active_low` inverts the pin, not the meaning,
     and the complement hypothesis (period − pulse) is dead — it predicts brightness moving the
     other way. (The straight line extrapolates to ~947 at E = 0, about 210 counts above the
     measured black level. Unexplained, not load-bearing for the polarity question, and not worth
     chasing before the recording.)
  2. **Exposure is 30000 µs, not the 4986 µs task 1.10 pinned.** That is a 90 % duty cycle, and it
     moves the exposure-midpoint stamp by **12.5 ms** — 1550× the Δ we are trying to resolve. Every
     `exposure_us` in the compose services is now wrong, and so is Δ. See 3R.4a for the choice this
     forces and 3R.14b for the experiment it enables.

  Also confirmed, incidentally: the v4l2 `exposure` control reads **520 µs** while the trigger holds
  the line for 30000 — which is task 1.10's finding from the other side. The sensor's own exposure
  register is inert under Fast Trigger, and anything reading it (Argus did) is reading a fiction.
- [~] 3R.4 **Brightness measured raw; the ISP-path check still owed.**
  `scripts/port/luma_stability.py` (30 s, four cameras, ISP path, Argus gain locked 16.0/4.0) is
  still the check that counts and needs 3R.4a settled first. What the raw v4l2 probe already shows,
  2026-08-31, room lit largely by daylight:

  - **Two of four cameras clip at 30 ms.** cam2 and cam4 face windows and sit at full scale
    (16383, i.e. p99.5 = max) ; cam1 and cam3 have **zero** clipped pixels. Blown highlights near the
    target destroy corner detection, so keep the AprilGrid away from the windows or bring the
    exposure down.
  - **No mains flicker detectable.** 60 frames at a deliberately short 2000 µs (a long exposure
    integrates the ripple away, which is the point): after a 5-frame startup step the per-frame mean
    is flat to **±0.8 counts out of ~1000**, with **no 3-frame beat**. At 30 fps a 50 Hz mains ripple
    (100 Hz) aliases to 10 Hz and would show as exactly that beat, so **50 Hz flicker is ruled out**.
    A 60 Hz ripple (120 Hz) aliases to ~0 Hz and would be *invisible* to this test, so this does not
    prove the absence of flicker in general — it proves there is none to cancel in this room, in
    daylight. **Re-run at night under artificial light** before treating 1.7's p2p question as
    closed, and note that this weakens the flicker argument for a long exposure (see 3R.4a).

- [x] 3R.4a **Choose ONE operating exposure, and use it for the calibration, for Δ, and for the VO.**
  This is now a decision, not a default: the rig was found at 30000 µs and the previous calibration
  ran at 4986 µs.

  Why it cannot be deferred: the stamp is `SOF − exposure/2`, so Δ is measured *at* an exposure.
  Calibrating Δ at 30 ms and then flying at 5 ms puts every camera stamp **12.5 ms** out — against a
  Δ of −8.06 ms and a sync budget of 1 µs. The two ends of the range trade against each other and
  neither is free:

  | | 30000 µs (as found) | ~5000 µs (the H7 setting) |
  |---|---|---|
  | duty cycle | 90 % of the frame period | 15 % |
  | motion blur | severe under any rig motion — and §5 is a *motion* test | negligible |
  | signal (cam1 raw mean, black ≈ 736) | 1270.7 | 1001.2 — the room measured genuinely dark |
  | highlights | **cam2 and cam4 clip at full scale** on the windows | no clipping anywhere |
  | readout margin | 3.3 ms, just inside the firmware's 1 ms refusal | ample |

  A dark room is the honest argument for 30 ms and is why it was set. Two measurements weaken it:
  the extra light is bought at the cost of **clipping two of the four cameras**, and there is **no
  50 Hz flicker to integrate away** (3R.4), so the one argument that would have favoured a long
  exposure on principle does not apply here. Against it stands a 90 % duty cycle in a change whose
  §5 is a *motion* test, and blur is not something a calibration undoes afterwards.

  **Prefer more light or more gain over more exposure.** If flicker does turn up at night, the
  exposures that cancel it are integer multiples of a mains half-cycle — 10 or 20 ms at 50 Hz,
  8.33 / 16.67 / 25 ms at 60 Hz — which is a better way to buy that property than 30 ms. Settle it
  before 3R.9: changing it mid-session changes blur and the frame-time model at once, and the two
  are then inseparable.

  Whatever is chosen: set it explicitly at the start of every session. **Settings do not survive a
  power cycle** — the F401 boots at 30.000 fps / 5000 µs / `pol 1`, and the current live state is
  *not* those defaults, so something already re-applies them by hand. `record_calib_session.sh`
  reads the generator rather than assuming, which is the behaviour to keep.

  **SETTLED 2026-08-31: 5000 µs**, chosen by the operator to eliminate blur, and set live
  (`ch1..4_exposure_us=5000  pulse_ns=4985740`). So **`EXPOSURE_US=4986`** — the measured pulse
  width, not the commanded 5000 — everywhere the stamp is computed, and the camera↔IMU stage and
  the VO both run at this exposure.

  The room is genuinely dark at 5 ms (cam1 raw mean 1001 on a 736 black level, i.e. ~265 counts of
  signal). **Add light before 3R.9** rather than winding the exposure back up: that is the trade this
  task exists to prevent making twice.

### B. Settle the optics before spending a session on them (~30 min)

- [ ] 3R.5 **Verify focus at the VO's working distance, then lock the barrels.** Sharpness at the
  centre *and* at four field points, on a textured target at **1.5–3 m** — not at the 0.3–0.6 m the
  calibration board sits at, or the rig ends up focused for the calibration rather than for the job.
  Then thread-lock. This is the point of no return and it is deliberately placed *before* the
  recording: any focus touch after it invalidates everything that follows, which is exactly how we
  got here.
- [ ] 3R.6 **Calipers on the printed AprilGrid** (the open item from 3.5b). Measure five tag pitches
  and divide. A print-scale error is common-mode across all four recordings, invisible to ring
  closure and to every consistency check run so far, and it scales VO translation output *directly* —
  the one error that a metric-scale claim cannot survive. Correct `tagSize` in
  `config/calib/april_6x6.yaml` if it disagrees with 0.055 m by more than ~0.3 %.
- [ ] 3R.7 *(optional, and much weakened)* **Reproject frames through the OLD `camN.yaml`.**
  As planned this had two purposes and 3R.1 removed both: it cannot measure "how far the refocus
  moved camera N's optics", because the module on port N is probably not the one those intrinsics
  were solved for; and it is no longer needed to detect a swap, since a swap is assumed. What remains
  is one gross sanity number for the writeup — *the old calibration no longer describes this rig* —
  which nothing downstream depends on. Do it only if a new solve looks odd and a second opinion is
  wanted.

### C. Record (host-side capture, ~2 h at the rig)

- [~] 3R.8 **Preserve the superseded calibration before writing anything new** — the rig-calibration
  spec requires it (*"Superseded calibrations are retained, not overwritten"*).

  **Archive DONE 2026-08-31.** `config/calib/archive/20260828/` holds every solver result, report and
  render from the 2026-08-28 session — 83 files, 19 MB, sha256 manifest verified — with
  `DELETED_INVENTORY.txt` naming each of the 63 raw files removed, and a table mapping each §3R result
  to the prior it should be checked against. The raw recordings are **deleted**: 30 GB of image bags
  and 5.6 GB of Kalibr `log.pkl` solver state, neither of which carries a number the reports do not.
  `datasets/calib_20260828_raw_keep/imu_stream.csv.gz` (26 MB) is the one exception — the only raw
  stream that could re-fit Δ if 3R.14's solve looks wrong.

  Still to do here: rename `config/calib/imx296_1456x1088/` → `imx296_1456x1088_20260828/`, likewise
  `rig_extrinsics_imx296.yaml`, the closed rig and `virtual_stereo_imx296.yaml`, each gaining a
  `superseded_by:` / `superseded_reason:` header naming this hardware change. **Disk: 146 G free
  after the archive**, against ~21 G for a full-rate session plus solver output — check headroom
  before recording, not during.
- [x] 3R.9 **Recordings done 2026-08-31/09-01.** Four single-camera sweeps re-recorded on 09-01
  after the first set proved unconverged (see 3R.11): 10 370 / 13 926 / 11 507 / 11 174 frames,
  every one at exactly 15.00 Hz with the board CSVs md5-verified and the board copy deleted.

  **The four pair recordings from 08-31 are reused unchanged** — the lenses and mounts have not
  been touched since, so those images are still valid; only the intrinsics they are solved against
  changed, and the new ones are *better* precisely in the image periphery where the pair overlap
  sits. 996 / 1234 / 1124 / 1433 simultaneous poses, 0.0 µs median skew.

  **Stage 9 (three cameras at once) was dropped, correctly.** The lens is H160 and the cameras sit
  90° apart, so adjacent pairs overlap by 70° while *diagonal* pairs have a 20° blind gap — and any
  triple contains a diagonal pair. It is geometrically impossible on this rig, not merely hard. It
  was my addition, not OmniNxt's.
- [x] 3R.10 **Camera↔IMU stage recorded four times** (2026-09-01), all via the ROS path with
  exposure-midpoint stamps and the full 200 Hz IMU in the bag — no CSV reconstruction needed this
  time: 5 ms (1101 img), 10 ms (812), DLPF 184 Hz (1600), DLPF 41 Hz (757). Excitation verified per
  run (gyro sd 0.17/0.16/0.33 rad/s, peak 1.91). Results in 3R.14 / 3R.14b / 3.6b.
- [x] 3R.11 **Stage 1 done — all four intrinsics re-solved on corner-inclusive sweeps (2026-09-01).**
  The first sweeps (2026-08-31) were re-done after the solve proved unconverged: fx moved 45 px
  between frame subsets of the *same* recording, with `xi` moving alongside it and `cx/cy` fixed —
  the signature of the Mei xi/focal ridge, not noise. Cause was no coverage at high incidence.

  | camera | xi | fx | fy | cx | cy | reproj px | cells short |
  |---|---|---|---|---|---|---|---|
  | cam1 | 2.0315 | 1618.81 | 1621.96 | 759.05 | 554.43 | 0.287 / 0.281 | 1 |
  | cam2 | 2.0572 | 1628.77 | 1629.48 | 743.03 | 525.38 | — | — |
  | cam3 | 1.9667 | 1562.58 | 1564.04 | 740.07 | 558.91 | — | **0** |
  | cam4 | 2.0011 | 1592.62 | 1595.76 | 728.59 | 550.52 | — | 1 |

  **Subset-stability check (the test that matters), full selection vs a thinner subset of the same
  recording:** cam1 agrees to **0.06 px** in fx; cam2 5.0; cam4 9.8; **cam3 34.4**. Only cam1 is
  demonstrably converged. Note cam3 has *perfect* cell coverage and the worst stability — so the
  8x8 cell grid is a poor proxy for what constrains the model, which is **incidence angle**, not
  image position. A corner *cell* can be filled by tags at moderate field angle.

  Carried forward: cam2/cam3/cam4 intrinsics are usable but carry 0.3-2.2 % focal uncertainty,
  which propagates to the pairwise baselines and to metric scale. Revisit if 3R.13's ring residual
  or the §5.1 scale check disappoints.
- [x] 3R.12 **Stage 2 done — four pairwise extrinsics on the new intrinsics (2026-09-01).**
  Solved from the 08-31 pair recordings via `scripts/calib/pair_extrinsics.py` (Kalibr's own
  `GridDetector` and target model, matching on the frame's own header stamp).

  | pair | cams | simultaneous views | baseline | rotation | round 1 |
  |---|---|---|---|---|---|
  | left | cam3 → cam1 | 994 | 155.4 mm | 90.01° | 147.8 mm / 90.9° |
  | front | cam1 → cam2 | 1178 | 152.7 mm | 89.43° | 148.7 mm / 90.4° |
  | right | cam2 → cam4 | 1067 | 153.2 mm | 90.90° | 149.2 mm / **92.2°** |
  | rear | cam4 → cam3 | 1364 | 161.1 mm | 90.64° | 149.2 mm / 90.1° |

  **The right pair, round 1's outlier at 92.2° with only 105 poses, is now 90.90° with 1067** — and
  it is the pair whose intrinsics moved most (cam2, −146 px in fx). The improvement landed where the
  theory said it would.

  **Baselines are ~4 mm longer than round 1 across the board**, which is the focal change showing up
  as expected: a shorter focal puts the board further away and lengthens the baseline.

  **But the baseline spread grew from 1.5 mm to 8.4 mm, and it is not random.** The two pairs
  involving **cam3** are the two highest (155.4, 161.1); the two without it agree to 0.5 mm
  (152.7, 153.2). cam3 is the camera that failed 3R.11's subset test worst (34.4 px of fx, drifting
  *downward* with fewer frames), and an underestimated focal inflates exactly these baselines.
- [x] 3R.13 **Ring closed 2026-09-01 — and the pre-closure misclosure more than halved.**

  | | round 1 | round 2 |
  |---|---|---|
  | before closure | **3.63°** / 9.2 mm | **1.58°** / 9.6 mm |
  | per-edge corrections | 0.57 – 1.31° | 0.23 – **0.94°** |
  | after closure | 0.0000° / 0.043 mm | 0.1436° / 0.000 mm |

  That halving is the clearest single verdict on the corner re-sweep: the corner coverage bought a
  real reduction in systematic bias, not merely different numbers.

  **Two things remain open, both pointing the same way.** 1.58° is still well above the ~0.5° a
  Monte-Carlo over each pair's own spread says random error alone would leave, so systematic
  per-recording bias persists — and the **rear pair takes the largest correction (0.94°)**, the pair
  containing cam3. Every independent line of evidence now converges on cam3's focal being the weak
  link: worst subset stability, both its baselines inflated, largest closure correction.

  **Anomaly worth resolving before this file is trusted:** the post-closure rotation residual is
  0.1436°, where the parameterisation (three poses in cam1's frame) makes closure *structurally*
  impossible to violate and round 1 reached 0.0000°. Translation closed cleanly to 0.000 mm. That
  looks like the optimiser stopping on a tolerance after 7 iterations rather than a data problem.
- [x] 3R.14 **Δ MEASURED 2026-09-01: +3.73 ms** (`timeshift_cam_imu = 0.003731525`), cam1 at
  4986 µs exposure, 1101 images and 14 693 IMU samples at 200.11 Hz, reprojection residual 0.45 px.
  Repeated on an independent recording at the same DLPF: **+3.7556 ms** — the two agree to **24 µs**,
  which is the only uncertainty figure that exists, since Kalibr reports none (it prints 18 digits).

  **The sign and magnitude both changed from last session's −8.06 ms, and the old value does not
  reproduce.** Two reasons to trust the new one: last session's IMU input had to be rebuilt from CSV
  because rosbag2 dropped it to 43 Hz (these carry the full 200 Hz in the bag), and it used the
  intrinsics we have since shown to be unconverged. `T_cam_imu` agrees to ~1e-3 across the two new
  solves. The "half the readout = 8.05 ms" coincidence is therefore dead — and it was a
  rolling-shutter quantity on a global-shutter sensor anyway.

  Written with provenance to `config/calib/imu_mpu9250.yaml`.
- [x] 3R.14b **Two exposures — Δ is CONSTANT in exposure.** +3.7315 ms at 4986 µs, +3.8612 ms at
  9986 µs. A mis-modelled `exposure/2` term predicted a **2.5 ms** shift; the measurement moved
  **0.13 ms**, so in `Δ = c + k·E` the slope is **k ≈ 0.026, i.e. zero**. The exposure-midpoint
  stamping is correct and Δ does not have to be re-measured when the exposure changes.

- [ ] 3R.14c **Align the gyro and accel filter delays, then re-measure Δ.** The two IMU paths
  currently disagree by **1.02 ms** — gyro 2.90 ms at 184 Hz, accel 1.88 ms at 218 Hz — and
  `imu_node` says so on every startup (*"group delays are REPORTED, not applied: the gyro lags the
  accel by 1.02 ms"*). Kalibr fits **one** Δ for both, so no single offset can correct two different
  lags; this is the leading suspect for the ~1.1 ms still unattributed in 3.6b.

  `accel_dlpf=2` (99 Hz, 2.88 ms) aligns them to **0.02 ms**, a 50× improvement, and incidentally
  puts the accel's corner near the 100 Hz Nyquist of the 200 Hz sample rate — 218 Hz is well past
  it, so the present setting aliases. Raising the *gyro* bandwidth instead would be the wrong move:
  a constant delay is fully absorbed by Δ, so there is nothing to gain, and 250 Hz aliases worse.

  Costs one 90 s recording plus a solve, since Δ moves by ~1 ms when the accel path changes.
  Recorded in `config/calib/imu_mpu9250.yaml` alongside the measured constant.

- [~] 3R.15 **Regenerated and measured 2026-09-01 — gates met, but the result is worse than round 1
  and points at cam3.** `VS_FOV=160`, 768×576, on the ring-closed extrinsics.

  | pair | cams | baseline | median \|dy\| | rms | round 1 \|dy\| | disparity valid | depth p10 / med / p90 |
  |---|---|---|---|---|---|---|---|
  | left  | cam3→cam1 | 0.1552 m | **2.06 px** | 3.11 | 0.45 | 14 % | 0.31 / 0.45 / 1.82 m |
  | front | cam1→cam2 | 0.1527 m | **1.32 px** | 2.86 | 1.79 | 17 % | 0.31 / 0.35 / 1.86 m |
  | right | cam2→cam4 | 0.1533 m | **2.86 px** | 4.07 | 2.55 | 11 % | 0.31 / 0.55 / 1.60 m |
  | rear  | cam4→cam3 | 0.1612 m | **3.67 px** | 4.52 | 0.66 | 10 % | 0.33 / 0.82 / 1.64 m |

  Gates: 100 % non-black on all four rectified views ✔; disparity sign consistent and depth positive
  and metrically plausible ✔; median |dy| reported ✔. Front improved (1.79 → 1.32); the other three
  regressed, and **the two worst are exactly the two pairs containing cam3** (left 2.06, rear 3.67),
  which also carries the worst subset stability (34.4 px vs cam1's 0.06), both baselines inflated
  (155.2/161.2 mm vs 152.7/153.3), and the largest ring correction (0.94°). Left **partial: to be
  re-run after the cam3 re-sweep**, which is expected to move the extrinsics feeding this stage.

  Figure: `config/calib/20260901/evidence/disparity_4pairs_round2.png` (rectified view + disparity per
  pair). It is a **2×2 montage of four independent pairs, not a fused omnidirectional depth** — nothing
  in this repo transforms the four into a common frame. Dense-disparity validity of 10–17 % here is the
  *scene* (repetitive tag grid at 0.3–0.6 m against a blank wall, worst case for block matching), not
  the rig; per 3.7 a calibration sweep remains *not* evidence either way, and a real verdict needs a
  textured scene at 1–3 m, which has still not been recorded.

  Three tool bugs had to be fixed before any of these numbers were valid, all of which had previously
  read as *bad data*: `vstereo_epipolar.py` and `vstereo_disparity.py` both re-derived the carve
  combination instead of taking the one the generator chose (keeping ~35 % of the image), and both
  assumed argument order set left/right — giving negative disparity, which `minDisparity=16` then
  silently discarded. Both now read the generator's `.yaml` sidecar and order the pair by baseline sign.

### E. Publish, and re-verify what can be verified offline

- [x] 3R.16 **PROMOTED 2026-09-01.** All four consumer files now carry the round-2 solve, tagged
  `calib_session: 20260901`, and the promoted set passes both gates of 3R.17 (frustum
  0.916-0.945; layout signs OK).

  - `config/calib/imx296_1456x1088/camN.yaml` — round-2 Mei intrinsics, each traced to the exact
    solve log that produced it (cam1←solve_cam1/log1, cam2←**solve_cam2b**/log1 i.e. the re-sweep,
    cam3←solve_cam3/log1, cam4←solve_cam4/log1). Reprojection 0.24-0.29 px on all four; **subset
    stability recorded alongside** (cam1 0.06, cam2 5.0, cam4 9.8, **cam3 34.4 px**) because that,
    not reprojection, is the number that separates them.
  - `config/rig/rig_extrinsics_imx296.yaml` — the ring-closed `rig_in_cam1`, with the frame
    convention stated (+x physically LEFT) and a **NEVER-MERGE banner** naming the round-1 mirror.
  - `config/rig/virtual_stereo_imx296.yaml` — regenerated from the promoted extrinsics. The carve
    combination is now *chosen* and the pair *ordered* in the file itself (`carve_yaw_deg`, and
    from/to swapped where the baseline sign required), rather than re-derived by each consumer —
    which is what made round 2 first read as a failure. Round-2 quality carried; round-1 p90 and
    signed-median deliberately **not** carried, since they were not re-measured.
  - Δ — already stated in `config/calib/imu_mpu9250.yaml` (3R.14), now also session-tagged.

  Conventions honoured: `fold_roll_for_vo.py` NOT run, and the raw-inverted assumption re-confirmed
  from `stage1_cam1_sel.bag` (printed text upside-down, floor at the top).

  **cam3 promoted as-is, deliberately.** Its re-sweep is outstanding and it is the weakest camera by
  every measure available. Promoting it unblocks the VO build, which is the cheapest test of whether
  its weakness actually matters; if tracking is poor, cam3 is the first thing to re-sweep. Recorded
  in cam3.yaml itself so the caveat travels with the file.

  Original text: write the results into the files the consumers actually read. This is §3.8 against
  the new session. Corrected 2026-08-31 after checking the launch file and the VO node:

  - `config/calib/imx296_1456x1088/camN.yaml` — tartancalib camchain converted to our
    `mu,mv,u0,v0` + `k2..k5`, with `sensor: imx296` and the true image size.
  - **`config/rig/rig_extrinsics_imx296.yaml`** — the ring-closed `rig_in_cam1` block. This is the
    file `bev_cuvslam.launch.py` loads (line 22), and the node reads `rig_in_cam1` specifically.
  - `config/rig/virtual_stereo_imx296.yaml` — regenerated per 3R.15.
  - Δ as a stated constant with provenance and direction.

  **Do NOT run `fold_roll_for_vo.py` on the Kalibr result.** That script belongs to a different
  lineage: it takes `rig_extrinsics_calibrated.yaml` (the panorama, feature-based chain, kept in the
  "nominal image-up" convention) and bakes `Rz180` in to produce `rig_extrinsics_vo.yaml`. Nothing in
  the current VO path reads that file. The Kalibr extrinsics are solved directly from the images the
  capture node publishes — raw sensor orientation, inverted — so they are already self-consistent
  with them, and folding the roll in again would put every camera 180 deg out.

  Confirm from one frame that the reconnect did not remount anything upright (3.9's roll still
  applies), because that assumption is what makes the previous paragraph true.
- [x] 3R.16b **State, and then fix, what frame the odometry comes out in.** `cuvslam_multicam_node`
  line 164: *"rig frame IS cam1's frame, which is how rig_in_cam1 is expressed"* — and it publishes
  that pose as `odom -> base_link`. So `base_link` is currently **cam1's raw optical frame**: an
  optical frame (z forward, x right, y down) that is additionally rolled 180 deg by the inverted
  mount. Neither the roll nor the optical->FLU convention is applied anywhere.

  Consequences, which differ by task and are why this is not merely cosmetic:
  - **§5.1 / §5.2 are unaffected.** A translation magnitude against a tape measure, and a
    return-to-origin drift, are frame-independent.
  - **Anything wanting a vehicle frame is wrong by that composition** — rviz, `tf` consumers, and any
    later IMU fusion, where the rig<->IMU rotation of `rig_layout.yaml` is expressed in FLU.

  So: either publish `rig_from_body` alongside (the 180 deg roll composed with optical->FLU, from
  `config/rig/rig_layout.yaml`), or declare `base_link` to be cam1's optical frame in the launch and
  README and let the consumer compose it. Decide before §6.2's documentation pass; do not leave it
  implicit, because a 180 deg roll produces trajectories that look entirely plausible.

  **DECIDED AND IMPLEMENTED 2026-09-01: the second option — name the frame truthfully.** The first
  option is not available: `rig_layout.yaml` has **no numeric rotation from body to cam1**.
  `position: front_left` is a label, not a rotation, and deriving one from the nominal layout would
  assume cam1 looks 45° off forward and sits level — on a rig that has already produced a camera
  ~19° off its nominal mount. Publishing a *computed-from-nominal* `base_link` would replace an
  obviously-wrong frame with a plausibly-wrong one, which is worse.

  Found while doing it, and larger than the task described: **`rig_layout.yaml` and the VO code used
  the word "rig" for two different frames.** The yaml's `R_imu_from_rig` was rig=FLU-body; the node's
  and `rig_extrinsics_imx296.yaml`'s `rig_in_cam1` is rig=cam1-optical. Composing the two — the
  obvious thing to do at the first IMU fusion — would have been wrong by (optical→FLU ∘ 180° roll)
  and would have looked plausible. Caught before any consumer existed: `R_imu_from_rig` was only ever
  *printed* by `scripts/imu/axis_check.py`, never composed.

  Changes: `rig_layout.yaml` now defines **body** (FLU, vehicle, = ROS `base_link`) and **rig**
  (cuVSLAM's word, = cam1 optical) explicitly at the top, and states that `R_body_from_cam1` is
  missing and why that blocks a true `base_link`; `R_imu_from_rig` → **`R_imu_from_body`** (and in
  `axis_check.py`). Both VO nodes default `base_frame` to **`cam1_optical_frame`**, not `base_link`
  (`cuvslam_multicam_node.cpp`, `bev_cuvslam_fused_node.cpp`, `bev_cuvslam.launch.py`,
  `fused_vo_params.yaml`), and `cuvslam_multicam_node` logs the frame and its caveat at startup —
  because the pose alone will never reveal the error.

  Unblocked by this: §5.1/§5.2 were already frame-independent and stay so. Still blocked: any tf
  consumer wanting a vehicle frame, until `R_body_from_cam1` is measured — that is new work, not
  part of this change. **The C++ is edited but not compiled** (no ROS 2 on this host); it builds or
  fails in the §4.5 board session like the rest of the node.

- [x] 3R.17 **Run 2026-09-01 as a dry run on the round-2 candidate — PASS, and it found something the
  check was not looking for.** **Real run against the promoted files done 2026-09-05: PASS.**
  `scripts/vo/verify_rig_build.sh` with no arguments, i.e. the tracked config the node actually
  loads (`rig_extrinsics_imx296.yaml` + `virtual_stereo_imx296.yaml` + `imx296_1456x1088`):
  frustum **0.936 / 0.934 / 0.916 / 0.945**, all 8 virtual cameras paired, and all three layout
  signs correct (cam2 x = −0.1083, cam4 z = −0.2181, cam3 x = +0.1128). Identical to the candidate
  figures, which is the expected result — 3R.16 promoted that candidate unchanged — and it is worth
  having as a fact rather than an inference, since promotion is exactly where a file gets swapped.

  Frustum, round-2 candidate (`closed.yaml` + `chains/`): **0.936 / 0.934 / 0.916 / 0.945**, all four
  pairs, clear of the hard-coded 0.5 gate. Round-1 tracked config reproduces 4.1b exactly
  (0.939/0.951/0.926/0.949). So the recalibration costs nothing in frustum overlap.

  **But the pairing labels flipped between rounds**, and chasing that down found that round 1 and
  round 2 place **cam2 and cam3 on opposite sides of cam1**:

  | camera | physical | round 1 x | round 2 x |
  |---|---|---|---|
  | cam2 | front_right | **+0.109** | **−0.108** |
  | cam3 | back_left   | **−0.100** | **+0.113** |

  cam4 differs by only 2.15°; cam2 and cam3 differ by ~179° about the rig-vertical axis. **Nothing in
  the pipeline caught this** — not the ring closure (1.58°), not the epipolar residuals, not the
  frustum graph, which scores ~0.94 *either way* because each fisheye contributes two carves and one
  always ends up facing a neighbour.

  **Round 2 is the correct one**, and the chain is traceable end to end: `board_sender.sh` maps
  port→topic explicitly (`CAMOF=( [c]=cam1 [d]=cam2 [e]=cam3 [f]=cam4 )`); `pair_left.yaml` puts cam3
  at x = +0.113 in cam1's frame and `pair_front.yaml` puts cam1 at x = +0.109 in cam2's;
  `closed.yaml` reflects both. And the frames the solve consumed are **raw and inverted** — verified
  by pulling a frame out of `stage1_cam1_sel.bag`, where the printed text reads upside-down and the
  floor is at the top. The VO node receives that same raw frame (the capture node memcpys, it does
  not rotate), so the 180° mount roll makes **+x physically LEFT** in cam1's frame, and round 2
  satisfies every sign. Round 1 is consistent with a solve on ISP-rotated (upright) frames
  (`csi_sender.sh FLIP=2`) — the mechanism is not proven, and does not need to be, because 3R.16
  replaces the file.

  **Consequence for 3R.16: replace the extrinsics wholesale. Never merge a round-1 value into a
  round-2 file** — the two are in mirrored frames and any mixture is geometrically incoherent while
  passing every check that exists.

  Added a check that is actually sensitive to it: `check_rig_poses.py` now takes the extrinsics and
  `rig_layout.yaml` as optional arguments and asserts each camera's position *sign* against the
  physical ring order. Verified both ways — round 2 exits 0, round 1 exits 1 naming both cameras.
  `verify_rig_build.sh` also takes the three paths as arguments now, so a candidate solve can be
  gated **before** promotion instead of only after.

  Original text: re-run the offline rig verification before the board session, unchanged from 4.1b:
  `scripts/vo/verify_rig_build.sh` (cuVSLAM's own frustum test on the poses the C++ emits — the
  0.939/0.951/0.926/0.949 figures will move and must stay clear of the hard-coded 0.5 gate) and
  `scripts/vo/check_rig_poses.py`. This is the only part of §4 checkable without a build, and running
  it first stops a bad rig file from being misdiagnosed as a ROS 2 wiring error on the TX2.
- [ ] 3R.18 **Exercise 1.5 — the stale-calibration guard — against the new files.** It has never been
  run; a recalibration is the exact event it exists for. If it can key on a session/date field, add
  one, so that the *next* refocus is caught by the node instead of by a drifting trajectory.

**Not redone, and why:** the capture node (§1) resolves its mapping at runtime; §2's target, containers
and model decision are unaffected by focus; 3.7's carve *methodology* and 4.1/4.1b's code stand — only
the numbers they consume change. 3.10 stays optional.

**Ordering:** A → B → C → D → E. Only A and the recording half of C need the rig; B is 30 minutes that
prevents a full redo; D and E are host-side. §4.5 and all of §5 stay blocked until 3R.16 lands, and
the TX2 build attempt (the first thing that will fail, since the node has never been compiled) can be
done during A, in parallel, since it needs no calibration.

## 4. Remove the sync workaround

- [x] 4.1 **Done.** Bundler and `sync_slop_ms` gone; sets formed by timestamp over a per-camera history (never by arrival order - separate DDS subscriptions say nothing about which trigger edge a frame came from), gated at `max_skew_us` (default 1000), each image carrying its own exposure-midpoint stamp, failures dropped and counted.
- [x] 4.1b **Rescoped and done: the node feeds cuVSLAM EIGHT VIRTUAL PINHOLES, not four fisheyes.** Task 2.9 closed the direct route - the lenses fit ~192 deg and cuVSLAM's only fisheye model is equidistant, capped below 180 - so the carve moved from an offline analysis step into the node. 8-camera Pinhole rig, remap tables built once at startup, `rectified_stereo_camera` left false on purpose (its horizontal-only tracker cannot move vertically and demands paired cameras share a rotation matrix to 1e-6, against our 1.0-1.4 deg).

  Verified without a board, since none of it can be run here: the hand-written Mei projection matches `cv2.omnidir.projectPoints` to 5e-13 px over 4000 rays and the built map is bit-identical at 25 sampled pixels; the quaternion conversion round-trips 20000 random rotations through all four trace branches to 1.6e-7; and `scripts/vo/verify_rig_build.sh` re-runs cuVSLAM's own frustum test on the poses the C++ emits, reproducing 0.939/0.951/0.926/0.949 with all 8 cameras paired. That last check exists because a sign error in `rig_from_fisheye * Ry` still yields a valid rig that cuVSLAM would accept while finding no stereo pairs at all - it drops the pairing to ~0.03.

  **NOT compiled.** The ROS 2 wiring needs an environment this host does not have.
- [x] 4.2 **DONE 2026-09-01 — the fused zero-copy node runs the carve on the GPU, with zero
  dropped sets.** Steady **8.8 Hz**, and the timing says something that changes the priorities:

  ```
  acquire + gpucopy = 26.3 ms
  Track             = 87.6 ms   <- the bottleneck
  ```

  **cuVSLAM's own `Track()` on 8 virtual cameras costs 87.6 ms.** It is 3× the entire capture
  and carve, and neither the remap work nor either resolution axis (4.5b) touches it. The
  images no longer go over DDS at all — the ~95 MB/s the modular path was shipping is gone.

  What it took, beyond the CUDA kernel: the node was still the IMX219 rig's in every respect
  (832×624 from a 1640×1232 sensor mode, four RAW fisheyes as `Distortion::Fisheye`, the old
  `projection_parameters` layout, one unified cam0 timestamp), plus `sensor_ids: [1,2,3,4]`
  asking for a fifth camera on a four-camera rig. All fixed; the ISP downscale is gone because
  the intrinsics are solved at 1456×1088 and a downscaled source leaves the carve working and
  merely wrong.

  **The bug worth remembering:** it first rejected **2673 of 2673** sets at exactly 33.3 ms on
  a rig with 8 µs of hardware skew. Acquiring one frame per camera in a loop does not give you
  a set — each consumer has its own queue and the four frames can sit on different edges. The
  modular node hit the identical wall for a different reason (DDS delivery order). Twice now
  the answer has been *align, then gate*, and twice the symptom read as "is the trigger
  running?" while the trigger was perfect.

  Shutdown also fixed: the blocking acquire outlasted launch's 5 s patience, so every run
  ended in SIGKILL, which leaks an Argus session and makes the *next* start fail with "Argus
  setup failed" — that cost a debugging cycle today.

  Original text: **GATE CONDITION NOW MET (2026-09-01) — this is the next piece of work.** The deferral below
  was conditional on the modular path proving too slow, and 4.5/4.5b have now shown exactly that:
  ~30 ms of CPU remap per set, half the sets dropped at 30 Hz and a third at 15 Hz, and both
  resolution axes measured to be bad trades. The images are also going out over DDS —
  4 × 1456×1088 mono8, ~95 MB/s at 15 Hz, plus a cv_bridge conversion on each side — which the fused
  node removes entirely by never leaving the GPU.

  The node itself is written and, as of today, **builds**: Argus NVMM Y-plane → CUDA device pointer
  via `NvEGLImageFromFd` + `cuGraphicsEGLRegisterImage`, publishing only `/cuvslam/odometry` + TF,
  with the bridge already validated against the CPU path by `scripts/port/egl_cuda_spike.cpp`.
  What it still needs is stated below: the virtual-pinhole carve as a **CUDA** remap on the NVMM
  buffer, plus 4.1's stamping fix. Note that ROS 2 Foxy has no practical loaned-message/shared-memory
  path, so single-process fusion is the right answer here rather than a DDS transport trick.

  Original text: **Deliberately deferred, and the reason is not effort.** The fused node exists to avoid a CPU round-trip (NVMM Y plane straight to CUDA). Adding the virtual-pinhole carve to it means a CUDA remap on the NVMM buffer - a CPU remap would negate the node's entire purpose. Until 4.5/section 5 show the modular path is too slow, the fused node has no justification to be rewritten twice. Note the stamping fix it needs is the same one 4.1 made; `iframe->getTime()` is the WRONG source (consumer-side - it reported cameras ~7 ms apart in capture-loop order), so this task's original wording is superseded by README 4.7.
- [x] 4.3 Done for the modular node AND, as of 2026-09-01, the fused node (sets / worst skew / dropped counter / windowed Hz + acquire + Track timings). Original: (sets / worst skew / drops / remap time every 5 s); pending for the fused node with 4.2. Report the drop counter and recent worst-case skew from both nodes; make a stopped trigger diagnosable as a trigger fault, not a camera failure (spec: *A stopped trigger is diagnosable*).
- [~] 4.4 `bev_cuvslam.launch.py` retargeted (ring-closed extrinsics, virtual-stereo config, skew gate, no `sync_slop_ms`). `fused_vo_params.yaml` and the fused run scripts wait on 4.2.
- [~] 4.5 **FIRST BUILD AND FIRST LIVE RUN, 2026-09-01.** `bev_cuvslam` had never been compiled;
  it now builds and runs on the board. Four findings, two of them defects that are fixed.

  **Build.** `docker compose run --rm build-ws` in `cuvslam-foxy:tx2`. libcuvslam did NOT need
  rebuilding (already CUDA-10.2 built at `third_party/cuVSLAM/build_tx2gpu/bin/libcuvslam.so`); the
  submodule pointer is identical on both branches, so the checkout preserved it. One blocker: the
  workspace's `install/cv_bridge` was a June artifact linked against OpenCV **4.8** (`.so.408`, from
  the VINS work in `/usr/local`) while the image ships 4.2 — it shadowed the image's own
  `ros-foxy-cv-bridge` and failed at link. Moved aside as `install/cv_bridge.ocv48.bak`; all three
  packages then built. Also: compose interpolates the WHOLE file, so `build-ws` refuses to run
  without `EXPOSURE_US` even though only `modular` uses it.

  **The rig is accepted by cuVSLAM, and 3R.17's offline check is now validated against it.** cuVSLAM's
  own frustum test found exactly four pairs, covering all 8 virtual cameras, and they are the four
  `verify_rig_build.sh` predicted:

  | cuVSLAM pair | ratio | offline prediction | ratio |
  |---|---|---|---|
  | 0-3  cam1₋₄₅–cam2₊₄₅ | 0.905 | cam1_L–cam2_R | 0.936 |
  | 1-4  cam1₊₄₅–cam3₋₄₅ | 0.948 | cam1_R–cam3_L | 0.934 |
  | 2-7  cam2₋₄₅–cam4₊₄₅ | 0.907 | cam2_L–cam4_R | 0.916 |
  | 5-6  cam3₊₄₅–cam4₋₄₅ | 0.952 | cam3_R–cam4_L | 0.945 |

  Agreement within 0.03 on every pair. The offline gate can be trusted before a board session.

  **Defect 1, fixed: the set matching raced DDS delivery.** With the trigger active the capture node
  measured **8 µs** of real skew, and the VO rejected **204 of 206** sets at exactly 33.3 ms — one
  frame period. It matched on cam1's *arrival* and then took the nearest-stamp candidate from each
  other camera, but at that instant the same-edge frames have usually not been delivered, so the
  nearest is the PREVIOUS edge. Now anchors on cam1's oldest buffered frame and waits until every
  other camera has delivered a frame at or after it. Worst skew **33.3 ms → 1 µs**, acceptance
  **1% → 48%**. It presented as "is the trigger running?", and the trigger was perfect.

  **Defect 2, environmental: `trigger_mode` resets to 0 on reboot.** The first run reported
  `trigger free-running` and dropped everything at 36.9 ms. `/sys/module/imx296/parameters/trigger_mode`
  was 0 after the morning's reboot. `record_calib_session.sh` already documents and preflights this;
  **the VO path does not** — see 4.6.

  **Remaining, and it is a throughput limit, not a sync problem:** ~50% of sets still drop. The
  capture node reports the same 33.33 ms events at its own source (314 over-limit of 2269, ~14%)
  while median offsets stay at 0 µs, so cameras are genuinely missing trigger edges. The cause is in
  the same log: **`remap 31271 us` per set** — 31 ms of CPU remap for 8 virtual cameras against a
  33 ms budget at 30 Hz. The modular node is saturated. Next: decimate to 15 Hz (§3.2b's
  `publish_every_n`, added for exactly this) and/or move to the fused node, which exists to avoid
  this CPU round-trip (4.2).

  Also: repeated runs leak an Argus session — one run died with `Argus setup failed` until
  `sudo systemctl restart nvargus-daemon`. Compose warns about it; worth automating.

  **THE RIG PRODUCED ODOMETRY, 2026-09-01.** `/cuvslam/odometry` publishing live, `frame_id: odom`,
  `child_frame_id: cam1_optical_frame` (3R.16b's rename visible on the wire), real pose and a
  populated covariance, **zero "tracking lost" warnings** across the run. cuVSLAM is tracking on the
  promoted round-2 calibration. This is the first odometry this rig has ever produced.

  **At 15 Hz (`PUBLISH_EVERY_N=2`):** source anomalies **13.8% → 3.0%** (70 over-limit of 2358),
  VO acceptance **48% → 64%**, worst skew still **1 µs**. Better, not fixed — and the remap is
  still 26-39 ms per set, so the CPU is near saturation even with the budget doubled to 66 ms.

  Original text: blocked on §3R (the rig is uncalibrated as it stands). Run both nodes on the board: confirm zero dropped sets with the trigger live, worst-case skew < 1 ms, and `/cuvslam/odometry` tracking with no "tracking lost".

## 5. Motion test (closes bring-up-end-to-end-vo 3.4 / 3.6)

**Blocked on §3R**: a scale check against a tape measure is meaningless on a stale calibration, and
a drift number would be attributed to the VO rather than to the optics. Tooling ready:
`scripts/vo/run_motion_test.sh` (board; refuses to record unless
`trigger_mode` is 1) and `scripts/vo/analyze_motion.py` (host). Needs the rig powered and
physically moved - the remaining items are not doable from here.

- [~] 4.5b **MEASURED 2026-09-01: lowering the resolution is a bad trade on both axes.**
  `scripts/vo/bench_remap.cpp`, standalone on the board (no cameras, no ROS, same maps the node
  builds, synthetic textured source), 30 iterations.

  | virtual | focal px | Mpix | ms/set |   | source | virtual | ms/set |
  |---|---|---|---|---|---|---|---|
  | 768×576 | 548.4 | 3.54 | 17.18 |   | 1456×1088 | 768×576 | 17.05 (100%) |
  | 640×480 | 457.0 | 2.46 | 16.53 |   | 1092×816  | 768×576 | 16.43 (96%) |
  | 512×384 | 365.6 | 1.57 | 10.46 |   | 728×544   | 768×576 | 15.92 (93%) |
  | 384×288 | 274.2 | 0.88 |  9.33 |   | | | |

  **Virtual**: 2.25× fewer output pixels buys 39%, and costs 33% of the virtual focal length —
  which §3.7 already warned about (at 480×360 tag detection collapsed from ~10/frame to 0.1).
  **Source**: 4× fewer source pixels buys **7%**. The maps are already fixed-point `CV_16SC2`
  (`convertMaps`), so the obvious lossless win was taken long ago.

  What the two sweeps locate together: ns/pixel *rises* as the output shrinks (4.9 → 10.5), and
  extrapolating the virtual sweep to zero output leaves **~8.5 ms/set of cost independent of both
  resolutions** — across 8 separate `cv::remap` calls, ~1 ms each. That is parallel-dispatch
  overhead, six threads spun up eight times per set, and it is recoverable at **zero quality cost**
  by batching the eight gathers into one parallel region. Worth one experiment before anything is
  traded away.

  Also: the benchmark measures 17 ms where the node reports **31 ms in situ**, so about half the
  node's remap time is contention with capture, cv_bridge, DDS and cuVSLAM over the same six cores —
  which no resolution change addresses either.

  Order of attack, revised on this evidence: (1) decimate to 15 Hz — doubles the budget 33→66 ms,
  costs nothing in quality, and 31 ms fits; (2) batch the remaps; (3) the fused node (4.2), which
  removes the CPU round-trip entirely and is the designed answer. Resolution reduction is last.

- [x] 4.8 **cuVSLAM v15 -> v17: whole pipeline re-verified 2026-09-02, and it is markedly faster.**

  | | v15.0.0 | v17.0.0 |
  |---|---|---|
  | odometry rate | 8.8 Hz | **~13.8 Hz** |
  | `Track()` | 87.6 ms | **~49 ms** |
  | acquire + gpucopy | 26.3 ms | 21-24 ms |
  | stationary drift (path length, ~37 s) | 2.50 m | **1.42 m** |

  `Track()` nearly halved, which was the bottleneck 4.2 identified. Total is now ~70 ms, so
  the node sits just under the 15 Hz decimated capture rate rather than half of it.

  Checked, in order: the offline gate compiles against the v17 headers and passes with
  **identical** frustum ratios (0.936/0.934/0.916/0.945) and layout signs, so the API surface
  we use and the rig geometry are unchanged; a forced clean rebuild of all three packages
  produced **zero errors**; the runtime reports `cuVSLAM 17.0.0` and tracks; the bag came back
  clean (metadata.yaml, 480 odometry + 480 tf) and `analyze_motion.py` read it and produced a
  verdict. The section 5 pipeline is unaffected.

  **One thing lost: v17 no longer prints the `FRUSTUM pair X-Y ratio ... thr 0.500` lines.**
  That was how 4.5 cross-checked `verify_rig_build.sh` against cuVSLAM's own computation, and
  it can no longer be repeated at runtime. The gate is still valid — v17's
  `libs/camera/frustum_intersection_graph.cpp:56` still has
  `intersected_num_points_ratio_threshold = 0.5`, the same constant our reimplementation
  assumes — but that is now verified by reading the source rather than by observing the
  library agree, and a future bump could change it silently. Worth re-checking that constant on
  every cuVSLAM upgrade.

- [x] 4.7 **DONE 2026-09-05, and it turned out the shared header already existed and nothing used it.**
  The fused node took raw 0-based Argus indices from `sensor_ids`, so it assumed bind order equals
  port order c,d,e,f. Task 1.1 established that this must not be assumed — a different boot shifted
  it, which is why `argus_capture_node` resolves at runtime. A shift here permutes the cameras
  against the extrinsics silently, and the frustum graph would still pass: it is the same shape
  of error as the round-1/round-2 mirror (3R.17).

  **The fix is smaller than the finding.** `bev_camera/include/bev_camera/argus_timing.hpp` already
  held the port table, the sysfs scan, the trigger probe and the exposure-midpoint rule, and its own
  header comment says it is *"used by BOTH the capture node and the fused VO node — these two facts
  must not be allowed to drift apart"*. **Nothing included it.** The capture node carried a verbatim
  copy of the port table and a near-copy of `frame_timing()`; the fused node had a third variant.
  The header was dead code and the drift it was written to prevent had already happened. Both nodes
  now include it, and `bev_cuvslam` gained a `bev_camera` build dependency (the CMake comment in
  `bev_camera/CMakeLists.txt` had been describing this arrangement as though it existed).

  **Verified live on the board**, not merely compiled:

      port c (imx296 2-001a) -> sensor-id 0 -> cam1     port e (imx296 7-001a) -> sensor-id 2 -> cam3
      port d (imx296 2-0018) -> sensor-id 1 -> cam2     port f (imx296 7-0018) -> sensor-id 3 -> cam4

  On this boot bind order *does* equal port order, so the resolved ids match the `[0,1,2,3]` that was
  hard-coded — the run proves the resolution works, not that it changed the answer today. That is the
  point: the value is that a shifted boot now moves with the hardware instead of silently mislabelling
  the rig. `sensor_ids` survives as an override and warns that the mapping is then unverified;
  a missing port throws rather than starting a partially populated rig.

- [x] 4.7b **Two further defects found in the fused node while doing 4.7, both from never being
  retargeted off the IMX219 rig. One of them is worth ~40 % of the odometry rate.**

  1. **No AE lock.** Under external trigger AE cannot reach its main actuator and hunts on gain
     instead — 1.4 measured a 3.5 Hz limit cycle at 171 % of mean luma and the capture node has
     clamped gain and locked AE ever since. The fused node never did, so the fused path has been
     feeding cuVSLAM hunting brightness all along.
  2. **Per-camera exposure in the midpoint stamp.** All four cameras fire on one trigger edge for one
     pulse width, so subtracting each camera's *own* Argus-reported half-exposure injects differences
     the hardware does not have — into the very stamps the 1 µs skew gate then judges. Now latched
     rig-wide via the shared header, with `exposure_us: 4986` so the stamp rests on the measured pulse
     width rather than the 0.521 ms fiction Argus reports under trigger (1.10).

  **Controlled A/B, same rig, same scene, minutes apart, ~45 s each:**

  | | AE locked | AE free |
  |---|---|---|
  | odometry rate | **14.7–15.0 Hz** | 9.8–12.0 Hz |
  | `Track()` | **43–45 ms, flat** | 56 → 79 ms, rising, plateaus ~78 |
  | acquire + gpucopy | 23 ms | 21–23 ms |

  **This answers the question 5.0b left open.** 5.0b recorded `Track()` rising monotonically within a
  run (70.9 → 93.2 ms over 24 s), ruled out thermal, clocks, CPU availability and library version, and
  was left with "the scene" as an explicitly unmeasured hypothesis. The rise is **AE hunting**: with AE
  free the curve climbs and plateaus, with AE locked it is flat for the whole run. `acquire+gpucopy` is
  unchanged between the two, so this is cuVSLAM's cost responding to the images, not a capture effect.

  Honest limits: one scene, one ~45 s pair, on wall power. 5.0b's run was on battery, so this
  identifies a sufficient mechanism with the same signature rather than proving that *that* run's rise
  had this cause. It does remove "the scene" as the leading hypothesis, and 4.8's reading of the
  morning tail rise as "the shutdown tail" is now doubly unsafe — that run had no AE lock either.

- [~] 4.6 **Preflight the VO path the way the calibration path is preflighted.** `trigger_mode` silently
  resets to 0 on every reboot and the VO then produces nothing but drops, while asking "is the trigger
  running?". `record_calib_session.sh` already checks and sets it; add the same to the VO run path,
  plus the F401 polarity/exposure read-back and an `nvargus-daemon` restart on a leaked session.

  **Partial, 2026-09-05 (4.7).** The fused node now reads `trigger_mode` at startup and warns
  explicitly that the cameras are free-running and that every set will be dropped by the skew gate —
  so the failure names itself instead of presenting as four broken cameras. That is the *diagnosis*
  half only. **Still owed:** the same check in the run scripts as a gate rather than a warning, the
  polarity and `pulse_ns` read-back that derives `exposure_us` instead of trusting the yaml (the stamp
  is SOF − exposure/2, so a stale value biases every timestamp rather than failing), and the
  `nvargus-daemon` restart. `scripts/log_rig.sh` already does the daemon restart and the generator
  read-back; the VO run path should reuse that block rather than grow a second copy.

- [~] 5.0 **Recording pipeline set up and deployed 2026-09-01; NOT run.** Waiting on rig time.

  `scripts/vo/run_motion_test.sh <label> <tape_m> [--record-images]`, driving a new
  `motion` compose service that runs capture + VO + recorder in one container.

  One real fault it had before: it launched `bev_cuvslam.launch.py`, which starts the **VO
  node only** — no capture node — so it would have recorded an empty run.

  *(An earlier version of this entry also blamed its use of `/dev/ttyTHS1`, calling that the
  superseded H7's port. That was wrong, and the operator corrected it: the F401 answers on
  **both** `/dev/ttyACM0` (USB CDC) and `/dev/ttyTHS1` (UART) — verified 2026-09-01, identical
  status from each. Its trigger check would have worked. Comments across `record_calib_session.sh`,
  `trigger_probe.py` and `docker-compose.yml` implied ttyTHS1 belonged only to the dead H7 and
  have been corrected too, since they are what led to the wrong conclusion.)*

  The preflight now **gates rather than warns**: `trigger_mode=1`, generator running,
  `polarity=active_low`, and it reads `pulse_ns` back to derive `exposure_us` instead of
  taking it on faith — the stamp is SOF − exposure/2, so a stale value does not fail, it
  biases every timestamp by half its error. Verified against the live generator: all three
  gates match and it derives 4986 µs. It also restarts `nvargus-daemon`, because a SIGKILLed
  run leaks a session and the next start dies with "Argus setup failed" (4.2).

  `--record-images` bags the four camera streams so a run is **replayable** — move the rig
  once, then re-run the VO against it as often as needed without being at the rig. This
  matters now that `Track()` at 87.6 ms is the bottleneck (4.2) and the tuning ahead wants
  many runs over the same motion.

  **Two passes, kept separate**: one with images for replay, one without for the live 5.1/5.2
  numbers. The recorder competes for CPU and I/O and can induce drops of its own, which on
  replay are indistinguishable from the rig misbehaving. Images cost ~95 MB/s at 15 Hz and the
  SD is known not to absorb 30 Hz of four cameras, so decimate and keep image runs short.

  **No existing bag can substitute.** Every IMX296 recording is 1–2 cameras (a consequence of
  the `PORTS` fix that cured the 50 % DDS loss); the only 4-camera bag is IMX219 at 1640×1232,
  free-running, from June. And 5.1/5.2 need ground truth captured at record time, which no bag
  supplies retroactively.

- [~] 5.0b **Short test log on BATTERY over WiFi, 2026-09-02 — the pipeline works end to end.**
  `tx2-wlan` (192.168.0.168). Clean bag with `metadata.yaml`, 202 odometry + 202 tf,
  `analyze_motion.py` read it and produced a verdict. Nothing about running on battery or over
  WiFi breaks the chain — the whole run is local to the board, so only ssh control traffic
  crosses the network.

  The preflight earned its keep again: the power-on had reset **`trigger_mode` to 0** and the
  F401 back to **`active_high`**, both of which it gates on. The F401 also answered on
  **`/dev/ttyTHS1` while `/dev/ttyACM0` was still enumerating** (ACM0's device node was
  timestamped a few seconds earlier) — exactly the case the operator flagged when correcting
  the "dead H7 port" claim, and a good reason `TRIG_PORT` is overridable.

  **Performance is roughly half the wall-power v17 run, and the cause is NOT the obvious ones.**

  | | wall power (09:02) | battery (11:42) |
  |---|---|---|
  | rate | 13.8 Hz | 8.4 Hz |
  | `Track()` first sample | 48.8 ms | 70.9 ms |
  | `Track()` last sample | 73.7 ms | 93.2 ms |

  Ruled out on the board: thermal (40-41 °C, every cooling-device state 0), clocks (GPU pinned
  1300.5 MHz, EMC 1866 MHz, `FreqOverride=1`, `jetson_clocks` applied), CPU availability (all
  six cores online at 2035200), and library version (both runs report `cuVSLAM 17.0.0`). The
  remaining candidate is the **scene** — cuVSLAM's cost scales with features, matches and map
  size, and the rig is somewhere different now — but that is a hypothesis, not a measurement.

  **`Track()` also rises within a run, and an earlier reading of that was wrong.** 4.8 recorded
  the morning run's tail rise as "the shutdown tail, not progressive degradation"; the battery
  run rises monotonically from the FIRST sample (70.9 -> 93.2 ms over ~24 s), so that reading
  was a guess and not a safe one. What is honestly established: the trend is real in this run,
  and it is not known whether it plateaus. It matters for §5 — a long run may start near 13 Hz
  and end considerably slower. **Cheap way to settle it:** one longer stationary run with every
  2 s sample captured, and see whether the curve flattens. Worth doing before a long motion run
  is planned around a rate.

  **SETTLED 2026-09-05 by 4.7b: the rise is AE hunting, and the remaining candidate named above
  ("the scene") was wrong.** With AE free the curve climbs 56 → 79 ms and plateaus; with AE locked
  it is flat at 43–45 ms for the whole run, at 14.7–15.0 Hz instead of ~10.5. The fused node had no
  AE lock at all until 4.7b, so every fused measurement in this file — including this task's battery
  numbers and 4.8's 13.8 Hz — was taken with AE hunting. Re-measure before drawing a rate conclusion.

- [~] 5.0c **First replayable 4-camera IMX296 motion log, replayed through VO end to end, 2026-09-03.**
  This closes the gap 5.0 flagged ("No existing bag can substitute" — every prior IMX296 recording
  was 1–2 cameras). `imglog_vio1_20260903_103102` (from `225`) is a full 4-camera raw log:
  1774 frames on cam1/2/3, 1773 on cam4, **88.9 s at 19.96 Hz** (~20 Hz, not the 30 Hz the header
  advertises), **99.9 % complete 4-camera sets** (1773 of 1774), and well synchronised — median
  inter-camera skew 0 µs, worst 146 µs cam1↔cam2. `check_log_sets.py` and a raw index-vs-`.raw`
  frame-count check both pass.

  **Correction, 2026-09-05 — "88.9 s" is the recording length, not the amount of data. See 5.0f.**
  The rig was only moving for **45.7 s** of it: 5.7 s stationary at the start and **37.5 s
  stationary at the end**, after the operator had stopped walking. Quoting the recording duration
  here was misleading, and the 30 s replay subset below made it worse — `--max-frames` takes the
  FIRST 600 frames, so the replay spent its first 5.7 s on a motionless rig and discarded 21 s of
  motion that came later. (The frames themselves are all genuinely there and all distinct; this is
  not truncation. Re-checked today: interior set completeness is 100.0 %, so even the "99.9 %"
  above was only the ragged final edge.)

  Converted to a Foxy rosbag2 with `scripts/port/raw_log_to_bag.py` (topics `/cam{1..4}/image_raw`,
  `sensor_msgs/msg/Image`, mono8 1456×1088), a 30 s / 600-frame subset pushed to the TX2, and
  replayed through `bev_cuvslam.launch.py` (the modular `cuvslam_multicam_node`) inside
  `cuvslam-foxy:tx2` at **rate 1.0**. The VO node gates sets on `header.stamp`, so a plain
  `ros2 bag play` (no sim time) reproduces the deployed pipeline faithfully.

  **Result — the pipeline works on real 4-camera data:** cuVSLAM 17.0.0 came up (4 fisheyes → 8
  virtual pinholes, sets gated at 1.0 ms), tracked continuously, and produced **328 odometry poses
  over 29.3 s = 11.18 Hz**, path length 13.94 m, straight-line displacement 4.424 m (wander 3.15).

  **Two honest caveats.** (1) ~**17 % of sets were dropped** at exactly 50/100/150 ms skew — i.e.
  the matcher pairing across frame boundaries because it fell behind. Source sync is 146 µs, so
  this is throughput, not the log: `Track()` at ~50–90 ms/set cannot keep up with a 50 ms period,
  so the input queue backs up and only ~55 % of the 20 Hz sets survive to produce odometry. (2) This
  is a **pre-recorded** motion with **no tape measure and no return-to-origin captured at record
  time**, so it validates 5.0/5.0b (replay end to end) and feeds 5.4 (rate), but it **cannot close
  5.1 or 5.2**, which need ground truth the log does not carry.

  Artefacts on the host: `datasets/imglog_vio1_20260903_103102_bag` (full, 7095 msgs) and
  `datasets/replay_out/odom_20260903_113841` (the recorded odometry). Compose note: the `shell`
  service run needs `EXPOSURE_US` and `MOTION_SECONDS` set to any value (the `motion`/`modular`
  services interpolate them at parse time even when they are not the target).

- [~] 5.0d **The 5.0c replay inspected visually, 2026-09-04 — see `add-replay-visual-diagnostics`.**
  5.0c established the pipeline *runs* using scalars (pose count, rate, path length). Those cannot
  separate a correct trajectory from a wrong one of the same length, so the same log was replayed into
  `scripts/vo/rerun_multicam.py`: 8 virtual pinholes with their features, the 4 raw fisheyes, the
  landmark map and the trajectory on one timeline. Four things it established that the scalars hid.

  **(1) One virtual camera is effectively blind in this log.** vcam3 (cam2's +45° carve) yields
  **~12 features/frame against 2402/frame overall** — the operator's head occludes it. So 5.0c's
  "tracked continuously" is true but not evenly sourced, and this qualifies 5.4's tracking result.
  A repeat run must keep the operator out of that carve.

  **(2) The map has large-range outliers.** `obs_20260903_140714` carries 26 759 landmarks of which
  only **24 234 are within 20 m — the map reaches 310 m in a 14 m indoor log.**

  **(3) Vertical drift is visible, and its cause is structural, not a tuning fault.** This path is
  pure VO — no IMU, no loop closure — so nothing pins roll and pitch to gravity and forward motion
  leaks into apparent vertical motion. Measured wander is **±0.4 m** over the walk. `bev_imu` +
  MPU9250 already exist in the repo and are the structural fix; this is evidence that they are
  justified, and it is the honest form of 5.2 that a log with no ground truth can support.

  **(4) Not all of the vertical motion was drift — and the operator was right where the fit was
  wrong.** A floor estimate taken along each pose's **own** up axis (immune to odometry tilt) puts the
  floor 0.22 m below the camera for t < 6 s and ~1.27 m from t = 7.4 s; the t = 6–9 s segment has
  **dz = +1.22 m with only 0.30 m of horizontal motion**, which drift cannot produce. That is the rig
  being lifted onto the head, i.e. real. **A global plane fitted once in the odometry frame reported
  the height swinging 0.03 → 1.56 m on what was a constant-height walk — it was measuring map drift,
  not the floor.** Fitting locally, near each pose in that pose's own frame, gives 1.36 m ± 0.20 m.

  **Caveat, unchanged from 5.0c:** this log carries no ground truth, so none of the above closes 5.1
  or 5.2. It sharpens what to measure when a run with ground truth is made.

- [x] 5.0e **Raw logging is now LOSSLESS at 20 fps, and the residual 0.1–0.2 % everything above
  quotes was startup, not storage. 2026-09-05.**

  5.0c and 5.0d both reported "99.9 % complete sets" and left it there, as if a fraction of a
  percent were simply the cost of recording. It was not. Locating the missed edges by sequence
  number instead of counting them showed the loss was **entirely in the first 0.15 s**:

  | | missed trigger edges |
  |---|---|
  | cam1 | seq 3–4 |
  | cam2 | seq 2–3 |
  | cam3 | seq 2 |
  | cam4 | **none at all** |

  and **58 s of steady state missed not one edge on any camera.** The four Argus streams come up
  ~215 ms apart and miss edges *independently* while they do, so those first few edges are
  incomplete sets — and an incomplete set from a synchronised rig is worth very little.

  **Fixed in the node rather than trimmed offline** (`log_warmup_ms`, default 500): the image log
  waits until every camera has delivered a frame, then settles, then writes. Gated on
  `image_log_dir` so the DDS/VO path is untouched. The skipped frames still get a frame-CSV row
  with `image=0`, so every trigger edge stays in the record. Verified over 88 s at 20 fps:

  ```
  COMPLETE 4-camera sets: 1763 of 1764  (99.9%)
    ragged boundary edges (run start/stop, not loss): 1
  INTERIOR sets: 1763 of 1763  (100.00%)   <- LOSSLESS
  effective SET rate: 20.00 Hz over 88.1 s
  ```

  `index == logged == .raw frame count` exactly on all four cameras, `writer q max 0–1/64`,
  `dropped 0`, worst steady-state skew **1 µs**. The one remaining incomplete set is the final
  trigger edge, which reaches some cameras and not others because the run is stopped while it is
  running — one edge, and a symptom of nothing. `check_log_sets.py` now reports interior
  completeness separately for exactly that reason: a lossless log used to read as 99.9 % and was
  indistinguishable at a glance from real dropping.

  **This also incidentally beat the documented stall rate.** `log_raw.sh` recorded "one brief
  global stall roughly every 80 s" at 20 fps (60 s had none, 90 s had one, 180 s had two). Three
  runs totalling 204 s of steady state produced **zero** interior loss, so that rate is at worst
  pessimistic now — but it was measured over 5.5 min and this is 3.4, so it is not yet retracted.

  **And 30 fps is confirmed impossible on one target, which is arithmetic rather than a bug.**
  Four cameras at 30 fps is 190 MB/s against eMMC's measured 136; the log_rig default put all
  four there and produced **95.4 % complete sets** with losses spread through the run. `log_raw.sh`
  checked *space* but not *bandwidth*, so that sailed through — it fits on disk easily. It now
  **refuses** and names the two fixes (lower the trigger; or split `LOG_DIRS` so each target stays
  inside its own measured limit), with `LOG_BANDWIDTH_OK=1` to override. The documented 30 fps
  split still passes.

  **Practical limit worth knowing:** at 20 fps on eMMC alone the *space* check caps a run at about
  90 s (126.7 MB/s against ~15.8 GB free, with its 1.4× startup allowance). Longer runs need
  `LOG_DIRS` — and note 2 cameras at 20 fps is 66 MB/s, which genuinely exceeds the SD's measured
  62.6, so the split has to be 3 on eMMC + 1 elsewhere rather than 2 + 2.

- [x] 5.0f **A recording's DURATION is not its usable CONTENT — the 90 s log holds 45.7 s of
  motion. Verified 2026-09-05 against the surviving log, not re-recorded.**

  The operator's recollection was that a claimed 90 s 20 fps log turned out to be "only around
  30 s". Both halves of that are right, for two separate reasons, and neither is data loss.

  **The bytes are all there.** `datasets/imglog_vio1_20260903_103102` holds 1774 frames per camera
  (1773 on cam4), and `index_rows × 1584128 == raw_bytes` to the byte on all four — zero shortfall.
  Nor are the frames stale: a 64-row central stripe from every frame gives **0 exact duplicates**,
  longest identical run 2, and no zero-difference frames. So there is no truncation and no frozen
  stream, which was the failure worth ruling out — a silently failing `copyToNvBuffer` would
  re-copy old pixels under fresh timestamps and every existing check would still pass, because
  `check_log_sets.py` reads only index timestamps and `locate_frame_loss.py` only `camN.csv`.

  **But the rig stopped moving 37.5 s before the recording did.** The frame-to-frame luma
  difference collapses from ~1.0–2.6 to ~0.10 after 50 s, and `imu0.csv` — already in the log —
  puts it exactly: |gyro| crosses the still baseline at **+5.8 s** and drops back at **+51.4 s**.

  | | frames | | |
  |---|---|---|---|
  | before motion | 113 | 6.4 % | 5.7 s stationary |
  | **during motion** | **912** | **51.4 %** | **45.7 s — the usable data** |
  | after motion | 749 | 42.2 % | 37.5 s stationary |

  **And the 30 s the operator remembers is the replay subset, which was chosen badly.** 5.0c
  converted `--max-frames 600` — the FIRST 600 frames, 30.2 s, of which only **487 (24.4 s) were
  moving**. So the replay saw about half the motion the log actually contained.

  **Fixed in the two tools that allowed it**, rather than by adding another script (nothing
  existing reads the `.raw` content or the IMU for this: `luma_stability.py` is a live ROS
  subscriber, `analyze_motion.py` reads VO output):
  - `check_log_sets.py` now reports the motion window from `imu0.csv` alongside completeness, and
    says so loudly when over a fifth of a log is a stationary rig. Completeness says the log is
    *intact*; the motion window says it is *useful*; neither implies the other, and printing only
    the first is how "90 s of 20 fps data" got said.
  - `raw_log_to_bag.py --motion` cuts to the moving window (±`--pad-s`, default 1 s) using the
    same threshold via a shared `motion_window()`, so the tool that reports the window and the
    tool that cuts on it cannot drift apart. On this log it keeps **952 frames / 47.6 s** against
    the old subset's 24.4 s of motion. `--max-frames` is now documented as taking the first N.

  Nothing here needs re-recording; the finding is that **runs should be quoted by motion duration,
  and stopped when the motion stops.** 37.5 s of stationary frames cost 3.7 GB of eMMC on a board
  whose free space caps a 20 fps run at ~90 s (5.0e).

- [x] 5.0g **The whole log now replays, on the host, at 15.8 Hz — and it shows the rig walking
  into a sunlit room that saturates all four cameras, which the VO reported as a 50 m teleport.
  2026-09-06.**

  Every §5 number so far came off a *subset*: the TX2 OOMs on a full 20 fps bag, and its
  `Track()` at 50-90 ms/set drops ~45 % of what survives. So an x86_64 offline path now replays
  the same Foxy bags with no TX2 in the loop (`docker-compose.host.yml`,
  `scripts/build_cuvslam_host.sh`, `scripts/vo/replay_host.sh`; `BEV_BUILD_ARGUS=OFF` drops the
  Argus targets so only the modular multicam node builds). Host cuVSLAM is a pristine v17 tree
  for sm_86 — **not** the CUDA-10.2 TX2 port, which the build script refuses to build on.

  **The input is the cleanest log yet.** `imglog_run1_20260906_090227` (2026-09-06, 20 fps):
  1155 of 1155 interior sets, **LOSSLESS**, 20.00 Hz over 57.7 s, and **100 % of frames inside
  the motion window** — 5.0f's lesson applied at record time. IMU lossless at 199 Hz; the range
  channel lossless against the trigger, 1772 readings at 20.01 Hz, its first working full run.

  **The replay itself is healthy.** 937 sets, 907 odometry poses, **15.76 Hz** against the TX2
  replay's 11.18 Hz (5.4), worst skew 1 us, 48 sets dropped on the 1 ms skew gate (5.1 %).
  For the first 43 s the trajectory is plausible: median step 27 mm, median speed 0.49 m/s,
  p95 0.98 m/s — a walk.

  **Then it breaks — and the cause is the scene, not the software.** Re-run at 0.25x to test
  the delivery-timing hypothesis, and the answer split cleanly in two:

  | | 0.5x replay | 0.25x replay |
  |---|---|---|
  | wall throughput vs input | 7.88 Hz vs 10 Hz — **overrunning** | 4.07 Hz vs 5 Hz — keeping up |
  | sets never reaching the matcher | 218 of 1155 | 194 of 1155 |
  | skew-gate drops | 48 | 31 |
  | **jump intervals (> 5 m/s)** | **8** | **2** |
  | path excluding jumps | 30.52 m | 30.71 m |

  So the SMALL discontinuities (0.25-0.73 m) were delivery: `SensorDataQoS()` is best-effort,
  the node was slower than the input, and DDS discarded images silently. Slowing the replay
  removed them, and the surviving trajectory reproduces to 0.2 m.

  **The 50 m jump is not delivery.** It survives at the *identical* sensor stamp in both runs
  (583.784941 -> 583.834926) with the same magnitude (50.18 m / 50.33 m) — but lands somewhere
  different each time, and one run reported a **negative variance**. What actually happens:

  1. From t = 46.4 s cuVSLAM returns the **same pose twelve sets running**, translation
     bit-identical (step exactly 0.000 m), covariance climbing 0.03 -> 2.6. It is not
     measuring; it is repeating the last estimate it was sure of.
  2. Then one pose ~50 m away, `cov[0]` = 381 in one run and **-33.8** in the other — a
     negative variance is a broken solve, not a large uncertainty.
  3. Then normal small steps resume, from a re-initialised origin.

  **Why:** the rig walked up to a blank white wall, at a FIXED 4.986 ms exposure (AE locked
  by design, 4.7 — `exposure [ns]` is the same constant on every row of `cam1.csv`). Between
  the good stretch and the freeze, cam1 goes:

  | | t = 30 s | t = 47 s |
  |---|---|---|
  | frame mean luma | 101.8 | 224.2 |
  | global std | 52.1 | 15.1 |
  | median local 16x16 std | 4.07 | **0.00** |
  | blocks with std < 2 (featureless) | 16 % | **88 %** |

  **It IS overexposure — the operator was right and two of my tests were wrong.** I first
  tested saturation at >= 250, which this pipeline never emits; then at 235, the extreme tail,
  and concluded "only 0.3 % at the ceiling, nothing is clipped". Both missed the actual white
  level. The histogram settles it: at t = 47 s, **88.8 % of cam1's pixels sit at exactly 227**,
  a single spike, with only 3.9 % above it — and every camera has the same mode 227, at
  **88.8 / 90.8 / 94.4 / 97.4 %**. Four independent sensors piling ~90 % of their pixels on one
  code is saturation, not a surface: a real scene at that level still carries shot noise.

  **And it is not a wall.** All four cameras blow at once, in all four directions, which no
  piece of geometry does. The saturated fraction tracks the route:

  | t (s) | frame mean | % pinned at 227 | features/frame |
  |---|---|---|---|
  | 0-9 | 165-209 | 30-68 % | 1887-3222 |
  | **12-42** | **102-152** | **0.6-12 %** | 2000-3400 |
  | **45-48** | **213-216** | **75-83 %** | collapses to 210 |
  | 51-57 | 198-209 | 32-65 % | recovering |

  The run starts bright, crosses a dim room, and re-enters a sunlit one at t ~ 43 s. Fixed
  exposure cannot hold both ends of that range, so the bright end clips to a constant and the
  features die with it. Note t = 6 s survived 68 % saturation with 1887 features — the collapse
  is a threshold effect somewhere above that, not a linear degradation.

  **The actuator is the STM32, not Argus.** AE is locked deliberately (4.7): under external
  trigger AE cannot reach its main actuator and hunts on gain instead. Exposure here IS the
  trigger pulse width — `ch_exposure_us=5000`, `pulse_ns=4985740` — so the fix is
  `j106-trigctl.py`, and "turn AE back on" is not available. What is still open is which pulse
  width covers the route; that needs a bracket, not a guess, because the clipped frames do not
  say how much brighter than 227 the bright room actually was.

  Secondary: the pipeline emits limited range (floor 16, tail 235), so some contrast is being
  discarded before any of this; and focus at close range is unverified (3R.5 still open).

  **cuVSLAM itself never said a word, and could not have.** `Track()` returned a valid pose on
  every one of those sets — `grep -c "tracking lost (no pose)"` over the replay log is **0** —
  so the library's own tracking messages never applied: all three of them
  ("images are not available", "Failed to track on the 2D tracking stage", "Failed to track on
  the PnP stage", `libs/odometry/multi_visual_odometry_base.cpp`) sit on paths that `return
  false`, which `cuvslam2.cpp` turns into an empty `world_from_rig`. Our freeze is the opposite
  case: `solveNextFrame` SUCCEEDED and produced `delta = prev.inverse() * world_from_rig`
  ~ identity, so `increment_pose` re-emitted the previous pose bit-for-bit. And `PoseEstimate`
  carries no status field at all — only `timestamp_ns` and `optional<world_from_rig>` — so
  there is no "degraded" flag to read. On top of that the library's logging is **off by
  default** (`SetVerbosity(0)`) and this node never called it; a `cuvslam_verbosity` parameter
  now exposes it, though for this failure it would have printed nothing.

  The one signal the library does give is the covariance, and that is why the negative variance
  matters: `static_pose_covariance_exp = static_info_exp.ldlt().solve(I)` inverts the
  information matrix, and with no features that matrix is rank-deficient, so the solve returns
  garbage rather than a large-but-valid uncertainty. A negative diagonal is the tracker saying
  the pose was unconstrained, in the only language it has.

  **The node published all of it without comment**, because `track_and_publish` only checked
  that `world_from_rig` was present. Fixed: `check_pose_health()` now warns on a bit-identical
  repeated pose, errors on a step over 5 m/s, and errors on a negative covariance diagonal.
  `scripts/vo/analyze_motion.py` gained the matching offline gate and now **suppresses** the
  5.1/5.2 verdict on a discontinuous run instead of averaging across a teleport — it was
  previously happy to report "89 m path, 45 m displacement" for this run. It also timed on the
  bag RECEIVE clock, which on a 0.5x replay halved every speed; it uses header stamps now.

  **The texture collapse is gradual and starts at t = 43 s**, and the run never recovers:
  featureless blocks go 7-16 % over t = 20-42 s, then 37 % at t = 43 s, 74 % at t = 45 s,
  88 % at t = 47 s, 45 % at t = 50 s, 53 % at t = 55 s.

  **Inside t = 1-43 s the result IS reproducible.** Three replays (0.5x, 0.25x, 0.25x with the
  health check) give a clean path length of **21.10 / 20.29 / 20.39 m** — about 4 % spread —
  with zero jumps in two of them. The third's nine "jumps" all sit between t = 0.50 and 0.95 s
  and are cuVSLAM's initialisation transient, so `analyze_motion.py --skip-s` (default 1.0 s)
  now excludes and names that window rather than suppressing the verdict over it.

  **What this costs §5:** run1 is unusable past t = 43 s, and `tape_metres.txt` was never
  written for it anyway, so 5.1 still has no measured run. Usable window: **t = 1-43 s**.

  **What the next recording has to do differently**, all three cheap:
  1. **Bracket the trigger pulse width over the actual route.** 4986 us is right for the dim
     room (mean 102, 3 % saturated) and far too long for the sunlit one (mean 216, 89 %
     saturated). Record the same walk at a few widths via `j106-trigctl.py` and take the one
     where neither end clips nor starves. Re-enabling AE is NOT an option here (4.7).
  2. Watch the new saturation warning while recording — it names the condition live, which is
     the whole point of adding it.
  3. Write `tape_metres.txt`. Without it the run cannot answer 5.1 no matter how clean it is.

  **Caveat on the range channel:** `range_cm` reads a flat 0.28-0.35 m right through the
  freeze while the rig is walking, and 46 % of the run is under 0.2 m. It is not seeing the
  scene — it is pointed at the rig or the operator. 5.5b needs it re-aimed before it can
  regress anything, and it is NOT independent confirmation of the wall.

- [ ] 5.1 Move the rig a measured straight-line distance; record `/cuvslam/odometry` + `/tf` and compare reported translation against the tape measure (spec: *Translation is recovered at true scale*, 5 %).
      **Now the highest-value open item in this change.** `add-replay-visual-diagnostics` 3.5 finds the
      floor **1.36 m** below a head-carried rig, against an expected ~1.60–1.75 m — a suspected
      **15–20 % scale underestimate**. Scale enters only through the extrinsic translations in
      `config/rig/rig_extrinsics_imx296.yaml`, so if this confirms, the fix is there and never a
      correction factor on the odometry. A tape measure settles it for free; do this before building
      any ToF or T265 comparison.

- [ ] 5.2 Return the rig to its starting pose and check the trajectory returns near the origin; record the drift.
      Partial evidence in 5.0d(3): ±0.4 m vertical wander, attributed to the absence of a gravity
      reference. A return-to-origin run is still required for the horizontal figure.
- [~] 5.3 **Answered offline, pending live confirmation.** cuVSLAM does not take declared stereo pairs: it samples a grid per camera, back-projects to 2 m and 4 m, and connects pairs exceeding 0.5 (`frustum_intersection_graph.cpp:33`). Re-running that on our ring-closed rig gives **0.939 / 0.951 / 0.926 / 0.949** against a 0.961 ceiling, all 8 virtual cameras paired, no spurious edges. The links form with room to spare and the `CUVSLAM_FRUSTUM_THRESHOLD` patch is not needed for the pinholes. Still to confirm on the live pipeline.
- [~] 5.4 **Rate confirmed on the 4-camera replay, tracking/drift pending live ground truth.**
  *(Rate superseded by 5.0g: 15.76 Hz on the host against the 11.18 Hz below. And tracking does
  NOT hold for the whole run — it freezes at t = 46.4 s on a featureless white wall and
  re-initialises 50 m away. That is a data-collection failure, not a tracking one.)* The
  2026-09-03 replay (5.0c) produced **11.18 Hz** odometry from a 20 Hz input — above the old bundled
  rig's ~8.5 Hz — with cuVSLAM tracking held for the whole 29 s run. It stays compute-limited on the
  TX2: `Track()` at ~50–90 ms/set means ~45 % of the 20 Hz sets never reach the tracker, and the
  survivors occasionally straddle a trigger boundary (the 17 % skew drops). Rate beats the old rig;
  **drift and "does tracking survive motion that broke the old rig" still need a run with ground
  truth**, since this pre-recorded log carries none.

- [ ] 5.10 **Replay loses ~1 set in 6 to DDS, at every rate. Fix the QoS, not the rate.**
      Measured 2026-09-06 on `run1_motion`, and it corrects 5.0g, which blamed the node being
      slower than the input:

      | replay rate | sets reaching the matcher | skew-gate drops | poses |
      |---|---|---|---|
      | 0.25x | 961 / 1155 (83 %) | 31 | 933 |
      | 0.5x | 937 / 1155 (81 %) | 48 | 907 |
      | 1.0x | 967 / 1155 (84 %) | 36 | 961 |

      **Not the conversion.** The bag's own header stamps are max **1 us** skew with zero sets
      over 1 ms, so `raw_log_to_bag.py` preserves the trigger's sync exactly.

      **Not compute either.** Instrumented `Track()` now reports per window: on the host it is
      **6.5-10 ms mean, ~15 ms max**, with the remap ~5 ms - about 15 ms/set, so ~65 Hz. At 1.0x
      the node sustained 16.7 Hz, i.e. every set that reached it. Compare the **TX2 at 50-90 ms**
      per set (5.4), which genuinely is compute-bound at ~11 Hz. Two different limits, and only
      the TX2's is the one 5.4 described.

      What is left is transport: `SensorDataQoS()` is **best-effort** over four 1.5 MB streams,
      DDS drops fragments silently, the orphans pair across trigger edges, and the set fails the
      1 ms gate at exactly one frame period. To do:
      - [ ] 5.10a Make the subscriber QoS a parameter and default replay to **RELIABLE** with a
            deeper queue; measure sets-reaching-matcher again at 1.0x.
      - [ ] 5.10b If reliable is not enough, raise the DDS buffers (Cyclone
            `MaxMessageSize`/`FragmentSize`, socket rcvbuf) - 20 Hz x 4 x 1.5 MB is 127 MB/s.
      - [ ] 5.10c **Bypass DDS for replay: read the bag in-process and call `Track()` directly.**
            Promoted from "consider" — it is now the blocker for every offline comparison, not a
            tidy-up. **It must be paired with `async_sba=false`**, and neither half works alone:

            - the reader alone removes frame drops but leaves BA on a background thread, whose
              iterations-per-frame follow the WALL CLOCK. Calling `Track()` in a tight loop just
              gives it a different wrong budget rather than a right one.
            - `async_sba=false` alone was tried on 2026-09-06 and made things worse: synchronous
              BA is slower, so with DDS still in the loop it dropped MORE sets (1022 vs 1153)
              and the two-rate divergence went from 1.13 m to 3.28 m.

            Together they give a fixed amount of optimisation per FRAME instead of per second,
            and two runs of the same bag become identical.

- [ ] 5.13 **Every host-side figure in this change answers "is A better than B on this data",
      never "what will the rig do".** The two need different setups and the distinction has been
      blurred throughout:

      | question | setup | property |
      |---|---|---|
      | is loop closure better than pure VO? | 5.10c + `async_sba=false` | deterministic, every frame, fair A/B |
      | what will the rig actually do? | TX2, sensor rate, async on | the real answer, and worse |

      **Corrected after reading the source — the wall-clock story here was wrong.** It said SBA
      is a free-running optimiser that "gets more iterations" the more wall-clock time it has.
      It is not. `pipelines/track_online_multi.cpp` triggers SBA **once per KEYFRAME**, inside
      `if (frameState == FrameState::Key)`, and each trigger is one bounded pass. `map/service.cpp`
      makes `inputs_ready_` a **flag, not a queue**: two keyframe triggers arriving while a pass
      runs coalesce into one.

      So the ceiling is **one SBA pass per keyframe**, and replay rate changes how many passes
      COALESCE, never how many iterations each does. A slow replay lets each keyframe get its own
      completed pass — approaching what the algorithm intends. A fast replay on a machine that
      cannot keep up merges passes and delivers **less** than intended. 0.4x is therefore not
      "over-optimised": it is at or near the design maximum, and 1.0x on this host may be below
      it. On the TX2, with `Track()` at 50-90 ms against a 50 ms budget, coalescing is guaranteed.

      The 1.67 m difference between the 1.0x and 0.4x lossless runs stands as a measurement; only
      the explanation of it changes.

      So: quote host replay results as replay results. The board's number has to come from the
      board.

- [ ] 5.14 **Keyframe fraction is a tracking-health signal, and it is free.** `Track()` is not the
      same work every frame: a normal frame solves PnP against the recent landmarks, a keyframe
      also triangulates, calls `map_.add_keyframe()` and triggers SBA. Measured on the host with
      SLAM off: **keyframe 16.6-20.7 ms against non-keyframe 12.1-13.4 ms**, so one mean over both
      hides a bimodal distribution and makes the keyframe cost read as jitter.

      The fraction is the useful part. The tracker declares a keyframe when it cannot carry on
      against the existing landmarks, so on run1:

      | t | keyframe % |
      |---|---|
      | 5-35 s | 54 -> 24 %, declining, healthy |
      | **45 s** | **100 %** |
      | 50-55 s | 97 %, 94 % |

      It hits 100 % in exactly the saturated window (5.0g). **"Every frame is a keyframe" is what
      tracking distress looks like from the inside** — the tracker re-anchoring constantly because
      nothing matches — and it costs triple, since SBA then fires every frame. The node reports the
      split and warns above 90 %. `State::keyframe` makes it free wherever the export is already on.

- [ ] 5.11 **Overexposure: the knob is GAIN, and it is pinned at 64x.** 5.0g said to shorten the
      trigger pulse. That works but is the harder half. `argus_capture_node` clamps, under AE
      lock, `setGainRange(16,16)` analog and `setIspDigitalGainRange(4,4)` digital - **64x total,
      fixed** - because AE under external trigger cannot reach its exposure actuator and hunts on
      gain (4.7). The clamp cured the hunt and nobody revisited the level.

      So brightness = 4.986 ms x 64. Gain is the free knob: it needs no MCU command and, unlike
      the pulse width, **does not move the exposure-midpoint stamp**, so nothing downstream has
      to re-read `exposure_us`. To do:
      - [ ] 5.11a Bracket `ae_gain` / `ae_dgain` over the sunlit room and find the pair that
            keeps the bright end under the 227 white level while the dim room still has features.
            Cheapest possible experiment: two parameters, one walk.
      - [ ] 5.11b Only if gain alone cannot span the route, bracket the pulse width too - and
            then handle the stamp: `exposure_us` is a static parameter today, so a changing pulse
            width silently corrupts the exposure-midpoint timestamp.
      - [ ] 5.11c **Closed-loop AE on the MCU pulse width** is the real answer for a route that
            spans rooms, and the actuator exists (`j106-trigctl.py`). Design it only after
            5.11a/b say how much range is actually needed. It must publish the commanded width
            per frame, not assume it.

- [ ] 5.12 **Auto-exposure for a route that spans rooms — the four levers, cheapest first.**
      The bind: exposure IS the trigger pulse width, so Argus AE cannot reach its main actuator
      and hunts on gain (4.7). "Turn AE back on" is not an option. What is:

      1. **Lower the fixed gain.** 16x analog x 4x digital = 64x is most of why 4.986 ms clips a
         sunlit room, and no dynamic control is needed to improve it. **Bias dark, not bright:**
         cuVSLAM tracks gradients, so an underexposed frame with 20 levels of local contrast
         still tracks while a clipped one has exactly zero — which is the t = 47 s failure. The
         run agrees: the dim room (mean 102) gave the most features, 2500-3400, and t = 6 s at
         68 % saturation still managed 1887. **Untested floor:** this log never went below mean
         ~100, so how dark is too dark is not known. Establish it while doing 5.11a.
      2. **Stop discarding range.** Output is limited-range 16-235, so ~14 % of the code space
         is gone before anything else. The sensor is 10-bit and we emit mono8; capturing 10-bit
         would give 4x the room to place the scene in.
      3. **Slow closed-loop GAIN control.** Argus gain is settable per request at runtime, so a
         custom controller — the saturated fraction `check_exposure()` already computes, with
         rate limiting and hysteresis — avoids the 3.5 Hz hunt that Argus's own AE produced. No
         MCU involvement, and it does **not** move the exposure-midpoint stamp. Must stay
         rig-wide: one gain for all four cameras.
      4. **Pulse-width AE** (5.11c) only if gain alone cannot span the route.

      **Explicitly rejected:** per-camera exposure or gain — it steps the panorama seams and
      mismatches the virtual-stereo pairs, which is the same reason exposure is rig-wide.
      Alternating short/long HDR frames — it breaks the synchronised-set assumption the whole
      rig is built on, and halves the rate.

## 6. Wrap-up

- [ ] 6.1 Tick `bring-up-end-to-end-vo` tasks 3.4/3.6 with the evidence from §5, or state precisely why they remain open.
- [ ] 6.2 Update `README.md` and `docs/` for the IMX296 rig: population, trigger prerequisite, `jetson-clocks`, new resolution, calibration layout. **Must also carry 3R.16b's outcome:** the odometry child frame is `cam1_optical_frame`, not `base_link`, and what a consumer has to compose to get a vehicle frame.
- [ ] 6.3 Update the project memory notes with the measured outcome (skew, rate, whether tracking is metric, Δ).
      Partly done: rig conventions, the row-vector frame trap, the drift/scale findings and the
      visualisation gotchas are in the repo memory notes and in `add-replay-visual-diagnostics`.
      "Whether tracking is metric" remains open pending 5.1.
- [ ] 6.4 Archive this change once §5 has a verdict.
