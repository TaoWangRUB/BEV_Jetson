# Scripts

Index of the helper scripts in this folder. "Runs on" = **TX2** (board, usually inside
`cuvslam-foxy:tx2`), **host** (laptop inside `bev-host-cuvslam`), or **dev** (workstation
python3 + opencv + numpy, no GPU VO).

**How to use them end to end** (run VO, log raw data, analyse): see
[README §3 Procedures](../README.md#3-procedures). This file is the catalogue; that section is the
runbook. Docker split: [docker-compose.yml](../docker-compose.yml) (TX2) vs
[docker-compose.host.yml](../docker-compose.host.yml) (host offline VO).

## Top level — build & run

| script | runs on | purpose |
|---|---|---|
| [setup_tx2_docker.sh](setup_tx2_docker.sh) | TX2 | one-time JetPack 4.6 / L4T R32.7 Docker prep for the BEV/cuVSLAM stack |
| [build_cuvslam_tx2gpu.sh](build_cuvslam_tx2gpu.sh) | TX2 | build cuVSLAM's GPU path on CUDA 10.2 with gcc-8 + C++14 (applies the port patch) |
| [build_cuvslam_host.sh](build_cuvslam_host.sh) | host | build cuVSLAM for x86_64 / CUDA 12.x (no TX2 port; `build_host/`) |
| [run_vo_fused_tx2.sh](run_vo_fused_tx2.sh) | TX2 | wrapper: run the **fused** zero-copy Argus→cuVSLAM VO (`docker compose`) |
| [run_vo_tx2.sh](run_vo_tx2.sh) | TX2 | wrapper: run the **modular** capture + cuVSLAM VO (`docker compose`) |
| [capture_montage_tx2.sh](capture_montage_tx2.sh) | TX2 | capture the 4 fisheye views + stitched panorama, montage into one image |

## calib/ — camera calibration

| script | runs on | purpose |
|---|---|---|
| [calib/online_calib.py](calib/online_calib.py) | dev/TX2 | online interactive fisheye (Kannala-Brandt) **intrinsic** calibration |
| [calib/offline_calib.py](calib/offline_calib.py) | dev | offline outlier-rejection fisheye **intrinsic** calibration |
| [calib/scale_calib.py](calib/scale_calib.py) | dev | **retired (IMX219/KB)** — exits; scaled KB `camN.yaml` to a downscaled output resolution |
| [calib/grid_view_tx2.sh](calib/grid_view_tx2.sh) | TX2 | live 2×3 IMX219 grid on the TX2 HDMI display (identify/label cameras) |
| [calib/capture_calib_sets.sh](calib/capture_calib_sets.sh) | TX2 | grab N raw 4-cam sets for **extrinsic** calibration (pan the rig) |
| [calib/extrinsic_calib.py](calib/extrinsic_calib.py) | dev | **retired (IMX219/KB)** — exits; feature-based relative-rotation **extrinsic** calibration. Port via `mei_project()` in `bev_panorama_node.cpp` |
| [calib/exposure_bracket.sh](calib/exposure_bracket.sh) | dev | sweep the rig-wide gain and report what each setting does to the histogram. Run it standing still, once in the BRIGHTEST part of the route and once in the DIMMEST |
| [calib/exposure_report.py](calib/exposure_report.py) | dev | clipping and surviving local contrast for one raw log. Detects the white level as the modal value, not `>=250` — this pipeline emits limited range and never reaches 250 |
| [calib/pano_tuner.py](calib/pano_tuner.py) | dev | **retired (IMX219/KB)** — exits; interactive panorama extrinsics tuner (web UI) |

| [calib/calibration_pipeline.ipynb](calib/calibration_pipeline.ipynb) | dev | **the end-to-end runbook** — every command actually used, each failure and why. Start here before any calibration work |
| [calib/record_calib_session.sh](calib/record_calib_session.sh) | TX2 | record one staged ROS session (preflight refuses a rig with the trigger off); **the only path with capture timestamps**, so the camera↔IMU stage must use it |
| [calib/pair_extrinsics.py](calib/pair_extrinsics.py) | dev | solve ONE adjacent pair's extrinsic from simultaneous target views (Kalibr detector + header-stamp matching; runs in the tartancalib container) |
| [calib/filter_bag.py](calib/filter_bag.py) | dev | pick the usable, coverage-spread frames from a ROS1 bag (runs in the tartancalib container; >=3 tags clears Kalibr's DLT floor) |
| [calib/select_frames.py](calib/select_frames.py) | dev | pick the frames worth solving: greedy on coverage ADDED against a per-cell quota, ties on sharpness over the target's bounding box |
| [calib/extract_quarterkalibr_bags.py](calib/extract_quarterkalibr_bags.py) | dev | split one session bag into the per-stage bags the solver takes |
| [calib/close_rig_ring.py](calib/close_rig_ring.py) | dev | make four independently-solved pairs one rigid body (18 dof, LM); prints per-edge corrections and the epipolar cost it bought |
| [calib/gen_virtual_stereo.py](calib/gen_virtual_stereo.py) | dev | carve each fisheye into two virtual pinholes at ±45° — required, since cuVSLAM's only fisheye model caps below 180° |
| [calib/vstereo_epipolar.py](calib/vstereo_epipolar.py) | dev | measure a virtual pair's epipolar residual by tag identity (ORB is worthless on a repetitive grid) |
| [calib/vstereo_disparity.py](calib/vstereo_disparity.py) | dev | dense disparity on a virtual pair — *not* evidence on a calibration sweep; needs a textured scene at 1–3 m |
| [calib/regen_vstereo.sh](calib/regen_vstereo.sh) | dev | regenerate all four virtual pairs on a closed rig and report measured vs closed |
| [calib/fold_roll_for_vo.py](calib/fold_roll_for_vo.py) | dev | **retired (IMX219/KB)** — exits; folded the 180° mounting roll into the VO extrinsics. The IMX296 calibration is solved on raw inverted frames, so no fold is needed |
| [calib/rig_design.py](calib/rig_design.py) | dev | rig geometry helper |
| [calib/cuvslam_frustum_check.py](calib/cuvslam_frustum_check.py) | dev | re-run cuVSLAM's own frustum-overlap test on our poses (its 0.5 threshold is hard-coded) |

See [docs/extrinsic_calibration.md](../docs/extrinsic_calibration.md) for the full extrinsic-calibration procedure.

## port/ — cuVSLAM CUDA-10.2 port & measurement

| script | runs on | purpose |
|---|---|---|
| [port/build_and_validate.sh](port/build_and_validate.sh) | TX2 | one command: build the Foxy image, build the port, run the smoke test |
| [port/smoke_test.cpp](port/smoke_test.cpp) | TX2 | runtime smoke test for the CUDA-10.2 cuVSLAM port (WarmUpGPU) |
| [port/egl_cuda_spike.cpp](port/egl_cuda_spike.cpp) | TX2 | EGL→CUDA zero-copy bridge spike (validates the NVMM→CUDA device pointer) |
| [port/downgrade_cuvslam_cpp17.py](port/downgrade_cuvslam_cpp17.py) | TX2 | downgrade cuVSLAM C++17 device syntax to C++14 for nvcc 10.2 |
| [port/regen_cuvslam_patch.sh](port/regen_cuvslam_patch.sh) | host | regenerate patch/cuvslam/0001-cuda102-tx2-port.patch after a cuVSLAM submodule bump |
| [port/regen_cunls_patch.sh](port/regen_cunls_patch.sh) | host | regenerate patch/cunls/0001-cuda102-tx2-port.patch (cuNLS, needed for USE_CUNLS/Multisensor) |
| [port/grab_views.py](port/grab_views.py) | TX2 | grab frames from ROS 2 image topics and/or montage 4 cams + panorama |
| [port/sync_check.py](port/sync_check.py) | TX2 | measure the timestamp spread across N camera topics |
| [port/topic_rate.py](port/topic_rate.py) | TX2 | count messages on topics over a window and print the rate |
| [port/luma_stability.py](port/luma_stability.py) | TX2 | brightness stability per camera (catches AE gain-hunting under external trigger) |
| [port/trigger_probe.py](port/trigger_probe.py) | TX2 | what the trigger is really doing, from raw V4L2: `--sweep` proves commanded exposure == asserted pulse (polarity), `--flicker` tests for mains beat. No ROS |

## stream/ — live preview and calibration capture

**Two pairs, and they are not interchangeable.** Pick by the job, not by the topic:

| job | use | why not the other |
|---|---|---|
| **look** at the cameras — focusing a lens, "is this one alive?", checking the trigger | `csi_sender.sh` + `csi_receiver.sh` | H.264/RTP/UDP into a native window: low latency, tracks your hand |
| **record** calibration data | `calib_sender.sh` + `calib_receiver.py` | MJPEG (no inter-frame artifact can smear a tag corner) + live AprilGrid coverage; its browser preview lags ~1 s, which is fine for filling a coverage grid and useless for focusing |

| script | runs on | purpose |
|---|---|---|
| [stream/csi_sender.sh](stream/csi_sender.sh) | TX2 | stream present cameras as H.264/RTP/UDP to the host, ISP-flipped upright, AE locked under trigger (`./csi_sender.sh [HOST_IP]`). `W=1456 H=1088 BR=16000000` for a **focus** check — the 640×480 default downscales away the detail you are judging |
| [stream/csi_receiver.sh](stream/csi_receiver.sh) | dev | receive into a labelled port grid; missing cams show "no signal". `PORTS="c" CW=1456 CH=1088 JITTER=20` for one camera at native size |
| [stream/calib_sender.sh](stream/calib_sender.sh) | TX2 | calibration capture: Argus → **hardware JPEG** → MJPEG over TCP 5000–5003, board does nothing else (load ~0.3). `RECORD_DIR=` also tees to board storage and reports per-camera frame counts on exit |
| [stream/calib_receiver.py](stream/calib_receiver.py) | dev | host end: decode, AprilGrid detection, **live coverage grid**, record frames. `--auto` keeps frames per coverage bin by itself |
| [stream/board_sender.sh](stream/board_sender.sh) | TX2 | capture + per-frame timing CSV + DDS publish, and nothing else — the ROS-path equivalent of `calib_sender.sh` |
| [stream/preview_server.py](stream/preview_server.py) | dev | standalone MJPEG preview server |

## vo/ — visual odometry bring-up and motion tests

| script | runs on | purpose |
|---|---|---|
| [vo/replay_host.sh](vo/replay_host.sh) | host | bag → modular cuVSLAM on x86 (`docker-compose.host.yml`); records `/cuvslam/odometry` + `/tf` under `datasets/replay_out/` |
| [vo/run_motion_test.sh](vo/run_motion_test.sh) | TX2 | record a motion test (§5). Preflight **gates, not warns**: refuses unless `trigger_mode=1`, the generator is running and `active_low`, and it reads the real pulse width back so `exposure_us` is never a stale guess. `--record-images` also bags the four camera streams, making the run replayable — move the rig once, re-run the VO against it as often as needed. Do two passes: one with images for replay, one without for the live numbers, because the recorder's own load is indistinguishable from the rig misbehaving |
| [vo/analyze_motion.py](vo/analyze_motion.py) | dev | odometry health then verdict: **continuity first** (frozen poses, teleports, gaps), then scale/drift/return-to-origin. `tape_metres.txt` optional; a discontinuous run suppresses the verdict rather than averaging over it |
| [vo/rerun_multicam.py](vo/rerun_multicam.py) | dev | **the replay viewer.** 8 virtual-pinhole panes with the tracked features, 4 raw fisheyes, landmark map and trajectory in one Rerun timeline. Two bags: odometry+observations first, source images via `--images`. `--t-range 40:52` to focus a window, `--panorama` / `--bev-fit-plane` for the stitch prototypes |
| [vo/make_panorama.py](vo/make_panorama.py) | dev | **360 panorama straight to a PNG/mp4**, from a raw image log or a camera bag — no Rerun scene and no TX2. Reuses `pano_maps`/`render_pano` from the viewer, defaults matched to `panorama_params.yaml`. `--depth 4.0` (finite sphere) vs `--depth inf` (rotation-only, ghosts everything indoors) |
| [vo/rerun_odometry.py](vo/rerun_odometry.py) | dev | lighter Rerun view: trajectory + landmark cloud + raw camera frames, no virtual pinholes |
| [vo/rerun_virtual_pinholes.py](vo/rerun_virtual_pinholes.py) | dev | just the 8 carves cuVSLAM consumes, from a camera bag alone — no VO run needed |
| [vo/render_multicam_video.py](vo/render_multicam_video.py) | dev | the same layout as an **mp4**/gif. Rerun cannot export video (only `--screenshot-to`, one frame), so this is the only way to get the scene as a movie. `--slam --panorama --bev-fit-plane` mirrors the .rrd: loop closures and hairline edges on the 3D panel, BEV on the left third and the 360 panorama on the right two thirds |
| [vo/bench_remap.cpp](vo/bench_remap.cpp) | TX2 | measure the virtual-pinhole remap against virtual and source resolution, standalone (no cameras, no ROS). Answers "should we lower the resolution" with a number: both axes are bad trades (task 4.5b) |
| [vo/verify_rig_build.sh](vo/verify_rig_build.sh) | dev | re-run cuVSLAM's frustum test on the poses the C++ actually emits — run it **before** a board session, or a bad rig file reads as a wiring bug |
| [vo/check_rig_poses.py](vo/check_rig_poses.py) | dev | sanity-check the rig poses fed to cuVSLAM |
| [vo/compare_modes.py](vo/compare_modes.py) | dev | A/B VO replays of one bag: `Track()` cost from the `_timing.csv` (`TIMING=1`), continuity, SE(3) shape difference against a reference, and scale against a slam-rs run. Include a same-config repeat - replays are not reproducible |
| [vo/verify_fig_primaries.sh](vo/verify_fig_primaries.sh) | dev | **does cuVSLAM actually track all 8 virtual cameras?** Runs the library's own frustum-graph selection on our topology - no GPU, no images, ~1 s. Also prints which primaries `performance` / `moderate` would keep (one per pair: 0 1 2 5). cuVSLAM declares camera 0 a depth camera on every Multicamera run and silently drops its stereo partner; on a ring of disjoint pairs that is one camera lost in every run. Needs patch/cuvslam/0003 |
| [vo/audit_frames.py](vo/audit_frames.py) | dev | **did every frame get tracked?** Closes a ledger between a camera bag and a replay's `/cuvslam/odometry`: an upper bound computed from the bag alone (nearest partner per cam1 frame vs the skew gate), against the poses actually published. Reports duplicates and splits losses into startup / shutdown / **mid-run** - only mid-run is a real loss |
| `docker compose run --rm logonly` | TX2 | **raw 4-camera image log, no ROS/DDS in the path.** Prefer [log_rig.sh](log_rig.sh) for cameras+IMU+range. See README §3.2 for rates |
| [docker_publish.sh](docker_publish.sh) | TX2 | push `cuvslam-foxy:tx2` to Docker Hub as `wtlove876/cuvslam-foxy:tx2` plus a dated tag |
| [port/check_log_sets.py](port/check_log_sets.py) | dev | set completeness + IMU/range health for a raw log — see README §3.3 |
| [port/locate_frame_loss.py](port/locate_frame_loss.py) | dev/TX2 | **where** a triggered rig loses a frame (sensor vs consumer) |
| [port/raw_log_to_bag.py](port/raw_log_to_bag.py) | dev | convert a raw image log into a **rosbag2 bag format v4** (Foxy-readable); `--motion` cuts to the IMU motion window |
| [port/topic_rate_probe.py](port/topic_rate_probe.py) | TX2 | count what a subscriber actually receives, to separate DDS from whatever consumes it |

### Running the Rerun viewer

`rerun-sdk` is not a system package here (PEP 668). The repo keeps a venv with system
site-packages, so `rosbags`, `cv2` and `numpy` come from the system and only rerun is local:

```bash
python3 -m venv --system-site-packages .venv     # once; needs python3.12-venv
.venv/bin/pip install rerun-sdk
```

It takes **two bags**: the VO output (with `/cuvslam/observations`, so record it with
`OBS=1 scripts/vo/replay_host.sh`) and the source images.

```bash
.venv/bin/python scripts/vo/rerun_multicam.py \
  datasets/replay_out/obs_run1_wall --images /tmp/run1_motion.bag \
  --t-range 40:52 --frames 240 --save /tmp/run1_wall.rrd
```

`--save` writes an `.rrd` to open later (`rerun file.rrd`), `--spawn` opens the desktop
viewer, `--serve` serves it over the web. **Port convention for `--serve`:** `--port` alone
still binds 9090 for the UI, so two concurrent servers need `--port` *and*
`--web-viewer-port`.

Without `--t-range`, `--frames` subsamples the whole run — on a 57 s log that is one pose in
five, enough to miss a 0.75 s tracking freeze completely.

**The full scene** adds the three prototype panes on top of the pinholes and trajectory:

```bash
.venv/bin/python scripts/vo/rerun_multicam.py \
  datasets/replay_out/obs_run1_wall --images /tmp/run1_motion.bag \
  --frames 1000 --map-radius 20 \
  --bev-fit-plane --panorama --fisheye \
  --save datasets/replay_out/run1_scene.rrd
```

- `--bev-fit-plane` — ground-plane BEV, with the plane **fitted per frame** from the landmarks
  around the current pose. Use this rather than `--bev-height`: `config/rig/ground_plane.yaml`
  is `status: unmeasured`, so any height passed by hand is provisional and assumes the floor is
  level. The per-frame fit is also drift-immune, which matters on a head-carried rig.
- `--panorama` — equirectangular 360 stitch, `--pano-depth auto` (the default) tracking the
  landmark cloud per frame. `inf` is rotation-only and ghosts everything within a few metres,
  because it collapses the four ~0.155 m baselines to one point.
- `--fisheye` — the raw ~192° lenses as textured spherical caps, 11 180 triangles each, since
  `rr.Pinhole` cannot express a field beyond 180°.

Budget roughly **1.2 s and 0.45 MB per frame** with all three on: a 965-frame run is about
20 minutes and 430 MB, against 4 minutes and 280 MB without them. Read the BEV pane with the
saturation warning in mind — where the cameras are blown the landmarks are junk, so the fitted
plane wanders (h 2.1–2.9 m across the t = 45–47 s window, against the 1.36 m measured
elsewhere).

## slamrs/ — slam-rs (Rust Basalt sqrt VIO) on raw logs

[slam-rs](https://github.com/rerun-io/examples-monorepo/tree/main/packages/slam-rs) runs any
number of cameras **with** the IMU and needs no camera overlap, which cuVSLAM's multicam mode
cannot. These feed it a `log_rig.sh` directory directly through its PyO3 API — no RRD, no catalog.
Build the core once in a checkout (`cargo build --release -p slam-rs-py`, then copy
`target/release/lib_core.so` to `slam_rs/_core.so`) and point `SLAM_RS_DIR` at it.

**On the TX2** (`build_slamrs.sh` with `PYTHON=` a conda-forge 3.12 env; 11 min build) slam-rs
gives the host's result - same sets tracked, positions within 1-14 mm median - at ~5x the cost.
Measured 2026-09-28, MAXN, alongside cuVSLAM replaying 20 s clips of the same logs at 20 fps
(`replay_host.sh` runs on the board too; patch 0003 applied):

| per camera set, median | host | TX2 | TX2 at 20 fps |
|---|---|---|---|
| slam-rs VIO, 1 thread (full logs) | 35-45 ms | 184-223 ms | no: offline only, ~5 Hz |
| slam-rs VIO, 4-6 threads (run6, 500 sets) | - | 148-151 ms | no: ~6.7 Hz, the estimator is one thread |
| cuVSLAM VO precision, `Track()` | 6-8 ms | 54-73 ms (callback 82-100) | no: 46-58 % of sets kept |
| cuVSLAM VO performance, `Track()` | 3-4 ms | 24-44 ms (callback 51-69) | borderline: 72-91 % kept |
| slam-rs VIO, GPU lane (frontend on wgpu, clocks locked) | 21-27 ms | not measured | - |

The modular node's remap is ~26 ms of each TX2 callback; the fused node does it on the GPU.

**The GPU lane** (`GPU=1 build_slamrs.sh`, `run_slamrs.py --gpu`) moves the frontend - pyramid, FAST
detection, KLT, stereo - onto CubeCL kernels through wgpu/Vulkan; the estimator stays one CPU thread in
either lane, by upstream's design. Measured 2026-10-16 on the host's RTX A2000 against the CPU lane of
the same build on the same logs (`compare_lanes.py`); the CPU lane of the `gpu-wgpu` build is
byte-identical to the CPU-only build:

| run5 / run6, median per set | CPU lane | GPU lane, clocks locked | GPU lane, clocks managed |
|---|---|---|---|
| `track()` | 51 / 44 ms | **21 / 27 ms** | 33 / 41 ms |
| frontend | 36 / 24 ms | 5.0 / 4.8 ms | 17 / 20 ms |
| estimator (`measure`) | 8.7 / 14.0 ms | 10.2 / 15.6 ms | 11.9 / 17.5 ms |
| sets tracked | 1756 / 1746 | same | same |
| position vs the CPU lane, RMS after rigid alignment | - | 1.1 / 0.3 cm | 1.1 / 0.3 cm |

So the lane is valid - identical tracking, trajectories within a centimetre, which is upstream's own
bar (design notes D64 / D83: inside the accuracy band and faster than the CPU lane). Two things to
know before quoting its speed:

- **Lock the clocks.** The lane loads the GPU 10-20 %, so the driver drops the A2000 from 1357 to
  ~300 MHz a few seconds in and every round trip takes 4x longer: the frontend goes 5 -> 20 ms and
  stays there even when the window is nearly empty. `sudo nvidia-smi -lgc 1357,1357` and
  `sudo nvidia-smi -lmc 5501,5501` for the measurement, `-rgc` / `-rmc` after. On the TX2 that is
  what `jetson_clocks` does.
- **The estimator is the floor.** With the frontend at 5 ms, `measure` is 10-16 ms median and
  29-45 ms p95 at keyframes, and ~1.5 ms slower in the GPU lane (the wgpu worker thread competes for
  the core). Four CPU frontend threads buy nothing on the host (41.5 vs 43.9 ms). The TX2's 184-223 ms
  per set holds the same split, unmeasured: the GPU lane can only take the frontend's share, and
  upstream's post-pin work (landmark blocks and normal-equation assembly on the CPU pool, `bfd4e17`)
  is what addresses the rest.

Not run on the TX2 yet (board unreachable 2026-10-16): it needs a Vulkan ICD on L4T 32.7 and the
`gpu-wgpu` build there.

**Compare against the right cuVSLAM replay.** Any replay made before `75dac4a` (2026-09-08) is
missing camera 0's stereo partner, and its teleports are that bug, not the tracker: run6 goes
from 2 jumps to 0 with the fix, run5 from one 51.8 m/s jump to 0. Take VO from a VO-only replay
(`OBS=1 replay_host.sh <bag> 0.5`), and SLAM from a separate `SLAM=1 SLAM_PRIMARIES='[0,2,4,6]'`
replay passed as `--slam-bag`. A SLAM replay that overruns its per-set budget drops sets, and the
widened gaps make the VO itself jump - README section 6 has the configs that avoid it. Check the
node log says what you asked for ("SLAM ON: ... over 4 primary cameras", "odometry multicam_mode"):
a launch argument the installed `install_host/` does not declare is dropped silently.

| script | runs on | purpose |
|---|---|---|
| [slamrs/make_calib.py](slamrs/make_calib.py) | dev | write [config/slamrs/bev_calib_s2.json](../config/slamrs/bev_calib_s2.json): KB4 fitted to the Mei intrinsics (slam-rs has no omni model; its KB4 uses `atan2` so it holds our >180° lenses), rig extrinsics, and the round-2 Kalibr `T_cam_imu` — whose only tracked copy is that JSON |
| [slamrs/fit_kb4.py](slamrs/fit_kb4.py) | dev | the KB4 fit on its own: residual per camera out to 100° incidence |
| [slamrs/build_slamrs.sh](slamrs/build_slamrs.sh) | dev / TX2 | build `slam_rs/_core.so` in the submodule without pixi; `GPU=1` adds the CubeCL/wgpu (Vulkan) frontend lane, `PYTHON=` a ≥3.12 interpreter on the TX2 |
| [slamrs/run_slamrs.py](slamrs/run_slamrs.py) | dev | replay a raw log through `slam_rs.Vio` at half resolution; IMU stamps shifted by Δ from `imu_mpu9250.yaml`; writes one pose row per set with `track_ms` and its `fe_ms` / `est_ms` split. `--gpu` runs the frontend lane on the GPU (needs the `GPU=1` build); `--dry-run` checks everything but slam-rs |
| [slamrs/compare_lanes.py](slamrs/compare_lanes.py) | dev | A/B two `run_slamrs.py` outputs of one log (CPU vs GPU lane, or two builds): sets tracked, position disagreement raw and rigid-aligned, path length, `track_ms` / `fe_ms` / `est_ms` medians and the speed ratio |
| [slamrs/compare.py](slamrs/compare.py) | dev | score it against cuVSLAM's `obs_*` bag: continuity, path length, and a Sim(3) fit whose **scale** is an independent check on cuVSLAM's (IMU vs camera baselines) |
| [slamrs/handeye.py](slamrs/handeye.py) | dev | cam1↔IMU rotation and time offset from the gyro vs cuVSLAM's rotation — a cross-check on Kalibr (rotation agrees to <1°; the offset is too coarse to use) |
| [slamrs/run_slamrs_lc.py](slamrs/run_slamrs_lc.py) | dev | the same replay with the **loop-closing backend** beside the VIO: writes the pure-VIO csv plus `slamrs_lc.csv` (corrected frames), `keyframes_lc.csv`, `loop_edges.csv`, `hypotheses.csv` (every candidate and why it was rejected), `timings.csv`. `--keyframes-out` also exports each keyframe's frames, pose and landmarks |
| [slamrs/loop_closure/](slamrs/loop_closure/) | dev | the backend: KB4 + rig (`camera.py`), keyframes from the VIO's snapshots (`keyframes.py`), ORB + landmark association + hash bag-of-words (`features.py`), spatial/appearance retrieval (`retrieval.py`), cuVSLAM's two-step PnP gates (`verify.py`), the 4-DoF pose graph (`posegraph.py`), `LoopClosureBackend.on_keyframe` (`backend.py`). Parameters in [config/slamrs/bev_lc_config.json](../config/slamrs/bev_lc_config.json); design in `openspec/changes/add-slamrs-loop-closure/design.md` |
| [slamrs/test_loop_closure.py](slamrs/test_loop_closure.py) | dev | synthetic tests on the real KB4 + extrinsics: model round trip, HashBow, the verifier recovers an injected loop pose and rejects a look-alike, determinism, the graph removes injected drift, the time gate |

**Loop closure (`run_slamrs_lc.py`).** slam-rs is pure odometry — upstream states "mapper output is
not" supported — so the backend lives beside it in Python and never touches the VIO's output. It is
cuVSLAM's backend design on slam-rs's keyframes: retrieve earlier keyframes by pose radius (older
than `min_separation_s` only; the radius is the KLT tracker's reach, 1 m on this rig — beyond it no
candidate ever produced a track); project the candidate's landmarks — every landmark the keyframe
observed, held in its own rig frame — and its neighbours' into the current cameras with the pose
guess and track their stored appearance, the frontend's own observation at keyframe time, with
Lucas-Kanade from there, forward-backward checked and NCC-gated (cuVSLAM's correspondence step;
every hypothesis records how many landmarks survived each stage). ORB and the hash bag-of-words are
off by default (`features.orb_enabled`): they only serve appearance retrieval after drift beyond the
radius, and on these raw fisheyes ORB descriptors are near-random across keyframes (Hamming 80 vs
122 random). Then solve a multi-camera
PnP-RANSAC on the best pair, refine jointly, and accept only through cuVSLAM's gates (≥ 40 landmarks,
inlier ratio ≥ lerp(0.5 @ 40 → 0.3 @ 200), refined pose consistent with the first). An accepted edge
goes into a 4-DoF pose graph (position + yaw; roll and pitch stay the VIO's) with a Huber kernel and
is solved at once; edges whose residual exceeds 3σ afterwards are dropped and the graph re-solved.
Everything is seeded and synchronous, so the same log gives the same bytes — the property
cuVSLAM's replay lacks (README §6). `compare.py <bag> <vio csv> <lc dir>` scores it;
`rerun_compare.py --lc-dir` draws the closed path in purple with its edges.

**Measured on run5 and run6** — the only revisits these logs hold are the near ones, so this
verifies the pipeline, not the benefit (that needs the closed-circuit log of
`openspec/changes/add-slamrs-loop-closure` task 7). The first column is the 2026-10-18 design
(Mac and Dell agree on run6 to the inlier count; run5's VIO differs by 8 cm between the two
builds, so its edges differ), the second the 2026-10-19 review's changes: observed landmarks in
the keyframe records, ORB off, the radius set to the tracker's reach, the analytic Jacobian.

| run5 / run6 | 10-18 design | after the review | after 9.7, depth confidence (defaults now) |
|---|---|---|---|
| keyframes / hypotheses / accepted | 251 / 603 / 3, 243 / 574 / 3 | 251 / 266 / **20**, 243 / 242 / **29** | 251 / 236 / **28**, 243 / 189 / **47** |
| accepted across the 70 s end-of-log loop | 0 / 0 | 0 / 0 | **10 / 19** (ratios 0.62–0.89) |
| accepted from the opposite heading (> 120°) | 0 / 0 | 12 / 19 | 13 / 18 |
| hypotheses with no track at all | 349 / 377 | 11 / 28 | 15 / 28 |
| landmarks per keyframe, median (weak share) | ~45 / ~55 | 165 / 220 | 166 / 220 (7 % weak, not tracked) |
| first-to-last distance, VIO → closed | 0.51 → 0.17, 0.33 → 0.11 m | 0.51 → 0.16, 0.33 → 0.11 m | 0.53 → 0.14, 0.33 → 0.12 m |
| closed vs VIO position, median / max | 0.14 / 0.48, 0.17 / 0.45 m | 0.14 / 0.50, 0.18 / 0.48 m | 0.15 / 0.54, 0.24 / 0.50 m |
| VIO vs cuVSLAM, Sim(3) RMS (the noise floor) | 0.65 / 0.78 m | same | same |
| per keyframe: features / verify (median, kf with candidates) | 44 / 379 ms (204), 36 / 418 ms (194) | 1 / 1263 ms (115), 1 / 964 ms (118) | 0 / 508 ms, 0 / 345 ms (Mac, three replays in parallel) |
| verify, whole run | 75 / 70 s | 107 / 94 s | 53 / 38 s |
| per accepted closure: pose-graph solve | 2.8 / 3.0 s | 1.1 / 1.2 s | 0.47 / 0.40 s (Mac) |

The review's changes, in order of effect: the 2 s post-closure throttle copied from cuVSLAM had
skipped 50 / 47 of the 111 / 110 keyframes that had an older keyframe within 1 m, so it is off by
default and a travelled-path gate (`min_path_m`, 2 m between the two keyframes) filters the rig
closing against itself while standing still; keyframe records hold every landmark the keyframe
observed, not only the ones it hosts; the radius is the tracker's reach; ORB is off; the pose graph
has its analytic Jacobian. The opposite-heading edges are the corridor walked out and back: only a
median 43 tracks survive the forward-backward check there against 507 same-direction, because the
walls are seen foreshortened the other way round — cuVSLAM's patch tracker has the same limit.

**Depth confidence (task 9.7, design D2c)** is what closes the long loop. The only revisit 70 s
apart in these logs — the rig returning to its first metres at the end — solved to the same 0.30 m /
7° correction on every one of its 111 hypotheses and was rejected on all of them, because the
early keyframes were recorded while the rig stood still and their landmarks' depths were frozen at
marginalisation with no parallax behind them. The tracker now keeps a finalised keyframe's record
open: a landmark's position follows the latest snapshot while it is alive, its parallax and
observation count (from the frontend's keypoint ids and the window poses) grow with it, and
landmarks under 2° of parallax or 5 observations are weak and never tracked. Two ideas that sound
right were measured not to be: a far / bearing-only class (a zero-parallax landmark on this rig is
near and unknown, not far) and a self-projection bound on the refresh (the early keyframes' frozen
poses are degrees off, so the later, better geometry fails the bound exactly where it is needed —
that disagreement is the drift the loop repairs). With the defaults the long loop closes on 10 / 19
edges with 106–270 inliers at ratios 0.62–0.89, which also retires the ratio-gate question of task
8.8: its denominator had been full of depthless landmarks. Not tracking the weak ones halves the
verification cost.

The pure-VIO csv is byte-identical with and without the backend (every column but the wall-clock
`track_ms`), and two replays on one build give byte-identical hypotheses, edges and corrected
trajectories (checked again after 9.7). The solve is the Python per-edge loop around SciPy's sparse
solver with the analytic Jacobian; vectorising it is the next step before the backend goes near a
thread.

## bev/ — bird's-eye ground stitch

| script | runs on | purpose |
|---|---|---|
| [bev/verify_ground_stitch.py](bev/verify_ground_stitch.py) | dev | check the whole BEV chain **without the rig**: render what each fisheye would see of a known floor, publish it as one synchronised set, and check `bev_ground_stitch` reproduces it. The projection is exact by construction, so what this actually tests is its *inputs* — rig frame, handedness, plane sign, which camera is which — every one of which otherwise yields a stitch that still looks like a picture of a floor. The texture is deliberately asymmetric (bright bar → forward, dark bar → left) because a checkerboard hides a 180° roll |
