# Scripts

Index of the helper scripts in this folder. "Runs on" = **TX2** (the board, usually inside the
`cuvslam-foxy:tx2` container) or **dev** (your workstation, needs python3 + opencv + numpy).

## Top level — build & run

| script | runs on | purpose |
|---|---|---|
| [setup_tx2_docker.sh](setup_tx2_docker.sh) | TX2 | one-time JetPack 4.6 / L4T R32.7 Docker prep for the BEV/cuVSLAM stack |
| [build_cuvslam_tx2gpu.sh](build_cuvslam_tx2gpu.sh) | TX2 | build cuVSLAM's GPU path on CUDA 10.2 with gcc-8 + C++14 (applies the port patch) |
| [run_vo_fused_tx2.sh](run_vo_fused_tx2.sh) | TX2 | wrapper: run the **fused** zero-copy Argus→cuVSLAM VO (`docker compose`) |
| [run_vo_tx2.sh](run_vo_tx2.sh) | TX2 | wrapper: run the **modular** capture + cuVSLAM VO (`docker compose`) |
| [capture_montage_tx2.sh](capture_montage_tx2.sh) | TX2 | capture the 4 fisheye views + stitched panorama, montage into one image |

## calib/ — camera calibration

| script | runs on | purpose |
|---|---|---|
| [calib/online_calib.py](calib/online_calib.py) | dev/TX2 | online interactive fisheye (Kannala-Brandt) **intrinsic** calibration |
| [calib/offline_calib.py](calib/offline_calib.py) | dev | offline outlier-rejection fisheye **intrinsic** calibration |
| [calib/scale_calib.py](calib/scale_calib.py) | dev | scale KB `camN.yaml` intrinsics to a downscaled output resolution |
| [calib/grid_view_tx2.sh](calib/grid_view_tx2.sh) | TX2 | live 2×3 IMX219 grid on the TX2 HDMI display (identify/label cameras) |
| [calib/capture_calib_sets.sh](calib/capture_calib_sets.sh) | TX2 | grab N raw 4-cam sets for **extrinsic** calibration (pan the rig) |
| [calib/extrinsic_calib.py](calib/extrinsic_calib.py) | dev | feature-based relative-rotation **extrinsic** calibration (joint solve + before/after render) |
| [calib/pano_tuner.py](calib/pano_tuner.py) | dev | interactive panorama extrinsics tuner (web UI; manual rot/trans) |

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
| [calib/fold_roll_for_vo.py](calib/fold_roll_for_vo.py) | dev | fold the 180° mounting roll into the VO extrinsics instead of hiding it downstream |
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
| `docker compose run --rm logonly` | TX2 | **raw 4-camera image log, no ROS/DDS in the path.** `argus_capture_node -p image_log_dir:=` writes `camN.raw` (concatenated mono8), `camN_index.csv` (exposure-midpoint stamp, byte offset) and `geometry.txt`. Read it with `numpy.memmap('cam1.raw','u1').reshape(-1,1088,1456)`. **Measured over 60 s (2026-09-03), on set completeness**: 30 fps split `LOG_DIRS=/logs,/logs,/sdlog,/ramlog` → **97.2% at 28.74 Hz**; 20 fps trigger with all four on eMMC (`TRIGGER_FPS=20 LOG_DIR=/logs`, generator set with `j106-trigctl.py fps 20`) → **99.9% at 19.93 Hz**. Across 5.5 min of 20 fps runs what remains is one brief global stall per ~80 s (all four cameras lose the same 1–3 edges) plus a 250 ms startup transient; it is **storage bandwidth** — decimating the images 10× while still logging every trigger edge gave zero stalls in 118 s. **Do not use `IMAGE_LOG_DIRECT` at 20 fps**: without the page cache eMMC cannot take 126.7 MB/s, and all four direct dropped 661 frames in 90 s. At 30 fps the losses begin ~30 s in and then run to the end, spread uniformly rather than periodically: the split saturating over the length of the run. **Prefer the 20 fps form** unless 30 fps is the requirement. Going through ROS topics to a recorder on the same board gave 6.1 fps |
| [docker_publish.sh](docker_publish.sh) | TX2 | push `cuvslam-foxy:tx2` to Docker Hub as `wtlove876/cuvslam-foxy:tx2` plus a dated tag. The image is Ubuntu 20.04 + ROS 2 Foxy + gcc-8/9 only — cuVSLAM, calibration and the repo are bind-mounted at runtime, so nothing project-specific is baked in. Exists because `docker image prune -af` wiped every image on the board on 2026-09-02 and cost an hour of rebuilding; with this, that is a `docker pull` |
| [port/check_log_sets.py](port/check_log_sets.py) | dev | the metric that matters for a triggered rig: how many trigger edges have **all four** cameras. Frame count and fps hide this — cameras drop *independently*, so 1.5% per-camera loss became 7% broken sets. Measured best: **97.4% complete sets at ~28 Hz** over 60 s |
| [port/locate_frame_loss.py](port/locate_frame_loss.py) | dev/TX2 | **where** a triggered rig loses a frame, which set completeness cannot say. The capture node records two counters per frame — `capture_id` (what the Argus session produced) and `seq` (what reached the consumer) — so a gap in both means the frame was never produced (sensor/driver missed the edge) while a gap in `seq` alone means it was produced and we failed to collect it. That distinction found the real bug: the per-frame CSV was flushed from the **capture thread**, so roughly every 5 s it blocked inside an ext4 journal commit (`data=ordered`, default 5 s) and the whole 4-camera set missed its trigger edge. Moving the CSV write onto its own thread took 20 fps from 99.5% to 99.9% complete sets |
| [port/raw_log_to_bag.py](port/raw_log_to_bag.py) | dev | convert a raw image log into a **rosbag2 bag format v4** — what Foxy reads. Recording to a bag on the board tops out at 6–7 fps (rosbag2's writer plateaus at 38–46 MB/s while DDS carries ≥142 and eMMC takes 136), and best_effort means the shortfall is *dropped*. So capture raw at full rate and convert here, where nothing is real time. Original per-frame exposure-midpoint stamps are preserved; `--compress` applies zstd. Note `rosbags`' own Writer emits v8/v9, which Foxy cannot open — hence the hand-written v4 |
| [port/topic_rate_probe.py](port/topic_rate_probe.py) | TX2 | count what a subscriber actually receives, to separate DDS from whatever consumes it. Measured: capture 30 Hz / 190 MB/s, subscriber 20–25 Hz / 141.8 MB/s, rosbag2 6.1 Hz / 38.5 MB/s — the transport is not the bottleneck, the writer is |
| [vo/bench_remap.cpp](vo/bench_remap.cpp) | TX2 | measure the virtual-pinhole remap against virtual and source resolution, standalone (no cameras, no ROS). Answers "should we lower the resolution" with a number: both axes are bad trades (task 4.5b) |
| [vo/verify_rig_build.sh](vo/verify_rig_build.sh) | dev | re-run cuVSLAM's frustum test on the poses the C++ actually emits — run it **before** a board session, or a bad rig file reads as a wiring bug |
| [vo/check_rig_poses.py](vo/check_rig_poses.py) | dev | sanity-check the rig poses fed to cuVSLAM |
| [vo/run_motion_test.sh](vo/run_motion_test.sh) | TX2 | record a motion test (§5). Preflight **gates, not warns**: refuses unless `trigger_mode=1`, the generator is running and `active_low`, and it reads the real pulse width back so `exposure_us` is never a stale guess. `--record-images` also bags the four camera streams, making the run replayable — move the rig once, re-run the VO against it as often as needed. Do two passes: one with images for replay, one without for the live numbers, because the recorder's own load is indistinguishable from the rig misbehaving |
| [vo/analyze_motion.py](vo/analyze_motion.py) | dev | compare recorded odometry against the tape measure: scale, drift, return-to-origin |

## bev/ — bird's-eye ground stitch

| script | runs on | purpose |
|---|---|---|
| [bev/verify_ground_stitch.py](bev/verify_ground_stitch.py) | dev | check the whole BEV chain **without the rig**: render what each fisheye would see of a known floor, publish it as one synchronised set, and check `bev_ground_stitch` reproduces it. The projection is exact by construction, so what this actually tests is its *inputs* — rig frame, handedness, plane sign, which camera is which — every one of which otherwise yields a stitch that still looks like a picture of a floor. The texture is deliberately asymmetric (bright bar → forward, dark bar → left) because a checkerboard hides a 180° roll |
