## Context

The stack runs on a Jetson TX2 (J106), JetPack 4.6.x / CUDA 10.2 / r440, from a single
`cuvslam-foxy:tx2` image (Ubuntu 20.04 + ROS 2 Foxy + gcc-8 + cmake 3.27, NVIDIA EGL
baked). `libcuvslam.so` is built and WarmUpGPU-validated in that image (confirmed
2026-06-20). The capture node (`bev_camera`) uses a headless `EGLDisplay` via
`EGL_PLATFORM_DEVICE_EXT` and creates `/camN/image_raw`; the VO node (`bev_cuvslam`)
links `libcuvslam.so` and calls `cuvslam::Odometry::Track(vector<Image>)`. Neither has
been run against the other. See [docs/build_and_run.md](../../../docs/build_and_run.md)
and [docs/cuvslam_tx2.md](../../../docs/cuvslam_tx2.md).

## Goals / Non-Goals

**Goals:**
- Sustained 4-camera capture at the configured rate, verified by `topic hz`.
- Correct runtime calibration loading (intrinsics + extrinsics).
- Capture → VO running together with `/odom` tracking real motion and the TF broadcast.
- Decide the container/DDS topology (one container vs. two over `--network host`).

**Non-Goals (this change) — but on the critical path to the final goal:**
- cam0 + 6th camera (i2c bus 1 @ 0x12) bring-up — only cam1–4 here.
- IMU fusion (IMU publisher + `robot_localization` EKF).
- Zero-copy fused single-process Argus→cuVSLAM node.
- Downstream perception: depth / occupancy grid, IPM surround BEV.

**Final goal (north star):** full surround **VIO** on **4–6 cameras + IMU** — i.e. all
six IMX219 feeding cuVSLAM, fused with the MPU-9250 IMU. This change is bring-up step 1
(4-cam VO, no IMU); the items above are the staged path there, not abandoned scope.

## Staged plan to the final goal

The migrations are deliberately sequenced so each step is validated before the next adds
complexity:

1. **(this change) 4-cam VO, non-zero-copy.** Prove tracking works end-to-end with the
   simple modular data path (Argus→NVMM→CPU→DDS→GPU). Correctness first; copies are fine.
2. **Zero-copy fused node.** *Only once step 1 tracks reliably*, collapse capture + VO
   into one process feeding Argus NVMM straight to cuVSLAM as GPU memory
   (`is_gpu_mem=true`, no GPU→CPU→DDS→GPU round-trip), publishing only odometry. Same
   container — no new image. This is the explicit next step, gated on step 1.
3. **6 cameras + IMU → VIO.** Add cam0 + the 6th camera (keep the overlap ring connected),
   and **loosely** couple the IMU via an EKF (`robot_localization`). Note: cuVSLAM v15
   cannot tight-couple an IMU in multicam mode — `Inertial` mode is single-stereo-camera
   only ([cuvslam2.h:221](../../../third_party/cuVSLAM/libs/cuvslam/cuvslam2.h#L221)), so
   the 4–6 cam + IMU north star is multicam VO + external IMU fusion, not cuVSLAM-internal VIO.

## Empirical findings (bring-up run, 2026-06-20)

What the actual board run flushed out (the unknowns this change existed to test):

- **Capture QoS**: the publisher must be **best-effort `SensorDataQoS`**. With default
  (reliable) QoS, a slow reliable subscriber (the VO node's GPU warmup + per-frame
  `Track()`) back-pressures and **blocks the Argus capture thread** — capture runs alone
  but freezes once VO attaches. Fixed.
- **Cross-container DDS discovery failed** (topics invisible from a 2nd container even
  with `--network host`) → confirmed the **single-container** topology for bring-up.
- **⛔ Hard blocker — camera sync.** cuVSLAM `Multicamera` rejects sets with per-camera
  timestamps >1 ms apart. The IMX219 rig has **no hardware trigger**; measured 4-cam
  spread is **~30–66 ms** and the cameras drift at *different* rates. A unified per-set
  timestamp clears cuVSLAM's 1 ms check, but ApproximateTime still can't reliably form
  4-way sets from unsynchronized, drifting, best-effort streams → `Track()` rarely fires
  → **no odometry**. This is the decision point: **hardware frame sync** (FSIN common
  trigger, if the modules support it) vs. **an async-multi-camera VIO like OpenVINS**
  (explicitly handles asynchronous cameras — cuVSLAM does not). This directly validates
  the VIO-options research: cuVSLAM is the wrong tool for an unsynchronized rig.
- **Calibration gap**: `cam2.yaml`/`cam4.yaml` load with default-looking intrinsics
  (f=(522,522), principal point ≈ image center) vs the real values in cam1/cam3 — those
  two cameras appear **uncalibrated**; redo their intrinsics before judging tracking.

## Decisions

- **Topology**: prefer running both nodes in **one container** for bring-up to sidestep
  cross-container DDS discovery entirely (start capture in the background, run VO in the
  foreground). The two-container `--network host` path stays documented as the
  alternative for recording bags / inspecting topics.
- **calib_dir**: the intrinsics are tracked at `scripts/config/calib/camN.yaml`; the old
  node default `config/calib/1640x1232` did not exist. Fix: set the default (node +
  launch) to `scripts/config/calib`. No gitignored/`1640x1232` path is involved — that
  was a stale note.
- **Validation order**: gate on capture rate first (cheap, isolates the camera side),
  then bring up VO — so a tracking failure isn't confounded by a frame-flow problem.

## Risks / Trade-offs

- **Timestamp sync across 4 cams** is the main unknown — if frames aren't coherently
  synchronized, cuVSLAM's stereo pairing degrades; may need to tighten the sync tolerance.
- **Frustum-overlap requirement (the load-bearing constraint).** `OdometryMode::Multicamera`
  requires *every camera to share frustum overlap with at least one other camera*
  ([cuvslam2.h:363-364](../../../third_party/cuVSLAM/libs/cuvslam/cuvslam2.h#L363));
  cuVSLAM auto-forms the stereo connections (`MulticameraMode::Precision`), so **no
  explicit stereo pairs are declared**. Our rig satisfies this: 4× 160° fisheye at 90°
  spacing → ≈70° azimuthal overlap between each adjacent pair → a connected ring
  (cam1↔cam2↔cam3↔cam4↔cam1; opposite cams don't overlap, which is allowed). The risk is
  not the mode but whether the overlap is *usable*: bad extrinsics or texture-poor
  side-overlap wedges can make auto-pairing fail to match features → no tracking. There
  is no multi-mono-without-overlap mode — if overlap were absent, the only fallback is
  `OdometryMode::Mono` (single camera, up-to-scale).
- **Fisheye (KB) intrinsics + rig extrinsics** are unproven against real frames; verify
  the auto-pairing actually connects the ring before trusting tracking.
- **3 copies + CPU round-trip** (Argus→NVMM→CPU→DDS→GPU) costs CPU/latency. This is an
  accepted, deliberate trade-off for bring-up — correctness before performance. It is not
  a permanent design: once this non-zero-copy version tracks reliably, the purely
  zero-copy fused node (staged plan step 2) replaces it. Keep the data path modular here
  so swapping in the fused node later is contained.
- TX2 concurrent FPS budget is tight; 4 cams + VO may not hit full rate — measure, then
  decide whether to drop capture rate or resolution.
