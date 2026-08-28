## Context

See proposal.md — *Why*. The design-relevant state:

- **Capture** ([argus_capture_node.cpp](../../../ros2/bev_camera/src/argus_capture_node.cpp)) takes
  `sensor_ids {1,2,3,4}` as a parameter and stamps each frame with the Argus
  `iframe->getTime()` value it already has. It copies NVMM→host and publishes `/camN/image_raw`.
- **Modular VO** ([cuvslam_multicam_node.cpp](../../../ros2/bev_cuvslam/src/cuvslam_multicam_node.cpp))
  caches the latest frame per camera; cam0 arriving triggers `Track()` on whatever the other three
  last delivered, gated by `sync_slop_ms` (default 120), then rewrites all four timestamps to cam0's
  so cuVSLAM's 1 ms `Multicamera` gate accepts the set.
- **Fused VO** ([bev_cuvslam_fused_node.cpp](../../../ros2/bev_cuvslam/src/bev_cuvslam_fused_node.cpp))
  acquires the four EGL streams in lock-step inside one loop, so its sets are already coherent — but
  it also stamps the set with cam0's time only (`ts0`).
- **cuVSLAM v15 constraint**: `Multicamera` mode and IMU fusion are mutually exclusive, and the
  frustum-overlap threshold for forming stereo links is hard-coded at 0.5. Neither changes here.
- **Board**: boots `LABEL j106imx296`; the STM32H7 emits the trigger, so the camera rate is set
  off-board. `jetson-clocks` is required for ≥3 concurrent streams.

## Goals / Non-Goals

**Goals:**

- One code path that works on the triggered rig, with no sensor-specific branches left behind in the
  VO nodes.
- Timestamps that mean what they say end to end, so Δ has something to be an offset *of*.
- A calibration set that is self-describing enough that applying a stale one is a startup error
  rather than a slow drift.

**Non-Goals:**

- Supporting both rigs at runtime. The IMX219 modules are not fitted; a compatibility mode would be
  untested code carrying the exact bug this change removes.
- IMU fusion in the VO. Δ is measured and recorded here; consuming it is the next change (and
  requires leaving cuVSLAM `Multicamera` mode).
- Panorama/BEV re-tuning, IMX296 colour work, ports A/B.

## Decisions

### D1: Resolve port→sensor-id inside the capture node, not in a launch wrapper

Argus assigns `sensor-id` in `/dev/video` bind order. The board's bind order is not port order and
is not stable across boots — verified live in `a5b0b28`, where binding port F before E produced
`video4=7-0012` and `video5=7-0010`, shifting every hard-coded cell by one. The node reads
`/sys/class/video4linux/videoN/name`, extracts the i2c name, and maps it through a port table
(`2-001a`→C, `2-0018`→D, `7-001a`→E, `7-0018`→F, plus the IMX219 `…-0010/0012` aliases).

*Alternative rejected*: have a shell wrapper resolve it and pass `sensor_ids` (what
`csi_sender.sh` does). It works, but it puts the rig's identity in two places and makes the node
silently wrong when launched directly. The node keeps a `sensor_ids` parameter as an explicit
override for bring-up.

### D2: Delete the bundler; gate on real skew and drop what fails

Each image goes to cuVSLAM with **its own** `timestamp_ns`. The set-matching rule becomes: a set is
formed from frames whose timestamps span less than `max_skew_us` (default 1000, i.e. cuVSLAM's own
gate); a set that fails is dropped and counted, never re-stamped. In the fused node the lock-step
acquire already yields such sets, so it needs the per-camera `getTime()` and the same gate rather
than a matcher.

*Alternative rejected*: `message_filters::ApproximateTime`. It was tried on the old rig and could
not match four drifting best-effort streams. With a shared trigger edge the matching problem is
trivial, and a hand-rolled matcher is what lets us publish the skew and drop counts that the spec's
health requirement calls for.

*Consequence*: if the trigger is not running, the VO produces nothing instead of producing something
wrong. That is deliberate — it is the failure mode the spec asks for, and the message says so.

### D3: One timebase, chosen at capture, with Δ absorbing the rest

Cameras are stamped with the Argus acquisition time; the IMU is stamped through the userspace path
`j106-record-sync.py` established. Rather than trying to make the two paths bias-free, the design
takes the J106 project's conclusion: the wake-path bias cancels only if both sides are measured the
same way, and whatever does not cancel *is* Δ. So the code applies exactly one documented offset in
one direction, and the recorded Δ carries the provenance of how it was obtained.

### D4: Get Δ from a single-camera + IMU Kalibr run; get extrinsics from the four-camera run

Kalibr's multi-camera solver needs overlap between consecutive cameras in the chain. This rig is
divergent by construction, and overlap between adjacent fisheyes is the known weak point (it is what
blocks cuVSLAM's 0.5 frustum gate and what OpenMAVIS's stereo init needs). Splitting the runs means
a hard four-camera solve cannot block Δ, which the VIO actually needs:

- `kalibr_calibrate_cameras` over all four (pinhole-equi / KB fisheye) → rig extrinsics.
- `kalibr_calibrate_imu_camera` with **one** camera + the MPU-9250 → Δ (and that camera's IMU pose).

*Fallback if the four-camera chain will not converge*: solve adjacent pairs and compose around the
ring, or keep the existing feature-based extrinsics (`config/rig/rig_extrinsics_vo.yaml`, residual
~1.5–1.9°) and take only Δ from Kalibr. The spec's loop-closure scenario is what decides which
result ships.

### D5: Kalibr runs on the host in Docker, over a converted bag

Kalibr is ROS1/Python2-era and will not build on the board. Record on the TX2 with `ros2 bag`
(reduced rate is fine — Kalibr wants ~4 Hz images and full-rate IMU), convert on the host with
`rosbags-convert`, and run Kalibr in a Noetic container against the host's existing Docker setup.
The alternative — a live ROS1 bridge — adds a timing path to a calibration whose whole purpose is
timing.

### D6: Retain the IMX219 calibration by moving, not deleting

`scripts/config/calib/cam{1..4}.yaml` move to `scripts/config/calib/imx219-1640x1232/`, and the new
IMX296 files take the canonical names. Note `scripts/config` appears in `.gitignore` while those
files are tracked, so the new ones need `git add -f` — easy to lose otherwise.

Each calibration file gains `sensor: imx296` and keeps its `image_width/height`; the VO nodes
compare both against the live capture configuration at startup and refuse a mismatch (spec:
*Resolution and calibration agree*).

## Risks / Trade-offs

- **The four-camera Kalibr solve does not converge on a divergent rig** → D4 splits Δ off so it is
  never blocked; documented fallbacks are pairwise composition or keeping the feature-based
  extrinsics.
- **Tracking is still not metric after sync is fixed** — cuVSLAM's hard-coded 0.5 frustum-overlap
  threshold may reject every pair regardless of timing → the spec requires the node to *say* it is
  running unscaled rather than presenting an unscaled pose as metric. If that is what we see, the
  outcome is evidence for the OpenMAVIS/D2SLAM route, not a defect in this change.
- **Argus timestamps may not be the true trigger instant** (ISP queueing could add a constant) → a
  constant offset lands in Δ and cancels; a *varying* one would show up as non-zero measured skew,
  which the health signal now reports.
- **Global shutter changes exposure behaviour**: AE is clamped, so a scene outside the fixed
  gain/exposure window is simply over- or under-exposed → the trigger pulse width is the exposure
  control; record the working pulse width with the calibration.
- **No fallback to the old rig** → rollback is `git revert` plus the retained IMX219 calibration
  directory; the board keeps its `j106fullfov`/`cam219` extlinux LABELs for the hardware side.

## Migration Plan

1. Land capture-node changes; verify all four ports resolve and stream at 1456×1088 with AE locked.
2. Recalibrate intrinsics; move the IMX219 set aside in the same commit that adds the IMX296 set.
3. Record the Kalibr bag once (target visible to all four cameras, then an IMU excitation sequence);
   solve extrinsics and Δ on the host.
4. Switch the VO nodes to real timestamps + skew gate, then run the motion test.
5. Close `bring-up-end-to-end-vo` tasks 3.4/3.6 with the motion-test evidence, or record why they
   remain open (e.g. overlap gate) and hand that to the next change.

## Open Questions

- **Trigger rate for VO.** 30 Hz is what the sync work used; whether the TX2 sustains 4×1456×1088 at
  30 Hz through `Track()` is a measurement, not a decision — set it from the first sustained run.
- **Whether Δ is stable across reboots** or needs re-measuring per session. One Kalibr run cannot
  answer it; note the value and re-check on the second recording.
