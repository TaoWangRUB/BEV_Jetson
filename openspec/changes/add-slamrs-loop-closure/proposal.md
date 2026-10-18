# Proposal

## Why

slam-rs gives the rig a deterministic, metric, gravity-aligned VIO that matches cuVSLAM to the
centimetre on the raw logs (`task/slam-rs-vio-test`, merged 2026-10-16) — but it is pure odometry.
Its estimator is a sliding window; everything outside the window is fixed, so drift accumulates with
distance and nothing ever pins the trajectory back to a place it has seen before. cuVSLAM has a
loop-closing backend, but `add-replay-visual-diagnostics` 1.7 established that on this rig it cannot
be measured: its replay is non-deterministic (1.13 m replay-to-replay noise), its backend starves the
tracker it is built on (README §6), and its cost grows as `N^0.72` in the map. slam-rs has no loop
closure at all — upstream states "mapper output is not" supported — so the question "does loop
closure help this rig" has no pipeline that can answer it.

This change gives slam-rs a loop-closing backend designed from the two published references that
actually describe one: cuVSLAM's online backend (spatial landmark retrieval → multi-camera PnP
hypothesis → gated acceptance → pose-graph optimisation on a separate, throttled thread) and
Basalt's visual-inertial mapping (ORB features distinct from the VIO points, VIO information as
relative-pose and roll-pitch factors). Basalt's own loop-closure code was never published or
evaluated — it is an unpublished hash-bag-of-words plus batch bundle adjustment — so only its factor
idea is taken; retrieval and verification follow cuVSLAM, the production design.

## What Changes

- **A keyframe memory beside the VIO.** `run_slamrs.py` gains an export of every keyframe the
  estimator votes: the four half-resolution frames, the IMU pose, and the keyframe's landmarks.
  The VIO itself is untouched — same `slam_rs._core`, same config, same poses as before.
- **Loop-candidate retrieval, two ways, both gated in time.** Spatial: keyframes whose VIO pose
  lies within a radius of the current one (cuVSLAM's query, the cheap path for small drift).
  Appearance: a bag-of-words over ORB descriptors from all four cameras (recovers revisits after
  the drift exceeds the radius, and revisits from a different heading because the ring sees every
  direction). Candidates younger than a configurable interval are never considered.
- **Geometric verification that follows cuVSLAM's gates.** ORB matches between the candidate
  keyframe and the current one, unprojected through the KB4 model to bearings; the candidate
  keyframe's VIO landmarks give the matched points their depth; a multi-camera PnP with RANSAC
  solves the relative pose; two-step refinement; acceptance requires a landmark floor, an inlier
  ratio that scales with the count, and agreement between the two steps. A rejected hypothesis
  leaves no trace in the graph.
- **A pose graph with VIO factors.** Keyframe nodes; sequential edges carry the VIO relative pose;
  roll and pitch are held to the VIO's gravity-aligned estimate (observable from the IMU, so the
  correction is 4-DoF: position and yaw, as VINS-Mono argues); loop edges carry the verified
  relative pose with a covariance scaled by its inlier ratio and a robust kernel. Optimised on
  every accepted closure, bounded by a throttling interval and a node cap, as cuVSLAM bounds it.
- **Outputs beside the VIO, never in place of it.** The optimised keyframe trajectory, the
  per-frame corrected poses, the accepted loop edges and the rejected hypotheses are written next
  to the pure-VIO CSV; `rerun_compare.py` draws the corrected path as a third line with the loop
  edges; `compare.py` reports return-to-start and loop-edge separation for both.
- **An offline, deterministic pipeline first.** The backend is written as an incremental object
  (`on_keyframe`) so it can run on its own thread later, but this change runs it synchronously on
  the raw logs on the host. The same log through the same build gives byte-identical output — the
  property cuVSLAM's replay lacks and the reason this pipeline can answer the question.

**Scope honesty.** No log in `datasets/` contains a real loop: the 63 "revisits" of run1 are a
median 2.45 s apart. The pipeline is verified on those near-revisits and on synthetic rigs; the
*benefit* of loop closure can only be measured on a closed-circuit log returned to the start with
`tape_metres.txt`, which is listed as a task, not claimed.

## Capabilities

### New Capabilities

- `vio-loop-closure`: recognise when the rig returns to a place its trajectory has already visited,
  verify that recognition geometrically, and produce a globally consistent trajectory beside the
  odometry — with bounded cost and deterministic output.

### Modified Capabilities

<!-- none. fused-vo and surround-panorama keep their requirements; the cuVSLAM SLAM path in
     bev_cuvslam is untouched and remains the comparison baseline. -->

## Impact

- `scripts/slamrs/run_slamrs.py`: optional keyframe export (new flags; default behaviour unchanged).
- New `scripts/slamrs/loop_closure/` package: retrieval, verification, pose graph, metrics.
- New `scripts/slamrs/run_slamrs_lc.py` driver and `scripts/slamrs/test_loop_closure.py`.
- `scripts/slamrs/rerun_compare.py`, `scripts/slamrs/compare.py`: read the optional outputs.
- Dependencies: OpenCV (already in `.venv`), SciPy (already), no new native libraries. GTSAM is
  installed in `.venv` as a later option for an incremental solver; this change uses SciPy.
- `config/slamrs/bev_lc_config.json`: the retrieval, verification and graph parameters.
- `scripts/README.md`: the slamrs/ section gains the loop-closure tools and the findings.
- Not affected: the TX2 image, the ROS nodes, `docker-compose*.yml`, the cuVSLAM port.
