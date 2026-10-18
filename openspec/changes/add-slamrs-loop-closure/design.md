# Design

## Context

See proposal.md — Why. What shapes the approach:

- **slam-rs exposes enough, and no more.** `Vio.snapshot()` returns the window poses, which of
  them are keyframes (`window_keyframe`), the marginalised ids, and the landmark ids, hosts and
  positions. It does not expose marginalisation priors or covariances, so Basalt-style non-linear
  factor *recovery* is not possible; the VIO's information enters as relative-pose edges with a
  modelled covariance and as fixed roll/pitch.
- **The rig is a ring, not a stereo pair.** Four ~192° fisheyes at 90°, overlapping ~12° at the
  seams. A revisit from a different heading is seen by a *different* camera. Retrieval and
  verification must therefore work across camera indices, not only camera-to-same-camera.
- **No omni model in OpenCV's PnP.** Bearings come from the KB4 fit in
  `config/slamrs/bev_calib_s2.json` (its `atan2` form holds past 180°); PnP and essential-matrix
  solvers are run on the forward hemisphere of each camera, projected to the normalised plane.
- **cuVSLAM is the production reference**, readable at
  `ackermann_rover_humble/src/cuVSLAM/libs/slam`: `LSIGrid` spatial retrieval,
  `LoopClosureSolverTwoStepsEasy` (40-landmark floor, inlier-ratio threshold interpolated from 0.5
  at 40 to 0.3 at 200, second step must land closer to the first than to the guess, covariance
  scaled by `1 + 99·(1 − ratio)`), `AsyncSlam` (throttling interval, node cap, one backend thread),
  `math/pgo.h` (6-DoF, robustifier 0.1, planar option).
- **Determinism is the point.** Everything here is synchronous and seeded; no wall-clock
  dependence anywhere, so the same log gives the same bytes.
- **Logs.** `log_rig.sh` directories (cam1..4.raw + index CSVs + imu0.csv) on the Dell host
  (192.168.0.225) and the TX2; none on this Mac yet. None contains a real loop (proposal —
  scope honesty).

## Goals / Non-Goals

**Goals:**

- An offline, deterministic loop-closing backend for slam-rs that runs on the raw logs on a laptop,
  structured so it can later be moved to its own thread (`on_keyframe` interface, throttle, cap).
- cuVSLAM's acceptance gates reproduced numerically, so the two backends are comparable.
- Metrics and a view that can decide "does loop closure help this rig" once a closed-circuit log
  exists.

**Non-Goals:**

- Changing slam-rs's Rust estimator, or upstreaming anything. Everything lives in
  `scripts/slamrs/`.
- Relocalisation against a *saved* map, map merging, or multi-session.
- Running on the TX2. The VIO is 5 Hz offline there (README); the backend inherits that.
- Replacing cuVSLAM's SLAM path. It stays as the comparison baseline.

## Decisions

**D1. Retrieval is spatial first, appearance second, both time-gated.**
cuVSLAM retrieves by pose only (kd-tree in the paper, `LSIGrid` in code) because its frontend drifts
little between closures; on a long circuit the VIO can drift past any radius, so appearance is kept
as the fallback. Appearance = ORB on all four cameras → a hash bag of words (16 random descriptor
bits → word, L1-normalised histogram, inverted index, L1 score), which is Basalt's `HashBow`:
needs no vocabulary training, is deterministic, and costs microseconds. *Alternative:* DBoW2-style
trained vocabulary — better recall in the literature but needs training data and a native library
with no macOS wheel. Revisit if HashBow's recall is the limiting factor on the circuit log.
Time gate: candidates younger than `min_separation_s` (default 10 s) are dropped — run1's
2.45 s "revisits" are exactly the noise cuVSLAM's `throttling_time_ms` exists to suppress.

**D2. Correspondences are KLT tracks of the stored landmark appearance, exactly as cuVSLAM does
it; ORB descriptors are only the fallback.** The first version matched ORB descriptors between
keyframes with depth from the VIO landmarks, and the real data overturned it: on these raw
fisheye frames the ORB descriptor of the same corner two keyframes apart is nearly as far from
its partner as from a random one (run5, Hamming 80 vs 122), and the detector misses most of the
corners the VIO tracks (nearest detection a median 10 px away). cuVSLAM never had this problem
because it does not match descriptors: it projects map landmarks into the current image with the
pose guess, tracks their stored patches with Lucas-Kanade from there, and keeps the tracks that
pass an NCC check. That is what `klt_pairs` does, with the keyframe's own observation of each
landmark (the frontend's `FlowFrame` at keyframe time) as the stored appearance, a
forward-backward check (Basalt's frontend's own test), and NCC ≥ 0.7. Cross-camera is free: the
prediction says which current camera a landmark lands in. Measured on run5 keyframes 34→36: 208
tracks, 105 PnP inliers, pose within 7 cm / 0.4° of the VIO, the same with the guess moved 30 cm.
*Cost:* the guess must be within the pyramid's reach (4 levels, 21 px window: roughly ±80 px),
which holds for spatial candidates; an appearance candidate found after large drift falls back
to the ORB route and is expected to verify rarely until descriptors are computed on rectified
virtual pinholes (open question).

**D2a. The keyframe record must be consistent with itself.** Two defects found on run5, both
invisible to the synthetic tests: (1) the landmarks alive when the VIO marginalises a keyframe are
a remnant (median 3 of ~30), so the record accumulates every landmark the keyframe ever hosted;
(2) a landmark's world position frozen at an earlier snapshot is inconsistent with the keyframe's
later pose, because the sliding window re-optimises both together — so each landmark is stored in
the host rig frame at the snapshot it was last seen in. Before: projection-to-observation error
median 5.7 px; after: 0.50 px, 87 % within 2 px. Landmarks younger than `min_landmark_age`
snapshots (3) are not used: their depth is still settling.

**D3. Verification is cuVSLAM's two-step PnP with its gates, per camera pair then jointly.**
The landmarks in sight are the candidate's plus those of its neighbouring keyframes (within
`neighbour_radius_m`, ±`neighbour_max_kf`, each brought into the candidate's frame through the
VIO relative pose, which is locally consistent) — cuVSLAM fetches every landmark near the guess.
Step 1: for each (source keyframe, camera pair), PnP-RANSAC (OpenCV EPnP, seeded, on
normalised-plane coordinates of bearings with z > 0.2, threshold 0.02 ≈ 5 px, which is where the
real correspondence errors sit) of the current bearings against the 3D points; the pair with most
inliers gives the first pose. Step 2: re-run from step 1's pose over all correspondences in one
joint non-linear refinement (SciPy `least_squares`, Huber). Accept iff: inliers ≥ 40; inlier
ratio ≥ lerp(0.5 @ 40 → 0.3 @ 200); step 2 closer to step 1 than to the odometry guess (cuVSLAM's
"first→second must beat cur→second", waived when the refined pose already agrees with the guess
within 2 cm / 0.2°, which cuVSLAM's version would reject by construction). Covariance = base ×
(1 + 99·(1 − ratio)). *Alternative:* a learned matcher (LightGlue) — more recall,
non-deterministic across hardware, out of scope for a deterministic baseline.

**D4. The graph is 4-DoF (x, y, z, yaw) with a robust kernel, solved by SciPy.**
Roll and pitch are observable from the IMU and the VIO is gravity-aligned, so VINS-Mono's 4-DoF
graph applies: fixing them keeps the map gravity-aligned (Basalt's roll-pitch factor, in its
strongest form) and halves the unknowns. Edges: sequential VIO relative poses with covariance
`diag(σ_t·d, σ_yaw·d)` growing with travelled distance `d` (no VIO covariance is exposed); loop
edges with D3's covariance and a Huber kernel (robustifier 0.1 m, cuVSLAM's). Solver: sparse
Levenberg–Marquardt via `scipy.optimize.least_squares` with an analytic sparse Jacobian; at ≤ 1000
keyframes this is well under a second per solve. *Alternative:* GTSAM (installed) with iSAM2 for
incremental solves — kept as a later option; a 4-DoF factor is not in its Python wrapper, and the
offline sizes do not need incremental solving yet.

**D5. Frames between keyframes are corrected by their keyframe's correction.**
`world_corr_from_frame = world_corr_from_kf · world_vio_from_kf⁻¹ · world_vio_from_frame` for the
preceding keyframe — cuVSLAM's `CalcFramePose` through `FindKeyframeByFrame`. Continuous within a
keyframe interval; the discontinuity at keyframe boundaries is the redistributed drift and is
reported, not hidden.

**D6. Bounded, incremental in structure, synchronous in execution.**
`LoopClosureBackend.on_keyframe(kf)` does retrieve → verify → (accept → optimise). Throttle:
no verification within `throttle_s` (default 2 s) of an accepted closure. Node cap: beyond
`max_nodes` the oldest keyframes without loop edges are merged into their successor (cuVSLAM's
`ReduceKeyframes`). Executed in the replay loop here; the interface is what a thread would call.

**D7. Verification of the implementation before any real loop exists.**
A synthetic rig test (`test_loop_closure.py`): random 3D points, the real KB4 + extrinsics, a
trajectory that returns to its start with injected drift and yaw error; the test asserts the solver
recovers the loop pose within tolerance, the gates reject a shuffled-correspondence look-alike,
and the graph removes the drift. Plus run5/run6: the backend must accept ≥ 1 near-revisit edge
whose separation in the odometry is already small, and change the trajectory by less than the
VIO-to-cuVSLAM disagreement (its noise floor).

## Risks / Trade-offs

- [ORB on 728×544 fisheye halves gives too few depth-associated points at the image edge, where
  the KB4 fit is worst] → association radius and ORB count are config; the per-pair inlier counts
  are logged so the limiting camera is visible; fall back to centre-weighted detection.
- [HashBow recall is unmeasured] → spatial retrieval carries the near-revisits; appearance recall
  is measured on the circuit log and reported; D1 names the upgrade path.
- [No VIO covariance → sequential edge weights are modelled, not estimated] → the weight model is
  one config block and the metrics are reported for a sweep; the graph is 4-DoF so the model only
  trades yaw against position drift.
- [A wrong accepted loop corrupts the whole trajectory] → cuVSLAM's gates plus a post-optimisation
  check: if any loop edge's residual exceeds 3σ after the solve, the edge is dropped and the graph
  re-solved (Huber already bounds its influence).
- [Verification cost on the host] → ORB + 16-pair matching is ~20–40 ms per candidate; throttle and
  a candidate cap (`max_candidates`, default 3) bound it per keyframe; the per-stage timings are in
  the output (spec: Cost is visible).

## Migration Plan

Nothing deployed changes. New scripts and one new config file; `run_slamrs.py` keeps its default
behaviour when the new flags are absent. Rollback is deleting the `loop_closure/` package.

## Open Questions

- Which camera pairs matter in practice on the circuit log (same-index only, or neighbours too)?
  Logged per edge; does not change the design.
- Whether a planar constraint (cuVSLAM's `planar_constraints`) helps on the lab floor — a config
  switch once the circuit log exists.
