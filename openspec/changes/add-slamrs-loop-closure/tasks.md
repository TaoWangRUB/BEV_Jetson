# Tasks

## 1. Environment and data

- [x] 1.1 Build `slam_rs/_core.so` on the host doing the work and verify `run_slamrs.py --dry-run`
      passes on one log (on macOS the artefact is `lib_core.dylib`; copy it to `_core.so`). Done on
      the Mac 2026-10-18 (cargo 1.99, CPU lane); the dry run and full replays pass.
- [x] 1.2 Copy at least two raw logs (run5, run6) and their cuVSLAM `obs_*` bags from the Dell host
      into `datasets/` and verify `run_slamrs.py` reproduces the tracked-set count recorded in
      `scripts/README.md`. run5: 1756 of 1762 sets tracked, run6: 1746 of 1758 — both equal to the
      Dell's reference csvs; run6 positions agree to 0.6 mm median, run5's reference
      (`slamrs_run5_cpu.csv`, older build) differs by 8 cm median with the same tracked set.
- [x] 1.3 Add `config/slamrs/bev_lc_config.json` holding every retrieval, verification and graph
      parameter named in design.md with its default, and verify the driver loads it.

## 2. Keyframe export from the VIO

- [x] 2.1 Keyframe export beside the VIO: `run_slamrs_lc.py --keyframes-out <dir>` writes, for every
      keyframe the estimator votes, the four half-resolution frames, the IMU pose, the landmarks
      (ids, positions, age) and the frontend's observations of them at keyframe time. The pure-VIO
      csv is identical with and without the backend (run5, first 400 sets: 400/400 rows equal in
      every column but the wall-clock `track_ms`) — spec: Odometry is preserved.
- [x] 2.2 `loop_closure/keyframes.py` (tracker + load/export) round-trips a run5 export: same stamps,
      poses, landmark counts and observations. Two defects found and fixed on real data (design D2a):
      landmarks accumulated over the keyframe's life, stored in the host rig frame at the snapshot
      they were last seen in — projection-to-observation error 5.7 px → 0.50 px median.

## 3. Retrieval

- [x] 3.1 Spatial retrieval (radius around the current VIO pose, minimum time separation): on run5
      near-revisits appear (470 of 574 run6 candidates are spatial) and nothing younger than the
      separation does (unit test).
- [x] 3.2 ORB-on-four-cameras + hash bag-of-words retrieval (16 bits, L1 scoring, inverted index,
      time gate): near-duplicates score highest (unit test) and the index is deterministic across
      two runs (unit test + run6 repeat).
- [x] 3.3 Both routes unit-tested on the synthetic rig in `test_loop_closure.py`.

## 4. Verification

- [x] 4.1 Correspondences: KLT tracking of the stored landmark appearance from the predicted
      projection, forward-backward checked and NCC-gated (cuVSLAM's method; design D2), with ORB
      descriptor matching kept as the fallback. On run5 keyframes 34→36: 208 tracks, 105 joint PnP
      inliers, pose within 7 cm / 0.4° of the VIO, unchanged with the guess moved 30 cm.
- [x] 4.2 Two-step multi-camera PnP with cuVSLAM's gates and covariance scaling: the synthetic test
      recovers an injected loop pose within 2 cm / 0.5° and rejects a shuffled-correspondence
      look-alike with the recorded reason.
- [x] 4.3 Every hypothesis is recorded to `hypotheses.csv` with reason and per-pair counts: run5 600
      rows (3 ok, 111 few_landmarks, 7 inconsistent_steps, 134 no_pair_solution, 345 no_matches),
      run6 574 rows (3 ok).

## 5. Pose graph

- [x] 5.1 4-DoF graph (sequential VIO edges with distance-scaled covariance, loop edges with Huber,
      fixed roll/pitch) solved with sparse LM: the synthetic test removes an injected 0.47 m drift
      with the loop edge residual inside its covariance and roll/pitch untouched.
- [x] 5.2 Throttling and the post-solve 3σ edge check: unit-tested (a closure inside the throttle
      window is skipped; a wrong edge is dropped and the graph re-solved). run5/run6: 19/20 keyframes
      throttled, 0 edges dropped.
- [x] 5.3 Per-frame correction from the preceding keyframe: continuity unit-tested; on run5/run6 the
      corrected frames follow the keyframes (compare.py: same path length as the VIO).
- [ ] 5.4 Node cap with keyframe merging (`max_nodes`, cuVSLAM's `ReduceKeyframes`): `reduce_to` is
      an explicit no-op; offline graphs of ≤ 260 nodes solve in ~0.8–1.0 s so the cap is not needed
      yet. Verify with a test when it is implemented.

## 6. Driver, outputs and metrics

- [x] 6.1 `run_slamrs_lc.py` writes `slamrs_vio.csv`, `slamrs_lc.csv`, `keyframes_lc.csv`,
      `loop_edges.csv`, `hypotheses.csv`, `timings.csv`, `summary.json`. Reproducibility verified:
      two run6 replays gave byte-identical `slamrs_vio.csv` (all columns but `track_ms`),
      `slamrs_lc.csv`, `keyframes_lc.csv`, `loop_edges.csv` and `hypotheses.csv` (spec: Acceptance
      is reproducible).
- [x] 6.2 `compare.py <bag> <vio csv> <lc dir> [tape]` reports path length, first-to-last distance and
      per-edge separation for both trajectories (verified on run5 and run6, output in README).
- [x] 6.3 `rerun_compare.py --lc-dir` draws the loop-closed path (purple) with its edges beside the
      three existing trajectories: `datasets/replay_out/lc_run{5,6}/compare.rrd`.
- [x] 6.4 run5 and run6: 3 near-revisit edges accepted on each (all spatial, 40–145 inliers, ratio
      0.51–0.64, every accepted pose 0.13–0.42 m and 2.5–5.4° from the odometry guess); the
      corrected path differs from the VIO by 0.15–0.18 m median, 0.45–0.49 m max — under the
      VIO-to-cuVSLAM Sim(3) disagreement (0.64–0.78 m RMS). First-to-last distance: run5 0.53 m →
      0.18 m, run6 0.33 m → 0.10 m. Per keyframe: features 21–26 ms, verification 224–239 ms median
      (p95 ≈ 360–410 ms, three candidates × KLT over the keyframes in sight), optimisation
      0.8–1.0 s per accepted closure. Recorded in `scripts/README.md`.

## 7. The log that answers the question

- [ ] 7.1 Record a closed-circuit log on the TX2 (two or three laps, 20 fps, start and end on a taped
      mark, `tape_metres.txt`) with `log_rig.sh`, and verify it replays through `run_slamrs.py`
- [ ] 7.2 Run the backend on it and record in `scripts/README.md` the return-to-start error of the
      VIO, the corrected trajectory and cuVSLAM's `slam_path`, with the per-stage timings

## 8. After the Dell review (2026-10-19) — the four changes, measured on run5 / run6

- [x] 8.1 Reproduce the Mac result on the Dell (x86, cargo 1.98; `pip install scipy>=1.14` in `.venv`,
      the system SciPy 1.11 breaks on NumPy 2): 9/9 synthetic tests pass; run6's three edges are
      identical to the Mac's to the inlier count; run5's three differ because run5's VIO itself
      differs by 8 cm between the two builds (1.2) — determinism holds per build, not per log.
- [x] 8.2 Spatial radius = the tracker's reach (design D1a). Measured on the "before" hypotheses:
      within 0.5 m 67–82 % of candidates produced tracks and 17–59 % reached 40; 0.5–1 m 2–11 %;
      beyond 1 m none, on 253 of 505 and 213 of 470 spatial hypotheses. `spatial_radius_m` 3 → 1.
- [x] 8.3 ORB off by default (D1a): every appearance hypothesis (98 / 104) died in KLT with zero
      tracks and the descriptor fallback the design described never ran (it only triggered without
      images). `features.orb_enabled` gates ORB, the bag of words, the appearance route and the
      fallback. Feature extraction 36–44 ms → 1 ms per keyframe.
- [x] 8.4 Keyframe records hold every observed landmark (D2b): `KeyframeTracker` accumulates the
      frontend's ids at keyframe time as they gain positions, with a `hosted` flag; unit-tested.
      Landmarks per keyframe median 45 → 165 (run5) / 220 (run6).
- [x] 8.5 Per-stage counts on every hypothesis (`n_sel … n_ncc` in `hypotheses.csv`): the
      forward-backward check is now the binding stage (9–13 % of forward tracks survive it; NCC then
      passes 78–85 %); the remaining `no_matches` (12 / 40) all tracked forward and died there.
- [x] 8.6 Analytic sparse Jacobian for the pose graph (D4), checked against central differences in
      a test. Solve per accepted closure 2.8–3.0 s → 1.1 s median on ~250 nodes; what remains is the
      Python per-edge loop in the residual and Jacobian, not the solver.

| run5 / run6 | before (10-18 design) | after |
|---|---|---|
| hypotheses verified | 603 / 574 | 162 / 166 |
| accepted (all spatial) | 3 / 3 | 9 / 9 |
| rejected with no track at all | 349 / 377 | 12 / 40 |
| landmarks per keyframe, median | ~45 / ~55 | 165 / 220 |
| first-to-last, VIO → closed | 0.51→0.17 / 0.33→0.11 m | 0.51→0.18 / 0.33→0.11 m |
| closed vs VIO, median / max | 0.14 / 0.48, 0.17 / 0.45 m | 0.14 / 0.48, 0.14 / 0.46 m |
| features per keyframe | 44 / 36 ms | 1 / 1 ms |
| verify per keyframe with candidates, median | 379 / 418 ms (204 / 194 kf) | 902 / 750 ms (61 / 63 kf) |
| verify, whole run | 75 / 70 s | 57 / 52 s |
| solve per accepted closure, median | 2.8 / 3.0 s | 1.1 / 1.1 s |

The global figures do not move: these logs only hold near-revisits, so three edges already pinned
what nine can. What the extra edges show is that the pipeline now finds them where they exist.

Open, deliberately left for the circuit log rather than tuned blind:

- [ ] 8.7 **The ratio gate is mis-scaled the other way now.** 76 (run5) / 29 (run6) hypotheses were
      rejected on cuVSLAM's inlier ratio with a median 106 / 44 joint inliers, all ≥ 40, because
      `tracked` now counts every NCC-passed track from thousands of projections (median 450 / 588)
      and the ratio reads 0.07–0.24 against a 0.3 floor. cuVSLAM's denominator is its own tracker's
      confirmed landmarks. Decide on the circuit log whether those are true closures: if so, the
      gate wants an absolute floor plus a ratio over the best pair, or a tighter NCC.
- [ ] 8.8 Stationary keyframes close trivially: the rig rests at the end of both logs, so keyframes
      28 s apart at the same spot (separation 0.5–3 cm) pass every gate. Harmless, useless; a
      travelled-path gate (≥ N m between the two keyframes) is the natural filter.
- [ ] 8.9 Verification cost per candidate rose 2–3× with the landmark count (thousands of
      projections across 4×4 camera pairs). Cap the landmarks per source keyframe (oldest / most
      observed first) before the backend goes near a thread.
- [ ] 8.10 Vectorise the pose graph's residual and Jacobian over edges (the 1.1 s is Python loops).
- [ ] 8.11 The forward-backward threshold (1.5 px) is the stage that decides; sweep it on the
      circuit log together with the NCC floor, with the stage counts as the readout.

## Workflow follow-up

- Archive the change after 6.1 is recorded; 7.x may complete after archive if the board is not
  reachable, in which case the README states that the benefit is unmeasured.
