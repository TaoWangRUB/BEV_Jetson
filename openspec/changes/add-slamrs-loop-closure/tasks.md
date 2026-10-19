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

- [x] 8.7 (was 8.8) **Throttle off, travelled-path gate on** (design D1b). The 2 s post-closure
      throttle had skipped 50 / 47 of the 111 / 110 keyframes that had an older keyframe within 1 m.
      Off: 32 / 33 accepted, of which 12 / 4 were a rig standing still at the end of the log closing
      against itself (under 5 cm apart). `min_path_m` = 2 m of travelled path between the two
      keyframes removes exactly those: **20 / 29 accepted, 0 trivial, 0 dropped**, 12 / 19 of them
      opposite-heading (the corridor walked out and back, > 120° between the passes).

| run5 / run6 | 10-18 design | 8.2–8.6 | + 8.7 (defaults now) |
|---|---|---|---|
| hypotheses verified | 603 / 574 | 162 / 166 | 266 / 242 |
| accepted (all spatial) | 3 / 3 | 9 / 9 | **20 / 29** |
| of which opposite-heading (> 120°) | 0 / 0 | 4 / 4 | 12 / 19 |
| of which trivial (< 5 cm apart) | 0 / 0 | 2 / 1 | 0 / 0 |
| rejected with no track at all | 349 / 377 | 12 / 40 | 11 / 28 |
| landmarks per keyframe, median | ~45 / ~55 | 165 / 220 | same |
| first-to-last, VIO → closed | 0.51→0.17 / 0.33→0.11 m | 0.51→0.18 / 0.33→0.11 m | 0.51→0.16 / 0.33→0.11 m |
| closed vs VIO, median / max | 0.14 / 0.48, 0.17 / 0.45 m | 0.14 / 0.48, 0.14 / 0.46 m | 0.14 / 0.50, 0.18 / 0.48 m |
| features per keyframe | 44 / 36 ms | 1 / 1 ms | same |
| verify, whole run | 75 / 70 s | 57 / 52 s | 107 / 94 s |
| solve per accepted closure, median | 2.8 / 3.0 s | 1.1 / 1.1 s | 1.1 / 1.2 s |

The global figures do not move: these logs only hold near-revisits, so three edges already pinned
what twenty can. What the extra edges show is that the pipeline now finds them where they exist,
including from the opposite direction — where only a median 43 tracks survive the forward-backward
check against 507 for same-direction revisits, since a 1.5 m corridor's walls are seen foreshortened
the other way round; cuVSLAM's patch tracker has the same limit.

Open, deliberately left for the circuit log rather than tuned blind:

- [ ] 8.8 **The ratio gate is mis-scaled the other way now.** With the defaults above 144 (run5) /
      58 (run6) hypotheses are rejected on cuVSLAM's inlier ratio — at the 8.2–8.6 stage 76 / 29,
      with a median 106 / 44 joint inliers, all ≥ 40 — because `tracked` now counts every
      NCC-passed track from thousands of projections (median 450 / 588) and the ratio reads
      0.07–0.24 against a 0.3 floor. cuVSLAM's denominator is its own tracker's confirmed
      landmarks. Decide on the circuit log whether those are true closures: if so, the gate wants
      an absolute floor plus a ratio over the best pair, or a tighter NCC.
- [ ] 8.9 Verification cost per candidate rose 2–3× with the landmark count (thousands of
      projections across 4×4 camera pairs). Cap the landmarks per source keyframe (oldest / most
      observed first) before the backend goes near a thread.
- [ ] 8.10 Vectorise the pose graph's residual and Jacobian over edges (the 1.1 s is Python loops).
- [ ] 8.11 The forward-backward threshold (1.5 px) is the stage that decides; sweep it on the
      circuit log together with the NCC floor, with the stage counts as the readout.

## 9. Mac review of the tuning branch (2026-10-19) — what 8.8 is really about

- [x] 9.1 The tuning branch reproduces on the Mac: 9/9 tests; run6 29 accepted (same as the Dell),
      run5 18 (Dell 20 — run5's VIO differs between the builds, 1.2/8.1); verification 502 / 550 ms
      median per keyframe with candidates; the Rerun scenes (`lc_run{5,6}/compare*.rrd`) regenerated.
- [x] 9.2 `run_slamrs_lc.py` aligned with the GPU-lane merge: the VIO csv carries `fe_ms` / `est_ms`
      like `run_slamrs.py`, and `--gpu` selects the lane.
- [x] 9.3 **8.8 measured: the ratio's denominator is not the lever.** Of run6's 91 hypotheses with
      ≥ 40 joint inliers, the pooled gate (current) passes 37; cuVSLAM's literal single-keyframe gate
      (inliers and ratio over the best source keyframe) passes 14, and "pooled floor + best-source
      ratio" 38. The candidate keyframe is the best source in only 6–14 % of cases (median 14 inliers
      per keyframe) — pooling the keyframes in sight is what makes closures possible on this rig, so a
      per-keyframe gate would cost closures, not add them.
- [x] 9.4 **cuVSLAM's second step re-tracks from the first solution; doing so here changes nothing.**
      On 116 hypotheses with ≥ 20 inliers: inliers 46 → 48 median, the pose moves 2 cm, gate passes
      38 → 37. Not adopted; the deviation was not the cause.
- [x] 9.5 **The end-of-log loop on run6 is real and every gate rejects it.** Keyframes 224–242 against
      13–17 are the only long revisit in the data (≈70 s apart). All 45 hypotheses solve to the same
      correction, 0.30 m / 7.0° — the VIO's own end-to-start gap — on ~50 inliers of ~570 tracks, the
      rest 10–20 px off. The cause is landmark depth, not tracking: the early keyframes' landmarks
      were frozen at marginalisation after the slow start, fit their own observations to 0.1 px (any
      depth does at low parallax), and the VIO later moves them a median 100 m (to infinity, p90
      1.4–2.3 km). Re-verifying with those later positions raises the inliers 42 → 162 and accepts 5
      of 30 (ratio 0.28, still under the floor); blindly refreshing every keyframe is wrong, though —
      mid-run keyframes' self-consistency degrades 0.4 → 1.9 px (D2a). Tracks at 5–10 px are a third of
      the accepted edges' tracks too, so the 5 px threshold is tight relative to the depth accuracy
      even where depths are fine.
- [x] 9.6 **8.9 measured:** capping each source keyframe to its 150 oldest landmarks keeps the
      acceptances (the ratio improves: 12 → 20 passes on a 25-hypothesis sample) and cuts
      verification 196 → 139 ms; 100 loses acceptances (15), 60 loses most (5). 150 is the number to
      adopt with 8.9.

Next, in order — each measurable on run6's long loop before the circuit log exists:

- [x] 9.7 **Per-landmark depth confidence (design D2c) — the end-of-log loop closes.** The tracker
      keeps finalised records open: a landmark's position follows the latest snapshot while it is
      alive, and its parallax and observation count (from the frontend's keypoint ids and the
      window poses) grow with it; landmarks under 2° of parallax or 5 observations are weak and not
      tracked. Measured on run6's 111 end-of-log hypotheses, which every earlier variant rejected:
      the far / bearing-only class gave 0–20 inliers of ~200 (a zero-parallax landmark here is near
      and unknown, not far); a 2 px or 20 px self-projection bound on the refresh gave 0 closures
      (the early keyframes' frozen poses are degrees off, and the later geometry fails the test where
      it matters); unbounded refresh without classes gave 10 edges; **unbounded refresh with the weak
      exclusion gives 19 long-loop edges (106–270 inliers of ~375, ratio 0.62–0.89), 47 edges in all,
      0 dropped**, run6 first-to-last 0.33 → 0.12 m; run5 28 edges, 10 across its own end-of-log
      loop, 0.53 → 0.14 m. Two run6 replays byte-identical. Verification 38 / 53 s per log (down
      from 94 / 107 s: weak landmarks are not tracked), solve 0.40 / 0.47 s. Three tests added:
      parallax measurement, the refresh rule, far landmarks as directions (the code path stays, off
      by default). Resolves 8.8: the ratio's denominator was full of depthless landmarks, not
      mis-scaled. README table, design D2c.
- [x] 9.8 Moot after 9.7: the accepted long-loop edges sit at ratios 0.62–0.89 with the single
      5 px threshold; no two-tier count is needed on run6. Re-check on the circuit log only if a
      true loop is rejected on ratio there.
- [ ] 9.9 8.10 (vectorised residual/Jacobian) next; 8.9's cap only if verification cost matters
      again after 9.7 halved it; then the circuit log (7.x) and the sweep (8.11); 5.4 (node cap)
      and threading after that.

## Workflow follow-up

- Archive the change after 6.1 is recorded; 7.x may complete after archive if the board is not
  reachable, in which case the README states that the benefit is unmeasured.
