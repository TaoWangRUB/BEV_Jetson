# Tasks

## 1. Environment and data

- [ ] 1.1 Build `slam_rs/_core.so` on the host doing the work and verify `run_slamrs.py --dry-run`
      passes on one log (on macOS the artefact is `lib_core.dylib`; copy it to `_core.so`)
- [ ] 1.2 Copy at least two raw logs (run5, run6) and their cuVSLAM `obs_*` bags from the Dell host
      into `datasets/` and verify `run_slamrs.py` reproduces the tracked-set count recorded in
      `scripts/README.md`
- [ ] 1.3 Add `config/slamrs/bev_lc_config.json` holding every retrieval, verification and graph
      parameter named in design.md with its default, and verify the driver loads it

## 2. Keyframe export from the VIO

- [ ] 2.1 Extend `run_slamrs.py` with `--keyframes-out <dir>`: on every snapshot keyframe write the
      four half-resolution frames, the IMU pose, the landmark ids/positions and the frame stamp;
      verify the pure-VIO CSV is byte-identical with and without the flag (spec: Odometry is
      preserved)
- [ ] 2.2 Write `loop_closure/keyframes.py` (load/iterate an export) and verify a round trip on a
      run5 export returns the same stamps, poses and landmark counts

## 3. Retrieval

- [ ] 3.1 Implement spatial retrieval (radius around the current VIO pose, minimum time separation)
      and verify on run5 that near-revisits appear and that nothing younger than the separation does
- [ ] 3.2 Implement ORB-on-four-cameras + hash bag-of-words retrieval (16 bits, L1 scoring, inverted
      index, time gate) and verify a keyframe's own near-duplicates score highest on run5 and that
      the index is deterministic across two runs
- [ ] 3.3 Unit-test both routes on the synthetic rig in `test_loop_closure.py`: the true revisit is
      retrieved, the time gate holds

## 4. Verification

- [ ] 4.1 Implement landmark-to-ORB depth association and cross-camera matching (16 pairs, Hamming 70,
      ratio 1.2, mutual) with KB4 unprojection, and verify on the synthetic rig that a known revisit
      yields correspondences whose bearings reproject within 1 px
- [ ] 4.2 Implement the two-step multi-camera PnP with cuVSLAM's gates and covariance scaling, and
      verify the synthetic test recovers an injected loop pose within 2 cm / 0.5° and rejects a
      shuffled-correspondence look-alike with the recorded reason
- [ ] 4.3 Record every hypothesis (accepted or rejected, with reason and per-pair inlier counts) to
      `hypotheses.csv` and verify the file lists one row per candidate on run5

## 5. Pose graph

- [ ] 5.1 Implement the 4-DoF graph (sequential VIO edges with distance-scaled covariance, loop edges
      with Huber, fixed roll/pitch) solved with sparse LM, and verify the synthetic test removes an
      injected drift so that the loop edge residual is within its covariance
- [ ] 5.2 Implement throttling, the node cap with keyframe merging, and the post-solve 3σ edge check,
      and verify by test that a closure inside the throttle window is skipped and a bad edge is
      dropped and re-solved
- [ ] 5.3 Implement per-frame correction from the preceding keyframe and verify continuity: no step
      larger than the VIO's own inter-frame motion except at keyframe boundaries, which are reported

## 6. Driver, outputs and metrics

- [ ] 6.1 Write `run_slamrs_lc.py` (replay → VIO → `on_keyframe` → outputs: `slamrs_lc.csv`,
      `keyframes_lc.csv`, `loop_edges.csv`, `hypotheses.csv`, `timings.csv`) and verify two runs on
      run5 give byte-identical outputs (spec: Acceptance is reproducible)
- [ ] 6.2 Extend `compare.py` to report path length, first-to-last distance and per-edge separation
      for both trajectories, plus `tape_metres.txt` when present, and verify the report on run5
- [ ] 6.3 Extend `rerun_compare.py` with the corrected path and loop edges and verify the `.rrd`
      shows three trajectories and the edges on run5
- [ ] 6.4 Verify on run5 and run6 that ≥ 1 near-revisit edge is accepted and the corrected path
      differs from the VIO by less than the VIO-to-cuVSLAM disagreement; record the numbers and the
      per-keyframe timings in `scripts/README.md`

## 7. The log that answers the question

- [ ] 7.1 Record a closed-circuit log on the TX2 (two or three laps, 20 fps, start and end on a taped
      mark, `tape_metres.txt`) with `log_rig.sh`, and verify it replays through `run_slamrs.py`
- [ ] 7.2 Run the backend on it and record in `scripts/README.md` the return-to-start error of the
      VIO, the corrected trajectory and cuVSLAM's `slam_path`, with the per-stage timings

## Workflow follow-up

- Archive the change after 6.4 is recorded; 7.x may complete after archive if the board is not
  reachable, in which case the README states that the benefit is unmeasured.
