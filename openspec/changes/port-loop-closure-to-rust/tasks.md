# Tasks

Numbers are from the Mac (arm64, cargo 1.99, release profile, one frontend thread), 2026-10-20,
against the Python fixtures of 2026-10-19 in `datasets/replay_out/lc_run5` and `lc_run6`.

## 1. Workspace and odometry parity (D7, D8)

- [x] 1.1 `rust/` workspace with the crate `slamrs-lc`; `slam-rs` by path into the submodule; the
  submodule's `Cargo.lock` copied in. With the resolver's own choices (`glam` 0.33.12 instead of
  0.33.6 through `kornia-algebra`) the run6 poses differed from Python's by 2.7 cm median; with
  the copy the VIO csv agrees row for row in every pose column on run5 and run6 (1762 / 1758 rows).
- [x] 1.2 Raw-log reader (`log.rs`): framesets, IMU preparation, 2× halving; `slamrs-lc debug-dump`
  compared bit for bit with what `run_slamrs.py` feeds the same core.
- [x] 1.3 `slamrs-lc replay <log> <out>` writes `slamrs_vio.csv` with the driver's columns in C
  `%.9g` (`fmtg.rs`, unit-tested against numpy's formatting). `--no-lc`, `--max-sets`, `--threads`,
  `--gpu`, `--calib`, `--config`, `--lc-config` as in the Python driver.

## 2. The backend, module for module

- [x] 2.1 `keyframes.rs`: pending records that follow the window, observed-landmark set (D2b),
  host-rig-frame storage (D2a), open records with refresh and parallax / observation confidence,
  weak exclusion (D2c). The records come out identical to Python's: 243 / 251 keyframes at the same
  stamps, 52 840 + 3 962 / 38 290 + 2 894 landmarks in the same classes, 15 414 refreshed on run6,
  `n_assoc` medians 183 / 138 on both sides.
- [x] 2.2 `retrieval.rs`: spatial candidates with the time and travelled-path gates (D1, D1b).
- [x] 2.3 `klt.rs` (D9): padded 8-bit pyramid with bit-exact `pyrDown`, pyramidal Lucas-Kanade
  with OpenCV's criteria, forward-backward and NCC checks; unit-tested on a synthetic shift.
  Rewriting the inner loops around a shared-offset window sampler (every sample of a window has
  the same fractional offset, so the four bilinear weights are computed once per iteration and
  the rows read contiguously) gave the same output to the byte and halved the cost: 545 → 306 ms
  median per keyframe with candidates on run6.
- [x] 2.4 `pnp.rs` (D10): RANSAC over four-point Gauss-Newton samples from the guess, consensus
  refinement, Huber-weighted joint refinement; unit-tested through 30 % outliers on the real
  calibration (pose recovered within 1 cm / 0.5°).
- [x] 2.5 `verify.rs`: cuVSLAM's correspondence step and two-step solver with the same gates and
  covariance scaling, the same stage counters. On run6 the medians per hypothesis are `n_sel`
  1657 = 1657, `n_proj` 2833 vs 2868, `n_klt` 1353 vs 1372, `n_fb` 164 vs 142, `n_ncc` 134 vs 127,
  tracked 111 = 111: the port's tracker passes a few more tracks through the forward-backward
  check, everything before it agrees.
- [x] 2.6 `posegraph.rs` (D11): 4-DoF graph, analytic Jacobian, block-sparse LM with block-Jacobi
  PCG, 3σ edge check; unit-tested on injected drift. A solve takes milliseconds (Python: 0.4–1.1 s),
  which closes 8.10 for the port.
- [x] 2.7 `backend.rs`: `on_keyframe`, in-sight neighbours, accept / solve / check, per-frame
  correction, and the writers for `slamrs_lc.csv`, `keyframes_lc.csv`, `loop_edges.csv`,
  `hypotheses.csv`, `timings.csv`, `summary.json` in the driver's formats, so `compare.py` and
  `rerun_compare.py --lc-dir` read the output unchanged.
- [x] 2.8 One bug worth recording: nalgebra's `Isometry3 * Vector3` applies the rotation only.
  Every point transform goes through `geom::tf`; before that the first replay tracked (KLT reaches
  the un-translated prediction) and solved nothing (`no_pair_solution` on 212 of 304 hypotheses).

## 3. Parity on the fixtures (D8)

- [x] 3.1 `scripts/slamrs/compare_port.py <python dir> <rust dir>`: VIO rows, keyframes, corrected
  poses, edge sets, reasons, stage medians, timings.
- [x] 3.2 run6: 52 edges accepted vs 47 (44 in common; 20 vs 19 across the 70 s end-of-log loop);
  corrected keyframe positions within 4 mm median / 4.8 cm max of Python's; first-to-last
  0.131 vs 0.127 m (VIO 0.33 m). The eight edges only the port accepts join the same keyframes
  as neighbouring accepted edges (140–144 → 78–116, 204 → 43).
- [x] 3.3 run5: 42 vs 28 (24 in common; 16 vs 11 long-loop); 6 mm / 2.7 cm; first-to-last
  0.151 vs 0.149 m (VIO 0.53 m).
- [x] 3.4 Deterministic: two replays of one build give byte-identical hypotheses, edges and
  trajectories (run6, `timings.csv` excepted).
- [x] 3.5 Cost on the Mac, one thread: the whole run6 replay 102 s (VIO 65 s, backend 46 s —
  the port verifies up to three candidates where Python stopped at its first acceptance, hence
  more hypotheses, 211 vs 189); verify 306 ms median per keyframe with candidates (Python 345 ms).

## 4. Not in this change

- [ ] 4.1 The backend on its own thread with cuVSLAM's throttling (D13); today it runs in the
  replay loop after every `track()`.
- [ ] 4.2 The GPU lane on the TX2: build with `--features gpu-wgpu`, run with `--gpu`, time the
  frontend and the backend against the 20 fps budget. Measured on the Mac first (2026-10-23,
  README "Performance of the native binary"): CPU 13.5 / 17.1 ms per set on run6 / run5, Metal
  lane 14.6 / 12.5 ms with a flat ~8 ms frontend, same closures; whole run6 replay 76 s on one
  thread once the frames are read on a prefetch thread (was 102 s).
- [ ] 4.3 Node cap (`max_nodes`, the reference's 5.4).
- [ ] 4.4 Rerun logging from the binary; the Python `rerun_compare.py --lc-dir` draws its output
  meanwhile (`datasets/replay_out/rs_lc_run6/compare.rrd`).
- [ ] 4.5 Whether to hold the port's forward-backward check to Python's edge count (2.5): both
  close the long loops and the corrected trajectories agree within centimetres, so the extra
  edges are left as they are until a closed-circuit log (reference 7.x) says which is right.
