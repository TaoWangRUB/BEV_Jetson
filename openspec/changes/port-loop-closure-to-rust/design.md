# Design

Decisions continue the numbering of `add-slamrs-loop-closure` (D1–D6 there).

## D7. A crate beside slam-rs, not a fork, not C++

`rust/slamrs-lc` depends on `slam-rs` by path into the submodule
(`third_party/rerun-examples/packages/slam-rs/crates/slam-rs`) and uses only its public API:
`Vio::with_backend`, `push_imu`, `track`, `estimator().snapshot()`, `frontend().frame()`, the
calibration and config loaders and the `RigCamera` projection models. The estimator is untouched.
A fork would have to be rebased on every upstream change; a C++ route (Basalt + a new backend)
would re-implement the whole pipeline against an unpublished loop closure. The crate is the
product; the Python package remains the reference and the bench.

## D8. Parity by fixtures, in two grades

The Python driver's outputs on run5 and run6 are the fixtures (`datasets/replay_out/lc_run5`,
`lc_run6`).

- **Byte-identical: the VIO.** Same raw-log reader (framesets by nearest stamp within 1 ms,
  stamp = floor(mean of four), IMU lag 3 731 525 ns, 2 s rest bias removal, along-gravity
  correction, non-advancing samples dropped, `(a+b+c+d+2)>>2` halving) and the same call sequence
  give the same inputs; the same dependency versions give the same arithmetic. This needed the
  submodule's `Cargo.lock` copied into `rust/`: with the resolver's own choice (`glam` 0.33.12
  instead of 0.33.6 through `kornia-algebra`) the poses differed by 2.7 cm median on run6. Output
  formatting is C `%.9g`, reproduced in `fmtg.rs`.
- **Within tolerance: the backend.** Its Lucas-Kanade is written from OpenCV's shape, not linked
  to OpenCV (D9), and its PnP is not EPnP (D10), so correspondences differ by a few and RANSAC's
  consensus by a few more. The acceptance test is: the same keyframes; accepted edges that
  overlap the reference's set with the long loop closed on both logs; corrected keyframe
  positions within centimetres; first-to-last distance within the same band. `compare_port.py`
  prints every stage (`n_sel … n_ncc`, inliers, reasons) so a divergence names its stage.

## D9. Lucas-Kanade re-implemented from OpenCV's shape

`calcOpticalFlowPyrLK` with `OPTFLOW_USE_INITIAL_FLOW` is reproduced structurally: an 8-bit
pyramid whose `pyrDown` is bit-exact (5-tap [1 4 6 4 1]², reflect-101 borders, `(s+128)>>8`),
levels built only while larger than the window, every level padded with a reflected border of
`win + 2` so a window may partly leave the image as OpenCV's may; a fixed template window with
central-difference gradients, Gauss-Newton on the displacement to 0.01 px or 30 iterations, the
displacement doubled between levels, the minimum-eigenvalue gate in the same units
(OpenCV's 1e-4 on Scharr/`FLT_SCALE` gradients = 0.1024 on true gradients), and `err` as the mean
absolute window residual. The forward-backward and NCC checks are as in Python. Memory is 8-bit
(≈1.6 MB per keyframe for four cameras, as before).

## D10. PnP by Gauss-Newton from the guess inside RANSAC

Python used `cv2.solvePnPRansac` (EPnP minimal solver, refinement on inliers). The port draws
four-point samples and solves each by Levenberg-Marquardt from the odometry guess, then counts
consensus, refines on the consensus set and recounts. This is sound here because of D1a: every
verified hypothesis comes with a guess inside the tracker's reach, and four points pin the pose
from that guess; a wrong local minimum on a bad sample loses that sample to consensus, as a wrong
EPnP solution would. Step 2 is the same Huber-weighted joint refinement (LM, numeric Jacobian)
with the same inlier recount and gates. Seeded xorshift, so a replay is reproducible.

## D11. Pose graph with block-sparse normal equations

The 4-DoF model, the distance-scaled sequential sigmas, the Huber loop edges, the 3σ post-check
and the per-frame correction are unchanged. The solver is LM with an analytic Jacobian (checked
numerically in Python, 8.6) on block-sparse normal equations solved by block-Jacobi
preconditioned conjugate gradients with node 0 fixed; it takes milliseconds where Python's loops
took 1.1 s, which closes 8.10.

## D12. What is not ported

- The appearance route (ORB detector, HashBow). Off by default since 8.3 and never the source of a
  closure on this rig. The config keys remain accepted and ignored.
- The far / bearing-only landmark class (`far_parallax_deg` > 0). Measured not to help (9.7);
  the key is read, the class is counted, correspondences are not made from it.
- Keyframe export/reload (`--keyframes-out`); the Python bench keeps it.
- The node cap (`max_nodes`, 5.4) — open in the reference as well.

## D13. Threads and the GPU lane

The backend is synchronous, called after every `track()`, as the reference. Its interface is
what a thread would call (`on_keyframe(records, index)` with the records owned by the tracker),
so moving it behind a channel with cuVSLAM's throttling is a wiring change, not a design change.
The frontend's GPU lane is a feature flag of the same binary (`--features gpu-wgpu`, `--gpu`);
measuring it on the TX2 is a follow-up task, not a claim of this change.
