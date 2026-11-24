# Proposal

## Why

`add-slamrs-loop-closure` (archived 2026-10-19) answered the question it set out to answer: the
slam-rs VIO plus the loop-closing backend recognises and closes real revisits on this rig, with a
deterministic replay — 47 accepted closures on run6 (19 across the 70 s end-of-log loop), 28 on
run5, first-to-last distance 0.33 → 0.12 m and 0.53 → 0.14 m. But what answered it is not a
package: the VIO is Rust reached through a PyO3 shell, the backend is Python plus OpenCV plus
SciPy, and the two meet in a Python driver. On the host the backend spends ~345 ms per verified
keyframe, mostly in Python loops around OpenCV calls; nothing of it can run on the TX2 as a node,
and the GPU lane that the frontend already has (`--features gpu-wgpu`) is unreachable from a
Python process that embeds the CPU build.

The user's requirement for the next step is explicit: one native implementation, Rust or C++,
including loop closure, that can become the rig's SLAM package and later run in real time on the
TX2. Rust beside slam-rs was chosen over a C++ route (Basalt's mapper is batch and its loop closure
unpublished; the port would have to be rewritten anyway) and over a fork of slam-rs (the backend
consumes the estimator's snapshot API and need not touch the estimator).

## What Changes

- **A Rust workspace in `rust/` with the crate `slamrs-lc`**, depending on the submodule's
  `slam-rs` crate by path and locked to the submodule's own `Cargo.lock` so the embedded VIO is the
  same binary code the Python build ran. One binary, `slamrs-lc replay <log> <out>`, replaces
  `run_slamrs_lc.py`: the same raw-log reader (framesets, IMU preparation, 2× halving), the same
  VIO call sequence, the same output files column for column, so `compare.py`, `rerun_compare.py`
  and the fixtures in `datasets/replay_out/lc_run*` read it unchanged.
- **The backend ported module for module** — keyframe records with the observed-landmark set, the
  host-rig-frame consistency rule, open records with refresh and depth confidence (D2a/b/c);
  spatial retrieval with the time and travelled-path gates (D1, D1b); KLT verification with the
  forward-backward and NCC checks and cuVSLAM's two-step solver and gates (D2, D3); the 4-DoF
  pose graph with the 3σ edge check and the per-frame correction (D4, D5); the synchronous
  `on_keyframe` interface (D6).
- **Parity as the acceptance test.** The Python outputs are fixtures. The VIO rows must be
  byte-identical; the backend's edges and corrected poses must match within a stated tolerance,
  since its Lucas-Kanade is a re-implementation of OpenCV's shape, not OpenCV. A tool,
  `scripts/slamrs/compare_port.py`, reports the comparison stage by stage so a divergence names
  the stage it came from.
- **Not ported, by measurement:** the appearance route (ORB + HashBow, off by default and never
  the source of a closure here), the far / bearing-only landmark class (measured not to help,
  9.7), the keyframe export and reload. The Python package stays as the reference and the
  research bench; the Rust crate is the product.

## Capabilities

- `vio-loop-closure`: ADDED requirement — a native implementation that reproduces the reference.

## Impact

- New: `rust/` (workspace, crate, lockfile copy), `scripts/slamrs/compare_port.py`.
- Unchanged: `scripts/slamrs/**`, `config/slamrs/*.json` (the crate reads the same files),
  the slam-rs submodule.
- Follow-ups the design names but this change does not claim: the backend on its own thread, the
  node cap, the GPU lane measured on the TX2, Rerun logging from the binary.
