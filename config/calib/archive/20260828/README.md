# IMX296 rig calibration — session 2026-08-28/29 (SUPERSEDED, archived 2026-08-31)

This is the complete surviving record of the calibration solved from
`datasets/calib_20260828`. **The raw recordings have been deleted** (36.4 GB — see
`DELETED_INVENTORY.txt` for exactly what, file by file, with sizes).

## Why it is superseded, and why it is still here

On 2026-08-31 the rig changed: cameras reconnected, **lenses refocused**, trigger source
changed to an F401. Refocusing moves focal length, principal point and the distortion curve
together, so every number below describes optics that no longer exist. §3R of
`openspec/changes/retarget-vo-to-imx296-rig/tasks.md` re-measures all of it.

It is kept because it is the **prior every new number gets checked against**, and because
**it can never be re-recorded** — the hardware state it describes is gone. Specifically:

| new result (§3R) | check it against |
|---|---|
| intrinsics, `omni-radtan` | `solve/*/log1-results-cam.txt` — 0.28–0.40 px reprojection, projection/distortion ± bounds |
| four pairwise baselines | `results/rig_ext.yaml` — 147.8 / 148.7 / 149.1 / 149.1 mm, agreeing to 1.3 mm |
| ring-closure residual | `results/closed.yaml` — 3.63° / 9.2 mm before, 0.0000° / 0.043 mm after, per-edge corrections 0.57–1.31° |
| Δ (camera↔IMU) | `results/rig_ext.yaml: camera_imu` — `timeshift_cam_imu_s: -0.008062483899950911` (**−8.06 ms**), residuals 0.366 px / 0.00156 rad s⁻¹ / 0.0424 m s⁻² |
| Δ's validity check | `solve/cam_imu/CAM_IMU_shift10-*` — the +10.000 ms injection that moved the estimate by −10.000 ms (residual −1.2 µs). Re-run this on the new fit |
| virtual-stereo epipolar | `evidence/` renders + the 0.45 / 1.79 / 2.55 / 0.66 px medians recorded in tasks.md 3.7 |

If the new Δ differs from −8.06 ms by roughly `(old_pulse − new_pulse)/2`, the remainder is
the constant readout term — which is the evidence task 3.6b is waiting for.

## What is here

- `results/` — the solved numbers. `intr_camN.yaml` (per-camera intrinsics), `rig_ext.yaml`
  (four pairwise extrinsics as measured, plus the camera↔IMU block), `closed.yaml`
  (ring-closed rig, the version the VO actually consumed), `imu.yaml`, `april_6x6.yaml`
  (the target as declared — note tasks 3R.6: the print scale was never verified with calipers,
  so a common-mode scale error in all of the above is possible and untested).
- `solve/` — every Kalibr/tartancalib run's `camchain`, `results-cam.txt` and `report-cam.pdf`.
  `f_cam1`, `s_cam2..4`, `solve_cam1_omni` are intrinsics; `p_pair_front|right|rear` are stereo;
  `cam_imu/` is the Δ fit and its shifted control.
  **No per-pair Kalibr report exists for the left pair** — it was solved by the repo's own
  `T_t_c` path rather than `kalibr_calibrate_cameras`; its numbers are in `rig_ext.yaml`.
- `evidence/` — virtual-stereo and disparity renders (fov 160 vs the wrong diagonal 190).
- `recording_metadata/` — each deleted bag's `metadata.yaml`: topics, message counts, duration.
  This is the index to what was recorded, kept after the recordings themselves were removed.
- `session_logs/`, `tools/` — the capture logs and the throwaway scripts used in the solve.
- `MANIFEST.sha256`, `DELETED_INVENTORY.txt`.

The in-force `camN.yaml` files derived from this session are in
`config/calib/imx296_1456x1088/` and in git history; task 3R.8 renames them with a
`superseded_by:` header rather than overwriting them.

## Deliberately not kept

- **Image bags** (`ros1/`, `ros1f/`, `CAM_*/`, 30 GB) — only useful for re-solving a
  calibration that is being replaced.
- **Kalibr `log.pkl` solver state** (5.6 GB) — derived; the reports and result files carry
  every number they contain.
- **`imu_stream.csv`** was the one exception: kept gzipped at
  `datasets/calib_20260828_raw_keep/imu_stream.csv.gz` (26 MB), untracked. It is the only raw
  stream small enough to be worth keeping, and it is the one that could re-fit Δ if the new
  camera↔IMU solve looks wrong.
