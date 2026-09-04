# Retired IMX219 rig extrinsics

These belong to the **previous 4×IMX219 rig**, not to the IMX296 rig now fitted. Kept for
provenance only — nothing in the tree reads them.

| File | Was |
|---|---|
| `rig_extrinsics_calibrated.yaml` | panorama lineage; rotations from `scripts/calib/extrinsic_calib.py` |
| `rig_extrinsics_vo.yaml` | the above with the 180° roll folded in (`fold_roll_for_vo.py`) |
| `rig_extrinsics.yaml` | earlier nominal layout |

**Their translations are nominal.** `t_xyz_m` is ±15 mm, giving ~21 mm adjacent baselines.
The IMX296 rig's *measured* adjacent baseline is **155.6 mm** (`rig_extrinsics_imx296.yaml`,
ring-closed 2026-09-01, √2 × 155.6 = 220.1 mm against a 220.0 mm measured diagonal). Do not
carry these numbers forward — they understate the real parallax by ~7×.

The matching IMX219 *intrinsics* (`scripts/config/{1640x1232,calib,832x624,1280x720,640x360}`,
KANNALA_BRANDT) were deleted on 2026-09-04. The current rig is calibrated in omni/Mei at
`config/calib/imx296_1456x1088`; the equidistant model cannot represent these >180° lenses.
