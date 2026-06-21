# Surround-rig extrinsic calibration

How to recalibrate the 4-camera fisheye rig's relative orientations so the 360° panorama seams line
up. This refines the **rotations** in `config/rig/rig_extrinsics.yaml` (the idealized 90°/level values
are off by real mounting error — last run found **cam4/port-f ~19°** off).

Method: match natural scene features across each adjacent camera's overlap, unproject to bearing
rays via the KB intrinsics, solve the relative rotations (RANSAC + Kabsch), then jointly refine all
camera rotations with a robust Gauss-Newton. No checkerboard needed.

## When to do it (matters a lot)

The earlier failure was a single close-range indoor snapshot (under-constrained + parallax). Do it:

- **Good, even lighting** — features need contrast; avoid blown-out windows / dark corners.
- **Distant, textured scene** — aim the rig where most content is **> ~5 m** away (across a large
  room, down a hallway, out a door). The rig baseline is ~3 cm, so near objects (<~0.5 m) carry
  parallax the rotation-only model can't fit (and they will always ghost in the panorama anyway).
- **Texture in every direction** — no camera should stare at a blank wall during the capture.
- **Many orientations** — slowly pan/tilt the whole rig through the capture so each seam sees varied
  scenes. Diverse views over-determine the rotation (a locally-fitting wrong rotation can't satisfy
  all of them) and average out parallax.

## Procedure

### 1. Capture (on the TX2)
```bash
cd /media/nvidia/workspace/BEV_Jetson
./scripts/calib/capture_calib_sets.sh 10 3      # 10 sets, 3 s apart (~35 s)
```
While it runs (prints `>>> grabbing setNN`), **slowly pan/tilt the whole rig**. Writes raw 4-cam
frames to `scripts/calib/capture/setNN/camX_image_raw.png`.

### 2. Pull the sets to the dev box
```bash
cd ~/workspace/BEV/scripts/calib/capture
rm -rf set* && scp -qr tx2-eth:/media/nvidia/workspace/BEV_Jetson/scripts/calib/capture/'set*' .
```

### 3. Calibrate (on the dev box — needs python3 + opencv + numpy)
```bash
cd ~/workspace/BEV
python3 scripts/calib/extrinsic_calib.py --images scripts/calib/capture/set*
# tune the inlier gate if needed:  --ransac-deg 1.0   (0.5 strict ... 2.0 loose)
```
Outputs `config/rig/rig_extrinsics_calibrated.yaml` and a before/after
`scripts/calib/capture/pano_before_after.png` (top = nominal, bottom = calibrated).

**Read the output critically:**
- Per-pair `inliers` should be > ~10 and `resid` < ~1°.
- The per-camera `refined by` corrections should be **stable** if you re-run with different
  `--ransac-deg`. Stable ⇒ real mounting deviation; jumpy ⇒ under-constrained (recapture better).
- Joint `RMS angular residual` should drop well below the nominal pair deltas (we got 2.2° vs 9–19°).
- Confirm the **bottom** of `pano_before_after.png` looks better than the top before deploying.

### 4. Deploy + verify on the board
`ros2/bev_cuvslam/config/panorama_params.yaml` already points at
`config/rig/rig_extrinsics_calibrated.yaml`. Just commit/push the new yaml, pull on the TX2, and
capture a live panorama:
```bash
# dev box
git add config/rig/rig_extrinsics_calibrated.yaml && git commit && git push
# TX2
git pull --no-recurse-submodules
./scripts/capture_montage_tx2.sh /tmp/bev.png      # then scp /tmp/bev.png off to view
```

### Optional — manual fine-tune
```bash
python3 scripts/calib/pano_tuner.py        # open http://localhost:8000
```
Sliders for each camera's yaw/pitch/roll + translation + scene depth; live panorama (scroll=zoom,
drag=pan, dbl-click=reset); **Save** → `rig_extrinsics_tuned.yaml`. Good for a final nudge from the
calibrated baseline; impractical as a from-scratch tool (24 coupled DOF).

## Scripts

| script | runs on | purpose |
|---|---|---|
| [scripts/calib/capture_calib_sets.sh](../scripts/calib/capture_calib_sets.sh) | TX2 | grab N raw 4-cam sets while you pan the rig |
| [scripts/calib/extrinsic_calib.py](../scripts/calib/extrinsic_calib.py) | dev box | feature-based joint rotation calibration + before/after render |
| [scripts/calib/pano_tuner.py](../scripts/calib/pano_tuner.py) | dev box | interactive manual rot/trans tuner (web UI) |

## Caveats / still open

- **Near-field ghosting is not a calibration error** — it's the ~3 cm baseline parallax; a
  single-center equirect panorama can't stitch objects <~0.5 m regardless of extrinsics.
- **VO extrinsics not updated** — `rig_extrinsics.yaml` (used by fused/modular cuVSLAM) still has the
  idealized rotations and additionally needs the 180° upside-down roll folded in. The calibrated
  rotations are currently applied to the **panorama only**.
- **Exposure compensation** for seam brightness jumps is not implemented.
- The capture writes files **root-owned** (container runs as root); the capture script `sudo rm`s the
  old sets, and the pull step `rm -rf`s the local copies first.
