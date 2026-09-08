## Why

The existing `bev_panorama_node` stitches the four fisheyes **rotation-only onto a sphere at
infinity** — it reads `q_wxyz` per camera and never reads the translations at all. Camera position
therefore cancels out of the mapping, which is exactly why nearby objects show parallax seams: two
cameras 150 mm apart genuinely see a 0.5 m object from different directions, and a model with no
baseline in it cannot represent that. **The parallax is inherent to the method, not a calibration
fault**, so no amount of recalibration removes it.

A bird's-eye view is a different projection with a different guarantee: project each camera onto the
**ground plane** using the full extrinsics (rotation *and* translation) plus the rig height, and every
point that actually lies on that plane maps to one place regardless of which camera saw it. Ground
features — lane markings, floor texture, the pallet edge the rover is about to hit — stitch seamlessly.

Two other things force the work now rather than later. The IMX296 rig has just been recalibrated with
**omni-radtan (Mei)** intrinsics, because the ~192° lenses cannot be represented by Kannala-Brandt
(`pinhole-equi` diverged on every camera) — and the panorama node implements **KB only**, so it
cannot consume the current calibration at all. And every value in `panorama_params.yaml` still
describes the IMX219 rig: 1640×1232, `fisheye_fov_half_deg: 65`, IMX219 `sensor_ids`, and extrinsics
feature-calibrated on cameras that are no longer fitted.

## What Changes

- **New ROS 2 node `bev_ground_stitch`** producing a metric top-down mosaic on `/bev/ground`, with a
  stated scale (metres per pixel) and a stated origin, so the output is a *measurement*, not a picture.
- **Ground-plane projection using full extrinsics.** Each output cell is a point on the ground plane
  in the rig frame; it is transformed into each camera by `T_cam_rig` and projected through that
  camera's model. Points on the plane are parallax-free by construction.
- **Mei (omni-radtan) camera model** in the stitch path, so the node consumes the calibration the rig
  actually has. The projection is already written and verified to 5e-13 px against
  `cv2.omnidir.projectPoints` (retarget-vo change, task 4.1b) — this is a port, not a derivation.
- **A measured ground plane.** The rig's height and orientation relative to the ground are a new
  calibration product; nothing in the current rig files states them. Recorded with provenance like
  every other calibrated quantity.
- **Seam blending with an overlap weight**, and a **parallax residual metric** that measures how far
  the same ground feature lands from itself in two cameras — the number this change is judged by.
- **Optional depth-aware stage** using the existing virtual-stereo pairs, to reduce (not eliminate)
  the smearing of objects standing *above* the plane. Kept separable so the ground-plane result can
  ship on its own.
- The existing `surround-panorama` capability is **left as it is** — still equirectangular, still
  IMX219-configured. It answers a different question (what is around me, at distance) and is not on
  this change's path.

**Scope honesty, stated once here and enforced in the specs:** a single-surface projection is
parallax-free *on that surface only*. A pole standing on the ground will still smear, because its top
is not on the plane and no ground-plane homography can place it correctly. "Eliminate parallax" is
therefore specified as **parallax-free for points on the calibrated ground plane, with a measured
residual**, and above-plane behaviour is characterised rather than claimed away.

## Capabilities

### New Capabilities

- `bev-ground-stitch`: metric top-down ground-plane mosaic from the four fisheyes, using full
  extrinsics and the Mei camera model, with a measured parallax residual on the ground plane.
- `ground-plane-calibration`: the rig's pose relative to the ground — height and orientation —
  measured, stated with provenance, and rejectable when stale.

### Modified Capabilities

<!-- none: surround-panorama keeps its current requirements and is out of scope here -->

## Impact

- **New**: `ros2/bev_ground/` (node + CUDA remap), `config/rig/ground_plane.yaml`,
  `scripts/calib/ground_plane_calib.py`, `scripts/viz/bev_view.sh`.
- **Reused unchanged**: the Mei projection and remap-table machinery from `bev_cuvslam`, the
  ring-closed extrinsics from the retarget-vo change, `argus_capture_node` and its synchronised sets.
- **Depends on**: `retarget-vo-to-imx296-rig` tasks 3R.11 (intrinsics), 3R.12/3R.13 (pairwise
  extrinsics + ring closure). The stitch needs one rigid set of camera poses; four independently
  solved pairs would put the seams in four inconsistent places.
- **Not affected**: `bev_panorama_node`, `cuvslam_multicam_node`, the capture path, the trigger.
- **Runs in** the existing `cuvslam-foxy:tx2` image — no new container.
- **Known limitation, by construction**: objects above the ground plane smear. The optional depth
  stage reduces it where stereo depth is valid; it does not remove it.
