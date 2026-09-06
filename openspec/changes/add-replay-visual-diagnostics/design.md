## Context

See `proposal.md` — Why. The constraints that shape this, none of them negotiable:

- **The rig is 4 × ~192° fisheyes calibrated omni-radtan (Mei)**, `intrinsics = [xi, fx, fy, cx, cy]`,
  `distortion = radtan [k1, k2, p1, p2]`, 1456×1088. `cv2.omnidir` is **not present** on the host
  (needs opencv-contrib), so the projection is the hand-port in `scripts/vo/rerun_virtual_pinholes.py`,
  already verified to 5e-13 px against `cv2.omnidir.projectPoints` in `retarget-vo` 4.1b.
- **cuVSLAM cannot consume a fisheye.** It supports equidistant under 180°, so each physical camera is
  carved into two virtual pinholes at yaw ±45° (768×576, fov 70°, focal 548.409 px). There are
  therefore **8 camera panes for 4 cameras**, and the mapping between them is not intuitive.
- **The modules are mounted inverted** (180° roll) and the calibration is solved on the raw inverted
  frames, so nothing downstream un-rolls them. Display-only correction, behind `--upright`.
- **Pure VO**: no IMU in this path, no loop closure. Nothing pins roll and pitch to gravity.
- **`matplotlib`'s `Axes3D` is broken on this host**, so any 3D check is hand-rolled OpenCV projection
  or goes through Rerun.

## Decision 1 — Rerun as the surface, one recording, many panes

Rejected: writing PNG/MP4 contact sheets. The defects being hunted are *temporal and cross-view* — a
feature that tracks well in one carve and not its neighbour, a plane that is right at second 3 and
wrong at second 9. A static sheet forces a guess about which frame to look at.

Rerun gives one timeline shared by all panes, so a 3D trajectory, 8 feature overlays, 4 fisheyes, a
BEV and a panorama scrub together. The recording is a file (`.rrd`), so a finding is reproducible by
sending someone a path rather than a description.

**Consequence — a real bug this caused.** The virtual pinholes own `rig/cam0` … `rig/cam7`. Logging
the physical cameras at `rig/cam1` … `rig/cam4` **collides**, and Rerun silently lets the later log
win. The symptom was that "cam1 raw" and "cam4 raw" looked identical; they were in fact vpin[1] and
vpin[4], two carves 1.63° apart. Physical cameras therefore live at `rig/raw_camN`. The `VCAMS` array
order is cuVSLAM's own vcam index, referenced by the `z` field of every observation — **reorder the
blueprint for display, never the array**.

## Decision 2 — Frame convention: rig FLU, and the row-vector trap stated explicitly

The viewer works in rig **FLU** — `+x` forward, `+y` left, `+z` up — because that is what
`config/rig/ground_plane.yaml` declares and what the layout (cam1 front-left +45°, cam2 front-right
−45°, cam3 rear-left +135°, cam4 rear-right −135°) is expressed in.

The deployed `bev_panorama_node` / `stitch_kernel.cu` uses the **old rig's RFU** (`+x` right,
`+y` forward, `+z` up). These are deliberately different and neither should be "fixed" to match the
other; the old node is still IMX219-configured and out of scope here.

**The trap, written down because it cost real time.** `R_rig_cam1` maps a cam1-optical vector into rig
FLU as `v_rig = R @ v_cam1`. Point arrays here are **row-vector** `(N,3)`, so:

| direction | row-vector form |
|---|---|
| cam1 → rig | `P @ R.T` |
| rig → cam1 | `P @ R` |

Using `P @ R` for cam1 → rig silently applies the inverse. It does not crash and it does not produce
absurd numbers — it produced a trajectory with a plausible 3.5 m vertical span and a plausible 1.28 m
plane at 14.9° tilt, all of which were wrong and were retracted. `bev_maps` needs the *other*
direction, so `P @ R` is correct there; the two live a few lines apart.

**Free standing check:** `R @ [0,1,0]` must equal `[0,0,1]`. And a geometric one that needs no ground
truth at all — the BEV seams must land on forward / left / right / rear, the bisectors of ±45°/±135°.
If `R_rig_cam1` disagreed with `rig_in_cam1`, they would not.

## Decision 3 — Fit the ground plane near each pose, not once in the map frame

First implementation fitted one plane by RANSAC over all landmarks in the odometry frame, then
transformed it per frame. On a log where the rig was carried **at constant height on the operator's
head**, it reported the height swinging 0.03 → 1.56 m.

The assumption it violates is *global map consistency*. Pure VO drifts; landmarks mapped at second 3
and second 25 are in frames that have rotated relative to each other, so no single plane fits both,
and the fit ends up measuring drift. Transforming a wrong global plane through an accurate pose
cannot recover.

Replaced by `plane_near_pose`: take landmarks within 5 m of the **current** pose, express them in
**that pose's own rig frame**, and fit there. Same log: **1.36 m median, std 0.20 m**. Local beats
global whenever the map drifts, which is always for pure VO.

**Sub-decision — lowest dense bin, not largest plane.** RANSAC's "biggest consensus set" is not the
floor indoors: walls and furniture put landmarks at every height, and a wall is a bigger plane than
the visible patch of floor. The estimator histograms height into 56 bins over (−3.0, −0.2) m, takes
the **lowest bin exceeding 25 % of the peak**, refines by SVD within a 0.12 m band, and rejects the
result if the normal is more than ~14° off vertical. A failed fit carries the previous plane forward
rather than blanking the pane.

## Decision 4 — Cap the BEV incidence angle; never suppress a frame by skipping the log

A ground point at planar radius `r` seen from height `h` arrives at incidence `arctan(r/h)`. As
`h → 0` the whole grid is grazing, so a handful of source pixels smear across the entire output. The
first version skipped rendering when `h < 0.05 m`.

**That was a user-visible bug.** Rerun holds the previous image for an entity that is not logged this
frame, so 64 of 220 frames showed a stale BEV — reported from the outside as "BEV is kind of frozen
and stretched". The rule that came out of it: **never suppress bad output by skipping a log; log
blank instead.**

The real fix is geometric: `--bev-max-incidence` (default 75°) limits the painted radius to
`h · tan 75° = 3.73 h`. Measured coverage of a 4 m extent: 96.4 % at `h = 1.5 m` (identical with the
cap off, so it costs nothing at working height), 53.7 % at 0.9 m, 2.5 % at 0.2 m, 0 % at 0.02 m. The
view degrades honestly to nothing instead of degrading to a lie.

## Decision 5 — Panorama onto a finite sphere, radius chosen by a photometric-invariant metric

Rotation-only equirectangular stitching assumes every ray comes from infinity, which cancels camera
position. The measured baselines are **0.153–0.221 m** and the log is indoors at 2–4 m, so the
assumption is badly violated and the first render ghosted visibly — doubled doorframes, a doubled
wall medallion, a doubled head.

Rays are therefore cast onto a sphere of finite radius, `v = (depth · d_cam1 − t_c) @ R_c`, the same
construction as `scripts/calib/pano_tuner.py`. Choosing the radius took **three attempts, and the
first two produced confidently wrong answers**:

1. **Mean |difference| between overlapping cameras — useless here.** It ranked infinity best and got
   *worse* monotonically from 1 m to 4 m. Per-camera exposure differs by ~40 grey levels, which is
   far larger than the misregistration being measured, so the metric was reading photometry.
2. **Edge NCC, averaged over whichever pairs overlap — unfair.** Gain-invariant, so the trend
   inverted and became sensible. But the *set* of overlapping pairs shrinks as the radius drops
   (opposite cameras stop overlapping at all), so different radii were scored on different pairs, and
   the 18-vs-12 pair counts made the means incomparable.
3. **Edge NCC over a fixed pair set** — the four adjacent 90° pairs, scored identically at every
   radius. Clean peak at **3.0 m**: NCC 0.058 with all four pairs positive (0.082 / 0.043 / 0.046 /
   0.061), against **−0.007 at infinity**. Confirmed visually.

**Default is still infinity**, so the prototype matches `bev_panorama_node`'s behaviour unless a
radius is passed. **Residual ghosting is inherent** — one radius cannot fit a scene spanning many
depths, and absolute NCC stays low (~0.06). Photometric compensation is `add-bev-ground-stitch` §5
and is deliberately not done here.

**Trap.** Checking a camera's azimuth with `wt[c].sum(0).argmax()` is wrong: the feather weight is
clipped at 1.0, so there is a broad plateau and `argmax` returns an arbitrary point on it. It
reported all four cameras mismatched, which was a false alarm. Use a circular-mean centroid.

## Decision 6 — What this change refuses to claim

- **Not absolute scale.** Seam parallax and the landmark plane share the same extrinsic translations.
  Their 7 cm agreement is self-consistency. Scale needs an external ruler.
- **Not a calibrated ground plane.** `ground_plane.yaml` stays `status: unmeasured`. The VO-landmark
  height is an interim cross-check with a 0.20 m spread and a suspected systematic error; writing it
  into the calibration would launder an estimate into a measurement.
- **Not a verdict on capture.** Replay cannot separate a defect recorded at capture from one
  introduced in VO.
