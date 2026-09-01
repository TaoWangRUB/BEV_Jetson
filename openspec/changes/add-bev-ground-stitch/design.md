## Context

See `proposal.md` — Why. The constraints that shape the approach, none of which are negotiable:

- **The lenses are ~192°, calibrated omni-radtan (Mei).** `pinhole-equi` (Kannala-Brandt) diverged on
  every camera, so KB is not an option to fall back to. A Mei camera is **not a projective map**: it
  unprojects through a unit sphere offset by `xi`. This single fact rules out the textbook BEV recipe
  (see Decision 2).
- **The extrinsics are one ring-closed set**, from `retarget-vo-to-imx296-rig` 3R.12/3R.13, in
  `config/rig/rig_extrinsics_imx296.yaml`. Not `rig_extrinsics_vo.yaml` — that file has the 180°
  mounting roll folded in for the VO path, and folding it in twice would silently rotate the mosaic.
- **The cameras are mounted upside-down**; raw frames are inverted. That is a property of the image,
  handled once at ingest, not a term in the geometry.
- **TX2 (L4T R32.7.6, CUDA 10.2), four 1456×1088 streams at 15 Hz**, sharing the board with capture.
  The compute budget is the reason for Decision 1.
- **The output frame is FLU** (x forward, y left, z up), sitting *on* the plane, with row 0 the
  most-forward row and column 0 the leftmost. Stated here because handedness and row order are the
  errors that still produce a picture of a floor (see task 2.6).
- **Nothing states where the ground is.** No rig file records height or tilt. That is a new measured
  product, not a constant to assume — the IMX219 rig had a camera ~19° off its nominal mount.

## Goals / Non-Goals

**Goals:**

- One static, precomputed mapping per camera from ground cell to source pixel, so the per-frame cost
  is a remap and a weighted sum.
- A ground plane obtained by the same detector/solver already used for calibration, so it carries
  provenance of the same kind as every other calibrated quantity here.
- A parallax residual that is *measured on real overlaps*, so the proposal's claim is falsifiable.

**Non-Goals:**

- Any change to `bev_panorama_node`. It stays equirectangular and IMX219-configured.
- A general multi-surface or mesh-based stitch. One plane, stated, with above-plane behaviour
  characterised (see spec `bev-ground-stitch`, "Above-plane objects are characterised").
- Fusing the four virtual-stereo pairs into a single omnidirectional depth. The optional depth stage
  overrides plane depth *per camera, in that camera's own pair*; it does not build a unified cloud.

## Decisions

### 1. Precompute a remap LUT per camera; do not project per frame

The mapping from output cell to source pixel depends only on the extrinsics, the intrinsics, the
plane, and the output grid — all fixed at startup. Build four `CV_32FC2` maps once and run
`cv::remap` per frame, on the CPU.

**Not** `cv::cuda::remap`, despite `bev_cuvslam` using it. That node captures via NVMM and never
touches host memory, so its remap is free of transfers. This node consumes ROS `Image` messages that
are *already in host memory*: a CUDA path would upload four 1456×1088 frames, remap, and download the
mosaic every cycle, and the transfers cost more than the arithmetic they save.

*Alternative — project every frame:* correct and simpler to read, but it repeats a Mei unprojection
and a distortion evaluation per output pixel per camera per frame. On a TX2 already running capture
that is the difference between fitting in the budget and not.

*Cost:* changing the plane or the grid means rebuilding the tables. That is a startup-time or
parameter-callback operation, and is stated as such rather than hidden.

### 2. Map through the full Mei model — **not** an inverse-perspective homography

The standard IPM recipe collapses ground-plane BEV to one 3×3 homography per camera. It is valid only
for a pinhole camera, because only then is the plane-to-image map projective. With a Mei model the
map is not projective, and a homography fit to it is wrong everywhere except where it was fit —
worst at the image periphery, which for a 192° lens is most of the useful ground.

So each output cell `(u,v)` becomes a rig-frame point `X` on the plane, is transformed by `T_cam_rig`,
and is projected by that camera's Mei model. The projection is already written and verified to 5e-13 px over
4000 rays against `cv2.omnidir.projectPoints` (retarget-vo task 4.1b) — this is a port, not a
derivation.

*It is copied, not shared.* `bev_cuvslam`'s CMake hard-`REQUIRE`s libcuvslam, and making the stitcher
unbuildable without a VO library is the worse coupling. The cost is a maintenance obligation: the two
copies must stay identical, and a check that they agree is a task, not a hope.

*Direction matters:* the LUT is built **backwards**, output → source, and sampled with `remap`.
Forward-warping source pixels into the output leaves holes wherever the ground is undersampled, which
near the far edge of a fisheye is everywhere.

### 3. The plane is a full `T_ground_rig`, not a height

Store a 4×4 transform in `config/rig/ground_plane.yaml`, with height and roll/pitch/yaw derived
alongside for a human to sanity-check. The rotation fixes not only *how far away* the plane is but
*which way the BEV x-axis points* — with only a height, the mosaic's orientation is unpinned and
"forward" in the output means nothing.

*Alternative — assume the rig is level and store height only:* one number, easy to measure, and wrong
in exactly the way the IMX219 rig was wrong. A 2° unmodelled tilt puts a point at 3 m range ~10 cm off.

### 4. Measure the plane from an AprilGrid lying on the floor

Lay the board flat, solve its pose in cam1 with the detector and solver already in the calibration
path, and take the board's plane as the ground plane. It gives all six degrees of freedom at once,
including the yaw that pins the BEV axes, and it is directly checkable — a known length on that board
must measure correctly in the output (spec `ground-plane-calibration`, "A known length measures
correctly").

*Cross-check, not primary:* IMU gravity gives roll and pitch independently. Agreement between the two
is the "disagreement is detectable" scenario; disagreement means one of them is wrong and the size of
the difference says how much.

*Rejected as primary — fit a plane to virtual-stereo points on the floor:* floors are texture-poor,
which is precisely what the round-2 disparity figure shows (10–17 % valid pixels on a scene chosen to
be easy). Fitting the reference surface to the least reliable measurement available inverts the
dependency.

### 5. Blend by normalised angular weight, computed into the same LUT stage

Each camera gets a per-cell weight that falls off toward the edge of its valid region; weights are
normalised across cameras per cell, so overlaps cross-fade and the sum is always 1 where any camera
sees. Weights are static, so they are built once beside the maps.

*Alternative — multi-band/Laplacian blending:* it exists to hide *misregistration*. On the plane there
is no misregistration to hide, by construction — what remains at a seam is photometric (independent AE
per camera), which multiband would smear rather than fix. Decision 6 addresses the actual cause, and
multiband costs TX2 time to do it worse.

### 6. Photometric gain compensation is separable and comes second

Four cameras with independent auto-exposure produce brightness steps at seams that look exactly like
stitching failures and are not. Estimate a per-camera gain on the overlap regions and apply it before
blending. Kept as its own task so a brightness step never gets mistaken for a geometric defect, and so
the geometric result can be judged before it is cosmetically improved.

### 7. Publish the metric contract as a latched YAML string, not `nav_msgs/MapMetaData`

`bev/ground` carries the image; `bev/ground/info` carries a latched `std_msgs/String` holding YAML:
frame and parent frame, resolution, the four per-side ranges, the plane status and normal, the literal
pixel→metres formula, and the name of the source-mask topic.

`MapMetaData` was the first choice — no custom package, semantics every ROS user knows. It was rejected
because it **cannot say the two things this output most needs to say**: that the plane is PROVISIONAL
(Decision 9), and what the plane normal is. An origin and a resolution presented in a standard message
read as authoritative, which is exactly wrong when the plane underneath them is a guess. The cost is
losing rviz's free map rendering; a string that states its own uncertainty is worth more here than a
typed message that cannot.

A companion `bev/ground/source` mask publishes which camera is dominant per cell (0 = no coverage),
so any seam in the mosaic can be attributed to a camera pair without re-deriving the weights.

### 8. An unmeasured plane blocks startup; a provisional one taints every output

`config/rig/ground_plane.yaml` ships `status: unmeasured`, and the node **refuses to start** against it.
There is no default height, because a plausible mosaic built on a guessed height is the failure mode
this whole change exists to avoid — it looks exactly like a correct one.

For look-only work before task 1.3 lands, the operator must set *two* parameters, not one:
`allow_unmeasured_plane:=true` **and** `provisional_height_m:=<metres>`. Requiring the height to be
typed in explicitly means no one gets a provisional plane by accident. Everything published then carries
`plane_status: PROVISIONAL`, in the info topic and in the log line, so a screenshot taken in this mode
cannot later be mistaken for a measurement.

### 9. Refuse to run on a calibration that does not match the rig

The calibration files carry a fingerprint — port→camera assignment plus calibration id/date. The node
compares it at startup and exits with a message naming the mismatch rather than producing a plausible,
wrong mosaic. Same for a ground plane whose recorded rig fingerprint differs (spec
`ground-plane-calibration`, "Remounting invalidates it").

## Risks / Trade-offs

- **cam3's intrinsics are currently the weakest of the four** (subset stability 34.4 px vs cam1's 0.06;
  both its baselines inflated; the two worst epipolar residuals) → its two seams will be the worst in
  the mosaic. *Mitigation:* the cam3 re-sweep lands before configs are written (retarget-vo 3R.16); the
  parallax residual is reported **per seam**, so a bad camera is localised rather than averaged away.
- **Ground-plane error scales with range** — a small tilt error is invisible at 0.5 m and metres wrong
  at the edge. *Mitigation:* validate a known length at two ranges, not one, and publish the residual
  as a function of range rather than a single number.
- **Directly under the rig is seen by no camera**; near the rig, cells are seen by three. *Mitigation:*
  publish a validity mask and leave the blind cone unpainted. An interpolated blind spot is a lie in
  the middle of a measurement product.
- **Mei unprojection is numerically delicate near the 180° limit** (xi near 1, incidence near grazing).
  *Mitigation:* build the tables in double precision at startup and mark cells whose reprojection does
  not round-trip within a pixel as invalid, rather than letting them fold back into the image.
- **Above-plane objects smear, always.** *Mitigation:* this is stated in the proposal, specified as a
  characterised behaviour rather than a fixed one, and is the entire reason the depth stage exists as
  an optional reducer. It is not presented as solved.
- **The depth stage depends on stereo that is only 10–17 % valid on the scenes recorded so far.**
  *Mitigation:* it stays behind a flag and off by default; the ground-plane result must stand alone.

## Migration Plan

Purely additive — a new node, new config, new topics. Nothing existing changes behaviour.

1. Land the plane calibration and its config first; it is the only new *measured* input.
2. Ship the geometric stitch with no gain compensation and no depth stage, and publish its parallax
   residual. That number is the change's verdict.
3. Add gain compensation, then the depth stage behind a flag.

*Rollback:* stop the node. `bev_panorama_node`, the capture path, the trigger, and the VO node are
untouched by every step.

## Open Questions

- **Default output extent and resolution.** 8 m × 8 m at 2 cm/px is the working assumption; the useful
  range is bounded by where the parallax residual grows past the tolerance, which step 2 measures. The
  defaults change no spec, no interface, and no task.
- **Gain-compensation model** — a single scalar per camera, or a scalar plus a slow vignette term. The
  overlap statistics from step 2 decide it; both fit the same task.
