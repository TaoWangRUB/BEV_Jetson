## 1. The viewer

- [x] 1.1 **`scripts/vo/rerun_multicam.py` — replay a recorded log into one Rerun timeline.**
  Reads the odometry + observations bag (`datasets/replay_out/obs_*`) and the raw image log
  (`datasets/imglog_*_bag`), and logs: the 8 virtual pinholes with their tracked features, the 4 raw
  fisheyes, the landmark map, and the trajectory. Reuses the verified Mei projection from
  `rerun_virtual_pinholes.py` and the bag readers from `rerun_odometry.py` — no second implementation.

- [x] 1.2 **Entity naming — fixed a silent view collision.** Virtual pinholes own `rig/cam0..cam7`;
  logging the physical cameras at `rig/cam1..cam4` collided and Rerun let the later log win. Symptom:
  "cam1 raw" and "cam4 raw" appeared identical — they were really vpin[1] and vpin[4], two carves
  1.63° apart. Physical cameras now live at `rig/raw_camN`. `VCAMS` order is cuVSLAM's own vcam index
  (referenced by each observation's `z` field); the blueprint is reordered for display, the array is not.

- [x] 1.3 **Fisheye domes** so a ~192° lens is viewable in 3D at all. `rr.Pinhole` cannot express a
  field beyond 180°, so each fisheye is textured onto a spherical cap mesh (11 180 triangles/camera,
  `--dome-radius`, `--tex-scale`, `--dome-stride`). Rerun's `Mesh3D` albedo must be **RGB** — a
  greyscale texture is rejected with "Bad albedo texture shape".

- [x] 1.4 **Inverted mounting handled at display only** (`--upright`), since the calibration is solved
  on the raw inverted frames and nothing downstream un-rolls them.

- [x] 1.5 **Viewer documented in `scripts/README.md`** — the four `vo/rerun_*` / `render_*` rows
  plus a "Running the Rerun viewer" section: the venv (rerun-sdk is not installable into the
  system python here, PEP 668 — `python3 -m venv --system-site-packages .venv` keeps rosbags,
  cv2 and numpy from the system), the two-bag invocation, and the `--serve` port convention
  (`--port` alone still binds 9090 for the UI, so concurrent servers need `--port` *and*
  `--web-viewer-port`).

  Two things had to be fixed before the documented invocation actually ran:

  - **`--t-range START:END`** on `rerun_multicam.py`. `--frames` subsampled the *whole* run, so
    on a 57 s log the default 180 frames is one pose in five — enough to miss the 0.75 s
    tracking freeze in `retarget-vo-to-imx296-rig` 5.0g completely. `--frames` now subsamples
    the selected window instead.
  - **`find_bag()` no longer trips over our own bag naming.** rosbags picks ROS 1 vs ROS 2 from
    the SUFFIX (`any(x.suffix != '.bag' ...)` in `highlevel/anyreader.py`), and README §3.3 tells
    people to write ROS 2 bags as `raw_log_to_bag.py -o /tmp/run1.bag` — a *directory* named
    `.bag`. Every viewer therefore got the rosbag1 reader and died with "Could not open file
    ...: Is a directory". It now hands back a suffix-free symlink; nothing is renamed.

  `replay_host.sh` gained `OBS=1` to publish and record `/cuvslam/landmarks` +
  `/cuvslam/observations` (naming the output `obs_*`), which is what the viewer needs and what
  the default rate-measuring runs must NOT carry.

- [x] 1.6 **Pane order walks the ring.** The two rows now run in opposite carve order —
  row 1 `cam1 +45, cam1 -45, cam2 +45, cam2 -45`, row 2 `cam3 -45, cam3 +45, cam4 -45, cam4 +45`
  — so reading each row left-to-right continues around the rig instead of jumping back across it
  at the row break. Display order only; `VCAMS` is cuVSLAM's own indexing and is what each
  observation's vcam field refers to, so the array is untouched.

- [ ] 1.7 **Loop closure: cuVSLAM has it, this node does not use it.** The viewer cannot show
  loop closures because nothing produces any — `cuvslam_multicam_node` builds a
  `cuvslam::Odometry`, which is pure VO with no pose graph. The library also ships
  `cuvslam::Slam` (`cuvslam2.h:885`), constructed as `Slam(rig, primary_cameras, config)` — so it
  takes a multicamera rig — with `throttling_time_ms` (minimum interval between loop-closure
  events), `max_map_size` (pose-graph cap, 300 for real-time), `enable_reading_internals` for the
  pose graph and loop closures, a `lc_status` field, a `LoopClosure` landmark layer, and
  `GetAllSlamPoses()`.

  This is the structural fix for the drift in 5.1/5.2, not just a visualisation: pure VO has
  nothing to pin the trajectory. Order of work:
  - [ ] 1.7a Stand up `cuvslam::Slam` alongside the existing `Odometry` in the node, behind a
        parameter, and publish the SLAM pose separately from the odometry pose. Do not silently
        replace `/cuvslam/odometry` — the §5 drift numbers are measured against pure VO.
  - [ ] 1.7b Publish loop-closure events and the pose graph; add a viewer pane keyed on
        `lc_status`, plus a corrected-trajectory line beside the raw VO one.
  - [ ] 1.7c Measure the cost on the TX2 before believing any of it. `Track()` there is already
        50-90 ms/set against a 20 Hz budget (`retarget-vo-to-imx296-rig` 5.10); SLAM runs its own
        thread unless `sync_mode`, but the board has no headroom to give away.
  - [ ] 1.7d Re-run 5.1/5.2 with loop closure on and state the drift both ways.

## 2. BEV prototype — satisfies `add-bev-ground-stitch` 2.6

- [x] 2.1 **`bev_maps()`: the ground-plane projection the node will implement, in Python.** Output cell
  → point on the plane in rig FLU → `T_cam_rig` → Mei projection → source pixel, per camera, with
  overlap weights. This is the offline mosaic check 2.6 asks for, and it did what 2.6 predicts: a
  wrong frame transform and a stale-image bug were both obvious here and would have been subtle in CUDA.

- [x] 2.2 **Supports a tilted plane, not just a level one.** Grid basis is built from the plane normal
  (`e1` = forward projected onto the plane, `e2 = n × e1`). **Verified**: with `normal = [0,0,1]` it
  reproduces the previous level-plane tables bit-exactly, so tilt support did not perturb the level case.

- [x] 2.3 **Seam sanity check that needs no ground truth**: seams land on forward/left/right/rear, the
  bisectors of ±45°/±135°, confirming `R_rig_cam1` agrees with `rig_in_cam1`. Free, and it is the check
  `add-bev-ground-stitch` 0.2 wants for "wrong file here rotates the entire mosaic".

- [x] 2.4 **Incidence cap instead of frame skipping** (commit `581cc64`). See design Decision 4. A
  ground point at radius `r` is seen at incidence `arctan(r/h)`, so `--bev-max-incidence` (75°) caps
  the painted radius at `3.73 h`. Coverage of a 4 m extent: **96.4 % at h = 1.5 m — identical with the
  cap off, so it is free at working height** — 53.7 % at 0.9 m, 2.5 % at 0.2 m, 0 % at 0.02 m.

- [x] 2.5 **Fixed a user-visible "frozen and stretched" BEV.** Both symptoms were `h → 0`. Frozen: an
  `h < 0.05` skip left the previous frame on screen for **64 of 220 frames**, because Rerun holds the
  last logged image for an entity. Stretched: at h = 0.05–0.15 m the camera is nearly *in* the plane
  and the grid projects to a grazing sliver. **Rule adopted: never suppress bad output by skipping a
  log — log blank.**

- [ ] 2.6 Cross-check the Python mosaic against the CUDA implementation once `ros2/bev_ground/` exists,
  cell for cell on one recorded set. The prototype's value is as a reference; that requires diffing it.

- [ ] 2.7 Characterise above-plane smearing quantitatively (`add-bev-ground-stitch` 4.3). The prototype
  can measure it — an object of known height, and how far its top lands from its base — and no such
  number exists yet.

## 3. Ground plane from VO landmarks — interim, feeds `add-bev-ground-stitch` §1

- [x] 3.1 **Discarded the global fit; it was measuring map drift** (superseded by `a13dc77`). A single
  RANSAC plane fitted over all landmarks in the **odometry frame** reported the height swinging
  **0.03 → 1.56 m** on a walk where the operator states the rig was carried at **constant height on
  their head**. The assumption it breaks is global map consistency: pure VO drifts, so landmarks from
  second 3 and second 25 are in mutually rotated frames and no single plane fits both.

- [x] 3.2 **`plane_near_pose()` — fit locally, in the pose's own frame** (commit `a13dc77`). Landmarks
  within 5 m of the current pose, expressed in that pose's rig frame; height histogrammed into 56 bins
  over (−3.0, −0.2) m; floor taken as the **lowest bin exceeding 25 % of the peak** — not the largest
  consensus set, because indoors a wall is a bigger plane than the visible floor — refined by SVD
  within a 0.12 m band, rejected if the normal is >14° off vertical.
  **Result on the same walk: 1.36 m median, std 0.20 m, range 1.04–1.99 m**, tilt 0.1–0.7° (against
  the global fit's 0.03–1.56 m and 2.95°). 191 distinct tables over 220 frames.

- [x] 3.3 **Cross-check at matched frames**: a seam-NCC sweep peaks at h ≈ 0.90 m; landmark RANSAC
  gives 0.97 m at the same frames — **agreement to 7 cm**. Recorded as a cross-check only: both derive
  from the same extrinsic translations, so this is self-consistency, **not** evidence about metres.

- [x] 3.4 **`ground_plane.yaml` deliberately left `status: unmeasured`, `height_m: null`.** Writing a
  0.20 m-spread estimate with a suspected systematic error into the calibration would launder an
  estimate into a measurement. Tasks `add-bev-ground-stitch` 1.1–1.5 (AprilGrid on the floor) still stand.

- [ ] 3.5 **Resolve the suspected 15–20 % scale underestimate.** The local floor sits **1.36 m** below
  the camera; a rig carried on an adult head should be ~1.60–1.75 m. If real, the error is in the
  extrinsic translations in `config/rig/rig_extrinsics_imx296.yaml`. **Blocked on one input: the
  measured height of the rig when worn.** This is the cheapest scale check available and it has been
  asked for and not yet answered.

## 4. Panorama prototype (commit `97877fa`)

- [x] 4.1 **`pano_maps()` / `render_pano()`: equirectangular stitch of the four raw fisheyes** on the
  *current* rig, which `bev_panorama_node` cannot serve (KB-only, IMX219-configured). 1280×356,
  azimuth 0 = rig forward, +azimuth toward left, ±50° elevation, feathered overlap weights.
  **99 % of the band covered, 85 % of it by 2+ cameras.**

- [x] 4.2 **Bearings verified against the extrinsics, not eyeballed.** Optical-axis azimuths
  **+45.0 / −43.8 / +134.4 / −134.8°**; blend-weight circular-mean centroids
  **+44.6 / −44.0 / +133.7 / −134.6°**. Both match the declared ±45 / ±135 layout.
  **Trap recorded**: the first check used `wt[c].sum(0).argmax()` and reported all four cameras
  mismatched. The feather weight is clipped at 1.0, so there is a broad plateau with no unique maximum
  and `argmax` returns an arbitrary point on it. False alarm; use a circular mean.

- [x] 4.3 **Finite-radius sphere to compensate the baselines.** Measured baselines are
  **0.153 / 0.155 / 0.219 / 0.221 / 0.153 / 0.161 m** and the scene is 2–4 m away, so rotation-only
  stitching ghosted visibly. Rays are cast as `v = (depth·d_cam1 − t_c) @ R_c` (construction borrowed
  from `scripts/calib/pano_tuner.py`). A later `auto` mode tracks the landmark cloud per frame at a
  low percentile (p25 — ghost displacement goes as baseline/depth, so erring near costs far less than
  erring far); measured 2.01–3.87 m, median 2.45 m over the walk.
  **Real but second-order** — see the correction in 4.5. This helps; the blend geometry mattered more.

- [x] 4.4 **Radius selected by measurement — after two metrics gave confidently wrong answers.**
  See design Decision 5. (1) Mean |diff| between overlapping cameras ranked **infinity best** because
  per-camera exposure differs by ~40 grey levels and swamped the geometry. (2) Edge NCC averaged over
  "whichever pairs overlap" was unfair, because the overlap set shrinks with radius (18 vs 12 pairs),
  so radii were scored on different content. (3) Edge NCC over a **fixed** set of the four adjacent
  90° pairs peaks cleanly at **3.0 m: NCC 0.058, all four pairs positive (0.082/0.043/0.046/0.061),
  against −0.007 at infinity.** Confirmed visually against `/tmp/pano_compare.png`.

- [x] 4.5 **CORRECTION — most of the ghosting was the blend geometry, not the method.** This task
  previously read "residual ghosting recorded as method-inherent, not tuned away". That was wrong,
  and the sphere-radius work in 4.3/4.4 was treating a symptom.

  **Root cause.** Weights feathered on absolute angle from each optical axis,
  `w = clip((fov_half − th)/feather, 0, 1)`, which is 1.0 for all `th < 65°`. Adjacent cameras are
  **90° apart**, so at the bisector both sit 45° off axis and **both get weight exactly 1.0**. The
  output was a literal 50/50 double exposure over a ~60° band at each of the four seams.
  **Measured: 65 % of the horizon, 39 % of the whole canvas.** No sphere radius can fix this — the
  radius only makes two views agree at *its* radius, and everything nearer or further doubles at
  full strength across that band.

  **Fix** (commit `22144d5`): rank cameras by angular distance and cross-fade only over
  `--pano-seam` (8°) where two are nearly equidistant, so each pixel comes from the camera looking
  most directly at it. Ranking against the best **valid** camera keeps weight 1.0 where only one
  camera sees, so coverage stays 100 % with no normalisation dip.

  **Detail retained in the overlap region: 0.87 → 1.85** relative to a single camera; equal-blend
  area 39 % → 4 %. Above 1.0 because nearest-axis also selects the *sharper* view — a ray far off
  axis lands in the fisheye's compressed periphery. **For scale, the per-frame auto radius alone
  moved the same number 0.87 → 0.96.** The blend geometry was worth roughly ten times the radius.

  What remains genuinely method-inherent is much smaller: a narrow mis-registered band at each seam
  for scene off the sphere radius, plus the photometric step (4.6).

- [x] 4.8 **The deployed `bev_panorama_node` has the same defect.** `bev_panorama_node.cpp:295`
  computes `fw = (fov_max - th)/feather` — the identical construction. With its shipped parameters
  (`fisheye_fov_half_deg: 80`, `feather_deg: 20`) weight is 1.0 for `th < 60°`, giving a **30°-wide
  50/50 band at each seam, ~33 % of the horizon**. Milder than the viewer's was, same mechanism.
  This matters now because the node has just been ported to Mei/IMX296 (`18319fb`), so it will ghost
  on the TX2 for the same reason. **Not fixed here** — the node is another change's surface.

- [ ] 4.9 Port the nearest-axis seam into `bev_panorama_node`. It is a change to the host-side table
  build only (the CUDA kernel already just weight-blends precomputed maps), so the kernel is untouched.
  Needs a second pass over the cameras to rank them, which the current single-pass loop does not do.

- [ ] 4.6 **Quantify the per-camera brightness step at each seam** — feeds `add-bev-ground-stitch` 5.1,
  which wants exactly this number so "a photometric defect is never later mistaken for a geometric one".
  The mismatch is plainly visible in the render and the ~40 grey-level figure above is an aggregate,
  not a per-seam measurement. Photometric compensation itself stays out of scope (`add-bev-ground-stitch` §5).

- [ ] 4.7 Decide whether the panorama prototype should be retargeted into `bev_panorama_node` (Mei model,
  IMX296 extrinsics, finite depth) or left as a diagnostic. `add-bev-ground-stitch`'s proposal
  explicitly leaves `surround-panorama` alone; this prototype changes the cost of revisiting that.

## 5. Findings that belong to other changes

- [x] 5.1 **Vertical drift is pure-VO drift, and its root cause is structural.** No IMU and no loop
  closure means nothing pins roll and pitch to gravity, so forward motion leaks into apparent vertical
  motion. `bev_imu` + MPU9250 already exist in the repo and are the structural fix. Recorded against
  `retarget-vo-to-imx296-rig` 5.2 (drift).

- [x] 5.2 **But not all of the vertical motion was drift — the operator was right and the fit was
  wrong.** A local floor measurement along each pose's **own** up axis (immune to odometry tilt) puts
  the floor 0.22 m below the camera for t < 6 s and ~1.27 m from t = 7.4 s. The t = 6–9 s segment has
  **dz = +1.22 m with only 0.30 m of horizontal motion**, which drift cannot produce: that is the rig
  being lifted onto the head. Drift is the ±0.4 m wander *after* it. **Operator ground truth beat the
  fit twice in this session; treat it as the reference.**

- [x] 5.3 **One virtual camera is effectively blind in this log.** vcam3 (cam2's +45° carve) yields
  **~12 features/frame against 2402/frame overall** — a person's head occludes it. Relevant to
  `retarget-vo-to-imx296-rig` §5: the run's tracking quality is not evenly sourced, and a repeat should
  keep the operator out of that carve.

- [x] 5.4 **The map has long-range outliers.** `obs_20260903_140714`: 440 odometry poses, 440
  observation sets, 26 759 landmarks, of which **24 234 within 20 m — the map reaches 310 m in a 14 m
  indoor log**. Rerun's 3D auto-framing collapses the room to a few pixels without `--map-radius`.

- [ ] 5.5 **Absolute scale is still unvalidated** — `retarget-vo-to-imx296-rig` 5.1 and
  `add-bev-ground-stitch` 4.1 both need it. Scale enters **only** through the extrinsic translations,
  so the fix on failure is the extrinsics; never post-scale the VO output. In order of value:
  - [ ] 5.5a Tape-measure a 5–10 m straight run and compare reported translation. Free, and the best
        anchor available. Nothing else should be built before this is done.
  - [ ] 5.5b ToF lidar: regress measured range against landmark range over **several** distances —
        slope is the scale error, intercept is a bias. One distance cannot separate the two.

        **Correction (2026-09-06): the channel is not useless, it is intermittently blocked.**
        5.0g called the beam "pointed at the rig or the operator" from its 0.30 m median and
        wrote it off. The operator's account is that a hand covers it part of the time, and
        segmenting on that is what the data supports: readings that are **both** under 0.5 m
        **and** near-constant (rolling std < 5 cm) are **68.6 %** of the run; the other
        **31.4 %** is scene, in three stretches over 1 s — t = 2.7-5.1, **t = 12.6-22.0**
        (9.4 s, 187 readings, 0.04-2.03 m) and t = 53.0-57.7. The middle one sits inside the
        well-exposed room, which makes it the only usable window.

        **But it still cannot anchor scale as recorded.** Inside that stretch |d(range)| per
        50 ms sample is a median 2 cm but p90 24 cm and max 158 cm, implying beam speeds to
        31.6 m/s: the beam is sweeping across objects, not tracking one surface. A regression
        needs the range and the VO to be measuring the same thing.
      - [ ] 5.5e **What a usable range run looks like.** Deliberately: keep the hand off the
            sensor, aim it at one flat wall, and translate along the beam through several
            distances (say 3 m to 0.5 m) with the wall filling the beam throughout. Then
            d(range) is directly comparable to the VO translation projected on the beam, and
            the regression slope is the scale error. Needs the rangefinder extrinsic, or at
            least its axis, which is still unmeasured — with 3+ distances the axis can be
            solved for alongside the scale.
  - [ ] 5.5c T265: the weaker reference. Compare **pairwise pose distances > 2 m apart** so the ~0.2 m
        lever arm between the devices is negligible, rather than aligning trajectories directly.
  - [ ] 5.5d Write the analysis scripts once a bag containing either sensor exists. Offered, not started.

- [x] 5.7 **The rig logger's range channel never worked, and the first run of it found four
  separate faults.** `bev_range` + the `rangelog` service were committed 2026-09-04 (`a870470`)
  and deployed, but had never been run once — `/home/nvidia/logs` was empty. Running it produced
  cameras and IMU with a `range0.csv` containing a correct provenance header and **zero data rows**,
  which is the worst possible failure shape: it reads as "the sensor said nothing" rather than as a
  bug. The sensor was fine throughout — `lidar_present=1`, streaming `!range_cm=N pulses=M` every
  15 pulses exactly.

  1. **The container never started.** `log_rig.sh` launched `rangelog` without `EXPOSURE_US`, and
     `docker compose run <one service>` interpolates the WHOLE file at parse time, so the `capture`
     service's `${EXPOSURE_US:?...}` aborted the command. `imulog` had always passed it. The
     failure was invisible because the call ended in `>/dev/null 2>&1 || true`, and the fallback
     diagnostic asked `docker logs` for a container a failed `run` never created. Same trap as
     `retarget-vo-to-imx296-rig` 5.0c hit with the `shell` service.
  2. **SIGINT never reached the node.** `rangelog` wraps it in `bash -lc`, which does not forward
     signals to a child it is waiting on, so `stop_signal: SIGINT` killed the shell and the node
     was SIGKILLed at the end of the grace period — skipping the destructor that closes the CSV.
     `imulog` has no wrapper and never had this. Fixed with `exec`.
  3. **The CSV buffered its rows** and only flushed in that destructor, so every reading was lost
     with it. Now flushed per row: they arrive at ~2 Hz, so the cost is nothing, and a crash or a
     power cut no longer empties the file.
  4. **A leaked Argus session** failed the next capture with "no session for 0". `log_rig.sh` now
     restarts `nvargus-daemon` in preflight every time, as `retarget-vo-to-imx296-rig` 4.6 asks.

  Verified end to end on `imglog_final_20260905_091337`: 276 sets at 14 us skew, 11 426 IMU
  samples, **17 range readings**, `pulses` advancing exactly +15.

- [x] 5.8 **The range->frame join is NOT the exact integer identity `a870470` claimed.** That commit
  and the node's header both stated `pulses` and the capture side's `seq` were the same counter, so
  "there is nothing here left to solve for". They are not. Both advance one per trigger edge, but
  `seq` is Argus's **per-session** counter (starts near 0) and `pulses` is the MCU's **free-running
  lifetime** counter: on the verification log cam1 `seq` began at 3 while `pulses` was ~60 550.
  A constant integer offset separates them and **no recording writes it down**.

  It is recoverable from the two CLOCK_MONOTONIC columns, but only to about **+-1 frame**, because
  the range `t_mono_ns` is the READ instant and carries 5-20 ms of acquisition plus ~2 ms of UART
  against a 33 ms frame period — which is exactly the ambiguity the pulse counter existed to remove.
  The claim is corrected in the node comment and in the CSV header (`frame_offset = UNMEASURED`).

- [ ] 5.9 **Pin the range->frame offset exactly.** Needs the capture side to see the MCU pulse
  counter, since the range node cannot: it owns the port for the run. Cheapest option is for
  `log_rig.sh` to read the counter at preflight (it already holds the port then) and record it with
  the first frame's `t_sof`. Until this is done, anything derived from a range-to-frame association
  is good to +-1 frame and must say so — including 5.5b, which is the reason it matters.

- [ ] 5.6 Tick or annotate `retarget-vo-to-imx296-rig` 5.0c with 5.2/5.3 above — the replay it records
  is the same log, and the vcam3 occlusion qualifies its tracking result.

## 6. Wrap-up

- [ ] 6.1 Record the frame-convention trap (row-vector `P @ R.T` vs `P @ R`, rig FLU vs the deployed
  node's RFU) in `docs/` where the next person implementing a projection will find it, not only in
  this change.
- [ ] 6.2 Decide the disposition of the panorama prototype (4.7) before archiving.
- [ ] 6.3 Archive this change once §3.5 has an answer and §5.5a has been run — those are the two open
  items that change conclusions rather than add polish.

## Artefacts

Host, all under gitignored `datasets/`:

| path | what |
|---|---|
| `datasets/replay_out/obs_20260903_140714/multicam_bevfit.rrd` | 105 MB — current recording, BEV + panorama panes |
| `datasets/replay_out/obs_20260903_140714/multicam.rrd` | 77 MB — cameras + map only |
| `datasets/replay_out/obs_20260903_140714/multicam_full.rrd` | 275 MB — with fisheye domes |
| `datasets/imglog_vio1_30s_bag` | source images for all of the above |

Commits on `feat/imx296-synced-vo`: `b513a25` (world-fixed BEV plane, superseded), `581cc64`
(incidence cap), `a13dc77` (per-pose plane fit), `97877fa` (panorama pane).

Regenerate:

```
python3 scripts/vo/rerun_multicam.py datasets/replay_out/obs_20260903_140714 \
  --images datasets/imglog_vio1_30s_bag --frames 180 \
  --bev-fit-plane --panorama --pano-depth 3.0 \
  --save datasets/replay_out/obs_20260903_140714/multicam_bevfit.rrd
```
