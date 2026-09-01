## 0. Gate on the calibration this change consumes

- [ ] 0.1 Confirm `retarget-vo-to-imx296-rig` 3R.16 has written
      `config/rig/rig_extrinsics_imx296.yaml` and `config/calib/imx296_1456x1088/camN.yaml`,
      including the cam3 re-sweep. Nothing below is meaningful on the current cam3 solve
      (subset stability 34.4 px, both baselines inflated, the two worst epipolar residuals).
- [ ] 0.2 Verify the extrinsics file is the ring-closed one and that the 180° mounting roll is
      **not** folded into it. Compare cam1→cam2 against `rig_extrinsics_vo.yaml`; they must differ
      by exactly that roll. Wrong file here rotates the entire mosaic and looks like a stitch bug.
- [ ] 0.3 Add the rig fingerprint (port→camera assignment + calibration id/date) to both calibration
      files if 3R.16 did not, so task 4.4 has something to compare against.

## 1. Ground-plane calibration

- [ ] 1.1 Write `scripts/calib/ground_plane_calib.py`: detect the AprilGrid lying flat on the floor
      in cam1, solve its pose, emit `T_ground_rig` — reusing the detector and solver already in the
      calibration path, not a second implementation.
- [ ] 1.2 Record one board-on-floor set with the rig in its normal mounted position. **Preview and
      wait for an explicit go before recording.**
- [ ] 1.3 Solve, and write `config/rig/ground_plane.yaml`: `T_ground_rig`, derived height and
      roll/pitch/yaw for human reading, the rig fingerprint, the source recording, and the residual.
- [ ] 1.4 Cross-check roll and pitch against IMU gravity from the same session. Record both numbers
      and their difference in the yaml — agreement is evidence, disagreement is a finding.
- [ ] 1.5 Sanity-check the derived height against a tape measure and record both.

## 2. Static mapping

- [ ] 2.1 Port the verified Mei projection (retarget-vo 4.1b) into `ros2/bev_ground/` — port, not
      re-derive; re-verify against `cv2.omnidir.projectPoints` on the way in.
- [ ] 2.2 Build the backward LUT: output cell → rig-frame point on the plane → `T_cam_rig` →
      Mei projection → source pixel. Double precision, one `CV_32FC2` map per camera.
- [ ] 2.3 Mark cells invalid where the projection does not round-trip within a pixel, or falls behind
      the camera, or outside the image. Emit a per-camera validity mask.
- [ ] 2.4 Build the per-cell blend weights: falloff toward each camera's valid-region edge, normalised
      across cameras so the sum is 1 wherever any camera sees.
- [ ] 2.5 Emit the combined validity mask, leaving the under-rig blind cone unpainted. Assert no cell
      is painted from zero cameras.
- [ ] 2.6 Offline check before any node exists: render one recorded synchronised set through the maps
      in Python and eyeball the mosaic. A wrong sign or a swapped camera is obvious here and subtle later.

## 3. The node

- [ ] 3.1 Scaffold `ros2/bev_ground/` as a Foxy package that builds inside `cuvslam-foxy:tx2`.
- [ ] 3.2 Subscribe to the four camera topics, form synchronised sets by timestamp, and stitch only
      complete sets — dropping incomplete ones rather than stitching stale frames.
- [ ] 3.3 De-invert the upside-down mounting once at ingest, and assert it is not also folded into
      the extrinsics.
- [ ] 3.4 Apply `cv::cuda::remap` per camera and the weighted sum, reusing the `bev_cuvslam` remap
      machinery rather than adding a second copy.
- [ ] 3.5 Publish `/bev/ground` (image) and latched `/bev/ground/info` (`nav_msgs/MapMetaData`:
      resolution and origin).
- [ ] 3.6 Parameters: output extent, resolution, config paths; rebuild the tables on change rather
      than silently keeping stale ones.

## 4. Correctness gates

- [ ] 4.1 Scale check: a known length on the ground measures correctly in the output at ~1 m and
      again at ~3 m. Two ranges, because a plane-tilt error is invisible at one.
- [ ] 4.2 Parallax residual: measure how far the same ground feature lands from itself between two
      cameras, **reported per seam**, not averaged. This is the number the change is judged by.
- [ ] 4.3 Characterise above-plane smearing: a vertical object of known height, and how far its top
      is displaced. Record it; do not claim it away.
- [ ] 4.4 Startup refusal: a mismatched rig fingerprint, or a ground plane recorded against a
      different fingerprint, must exit with a message naming the mismatch — not produce a mosaic.
- [ ] 4.5 Timing on the TX2 with capture running: sustained stitch rate at the 15 Hz capture rate,
      and what it costs the rest of the board.

## 5. Photometric

- [ ] 5.1 Quantify the brightness step at each seam with no compensation, so a photometric defect is
      never later mistaken for a geometric one.
- [ ] 5.2 Estimate a per-camera gain on the overlap regions and apply it before blending.
- [ ] 5.3 Re-measure 5.1 and 4.2 after compensation — gain must not have moved the geometry.

## 6. Optional depth stage (behind a flag, off by default)

- [ ] 6.1 Record a textured scene at 1–3 m and measure virtual-stereo validity on it. The calibration
      sweeps give 10–17 % valid, which is the scene, not the rig — this stage needs a real number
      before it is worth building.
- [ ] 6.2 Per camera, override plane depth with virtual-stereo depth where it is valid and disagrees
      with the plane beyond a threshold.
- [ ] 6.3 Re-measure 4.3 with the stage on: state how much above-plane smearing it removes, and that
      it does not remove all of it.

## 7. Documentation

- [ ] 7.1 `scripts/viz/bev_view.sh` to display `/bev/ground` on the host, following the existing
      `rviz_from_tx2.sh` pattern (Foxy + CycloneDDS + private domain).
- [ ] 7.2 Add the new scripts to `scripts/README.md`.
- [ ] 7.3 Record the measured parallax residual, scale accuracy, and above-plane characterisation
      in the change's evidence, and state plainly which part of "eliminate parallax" is met and which
      is out of reach by construction.
