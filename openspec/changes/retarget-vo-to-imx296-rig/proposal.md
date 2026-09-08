## Why

The rig changed underneath the software. Ports C–F now carry 4× Sony IMX296LQR global-shutter
modules driven by an STM32H7 hardware trigger — measured **~1 µs inter-camera skew**, replacing the
free-running IMX219 rig whose sets landed 30–86 ms apart and drifted. The BEV stack still assumes the
old rig everywhere: IMX219 intrinsics at 1640×1232, hard-coded Argus `sensor_ids {1,2,3,4}`, and a
latest-frame **bundler + unified per-set timestamp** that exists solely to hide up to ~120 ms of
skew from cuVSLAM's 1 ms `Multicamera` gate. That workaround is now both unnecessary and actively
harmful: it discards true per-frame time on the one rig that finally has it, and it is the
documented main accuracy limiter under motion.

The J106 bring-up project closed all of its changes and explicitly handed the last item here: the
camera↔IMU time offset **Δ** is to be obtained with Kalibr in BEV, alongside the extrinsics BEV needs
anyway (`auvidea-j106-tx2` commit `a21f4b0`).

## What Changes

- **Capture node retargeted to IMX296.** 1456×1088 native mode; port→Argus-sensor-id resolved at
  **runtime** from each `/dev/videoN`'s i2c name (Argus numbers cameras in bind order, not port
  order, so any hard-coded map silently mislabels cameras); 4 cameras on ports C–F.
- **AE locked under external trigger.** In Fast Trigger mode the exposure *is* the XTRIG pulse
  width, so Argus AE cannot move its main actuator and hunts on gain instead — a measured 3.5 Hz
  limit cycle swinging 171 % of mean luma. The node detects trigger mode and clamps gain, matching
  what `scripts/stream/csi_sender.sh` already does.
- **Intrinsics recalibrated** for the IMX296 modules at 1456×1088 (the tracked `cam{1..4}.yaml` are
  IMX219 KANNALA_BRANDT at 1640×1232 — wrong sensor, wrong resolution, wrong lenses).
- **Extrinsics + camera↔IMU Δ from Kalibr**, replacing the hand/feature-derived rig extrinsics for
  the VO path, and giving the first measured Δ against the MPU-9250.
- **BREAKING — the sync workaround is removed.** The bundler and the unified per-set timestamp go
  away; cuVSLAM receives real per-frame timestamps and a genuinely synchronized set. Frame sets that
  fail the sync gate are dropped and counted rather than papered over.
- **The motion test is re-run** to close `bring-up-end-to-end-vo` tasks 3.4 / 3.6: confirm tracking
  under real rig motion and that it is metric (stereo links form across the divergent fisheye pairs),
  not drifting mono.

## Capabilities

### New Capabilities
- `synchronized-capture`: hardware-triggered 4×IMX296 Argus capture — correct port→sensor mapping,
  native mode/resolution, AE locked in trigger mode, true per-frame timestamps, and an observable
  sync health signal (measured skew, dropped sets).
- `rig-calibration`: the calibration inputs the VO consumes — IMX296 intrinsics, Kalibr rig
  extrinsics, and a stated camera↔IMU offset Δ with its provenance.

### Modified Capabilities
- `fused-vo`: the parity requirement currently pins the fused node to the *bundler + unified per-set
  timestamp* behaviour. With hardware sync that clause is removed: the node SHALL pass real
  per-frame timestamps and rely on the hardware trigger, and the tracking bar rises from "≥ ~8.5 Hz,
  no tracking lost" to metric tracking under motion at the capture rate.

## Impact

- **Nodes**: `ros2/bev_camera/src/argus_capture_node.cpp` (mode, sensor resolution, AE),
  `ros2/bev_cuvslam/src/bev_cuvslam_fused_node.cpp` and the modular multicam node (bundler removal,
  timestamp handling), `ros2/bev_cuvslam/config/fused_vo_params.yaml`.
- **Config**: `scripts/config/calib/cam{1..4}.yaml` (new intrinsics), `config/rig/rig_extrinsics_vo.yaml`
  (Kalibr result), a new stated Δ constant.
- **Scripts**: `scripts/calib/*` (capture + solve at the new resolution), `scripts/run_vo_tx2.sh`,
  `scripts/run_vo_fused_tx2.sh`.
- **Hardware/board**: boots `LABEL j106imx296`; requires `jetson-clocks` for ≥3 concurrent streams;
  the STM32 trigger must be running (`j106-trigctl.py`) or triggered cameras produce no frames at all.
- **Cross-repo**: recording for Kalibr uses `auvidea-j106-tx2/tools/j106-record-sync.py`; Δ closes the
  descoped task from that repo.
- **Out of scope**: panorama/BEV re-tuning for the new sensor (the `surround-panorama` spec keeps its
  IMX219-era geometry until VO is closed out), IMX296 ISP colour/vignetting work, refitting ports A/B.
