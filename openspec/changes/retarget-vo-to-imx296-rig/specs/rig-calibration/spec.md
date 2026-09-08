## Purpose

Hold the measured geometry and timing the VO consumes — per-camera intrinsics, the rig's
camera-to-camera extrinsics, and the camera↔IMU time offset — each stated with the sensor,
resolution, and provenance it was measured under, so a stale calibration cannot be applied silently.

## ADDED Requirements

### Requirement: Intrinsics match the fitted sensor and capture resolution

Each camera SHALL have intrinsics measured on the IMX296 module fitted to its port, at the
resolution used for capture, recorded with a fisheye model appropriate to the lens.

#### Scenario: Calibration declares what it was measured on

- **WHEN** a calibration file is loaded
- **THEN** it states the camera model, the image size, and the sensor family it was measured on
- **AND** a consumer can reject it if any of those disagree with the live configuration

#### Scenario: Reprojection error is within tolerance

- **WHEN** intrinsics are solved for a camera
- **THEN** the RMS reprojection error is under 1.0 px
- **AND** the result is rejected, not published, if it is not

### Requirement: Rig extrinsics are jointly estimated across all four cameras

The rig SHALL have a single set of camera-to-camera extrinsics estimated jointly from one recording
of a common calibration target, expressed in a stated body frame, and covering the mounting
orientation of the physical modules.

#### Scenario: Extrinsics are consistent around the rig

- **WHEN** the four camera poses are composed around the ring
- **THEN** the loop closes to within the stated solver residual
- **AND** the per-camera residuals are reported alongside the result

#### Scenario: Mounting orientation is captured, not assumed

- **WHEN** the modules are mounted rotated relative to the body frame
- **THEN** that rotation is present in the published extrinsics
- **AND** the file states which convention (raw sensor frame or corrected frame) its poses are in

#### Scenario: A calibration is traceable to its recording

- **WHEN** extrinsics are published
- **THEN** they name the recording and the date they were solved from

### Requirement: The camera↔IMU time offset is a single stated constant

The rig SHALL carry one camera↔IMU time offset Δ, stated with its provenance and its estimated
uncertainty. Because all cameras share one trigger edge, Δ has no per-camera component.

#### Scenario: Delta is stated with provenance

- **WHEN** Δ is recorded
- **THEN** it states the value, how it was obtained, and the uncertainty of that method
- **AND** an offset that has not been measured is marked as unmeasured rather than defaulted to zero

#### Scenario: Delta is applied consistently

- **WHEN** camera and IMU samples are placed on a common timebase
- **THEN** Δ is applied in one documented direction, stated in the calibration
- **AND** the same convention is used by every consumer

### Requirement: Superseded calibrations are retained, not overwritten

Replacing a calibration SHALL preserve the previous one and record why it was superseded.

#### Scenario: The IMX219-era calibration survives the switch

- **WHEN** IMX296 intrinsics replace the IMX219 files
- **THEN** the IMX219 calibration remains recoverable and is labelled with the rig it belongs to
- **AND** a board refitted with IMX219 modules can use it without re-measurement
