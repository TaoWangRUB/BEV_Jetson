## MODIFIED Requirements

### Requirement: Fused odometry matches the modular pipeline

The fused node SHALL produce odometry equivalent to the modular capture→VO pipeline (same
calibration, rig, and per-frame timestamps), at equal or better rate. Both pipelines SHALL rely on
the hardware trigger for cross-camera synchronization and SHALL pass each frame's own timestamp to
cuVSLAM — neither may bundle latest-available frames nor rewrite a set to a single synthesised
timestamp.

#### Scenario: Tracking parity

- **WHEN** the fused node runs with the same `scripts/config/calib` + `config/rig` inputs
- **THEN** `/cuvslam/odometry` publishes with live tracking (no "tracking lost") at ≥ the modular node's rate
- **AND** the pose holds near origin when stationary and tracks under rig motion

#### Scenario: Real timestamps reach cuVSLAM

- **WHEN** a synchronized set is passed to `Track()`
- **THEN** each image carries its own acquisition timestamp
- **AND** no unified per-set timestamp is substituted

#### Scenario: An unsynchronized set is dropped, not repaired

- **WHEN** a set's inter-camera skew exceeds the configured limit
- **THEN** the set is dropped and counted
- **AND** the node does not fabricate a timestamp to make it acceptable to cuVSLAM

#### Scenario: Lower CPU than the modular pipeline

- **WHEN** the fused node and the modular two-node pipeline are each run at the same camera rate
- **THEN** the fused node uses measurably less CPU (no per-frame NvBufferMemMap memcpy + DDS image serialization)

## ADDED Requirements

### Requirement: Tracking is metric under rig motion

With a synchronized rig the VO SHALL track real motion metrically — scale established from
cross-camera stereo links, not drifting as an unscaled monocular estimate.

#### Scenario: Translation is recovered at true scale

- **WHEN** the rig is moved a measured distance along a straight path
- **THEN** the odometry translation matches the measured distance within 5 %
- **AND** the trajectory returns near its start when the rig is returned to its starting pose

#### Scenario: Stereo links actually form

- **WHEN** VO runs on the four-camera rig
- **THEN** cross-camera feature associations are established between at least one overlapping pair
- **AND** if no pair passes the overlap gate, the node reports that it is running unscaled rather than presenting the pose as metric
