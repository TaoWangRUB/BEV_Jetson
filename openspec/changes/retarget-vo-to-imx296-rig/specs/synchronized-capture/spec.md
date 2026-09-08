## Purpose

Deliver genuinely simultaneous 4-camera image sets from the hardware-triggered IMX296 rig: each set
carries frames exposed within microseconds of each other, correctly attributed to physical camera
positions, with stable brightness and true per-frame timestamps that downstream VO can trust.

## ADDED Requirements

### Requirement: Cameras are identified by physical port, not by Argus index

The capture path SHALL resolve each physical carrier port (C, D, E, F) to its Argus `sensor-id` at
runtime, from the i2c name reported by the corresponding `/dev/videoN`. It SHALL NOT rely on a
static port→index map.

Argus numbers cameras in `/dev/video` bind order, which is not port order and varies between boots,
so a static map mislabels cameras silently — producing a rig whose extrinsics are attached to the
wrong images.

#### Scenario: Ports map correctly regardless of bind order

- **WHEN** capture starts on a board whose IMX296 modules enumerated in an order different from port order
- **THEN** each published camera stream corresponds to the physical port recorded in its calibration
- **AND** the resolved port→sensor-id mapping is logged at startup

#### Scenario: Sensor family is recognised by either address

- **WHEN** a port carries an IMX296 (i2c `…-001a` / `…-0018`) or an IMX219 (`…-0010` / `…-0012`)
- **THEN** the port is resolved for either family without a configuration flag

#### Scenario: A missing camera is reported, not silently dropped

- **WHEN** fewer than the configured number of ports resolve to a live device
- **THEN** capture reports which ports are missing and refuses to start a rig it cannot fully populate

### Requirement: Capture uses the IMX296 native geometry

The capture path SHALL acquire at the IMX296's native 1456×1088 and SHALL reject a configuration
whose resolution does not match the calibration files it loads.

#### Scenario: Resolution and calibration agree

- **WHEN** the node starts with calibration declaring an image size different from the configured capture size
- **THEN** it fails at startup with a message naming both sizes
- **AND** it does not publish frames under a mismatched calibration

### Requirement: Exposure is stable under external trigger

When the sensor driver is in external-trigger mode, the capture path SHALL prevent Argus
auto-exposure from modulating gain.

In Fast Trigger mode the exposure time *is* the trigger pulse width, so AE cannot move its main
actuator and instead hunts on gain — measured as a 3.5 Hz limit cycle swinging 150 luma levels
peak-to-peak (171 % of the mean).

#### Scenario: Trigger mode is detected and AE is clamped

- **WHEN** the driver reports external-trigger mode active
- **THEN** capture runs with AE locked and gain clamped
- **AND** it logs that it did so, with the gain values used

#### Scenario: Free-running capture is unaffected

- **WHEN** the driver is not in trigger mode
- **THEN** capture leaves auto-exposure enabled, unchanged from free-running behaviour

#### Scenario: Brightness holds steady

- **WHEN** a triggered stream is measured over a stationary scene for at least 30 s
- **THEN** frame-mean luma varies by less than 5 levels peak-to-peak

### Requirement: Frame sets carry true per-frame timestamps

Each published frame SHALL carry the timestamp of its own exposure. The capture path SHALL NOT
substitute a shared or synthesised timestamp across a set.

#### Scenario: Timestamps are per-frame

- **WHEN** a set of 4 triggered frames is published
- **THEN** each frame's timestamp is its own acquisition time
- **AND** the spread across the set reflects the measured hardware skew rather than being identically zero

### Requirement: Synchronization health is observable

The capture path SHALL expose a measurable indication of set quality: the inter-camera skew of
recent sets and the number of sets rejected for exceeding the configured skew limit.

#### Scenario: Hardware sync is confirmed at runtime

- **WHEN** capture runs with the trigger active
- **THEN** reported worst-case inter-camera skew stays below 1 ms
- **AND** the rejected-set count stays at zero over a sustained run

#### Scenario: A stopped trigger is diagnosable

- **WHEN** the external trigger stops while capture is running
- **THEN** the loss of frames is reported as a trigger fault, distinguishable from a camera failure
