# bev-ground-stitch Specification

## Purpose

A metric top-down view of the ground around the rig, stitched from the four fisheyes by projecting
onto the calibrated ground plane, so that a feature lying on that plane appears once and in the
right place regardless of which camera saw it.

## ADDED Requirements

### Requirement: Ground features are parallax-free, and the residual is measured

The stitch SHALL project through the full camera poses — rotation AND translation — onto the
calibrated ground plane, so that a point on that plane maps to the same output cell from every
camera that sees it. The residual SHALL be measured and reported, not assumed.

#### Scenario: The same ground feature lands in one place

- **WHEN** a feature lying on the ground plane is visible to two cameras at once
- **THEN** its position in the output differs between those cameras by no more than a stated
  tolerance, expressed in millimetres on the ground
- **AND** that residual is reported per overlap region, so a bad seam is attributable to a pair

#### Scenario: The residual is a published number, not an impression

- **WHEN** a stitch configuration is produced
- **THEN** the parallax residual is measured against a target of known geometry placed across a seam
- **AND** the result is recorded with the calibration and the rig height it was measured under

#### Scenario: Above-plane objects are characterised, not claimed correct

- **WHEN** an object stands above the ground plane
- **THEN** it is documented that it smears, and by how much as a function of its height and its
  distance from the rig
- **AND** the output is not represented as parallax-free for such objects

### Requirement: The output is metric and its frame is stated

The output SHALL be a metric raster: a stated ground resolution in metres per pixel, a stated extent,
and an origin tied to a named rig frame. A consumer SHALL be able to convert any output pixel to a
position on the ground without consulting the implementation.

#### Scenario: A pixel converts to a ground position

- **WHEN** a consumer reads an output pixel coordinate
- **THEN** the published metadata is sufficient to convert it to metres in the stated frame
- **AND** the frame is the one named in the rig layout, not an implicit camera frame

#### Scenario: Scale is verifiable against a ruler

- **WHEN** an object of known length lies on the ground plane
- **THEN** its length measured in the output agrees with its true length within a stated tolerance

### Requirement: The stitch consumes the calibration the rig actually has

The stitch SHALL use the per-camera model the rig is calibrated with, including fisheye models whose
field exceeds 180 degrees, and SHALL refuse to run against a calibration whose declared sensor,
resolution or model does not match the live configuration.

#### Scenario: A camera model that covers the lens

- **WHEN** the cameras are calibrated with a model able to represent rays beyond 90 degrees incidence
- **THEN** the stitch consumes that calibration directly, without refitting to a narrower model

#### Scenario: A stale calibration is refused

- **WHEN** the calibration's stated resolution, sensor or model disagrees with the running cameras
- **THEN** the node refuses to start and names the disagreement
- **AND** it does not silently fall back to defaults

### Requirement: Overlaps blend, and every seam is attributable

Where cameras overlap, the output SHALL blend rather than hard-cut, and the contributing cameras
SHALL be recoverable for any output cell so that a visible artefact can be traced to a camera pair.

#### Scenario: No hard seam line

- **WHEN** an output cell is covered by more than one camera
- **THEN** the contributions are weighted and blended across the overlap band

#### Scenario: A seam can be attributed

- **WHEN** an artefact appears in the output
- **THEN** the tooling can report which cameras contributed to that region

### Requirement: Runs live on the rig at the capture rate

The stitch SHALL run on the board within the existing runtime, keep up with the synchronised capture
rate, and SHALL consume only whole synchronised camera sets.

#### Scenario: Keeps up with capture

- **WHEN** the node runs on the board with all four cameras publishing
- **THEN** it produces output at the capture rate without unbounded queue growth

#### Scenario: Only synchronised sets are stitched

- **WHEN** the four images of a set are not from the same trigger edge
- **THEN** that set is dropped and counted, rather than stitched from mismatched frames
