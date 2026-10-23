# vio-loop-closure Specification (delta)

## ADDED Requirements

### Requirement: A native implementation reproduces the reference

The capability SHALL be available as a single native program, with no interpreter in the
processing path, that embeds the odometry and the loop-closing backend and writes the same output
artefacts as the reference implementation. Its odometry output SHALL be identical to the
reference's, and its loop closures SHALL reproduce the reference's on the same logs within a
stated tolerance.

#### Scenario: Odometry is bit-identical

- **WHEN** a raw log is replayed by the native program and by the reference
- **THEN** the odometry trajectory files agree row for row in every pose column

#### Scenario: Closures reproduce

- **WHEN** a log on which the reference accepted closures is replayed by the native program
- **THEN** the native program selects the same keyframes, accepts constraints between the same
  places including every long loop the reference closed, and its corrected trajectory's
  first-to-last distance lies within the reference's band

#### Scenario: A divergence names its stage

- **WHEN** the native and reference outputs for a log are compared
- **THEN** the comparison reports, per verification stage, the counts on each side, so that a
  difference is attributed to retrieval, tracking, solving, gating or optimisation
