# ground-plane-calibration Specification

## Purpose

The rig's pose relative to the ground — how high it sits and how it is tilted — measured and stated
with provenance. Every ground-plane projection depends on it, and an error here is indistinguishable
from a calibration error in the cameras themselves.

## ADDED Requirements

### Requirement: The ground plane is measured and stated with provenance

The rig SHALL carry a stated ground plane: the height of the rig frame above it and its orientation,
recorded with how it was obtained, when, and its estimated uncertainty. An unmeasured plane SHALL be
marked as such rather than defaulted.

#### Scenario: The plane states how it was obtained

- **WHEN** the ground plane is recorded
- **THEN** it states the height, the orientation, the method, the date and the uncertainty
- **AND** a plane that has not been measured is marked unmeasured, never assumed level at a guessed height

#### Scenario: Uncertainty is propagated to the output

- **WHEN** the plane carries a stated uncertainty
- **THEN** the resulting ground-position error at the edge of the output extent is stated with it,
  because a small tilt error grows with distance from the rig

### Requirement: The plane is validated against known geometry

The measured plane SHALL be checked against an independent object of known size on the ground, and
the check SHALL be recorded alongside the result.

#### Scenario: A known length measures correctly

- **WHEN** an object of known length lies flat on the ground within the output extent
- **THEN** its measured length agrees with its true length within the stated tolerance
- **AND** the agreement is checked near the edge of the extent as well as near the rig, since tilt
  error is smallest underneath and largest far away

### Requirement: A changed rig invalidates the plane

Anything that moves the cameras relative to the ground SHALL invalidate the stated plane, and the
system SHALL be able to detect that the plane no longer matches the rig.

#### Scenario: Remounting invalidates it

- **WHEN** the rig height, its mounting, or the camera extrinsics change
- **THEN** the stated plane is marked superseded and is not used for a new stitch
- **AND** the previous value is retained rather than overwritten

#### Scenario: Disagreement is detectable

- **WHEN** the plane no longer describes the rig
- **THEN** the parallax residual on ground features rises above its stated tolerance, which is the
  observable symptom a user can check without re-measuring
