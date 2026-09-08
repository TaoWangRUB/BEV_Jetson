# replay-diagnostics Specification

## Purpose

The ability to reconstruct, from a recorded log, what the VO pipeline saw and what it believed — the
per-camera imagery, the features it tracked, the map it built and the trajectory it produced, on one
shared timeline — so that geometric and calibration defects are **visible** rather than inferred from
summary statistics. A pose count and a path length cannot distinguish a correct trajectory from a
wrong one of the same length.

## ADDED Requirements

### Requirement: A recorded run is reconstructable offline

A recorded log SHALL be replayable on a host into a view showing the source imagery, the tracked
features, the map and the trajectory together on one timeline, without the target hardware.

#### Scenario: Every camera the tracker used is shown

- **WHEN** a log is replayed
- **THEN** each camera the tracker consumed is displayed with the features it contributed
- **AND** where physical cameras are subdivided into virtual ones, both the physical and the virtual
  views are available, since a defect in one is not visible in the other

#### Scenario: Views share a timeline

- **WHEN** a defect is observed at some instant
- **THEN** every other view can be inspected at that same instant
- **AND** the reconstruction is a persistable artefact, so a finding is reproducible by reference
  rather than by description

#### Scenario: Distinct entities are distinctly named

- **WHEN** two views derive from different sources
- **THEN** they occupy distinct identities in the output
- **AND** a naming collision that would cause one view to silently display another's content is
  treated as a defect, not a display quirk

### Requirement: Absent data is shown as absent

A view that cannot be produced for a given instant SHALL be rendered as empty, and SHALL NOT be
omitted in a way that leaves a previous instant's content on display.

#### Scenario: A frame that cannot be rendered

- **WHEN** a derived view is invalid or unavailable at some instant
- **THEN** the view is blank at that instant
- **AND** it does not retain the last successfully rendered content, which would read as the view
  being frozen and would misattribute the fault

#### Scenario: Degraded output degrades visibly

- **WHEN** a derived view is valid but of degraded quality because the geometry is unfavourable
- **THEN** the degraded region is excluded on a stated geometric criterion rather than the whole
  view being suppressed
- **AND** the criterion and the resulting coverage are stated

### Requirement: Projections are verified against the rig's own geometry before they are trusted

A view that projects through the camera models and extrinsics SHALL be checked against a property
derivable from the rig geometry alone, and the check SHALL be recorded.

#### Scenario: Seams fall where the layout says they must

- **WHEN** a multi-camera projection is produced
- **THEN** the boundaries between camera contributions fall at the bisectors of the cameras'
  declared bearings
- **AND** a disagreement is reported as an inconsistency between the rig layout and the extrinsics,
  since it cannot be resolved by adjusting the view

#### Scenario: Camera bearings are recovered from the extrinsics

- **WHEN** a panoramic or top-down projection is built
- **THEN** each camera's recovered bearing in the output agrees with its declared bearing in the rig
  layout within a stated tolerance
- **AND** the recovered bearing is computed by a method that is well defined where the blend weight
  is constant, since a weight that saturates has no unique maximum

### Requirement: A derived scene estimate states its method and its validity

An estimate of scene geometry derived from the map SHALL state how it was obtained and under what
assumption, and SHALL NOT be presented as a calibrated measurement.

#### Scenario: An estimate that assumes a consistent map

- **WHEN** an estimate is derived from map points spanning a period over which the map may have
  drifted
- **THEN** the assumption of global consistency is stated
- **AND** where that assumption does not hold, the estimate is formed locally, from points near the
  pose it describes, expressed in that pose's own frame

#### Scenario: An estimate is not promoted to a calibration

- **WHEN** an estimate agrees with an independent quantity
- **THEN** it is recorded as a cross-check with its spread
- **AND** it is not written into a calibration file that is marked unmeasured, since agreement
  between two quantities derived from the same extrinsics is self-consistency and not measurement

### Requirement: Method-inherent error is characterised, not removed

Where a view relies on an approximation of scene geometry, the residual error SHALL be characterised
and attributed to the method.

#### Scenario: A single-surface approximation

- **WHEN** the scene is projected onto a single surface
- **THEN** content not on that surface is documented as misregistered, with the magnitude and the
  conditions under which it grows
- **AND** the view is not represented as correct for such content

#### Scenario: A parameter chosen by measurement

- **WHEN** a free parameter of the approximation is selected empirically
- **THEN** the metric used is invariant to quantities it is not intended to measure
- **AND** candidate values are scored over an identical sample, so that values are compared on the
  same content
