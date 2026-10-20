# vio-loop-closure Specification

## Purpose
The ability to recognise, from the rig's own imagery and odometry, that it has returned to a place
it already visited; to verify that recognition geometrically before trusting it; and to produce a
globally consistent trajectory beside the odometry, with bounded cost and deterministic output.
Pure odometry has nothing to pin the trajectory to; this capability is what pins it.

## Requirements

### Requirement: Odometry is preserved

The loop-closing backend SHALL consume the odometry's keyframe poses, keyframe imagery and
landmarks, and SHALL NOT alter the odometry's own output. The pure-odometry trajectory remains
available, unchanged, beside every corrected output.

#### Scenario: Running with loop closure does not change the odometry

- **WHEN** a log is processed with the backend enabled and again with it disabled
- **THEN** the pure-odometry trajectory written in both runs is identical pose for pose

#### Scenario: Corrected outputs are separate artefacts

- **WHEN** the backend produces a corrected trajectory
- **THEN** it is written as a separate output next to the odometry trajectory, never in its place

### Requirement: Revisits are recognised

The backend SHALL propose loop candidates for a keyframe from among earlier keyframes, by two
independent routes: proximity of the odometry pose, and visual similarity of the imagery across all
cameras of the rig. Candidates younger than a configurable minimum time separation SHALL never be
proposed.

#### Scenario: A nearby earlier keyframe is proposed

- **WHEN** an earlier keyframe lies within the configured radius of the current odometry pose and is
  older than the minimum separation
- **THEN** it is among the candidates for the current keyframe

#### Scenario: A visually similar keyframe is proposed despite drift

- **WHEN** an earlier keyframe shows the same place but the odometry has drifted beyond the radius
- **THEN** the appearance route can still propose it, from any heading the ring of cameras covers

#### Scenario: Recent keyframes are never candidates

- **WHEN** an earlier keyframe is younger than the minimum separation
- **THEN** it is not proposed by either route, however similar or near it is

### Requirement: A hypothesis is accepted only when geometry confirms it

A candidate SHALL become a loop constraint only after a relative pose between the two keyframes has
been solved from image correspondences with metric depth, and that solution passes acceptance gates:
a minimum number of supporting correspondences, a minimum inlier ratio that rises as the support
falls, and consistency between an initial solution and a refined one. A rejected hypothesis SHALL
leave no trace in the trajectory and SHALL be recorded with the reason for rejection.

#### Scenario: A true revisit yields a constraint

- **WHEN** the candidate shows the same place and enough correspondences agree on one relative pose
- **THEN** a loop constraint with that relative pose and a confidence derived from the inlier ratio
  is added

#### Scenario: A look-alike is rejected

- **WHEN** the candidate is visually similar but its correspondences do not agree on one relative
  pose above the gates
- **THEN** no constraint is added, the trajectory is unchanged, and the rejection and its reason are
  recorded

#### Scenario: Acceptance is reproducible

- **WHEN** the same log is processed twice with the same configuration
- **THEN** the same hypotheses are accepted and rejected, with identical relative poses

### Requirement: The corrected trajectory is globally consistent

The backend SHALL maintain a graph of keyframe poses whose edges are the odometry's relative poses
between consecutive keyframes and the accepted loop constraints, SHALL keep roll and pitch at the
odometry's gravity-aligned values, and SHALL re-optimise the graph whenever a constraint is accepted.
Every frame of the odometry SHALL receive a corrected pose derived from its keyframe's correction.

#### Scenario: A loop closes the drift

- **WHEN** a constraint joins the current keyframe to an earlier one
- **THEN** after optimisation the two keyframes' separation agrees with the constraint within its
  confidence, and the poses between them are redistributed rather than teleported at one frame

#### Scenario: Gravity alignment survives optimisation

- **WHEN** the graph is optimised
- **THEN** every keyframe's roll and pitch remain within the odometry's own uncertainty of their
  odometry values

#### Scenario: Frames between keyframes are corrected

- **WHEN** the optimised keyframe poses are written
- **THEN** every odometry frame has a corrected pose continuous with its neighbouring keyframes

### Requirement: Cost is bounded

The backend SHALL enforce a minimum interval between accepted closures and a cap on graph nodes,
both configurable, and SHALL report its per-keyframe processing time so that its cost can be compared
with the odometry's budget.

#### Scenario: Closures are throttled

- **WHEN** a constraint was accepted less than the configured interval ago
- **THEN** no new hypothesis is verified until the interval has elapsed

#### Scenario: Cost is visible

- **WHEN** a log has been processed
- **THEN** the output reports retrieval, verification and optimisation time per keyframe

### Requirement: The effect is measurable

For every processed log the backend SHALL report, for both the odometry and the corrected
trajectory: path length, the distance between the first and last pose, and, for each accepted loop
constraint, the separation of its two keyframes in each trajectory. Where a reference measurement of
the return-to-start distance exists for the log, it SHALL be reported beside both.

#### Scenario: A closed-circuit log is scored

- **WHEN** a log recorded as a closed circuit with a measured return-to-start distance is processed
- **THEN** the report shows that distance beside the odometry's and the corrected trajectory's
  first-to-last separation

#### Scenario: Loop edges are inspectable

- **WHEN** the comparison view of a log is rendered
- **THEN** the corrected trajectory is drawn beside the odometry with each accepted constraint drawn
  between the two keyframes it joins
