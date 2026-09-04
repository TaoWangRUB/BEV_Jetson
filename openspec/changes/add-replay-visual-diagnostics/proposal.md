## Why

`retarget-vo-to-imx296-rig` 5.0c produced the first replayable 4-camera IMX296 motion log and
replayed it through cuVSLAM end to end. It established that the pipeline *runs*. It could not
establish whether what the pipeline **believes** is right, because the only outputs inspected were
scalars: pose count, rate, path length. A trajectory of the right length can still be the wrong
shape, and a map with 26 759 landmarks can be 90 % noise, and neither shows up in a number.

Two open tasks depend on exactly that distinction and are currently blocked on it. `add-bev-ground-stitch`
task 2.6 requires rendering a recorded synchronised set through the projection **in Python, before
any node exists**, on the grounds that "a wrong sign or a swapped camera is obvious here and subtle
later". And `add-bev-ground-stitch` §1 needs a ground plane, which the rig does not have —
`config/rig/ground_plane.yaml` is still `status: unmeasured`, `height_m: null`.

There is also a class of defect that only a picture catches. Two of the bugs found while building
this were invisible to every scalar check: the four raw fisheyes were being logged into entity paths
that collided with the virtual pinholes, so two panes silently showed the same camera; and a
plane-fitting frame transform was inverted, which produced *plausible* numbers that were wrong. Both
were found by looking.

## What Changes

- **A host-side replay viewer**, `scripts/vo/rerun_multicam.py`, that takes a recorded odometry +
  observations bag plus the raw image log and reconstructs what VO saw: the 8 virtual pinholes with
  their tracked features, the 4 raw fisheyes, the landmark map, and the trajectory, on one timeline.
- **A BEV pane** that is the Python prototype `add-bev-ground-stitch` 2.6 asks for — the same
  ground-plane projection the node will implement, through the Mei model and the full extrinsics,
  rendered per frame so the mosaic can be eyeballed against the source images.
- **A panorama pane** that is the equivalent prototype for the equirectangular stitch, on the
  **current** rig, which the deployed `bev_panorama_node` cannot serve because it is KB-only and
  still configured for the IMX219 rig.
- **A ground-plane estimate from VO landmarks**, fitted near each pose rather than once globally, as
  an *interim* number and a cross-check — explicitly **not** a substitute for the AprilGrid
  measurement in `add-bev-ground-stitch` 1.1–1.5.
- **The findings**, recorded here rather than lost in a terminal: what the replay showed about drift,
  scale, ground height, camera occlusion, and the limits of both stitch methods.

**Scope honesty.** This is a diagnostic, not a product. It runs on the host, offline, on recorded
logs; nothing here ships to the TX2 and nothing here is on the runtime path. Its value is that it
makes the other changes' correctness gates checkable before their nodes exist.

**Scale honesty, stated once here.** Nothing in this change validates absolute scale. The BEV seam
alignment and the landmark plane fit are both derived from the same extrinsic translations, so their
agreement is self-consistency and not evidence about metres. An independent ruler is required and is
listed as a task, not claimed as done.

## Capabilities

### New Capabilities

- `replay-diagnostics`: reconstruct, from a recorded log, what the VO pipeline saw and believed —
  per-camera imagery, tracked features, map and trajectory on one timeline — so that geometric and
  calibration defects are visible rather than inferred.

### Modified Capabilities

<!-- none: this change adds host tooling and evidence. The findings feed
     add-bev-ground-stitch and retarget-vo-to-imx296-rig; those changes keep their own requirements. -->

## Impact

- **New**: `scripts/vo/rerun_multicam.py` (viewer, BEV prototype, panorama prototype, plane fitting).
- **Reused unchanged**: the Mei projection from `scripts/vo/rerun_virtual_pinholes.py`, the bag
  readers from `scripts/vo/rerun_odometry.py`, the finite-depth sphere idea from
  `scripts/calib/pano_tuner.py`, `config/rig/rig_extrinsics_imx296.yaml`, `config/rig/ground_plane.yaml`.
- **Feeds**: `add-bev-ground-stitch` 2.6 (offline mosaic check), 4.3 (above-plane smearing), 5.1
  (seam brightness step); `retarget-vo-to-imx296-rig` 5.1/5.2 (scale and drift).
- **Does not affect**: any node, the container images, the capture path, the trigger. Host-only.
- **Known limitation, by construction**: replay shows what was *recorded*. It cannot distinguish a
  defect introduced at capture from one introduced in VO, and it carries no ground truth of its own.
