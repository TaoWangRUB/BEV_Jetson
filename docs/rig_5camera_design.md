# Five cameras at 72°: the rig change

**Decision:** move from 4 cameras at 90° to **5 cameras at 72°**, to close the blind cones
that the 4-camera carve leaves. Verified geometrically with `scripts/calib/rig_design.py`;
not yet built.

## Why 4 cameras cannot close the gap, and 5 can

Carve each fisheye into two virtual pinholes at ±S/2 (S = camera separation) with pinhole
field exactly S. They then span [−S, 0] and [0, S]: they **meet on the optical axis**, so
there is no blind cone and no self-overlap, and N of them tile exactly 360°.

At S = 90° that rule needs a pinhole reaching 90° off-axis. The lens is D190/**H160**, so
it reaches 80°. **Four cameras cannot tile the ring with this lens at any pinhole field** —
that is the origin of today's four 20° blind cones, and it is geometric, not a tuning
choice. Five at 72° is the first count that fits, with 8° of margin.

| cams | sep | fov_pin | focal px | baseline | f·B | fits H160? |
|---|---|---|---|---|---|---|
| 4 | 90.0° | 90.0° | 384.0 | 147.2 mm | 56.5 | **no** — needs 90°, lens gives 80° |
| **5** | **72.0°** | **72.0°** | **528.5** | **122.4 mm** | **64.7** | **yes** |
| 6 | 60.0° | 60.0° | 665.1 | 104.1 mm | 69.2 | yes |
| 7 | 51.4° | 51.4° | 797.4 | 90.3 mm | 72.0 | yes |

(ring radius 104.1 mm, as built today. Current 4-camera rig runs fov_pin 70° / focal 548 /
f·B 80.7 — it beats these on f·B *only* because it declines to cover the full ring.)

Note the trend: f·B ∝ R·cos(S/2), so at a fixed ring radius **more cameras is
monotonically better for depth precision**. The limit is CSI ports, trigger channels and
compute, not geometry. Six would be better than five on every optical axis; five was
chosen to keep a spare CSI port and a spare trigger channel.

## The design

- 5× IMX296, separation 72°, ports c–f plus **port a**
- carve at ±36°, virtual pinhole **fov 72°, 768×576, focal 528.5 px**, zero distortion
- pinhole reaches 72° off-axis against the lens's 80° — 8° margin
- **zero blind cone**, ring covered exactly once
- **10 virtual cameras**, 5 stereo pairs

Predicted cuVSLAM frustum graph (its own algorithm, run offline): all five facing pairs at
**0.961**, which is the ceiling — the gate samples 31×31 = 961 points against a nominal
denominator of 1000. No spurious edges, every virtual camera degree 1, none dropped.

## Ring radius: the one open decision

Baseline is 2R·sin(S/2), so tightening the separation shortens it unless the ring grows.

| ring radius | baseline | f·B | vs today (80.7) |
|---|---|---|---|
| 104.1 mm (as built) | 122.4 mm | 64.7 | −20 % |
| 125.0 mm | 147.0 mm | 77.7 | −4 % |

Keeping the ring costs 20 % of depth precision; growing it to 125 mm costs 21 mm of rig
radius and recovers almost all of it. This is a mechanical call, not a software one.

## Hardware prerequisites

1. **A fifth IMX296 on port a.** The J106 carries 6 CSI ports; b is empty and a is unused
   (README §26–48), so the port exists. The 6-CSI device tree already works — it ran the
   old IMX219 rig.
2. **A fifth trigger channel — this is the blocker.** `j106-trigctl.py status` reports
   **four** channels. Either fan one channel out to two cameras (they take the same period
   and pulse width, so this is electrically reasonable if the driver sources the extra
   opto-isolator), or revise the trigger board. Nothing else in the plan is gated on new
   hardware; this is.
3. Bandwidth: 5 × 1456×1088 × 30 Hz ≈ 238 MB/s raw, up from 190. Within what the capture
   path already handles, but the recording path was sized for four.

## What this invalidates

Everything downstream of the rig geometry. The calibration data collected 2026-08-28 is a
4-camera dataset and does not carry over:

- intrinsics for the new camera (the existing four stay valid — same lens, same sensor)
- **all five pairwise extrinsics** — re-record, the pairs are different pairs
- **ring closure** — five hops now, `close_rig_ring.py` needs its RING table extended
- camera↔IMU Δ — re-measure, though −8.06 ms should reappear if the hypothesis holds,
  which makes this a useful check on 3.6b rather than pure rework
- `virtual_stereo_imx296.yaml` — new fov, new focal, five pairs

Worth doing at the same time, since the rig is on the bench either way: a recording where
**three or more cameras see the board simultaneously**, which is the only way to attack the
~1° systematic per-recording bias that ring closure can redistribute but not remove.

## Status

Geometry verified, hardware not built, no calibration re-recorded. §4 and §5 of
`retarget-vo-to-imx296-rig` still describe the 4-camera rig.
