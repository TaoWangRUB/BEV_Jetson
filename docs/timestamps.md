# Timestamps — camera and IMU on one clock

The four cameras share one hardware trigger edge, so **the rig's sync is a solved hardware problem
and every remaining timing error is created in software.** This document is the convention the whole
stack follows; the measurements behind it live in the J106 project (`auvidea-j106-tx2`, README §5
"Latency and where to timestamp" and §5a "Camera ⟷ IMU timebase").

**The contract, in one line:** every sensor message carries the **exposure midpoint** (cameras) or
the **data-ready edge** (IMU), both on **`CLOCK_MONOTONIC`**, and the residual camera↔IMU offset
**Δ** is a single stated constant — never silently assumed to be zero.

## The four rules

**1. One clock: `CLOCK_MONOTONIC`.** V4L2 stamps buffers with it (flag `0x00002001`, verified on
this board), so everything else follows it rather than converting. Two traps:

- The GPIO chardev's own event timestamp is `CLOCK_REALTIME` (`gpiolib.c`, `ktime_get_real_ns()`).
  Use the chardev to *wait* for an edge, then take your own `clock_gettime(CLOCK_MONOTONIC)` —
  mixing the two misdates everything by the REALTIME↔MONOTONIC offset, and NTP can slew one under
  you.
- ⚠ **`header.stamp` in this stack is monotonic, not ROS system time.** `argus_capture_node`
  publishes the frame's monotonic time in the header. Never compare it against `now()`, and any IMU
  node feeding this pipeline must stamp the same way. It is the difference between the two streams
  that must be right, and a bag replayed later must not be re-dated.

**2. Stamp the exposure midpoint, not the frame's arrival.** Three timestamps look like a capture
time and are not:

| Source | What it actually is | Usable? |
|---|---|---|
| `EGLStream::IFrame::getTime()` | consumer-side frame time | ❌ measured ~7 ms apart *in the order the capture loop visits the cameras* |
| `nvarguscamerasrc` GstBuffer PTS | output-stamped (~0.7 ms, transport only) | ❌ and the RTP path discards capture time entirely |
| `ICaptureMetadata::getSensorTimestamp()` | kernel **SOF** — first data out of the sensor | ✅ this is the one |

SOF is the start of **readout**, which on a global shutter is *after* the exposure has finished, so
the instant the frame depicts is:

```
t_frame = SOF − exposure/2
```

Half an exposure is a constant bias against the IMU that would otherwise hide inside Δ and move
whenever the exposure changed. Under the hardware trigger the true exposure **is the trigger pulse
width**, and **Argus does not know it** — measured on this rig Argus reported 0.521 ms while the
STM32 was emitting 4.986 ms, so trusting Argus put the stamp 2.2 ms off. Read the real value from
the generator and pass it as `exposure_us`:

```bash
# on the TX2 — the MCU is on the M110 UART, not a USB CDC port
sudo python3 tools/j106-trigctl.py --port /dev/ttyTHS1 status
#   period_us=33333  ch1_exposure_us=5000 pulse_ns=4985740  (all four channels equal)
```

⚠ Re-read it after any MCU reset: the firmware boots at its compiled-in defaults and says so
nowhere. Note `pulse_ns` (4.98574 ms) is the value to use, not the commanded 5000 µs.

One exposure is used for the whole rig: all four cameras expose on the same edge, so a per-camera
value would inject differences the hardware does not have. The firmware *can* set per-channel
exposures (`raw 'exp <ch> <us>'`) — if that is ever used, this assumption and the single
`exposure_us` parameter both stop holding.

The other latencies are *not* timestamp corrections — they are delivery costs (measured at
1456×1088): readout+MIPI→VI 16.1 ms, ISP 1.9 ms, raw-V4L2 buffer age at `DQBUF` **66.7 ms**. Note
bypassing the ISP makes latency *worse*: use Argus for pixels, and the raw V4L2 path only for
*proving* sync.

**3. Match frames by timestamp, never by position.** Each camera's queue advances independently, so
"one frame from each camera per loop iteration" can mix adjacent trigger edges — it read as 35 ms of
skew on a rig whose real skew is 1 µs. The capture node matches each of camera 0's frames to the
nearest frame from every other camera and reports the spread; sets beyond `max_skew_us` (default
1000, cuVSLAM's own gate) are counted, not repaired.

For **recordings used in calibration**, go further and *fit* the frame times: the trigger is
hardware-periodic, so `t[k] = a·k + b` indexed by the V4L2 `sequence` field (not arrival order — one
dropped frame in 600 shifts the fitted period by ~5800 ppm). With 1.5 µs of per-frame jitter the
phase error falls as 1/√N. The slope `a` also absorbs the fact that the STM32 free-runs against the
Tegra (−12.3 ppm de-slewed), which would otherwise make a once-calibrated offset go stale.
`j106-frametime.py` and `j106-record-sync.py` in the J106 repo do this.

## What the capture node records

Every frame is published twice: the pixels on `/camN/image_raw`, and its timing on
`/camN/frame_meta` (`bev_camera/msg/FrameMeta`) under the **same stamp**, so a bag keeps the
timing even when images are throttled or dropped for bandwidth. The message carries the exposure
midpoint (`header.stamp`), the raw `sof_ns` so the correction stays undoable, the `exposure_ns` it
was derived from, and **two** sequence counters:

| field | side | what a gap in it means |
|---|---|---|
| `capture_id` | Argus session | the session did not produce that capture |
| `frame_number` | consumer | it was produced but never reached us |

`image_published` says whether the pixels for that stamp actually went out: the NVMM map or copy
can fail after the frame has been acquired and timed, and the timing record still stands. False
means "expect no image at this stamp" rather than leaving a consumer with an unmatched record.

Set `frame_log_dir` and the node also writes `camN.csv` per camera — the same rows in the shape
`j106-frametime.py` fits, under a provenance header (clock, timestamp convention, port, sensor,
resolution, trigger state and rate, exposure source, and Δ marked `UNMEASURED`):

```
#timestamp [ns],seq,capture_id,t_sof [ns],exposure [ns],image
11521506304000,355,355,11521508797000,4986000,1
```

Measured over 30 s with no subscribers: 859 frames per camera at 29.86–30.03/s, with **0–3 lost
frames**, all in a single startup gap. In steady state `seq` advances exactly one per trigger edge
(351 of 353 intervals in a shorter run), so it is a valid index for a frame-time fit — but the
startup transient is not, so drop the leading rows or reject any row where `seq` disagrees with
`round(Δt / period)`.

⚠️ A subscriber can make it look far worse than it is: measuring with a Python node that decodes
all four streams showed 55–211 gaps per 30 s. That loss is in delivery, not capture. Judge capture
health from `frame_meta`/the CSV, not from what a consumer received.

## The IMU side

`bev_imu`'s `imu_node` publishes `sensor_msgs/Imu` on `/imu0` with `header.stamp` taken **at the
data-ready edge**, on the same `CLOCK_MONOTONIC` as the cameras. It talks to `/dev/spidev1.0` and
the GPIO character device directly, because the sample loop must not do anything that can stall
between the edge and the timestamp. Three board-specific parts are easy to get silently wrong and
are worth knowing before touching that file: the INT line is inverted with no pull-up, so the
sensor is configured **push-pull** and the assertion arrives as a **falling** edge; the chardev's
own event stamp is `CLOCK_REALTIME` and must be discarded (it is used only to *wait*); and the
kernel here is 4.9, so only the **v1** GPIO event ABI exists.

Two things the node reports and does **not** apply:

- **DLPF group delay** — the edge marks when the *filtered* sample was ready, and the gyro path lags
  the accel path by ~1.0 ms at every matched bandwidth. One correction cannot serve both, so both
  are logged and left to the consumer.
- **Δ** — logged as `UNMEASURED` until §3 of the retarget change measures it.

It needs `privileged` (device-cgroup access to `/dev/spidev1.0` and `/dev/gpiochip*`, plus
`CAP_SYS_NICE` for `SCHED_FIFO`); `docker compose run --rm imu` sets that up.

⚠️ **Anything that can pause the sample loop costs samples, not just jitter.** A stalled loop
misses data-ready edges outright — the data registers only ever hold the newest sample, so an edge
seen late is a sample gone. An early Python implementation of this node lost 29 samples in 18 s to
a single cyclic-GC pause (one 78 ms gap) against 0 for the same hardware read in C. That is why the
loop is C++, allocates nothing per sample, and does the timestamp before anything else.

Reference figures for the same hardware, `j106-imu-read.py` at 200 Hz under `chrt -f 80`:
interval sd 15.7 µs, max 5137 µs, 0 dropped in 3600 samples.

**4. Δ is stated, with provenance — or marked unmeasured.** After the fit, exactly one unknown is
left: the offset between the camera timebase and the IMU timebase. It is **one constant for the whole
rig**, not one per camera, because a shared trigger edge leaves no per-camera component. Two routes:

- **Estimate it** with Kalibr from a recording (`td`), ~0.1–1 ms — the route this project takes
  (see the `retarget-vo-to-imx296-rig` change, §3).
- **Measure it** by echoing the trigger into a GPIO and comparing against the fitted frame time,
  ~1 µs — documented in the J106 repo's `hw-trigger/WIRING.md` §4.4. ⚠ The echo must be timestamped
  through **the same userspace wake path as the IMU**, so the ~50 µs wake latency is common mode and
  cancels; taking the kernel IRQ stamp instead measures the wrong quantity and leaves Δ ~50 µs short.

Until one of them happens, Δ is recorded as **unmeasured** rather than as zero.

## Things that are already known to bite

- **NTP slews `CLOCK_MONOTONIC`** (+48.45 ppm measured here; only `CLOCK_MONOTONIC_RAW` is free of
  it). It is common mode between camera and IMU so it does not harm Δ — but it corrupts any
  statement about *rate*, and the servo makes a frame-time fit residual wander (30.9 µs with
  `systemd-timesyncd` running vs 8.4 µs without). Stop it for long calibration recordings.
- **The MPU-9250's DLPF group delay differs between gyro and accel** — the gyro lags the accel by
  ≈1.0 ms at every matched bandwidth. A front end that treats one timestamp as covering both
  inherits that error.
- **The IMU must be stamped at its data-ready edge**, not at the SPI read. Waking userspace on the
  edge costs a median 50 µs (bias, absorbed by Δ) with a MAD of 2.8 µs (the real limit, and the same
  order as the camera side's 1.5 µs jitter). Run the reader `SCHED_FIFO`.
- **`FSYNC` is not available** on the J106 (pin not brought out) and Tegra GTE hardware GPIO
  timestamping is Xavier-only — so waking on the edge is the best this board allows.

