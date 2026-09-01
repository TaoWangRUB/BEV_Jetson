#!/usr/bin/env python3
"""Probe what the trigger generator is ACTUALLY doing to the sensors, from raw V4L2.

Two questions that a generator swap or a room change makes live again, and that no
label answers:

  --sweep     Does the commanded exposure equal the ASSERTED pulse, or its complement?
              Under Fast Trigger the pulse width IS the exposure, so a polarity flip
              would mean the true exposure is (period - pulse) and every timestamp is
              biased by half the difference. Fix the scene, step the exposure, watch
              the brightness: RISING means commanded == asserted. This mattered on
              2026-08-31, when the F401 reported active_low where the H7 had reported
              active_high, and the label alone could not say whether the MEANING had
              inverted too.

  --flicker   Is there mains flicker to worry about? At 30 fps a 50 Hz mains ripple
              (100 Hz) aliases to 10 Hz -> a 3-frame beat in the per-frame mean. A
              60 Hz ripple (120 Hz) aliases to ~0 Hz and is INVISIBLE to this test,
              so a flat result rules out 50 Hz and proves nothing about 60. Run it at
              a SHORT exposure: a long one integrates the ripple away, which is the
              whole point of the measurement.

Deliberately raw V4L2 and no ROS: this runs on the bare board with nothing else up,
and it must not depend on the ISP or on AE settings. For brightness stability through
the ISP over time (AE gain-hunting), use luma_stability.py next door instead - that is
a different question and a different path.

    ssh tx2-eth
    python3 trigger_probe.py --sweep 5000 15000 30000
    python3 trigger_probe.py --flicker --exposure 2000 --frames 60

The exposure in force when the script starts is RESTORED on exit, including on Ctrl-C:
leaving a rig at a probe's exposure silently biases every subsequent stamp.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

TRIGCTL = os.environ.get("TRIGCTL", "/home/nvidia/tools/j106-trigctl.py")
TRIG_PORT = os.environ.get("TRIG_PORT", "/dev/ttyACM0")   # F401 (USB CDC); the H7 was /dev/ttyTHS1
W, H = 1456, 1088


def trigctl(*args):
    """One retry: ModemManager probes a new ACM port with AT commands and can hold it."""
    # NB: the board is Python 3.6 - no capture_output=, no text=.
    cmd = ["python3", TRIGCTL, "--port", TRIG_PORT, *args]
    for _ in range(2):
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        if r.returncode == 0:
            return r.stdout
    return r.stdout + r.stderr


def read_exposure_us():
    """The MEASURED pulse width, not the commanded value - they differ by ~14 us."""
    m = re.search(r"ch1_exposure_us=(\d+)\s+pulse_ns=(\d+)", trigctl("status"))
    if not m:
        sys.exit(f"cannot read trigger status on {TRIG_PORT} — is the generator attached?")
    return int(m.group(1)), (int(m.group(2)) + 500) // 1000


def grab(dev, count):
    """Return the LAST frame as float32, de-strided.

    Tegra pads the line stride (3072 bytes for 1456 px of 16-bit, not 2912). Decoding
    at width*2 gives a plausible-looking image of pure diagonal streaks, so the stride
    is derived from the buffer rather than assumed.
    """
    with tempfile.NamedTemporaryFile(suffix=".raw") as tf:
        subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{dev}",
             "--set-fmt-video", f"width={W},height={H},pixelformat=BG10",
             "--stream-mmap", f"--stream-count={count}", "--stream-to", tf.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buf = np.fromfile(tf.name, dtype=np.uint8)
    if buf.size < H * W * 2:
        sys.exit(
            f"/dev/video{dev} returned {buf.size} bytes. In order of likelihood:\n"
            f"  1. ANOTHER CONSUMER HOLDS THE CAMERA. Argus serves one at a time, so a running\n"
            f"     csi_sender.sh / calib_sender.sh / ROS capture blocks raw V4L2 entirely.\n"
            f"     Check with: ps -eo args | grep -c '[g]st-launch'\n"
            f"  2. trigger_mode=1 but the generator is stopped — no edges, no frames.\n"
            f"  3. the camera needs a reset (README 4.3).")
    stride = buf.size // count // H
    if stride < W * 2:
        sys.exit(f"unexpected geometry: {buf.size} bytes / {count} frames / {H} rows = stride {stride}")
    frame = buf[-stride * H:].reshape(H, stride)[:, :W * 2].copy()
    return frame.view(np.uint16).astype(np.float32)


def stats(f):
    return dict(min=float(f.min()), mean=float(f.mean()),
                p99=float(np.percentile(f, 99)), max=float(f.max()),
                clipped=100.0 * float((f >= 16300).mean()))


def sweep(devs, exposures, count):
    rows = []
    for e in exposures:
        trigctl("exposure", str(e))
        time.sleep(1)
        _, pulse = read_exposure_us()
        for d in devs:
            s = stats(grab(d, count))
            rows.append((e, pulse, d, s))
            print(f"  cmd={e:6d} us  pulse={pulse:6d} us  video{d}  "
                  f"min={s['min']:6.0f}  mean={s['mean']:8.1f}  p99={s['p99']:7.0f}  "
                  f"clipped={s['clipped']:5.2f}%")
    for d in devs:
        r = [x for x in rows if x[2] == d]
        if len(r) < 2:
            continue
        slopes = [(b[3]["mean"] - a[3]["mean"]) / (b[0] - a[0]) for a, b in zip(r, r[1:])]
        print(f"\nvideo{d}: incremental slope " +
              ", ".join(f"{s:.4f}" for s in slopes) + " counts/us")
        if all(s > 0 for s in slopes):
            print("  -> brightness RISES with commanded exposure: commanded == ASSERTED width.")
            print("     The complement hypothesis (exposure = period - pulse) predicts the")
            print("     opposite and is therefore excluded, whatever the polarity label says.")
        else:
            print("  -> brightness FALLS with commanded exposure. The asserted window is the")
            print("     COMPLEMENT of the commanded value: every exposure-midpoint stamp is")
            print("     wrong by (period - 2*pulse)/2. Fix polarity before recording anything.")


def flicker(dev, frames, fps):
    f = []
    with tempfile.NamedTemporaryFile(suffix=".raw") as tf:
        subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{dev}",
             "--set-fmt-video", f"width={W},height={H},pixelformat=BG10",
             "--stream-mmap", f"--stream-count={frames}", "--stream-to", tf.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buf = np.fromfile(tf.name, dtype=np.uint8)
    n = frames
    if buf.size < n * H * W * 2:
        sys.exit(f"got {buf.size} bytes for {n} frames — another consumer probably holds the "
                 f"camera (Argus serves one at a time); stop csi_sender.sh / the ROS capture.")
    stride = buf.size // n // H
    fsz = stride * H
    for i in range(n):
        img = buf[i * fsz:(i + 1) * fsz].reshape(H, stride)[:, :W * 2].copy().view(np.uint16)
        f.append(float(img.mean()))
    m = np.array(f)
    # v4l2 hands back a few startup frames before the stream settles; they read as a step
    # and would dominate the spectrum.
    m = m[5:]
    m = m - m.mean()
    print(f"  frames={len(m)} (5 startup dropped)  sd={m.std():.2f}  p2p={m.ptp():.2f} counts")
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    freq = np.fft.rfftfreq(len(m), d=1.0 / fps)
    for i in np.argsort(spec[1:])[::-1][:3] + 1:
        print(f"    peak {freq[i]:5.2f} Hz  amp {spec[i]:6.1f}")
    beat = spec[(freq > 8) & (freq < 12)]
    if beat.size and beat.max() > 4 * np.median(spec[1:]):
        print("  -> a ~10 Hz beat is present: 50 Hz mains flicker (100 Hz ripple aliased by")
        print("     30 fps sampling). Exposures that are whole multiples of a 10 ms half-cycle")
        print("     (10, 20, 30 ms) integrate it away.")
    else:
        print("  -> no ~10 Hz beat: 50 Hz mains flicker is RULED OUT.")
        print("     NOT a proof of no flicker: 120 Hz (60 Hz mains) aliases to ~0 Hz at 30 fps")
        print("     and is invisible here. Re-run under artificial light before concluding.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", nargs="*", type=int, metavar="US",
                    help="exposures to step through (default 5000 15000 30000)")
    ap.add_argument("--flicker", action="store_true")
    ap.add_argument("--exposure", type=int, default=2000,
                    help="exposure for --flicker; keep it SHORT or the ripple integrates away")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=30.0, help="trigger rate, for the alias maths")
    ap.add_argument("--devs", type=int, nargs="+", default=[0],
                    help="/dev/videoN indices (0..3 = ports c,d,e,f on this rig)")
    ap.add_argument("--count", type=int, default=8, help="frames to grab per measurement")
    a = ap.parse_args()

    mode = None
    if os.path.exists("/sys/module/imx296/parameters/trigger_mode"):
        mode = open("/sys/module/imx296/parameters/trigger_mode").read().strip()
    if mode != "1":
        print(f"⚠ trigger_mode={mode} — the sensors are NOT following the trigger, so nothing "
              f"below measures the trigger. Set it to 1 first.", file=sys.stderr)

    cmd0, pulse0 = read_exposure_us()
    print(f"trigger: commanded {cmd0} us, measured pulse {pulse0} us  (restored on exit)\n")
    try:
        if a.flicker:
            trigctl("exposure", str(a.exposure))
            time.sleep(1)
            print(f"flicker probe at {a.exposure} us, {a.frames} frames, video{a.devs[0]}:")
            flicker(a.devs[0], a.frames, a.fps)
        else:
            ex = a.sweep if a.sweep else [5000, 15000, 30000]
            print(f"exposure sweep on video{a.devs}:")
            sweep(a.devs, ex, a.count)
    finally:
        trigctl("exposure", str(cmd0))
        print(f"\nrestored exposure to {cmd0} us")


if __name__ == "__main__":
    main()
