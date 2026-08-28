#!/usr/bin/env python3
"""Work out how the MPU-9250's axes sit relative to the rig, from a motion recording.

Nothing documents this: the J106 README does not state the IMU's mounting, and the
datasheet gives axes relative to package pin 1, which says nothing about how it was
soldered. Gravity alone pins only the vertical axis; which way +X and +Y point
relative to the rig's forward direction needs the rig to be MOVED along each one.

Record with the J106 reader, then analyse here:

    # on the board
    sudo chrt -f 80 python3 j106-imu-read.py -n 16000 -o /tmp/axis.csv
    #   hold still ~5 s, then: front/back, pause, left/right, pause, up/down
    # on the host
    python3 scripts/imu/axis_check.py /tmp/axis.csv

Each motion is found as a burst of acceleration above the still baseline, the axis
that moved is the one with the largest spread, and the sign comes from the FIRST
excursion — which is why the recording has to start each pair in the named direction.

The result is the reference to check Kalibr's camera->IMU rotation against. A gross
axis swap in a calibration is otherwise easy to accept, because the numbers still
look plausible.
"""
import argparse
import math
import statistics
import sys

AXES = "xyz"


def load(path):
    t, acc, gyr = [], [], []
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 7:
            continue
        try:                       # the reader writes a bare column-name row after its '#' block
            int(f[0])
        except ValueError:
            continue
        t.append(int(f[0]) / 1e9)
        acc.append((float(f[1]), float(f[2]), float(f[3])))
        gyr.append((float(f[4]), float(f[5]), float(f[6])))
    return t, acc, gyr


def smooth(v, n):
    out, acc = [], 0.0
    for i, x in enumerate(v):
        acc += x
        if i >= n:
            acc -= v[i - n]
        out.append(acc / min(i + 1, n))
    return out


def find_gravity(acc, rate, window_s=2.0):
    """Quietest window in the recording is the best gravity estimate.

    Not simply "the first N samples": the recording may start while the rig is still
    being picked up, and a gravity vector taken during motion poisons every residual
    that follows.
    """
    n = int(window_s * rate)
    mag = [math.sqrt(sum(c * c for c in a)) for a in acc]
    best, best_var = 0, None
    for i in range(0, len(acc) - n, max(1, n // 4)):
        var = statistics.pvariance(mag[i:i + n])
        if best_var is None or var < best_var:
            best, best_var = i, var
    seg = acc[best:best + n]
    g = [statistics.mean(s[i] for s in seg) for i in range(3)]
    return g, best / rate, math.sqrt(best_var)


def bursts(energy, rate, thresh, min_len_s, merge_gap_s):
    """Contiguous stretches above threshold, merged across short gaps.

    The gap merge is what makes "front then back" one motion rather than two: the
    rig is momentarily still between the two pushes, and they describe one axis.
    """
    on = [e > thresh for e in energy]
    spans, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            spans.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) / rate < merge_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if (s[1] - s[0]) / rate >= min_len_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--labels", nargs="*", default=["front", "left", "up"],
                    help="what the motions were, in the order they were performed")
    ap.add_argument("--thresh", type=float, default=1.5,
                    help="m/s^2 above the still baseline that counts as a deliberate move")
    a = ap.parse_args()

    t, acc, gyr = load(a.csv)
    if len(t) < 100:
        sys.exit("not enough samples in %s" % a.csv)
    rate = (len(t) - 1) / (t[-1] - t[0])
    print("%d samples, %.1f s, %.1f Hz" % (len(t), t[-1] - t[0], rate))

    g, g_at, g_noise = find_gravity(acc, rate)
    print("gravity = (%+.2f, %+.2f, %+.2f) m/s^2, |g| = %.2f   (quietest window at t=%.1f s, "
          "residual noise %.3f)" % (g[0], g[1], g[2], math.sqrt(sum(v * v for v in g)),
                                    g_at, g_noise))
    # An accelerometer at rest measures SPECIFIC FORCE, which points UP (it is the
    # reaction to gravity), so the measured vector names the up direction, not down.
    gdir = max(range(3), key=lambda i: abs(g[i]))
    print("  -> rig UP is IMU %s%s  (so IMU %s%s points down)\n"
          % ("+" if g[gdir] > 0 else "-", AXES[gdir],
             "-" if g[gdir] > 0 else "+", AXES[gdir]))

    resid = [[a_[i] - g[i] for a_ in acc] for i in range(3)]
    energy = smooth([math.sqrt(sum(resid[i][k] ** 2 for i in range(3)))
                     for k in range(len(acc))], int(0.1 * rate))

    spans = bursts(energy, rate, a.thresh, 0.4, 2.0)
    print("found %d motion burst(s)" % len(spans))

    results = []
    for k, (i0, i1) in enumerate(spans):
        seg = [[resid[i][j] for j in range(i0, i1)] for i in range(3)]
        spread = [statistics.pstdev(s) for s in seg]
        axis = spread.index(max(spread))
        sign = 0
        for j in range(len(seg[axis])):
            if abs(seg[axis][j]) > 2.0:
                sign = 1 if seg[axis][j] > 0 else -1
                break
        # Rotation contaminates a translation test: tilting swings 10.4 m/s^2 of gravity
        # into the horizontal axes, so flag any burst with real angular rate.
        rot = max(statistics.pstdev([gyr[j][i] for j in range(i0, i1)]) for i in range(3))
        label = a.labels[k] if k < len(a.labels) else "motion %d" % (k + 1)
        results.append((label, axis, sign, spread, rot, (t[i0] - t[0], t[i1] - t[0])))

    print("\n---- axis map --------------------------------------------------------")
    for label, axis, sign, spread, rot, (ts, te) in results:
        arrow = "+" if sign > 0 else ("-" if sign < 0 else "?")
        warn = "   <-- ROTATION %.2f rad/s: may be tilt, not translation" % rot if rot > 0.5 else ""
        print("  rig %-6s -> IMU %s%s   t=%5.1f-%5.1f s  spread x=%.2f y=%.2f z=%.2f%s"
              % (label, arrow, AXES[axis], ts, te, spread[0], spread[1], spread[2], warn))
    print("  rig up     -> IMU %s%s   (from gravity, independent of the motions above)"
          % ("+" if g[gdir] > 0 else "-", AXES[gdir]))

    picked = [r[1] for r in results]
    if len(set(picked)) != len(picked):
        print("\nTwo motions picked the SAME axis — one of them tilted the rig or was too "
              "gentle. Re-record that one; the axes must come out distinct.")


if __name__ == "__main__":
    main()
