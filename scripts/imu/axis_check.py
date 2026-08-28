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


def quiet_windows(acc, rate, min_s=1.5, tol=0.12):
    """Stretches where the rig is holding still, and the gravity vector in each.

    A static pose is a far better measurement than a translation: gravity is 10.4 m/s^2
    of noise-free signal, where a hand-pushed slide gives ~1 m/s^2 against the same
    noise. Tilt the rig into a pose, hold it, and the axis that takes gravity IS the
    axis pointing up - no thresholds, no sign guessing from a first excursion.
    """
    n = int(min_s * rate)
    mag = [math.sqrt(sum(c * c for c in a)) for a in acc]
    quiet = []
    i = 0
    while i + n <= len(acc):
        if statistics.pstdev(mag[i:i + n]) < tol:
            j = i + n
            while j < len(acc) and abs(mag[j] - mag[i]) < 4 * tol:
                j += 1
            quiet.append((i, j))
            i = j
        else:
            i += max(1, n // 8)
    out = []
    for i, j in quiet:
        seg = acc[i:j]
        g = [statistics.mean(s[k] for s in seg) for k in range(3)]
        out.append((i, j, g))
    return out


def report_static(t, acc, rate):
    poses = quiet_windows(acc, rate)
    print("found %d static pose(s)" % len(poses))
    print("\n---- gravity per pose ------------------------------------------------")
    for k, (i, j, g) in enumerate(poses):
        mag = math.sqrt(sum(v * v for v in g))
        axis = max(range(3), key=lambda m: abs(g[m]))
        # The measured vector is specific force: it points UP.
        up = "%s%s" % ("+" if g[axis] > 0 else "-", AXES[axis])
        off = math.degrees(math.acos(min(1.0, abs(g[axis]) / mag)))
        print("  pose %d  t=%5.1f-%5.1f s   g=(%+6.2f,%+6.2f,%+6.2f)  |g|=%5.2f   "
              "up = IMU %s   (%.1f deg off that axis)"
              % (k + 1, t[i] - t[0], t[j - 1] - t[0], g[0], g[1], g[2], mag, up, off))
    print("\nIn each pose the axis carrying gravity is the one pointing UP. Tilt the rig "
          "nose-up and that axis is FORWARD; roll it left-side-up and it is LEFT.")
    return poses


def solve_rotation(poses, labels):
    """Build the rig->IMU rotation from held poses, using the LEVEL pose as reference.

    A pose does not have to be a clean 90 degree tilt, and in practice never is. In the
    level pose the measured gravity direction IS rig +z in IMU coordinates. Tilt the
    nose up by any angle and the measured direction gains a component along rig +x; the
    part of it perpendicular to the level direction therefore points along rig +x, no
    matter how far it was tilted. Same for left-side-up and rig +y.

    So the tilt angle never has to be known or controlled - only held still. What the
    angle does control is precision: a 10 degree tilt projects a short vector and its
    direction is noisier than a 40 degree one.
    """
    import numpy as np
    groups = {}
    for (i, j, g), label in zip(poses, labels):
        v = np.array(g, dtype=float)
        groups.setdefault(label, []).append(v / np.linalg.norm(v))
    for k in ("level", "nose_up", "left_up"):
        if k not in groups:
            print("\nneed level / nose_up / left_up poses to solve a rotation; got %s"
                  % sorted(groups))
            return None

    e3 = np.mean(groups["level"], axis=0)
    e3 /= np.linalg.norm(e3)

    def perpendicular(vs, name):
        outs = []
        for v in vs:
            p = v - np.dot(v, e3) * e3      # the part the tilt added
            n = np.linalg.norm(p)
            tilt = math.degrees(math.asin(min(1.0, n)))
            if n < 0.05:                     # under ~3 deg the direction is mostly noise
                print("  %s pose tilted only %.1f deg - too little to trust, skipped"
                      % (name, tilt))
                continue
            outs.append(p / n)
        return outs

    print("\n---- rig -> IMU rotation --------------------------------------------")
    e1s, e2s = perpendicular(groups["nose_up"], "nose_up"), perpendicular(groups["left_up"], "left_up")
    if not e1s or not e2s:
        print("  not enough tilt to solve")
        return None
    if len(e1s) > 1:
        spread = max(math.degrees(math.acos(max(-1, min(1, float(np.dot(a, b))))))
                     for a in e1s for b in e1s)
        print("  nose_up repeats agree to %.1f deg" % spread)
    e1 = np.mean(e1s, axis=0); e1 /= np.linalg.norm(e1)
    e2 = np.mean(e2s, axis=0); e2 /= np.linalg.norm(e2)

    ang = lambda a, b: math.degrees(math.acos(max(-1, min(1, float(np.dot(a, b))))))
    M = np.column_stack([e1, e2, e3])
    print("  measured axis angles: front^left %.1f, left^up %.1f, front^up %.1f  (90 = perfect)"
          % (ang(e1, e2), ang(e2, e3), ang(e1, e3)))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:                 # never accept a reflection
        U[:, -1] *= -1
        R = U @ Vt
    moved = math.degrees(math.acos(max(-1, min(1, (np.trace(R.T @ M) - 1) / 2))))
    print("  orthonormalisation moved it %.2f deg  (det = %+.3f)" % (moved, np.linalg.det(R)))
    np.set_printoptions(precision=3, suppress=True)
    print("  R_imu_from_rig =\n%s" % np.array2string(R, prefix="    "))
    for k, name in enumerate(("front", "left", "up")):
        c = R[:, k]
        ax = int(np.argmax(np.abs(c)))
        off = math.degrees(math.acos(min(1.0, abs(c[ax]))))
        print("  rig %-5s -> IMU %s%s   (%.1f deg off that axis)"
              % (name, "+" if c[ax] > 0 else "-", AXES[ax], off))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--static", action="store_true",
                    help="analyse held poses (gravity) instead of motion bursts")
    ap.add_argument("--poses", nargs="*", default=[],
                    help="label each detected pose in order: level | nose_up | left_up | other")
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

    if a.static:
        poses = report_static(t, acc, rate)
        if a.poses:
            solve_rotation(poses, a.poses)
        return

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
