#!/usr/bin/env python3
"""Are the frames in a raw log still SETS? Frame count and fps do not answer that.

A hardware-triggered 4-camera rig only has value as sets: four frames from the same trigger
edge. If the cameras lose DIFFERENT frames - which is exactly what a per-camera writer queue
that drops on overflow will do - the totals can look fine while most edges are missing at
least one camera. Rate is the wrong metric; set completeness is the right one.

Matches every camera's frames by timestamp (all four should land within the measured ~66 us
of hardware skew) and reports how many edges have all four.

  check_log_sets.py <dir> [<dir> ...]      # one dir per target when the log is split
"""
import glob
import os
import sys

TOL_NS = 5_000_000          # 5 ms: far wider than the 66 us real skew, far under a 33 ms period

# How far above the measured still baseline |gyro| has to sit before the rig counts as
# moving, rad/s. 0.03 is ~1.7 deg/s: far above the MPU-9250's noise (the baseline measures
# ~0.024 including bias) and far below anything a walking operator produces (0.2-1.0).
IMU_STILL_MARGIN = 0.03


def motion_window(dirs):
    """When was the rig actually MOVING? -> (lo_ns, hi_ns, baseline, thresh, imu_t0, imu_t1).

    Set completeness says the log is intact; it says nothing about whether the rig did
    anything, and those are separately necessary. A stationary rig produces a flawless log
    that is worth nothing to VO, and reporting only "88.9 s, 99.9% complete" is how a
    recording with 46 s of motion in it came to be described as 90 s of data.

    lo_ns is None when the rig never moved. Returns None when there is no imu0.csv.
    """
    for d in dirs:
        p = os.path.join(d, "imu0.csv")
        if not os.path.exists(p):
            continue
        t, w = [], []
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            if len(f) < 7:      # the tail line of a SIGKILLed run is truncated
                continue
            try:
                t.append(int(f[0]))
                w.append((float(f[4]) ** 2 + float(f[5]) ** 2 + float(f[6]) ** 2) ** 0.5)
            except ValueError:
                continue
        if len(t) < 100:
            continue
        # 10th percentile, not the minimum: the still level includes gyro bias, and one quiet
        # sample would put the baseline below it and make the threshold too tight.
        base = sorted(w)[len(w) // 10]
        thr = base + IMU_STILL_MARGIN
        hot = [i for i, v in enumerate(w) if v > thr]
        if not hot:
            return (None, None, base, thr, t[0], t[-1])
        return (t[hot[0]], t[hot[-1]], base, thr, t[0], t[-1])
    return None


def report_imu(dirs, cam_t0, cam_t1):
    """Rate / continuity of imu0.csv. Returns True if a file was found."""
    for d in dirs:
        p = os.path.join(d, "imu0.csv")
        if not os.path.exists(p):
            continue
        t, seq = [], []
        n_bad = 0
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            if len(f) < 9:
                n_bad += 1
                continue
            try:
                t.append(int(f[0]))
                seq.append(int(f[8]))
            except ValueError:
                n_bad += 1
        if len(t) < 2:
            print("\nIMU: imu0.csv present but fewer than 2 valid rows")
            return True
        span = (t[-1] - t[0]) / 1e9
        rate = (len(t) - 1) / span
        dts = [(b - a) / 1e6 for a, b in zip(t, t[1:])]
        med = sorted(dts)[len(dts) // 2]
        gaps = sum(1 for d in dts if d > 1.5 * med)
        seq_gaps = sum(1 for a, b in zip(seq, seq[1:]) if b - a != 1)
        # Does the IMU bracket the cameras? log_rig starts it first and stops it last.
        lead = (cam_t0 - t[0]) / 1e9 if cam_t0 else None
        trail = (t[-1] - cam_t1) / 1e9 if cam_t1 else None
        print("\nIMU (imu0.csv):")
        print("  %d samples over %.2f s -> %.2f Hz  (median dt %.3f ms)"
              % (len(t), span, rate, med))
        print("  seq gaps: %d   timing gaps (>1.5 med): %d   truncated/bad rows: %d"
              % (seq_gaps, gaps, n_bad))
        if lead is not None:
            print("  brackets cameras: lead %+.2f s, trail %+.2f s%s"
                  % (lead, trail,
                     "" if lead > 0 and trail > 0 else
                     "   <- WARNING: IMU does not fully cover the camera span"))
        if abs(rate - 200) > 5 and abs(rate - 1000) > 50:
            print("  ^ rate is far from the usual 200 Hz — check IMU_RATE / DLPF config")
        return True
    return False


def report_range(dirs, cam_t0, cam_t1):
    """Rate / pulse continuity of range0.csv. Returns True if a file was found."""
    for d in dirs:
        p = os.path.join(d, "range0.csv")
        if not os.path.exists(p):
            continue
        t, cm, pulses = [], [], []
        divisor = None
        for line in open(p):
            line = line.strip()
            if line.startswith("#") and "divisor=" in line:
                try:
                    divisor = int(line.split("divisor=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
                continue
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            if len(f) < 3:
                continue
            try:
                t.append(int(f[0])); cm.append(int(f[1])); pulses.append(int(f[2]))
            except ValueError:
                continue
        print("\nRANGE (range0.csv):")
        if len(t) < 2:
            print("  present but fewer than 2 readings — sensor absent or auto stream never started")
            return True
        span = (t[-1] - t[0]) / 1e9
        rate = (len(t) - 1) / span
        dp = [b - a for a, b in zip(pulses, pulses[1:])]
        # Expected step is the configured divisor (default now 1 = every trigger edge).
        step = divisor if divisor and divisor > 0 else (sorted(dp)[len(dp) // 2] if dp else 1)
        skipped = sum(max(0, d // step - 1) for d in dp) if step else 0
        lead = (cam_t0 - t[0]) / 1e9 if cam_t0 else None
        trail = (t[-1] - cam_t1) / 1e9 if cam_t1 else None
        print("  %d readings over %.2f s -> %.2f Hz  (divisor=%s, median pulse step=%d)"
              % (len(t), span, rate, divisor if divisor is not None else "?",
                 sorted(dp)[len(dp) // 2] if dp else 0))
        print("  range_cm: min=%d median=%d max=%d"
              % (min(cm), sorted(cm)[len(cm) // 2], max(cm)))
        print("  missed pulse steps: %d%s"
              % (skipped, "   <- LOSSLESS vs trigger" if skipped == 0 else ""))
        if lead is not None:
            print("  vs cameras: first reading %+.2f s, last %+.2f s relative to cam span ends"
                  % (lead, trail))
        if divisor and divisor > 1:
            print("  ^ divisor=%d — raise to 1 for one reading per trigger edge "
                  "(RANGE_DIV=1)." % divisor)
        return True
    print("\nRANGE: no range0.csv — recording has no rangefinder channel")
    return False


def load(dirs):
    frames = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "cam*_index.csv"))):
            cam = os.path.basename(f)[: -len("_index.csv")]
            ts = [int(l.split(",")[0]) for l in open(f)
                  if l.strip() and not l.startswith(("#", "stamp"))]
            frames.setdefault(cam, []).extend(ts)
    for cam in frames:
        frames[cam].sort()
    return frames


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    frames = load(dirs)
    if not frames:
        sys.exit("no cam*_index.csv found")
    cams = sorted(frames)
    print("cameras: %s" % ", ".join("%s(%d)" % (c, len(frames[c])) for c in cams))

    # Anchor on the camera with the MOST frames, so a set that is missing the anchor camera
    # still counts as incomplete rather than silently vanishing.
    anchor = max(cams, key=lambda c: len(frames[c]))
    others = [c for c in cams if c != anchor]
    idx = {c: 0 for c in others}
    complete = 0
    missing_per_cam = {c: 0 for c in others}

    ok_edge = []

    for t in frames[anchor]:
        got = 0
        for c in others:
            lst = frames[c]
            i = idx[c]
            while i + 1 < len(lst) and abs(lst[i + 1] - t) <= abs(lst[i] - t):
                i += 1
            idx[c] = i
            if i < len(lst) and abs(lst[i] - t) <= TOL_NS:
                got += 1
            else:
                missing_per_cam[c] += 1
        ok_edge.append(got == len(others))
        if got == len(others):
            complete += 1

    n = len(frames[anchor])
    print("anchor %s: %d edges" % (anchor, n))
    print("COMPLETE 4-camera sets: %d of %d  (%.1f%%)" % (complete, n, 100.0 * complete / n))
    for c in others:
        print("  %s missing from %d edges (%.1f%%)" % (c, missing_per_cam[c],
                                                       100.0 * missing_per_cam[c] / n))

    # SEPARATE THE RAGGED ENDS FROM ACTUAL LOSS, because conflating them made a lossless
    # log read as 99.9% and there was no way to tell that from real dropping.
    #
    # A run is stopped while it is running, so the last trigger edge is served to some
    # cameras and not others - the capture loop reads them in order and simply stops. That
    # is ONE edge and it is not a symptom of anything. Interior loss is the opposite: it
    # means the rig or the storage could not keep up, and it is what has to be zero.
    # Measured 2026-09-05, 20 fps, 88 s: 1763 of 1764 sets, and the single incomplete one
    # was the final edge - interior 1763 of 1763. Reporting only the 99.9% would have
    # looked like the same thing as the 95.4% that 30 fps on one target produces.
    if not any(ok_edge):
        print("NO complete sets at all - the cameras are not producing matched edges.")
    else:
        lo = ok_edge.index(True)
        hi = len(ok_edge) - 1 - ok_edge[::-1].index(True)
        inner = ok_edge[lo:hi + 1]
        holes = [lo + k for k, v in enumerate(inner) if not v]
        print("  ragged boundary edges (run start/stop, not loss): %d" % (n - len(inner)))
        print("INTERIOR sets: %d of %d  (%.2f%%)%s"
              % (len(inner) - len(holes), len(inner),
                 100.0 * (len(inner) - len(holes)) / len(inner),
                 "   <- LOSSLESS" if not holes else ""))
        if holes:
            print("  interior edges missing a camera, by anchor index: %s%s"
                  % (holes[:20], " ..." if len(holes) > 20 else ""))
            print("  ^ THIS is real loss. Check write bandwidth before anything else.")

    span = (frames[anchor][-1] - frames[anchor][0]) / 1e9
    print("effective SET rate: %.2f Hz over %.1f s" % (complete / span, span))

    # AND SAY HOW MUCH OF IT THE RIG WAS MOVING FOR, when the IMU is in the log.
    #
    # This report used to end at the line above, and that is precisely how a recording was
    # described as "90 s of 20 fps data" when 42% of its frames were a rig sitting still
    # after the operator had stopped walking. Both numbers are needed: completeness says the
    # log is INTACT, the motion window says it is USEFUL, and neither implies the other.
    mw = motion_window(dirs)
    if mw:
        lo, hi, base, thr, imu_t0, imu_t1 = mw
        f0 = frames[anchor][0]
        print("\nMOTION (from imu0.csv; still baseline %.4f rad/s, threshold %.4f):" % (base, thr))
        print("  all times relative to the first camera frame; IMU covers %+.1f s to %+.1f s"
              % ((imu_t0 - f0) / 1e9, (imu_t1 - f0) / 1e9))
        if lo is None:
            print("  the gyro NEVER crossed the threshold - this log is a STATIONARY rig.")
        else:
            inside = sum(1 for t in frames[anchor] if lo <= t <= hi)
            before = sum(1 for t in frames[anchor] if t < lo)
            after = sum(1 for t in frames[anchor] if t > hi)
            rate = len(frames[anchor]) / span
            print("  motion runs %+.1f s to %+.1f s (%.1f s)"
                  % ((lo - f0) / 1e9, (hi - f0) / 1e9, (hi - lo) / 1e9))
            print("  frames before motion : %5d  (%4.1f%%)  %5.1f s stationary"
                  % (before, 100.0 * before / len(frames[anchor]), before / rate))
            print("  frames DURING motion : %5d  (%4.1f%%)  %5.1f s   <- the usable data"
                  % (inside, 100.0 * inside / len(frames[anchor]), inside / rate))
            print("  frames after motion  : %5d  (%4.1f%%)  %5.1f s stationary"
                  % (after, 100.0 * after / len(frames[anchor]), after / rate))
            if before + after > 0.2 * len(frames[anchor]):
                print("  ^ over a fifth of this log is a rig that is not moving. Quote the")
                print("    MOTION duration, not the recording duration, as what it contains.")
            print("  to convert only the moving part:")
            print("    raw_log_to_bag.py <dir> -o out.bag --motion")

    # AUX SENSORS. Completeness of the cameras does not imply the IMU or rangefinder
    # actually wrote anything — log_rig treats range as optional, and a failed start used
    # to leave a directory with perfect cam*.raw and no range0.csv at all.
    f0, f1 = frames[anchor][0], frames[anchor][-1]
    if not report_imu(dirs, f0, f1):
        print("\nIMU: no imu0.csv — recording has no IMU channel")
    report_range(dirs, f0, f1)

    print("\n(raw per-camera fps is not the number that matters here - a log whose sets are")
    print(" broken cannot be replayed as a synchronised rig, whatever its frame count.)")


if __name__ == "__main__":
    main()
