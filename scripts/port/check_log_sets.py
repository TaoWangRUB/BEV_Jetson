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
        if got == len(others):
            complete += 1

    n = len(frames[anchor])
    print("anchor %s: %d edges" % (anchor, n))
    print("COMPLETE 4-camera sets: %d of %d  (%.1f%%)" % (complete, n, 100.0 * complete / n))
    for c in others:
        print("  %s missing from %d edges (%.1f%%)" % (c, missing_per_cam[c],
                                                       100.0 * missing_per_cam[c] / n))
    span = (frames[anchor][-1] - frames[anchor][0]) / 1e9
    print("effective SET rate: %.2f Hz over %.1f s" % (complete / span, span))
    print("(raw per-camera fps is not the number that matters here - a log whose sets are")
    print(" broken cannot be replayed as a synchronised rig, whatever its frame count.)")


if __name__ == "__main__":
    main()
