#!/usr/bin/env python3
"""Choose the frames worth calibrating from everything that was recorded.

Recording at full rate and selecting afterwards beats decimating live, because live
decimation keeps every Nth frame whether or not it is any good. Offline there is
evidence to choose on:

  sharpness   a 5 ms exposure while the target is being swept blurs some frames. A
              blurred tag still detects, but its corners land a pixel or two off and
              the solver has no way to know - it just fits worse.
  novelty     fifty frames of the target in the same place say little more than one.
              What a fisheye calibration needs is the PERIPHERY, so frames are picked
              greedily for the coverage they ADD rather than for being sharp somewhere
              already well covered.

Solvers also need this: tartancalib on nine thousand frames is not a long run, it is
an abandoned one. A few hundred well-spread frames is the working size.

  python3 scripts/calib/select_frames.py datasets/calib/CAM_A/cam1 --out selected/ -n 150
"""
import argparse
import os
import shutil

import cv2
import numpy as np

GRID = 8

_p = cv2.aruco.DetectorParameters()
_p.markerBorderBits = 2
_p.adaptiveThreshWinSizeStep = 1
_p.adaptiveThreshWinSizeMin = 3
_det = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), _p)


def score(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    corners, ids, _ = _det.detectMarkers(img)
    if ids is None or len(ids) < 4:
        return None
    cells = set()
    xs, ys = [], []
    for c in corners:
        pts = c[0]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        xs.append(cx); ys.append(cy)
        cells.add((min(GRID - 1, int(cy / img.shape[0] * GRID)),
                   min(GRID - 1, int(cx / img.shape[1] * GRID))))
    # Sharpness measured on the target's own bounding box, not the whole frame: a sharp
    # background says nothing about the tags, and on a fisheye the background is usually
    # a different depth entirely.
    x0, x1 = int(max(0, min(xs) - 40)), int(min(img.shape[1], max(xs) + 40))
    y0, y1 = int(max(0, min(ys) - 40)), int(min(img.shape[0], max(ys) + 40))
    roi = img[y0:y1, x0:x1]
    sharp = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size else 0.0
    return {"path": path, "ntags": len(ids), "cells": cells, "sharp": sharp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--count", type=int, default=150)
    ap.add_argument("--min-sharp", type=float, default=0.0,
                    help="drop frames below this Laplacian variance (0 = keep all)")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.indir) if f.endswith((".jpg", ".png")))
    print("scoring %d frames..." % len(files))
    scored = []
    for i, f in enumerate(files):
        s = score(os.path.join(a.indir, f))
        if s and s["sharp"] >= a.min_sharp:
            scored.append(s)
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, len(files)))
    if not scored:
        raise SystemExit("no frames had a detectable target")
    print("%d frames have >=4 tags (%.0f%%)" % (len(scored), 100 * len(scored) / len(files)))

    # Greedy: repeatedly take the frame adding the most uncovered cells, breaking ties on
    # sharpness. Once nothing new is left, fall back to the sharpest remaining, so the
    # tail of the selection is at least high quality rather than arbitrary.
    covered, picked = set(), []
    pool = list(scored)
    while pool and len(picked) < a.count:
        best = max(pool, key=lambda s: (len(s["cells"] - covered), s["sharp"]))
        picked.append(best)
        covered |= best["cells"]
        pool.remove(best)

    os.makedirs(a.out, exist_ok=True)
    for s in picked:
        shutil.copy2(s["path"], a.out)

    occ = np.zeros((GRID, GRID), int)
    for s in picked:
        for (r, c) in s["cells"]:
            occ[r, c] += 1
    print("\nselected %d frames -> %s" % (len(picked), a.out))
    print("mean tags %.1f, median sharpness %.0f"
          % (np.mean([s["ntags"] for s in picked]),
             np.median([s["sharp"] for s in picked])))
    print("\ncoverage of the selection (%dx%d):" % (GRID, GRID))
    for row in occ:
        print("  " + " ".join("%4d" % v for v in row))
    empty = int((occ == 0).sum())
    print("\n%d/%d cells empty%s" % (empty, GRID * GRID,
          "  <-- the target never went there; re-shoot rather than trust the edges"
          if empty else "  (full coverage)"))


if __name__ == "__main__":
    main()
