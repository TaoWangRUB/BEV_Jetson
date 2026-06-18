#!/usr/bin/env python3
"""Offline fisheye (Kannala-Brandt) intrinsic calibration from a directory of
checkerboard frames (captured with scripts/calib/capture_frames.sh).

The capture is "blind" (no live feedback), so this tool does the work afterward:
  - auto-detects the checkerboard grid (so you don't have to get inner-corner
    counts exactly right),
  - reports detection rate AND a 3x3 coverage map (edge/corner coverage is what
    matters for a 160 deg fisheye),
  - calibrates with the fisheye model cuVSLAM / VINS use and writes the
    KANNALA_BRANDT yaml + annotated previews.

File-based — needs only OpenCV + numpy (no GStreamer):
    pip install opencv-python-headless numpy
    ./fisheye_calib.py <frames-dir> --id N [--board 11x9] [--square 30]

--board is the number of SQUARES (e.g. 11x9); inner corners = squares-1. If the
guess fails, nearby grids are tried automatically.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("frames", help="directory of *.jpg/*.png checkerboard frames")
ap.add_argument("--id", type=int, default=0, help="camera id (output naming)")
ap.add_argument("--board", default="11x9", help="checkerboard SQUARES, e.g. 11x9")
ap.add_argument("--square", type=float, default=30.0, help="square size (mm)")
ap.add_argument("--min-views", type=int, default=10)
ap.add_argument("--out", default="config/calib")
args = ap.parse_args()

sx, sy = (int(v) for v in args.board.lower().split("x"))
# inner corners = squares - 1; build candidate grids around the guess (and swap)
guess = (sx - 1, sy - 1)
CANDIDATES = []
for c in [guess, (guess[1], guess[0]), (sx, sy), (sx - 2, sy - 2)]:
    if c[0] > 2 and c[1] > 2 and c not in CANDIDATES:
        CANDIDATES.append(c)

FIND = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) +
               glob.glob(os.path.join(args.frames, "*.png")))
if not files:
    sys.exit(f"no frames in {args.frames}")
grays = [(f, cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)) for f in files
         if cv2.imread(f) is not None]
size = grays[0][1].shape[::-1]

# --- auto-detect the grid on a sample ---------------------------------------
sample = grays[:max(8, len(grays) // 4)]
best, best_n = None, 0
for cand in CANDIDATES:
    n = sum(cv2.findChessboardCorners(g, cand, FIND)[0] for _, g in sample)
    print(f"  grid {cand[0]}x{cand[1]} inner corners -> {n}/{len(sample)} sample hits")
    if n > best_n:
        best, best_n = cand, n
if best is None or best_n == 0:
    sys.exit("No checkerboard found with any candidate grid. Check --board and "
             "that the board is fully visible / in focus, then recapture.")
BOARD = best
print(f"using grid {BOARD[0]}x{BOARD[1]} inner corners ({BOARD[0]+1}x{BOARD[1]+1} squares)")

objp = np.zeros((1, BOARD[0] * BOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
objp *= args.square

os.makedirs(args.out, exist_ok=True)
prevdir = os.path.join(args.out, f"cam{args.id}_preview")
os.makedirs(prevdir, exist_ok=True)

objpoints, imgpoints, used = [], [], []
cov = np.zeros((3, 3), int)  # 3x3 coverage map of board centers
for f, gray in grays:
    ok, corners = cv2.findChessboardCorners(gray, BOARD, FIND)
    if not ok:
        continue
    corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), SUBPIX)
    objpoints.append(objp.copy())
    imgpoints.append(corners)
    used.append(f)
    cx, cy = corners[:, 0, 0].mean(), corners[:, 0, 1].mean()
    cov[min(2, int(cy / size[1] * 3)), min(2, int(cx / size[0] * 3))] += 1
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, BOARD, corners, ok)
    cv2.imwrite(os.path.join(prevdir, "det_" + os.path.basename(f)), vis)

print(f"\ndetected board in {len(used)}/{len(grays)} frames")
print("coverage (3x3 image regions, board-center counts):")
for row in cov:
    print("   " + " ".join(f"{v:3d}" for v in row))
empty = int((cov == 0).sum())
if empty:
    print(f"WARNING: {empty}/9 image regions have NO views — for a 160 deg fisheye, "
          "recapture hitting the empty regions (esp. edges/corners).")
if len(used) < args.min_views:
    sys.exit(f"only {len(used)} good views (< {args.min_views}). Recapture with more "
             "coverage; capture more frames and move the board slowly.")

K = np.zeros((3, 3))
D = np.zeros((4, 1))
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, size, K, D, flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
print(f"\nRMS reprojection error = {rms:.4f} px   (good < ~0.5)")
print("K =\n", K, "\nD =", D.ravel())

for f in used[:3]:
    img = cv2.imread(f)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)
    cv2.imwrite(os.path.join(prevdir, "undist_" + os.path.basename(f)),
                cv2.remap(img, m1, m2, cv2.INTER_LINEAR))

np.savez(os.path.join(args.out, f"cam{args.id}.npz"), K=K, D=D, size=size, rms=rms)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
k = D.ravel()
with open(os.path.join(args.out, f"cam{args.id}.yaml"), "w") as fp:
    fp.write(f"""%YAML:1.0
---
model_type: KANNALA_BRANDT
camera_name: cam{args.id}
image_width: {size[0]}
image_height: {size[1]}
distortion_parameters:
   k2: {k[0]:.10f}
   k3: {k[1]:.10f}
   k4: {k[2]:.10f}
   k5: {k[3]:.10f}
projection_parameters:
   mu: {fx:.10f}
   mv: {fy:.10f}
   u0: {cx:.10f}
   v0: {cy:.10f}
""")
print(f"\nsaved {args.out}/cam{args.id}.yaml + cam{args.id}.npz + previews in {prevdir}/")
