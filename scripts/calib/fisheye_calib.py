#!/usr/bin/env python3
"""Offline fisheye (Kannala-Brandt) intrinsic calibration from a directory of
checkerboard frames (captured with scripts/calib/capture_frames.sh).

Outputs the K + D that cuVSLAM / VINS use (KANNALA_BRANDT), a VINS-style yaml,
and annotated detection + undistortion previews for a visual sanity check.

File-based — needs only OpenCV + numpy (no GStreamer):
    pip install opencv-python-headless numpy
    ./fisheye_calib.py <frames-dir> --id N --cols 10 --rows 8 --square 30

`--cols`/`--rows` are the INNER corner counts of the checkerboard.
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
ap.add_argument("--cols", type=int, default=10, help="inner corners per row")
ap.add_argument("--rows", type=int, default=8, help="inner corners per column")
ap.add_argument("--square", type=float, default=30.0, help="square size (mm)")
ap.add_argument("--out", default="config/calib")
args = ap.parse_args()
BOARD = (args.cols, args.rows)

objp = np.zeros((1, BOARD[0] * BOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
objp *= args.square

files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) +
               glob.glob(os.path.join(args.frames, "*.png")))
if not files:
    sys.exit(f"no frames in {args.frames}")

find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
os.makedirs(args.out, exist_ok=True)
prevdir = os.path.join(args.out, f"cam{args.id}_preview")
os.makedirs(prevdir, exist_ok=True)

objpoints, imgpoints, used, size = [], [], [], None
for f in files:
    img = cv2.imread(f)
    if img is None:
        continue
    size = img.shape[:2][::-1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, BOARD, find_flags)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used.append(f)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, BOARD, corners, ok)
        cv2.imwrite(os.path.join(prevdir, "det_" + os.path.basename(f)), vis)

print(f"detected checkerboard in {len(used)}/{len(files)} frames")
if len(used) < 10:
    sys.exit("need >= 10 good views — recapture with better full-FOV coverage")

K = np.zeros((3, 3))
D = np.zeros((4, 1))
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, size, K, D, flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
print(f"RMS reprojection error = {rms:.4f} px   (good < ~0.5)")
print("K =\n", K, "\nD =", D.ravel())

# undistortion sanity previews
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
print(f"saved {args.out}/cam{args.id}.yaml + cam{args.id}.npz + previews in {prevdir}/")
