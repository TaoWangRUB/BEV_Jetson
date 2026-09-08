#!/usr/bin/env python3
"""Solve the extrinsic between ONE adjacent camera pair from simultaneous target views.

The method (calibration_pipeline.ipynb section 6, made runnable). Each camera solves the
AprilGrid pose independently with KALIBR'S OWN detector and its own fitted geometry; the
extrinsic is the transform between those two poses, averaged over every instant BOTH
cameras saw the board:

    T_b_a = inv(T_target_b) @ T_target_a          per simultaneous frame
    R     = SVD-projected mean of the rotations   (a mean of rotation matrices is not one)
    t     = mean of the translations

Two things here are load-bearing, both learned the hard way:

  * KALIBR'S DETECTOR, not a hand-rolled board model. Reimplementing the correspondence
    plateaued at 11 px residual and produced a 1 m baseline on a 15 cm rig, even though the
    camera model round-tripped exactly. `GridCalibrationTargetAprilgrid` already defines it.

  * MATCH ON THE FRAME'S OWN HEADER STAMP, never on bag arrival time. The two images of a
    pair are ~1.58 MB each and arrive over DDS tens of ms apart; matching on arrival
    reports ZERO simultaneous pairs on a rig whose sensors are triggered to 1 us. This was
    re-confirmed on 2026-08-31: 0 pairs by arrival time, 996 by header stamp.

Runs in the tartancalib container (ROS1 python3 + kalibr on the path):

  docker run --rm --entrypoint /bin/bash -v $PWD:/repo -v <dataset>:/data tartancalib:latest \\
    -lc 'source /catkin_ws/devel/setup.bash && python3 /repo/scripts/calib/pair_extrinsics.py \\
         --bag /data/ros1/stage6_front_cam1_cam2.bag --cam-a cam1 --cam-b cam2 \\
         --chain-a /data/solve_cam1/log1-camchain.yaml --chain-b /data/solve_cam2/log1-camchain.yaml \\
         --target /repo/config/calib/april_6x6.yaml --pair front --out /data/pair_front.yaml'

Output is one block in the shape of config/rig/rig_extrinsics_imx296.yaml, so the four
pairs concatenate into the rig file that close_rig_ring.py consumes.
"""
import argparse
import sys

import numpy as np
import rosbag
import aslam_cv as acv
import aslam_cameras_april as acv_april
import kalibr_common as kc
from cv_bridge import CvBridge


def build(chain_yaml, target_yaml):
    """A detector bound to THIS camera's fitted geometry, plus the shared target model."""
    chain = kc.ConfigReader.CameraChainParameters(chain_yaml)
    cam = kc.AslamCamera.fromParameters(chain.getCameraParameters(0))
    tp = kc.ConfigReader.CalibrationTargetParameters(target_yaml).getTargetParams()
    opts = acv_april.AprilgridOptions()
    # 7 on a 6x6 board - the same floor Kalibr itself uses. Fewer tags than this and the
    # pose fit is unstable, which shows up as an outlier extrinsic rather than as an error.
    opts.minTagsForValidObs = int(max(tp['tagRows'], tp['tagCols']) + 1)
    grid = acv_april.GridCalibrationTargetAprilgrid(
        tp['tagRows'], tp['tagCols'], tp['tagSize'], tp['tagSpacing'], opts)
    o = acv.GridDetectorOptions()
    o.filterCornerOutliers = False
    return acv.GridDetector(cam.geometry, grid, o)


def poses(bag, topic, det):
    """header-stamp(ns) -> T_target_camera (4x4)."""
    out, bridge = {}, CvBridge()
    for _, msg, _ in rosbag.Bag(bag).read_messages(topics=[topic]):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        ok, obs = det.findTarget(acv.Time(msg.header.stamp.secs, msg.header.stamp.nsecs),
                                 np.array(img))
        if ok:
            out[msg.header.stamp.to_nsec()] = obs.T_t_c().T()
    return out


def average(mats):
    """Mean pose. The mean of rotation matrices is not a rotation - project it back."""
    R = np.mean([m[:3, :3] for m in mats], axis=0)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:                       # reflection, not a rotation
        u[:, -1] *= -1
        R = u @ vt
    t = np.mean([m[:3, 3] for m in mats], axis=0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def angle_between(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--cam-a", required=True, help="the camera whose pose is EXPRESSED (from)")
    ap.add_argument("--cam-b", required=True, help="the camera it is expressed IN (to)")
    ap.add_argument("--chain-a", required=True)
    ap.add_argument("--chain-b", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--pair", required=True, help="left|front|right|rear - the yaml key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-skew-us", type=int, default=1000)
    a = ap.parse_args()

    pa = poses(a.bag, "/%s/image_raw" % a.cam_a, build(a.chain_a, a.target))
    pb = poses(a.bag, "/%s/image_raw" % a.cam_b, build(a.chain_b, a.target))
    print("detections: %s %d, %s %d" % (a.cam_a, len(pa), a.cam_b, len(pb)))

    tol = a.max_skew_us * 1000
    kb = np.array(sorted(pb), dtype=np.int64)
    rel = []
    for t, Ta in sorted(pa.items()):
        if not len(kb):
            break
        i = int(np.argmin(np.abs(kb - t)))
        if abs(int(kb[i]) - t) <= tol:
            rel.append(np.linalg.inv(pb[int(kb[i])]) @ Ta)      # T_b_a
    if len(rel) < 10:
        sys.exit("only %d simultaneous views - not enough to average (need the target visible "
                 "to BOTH cameras at the same instant, in their overlap wedge)" % len(rel))

    T = average(rel)
    spread = np.array([angle_between(T[:3, :3], m[:3, :3]) for m in rel])
    tspread = np.array([np.linalg.norm(T[:3, 3] - m[:3, 3]) for m in rel])
    baseline = float(np.linalg.norm(T[:3, 3]))
    # A square rig puts adjacent cameras 90 deg apart; the deviation from 90 is the number
    # to be suspicious of, since nothing in this solve enforces it.
    rot = angle_between(np.eye(3), T[:3, :3])

    print("simultaneous views %d | baseline %.4f m | rotation %.2f deg" % (len(rel), baseline, rot))
    print("angular spread: median %.2f deg, p90 %.2f deg" % (np.median(spread), np.percentile(spread, 90)))
    print("translation spread: median %.1f mm, p90 %.1f mm" % (np.median(tspread) * 1e3,
                                                               np.percentile(tspread, 90) * 1e3))
    with open(a.out, "w") as f:
        f.write("%s:            # %s expressed in %s\n" % (a.pair, a.cam_a, a.cam_b))
        f.write("  from: %s\n  to: %s\n" % (a.cam_a, a.cam_b))
        f.write("  simultaneous_poses: %d\n" % len(rel))
        f.write("  angular_spread_deg: {median: %.2f, p90: %.2f}\n"
                % (np.median(spread), np.percentile(spread, 90)))
        f.write("  rotation_deg: %.2f\n" % rot)
        f.write("  baseline_m: %.4f\n" % baseline)
        f.write("  T_to_from:\n")
        for r in T:
            f.write("    - [%s]\n" % ", ".join("%.6g" % v for v in r))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
