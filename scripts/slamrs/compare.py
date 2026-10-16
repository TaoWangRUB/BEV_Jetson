"""Score a slam-rs trajectory against cuVSLAM on the same log.

No ground truth exists, so this reports what CAN disagree:
  - continuity: max step speed, poses lost
  - path length and return-to-start for each
  - Sim(3) Umeyama alignment slam-rs -> cuVSLAM over common stamps: scale factor + RMS residual.
    slam-rs scale comes from the IMU, cuVSLAM's from the camera baselines; the ratio is the
    independent scale cross-check README section 8 is missing.

  python3 scripts/slamrs/compare.py <cuvslam obs bag .db3> [slamrs.csv]
"""
import json
import sys
from pathlib import Path

import numpy as np

from handeye import quat_to_R, read_odom  # noqa: E402

CALIB = Path(__file__).resolve().parents[2] / "config/slamrs/bev_calib_s2.json"


def slamrs_cam1(s, calib=CALIB):
    """slam-rs reports the IMU pose; cuVSLAM tracks cam1. Compare like with like: the ~10 cm
    lever arm between them swings with every rotation and biases a displacement ratio."""
    c = json.load(open(calib))["value0"]["T_imu_cam"][0]
    R_wi = quat_to_R(np.stack([s["qx"], s["qy"], s["qz"], s["qw"]], -1))
    p_wi = np.stack([s["tx"], s["ty"], s["tz"]], -1)
    return p_wi + R_wi @ np.array([c["px"], c["py"], c["pz"]])


def summary(name, t, p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    v = d / (np.diff(t) * 1e-9)
    jumps = v > 5.0
    L = d[~jumps].sum()
    print(f"{name:8s} poses {len(t):5d}  span {(t[-1] - t[0]) / 1e9:6.1f}s  path(excl jumps) {L:7.2f} m  "
          f"jumps>5m/s {jumps.sum():3d} (max {v.max():6.1f} m/s)  end-start {np.linalg.norm(p[-1] - p[0]):6.2f} m")
    return jumps


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, D, Vt = np.linalg.svd(B.T @ A / len(src))
    Sg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ Sg @ Vt
    s = np.trace(np.diag(D) @ Sg) / (A ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def window_scale(tc, pc, ts, ps, window_s=1.0, min_move=0.3):
    """cuVSLAM / slam-rs displacement over matched windows - needs no alignment.

    A Sim(3) scale is only as good as the shapes agree, and it is not symmetric: on run5 fitting
    one way gave 0.88 and the other 1.10. The ratio of displacements over the same 1 s, taken
    per window and summarised by the median, does not depend on either frame. Windows across a
    cuVSLAM jump (> 5 m/s) or where the rig barely moved are skipped.
    """
    common, ic, is_ = np.intersect1d(tc, ts, return_indices=True)
    r, k = [], 0
    while k < len(common):
        j = np.searchsorted(common, common[k] + int(window_s * 1e9))
        if j >= len(common):
            break
        dt = (common[j] - common[k]) * 1e-9
        dc = np.linalg.norm(pc[ic[j]] - pc[ic[k]])
        ds = np.linalg.norm(ps[is_[j]] - ps[is_[k]])
        if abs(dt - window_s) < 0.2 and dc / dt < 5.0 and ds > min_move:
            r.append(dc / ds)
        k = j
    return np.array(r)


def main(bag, csv=None):
    od = read_odom(bag)
    tc, pc = od[:, 0].astype(np.int64), od[:, 1:4]
    jc = summary("cuVSLAM", tc, pc)
    if csv is None:
        return
    s = np.genfromtxt(csv, delimiter=",", names=True)
    ts = s["t_ns"].astype(np.int64)
    trk = s["status"] == 1
    ps = slamrs_cam1(s)
    print(f"slam-rs  tracking on {trk.sum()} / {len(ts)} sets; median track {np.median(s['track_ms']):.1f} ms, "
          f"p95 {np.percentile(s['track_ms'], 95):.1f} ms")
    summary("slam-rs", ts[trk], ps[trk])
    # common stamps, before cuVSLAM's first jump so its frame is still the one it started in
    first_jump = tc[1:][jc][0] if jc.any() else tc[-1] + 1
    common, ic, is_ = np.intersect1d(tc[tc < first_jump], ts[trk], return_indices=True)
    if len(common) < 50:
        print("too few common stamps"); return
    sc, R, t = umeyama(ps[trk][is_], pc[ic])
    res = np.linalg.norm((sc * (R @ ps[trk][is_].T)).T + t - pc[ic], axis=1)
    for w in (1.0, 2.0):
        r = window_scale(tc, pc, ts[trk], ps[trk], w)
        print(f"scale, {w:.0f} s windows: cuVSLAM/slam-rs displacement median {np.median(r):.3f} "
              f"(IQR {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f}, n={len(r)})  <1 = cuVSLAM shorter")
    print(f"Sim3 slam-rs -> cuVSLAM on {len(common)} common poses (up to cuVSLAM's first jump, "
          f"{(common[-1] - common[0]) / 1e9:.1f}s): scale {sc:.3f}  RMS {np.sqrt((res ** 2).mean()):.3f} m  max {res.max():.3f} m")
    # umeyama maps src (slam-rs) onto dst (cuVSLAM): dst = sc * R @ src + t, so sc IS cuVSLAM / slam-rs
    print(f"  Sim3 scale {sc:.3f} (cuVSLAM/slam-rs) - shape-dependent and asymmetric; prefer the window median")
    # gravity check: slam-rs world is gravity-aligned (+Z up); vertical drift of a ground rig should be small
    z = ps[trk][:, 2]
    print(f"slam-rs vertical range {z.max() - z.min():.2f} m (world +Z is gravity-up)")


if __name__ == "__main__":
    main(*sys.argv[1:])
