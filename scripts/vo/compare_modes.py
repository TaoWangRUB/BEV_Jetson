"""A/B two or more cuVSLAM VO replays of the same bag: cost per set, continuity, and shape.

  python3 scripts/vo/compare_modes.py REF_DIR OTHER_DIR [OTHER_DIR ...] [--slamrs slamrs.csv]

Each DIR is a replay_host.sh output directory; its <DIR>_timing.csv (TIMING=1 or SLAM=1) is
read if present. Trajectories are compared against REF with an SE(3) fit over common stamps -
no scale, so a scale change shows up as residual. Replays are NOT reproducible (async SBA,
cuvslam_multicam_node.cpp): include a same-config replay at another rate as one of the OTHERs
to see the noise floor, or a difference means nothing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slamrs"))
from compare import slamrs_cam1, window_scale  # noqa: E402
from handeye import read_odom  # noqa: E402

JUMP_MPS = 5.0


def load(d):
    od = read_odom(str(next(Path(d).glob("*.db3"))))
    t = od[:, 0].astype(np.int64)
    return t, od[:, 1:4]


def timing(d):
    f = Path(str(d).rstrip("/") + "_timing.csv")
    if not f.exists():
        return None
    return np.genfromtxt(f, delimiter=",", names=True)


def se3_residual(t_ref, p_ref, t, p):
    common, i, j = np.intersect1d(t_ref, t, return_indices=True)
    A, B = p[j] - p[j].mean(0), p_ref[i] - p_ref[i].mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    r = np.linalg.norm(A @ R.T - B, axis=1)
    return len(common), np.sqrt((r ** 2).mean()), r.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--slamrs", type=Path)
    a = ap.parse_args()

    t_ref, p_ref = load(a.ref)
    if a.slamrs:
        s = np.genfromtxt(a.slamrs, delimiter=",", names=True)
        ok = s["status"] == 1
        ts, ps = s["t_ns"][ok].astype(np.int64), slamrs_cam1(s)[ok]
    print(f"{'replay':34s} {'poses':>5s} {'jumps':>5s} {'vmax':>5s} {'path m':>6s} {'end-st':>6s} "
          f"{'track ms mean/p95/max':>22s} {'kf/non-kf ms':>13s} {'cb p95':>6s} {'vs REF rms/max m':>16s} {'scale/slam-rs':>13s}")
    for d in [a.ref] + a.others:
        t, p = load(d)
        dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        v = dp / (np.diff(t) * 1e-9)
        tm = timing(d)
        if tm is not None:
            tr = tm["track_us"] / 1e3
            kf = tm["keyframe"] == 1
            tstr = f"{tr.mean():5.1f}/{np.percentile(tr, 95):5.1f}/{tr.max():6.1f}"
            kstr = f"{tr[kf].mean():5.1f}/{tr[~kf].mean():5.1f}" if kf.any() else "-"   # flagged with SLAM only
            cstr = f"{np.percentile(tm['callback_us'] / 1e3, 95):6.1f}"
        else:
            tstr, kstr, cstr = "-", "-", "-"
        n, rms, mx = se3_residual(t_ref, p_ref, t, p) if d != a.ref else (len(t), 0.0, 0.0)
        sc = f"{np.median(window_scale(t, p, ts, ps)):.3f}" if a.slamrs else "-"
        print(f"{Path(d).name:34s} {len(t):5d} {(v > JUMP_MPS).sum():5d} {v.max():5.1f} {dp[v <= JUMP_MPS].sum():6.1f} "
              f"{np.linalg.norm(p[-1] - p[0]):6.2f} {tstr:>22s} {kstr:>13s} {cstr:>6s} {rms:7.2f}/{mx:6.2f}   {sc:>13s}")


if __name__ == "__main__":
    main()
