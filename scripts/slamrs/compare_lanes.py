"""A/B two run_slamrs.py outputs of the same log: CPU lane vs GPU lane, or any two builds.

Upstream's rule for the GPU lane (design-notes D64/D83) is "accuracy inside the band and faster
than the CPU lane on the same machine", not bit-equality. So this reports, over the common stamps:
  - sets tracked (status) in each
  - trajectory disagreement: median / p95 / max position difference, raw (both lanes start at the
    same origin) and after a rigid alignment at the first common pose
  - path length and start-to-end for each
  - track_ms median / p95 per lane, with the frontend (fe_ms) and estimator (est_ms) split when the
    CSV has it, and the speedup ratio

  python3 scripts/slamrs/compare_lanes.py <ref.csv> <other.csv> [--label-a cpu --label-b gpu]
"""
import argparse
from pathlib import Path

import numpy as np


def load(p):
    s = np.genfromtxt(p, delimiter=",", names=True)
    return s


def stats(v):
    v = np.asarray(v, float)
    return f"median {np.median(v):7.2f}  p95 {np.percentile(v, 95):7.2f}  max {v.max():7.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    o = ap.parse_args()
    A, B = load(o.a), load(o.b)
    la, lb = o.label_a, o.label_b
    print(f"{la}: {o.a}\n{lb}: {o.b}")
    for name, s in ((la, A), (lb, B)):
        ok = s["status"] > 0 if "status" in s.dtype.names else np.ones(len(s), bool)
        p = np.c_[s["tx"], s["ty"], s["tz"]]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt = np.diff(s["t_ns"]) * 1e-9
        print(f"  {name:>4}: {len(s)} sets, status>0 in {ok.sum()}, path {step.sum():.2f} m, "
              f"start->end {np.linalg.norm(p[-1] - p[0]):.2f} m, max step {np.max(step / np.maximum(dt, 1e-3)):.2f} m/s")
        print(f"        track_ms {stats(s['track_ms'])}")
        if "fe_ms" in s.dtype.names:
            print(f"        fe_ms    {stats(s['fe_ms'])}")
            print(f"        est_ms   {stats(s['est_ms'])}")
    ta, tb = A["t_ns"].astype(np.int64), B["t_ns"].astype(np.int64)
    _, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    pa = np.c_[A["tx"], A["ty"], A["tz"]][ia]
    pb = np.c_[B["tx"], B["ty"], B["tz"]][ib]
    print(f"  common stamps: {len(ia)}")
    d = np.linalg.norm(pa - pb, axis=1)
    print(f"  |pos {la} - pos {lb}| raw (m)            {stats(d)}")
    # rigid alignment (Umeyama without scale) so a different initial yaw does not read as drift
    ma, mb = pa.mean(0), pb.mean(0)
    H = (pb - mb).T @ (pa - ma)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    pb_al = (pb - mb) @ R.T + ma
    d2 = np.linalg.norm(pa - pb_al, axis=1)
    print(f"  |pos {la} - pos {lb}| rigid-aligned (m)  {stats(d2)}   (RMS {np.sqrt((d2 ** 2).mean()):.3f})")
    sa, sb = np.median(A["track_ms"]), np.median(B["track_ms"])
    print(f"  speed: {lb} / {la} track_ms median = {sb:.1f} / {sa:.1f} -> {la} is {sa / sb:.2f}x of {lb}; "
          f"{lb} is {sa / sb:.2f}x faster" if sb < sa else
          f"  speed: {lb} / {la} track_ms median = {sb:.1f} / {sa:.1f} -> {lb} is {sb / sa:.2f}x SLOWER")


if __name__ == "__main__":
    main()
