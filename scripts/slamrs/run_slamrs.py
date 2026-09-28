"""Replay a BEV raw log (log_rig.sh output) through slam-rs, straight from the .raw files.

  SLAM_RS_DIR=<slam-rs checkout> python3 scripts/slamrs/run_slamrs.py <imglog dir> <out.csv> [--dry-run] [--max-sets N] [--profile fast|reference]

--dry-run exercises everything except slam-rs (frameset matching, IMU prep, image decode).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import os

REPO = Path(__file__).resolve().parents[2]
# the third_party/rerun-examples submodule, with slam_rs/_core.so built by build_slamrs.sh
SLAMRS = Path(os.environ.get("SLAM_RS_DIR", REPO / "third_party/rerun-examples/packages/slam-rs"))
SCALE = 2
IMU_LAG_NS = 3_731_525  # config/calib/imu_mpu9250.yaml timeshift_cam_imu (t_imu = t_cam + shift), DLPF 184 Hz
SET_TOL_NS = 1_000_000


def load_framesets(d):
    idx = [np.loadtxt(d / f"cam{i}_index.csv", delimiter=",", comments=("#", "stamp"), dtype=np.int64) for i in (1, 2, 3, 4)]
    sets = []
    for t, off in idx[0]:
        row = [(t, off)]
        for k in (1, 2, 3):
            j = np.searchsorted(idx[k][:, 0], t)
            cands = [c for c in (j - 1, j) if 0 <= c < len(idx[k])]
            c = min(cands, key=lambda c: abs(idx[k][c, 0] - t))
            if abs(idx[k][c, 0] - t) > SET_TOL_NS:
                break
            row.append(tuple(idx[k][c]))
        else:
            sets.append((int(np.mean([r[0] for r in row])), [int(r[1]) for r in row]))
    return sets, len(idx[0])


def load_imu(d):
    imu = np.loadtxt(d / "imu0.csv", delimiter=",", comments="#")
    t = imu[:, 0].astype(np.int64) - IMU_LAG_NS
    acc, gyr = imu[:, 1:4].copy(), imu[:, 4:7].copy()
    rest = t < t[0] + 2_000_000_000  # every log starts with the rig still
    gyr -= gyr[rest].mean(0)
    a0 = acc[rest].mean(0)
    # |a| at rest reads ~10.4 (uncalibrated zero-g offset). Only the component along gravity
    # is observable from one pose, so remove exactly that.
    acc -= a0 - 9.80665 * a0 / np.linalg.norm(a0)
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], gyr[keep], acc[keep], np.linalg.norm(a0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-sets", type=int, default=0)
    ap.add_argument("--profile", default="fast")
    ap.add_argument("--calib", type=Path, default=REPO / "config/slamrs/bev_calib_s2.json")
    ap.add_argument("--config", type=Path, default=REPO / "config/slamrs/bev_vio_config.json")
    a = ap.parse_args()

    geo = dict(l.split() for l in open(a.log / "geometry.txt"))
    W, H, bpf = int(geo["width"]), int(geo["height"]), int(geo["bytes_per_frame"])
    raws = [np.memmap(a.log / f"cam{i}.raw", dtype=np.uint8, mode="r") for i in (1, 2, 3, 4)]
    sets, n1 = load_framesets(a.log)
    ti, gyr, acc, a0n = load_imu(a.log)
    print(f"{len(sets)} complete sets of {n1} cam1 frames; IMU {len(ti)} samples, |a|rest {a0n:.3f}")
    if a.max_sets:
        sets = sets[: a.max_sets]

    def frames(offs):
        return [np.ascontiguousarray(cv2.resize(raws[k][o:o + bpf].reshape(H, W), (W // SCALE, H // SCALE),
                                                interpolation=cv2.INTER_AREA)) for k, o in enumerate(offs)]

    if a.dry_run:
        f = frames(sets[len(sets) // 2][1])
        print("frame shapes", [x.shape for x in f], "mean luma", [round(float(x.mean()), 1) for x in f])
        print("set dt ms median", np.median(np.diff([s[0] for s in sets])) / 1e6,
              "IMU covers sets:", ti[0] < sets[0][0], ti[-1] > sets[-1][0])
        return

    sys.path.insert(0, str(SLAMRS))
    from slam_rs import _core

    calib = _core.Calibration.from_json(a.calib.read_text())
    cfg = _core.VioConfig.from_json(a.config.read_text())
    vio = _core.Vio(calib, cfg)
    ii = 0
    rows, t0 = [], time.perf_counter()
    for n, (t, offs) in enumerate(sets):
        j = np.searchsorted(ti, t + 20_000_000)  # IMU a little past the frame
        if j > ii:
            vio.push_imu_batch(ti[ii:j], gyr[ii:j], acc[ii:j])
            ii = j
        tc = time.perf_counter()
        r = vio.track(t, frames(offs))
        dt_ms = (time.perf_counter() - tc) * 1e3
        snap = vio.snapshot()
        nobs = snap.num_observations if snap else 0
        nlm = len(snap.landmark_ids) if snap else 0
        rows.append([t, int(r.status), *r.world_from_rig, *r.velocity, *r.accel_bias, *r.gyro_bias, nobs, nlm, dt_ms])
        if n % 100 == 0:
            p = r.world_from_rig
            print(f"{n:5d} t={(t - sets[0][0]) / 1e9:6.2f}s status={int(r.status)} pos={np.round(p[:3], 3)} "
                  f"lm={nlm} obs={nobs} track={dt_ms:.1f}ms", flush=True)
    hdr = "t_ns,status,tx,ty,tz,qx,qy,qz,qw,vx,vy,vz,bax,bay,baz,bgx,bgy,bgz,num_obs,num_landmarks,track_ms"
    np.savetxt(a.out, np.array(rows), delimiter=",", header=hdr, comments="", fmt=["%d", "%d"] + ["%.9g"] * 19)
    print(f"wrote {a.out}: {len(rows)} poses in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
