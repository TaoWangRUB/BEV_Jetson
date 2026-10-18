"""Replay a BEV raw log through slam-rs VIO with the loop-closing backend beside it.

  .venv/bin/python scripts/slamrs/run_slamrs_lc.py <imglog dir> <out dir> [--max-sets N] [--threads N]
        [--lc-config config/slamrs/bev_lc_config.json] [--keyframes-out <dir>] [--no-lc]

The VIO call sequence is run_slamrs.py's, so `<out dir>/slamrs_vio.csv` carries the same rows as
run_slamrs.py would write (track_ms is wall time and differs run to run; every other column is
identical). Beside it:

  slamrs_lc.csv      every frame's corrected pose (same columns as the VIO csv, no timings)
  keyframes_lc.csv   keyframe stamps with VIO and corrected poses
  loop_edges.csv     accepted constraints with their separation in each trajectory
  hypotheses.csv     every candidate verified, accepted or not, and why
  timings.csv        per-keyframe cost of feature extraction, retrieval, verification, optimisation
  summary.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_slamrs as base  # noqa: E402  (loaders, SCALE, IMU_LAG_NS, SLAMRS)
from loop_closure.backend import LoopClosureBackend  # noqa: E402
from loop_closure.camera import Rig, T_to_pose7, pose7_to_T  # noqa: E402
from loop_closure.keyframes import KeyframeTracker, export_keyframe  # noqa: E402

REPO = HERE.parents[1]
HDR = "t_ns,status,tx,ty,tz,qx,qy,qz,qw,vx,vy,vz,bax,bay,baz,bgx,bgy,bgz,num_obs,num_landmarks,track_ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-sets", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--calib", type=Path, default=REPO / "config/slamrs/bev_calib_s2.json")
    ap.add_argument("--config", type=Path, default=REPO / "config/slamrs/bev_vio_config.json")
    ap.add_argument("--lc-config", type=Path, default=REPO / "config/slamrs/bev_lc_config.json")
    ap.add_argument("--keyframes-out", type=Path, help="also export every keyframe (PNGs, pose, landmarks)")
    ap.add_argument("--no-lc", action="store_true", help="VIO only: write slamrs_vio.csv and stop")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    geo = dict(l.split() for l in open(a.log / "geometry.txt"))
    W, H, bpf = int(geo["width"]), int(geo["height"]), int(geo["bytes_per_frame"])
    raws = [np.memmap(a.log / f"cam{i}.raw", dtype=np.uint8, mode="r") for i in (1, 2, 3, 4)]
    sets, n1 = base.load_framesets(a.log)
    ti, gyr, acc, a0n = base.load_imu(a.log)
    print(f"{len(sets)} complete sets of {n1} cam1 frames; IMU {len(ti)} samples, |a|rest {a0n:.3f}")
    if a.max_sets:
        sets = sets[: a.max_sets]

    import cv2
    def frames(offs):
        return [np.ascontiguousarray(cv2.resize(raws[k][o:o + bpf].reshape(H, W), (W // base.SCALE, H // base.SCALE),
                                                interpolation=cv2.INTER_AREA)) for k, o in enumerate(offs)]

    sys.path.insert(0, str(base.SLAMRS))
    from slam_rs import _core
    calib = _core.Calibration.from_json(a.calib.read_text())
    cfg = _core.VioConfig.from_json(a.config.read_text())
    vio = _core.Vio(calib, cfg, threads=a.threads)

    lc_cfg = json.load(open(a.lc_config))
    rig = Rig.from_json(a.calib, lc_cfg["features"]["theta_max_deg"])
    backend = None if a.no_lc else LoopClosureBackend(rig, lc_cfg, a.out)
    tracker = KeyframeTracker()

    ii, rows, poses, t0 = 0, [], [], time.perf_counter()
    n_kf = 0
    for n, (t, offs) in enumerate(sets):
        j = np.searchsorted(ti, t + 20_000_000)
        if j > ii:
            vio.push_imu_batch(ti[ii:j], gyr[ii:j], acc[ii:j])
            ii = j
        ims = frames(offs)
        tc = time.perf_counter()
        r = vio.track(t, ims)
        dt_ms = (time.perf_counter() - tc) * 1e3
        snap = vio.snapshot()
        nobs = snap.num_observations if snap else 0
        nlm = len(snap.landmark_ids) if snap else 0
        rows.append([t, int(r.status), *r.world_from_rig, *r.velocity, *r.accel_bias, *r.gyro_bias, nobs, nlm, dt_ms])
        poses.append((t, pose7_to_T(r.world_from_rig)))
        if backend is not None or a.keyframes_out:
            for kf in tracker.update(t, ims, snap, vio.flow_frame()):
                n_kf += 1
                if a.keyframes_out:
                    export_keyframe(a.keyframes_out, kf)
                if backend is not None:
                    h = backend.on_keyframe(kf)
                    if h is not None:
                        print(f"  LOOP kf {h.cur} -> {h.cand} ({h.route}) inliers {h.inliers2}/{h.tracked} "
                              f"ratio {h.ratio:.2f} best pair {h.best_pair[0] + 1}-{h.best_pair[1] + 1}", flush=True)
        if n % 100 == 0:
            p = r.world_from_rig
            print(f"{n:5d} t={(t - sets[0][0]) / 1e9:6.2f}s status={int(r.status)} pos={np.round(p[:3], 3)} "
                  f"lm={nlm} obs={nobs} kf={n_kf} track={dt_ms:.1f}ms", flush=True)
    for kf in tracker.flush():
        n_kf += 1
        if a.keyframes_out:
            export_keyframe(a.keyframes_out, kf)
        if backend is not None:
            backend.on_keyframe(kf)

    np.savetxt(a.out / "slamrs_vio.csv", np.array(rows), delimiter=",", header=HDR, comments="",
               fmt=["%d", "%d"] + ["%.9g"] * 19)
    print(f"wrote {a.out / 'slamrs_vio.csv'}: {len(rows)} poses in {time.perf_counter() - t0:.0f} s; {n_kf} keyframes")
    if backend is None:
        return
    backend.finish()
    ts = np.array([t for t, _ in poses])
    corr = backend.correct_frames(ts, [T for _, T in poses])
    lc_rows = [[t, rows[k][1], *T_to_pose7(corr[k])] for k, t in enumerate(ts)]
    np.savetxt(a.out / "slamrs_lc.csv", np.array(lc_rows), delimiter=",", header="t_ns,status,tx,ty,tz,qx,qy,qz,qw",
               comments="", fmt=["%d", "%d"] + ["%.9g"] * 7)
    s = backend.write()
    print(f"loop closure: {s['keyframes']} keyframes, {s['hypotheses']} hypotheses, {s['accepted']} accepted, "
          f"{len(s['dropped'])} dropped; feat {s['feat_ms_median']:.1f} ms, verify {s['verify_ms_median']:.1f} ms median")


if __name__ == "__main__":
    main()
