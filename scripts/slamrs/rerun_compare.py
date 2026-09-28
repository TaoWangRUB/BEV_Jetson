"""slam-rs VIO beside cuVSLAM VO and cuVSLAM SLAM, in one Rerun scene.

  .venv/bin/python scripts/slamrs/rerun_compare.py <imglog dir> <slamrs.csv> <cuvslam obs bag dir>
        [--save out.rrd] [--spawn] [--image-stride 4] [--no-images]

Everything is drawn in slam-rs's world (gravity-aligned, +Z up, metric from the IMU). cuVSLAM's
frame is cam1's optical frame at its first pose, so it is brought across through the Kalibr
T_imu_cam1 in config/slamrs/bev_calib_s2.json - two ways, one per 3D view:

  anchored  cuVSLAM's cam1 pose made to coincide with slam-rs's at their first common stamp,
            and nothing else. No fit and NO scale: any scale disagreement shows as the two
            paths separating in proportion to distance travelled.
  sim3      Umeyama Sim(3) over the common stamps up to cuVSLAM's first jump - shape only.

cuVSLAM VO is /cuvslam/odometry, which deliberately keeps its tracking-loss teleports (README
3.1); it is split into segments there instead of drawing a chord across them. cuVSLAM SLAM is
the LAST /cuvslam/slam_path (the most optimised graph), as in vo/rerun_multicam.py.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vo"))
from compare import umeyama, window_scale  # noqa: E402
from handeye import quat_to_R, read_odom  # noqa: E402
from rerun_multicam import read_slam, split_on_gaps  # noqa: E402
from rerun_odometry import find_bag  # noqa: E402
from run_slamrs import load_framesets  # noqa: E402

REPO = HERE.parents[1]
JUMP_MPS = 5.0
C_SLAMRS, C_VO, C_SLAM = 0xE8A33DFF, 0x3D9BE8FF, 0x52C46BFF   # orange, blue, green


def pose(T, p, q):
    T = np.repeat(np.eye(4)[None], len(p), 0)
    T[:, :3, :3] = quat_to_R(q)
    T[:, :3, 3] = p
    return T


def apply(A, P):
    return P @ A[:3, :3].T + A[:3, 3]


def segments(P, t):
    """Split on tracking jumps (> JUMP_MPS) and on time gaps, like the VO viewer does."""
    if len(P) < 2:
        return []
    v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)
    out = []
    for seg_idx in np.split(np.arange(len(P)), np.where(v > JUMP_MPS)[0] + 1):
        out += split_on_gaps(P[seg_idx], t[seg_idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("slamrs_csv", type=Path)
    ap.add_argument("cuvslam_bag", type=Path, help="obs_* bag directory (odometry + slam topics)")
    ap.add_argument("--calib", type=Path, default=REPO / "config/slamrs/bev_calib_s2.json")
    ap.add_argument("--image-stride", type=int, default=4, help="log every Nth camera set")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--spawn", action="store_true")
    a = ap.parse_args()

    # slam-rs: T_world_imu -> T_world_cam1 through the same extrinsic it was run with
    s = np.genfromtxt(a.slamrs_csv, delimiter=",", names=True)
    ok = s["status"] == 1
    ts = s["t_ns"][ok].astype(np.int64)
    T_w_imu = pose(None, np.stack([s["tx"], s["ty"], s["tz"]], -1)[ok],
                   np.stack([s["qx"], s["qy"], s["qz"], s["qw"]], -1)[ok])
    c = json.load(open(a.calib))["value0"]["T_imu_cam"][0]
    T_imu_c1 = pose(None, np.array([[c["px"], c["py"], c["pz"]]]),
                    np.array([[c["qx"], c["qy"], c["qz"], c["qw"]]]))[0]
    T_w_c1 = T_w_imu @ T_imu_c1
    Ps = T_w_c1[:, :3, 3]

    # cuVSLAM VO (cam1 pose in its odom frame)
    db = next(a.cuvslam_bag.glob("*.db3"))
    od = read_odom(str(db))
    tc = od[:, 0].astype(np.int64)
    T_o_c1 = pose(None, od[:, 1:4], od[:, 4:8])
    Pc = od[:, 1:4]
    slam_path, slam_t, lc, *_ = read_slam(find_bag(a.cuvslam_bag))

    t0 = ts[0]
    common, ic, is_ = np.intersect1d(tc, ts, return_indices=True)
    A_anchor = T_w_c1[is_[0]] @ np.linalg.inv(T_o_c1[ic[0]])
    v = np.linalg.norm(np.diff(Pc, axis=0), axis=1) / (np.diff(tc) * 1e-9)
    first_jump = tc[1:][v > JUMP_MPS][0] if (v > JUMP_MPS).any() else tc[-1] + 1
    fit = common < first_jump
    sc, R, t = umeyama(Pc[ic[fit]], Ps[is_[fit]])        # cuVSLAM -> slam-rs
    A_sim3 = np.eye(4)
    A_sim3[:3, :3], A_sim3[:3, 3] = sc * R, t
    res = np.linalg.norm(apply(A_sim3, Pc[ic[fit]]) - Ps[is_[fit]], axis=1)

    vs = np.linalg.norm(np.diff(Ps, axis=0), axis=1) / (np.diff(ts) * 1e-9)
    ws = window_scale(tc, Pc, ts, Ps)
    lines = [
        f"# {a.log.name}",
        "",
        "| | slam-rs VIO | cuVSLAM VO |",
        "|---|---|---|",
        f"| poses | {len(ts)} of {len(s)} sets | {len(tc)} |",
        f"| jumps > {JUMP_MPS:g} m/s | {(vs > JUMP_MPS).sum()} (max {vs.max():.1f} m/s) | {(v > JUMP_MPS).sum()} (max {v.max():.1f} m/s) |",
        f"| end - start | {np.linalg.norm(Ps[-1] - Ps[0]):.2f} m | {np.linalg.norm(Pc[-1] - Pc[0]):.2f} m |",
        f"| median track() | {np.median(s['track_ms']):.1f} ms (1 CPU thread, 728x544) | - |",
        "",
        f"**Scale**: over matched 1 s windows cuVSLAM moves **{np.median(ws):.3f}x** as far as slam-rs "
        f"(median of {len(ws)}, IQR {np.percentile(ws, 25):.2f}-{np.percentile(ws, 75):.2f}). "
        f"Sim3 fit (right view, {fit.sum()} poses up to cuVSLAM's first jump): scale {1 / sc:.3f}, "
        f"residual RMS {np.sqrt((res ** 2).mean()):.2f} m / max {res.max():.2f} m - shape-dependent, prefer the window median.",
        "",
        "slam-rs is metric only as far as the MPU-9250 accelerometer is: at rest it reads "
        f"|a| = 10.33-10.35 m/s^2, and only the along-gravity part of that was removed.",
        "",
        "Colours: **orange** slam-rs VIO, **blue** cuVSLAM VO, **green** cuVSLAM SLAM (optimised path).",
    ]
    print("\n".join(lines))

    rr.init("bev_slamrs_vs_cuvslam", spawn=a.spawn)
    if a.save:
        rr.save(a.save)
    cams = [f"cams/cam{i}" for i in (1, 2, 3, 4)]
    eye = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(row_shares=[5, 2, 2], contents=[
        rrb.Horizontal(contents=[
            rrb.Spatial3DView(name="anchored at first pose - raw scales", origin="/anchored", eye_controls=eye),
            rrb.Spatial3DView(name="Sim(3)-fitted - shape only", origin="/sim3", eye_controls=eye),
            rrb.TextDocumentView(name="numbers", origin="/summary")], column_shares=[3, 3, 2]),
        rrb.Horizontal(contents=[
            rrb.TimeSeriesView(name="height z (m), anchored", origin="/plots/z"),
            rrb.TimeSeriesView(name="speed (m/s), anchored", origin="/plots/speed"),
            rrb.TimeSeriesView(name="slam-rs: landmarks / track ms", origin="/plots/slamrs")]),
        rrb.Horizontal(contents=[rrb.Spatial2DView(name=n.split("/")[-1], origin=n) for n in cams]),
    ]), rrb.TimePanel(state="collapsed"), collapse_panels=True))

    rr.log("summary", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
    for view in ("anchored", "sim3"):
        rr.log(view, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for key, name, col in (("slamrs", "slam-rs VIO", C_SLAMRS), ("cuvslam_vo", "cuVSLAM VO", C_VO),
                           ("cuvslam_slam", "cuVSLAM SLAM", C_SLAM)):
        rr.log(f"plots/z/{key}", rr.SeriesLines(colors=[col], names=name), static=True)
        rr.log(f"plots/speed/{key}", rr.SeriesLines(colors=[col], names=name), static=True)
    rr.log("plots/slamrs/landmarks", rr.SeriesLines(colors=[C_SLAMRS], names="landmarks"), static=True)
    rr.log("plots/slamrs/track_ms", rr.SeriesLines(colors=[0x999999FF], names="track() ms"), static=True)

    tc_s, ts_s = (tc - t0) * 1e-9, (ts - t0) * 1e-9
    for view, A in (("anchored", A_anchor), ("sim3", A_sim3)):
        rr.log(f"{view}/slamrs_vio", rr.LineStrips3D(segments(Ps, ts_s), colors=[C_SLAMRS], radii=0.015), static=True)
        rr.log(f"{view}/cuvslam_vo", rr.LineStrips3D(segments(apply(A, Pc), tc_s), colors=[C_VO], radii=0.01), static=True)
        if len(slam_path):
            st = (slam_t * 1e9 - t0) * 1e-9
            rr.log(f"{view}/cuvslam_slam", rr.LineStrips3D(segments(apply(A, slam_path), st), colors=[C_SLAM], radii=0.01),
                   static=True)
        if len(lc):
            rr.log(f"{view}/loop_closures", rr.Points3D(apply(A, lc), colors=[C_SLAM], radii=0.05), static=True)
        rr.log(f"{view}/start", rr.Points3D([Ps[0]], colors=[0xFFFFFFFF], radii=0.08, labels=["start"]), static=True)

    # time-varying: heads and plots
    Pc_anch = apply(A_anchor, Pc)
    slam_od = read_odom(str(db), "/cuvslam/slam_odometry") if len(slam_path) else None
    for i in range(len(ts)):
        rr.set_time("time", duration=ts_s[i])
        for view in ("anchored", "sim3"):
            rr.log(f"{view}/head_slamrs", rr.Points3D([Ps[i]], colors=[C_SLAMRS], radii=0.06))
        rr.log("plots/z/slamrs", rr.Scalars(Ps[i, 2]))
        if i:
            rr.log("plots/speed/slamrs", rr.Scalars(vs[i - 1]))
        rr.log("plots/slamrs/landmarks", rr.Scalars(s["num_landmarks"][ok][i]))
        rr.log("plots/slamrs/track_ms", rr.Scalars(s["track_ms"][ok][i]))
    for i in range(len(tc)):
        rr.set_time("time", duration=tc_s[i])
        rr.log("anchored/head_vo", rr.Points3D([Pc_anch[i]], colors=[C_VO], radii=0.05))
        rr.log("sim3/head_vo", rr.Points3D([apply(A_sim3, Pc[i:i + 1])[0]], colors=[C_VO], radii=0.05))
        rr.log("plots/z/cuvslam_vo", rr.Scalars(Pc_anch[i, 2]))
        if i and v[i - 1] < JUMP_MPS:
            rr.log("plots/speed/cuvslam_vo", rr.Scalars(v[i - 1]))
    if slam_od is not None and len(slam_od):
        Pl = apply(A_anchor, slam_od[:, 1:4])
        tl = (slam_od[:, 0].astype(np.int64) - t0) * 1e-9
        for i in range(len(tl)):
            rr.set_time("time", duration=tl[i])
            rr.log("plots/z/cuvslam_slam", rr.Scalars(Pl[i, 2]))

    if not a.no_images:
        geo = dict(line.split() for line in open(a.log / "geometry.txt"))
        W, H, bpf = int(geo["width"]), int(geo["height"]), int(geo["bytes_per_frame"])
        raws = [np.memmap(a.log / f"cam{i}.raw", dtype=np.uint8, mode="r") for i in (1, 2, 3, 4)]
        sets, _ = load_framesets(a.log)
        for t, offs in sets[:: a.image_stride]:
            rr.set_time("time", duration=(t - t0) * 1e-9)
            for k, o in enumerate(offs):
                img = cv2.resize(raws[k][o:o + bpf].reshape(H, W), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
                rr.log(cams[k], rr.Image(img[::-1, ::-1].copy()).compress(jpeg_quality=70))  # mounted inverted
    print(f"saved {a.save}" if a.save else "done")


if __name__ == "__main__":
    main()
