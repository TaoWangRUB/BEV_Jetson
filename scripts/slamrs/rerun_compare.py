"""slam-rs VIO beside cuVSLAM VO and cuVSLAM SLAM, in one Rerun scene.

  .venv/bin/python scripts/slamrs/rerun_compare.py <imglog dir> <slamrs.csv> <cuvslam obs bag dir>
        [--slam-bag <obs bag dir>] [--lc-dir <loop-closure output dir>] [--save out.rrd] [--spawn]
        [--image-stride 4] [--no-images] [--video out.mp4 [--video-speed 4] [--video-size 1920x1080]]

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

--slam-bag takes SLAM from a different replay than VO. Use it: a SLAM=1 replay is only
lossless when slow enough for the backend (README 6) - at 0.4x on run6 it overran, sets queued
and the VO in THAT bag jumped 18.6 m across the gap. VO should come from a VO-only replay. The
SLAM bag's frame is tied to the VO bag's at their first common stamp.

--video renders the scene to an mp4 instead (Rerun itself cannot export video): a headless viewer
of its own, the time-varying data streamed into it in time order, and one screenshot per frame, so
the movie is exactly what the viewer draws, at --video-speed times real time.

Either input can instead be a scripts/vo/replay_raw.sh output directory: cuVSLAM run in-process on
the raw log, every set and deterministic - the same input slam-rs gets, which a bag replay is not
(cut to the motion window, sets dropped whenever SLAM overruns the replay rate).
"""
import argparse
import csv
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
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
C_LC = 0xB65CE8FF                                              # purple: slam-rs loop-closed


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


def load_cuvslam_odom(d):
    """cuVSLAM VO, rows (t_ns, x, y, z, qx, qy, qz, qw): from a scripts/vo/replay_raw.sh output
    directory (odometry.csv) or from a replay_host.sh bag (/cuvslam/odometry)."""
    if (d / "odometry.csv").exists():
        o = np.genfromtxt(d / "odometry.csv", delimiter=",", names=True)
        o = o[o["status"] == 1]
        return np.stack([o[k] for k in ("t_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw")], -1)
    return read_odom(str(next(d.glob("*.db3"))))


def load_cuvslam_slam(d):
    """cuVSLAM SLAM: the final optimised path (positions, stamps in s), the loop-closure poses and
    the number of closure EVENTS. All empty for a run without SLAM."""
    if (d / "odometry.csv").exists():                     # replay_raw.sh
        if not (d / "slam_path.csv").exists():
            return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), 0
        sp = np.genfromtxt(d / "slam_path.csv", delimiter=",", names=True, ndmin=1)
        with open(d / "loop_closures.csv") as f:
            rows = list(csv.DictReader(f))
        lc = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in rows if r["kind"] == "lc_pose"])
        return (np.stack([sp["tx"], sp["ty"], sp["tz"]], -1), sp["t_ns"] * 1e-9, lc.reshape(-1, 3),
                sum(r["kind"] == "event" for r in rows))
    # A bag: the node publishes /cuvslam/loop_closures once per closure event, each message
    # carrying every closure pose so far - the message count is the number of closures.
    sdb = next(d.glob("*.db3"))
    with sqlite3.connect(f"file:{sdb}?mode=ro&immutable=1", uri=True) as con:   # bags are root-owned
        n_events = con.execute("SELECT COUNT(*) FROM messages JOIN topics ON messages.topic_id = topics.id "
                               "WHERE topics.name = '/cuvslam/loop_closures'").fetchone()[0]
    path, t, lc, *_ = read_slam(find_bag(d))
    return path, t, lc, n_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("slamrs_csv", type=Path)
    ap.add_argument("cuvslam_bag", type=Path,
                    help="cuVSLAM VO: an obs_* bag directory with /cuvslam/odometry, or a replay_raw.sh output")
    ap.add_argument("--slam-bag", type=Path, help="SLAM=1 replay to take the SLAM path from (default: cuvslam_bag)")
    ap.add_argument("--calib", type=Path, default=REPO / "config/slamrs/bev_calib_s2.json")
    ap.add_argument("--image-stride", type=int, default=4, help="log every Nth camera set")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--lc-dir", type=Path, help="run_slamrs_lc.py or `slamrs-lc replay` output: draws the loop-closed slam-rs path and its edges")
    ap.add_argument("--video", type=Path, help="render the scene to this mp4 instead of saving or showing it")
    ap.add_argument("--video-speed", type=float, default=4.0, help="times real time")
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--video-size", default="1920x1080", help="viewer size in points, as --window-size")
    ap.add_argument("--video-hold", type=float, default=2.0, help="seconds to hold the last frame")
    ap.add_argument("--video-port", type=int, default=9890, help="port of the headless viewer")
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

    # slam-rs with loop closure (run_slamrs_lc.py): corrected frames and the accepted edges, same world
    Pl_lc, tl_lc, edges_lc, n_edges = None, None, [], 0
    if a.lc_dir:
        l = np.genfromtxt(a.lc_dir / "slamrs_lc.csv", delimiter=",", names=True)
        okl = l["status"] == 1
        tl_lc = l["t_ns"][okl].astype(np.int64)
        Pl_lc = (pose(None, np.stack([l["tx"], l["ty"], l["tz"]], -1)[okl],
                      np.stack([l["qx"], l["qy"], l["qz"], l["qw"]], -1)[okl]) @ T_imu_c1)[:, :3, 3]
        kf = np.genfromtxt(a.lc_dir / "keyframes_lc.csv", delimiter=",", names=True)
        kf_c1 = (pose(None, np.stack([kf["lc_tx"], kf["lc_ty"], kf["lc_tz"]], -1),
                      np.stack([kf["lc_qx"], kf["lc_qy"], kf["lc_qz"], kf["lc_qw"]], -1)) @ T_imu_c1)[:, :3, 3]
        try:
            ed = np.genfromtxt(a.lc_dir / "loop_edges.csv", delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
            # (time the closure was accepted, current keyframe pos, candidate keyframe pos, label)
            edges_lc = sorted(
                (int(e["t_cur_ns"]), kf_c1[int(e["cur"])], kf_c1[int(e["cand"])],
                 f"LC kf{int(e['cur'])}->kf{int(e['cand'])} {int(e['inliers'])}/{int(e['tracked'])}") for e in ed)
        except (ValueError, IndexError):
            edges_lc = []
        n_edges = len(edges_lc)

    # cuVSLAM VO (cam1 pose in its odom frame)
    od = load_cuvslam_odom(a.cuvslam_bag)
    tc = od[:, 0].astype(np.int64)
    T_o_c1 = pose(None, od[:, 1:4], od[:, 4:8])
    Pc = od[:, 1:4]
    slam_bag = a.slam_bag or a.cuvslam_bag
    slam_path, slam_t, lc, n_lc_events = load_cuvslam_slam(slam_bag)
    A_vo_slam = np.eye(4)            # SLAM bag frame -> VO bag frame
    if slam_bag != a.cuvslam_bag:
        od2 = load_cuvslam_odom(slam_bag)
        _, i1, i2 = np.intersect1d(tc, od2[:, 0].astype(np.int64), return_indices=True)
        A_vo_slam = T_o_c1[i1[0]] @ np.linalg.inv(pose(None, od2[i2[:1], 1:4], od2[i2[:1], 4:8])[0])
    slam_path = apply(A_vo_slam, slam_path) if len(slam_path) else slam_path
    lc = apply(A_vo_slam, lc) if len(lc) else lc

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

    # one column per path, so the odometries and the two loop-closed results sit side by side.
    # The paths cover different spans - slam-rs the whole log, a cuVSLAM bag cut to the motion window
    # (raw_log_to_bag.py --motion), a SLAM replay that ends early - so end - start is also given
    # over the span every path covers.
    paths = {"slam-rs VIO": (ts, Ps)}
    if Pl_lc is not None:
        paths["slam-rs + loop closure"] = (tl_lc, Pl_lc)
    paths["cuVSLAM VO"] = (tc, Pc)
    if len(slam_path):
        paths["cuVSLAM SLAM"] = ((slam_t * 1e9).astype(np.int64), slam_path)
    c0, c1 = max(t[0] for t, _ in paths.values()), min(t[-1] for t, _ in paths.values())

    def column(t_ns, P, poses=None, closures="-", track="-"):
        sp = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(t_ns) * 1e-9, 1e-9)
        i0, i1 = np.searchsorted(t_ns, c0), np.searchsorted(t_ns, c1, side="right") - 1
        return [poses or f"{len(P)}", f"{(t_ns[0] - t0) * 1e-9:.1f} - {(t_ns[-1] - t0) * 1e-9:.1f} s",
                f"{(sp > JUMP_MPS).sum()} (max {sp.max():.1f} m/s)", f"{np.linalg.norm(P[-1] - P[0]):.2f} m",
                f"{np.linalg.norm(P[i1] - P[i0]):.2f} m", closures, track]
    extra = {"slam-rs VIO": dict(poses=f"{len(ts)} of {len(s)} sets",
                                 track=f"{np.median(s['track_ms']):.1f} ms (1 CPU thread, 728x544)"),
             "slam-rs + loop closure": dict(closures=f"{n_edges} accepted edges"),
             "cuVSLAM SLAM": dict(poses=f"{len(slam_path)} optimised",
                                  closures=f"{n_lc_events} events ({len(lc)} closure poses)")}
    cols = {name: column(*tp, **extra.get(name, {})) for name, tp in paths.items()}
    rows = ["poses", "time span", f"jumps > {JUMP_MPS:g} m/s", "end - start",
            f"end - start over the common {(c0 - t0) * 1e-9:.1f} - {(c1 - t0) * 1e-9:.1f} s",
            "loop closures", "median track()"]
    lines = [
        f"# {a.log.name}",
        "",
        "| | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ] + [f"| {r} | " + " | ".join(c[k] for c in cols.values()) + " |" for k, r in enumerate(rows)] + [
        "",
        "cuVSLAM's columns are in its own scale (camera baselines), slam-rs's in the IMU's; see the scale below.",
        "",
        f"**Scale**: over matched 1 s windows cuVSLAM moves **{np.median(ws):.3f}x** as far as slam-rs "
        f"(median of {len(ws)}, IQR {np.percentile(ws, 25):.2f}-{np.percentile(ws, 75):.2f}). "
        f"Sim3 fit (right view, {fit.sum()} poses up to cuVSLAM's first jump): scale {1 / sc:.3f}, "
        f"residual RMS {np.sqrt((res ** 2).mean()):.2f} m / max {res.max():.2f} m - shape-dependent, prefer the window median.",
        "",
        "slam-rs is metric only as far as the MPU-9250 accelerometer is: at rest it reads "
        f"|a| = 10.33-10.35 m/s^2, and only the along-gravity part of that was removed.",
        "",
        "Colours: **orange** slam-rs VIO, **blue** cuVSLAM VO, **green** cuVSLAM SLAM (optimised path)"
        + (", **purple** slam-rs loop-closed path with its accepted edges." if Pl_lc is not None else "."),
    ] + ([
        "",
        f"slam-rs and its loop closure from `{a.lc_dir.name}`.",
    ] if Pl_lc is not None else []) + [
        "",
        f"cuVSLAM VO from `{a.cuvslam_bag.name}`, SLAM from `{slam_bag.name}`.",
    ]
    print("\n".join(lines))

    viewer = client = None
    if a.video:
        # A headless viewer of our own: the data is streamed into it in time order and it is
        # screenshotted after each step, so every frame is exactly what the viewer draws.
        from rerun.experimental import ViewerClient
        url = f"rerun+http://127.0.0.1:{a.video_port}/proxy"
        viewer = subprocess.Popen([str(Path(sys.executable).parent / "rerun"), "--headless", "--port",
                                   str(a.video_port), "--window-size", a.video_size],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(150):
            try:
                client = ViewerClient.connect(url)
                break
            except ConnectionError:
                time.sleep(0.2)
        else:
            os.killpg(viewer.pid, signal.SIGTERM)
            sys.exit(f"the headless viewer never came up on {url}")
        rr.init("bev_slamrs_vs_cuvslam", spawn=False)
        rr.connect_grpc(url)
    else:
        rr.init("bev_slamrs_vs_cuvslam", spawn=a.spawn)
        if a.save:
            rr.save(a.save)
    cams = [f"cams/cam{i}" for i in (1, 2, 3, 4)]
    eye = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)
    linked = rrb.TimeAxis(link="LinkToGlobal")
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(row_shares=[5, 2, 2], contents=[
        rrb.Horizontal(contents=[
            rrb.Spatial3DView(name="anchored at first pose - raw scales", origin="/anchored", eye_controls=eye),
            rrb.Spatial3DView(name="Sim(3)-fitted - shape only", origin="/sim3", eye_controls=eye),
            rrb.TextDocumentView(name="numbers", origin="/summary")], column_shares=[3, 3, 2]),
        rrb.Horizontal(contents=[
            # one time window for the three plots: Rerun pans and zooms each one on its own otherwise
            rrb.TimeSeriesView(name="height z (m), anchored", origin="/plots/z", axis_x=linked),
            rrb.TimeSeriesView(name="speed (m/s), anchored", origin="/plots/speed", axis_x=linked),
            rrb.TimeSeriesView(name="slam-rs: landmarks / track ms", origin="/plots/slamrs", axis_x=linked)]),
        rrb.Horizontal(contents=[rrb.Spatial2DView(name=n.split("/")[-1], origin=n) for n in cams]),
    ]), rrb.TimePanel(state="collapsed", playback_speed=a.video_speed if a.video else None,
                   play_state="following" if a.video else None), collapse_panels=True))

    if a.video:
        # the pane is a quarter of the width: one row per path instead of one column
        colour = {"slam-rs VIO": "orange", "slam-rs + loop closure": "purple", "cuVSLAM VO": "blue",
                  "cuVSLAM SLAM": "green"}
        lines = [f"# {a.log.name}", "", f"Played at {a.video_speed:g}x.", "",
                 "| path | first-to-last | loop closures |", "|---|---|---|"] + [
            f"| **{colour[n]}** {n} | {c[4]} | {c[5]} |" for n, c in cols.items()] + [
            "", "Left: anchored at the first pose, each path in its own scale. Right: cuVSLAM Sim(3)-fitted "
            "onto slam-rs, shape only.", "",
            f"cuVSLAM's scale is its own: over 1 s windows it moves {np.median(ws):.2f}x as far as slam-rs."]
    rr.log("summary", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
    for view in ("anchored", "sim3"):
        rr.log(view, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for key, name, col in (("slamrs", "slam-rs VIO", C_SLAMRS), ("cuvslam_vo", "cuVSLAM VO", C_VO),
                           ("cuvslam_slam", "cuVSLAM SLAM", C_SLAM), ("slamrs_lc", "slam-rs loop-closed", C_LC)):
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
        if Pl_lc is not None:
            rr.log(f"{view}/slamrs_lc", rr.LineStrips3D(segments(Pl_lc, (tl_lc - t0) * 1e-9), colors=[C_LC], radii=0.012),
                   static=True)
            # the edges and their markers are logged on the timeline below, when each closure happens
        rr.log(f"{view}/start", rr.Points3D([Ps[0]], colors=[0xFFFFFFFF], radii=0.08, labels=["start"]), static=True)

    # time-varying: heads, plots, closures, images. Straight to Rerun, or buffered (--video) to be
    # streamed in time order.
    events = [] if a.video else None

    class Timeline:
        t = 0.0

        def at(self, t_s):
            self.t = float(t_s)
            if events is None:
                rr.set_time("time", duration=self.t)

        def log(self, path, arch):
            if events is None:
                rr.log(path, arch)
            else:
                events.append((self.t, len(events), path, arch))

    tl = Timeline()
    Pc_anch = apply(A_anchor, Pc)
    for i in range(len(ts)):
        tl.at(ts_s[i])
        for view in ("anchored", "sim3"):
            tl.log(f"{view}/head_slamrs", rr.Points3D([Ps[i]], colors=[C_SLAMRS], radii=0.06))
        tl.log("plots/z/slamrs", rr.Scalars(Ps[i, 2]))
        if i:
            tl.log("plots/speed/slamrs", rr.Scalars(vs[i - 1]))
        tl.log("plots/slamrs/landmarks", rr.Scalars(s["num_landmarks"][ok][i]))
        tl.log("plots/slamrs/track_ms", rr.Scalars(s["track_ms"][ok][i]))
    for i in range(len(tc)):
        tl.at(tc_s[i])
        tl.log("anchored/head_vo", rr.Points3D([Pc_anch[i]], colors=[C_VO], radii=0.05))
        tl.log("sim3/head_vo", rr.Points3D([apply(A_sim3, Pc[i:i + 1])[0]], colors=[C_VO], radii=0.05))
        tl.log("plots/z/cuvslam_vo", rr.Scalars(Pc_anch[i, 2]))
        if i and v[i - 1] < JUMP_MPS:
            tl.log("plots/speed/cuvslam_vo", rr.Scalars(v[i - 1]))
    if len(slam_path):
        # cuVSLAM SLAM's height and speed from its FINAL optimised path - the green line in 3D, and
        # the same kind of path as slam-rs's loop-closed one. Its live corrected pose
        # (/cuvslam/slam_odometry) steps at every closure (up to 12 m/s on run6), and those
        # corrections would read as speed spikes. slam_path is already in the VO bag's frame.
        Pg = apply(A_anchor, slam_path)
        tg = (slam_t * 1e9 - t0) * 1e-9
        vg = np.linalg.norm(np.diff(Pg, axis=0), axis=1) / np.maximum(np.diff(tg), 1e-9)
        for i in range(len(tg)):
            tl.at(tg[i])
            tl.log("plots/z/cuvslam_slam", rr.Scalars(Pg[i, 2]))
            if i and vg[i - 1] < JUMP_MPS:
                tl.log("plots/speed/cuvslam_slam", rr.Scalars(vg[i - 1]))
    if Pl_lc is not None:
        # purple head along the loop-closed path, and its height / speed traces
        tl_s = (tl_lc - t0) * 1e-9
        vl = np.linalg.norm(np.diff(Pl_lc, axis=0), axis=1) / np.maximum(np.diff(tl_s), 1e-9)
        for i in range(len(tl_s)):
            tl.at(tl_s[i])
            for view in ("anchored", "sim3"):
                tl.log(f"{view}/head_lc", rr.Points3D([Pl_lc[i]], colors=[C_LC], radii=0.06))
            tl.log("plots/z/slamrs_lc", rr.Scalars(Pl_lc[i, 2]))
            if i and vl[i - 1] < JUMP_MPS:
                tl.log("plots/speed/slamrs_lc", rr.Scalars(vl[i - 1]))
        # each accepted closure appears when it happened: a labelled marker on both keyframes and
        # the edge between them; everything accepted so far stays visible from then on
        for n, (t_acc, p_cur, p_cand, label) in enumerate(edges_lc):
            tl.at((t_acc - t0) * 1e-9)
            so_far = edges_lc[: n + 1]
            for view in ("anchored", "sim3"):
                tl.log(f"{view}/slamrs_lc_edges", rr.LineStrips3D([[e[1], e[2]] for e in so_far], colors=[C_LC], radii=0.02))
                tl.log(f"{view}/slamrs_lc_closures",
                       rr.Points3D([e[1] for e in so_far] + [e[2] for e in so_far], colors=[C_LC], radii=0.05,
                                   labels=[e[3] for e in so_far] + [""] * len(so_far),
                                   show_labels=not a.video))   # a pile of labels mid-movie

    if not a.no_images:
        geo = dict(line.split() for line in open(a.log / "geometry.txt"))
        W, H, bpf = int(geo["width"]), int(geo["height"]), int(geo["bytes_per_frame"])
        raws = [np.memmap(a.log / f"cam{i}.raw", dtype=np.uint8, mode="r") for i in (1, 2, 3, 4)]
        sets, _ = load_framesets(a.log)
        for t, offs in sets[:: a.image_stride]:
            tl.at((t - t0) * 1e-9)
            for k, o in enumerate(offs):
                img = cv2.resize(raws[k][o:o + bpf].reshape(H, W), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
                tl.log(cams[k], rr.Image(img[::-1, ::-1].copy()).compress(jpeg_quality=70))  # mounted inverted
    if a.video:
        render_video(a, events, client, max(e[0] for e in events))
        rr.disconnect()
        os.killpg(viewer.pid, signal.SIGTERM)   # the launcher and the viewer it started
        print(f"saved {a.video}")
    else:
        print(f"saved {a.save}" if a.save else "done")


def render_video(a, events, client, t_end):
    """Stream the buffered events into the headless viewer in time order, one screenshot per frame
    of the movie (--video-speed seconds of log per second of video), then encode with ffmpeg."""
    rec = rr.get_global_data_recording()
    events.sort(key=lambda e: (e[0], e[1]))
    dt = a.video_speed / a.video_fps
    n = int(np.ceil(t_end / dt)) + 1
    frames = Path(tempfile.mkdtemp(prefix="rerun_compare_video_"))
    j = 0
    t_wall = time.time()
    for k in range(n):
        while j < len(events) and events[j][0] <= k * dt:
            t, _, path, arch = events[j]
            rr.set_time("time", duration=t)
            rr.log(path, arch)
            j += 1
        rec.flush(timeout_sec=10)
        time.sleep(0.08)          # let the viewer ingest what was just sent before it draws
        client.save_screenshot(str(frames / f"{k:05d}.png"))
        if k % 100 == 0:
            print(f"  frame {k}/{n} (t = {k * dt:.1f} s), {time.time() - t_wall:.0f} s")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.video_fps), "-i", str(frames / "%05d.png"),
                    "-vf", f"tpad=stop_mode=clone:stop_duration={a.video_hold},scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(a.video)], check=True)
    shutil.rmtree(frames)


if __name__ == "__main__":
    main()
