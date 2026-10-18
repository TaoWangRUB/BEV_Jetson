"""Synthetic tests for the loop-closure backend (design D7). Runs with pytest or as a script.

The real KB4 + extrinsics are used throughout; what is synthetic is the scene and the trajectory,
so the checks are about the solver and the gates, not about ORB on real pixels (that is run5).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loop_closure.camera import Rig, Rz, inv, rot_angle_deg, rotvec_to_R, se3  # noqa: E402
from loop_closure.features import CamFeatures, HashBow, KeyframeFeatures  # noqa: E402
from loop_closure.posegraph import PoseGraph4DoF  # noqa: E402
from loop_closure.retrieval import Retriever  # noqa: E402
from loop_closure.verify import verify  # noqa: E402

REPO = HERE.parents[1]
CFG = json.load(open(REPO / "config/slamrs/bev_lc_config.json"))
RIG = Rig.from_json(REPO / "config/slamrs/bev_calib_s2.json", CFG["features"]["theta_max_deg"])


def test_kb4_roundtrip():
    rng = np.random.RandomState(0)
    cam = RIG.cams[0]
    b = rng.normal(size=(2000, 3))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    uv, ok = cam.project(b)
    assert ok.sum() > 500
    b2 = cam.unproject(uv[ok])
    ang = np.degrees(np.arccos(np.clip((b[ok] * b2).sum(1), -1, 1)))
    assert ang.max() < 0.05, ang.max()
    # a ray past 90 deg incidence is representable (the ~192 deg lens); the sensor is narrower
    # than the image circle, so only the corners reach that far
    th = np.radians(92)
    phi = np.arctan2(cam.height - cam.cy, cam.width - cam.cx)
    uv, ok = cam.project(np.array([[np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)]]))
    assert ok[0], uv


def test_hashbow_scores_near_duplicates():
    rng = np.random.RandomState(1)
    hb = HashBow(16, 0)
    base = rng.randint(0, 256, size=(600, 32), dtype=np.uint8)
    noisy = base.copy()
    flip = rng.rand(*np.unpackbits(noisy, axis=1).shape) < 0.03     # ~8 of 256 bits per descriptor
    noisy = np.packbits(np.unpackbits(noisy, axis=1) ^ flip.astype(np.uint8), axis=1)
    others = [rng.randint(0, 256, size=(600, 32), dtype=np.uint8) for _ in range(20)]
    hb.add(0, hb.bow(base))
    for k, d in enumerate(others):
        hb.add(k + 1, hb.bow(d))
    res = hb.query(hb.bow(noisy), 3)
    assert res[0][0] == 0 and res[0][1] > 0.4, res
    assert res[1][1] < 0.1, res
    assert hb.query(hb.bow(noisy), 3) == res   # deterministic


def _synth_keyframe(rng, T_world_rig, points_world, desc, assoc=True):
    """Features as `extract` would produce them, but from projected synthetic points."""
    T_rig_world = inv(T_world_rig)
    P_rig = (T_rig_world[:3, :3] @ points_world.T).T + T_rig_world[:3, 3]
    cams = []
    for k, cam in enumerate(RIG.cams):
        Tci = RIG.T_cam_imu[k]
        P = (Tci[:3, :3] @ P_rig.T).T + Tci[:3, 3]
        uv, ok = cam.project(P)
        ok &= P[:, 2] > 0.1
        idx = np.nonzero(ok)[0]
        uv_n = uv[idx] + rng.normal(scale=0.3, size=(len(idx), 2))
        if assoc:   # the candidate side: descriptors at the landmark projections
            cams.append(CamFeatures(uv_n, desc[idx], cam.unproject(uv_n), uv[idx], desc[idx], idx.astype(np.int64)))
        else:       # the current side: detections only
            cams.append(CamFeatures(uv_n, desc[idx], cam.unproject(uv_n), np.zeros((0, 2)),
                                    np.zeros((0, 32), np.uint8), np.zeros(0, np.int64)))
    return KeyframeFeatures(cams, P_rig, {}, int(sum(len(c.lm_idx) for c in cams)))


def _scene(rng, n=1500):
    pts = rng.uniform([-6, -6, -1.0], [6, 6, 2.5], size=(n, 3))
    desc = rng.randint(0, 256, size=(n, 32), dtype=np.uint8)
    return pts, desc


def test_verify_recovers_loop_pose_and_rejects_lookalike():
    rng = np.random.RandomState(2)
    pts, desc = _scene(rng)
    T_cand = se3(Rz(0.3), [0.5, -0.2, 0.0])
    T_cur_true = se3(Rz(0.3 + 0.25), [0.9, 0.4, 0.05])           # revisit from a different heading
    fc = _synth_keyframe(rng, T_cand, pts, desc, assoc=True)
    fk = _synth_keyframe(rng, T_cur_true, pts, desc, assoc=False)
    T_true = inv(T_cur_true) @ T_cand                              # T_cur_from_cand
    T_guess = se3(rotvec_to_R([0, 0, 0.03]), [0.0, 0.0, 0.0]) @ T_true @ se3(np.eye(3), [0.3, -0.2, 0.0])  # drifted odometry
    h = verify(0, 1, "spatial", fc, fk, RIG, T_guess, CFG["verify"], CFG["features"]["min_bearing_z"])
    assert h.accepted, (h.reason, h.tracked, h.inliers1, h.inliers2, h.ratio, h.pair_counts)
    D = inv(T_true) @ h.T_cur_from_cand
    assert np.linalg.norm(D[:3, 3]) < 0.02, np.linalg.norm(D[:3, 3])
    assert rot_angle_deg(D[:3, :3]) < 0.5, rot_angle_deg(D[:3, :3])
    assert h.inliers2 >= 40 and h.ratio > 0.5
    # a look-alike: same descriptors, geometry shuffled -> no consistent pose
    perm = rng.permutation(len(pts))
    fk_bad = _synth_keyframe(rng, T_cur_true, pts[perm], desc, assoc=False)
    h2 = verify(0, 1, "appearance", fc, fk_bad, RIG, T_guess, CFG["verify"], CFG["features"]["min_bearing_z"])
    assert not h2.accepted, (h2.reason, h2.inliers2, h2.ratio)
    assert h2.reason != "ok"


def test_verify_is_deterministic():
    rng = np.random.RandomState(3)
    pts, desc = _scene(rng)
    T_cand, T_cur = se3(np.eye(3), [0, 0, 0]), se3(Rz(0.1), [0.4, 0.1, 0])
    fc = _synth_keyframe(rng, T_cand, pts, desc)
    fk = _synth_keyframe(rng, T_cur, pts, desc, assoc=False)
    g = inv(T_cur) @ T_cand @ se3(np.eye(3), [0.15, 0.0, 0.0])   # the odometry guess is off by 15 cm
    a = verify(0, 1, "spatial", fc, fk, RIG, g, CFG["verify"], CFG["features"]["min_bearing_z"])
    b = verify(0, 1, "spatial", fc, fk, RIG, g, CFG["verify"], CFG["features"]["min_bearing_z"])
    assert a.accepted and b.accepted, (a.reason, b.reason)
    # and a perfect guess must not be rejected by the consistency gate either
    c = verify(0, 1, "spatial", fc, fk, RIG, inv(T_cur) @ T_cand, CFG["verify"], CFG["features"]["min_bearing_z"])
    assert c.accepted, c.reason
    assert np.array_equal(a.T_cur_from_cand, b.T_cur_from_cand) and a.inliers2 == b.inliers2


def _square_lap(n_per_side=10, side=4.0):
    """Keyframe poses around a square, returning to the start; heading along the path."""
    poses = []
    for leg in range(4):
        yaw = leg * np.pi / 2
        d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        start = {0: [0, 0, 0], 1: [side, 0, 0], 2: [side, side, 0], 3: [0, side, 0]}[leg]
        for k in range(n_per_side):
            poses.append(se3(Rz(yaw), np.array(start, float) + d * side * k / n_per_side))
    poses.append(se3(Rz(0.0), [0.0, 0.0, 0.0]))   # back at the start
    return poses


def _drifted(poses, yaw_rate=0.004, scale=1.03):
    """Integrate the true relative motions with a yaw bias and a scale error, like a drifting VIO."""
    out = [poses[0]]
    for k in range(1, len(poses)):
        rel = inv(poses[k - 1]) @ poses[k]
        rel = se3(Rz(yaw_rate) @ rel[:3, :3], rel[:3, 3] * scale)
        out.append(out[-1] @ rel)
    return out


def test_posegraph_closes_drift():
    truth = _square_lap()
    vio = _drifted(truth, yaw_rate=0.004, scale=1.05)           # ~0.25 deg/keyframe, 5 % scale
    g = PoseGraph4DoF(CFG["graph"])
    for k, T in enumerate(vio):
        g.add_node(k * 10 ** 9, T)
    end_err_vio = np.linalg.norm(vio[-1][:3, 3] - truth[-1][:3, 3])
    assert end_err_vio > 0.3, end_err_vio                       # the drift is real (~0.47 m)
    # loop edge: last node sees the first, measured exactly
    T_last_from_first = inv(truth[-1]) @ truth[0]
    g.add_loop_edge(len(vio) - 1, 0, T_last_from_first, 0.05, np.radians(1.0))
    r = g.optimize()
    P = g.poses()
    assert r.max() < 3.0, r
    assert np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3]) < 0.08
    # the drift is redistributed: no single step larger than the true step + 10 cm
    steps = [np.linalg.norm(P[k][:3, 3] - P[k - 1][:3, 3]) for k in range(1, len(P))]
    true_steps = [np.linalg.norm(truth[k][:3, 3] - truth[k - 1][:3, 3]) for k in range(1, len(truth))]
    assert max(s - t for s, t in zip(steps, true_steps)) < 0.1
    # roll/pitch untouched: every rotation is still a pure yaw
    for T in P:
        assert abs(T[2, 2] - 1) < 1e-9
    # the pure-VIO poses used to build the graph are unchanged
    for k, T in enumerate(vio):
        assert np.array_equal(g.R_vio[k], T[:3, :3]) and np.array_equal(g.p_vio[k], T[:3, 3])


def test_retrieval_time_gate_and_radius():
    hb = HashBow(16, 0)
    r = Retriever(hb, CFG["retrieval"])

    class KF:
        def __init__(self, t, p):
            self.t_ns, self.T_vio = t, se3(np.eye(3), p)

    class F:
        bow = {}

    r.add(0, KF(0, [0, 0, 0]), F())
    r.add(1, KF(5 * 10 ** 9, [0.5, 0, 0]), F())           # near but too young for index 2
    r.add(2, KF(11 * 10 ** 9, [0.2, 0, 0]), F())
    c = r.candidates(2, F())
    assert [x[0] for x in c] == [0], c
    r.add(3, KF(40 * 10 ** 9, [20, 0, 0]), F())            # far from everything
    assert r.candidates(3, F()) == []


def test_backend_throttle_and_bad_edge_drop(tmp_path=None):
    import tempfile
    from loop_closure.backend import LoopClosureBackend
    from loop_closure.keyframes import KeyframeRecord
    rng = np.random.RandomState(4)
    out = Path(tmp_path or tempfile.mkdtemp())
    be = LoopClosureBackend(RIG, CFG, out, log=lambda *a: None)
    truth = _square_lap()
    vio = _drifted(truth, yaw_rate=0.004, scale=1.05)

    def kf(k):
        ims = [rng.randint(0, 255, size=(c.height, c.width), dtype=np.uint8) for c in RIG.cams]
        return KeyframeRecord(k * 10 ** 9, vio[k], ims, np.zeros(0, np.int64), np.zeros((0, 3)))

    for k in range(len(vio)):
        be.on_keyframe(kf(k))
        if k == 20:
            be.last_accept_t = 20 * 10 ** 9      # pretend a closure was accepted here
    notes = [r.get("note", "") for r in be.timings]
    assert notes[21] == "throttled" and notes[22] == "" , notes[19:24]   # throttle_s = 2 s
    # a good loop edge closes the lap; then a wrong one is dropped by the 3-sigma check
    be.graph.add_loop_edge(len(vio) - 1, 0, inv(truth[-1]) @ truth[0], 0.05, np.radians(1.0))
    be._solve_and_check()
    assert be.dropped == []
    P = be.graph.poses()
    assert np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3]) < 0.08
    bad = be.graph.add_loop_edge(30, 5, se3(np.eye(3), [3.0, 0.0, 0.0]), 0.05, np.radians(1.0))
    be.edges.append((None, bad))
    be._solve_and_check()
    assert len(be.dropped) == 1 and be.dropped[0][:2] == (30, 5), be.dropped
    assert bad not in be.graph.edges
    # frames between keyframes are corrected continuously (design D5)
    ts = np.arange(0, len(vio) * 10 ** 9, 10 ** 8)
    Tf = [vio[int(t // 10 ** 9)] @ se3(np.eye(3), [0.04 * ((t % 10 ** 9) / 1e8), 0, 0]) for t in ts]
    C = be.correct_frames(ts, Tf)
    steps = np.linalg.norm(np.diff(C[:, :3, 3], axis=0), axis=1)
    assert steps.max() < 0.25, steps.max()
    s = be.write()
    assert (out / "hypotheses.csv").exists() and (out / "timings.csv").exists() and s["keyframes"] == len(vio)


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
