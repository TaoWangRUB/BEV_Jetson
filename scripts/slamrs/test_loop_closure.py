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
    r = Retriever(hb, dict(CFG["retrieval"], min_path_m=0.0))

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
    # the travelled-path gate (D1b): standing still is old enough and close, but no path lies
    # between the keyframes; going away and coming back does
    r2 = Retriever(HashBow(16, 0), dict(CFG["retrieval"], min_path_m=2.0))
    r2.add(0, KF(0, [20, 0, 0]), F())
    r2.add(1, KF(20 * 10 ** 9, [20.01, 0, 0]), F())
    assert r2.candidates(1, F()) == [], r2.candidates(1, F())
    r2.add(2, KF(30 * 10 ** 9, [23, 0, 0]), F())           # away: 3 m out ...
    r2.add(3, KF(45 * 10 ** 9, [20.02, 0, 0]), F())        # ... and back: path 6 m, distance 2 cm
    assert sorted(x[0] for x in r2.candidates(3, F())) == [0, 1], r2.candidates(3, F())   # nearest first


def test_backend_throttle_and_bad_edge_drop(tmp_path=None):
    import tempfile
    from loop_closure.backend import LoopClosureBackend
    from loop_closure.keyframes import KeyframeRecord
    rng = np.random.RandomState(4)
    out = Path(tmp_path or tempfile.mkdtemp())
    cfg = json.loads(json.dumps(CFG))
    cfg["backend"]["throttle_s"] = 2.0                     # the throttle itself is under test; default is off
    be = LoopClosureBackend(RIG, cfg, out, log=lambda *a: None)
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


def test_posegraph_jacobian_matches_finite_differences():
    truth = _square_lap()
    vio = _drifted(truth, yaw_rate=0.004, scale=1.05)
    g = PoseGraph4DoF(CFG["graph"])
    for k, T in enumerate(vio):
        g.add_node(k * 10 ** 9, T)
    g.add_loop_edge(len(vio) - 1, 0, inv(truth[-1]) @ truth[0], 0.05, np.radians(1.0))
    g.edges[-1].weight = 0.7                                     # a Huber-reweighted loop edge
    x = g._pack() + np.random.RandomState(5).normal(scale=0.05, size=g._pack().shape)
    J = g._jacobian(x, g.edges).toarray()
    eps = 1e-6
    Jn = np.empty_like(J)
    for c in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[c] += eps
        xm[c] -= eps
        Jn[:, c] = (g._residuals(xp, g.edges) - g._residuals(xm, g.edges)) / (2 * eps)
    assert np.abs(J - Jn).max() < 1e-4 * max(1.0, np.abs(Jn).max()), np.abs(J - Jn).max()
    assert J.shape == (4 * len(g.edges), 4 * (len(vio) - 1))


def test_keyframe_record_keeps_observed_landmarks():
    """D2b: a keyframe's record holds every landmark it observed once that landmark has a
    position, not only the ones it hosts; positions follow the keyframe's own latest pose."""
    from loop_closure.keyframes import KeyframeTracker

    class Flow:
        camera_count = 4

        def __init__(self, t, ids):
            self.t_ns, self._ids = t, np.asarray(ids, np.int64)

        def ids(self, c):
            return self._ids if c == 0 else np.zeros(0, np.int64)

        def positions(self, c):
            return np.zeros((len(self.ids(c)), 2))

    class Snap:
        def __init__(self, kf_ids, wt, hosts, lids, lpos, marg):
            self.kf_ids, self.window_t_ns, self.marginalized = kf_ids, wt, marg
            self.window_poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (len(wt), 1))   # identity poses
            self.landmark_hosts, self.landmark_ids, self.landmark_positions = hosts, lids, lpos

    ims = [np.zeros((4, 4), np.uint8)] * 4
    tr = KeyframeTracker()
    # keyframe 10 observes ids 1, 2, 3; 1 is hosted by 10, 2 by the older keyframe 5, 3 has no position yet
    out = tr.update(10, ims, Snap([5, 10], [5, 10], [10, 5], [1, 2], [[1, 0, 0], [0, 2, 0]], []), Flow(10, [1, 2, 3]))
    assert out == [] and 10 in tr.pending
    # next frame: 3 is triangulated, hosted by 11
    out = tr.update(11, ims, Snap([5, 10, 11], [5, 10, 11], [10, 5, 11], [1, 2, 3],
                                  [[1, 0, 0], [0, 2, 0], [0, 0, 3]], []), Flow(11, [1, 2, 3]))
    assert out == []
    # keyframe 10 is marginalised
    out = tr.update(12, ims, Snap([5, 11], [5, 11, 12], [5, 11], [2, 3], [[0, 2, 0], [0, 0, 3]], [10]), Flow(12, [2, 3]))
    assert len(out) == 1
    kf = out[0]
    assert kf.lm_ids.tolist() == [1, 2, 3]
    assert kf.lm_hosted.tolist() == [True, False, False]
    assert np.allclose(kf.lm_pos, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert kf.lm_age.tolist() == [2, 2, 1]


def _flow_and_snap():
    class Flow:
        camera_count = 4

        def __init__(self, t, ids, uv=None):
            self.t_ns, self._ids = t, np.asarray(ids, np.int64)
            self._uv = np.asarray(uv, np.float64) if uv is not None else np.zeros((len(self._ids), 2))

        def ids(self, c):
            return self._ids if c == 0 else np.zeros(0, np.int64)

        def positions(self, c):
            return self._uv if c == 0 else np.zeros((0, 2))

    class Snap:
        def __init__(self, kf_ids, wt, poses, hosts, lids, lpos, marg):
            self.kf_ids, self.window_t_ns, self.marginalized = kf_ids, wt, marg
            self.window_poses = np.asarray(poses, dtype=np.float64)
            self.landmark_hosts, self.landmark_ids, self.landmark_positions = hosts, lids, np.asarray(lpos, float)

    return Flow, Snap


def test_tracker_measures_parallax_and_observations():
    """D2c: parallax is the angle at the landmark between the first and latest rig positions that
    observed it; a landmark seen without motion has none and is classified far."""
    from loop_closure.keyframes import CLASS_3D, CLASS_FAR, KeyframeTracker
    Flow, Snap = _flow_and_snap()
    ims = [np.zeros((4, 4), np.uint8)] * 4
    tr = KeyframeTracker()
    at = lambda x: [x, 0, 0, 0, 0, 0, 1.0]
    # landmark 1 at [0, 2, 0], seen from x = 0 .. 1 over six frames; landmark 2 at [5, 0, 0] seen the same way
    L = [[0, 2, 0], [5, 0, 0]]
    for k, x in enumerate(np.linspace(0, 1, 6)):
        t = 10 + k
        kfs = [10] if k < 5 else [10, 15]
        tr.update(t, ims, Snap(kfs, [10, t] if t != 10 else [10], [at(0), at(x)] if t != 10 else [at(0)],
                               [10, 10], [1, 2], L, []), Flow(t, [1, 2]))
    out = tr.update(16, ims, Snap([15], [15, 16], [at(1), at(1)], [15], [3], [[0, 0, 9]], [10]), Flow(16, [3]))
    assert len(out) == 1
    kf = out[0]
    assert kf.lm_ids.tolist() == [1, 2]
    assert kf.lm_nobs.tolist() == [6, 6]
    assert abs(kf.lm_parallax_deg[0] - np.degrees(np.arctan(0.5))) < 1e-6, kf.lm_parallax_deg   # 26.6 deg
    assert abs(kf.lm_parallax_deg[1]) < 1e-6                                                 # moved along its ray
    assert np.allclose(kf.lm_baseline, [1.0, 1.0])
    cls = kf.depth_class(dict(min_obs=5, min_parallax_deg=2.0, far_parallax_deg=1.0))
    assert cls.tolist() == [CLASS_3D, CLASS_FAR]
    assert kf.depth_class(dict(min_obs=7, min_parallax_deg=2.0, far_parallax_deg=1.0)).tolist() == [2, 2]


def test_tracker_refresh_keeps_only_consistent_positions():
    """D2c: after finalisation a landmark's position is replaced from a later snapshot only if the
    keyframe's own observation still projects within refresh_max_px under the frozen pose."""
    from loop_closure.keyframes import KeyframeTracker
    Flow, Snap = _flow_and_snap()
    ims = [np.zeros((RIG.cams[0].height, RIG.cams[0].width), np.uint8)] * 4
    tr = KeyframeTracker(rig=RIG, cfg=dict(refresh_enabled=True, refresh_max_px=2.0))
    # two landmarks in front of camera 1 of a keyframe at the identity pose
    Tc = RIG.T_imu_cam[0]
    P_cam = np.array([[0.2, 0.1, 2.0], [-0.3, 0.2, 3.0]])
    L = P_cam @ Tc[:3, :3].T + Tc[:3, 3]                    # rig frame == world frame (identity pose)
    uv, ok = RIG.cams[0].project(P_cam)
    assert ok.all()
    I = [0, 0, 0, 0, 0, 0, 1.0]
    tr.update(10, ims, Snap([10], [10], [I], [10, 10], [1, 2], L, []), Flow(10, [1, 2], uv))
    out = tr.update(11, ims, Snap([11], [10, 11], [I, I], [10, 10], [1, 2], L, [10]), Flow(11, [1, 2], uv))
    assert len(out) == 1 and out[0].lm_refreshed.tolist() == [0, 0]
    kf = out[0]
    # later snapshot: landmark 1 moves 1 cm along its ray (consistent), landmark 2 moves 0.5 m sideways
    L2 = L.copy()
    L2[0] = (P_cam[0] * 1.005) @ Tc[:3, :3].T + Tc[:3, 3]
    L2[1] = L[1] + np.array([0.5, 0.0, 0.0])
    tr.update(12, ims, Snap([11], [11, 12], [I, I], [11, 11], [1, 2], L2, []), Flow(12, [1, 2], uv))
    assert kf.lm_refreshed.tolist() == [1, 0], kf.lm_refreshed
    assert np.allclose(kf.lm_pos[0], L2[0]) and np.allclose(kf.lm_pos[1], L[1])


def test_refine_uses_far_landmarks_as_directions():
    """D2c: far landmarks constrain the rotation only; with them a yaw error in the guess is
    recovered and they count as inliers although their stored depth is meaningless."""
    from loop_closure.verify import refine
    rng = np.random.RandomState(6)
    near = rng.uniform([-3, -3, -1], [3, 3, 2], size=(60, 3))
    far_dir = rng.normal(size=(80, 3)); far_dir /= np.linalg.norm(far_dir, axis=1, keepdims=True)
    far_true = far_dir * 300.0                          # where they really are
    far_stored = far_dir * 2.0                          # the depth the record froze: wrong by 150x
    T_true = se3(Rz(0.15), [0.3, -0.2, 0.05])           # T_cur_from_cand
    P_store = np.vstack([near, far_stored]); P_real = np.vstack([near, far_true])
    far = np.r_[np.zeros(60, bool), np.ones(80, bool)]
    cam_of, B, seen = [], [], []
    for n, p in enumerate(P_real):
        best = None
        for j in range(4):
            Tc = RIG.T_cam_imu[j] @ T_true
            pc = Tc[:3, :3] @ p + Tc[:3, 3]
            uv, ok = RIG.cams[j].project(pc[None])
            if ok[0] and pc[2] > 0.2 and (best is None or pc[2] > best[0]):
                best = (pc[2], j, pc / np.linalg.norm(pc))
        if best is None:
            continue                                     # outside every camera: no observation
        seen.append(n); cam_of.append(best[1]); B.append(best[2] + rng.normal(scale=0.001, size=3))
    seen, cam_of, B = np.array(seen), np.array(cam_of), np.array(B)
    P_store, far = P_store[seen], far[seen]
    assert far.sum() > 60 and (~far).sum() > 50
    T0 = se3(Rz(0.15 + 0.05), [0.3, -0.2, 0.05])        # 2.9 deg of yaw error in the guess
    T, inl, err = refine(T0, P_store, B, cam_of, RIG, CFG["verify"], CFG["features"]["min_bearing_z"], far=far)
    D = inv(T_true) @ T
    assert rot_angle_deg(D[:3, :3]) < 0.3, rot_angle_deg(D[:3, :3])
    assert np.linalg.norm(D[:3, 3]) < 0.03
    assert inl[far].sum() > 0.9 * far.sum() and inl[~far].sum() > 0.9 * (~far).sum(), (inl[far].sum(), inl[~far].sum())
    # the same far points treated as 3D points are all outliers: that is why they must not count
    T_b, inl_b, _ = refine(T0, P_store, B, cam_of, RIG, CFG["verify"], CFG["features"]["min_bearing_z"], far=None)
    assert inl_b[far].sum() < 10


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
