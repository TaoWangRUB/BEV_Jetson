"""Geometric verification of a loop candidate: cuVSLAM's two-step solver, in Python.

Step 1 - per camera pair (candidate camera i, current camera j): match ORB descriptors (Basalt's
constants: Hamming <= 70, ratio 1.2, mutual), take the candidate's landmark-associated keypoints as
3D points in its rig frame, and solve PnP-RANSAC on the current keypoints' bearings. The pair with
most inliers gives the first pose (cuVSLAM: `LoopClosureSolverSimpleSingleKF` picks the keyframe
owning most tracked landmarks).

Step 2 - refine jointly over all pairs' matches from step 1's pose with a Huber loss, recount the
inliers, and apply cuVSLAM's gates (`LoopClosureSolverTwoStepsEasy`): >= 40 landmarks, an inlier
ratio of at least lerp(0.5 @ 40 -> 0.3 @ 200), and the refined pose closer to the first than to the
odometry guess. The covariance is scaled by `1 + (max-1) * (1 - ratio)`.

The relative pose returned is T_cur_from_cand: candidate-rig coordinates into current-rig
coordinates, i.e. T_cur^-1 T_cand.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import R_to_rotvec, inv, rot_angle_deg, rotvec_to_R, se3


@dataclass
class Hypothesis:
    cand: int
    cur: int
    route: str
    accepted: bool = False
    reason: str = ""
    T_cur_from_cand: np.ndarray = None
    sigma_t: float = 0.0
    sigma_yaw: float = 0.0
    tracked: int = 0          # matches with a landmark, all pairs
    inliers1: int = 0         # best pair's RANSAC inliers (step 1)
    inliers2: int = 0         # joint inliers (step 2)
    ratio: float = 0.0
    best_pair: tuple = (-1, -1)
    pair_counts: dict = field(default_factory=dict)
    guess_diff_m: float = 0.0
    guess_diff_deg: float = 0.0


def match_pair(fa, fb, cfg):
    """Mutual Hamming matches a->b restricted to a's landmark-associated keypoints.
    Returns (ia, ib) index arrays."""
    sel = np.nonzero(fa.lm >= 0)[0]
    if len(sel) < 2 or len(fb.desc) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    da, db = fa.desc[sel], fb.desc[: len(fb.desc)]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    hmax, ratio = int(cfg["hamming_max"]), float(cfg["ratio_test"])

    def best(d1, d2):
        out = {}
        for m in bf.knnMatch(d1, d2, k=2):
            if len(m) == 0:
                continue
            m0 = m[0]
            m1 = m[1].distance if len(m) > 1 else 10 ** 6
            if m0.distance < hmax and m0.distance * ratio <= m1:
                out[m0.queryIdx] = m0.trainIdx
        return out

    ab, ba = best(da, db), best(db, da)
    ia, ib = [], []
    for qa, tb in ab.items():
        if ba.get(tb) == qa:
            ia.append(sel[qa])
            ib.append(tb)
    o = np.argsort(ia, kind="stable")
    return np.asarray(ia, np.int64)[o], np.asarray(ib, np.int64)[o]


def _normalised(b, min_z):
    ok = b[:, 2] > min_z
    xy = np.zeros((len(b), 2))
    xy[ok] = b[ok, :2] / b[ok, 2:3]
    return xy, ok


def pnp_pair(P_rig, bearings, T_imu_cam_j, cfg, min_z):
    """P_rig (n,3) in the candidate rig frame observed as `bearings` in current camera j.
    Returns (T_cur_from_cand, inlier mask) or (None, None)."""
    xy, ok = _normalised(bearings, min_z)
    idx = np.nonzero(ok)[0]
    if len(idx) < 6:
        return None, None
    cv2.setRNGSeed(int(cfg["seed"]))
    ret, rvec, tvec, inl = cv2.solvePnPRansac(
        P_rig[idx].reshape(-1, 1, 3).astype(np.float64), xy[idx].reshape(-1, 1, 2).astype(np.float64),
        np.eye(3), None, iterationsCount=int(cfg["ransac_iters"]),
        reprojectionError=float(cfg["ransac_thresh_norm"]), confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ret or inl is None or len(inl) < 6:
        return None, None
    T_camj_from_cand = se3(rotvec_to_R(rvec.ravel()), tvec.ravel())
    mask = np.zeros(len(P_rig), bool)
    mask[idx[inl.ravel()]] = True
    return T_imu_cam_j @ T_camj_from_cand, mask


def _residuals(x, P_rig, bearings, cam_of, T_cam_imu, min_z, huge=10.0):
    T = se3(rotvec_to_R(x[:3]), x[3:6])
    out = np.empty(len(P_rig) * 2)
    for j, Tci in enumerate(T_cam_imu):
        m = cam_of == j
        if not m.any():
            continue
        Tc = Tci @ T
        P = (Tc[:3, :3] @ P_rig[m].T).T + Tc[:3, 3]
        xy_obs, ok = _normalised(bearings[m], min_z)
        z = P[:, 2]
        okp = ok & (z > 1e-3)
        xy = np.zeros((len(P), 2))
        xy[okp] = P[okp, :2] / z[okp, None]
        r = xy - xy_obs
        r[~okp] = huge
        out[np.repeat(m, 2)] = r.ravel()
    return out


def refine(T0, P_rig, bearings, cam_of, rig, cfg, min_z):
    """Joint non-linear refinement over all pairs; returns (T, inlier mask, per-point error)."""
    x0 = np.concatenate([R_to_rotvec(T0[:3, :3]), T0[:3, 3]])
    thr = float(cfg["ransac_thresh_norm"])
    res = least_squares(_residuals, x0, args=(P_rig, bearings, cam_of, rig.T_cam_imu, min_z),
                        loss="huber", f_scale=thr, max_nfev=200, xtol=1e-10, ftol=1e-10)
    T = se3(rotvec_to_R(res.x[:3]), res.x[3:6])
    r = _residuals(res.x, P_rig, bearings, cam_of, rig.T_cam_imu, min_z).reshape(-1, 2)
    err = np.linalg.norm(r, axis=1)
    return T, err <= thr, err


def _lerp_inv(a, b, v):
    return float(np.clip((v - a) / (b - a), 0.0, 1.0))


def verify(cand_idx, cur_idx, route, fc, fk, rig, T_guess, cfg, min_z):
    """fc: candidate KeyframeFeatures (has the 3D landmarks), fk: current KeyframeFeatures.
    T_guess: odometry's T_cur_from_cand."""
    h = Hypothesis(cand_idx, cur_idx, route)
    P_all, B_all, cam_all, counts = [], [], [], {}
    pairs = []
    for i, fa in enumerate(fc.cams):
        for j, fb in enumerate(fk.cams):
            ia, ib = match_pair(fa, fb, cfg)
            if len(ia) == 0:
                continue
            counts[(i, j)] = int(len(ia))
            P = fc.lm_pos_rig[fa.lm[ia]]
            B = fb.bearing[ib]
            pairs.append((i, j, P, B))
            P_all.append(P)
            B_all.append(B)
            cam_all.append(np.full(len(ia), j, np.int64))
    h.pair_counts = {f"{i + 1}-{j + 1}": n for (i, j), n in sorted(counts.items())}
    h.tracked = int(sum(counts.values()))
    if h.tracked == 0:
        h.reason = "no_matches"
        return h
    # step 1: best single pair
    best = None
    for i, j, P, B in pairs:
        if len(P) < int(cfg["min_pair_matches"]):
            continue
        T1, mask = pnp_pair(P, B, rig.T_imu_cam[j], cfg, min_z)
        if T1 is None:
            continue
        if best is None or mask.sum() > best[0]:
            best = (int(mask.sum()), T1, (i, j))
    if best is None:
        h.reason = "no_pair_solution"
        return h
    h.inliers1, T1, h.best_pair = best
    # step 2: joint refinement from T1
    P_all, B_all, cam_all = np.concatenate(P_all), np.concatenate(B_all), np.concatenate(cam_all)
    T2, inl, _ = refine(T1, P_all, B_all, cam_all, rig, cfg, min_z)
    h.inliers2 = int(inl.sum())
    h.ratio = h.inliers2 / h.tracked
    h.T_cur_from_cand = T2
    D = inv(T_guess) @ T2
    h.guess_diff_m = float(np.linalg.norm(D[:3, 3]))
    h.guess_diff_deg = float(rot_angle_deg(D[:3, :3]))
    # gates (cuVSLAM LoopClosureSolverTwoStepsEasy). The consistency gate asks that the refined
    # pose land closer to the first solution than to the odometry guess; unlike cuVSLAM it is
    # waived when the refined pose already agrees with the guess within a tolerance, since a
    # near-perfect guess would otherwise be rejected by construction.
    d12 = inv(T1) @ T2
    agrees = (h.guess_diff_m <= float(cfg["consistency_tol_m"])
              and h.guess_diff_deg <= float(cfg["consistency_tol_deg"]))
    if not agrees and np.linalg.norm(d12[:3, 3]) > h.guess_diff_m and rot_angle_deg(d12[:3, :3]) > h.guess_diff_deg:
        h.reason = "inconsistent_steps"
        return h
    if h.inliers2 < int(cfg["min_landmarks"]):
        h.reason = "few_landmarks"
        return h
    f = _lerp_inv(float(cfg["ratio_lo_at"]), float(cfg["ratio_hi_at"]), h.inliers2)
    thr = max((1 - f) * float(cfg["ratio_lo"]) + f * float(cfg["ratio_hi"]), float(cfg["ratio_hi"]))
    if h.ratio < thr:
        h.reason = f"low_ratio<{thr:.2f}"
        return h
    scale = 1.0 + (float(cfg["cov_scale_max"]) - 1.0) * (1.0 - h.ratio)
    h.sigma_t = float(cfg["base_sigma_t_m"]) * np.sqrt(scale)
    h.sigma_yaw = np.radians(float(cfg["base_sigma_yaw_deg"])) * np.sqrt(scale)
    h.accepted, h.reason = True, "ok"
    return h
