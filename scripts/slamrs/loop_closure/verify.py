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
    method: str = ""
    # KLT stage counts over every source keyframe and camera pair (a landmark seen by two source
    # cameras, or landing in two current cameras, counts once per pair): why a candidate died
    n_sel: int = 0            # landmarks with an observation and enough age
    n_proj: int = 0           # ... that project into a current camera with the guess
    n_klt: int = 0            # ... that Lucas-Kanade tracked forward
    n_fb: int = 0             # ... and tracked back to within klt_fb_max_px
    n_err: int = 0            # ... with a tracking error under klt_max_err
    n_ncc: int = 0            # ... whose patch NCC >= klt_ncc_min (= the correspondences used)
    # depth confidence (design D2c): what the correspondences were made of
    n_weak: int = 0           # landmarks skipped for lack of parallax and observations
    tracked3d: int = 0        # correspondences on landmarks with a trusted depth (the gates' denominator)
    far_tracked: int = 0      # correspondences on far / zero-parallax landmarks (bearing-only support)
    far_inliers: int = 0      # ... of which consistent with the refined rotation


def match_pair(fa, fb, cfg):
    """Mutual Hamming matches from a's landmark descriptors to b's detected keypoints.
    Returns (landmark indices into a's keyframe landmarks, keypoint indices into b)."""
    if len(fa.lm_desc) < 2 or len(fb.desc) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    da, db = fa.lm_desc, fb.desc
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
            ia.append(fa.lm_idx[qa])
            ib.append(tb)
    o = np.argsort(ia, kind="stable")
    return np.asarray(ia, np.int64)[o], np.asarray(ib, np.int64)[o]


def _ncc(a, b):
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-6 else -1.0


def klt_pairs(src, cur, T_cur_from_src, rig, cfg, min_z):
    """cuVSLAM's correspondence step: project the source keyframe's landmarks into the current
    cameras with the pose guess, track their stored appearance from the source image with
    Lucas-Kanade starting at the prediction, keep tracks that pass an NCC check. Returns
    [(i, j, P (n,3) in the source rig frame, bearings (n,3) in current camera j)].

    Why not descriptors: on these raw fisheye frames ORB descriptors of the same corner two
    keyframes apart are nearly as far apart as random ones (run5: Hamming 80 vs 122), and the
    detector misses most tracked corners. Tracking a patch from a predicted position needs no
    detector and no descriptor, only a guess within the pyramid's reach.

    Also returns the stage counts (see Hypothesis.n_*), so a candidate that yields nothing says
    at which stage it died. Each returned pair carries a per-correspondence class: CLASS_3D for
    landmarks with a trusted depth, CLASS_FAR for zero-parallax ones, used as directions only
    (design D2c); CLASS_WEAK landmarks are not tracked.
    """
    from .keyframes import CLASS_FAR, CLASS_WEAK
    out = []
    stats = dict(sel=0, proj=0, klt=0, fb=0, err=0, ncc=0, weak=0)
    if src.lm_obs is None or src.images is None or cur.images is None or len(src.lm_ids) == 0:
        return out, stats
    fs = src.features
    if getattr(src, "dirty", False):
        fs.lm_pos_rig = src.lm_pos_rig()
        src.dirty = False
    win, lev = int(cfg["klt_win"]), int(cfg["klt_levels"])
    half = int(cfg["klt_patch"]) // 2
    ncc_min, max_err = float(cfg["klt_ncc_min"]), float(cfg["klt_max_err"])
    pos = {int(i): n for n, i in enumerate(src.lm_ids)}
    min_age = int(cfg.get("min_landmark_age", 1))
    # the confidence thresholds live in the features config; the backend copies them into cfg
    cls = src.depth_class(cfg) if hasattr(src, "depth_class") else np.zeros(len(src.lm_ids), np.int64)
    for i, (ids, uv_s) in src.lm_obs.items():
        cand = [(n, pos[int(k)]) for n, k in enumerate(ids) if int(k) in pos and src.lm_age[pos[int(k)]] >= min_age]
        sel = [(n, l) for n, l in cand if cls[l] != CLASS_WEAK]
        stats["weak"] += len(cand) - len(sel)
        stats["sel"] += len(sel)
        if len(sel) < 3:
            continue
        n_obs = np.array([s[0] for s in sel]); l_idx = np.array([s[1] for s in sel])
        P_src = fs.lm_pos_rig[l_idx].copy()
        kind_sel = cls[l_idx]
        far_sel = kind_sel == CLASS_FAR
        if far_sel.any():
            # a far landmark is a direction: the host camera's observation ray, in the rig frame.
            # (Not the ray from the rig origin to the frozen point: at the frozen, false depth
            # that ray is off by the camera's lever arm, run6: 0 of 210 far inliers.)
            d = rig.cams[i].unproject(uv_s[n_obs[far_sel]]) @ rig.T_imu_cam[i][:3, :3].T
            P_src[far_sel] = d
        P_cur = P_src @ T_cur_from_src[:3, :3].T + T_cur_from_src[:3, 3]
        P_cur[far_sel] = P_src[far_sel] @ T_cur_from_src[:3, :3].T          # rotation only
        for j, cam in enumerate(rig.cams):
            Tc = rig.T_cam_imu[j]
            Pc = P_cur @ Tc[:3, :3].T + Tc[:3, 3]
            Pc[far_sel] = P_cur[far_sel] @ Tc[:3, :3].T
            uv_pred, ok = cam.project(Pc)
            ok &= Pc[:, 2] > 0.02
            stats["proj"] += int(ok.sum())
            if ok.sum() < 3:
                continue
            k_idx = np.nonzero(ok)[0]
            p0 = uv_s[n_obs[k_idx]].astype(np.float32).reshape(-1, 1, 2)
            p1 = uv_pred[k_idx].astype(np.float32).reshape(-1, 1, 2).copy()
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                src.images[i], cur.images[j], p0, p1, winSize=(win, win), maxLevel=lev,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            # forward-backward check (Basalt's frontend does the same): track back to the source
            # and keep only tracks that return to where they started
            p0b = p0.copy()
            p0b, stb, _ = cv2.calcOpticalFlowPyrLK(
                cur.images[j], src.images[i], p1, p0b, winSize=(win, win), maxLevel=lev,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            fb = np.linalg.norm(p0b.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
            stats["klt"] += int((st.ravel() == 1).sum())
            p1 = p1.reshape(-1, 2); st = (st.ravel() == 1) & (stb.ravel() == 1) & (fb <= float(cfg["klt_fb_max_px"]))
            err = err.ravel()
            stats["fb"] += int(st.sum())
            stats["err"] += int((st & (err < max_err)).sum())
            keep = []
            for n in np.nonzero(st & (err < max_err))[0]:
                x0, y0 = p0[n, 0]; x1, y1 = p1[n]
                if not (half <= x1 < cam.width - half and half <= y1 < cam.height - half):
                    continue
                pa = cv2.getRectSubPix(src.images[i], (2 * half + 1, 2 * half + 1), (float(x0), float(y0)))
                pb = cv2.getRectSubPix(cur.images[j], (2 * half + 1, 2 * half + 1), (float(x1), float(y1)))
                if _ncc(pa, pb) >= ncc_min:
                    keep.append(n)
            stats["ncc"] += len(keep)
            if len(keep) < 3:
                continue
            keep = np.array(keep)
            out.append((i, j, P_src[k_idx[keep]], cam.unproject(p1[keep]), kind_sel[k_idx[keep]]))
    return out, stats


def orb_pairs(fs, fk, cfg):
    """The descriptor route: mutual ORB matches from the source keyframe's landmark descriptors to
    the current keyframe's detections, per camera pair. Needs a core built with ORB on
    (features.orb_enabled); the fallback for a candidate KLT cannot reach, or for bare features.
    Every correspondence is CLASS_3D: bare features carry no confidence."""
    found = []
    for i, fa in enumerate(fs.cams):
        for j, fb in enumerate(fk.cams):
            ia, ib = match_pair(fa, fb, cfg)
            if len(ia):
                found.append((i, j, fs.lm_pos_rig[ia], fb.bearing[ib], np.zeros(len(ia), np.int64)))
    return found


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


def _residuals(x, P_rig, bearings, cam_of, T_cam_imu, min_z, far=None, huge=10.0):
    """Normalised-plane residuals of every correspondence under T = (rotvec, t). Rows flagged
    `far` are directions: their point is taken at infinity along its bearing from the candidate
    rig, so only the rotation acts on them (design D2c)."""
    T = se3(rotvec_to_R(x[:3]), x[3:6])
    out = np.empty(len(P_rig) * 2)
    if far is None:
        far = np.zeros(len(P_rig), bool)
    for j, Tci in enumerate(T_cam_imu):
        m = cam_of == j
        if not m.any():
            continue
        Tc = Tci @ T
        P = (Tc[:3, :3] @ P_rig[m].T).T + Tc[:3, 3]
        fm = far[m]
        if fm.any():
            d = P_rig[m][fm]
            d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            P[fm] = (Tc[:3, :3] @ d.T).T
        xy_obs, ok = _normalised(bearings[m], min_z)
        z = P[:, 2]
        okp = ok & (z > 1e-3)
        xy = np.zeros((len(P), 2))
        xy[okp] = P[okp, :2] / z[okp, None]
        r = xy - xy_obs
        r[~okp] = huge
        out[np.repeat(m, 2)] = r.ravel()
    return out


def refine(T0, P_rig, bearings, cam_of, rig, cfg, min_z, far=None):
    """Joint non-linear refinement over all pairs; returns (T, inlier mask, per-point error)."""
    x0 = np.concatenate([R_to_rotvec(T0[:3, :3]), T0[:3, 3]])
    thr = float(cfg["ransac_thresh_norm"])
    res = least_squares(_residuals, x0, args=(P_rig, bearings, cam_of, rig.T_cam_imu, min_z, far),
                        loss="huber", f_scale=thr, max_nfev=200, xtol=1e-10, ftol=1e-10)
    T = se3(rotvec_to_R(res.x[:3]), res.x[3:6])
    r = _residuals(res.x, P_rig, bearings, cam_of, rig.T_cam_imu, min_z, far).reshape(-1, 2)
    err = np.linalg.norm(r, axis=1)
    return T, err <= thr, err


def _lerp_inv(a, b, v):
    return float(np.clip((v - a) / (b - a), 0.0, 1.0))


def verify(cand_idx, cur_idx, route, cand, cur, rig, T_guess, cfg, min_z, neighbours=()):
    """cand, cur: KeyframeRecords (with .features and .images) or bare KeyframeFeatures.
    T_guess: odometry's T_cur_from_cand. neighbours: [(record or features, T_cand_from_nb)] -
    the keyframes in sight of the candidate, whose landmarks join step 2 (cuVSLAM fetches every
    landmark near the guess, then picks the single keyframe owning most of them for step 1).

    Correspondences come from KLT tracking of the stored landmark appearance (klt_pairs) when
    images and observations are available; ORB descriptor matching (orb_pairs) serves bare
    features (no images) and, with cfg["orb_fallback"], a candidate KLT found nothing on."""
    from .keyframes import CLASS_3D, CLASS_FAR
    h = Hypothesis(cand_idx, cur_idx, route)
    P_all, B_all, cam_all, kind_all, counts = [], [], [], [], {}
    pairs = []
    fk = getattr(cur, "features", cur)
    sources = [(cand, np.eye(4), True)] + [(nb, T, False) for nb, T in neighbours]
    for s, T_cand_from_src, own in sources:
        fs = getattr(s, "features", s)
        found = []
        if getattr(s, "images", None) is not None and getattr(cur, "images", None) is not None \
                and getattr(s, "lm_obs", None) is not None:
            # guess for this source: T_cur_from_src = T_cur_from_cand @ T_cand_from_src
            found, st = klt_pairs(s, cur, T_guess @ T_cand_from_src, rig, cfg, min_z)
            h.n_sel += st["sel"]; h.n_proj += st["proj"]; h.n_klt += st["klt"]
            h.n_fb += st["fb"]; h.n_err += st["err"]; h.n_ncc += st["ncc"]; h.n_weak += st["weak"]
            h.method = "klt"
            if not found and bool(cfg.get("orb_fallback", False)):
                found = orb_pairs(fs, fk, cfg)
                h.method = "klt+orb"
        else:
            h.method = "orb"
            found = orb_pairs(fs, fk, cfg)
        for i, j, P_src, B, kind in found:
            if own:
                counts[(i, j)] = int(len(P_src))
            P = P_src @ T_cand_from_src[:3, :3].T + T_cand_from_src[:3, 3]
            P[kind == CLASS_FAR] = P_src[kind == CLASS_FAR] @ T_cand_from_src[:3, :3].T   # directions
            # step 1 may start from any source keyframe's pair: cuVSLAM picks the keyframe
            # owning most of the fetched landmarks, which need not be the candidate itself
            pairs.append((i, j, P, B, kind))
            P_all.append(P)
            B_all.append(B)
            cam_all.append(np.full(len(P), j, np.int64))
            kind_all.append(kind)
    h.pair_counts = {f"{i + 1}-{j + 1}": n for (i, j), n in sorted(counts.items())}
    h.tracked = int(sum(len(p) for p in P_all))
    if h.tracked == 0:
        h.reason = "no_matches"
        return h
    kind_all = np.concatenate(kind_all)
    h.tracked3d = int((kind_all == CLASS_3D).sum())
    h.far_tracked = int((kind_all == CLASS_FAR).sum())
    # step 1: best single pair, on trusted depths only (a direction cannot place the rig)
    best = None
    for i, j, P, B, kind in pairs:
        m3 = kind == CLASS_3D
        if m3.sum() < int(cfg["min_pair_matches"]):
            continue
        T1, mask = pnp_pair(P[m3], B[m3], rig.T_imu_cam[j], cfg, min_z)
        if T1 is None:
            continue
        if best is None or mask.sum() > best[0]:
            best = (int(mask.sum()), T1, (i, j))
    if best is None:
        h.reason = "no_pair_solution"
        return h
    h.inliers1, T1, h.best_pair = best
    # step 2: joint refinement from T1 over everything; far landmarks constrain the rotation only
    P_all, B_all, cam_all = np.concatenate(P_all), np.concatenate(B_all), np.concatenate(cam_all)
    T2, inl, _ = refine(T1, P_all, B_all, cam_all, rig, cfg, min_z, far=kind_all == CLASS_FAR)
    h.inliers2 = int(inl[kind_all == CLASS_3D].sum())
    h.far_inliers = int(inl[kind_all == CLASS_FAR].sum())
    # the gates count landmarks whose depth is trusted: a zero-parallax landmark's 3D error says
    # nothing about the hypothesis, so it is neither an inlier nor an outlier (design D2c)
    h.ratio = h.inliers2 / max(h.tracked3d, 1)
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
