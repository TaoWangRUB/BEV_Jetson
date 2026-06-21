#!/usr/bin/env python3
"""Feature-based relative-rotation extrinsic calibration for the 4-camera surround fisheye rig.

The cameras sit ~90 deg apart with a thin (~37 deg) overlap at the fisheye edges. There is no
shared checkerboard, so we calibrate the ROTATIONS from natural scene features in each adjacent
overlap band:

  1. SIFT-match features between adjacent cameras (ring order cam3->cam4->cam1->cam2->cam3).
  2. Unproject matches to unit bearing rays via each camera's KB (OpenCV-fisheye) intrinsics.
  3. For a far scene the two cameras are related by a pure rotation: b_i = R_ij b_j. Estimate
     R_ij robustly (RANSAC + Kabsch/orthogonal-Procrustes on the inlier bearings).
  4. Anchor cam3 (front) at its current orientation, chain the relative rotations around the ring,
     and distribute the loop-closure residual equally (matrix-log) so the 4 rotations are consistent.
  5. Write rig_extrinsics_calibrated.yaml (rotations refined, translations/imu kept).

Convention (matches bev_panorama_node.cpp): R = quat 'camera->rig'; the upside-down mount adds a
180 deg roll Rz180=diag(-1,-1,1). Optical bearing b = Rz180 . R^T . g, so the optical rotation is
Q = R . Rz180 and R = Q . Rz180 (Rz180 is its own inverse).

Pure-rotation caveat: the rig baseline is ~3 cm, so nearby objects (<~1 m) carry parallax that
this rotation-only model can't absorb. Calibrate against a far/feature-rich scene for best results.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# Ring order by azimuth (panorama az=0=+Y front, +az toward +X):
#   cam3(+Y,0) -> cam4(+X,90) -> cam1(-Y,180) -> cam2(-X,270) -> cam3
RING = ["cam3", "cam4", "cam1", "cam2"]
PAIRS = [("cam3", "cam4"), ("cam4", "cam1"), ("cam1", "cam2"), ("cam2", "cam3")]
ANCHOR = "cam3"
RZ180 = np.diag([-1.0, -1.0, 1.0])


def load_intrinsics(calib_dir, cam):
    """Read the OpenCV-style KB yaml -> (K, D) for cv2.fisheye."""
    with open(os.path.join(calib_dir, cam + ".yaml")) as f:
        txt = "\n".join(l for l in f if not l.startswith("%YAML") and l.strip() != "---")
    y = yaml.safe_load(txt)
    pp, dp = y["projection_parameters"], y["distortion_parameters"]
    K = np.array([[pp["mu"], 0, pp["u0"]], [0, pp["mv"], pp["v0"]], [0, 0, 1]], float)
    D = np.array([dp["k2"], dp["k3"], dp["k4"], dp["k5"]], float)  # KB k2..k5 == cv2 fisheye D[0..3]
    return K, D, int(y["image_width"]), int(y["image_height"])


def quat_to_R(q_wxyz):
    """camera->rig rotation from quat wxyz (same formula as the panorama node)."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], float)


def R_to_quat(R):
    """rotation matrix -> quat wxyz (Shepperd's method)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def so3_log(R):
    """rotation matrix -> rotation vector (axis*angle)."""
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return axis * ang


def so3_exp(v):
    ang = np.linalg.norm(v)
    if ang < 1e-9:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def unproject(pts, K, D):
    """pixel points (N,2) -> unit bearing rays (N,3) in the optical frame (cv2 fisheye)."""
    p = pts.reshape(-1, 1, 2).astype(np.float64)
    norm = cv2.fisheye.undistortPoints(p, K, D).reshape(-1, 2)  # normalized (a,b)=(X/Z,Y/Z)
    b = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def kabsch(bi, bj):
    """least-squares rotation R s.t. bi ~ R bj (both (N,3) unit)."""
    H = bj.T @ bi
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def ransac_rotation(bi, bj, iters=2000, thresh_deg=0.5, seed=0):
    """robust R (bi ~ R bj) with RANSAC; returns (R, inlier_mask, residual_deg)."""
    rng = np.random.default_rng(seed)
    n = len(bi)
    thr = np.deg2rad(thresh_deg)
    best_in = np.zeros(n, bool)
    if n < 3:
        return np.eye(3), best_in, np.inf
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R = kabsch(bi[idx], bj[idx])
        ang = np.arccos(np.clip(np.sum(bi * (bj @ R.T), axis=1), -1, 1))
        inl = ang < thr
        if inl.sum() > best_in.sum():
            best_in = inl
    if best_in.sum() >= 3:
        R = kabsch(bi[best_in], bj[best_in])
        ang = np.arccos(np.clip(np.sum(bi * (bj @ R.T), axis=1), -1, 1))
        best_in = ang < thr
        R = kabsch(bi[best_in], bj[best_in])
        res = np.rad2deg(np.arccos(np.clip(np.sum(bi[best_in] * (bj[best_in] @ R.T), axis=1), -1, 1))).mean()
        return R, best_in, res
    return np.eye(3), best_in, np.inf


def match_pair(imgA, imgB, KA, DA, KB_, DB, ratio=0.75):
    """SIFT-match A,B; return matched bearing rays (bA, bB) and pixel coords for viz."""
    sift = cv2.SIFT_create(nfeatures=8000)
    kA, dA = sift.detectAndCompute(imgA, None)
    kB, dB = sift.detectAndCompute(imgB, None)
    if dA is None or dB is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(dA, dB, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None
    pA = np.array([kA[m.queryIdx].pt for m in good])
    pB = np.array([kB[m.trainIdx].pt for m in good])
    return unproject(pA, KA, DA), unproject(pB, KB_, DB), pA, pB


def render_panorama(imgs, intr, rots, out_w=1920, out_h=540, el_max_deg=50.0,
                    fov_half_deg=65.0, feather_deg=20.0):
    """Replicate the panorama node's equirect stitch in numpy (R = camera->rig, flip180 applied)."""
    el_max = np.deg2rad(el_max_deg); fov = np.deg2rad(fov_half_deg); feath = np.deg2rad(feather_deg)
    xs = (2 * np.pi) * (np.arange(out_w) + 0.5) / out_w - np.pi
    ys = el_max - (2 * el_max) * (np.arange(out_h) + 0.5) / out_h
    az, el = np.meshgrid(xs, ys)
    dr = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1)  # (H,W,3)
    acc = np.zeros((out_h, out_w), np.float32); wsum = np.zeros((out_h, out_w), np.float32)
    for cam, img in imgs.items():
        K, D, w, h = intr[cam]
        R = rots[cam]
        dcam = dr @ R  # d_cam = R^T d_rig  -> (H,W,3) since (dr @ R)_k = sum_j dr_j R_jk = (R^T dr)_k
        X, Y, Z = dcam[..., 0].copy(), dcam[..., 1].copy(), dcam[..., 2]
        X = -X; Y = -Y  # flip_180 (upside-down mount)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = X / Z; b = Y / Z
            r = np.sqrt(a * a + b * b); th = np.arctan(r)
            th2 = th * th
            thd = th * (1 + D[0] * th2 + D[1] * th2**2 + D[2] * th2**3 + D[3] * th2**4)
            s = np.where(r > 1e-9, thd / r, 1.0)
            u = K[0, 0] * a * s + K[0, 2]; v = K[1, 1] * b * s + K[1, 2]
        valid = (Z > 1e-6) & (th < fov) & (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
        wgt = np.clip((fov - th) / feath, 0, 1).astype(np.float32) * valid
        samp = cv2.remap(img, u.astype(np.float32), v.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        acc += samp.astype(np.float32) * wgt; wsum += wgt
    out = np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-6), 0)
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", default="scripts/config/1640x1232")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics.yaml")
    ap.add_argument("--images", nargs="+", default=["scripts/calib/capture"],
                    help="one or more dirs each with camN_image_raw.png (RAW). Multiple distant/"
                         "textured scenes make the rotation over-determined -> reliable calibration.")
    ap.add_argument("--out", default="config/rig/rig_extrinsics_calibrated.yaml")
    ap.add_argument("--ransac-deg", type=float, default=0.5)
    ap.add_argument("--render", default="scripts/calib/capture",
                    help="dir to write before/after panorama PNGs")
    args = ap.parse_args()

    cams = ["cam1", "cam2", "cam3", "cam4"]
    intr = {c: load_intrinsics(args.calib_dir, c) for c in cams}
    sets = []                                   # list of {cam: image} per capture set
    for d in args.images:
        s = {}
        for c in cams:
            im = cv2.imread(os.path.join(d, c + "_image_raw.png"), cv2.IMREAD_GRAYSCALE)
            if im is None:
                sys.exit("missing image: " + os.path.join(d, c + "_image_raw.png"))
            s[c] = im
        sets.append(s)
    imgs = sets[0]                              # first set used for the before/after render
    print(f"loaded {len(sets)} capture set(s)")
    with open(args.rig) as f:
        rig = yaml.safe_load(f)
    R_nom = {c: quat_to_R(rig["cameras"][c]["q_wxyz"]) for c in cams}          # camera->rig
    Q_nom = {c: R_nom[c] @ RZ180 for c in cams}                               # optical->rig

    # 1-3. per-pair RANSAC -> inlier bearing correspondences (the global solve uses these)
    print("=== pairwise relative-rotation estimation ===")
    pair_inliers = {}                          # (ci,cj) -> (bi_in, bj_in)
    MIN_INLIERS = 6
    for ci, cj in PAIRS:
        bis, bjs = [], []
        for s in sets:                         # accumulate correspondences across all capture sets
            m = match_pair(s[ci], s[cj], intr[ci][0], intr[ci][1], intr[cj][0], intr[cj][1])
            if m is not None:
                bis.append(m[0]); bjs.append(m[1])
        if not bis:
            print(f"  {ci}-{cj}: too few matches -> skipped")
            continue
        bi, bj = np.vstack(bis), np.vstack(bjs)
        R, inl, res = ransac_rotation(bi, bj, thresh_deg=args.ransac_deg)
        Rnom_ij = Q_nom[ci].T @ Q_nom[cj]
        delta = np.rad2deg(np.linalg.norm(so3_log(R.T @ Rnom_ij)))
        ok = int(inl.sum()) >= MIN_INLIERS and np.isfinite(res)
        print(f"  {ci}-{cj}: {len(bi):4d} matches, {int(inl.sum()):4d} inliers, "
              f"resid {res:.3f} deg, delta-vs-nominal {delta:.2f} deg{'' if ok else '   [WEAK -> dropped]'}")
        if ok:
            pair_inliers[(ci, cj)] = (bi[inl], bj[inl])
    if not pair_inliers:
        sys.exit("no usable pairs")

    # 4. global robust solve: optimize the 3 non-anchor cams' rotations jointly over ALL inlier
    #    correspondences. Q_c = exp(delta_c) Q_nom_c (anchor delta=0). Residual per match = b_i - R_ij b_j.
    free = [c for c in cams if c != ANCHOR]
    idx = {c: k for k, c in enumerate(free)}
    M = 3 * len(free)

    def make_Q(p):
        Q = {ANCHOR: Q_nom[ANCHOR].copy()}
        for c in free:
            Q[c] = so3_exp(p[3 * idx[c]:3 * idx[c] + 3]) @ Q_nom[c]
        return Q

    def resid(p):
        Q = make_Q(p)
        out = []
        for (ci, cj), (bi_in, bj_in) in pair_inliers.items():
            Rij = Q[ci].T @ Q[cj]
            out.append((bi_in - bj_in @ Rij.T).ravel())
        return np.concatenate(out)

    # robust Gauss-Newton (numerical Jacobian, Huber IRLS) -- scipy 1.11 is ABI-broken vs numpy 2.x
    p = np.zeros(M)
    huber = 0.01  # ~0.57 deg
    r0 = np.rad2deg(np.sqrt(np.mean(resid(p) ** 2)))
    for _ in range(30):
        r = resid(p)
        J = np.zeros((len(r), M))
        eps = 1e-6
        for k in range(M):
            dp = p.copy(); dp[k] += eps
            J[:, k] = (resid(dp) - r) / eps
        a = np.abs(r)
        w = np.where(a < huber, 1.0, huber / np.maximum(a, 1e-12))  # Huber weights
        Jw = J * w[:, None]
        step, *_ = np.linalg.lstsq(Jw.T @ Jw + 1e-9 * np.eye(M), -Jw.T @ (w * r), rcond=None)
        if np.linalg.norm(step) < 1e-7:
            p += step; break
        p += step
    Q = make_Q(p)
    r1 = np.rad2deg(np.sqrt(np.mean(resid(p) ** 2)))
    print(f"=== global solve: RMS angular residual {r0:.3f} -> {r1:.3f} deg over {len(pair_inliers)} pairs ===")

    # 5. write refined extrinsics (R = Q . Rz180); keep translations/imu
    out_rig = {k: v for k, v in rig.items()}
    for c in cams:
        Rcam = Q[c] @ RZ180
        q = R_to_quat(Rcam)
        out_rig["cameras"][c] = dict(rig["cameras"][c])
        out_rig["cameras"][c]["q_wxyz"] = [round(float(x), 8) for x in q]
        d = np.rad2deg(np.linalg.norm(so3_log(R_nom[c].T @ Rcam)))
        print(f"  {c}: rotation refined by {d:.2f} deg")
    with open(args.out, "w") as f:
        f.write("# Calibrated by scripts/calib/extrinsic_calib.py (feature-based relative-rotation).\n")
        f.write("# Rotations refined from scene-feature overlap; translations/imu unchanged.\n")
        yaml.safe_dump(out_rig, f, sort_keys=False, default_flow_style=None)
    print("wrote", args.out)

    # before/after panorama for visual validation
    if args.render:
        before = render_panorama(imgs, intr, R_nom)
        after = render_panorama(imgs, intr, {c: Q[c] @ RZ180 for c in cams})
        bar = np.full((6, before.shape[1]), 255, np.uint8)
        cv2.imwrite(os.path.join(args.render, "pano_before_after.png"),
                    np.vstack([before, bar, after]))
        print("wrote", os.path.join(args.render, "pano_before_after.png"), "(top=before, bottom=after)")


if __name__ == "__main__":
    main()
