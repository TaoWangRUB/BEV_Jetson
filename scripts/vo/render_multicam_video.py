#!/usr/bin/env python3
"""Render a cuVSLAM-tutorial-style multicamera VO video (no Rerun needed).

  render_multicam_video.py <cloud_or_odom_bag> --images <camera_bag>
        [--calib DIR] [--vstereo YAML] [--rig YAML] [--out FILE.mp4]
        [--frames N] [--fps F] [--gif]

Layout mirrors nvidia-isaac/cuVSLAM's tutorial_multicamera_edex.gif: the 8 virtual
pinholes cuVSLAM consumes across the top and bottom, a 3D trajectory + landmark map in
the middle. The coloured dots on each pane are the real cuVSLAM final landmarks
reprojected into that virtual camera at the current pose (color keyed by landmark id),
which is the offline-honest stand-in for the tracker's per-frame observations.
"""
import sys, os, argparse, pathlib, numpy as np, cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_virtual_pinholes import load_omni, build_map, rot_y  # noqa: E402
from rerun_odometry import read_bag, find_bag, read_images       # noqa: E402

CAMS = ["cam1", "cam2", "cam3", "cam4"]
# 8 virtual cams as the node orders them: camN at yaw -45 then +45.
VCAMS = [(c, s) for c in CAMS for s in (-1, +1)]
PANE_W, PANE_H = 320, 240
MID_H = 520


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def color_from_id(i):  # matches the NVIDIA example's palette
    return (int((i * 17) % 256), int((i * 31) % 256), int((i * 47) % 256))


def view_basis(azim, elev):
    """Screen x/y unit vectors for an orbital view (world is X-right, Y-down, Z-fwd)."""
    a, e = np.radians(azim), np.radians(elev)
    up = np.array([0.0, -1.0, 0.0])                      # world up = -Y
    fwd = np.array([np.cos(e) * np.sin(a), -np.sin(e), np.cos(e) * np.cos(a)])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    trueup = np.cross(fwd, right)
    return right, -trueup                                # screen +y points down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag with /cuvslam/odometry (+ /cuvslam/landmarks)")
    ap.add_argument("--images", required=True, help="source camera bag")
    ap.add_argument("--calib", default="config/calib/imx296_1456x1088")
    ap.add_argument("--vstereo", default="config/rig/virtual_stereo_imx296.yaml")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics_imx296.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=200, help="max composited frames")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--map-radius", type=float, default=20.0,
                    help="display-only: drop landmarks further than this (m) from the "
                         "trajectory. Low-parallax features triangulate to hundreds of "
                         "metres. 0 = keep all")
    ap.add_argument("--gif", action="store_true", help="also write a .gif")
    ap.add_argument("--upright", action=argparse.BooleanOptionalAction, default=True,
                    help="display-only: undo the 180 mount roll (panes, features and 3D "
                         "together). --no-upright shows exactly what cuVSLAM consumes.")
    a = ap.parse_args()

    vs = yaml.safe_load(open(a.vstereo))["virtual_pinhole"]
    W, H, focal = int(vs["width"]), int(vs["height"]), float(vs["focal_px"])
    cx, cy = W / 2.0, H / 2.0
    omni = {c: load_omni(pathlib.Path(a.calib) / f"{c}.yaml") for c in CAMS}
    rig = yaml.safe_load(open(a.rig))["rig_in_cam1"]
    signs = {-1: np.radians(-45), +1: np.radians(45)}
    maps = {(c, s): build_map(omni[c], signs[s], focal, W, H) for c in CAMS for s in (-1, 1)}
    # cam1_from_vcam pose for each virtual camera (virtual shares the fisheye optical centre).
    T_cam1_v = {}
    for c, s in VCAMS:
        M = np.array(rig[c])
        R = M[:3, :3] @ rot_y(signs[s])
        T_cam1_v[(c, s)] = (R, M[:3, 3])

    odom_bag = find_bag(pathlib.Path(a.bag))
    ts, P, Q, child, clouds = read_bag(odom_bag)
    if len(P) < 2:
        sys.exit("need at least 2 poses")
    lm = max((c[1] for c in clouds), key=len) if clouds else np.zeros((0, 3), np.float32)
    if a.map_radius > 0 and len(lm):
        lm = lm[np.linalg.norm(lm - np.asarray(P).mean(0), axis=1) < a.map_radius]
    # Thin the global map so reprojected dots read as sparse features, not confetti.
    lm_draw = lm[:: max(1, len(lm) // 4000)] if len(lm) else lm
    lm_col = np.array([color_from_id(i * 7) for i in range(len(lm_draw))], np.uint8)
    from rerun_multicam import read_observations
    obs = read_observations(odom_bag)
    if obs:
        print("using %d frames of real cuVSLAM observations" % len(obs))

    # Carve virtual panes for every source frame set, keyed by stamp.
    src = read_images(find_bag(pathlib.Path(a.images)), stride=1)
    stamps = np.array([s for s, _ in src["/cam1/image_raw"]])
    # Each camera stamps its own exposure midpoint, so the four differ by a few ms even
    # though the trigger is shared - match on nearest stamp, not an exact key.
    cam_ts = {c: np.array([s for s, _ in src[f"/{c}/image_raw"]]) for c in CAMS}
    cam_im = {c: [im for _, im in src[f"/{c}/image_raw"]] for c in CAMS}

    def frame_at(c, t, tol=0.03):
        k = int(np.abs(cam_ts[c] - t).argmin())
        return cam_im[c][k] if abs(cam_ts[c][k] - t) <= tol else None

    # Pick the poses to render, and the nearest source frame set for each.
    step = max(1, len(P) // a.frames)
    idxs = list(range(0, len(P), step))
    print("compositing %d frames from %d poses, %d landmarks" % (len(idxs), len(P), len(lm)))

    out = pathlib.Path(a.out) if a.out else odom_bag / "multicam_vo.mp4"
    lm_sub = lm[:: max(1, len(lm) // 6000)] if len(lm) else lm
    canvas_w = 4 * PANE_W

    # Cameras are mounted upside-down (180 about the optical axis), so odom's +Y points
    # physically up. Undo it for display only - a proper rotation, so nothing is mirrored;
    # the panes get the same 180 rotation below.
    roll = np.array([-1.0, -1.0, 1.0]) if a.upright else np.array([1.0, 1.0, 1.0])
    Pd = P * roll
    lm_subd = lm_sub * roll if len(lm_sub) else lm_sub

    # Fit an orthographic view, framed on the trajectory (the map spreads far past it).
    sx, sy = view_basis(azim=-60, elev=22)
    tpx, tpy = Pd @ sx, Pd @ sy
    cx3, cy3 = (tpx.max() + tpx.min()) / 2, (tpy.max() + tpy.min()) / 2
    half = max(tpx.max() - tpx.min(), tpy.max() - tpy.min(), 1.0) / 2 * 1.8
    pad = 30
    scale = (MID_H - 2 * pad) / (2 * half)
    ox = canvas_w / 2 - scale * cx3
    oy = MID_H / 2 - scale * cy3

    def to2d(pts):
        return np.column_stack([pts @ sx * scale + ox, pts @ sy * scale + oy]).astype(np.int32)

    lm2d = to2d(lm_subd) if len(lm_subd) else np.zeros((0, 2), np.int32)
    traj2d = to2d(Pd)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                             (canvas_w, PANE_H * 2 + MID_H))
    gif_frames = []

    for k, i in enumerate(idxs):
        # nearest source frame set to this pose
        j = int(np.abs(stamps - ts[i]).argmin())
        tref = stamps[j]
        R_wr, t_wr = quat_to_R(Q[i]), P[i]

        panes = []
        ob = obs.get(ts[i])
        for vi, (c, s) in enumerate(VCAMS):
            fish = frame_at(c, tref)
            if fish is None:
                panes.append(np.zeros((PANE_H, PANE_W, 3), np.uint8)); continue
            v = cv2.remap(fish, *maps[(c, s)], cv2.INTER_LINEAR)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            if ob is not None:
                p = ob[ob[:, 2].astype(int) == vi]
                for uu, vy, _, oid in p:
                    col = color_from_id(int(oid))
                    cv2.circle(v, (int(uu), int(vy)), 4, (col[2], col[1], col[0]), -1)
            elif len(lm_draw):
                Rc, tc = T_cam1_v[(c, s)]
                R_wv = R_wr @ Rc
                t_wv = R_wr @ tc + t_wr
                Xc = (lm_draw - t_wv) @ R_wv          # world -> vcam (R_wv^T @ (X - t))
                z = Xc[:, 2]
                m = z > 0.05
                u = focal * Xc[m, 0] / z[m] + cx
                vv = focal * Xc[m, 1] / z[m] + cy
                cols = lm_col[m]
                inb = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                for uu, vy, col in zip(u[inb], vv[inb], cols[inb]):
                    cv2.circle(v, (int(uu), int(vy)), 3, (int(col[2]), int(col[1]), int(col[0])), -1)
            if a.upright:
                v = cv2.rotate(v, cv2.ROTATE_180)   # same 180 roll as the 3D view
            v = cv2.resize(v, (PANE_W, PANE_H))
            cv2.putText(v, f"{c} {'+' if s > 0 else '-'}45", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            panes.append(v)

        # 3D panel: static landmark cloud + trajectory so far + current pose.
        mid = np.full((MID_H, canvas_w, 3), 18, np.uint8)
        for x, y in lm2d:
            if 0 <= x < canvas_w and 0 <= y < MID_H:
                mid[y, x] = (170, 170, 170)
        cv2.polylines(mid, [traj2d[: i + 1]], False, (255, 190, 40), 2, cv2.LINE_AA)
        cv2.circle(mid, tuple(traj2d[0]), 6, (0, 220, 0), -1)
        cv2.circle(mid, tuple(traj2d[i]), 7, (0, 0, 255), -1)
        path_m = np.linalg.norm(np.diff(P[: i + 1], axis=0), axis=1).sum() if i else 0.0
        cv2.putText(mid, "cuVSLAM multicam VO   frame %d/%d   path %.1f m" % (i, len(P), path_m),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

        top = np.hstack(panes[:4])
        bot = np.hstack(panes[4:])
        frame = np.vstack([top, mid, bot])
        writer.write(frame)
        if a.gif:
            gif_frames.append(cv2.cvtColor(cv2.resize(frame, (canvas_w // 2, (PANE_H * 2 + MID_H) // 2)),
                                           cv2.COLOR_BGR2RGB))
        if k % 20 == 0:
            print("  frame %d/%d" % (k, len(idxs)))

    writer.release()
    print("wrote %s" % out)
    if a.gif:
        import imageio
        gp = out.with_suffix(".gif")
        imageio.mimsave(str(gp), gif_frames, fps=a.fps)
        print("wrote %s" % gp)


if __name__ == "__main__":
    main()
