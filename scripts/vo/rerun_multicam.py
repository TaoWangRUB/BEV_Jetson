#!/usr/bin/env python3
"""cuVSLAM multicamera Rerun view, mirroring nvidia-isaac/cuVSLAM's multicamera_edex.

  rerun_multicam.py <cloud_or_odom_bag> --images <camera_bag>
        [--calib DIR] [--vstereo YAML] [--rig YAML]
        [--frames N] [--save [out.rrd]] [--spawn] [--serve]

Same blueprint as track_multicamera_r2b.py: 8 image panes (the virtual pinholes cuVSLAM
consumes) around a central 3D view with the trajectory, landmark cloud, moving rig body
and camera frusta. The Points2D on each pane are the real cuVSLAM final landmarks
reprojected into that virtual camera (color keyed by landmark id) - the offline-honest
stand-in for the tracker's per-frame observations, which only exist while tracking runs.
"""
import sys, os, argparse, pathlib, numpy as np, cv2, yaml
import rerun as rr
import rerun.blueprint as rrb
from rosbags.highlevel import AnyReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_virtual_pinholes import load_omni, build_map, rot_y, project_omni   # noqa: E402
from rerun_odometry import read_bag, find_bag, read_images, TS    # noqa: E402
from render_multicam_video import CAMS, VCAMS, quat_to_R, color_from_id  # noqa: E402


def read_observations(bag):
    """{stamp: Nx4 (u, v, vcam_index, id)} - the features cuVSLAM tracked per frame."""
    out = {}
    with AnyReader([bag], default_typestore=TS) as r:
        conns = [c for c in r.connections if c.topic == "/cuvslam/observations"]
        for con, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, con.msgtype)
            a = (np.frombuffer(bytes(m.data), np.uint8)
                 .reshape(m.width, m.point_step)[:, :16].copy()
                 .view(np.float32).reshape(-1, 4))
            out[m.header.stamp.sec + m.header.stamp.nanosec * 1e-9] = a
    return out


def read_slam(bag):
    """The OPTIMISED SLAM trajectory, the loop-closure sites, and the loop-closure edges.

    The trajectory comes from /cuvslam/slam_path (the node's GetAllSlamPoses), NOT from
    accumulating /cuvslam/slam_odometry. A loop closure re-optimises the whole pose graph,
    so a line built by appending the per-frame corrected pose is stale everywhere behind
    the head and steps at every closure — which is exactly how it looked. cuVSLAM's own app
    re-reads the full pose list for the same reason ("overwrite all slam poses in the end
    after LCs and PGOs", tools/cuvslam_app/cuvslam_app.py). The LAST slam_path message is
    the most optimised one, so that is the one to draw.

    All three come back empty for a bag recorded without SLAM=1, and the viewer then just
    shows the pure-VO trajectory instead of failing.
    """
    path, lc, edges = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), \
        np.zeros((0, 2, 3), np.float32)
    path_t, lc_t = np.zeros(0), np.zeros(0)
    with AnyReader([bag], default_typestore=TS) as r:
        def last(topic):
            out = None
            for con, _, raw in r.messages(
                    connections=[c for c in r.connections if c.topic == topic]):
                out = r.deserialize(raw, con.msgtype)
            return out
        m = last("/cuvslam/slam_path")
        if m is not None:
            path = np.array([[q.pose.position.x, q.pose.position.y, q.pose.position.z]
                             for q in m.poses], np.float32).reshape(-1, 3)
            path_t = np.array([q.header.stamp.sec + q.header.stamp.nanosec * 1e-9
                               for q in m.poses])
        m = last("/cuvslam/loop_closures")
        if m is not None:
            lc = np.array([[q.pose.position.x, q.pose.position.y, q.pose.position.z]
                           for q in m.poses], np.float32).reshape(-1, 3)
            lc_t = np.array([q.header.stamp.sec + q.header.stamp.nanosec * 1e-9
                             for q in m.poses])
        m = last("/cuvslam/loop_closure_edges")
        if m is not None:
            e = np.array([[q.position.x, q.position.y, q.position.z]
                          for q in m.poses], np.float32).reshape(-1, 3)
            edges = e[:len(e) // 2 * 2].reshape(-1, 2, 3)   # consecutive pairs = one edge
    return path, path_t, lc, lc_t, edges


def R_to_quat(R):
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
    return np.array([x, y, z, w])


def fisheye_dome(o, img_w, img_h, radius, n_th=48, n_ph=144, th_max=93.0):
    """Spherical cap UV-mapped through the Mei model, so the raw 192 deg image can sit
    on the rig in 3D. rr.Pinhole cannot express a fisheye - it diverges at 180 deg."""
    th = np.radians(np.linspace(1e-3, th_max, n_th))
    ph = np.radians(np.linspace(0.0, 360.0, n_ph, endpoint=False))
    T, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(T) * np.cos(PH), np.sin(T) * np.sin(PH), np.cos(T)], -1)
    d = d.reshape(-1, 3)
    # Forward projection only: pick the directions, look up where they land. No inverse.
    u, v = project_omni(o, d[:, 0], d[:, 1], d[:, 2])
    ok = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    idx = np.arange(n_th * n_ph).reshape(n_th, n_ph)
    nxt = np.roll(np.arange(n_ph), -1)
    v00, v01, v10, v11 = idx[:-1], idx[:-1][:, nxt], idx[1:], idx[1:][:, nxt]
    tris = np.concatenate([np.stack([v00, v10, v11], -1).reshape(-1, 3),
                           np.stack([v00, v11, v01], -1).reshape(-1, 3)])
    tris = tris[ok[tris].all(1)]
    uv = np.stack([u / img_w, v / img_h], -1).astype(np.float32)
    return (d * radius).astype(np.float32), tris.astype(np.uint32), uv


def bev_maps(omni, rig, R_rig_cam1, height, extent, ppm, normal=None, max_incidence=75.0,
             iw=1456, ih=1088):
    """Ground-plane remap tables, one per camera, plus the per-pixel owner.

    `height` is the drop from cam1's optical centre to the plane along `normal` (rig FLU,
    default level). The grid stays ego-centric: rig-forward projected onto the plane.
    Ownership is the camera whose optical axis is closest to the pixel's ray, which puts
    the seams on the bisectors between neighbouring cameras.

    A ground point at planar radius r is seen at incidence arctan(r/h), so its ground
    sample distance blows up as the rig nears the plane. Past `max_incidence` the pixels
    are dropped: that is what stops a low pass smearing a few source pixels across the
    whole square, and it shrinks the BEV to nothing as h -> 0, which is the truth."""
    n = int(2 * extent * ppm)
    nrm = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float)
    nrm = nrm / np.linalg.norm(nrm)
    e1 = np.array([1.0, 0.0, 0.0]) - nrm * nrm[0]  # forward, projected onto the plane
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)                         # left
    fwd = np.linspace(extent, -extent, n)          # image row 0 = forward
    left = np.linspace(extent, -extent, n)         # image col 0 = left
    X, Y = np.meshgrid(fwd, left, indexing="ij")
    P = (-height * nrm + X[..., None] * e1 + Y[..., None] * e2).reshape(-1, 3)
    P_cam1 = P @ R_rig_cam1                        # v_cam1 = R_rig_cam1^T @ v_rig
    r_max = height * np.tan(np.radians(max_incidence))
    resolvable = (np.hypot(X, Y) <= r_max).reshape(-1)
    mx, my, cosang = {}, {}, []
    for c in CAMS:
        R_c, t_c = np.array(rig[c])[:3, :3], np.array(rig[c])[:3, 3]
        v = (P_cam1 - t_c) @ R_c
        rng = np.linalg.norm(v, axis=1)
        u, w = project_omni(omni[c], v[:, 0], v[:, 1], v[:, 2])
        ok = resolvable & (v[:, 2] > 0) & (u >= 0) & (u < iw) & (w >= 0) & (w < ih)
        mx[c] = np.where(ok, u, -1).reshape(n, n).astype(np.float32)
        my[c] = np.where(ok, w, -1).reshape(n, n).astype(np.float32)
        cosang.append(np.where(ok, v[:, 2] / rng, -2.0).reshape(n, n))
    stack = np.stack(cosang)
    owner = np.where(stack.max(0) > -2.0, stack.argmax(0), -1)
    return mx, my, owner


def pano_maps(omni, rig, R_rig_cam1, out_w=1280, el_max_deg=50.0, fov_half_deg=90.0,
              feather_deg=25.0, seam_deg=8.0, depth=None, iw=1456, ih=1088):
    """Equirectangular remap tables and feather weights, one per camera.

    Same convention as the panorama node's spec: az=0 is forward, elevation runs +el_max
    at row 0. With depth=None the sphere is at infinity and the stitch is rotation only,
    which is exact for distant scene and ghosts badly on anything close - the baselines
    are ~0.15 m. A finite `depth` puts the sphere at that radius instead, cancelling
    parallax for scene at roughly that distance.

    Each pixel is taken from the camera whose optical axis is CLOSEST to it, cross-fading
    only over `seam_deg` where two cameras are nearly equidistant. Feathering on absolute
    angle from the axis instead - the obvious thing, and what this used to do - gives every
    camera full weight out to fov_half-feather, which is 65 deg here. Adjacent cameras are
    90 deg apart, so both sit 45 deg off axis at the bisector and both get weight 1.0: the
    output becomes an exact 50/50 average of two cameras over a ~60 deg band, four times
    round, i.e. 65% of the horizon is a literal double exposure. That is the ghost, and no
    choice of `depth` removes it - the sphere only makes the two views agree at ITS radius,
    and everything nearer or further doubles at full strength. Nearest-axis confines the
    disagreement to the seam, and picks the sharper view besides, since a ray far off axis
    lands in the fisheye's compressed periphery."""
    out_h = int(round(out_w * (2 * el_max_deg) / 360.0))       # square pixels
    el_max = np.radians(el_max_deg)
    az = 2 * np.pi * (np.arange(out_w) + 0.5) / out_w - np.pi
    el = el_max - 2 * el_max * (np.arange(out_h) + 0.5) / out_h
    A, E = np.meshgrid(az, el)
    # rig FLU: +x forward, +y left, +z up, so az sweeps forward -> left.
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    d_cam1 = d @ R_rig_cam1
    fov, feath, seam = np.radians(fov_half_deg), np.radians(feather_deg), np.radians(seam_deg)
    mx, my, th, ok = {}, {}, {}, {}
    for c in CAMS:
        R_c, t_c = np.array(rig[c])[:3, :3], np.array(rig[c])[:3, 3]
        v = d_cam1 @ R_c if depth is None else (depth * d_cam1 - t_c) @ R_c
        th[c] = np.arccos(np.clip(v[:, 2] / np.maximum(np.linalg.norm(v, axis=1), 1e-9), -1, 1))
        u, w = project_omni(omni[c], v[:, 0], v[:, 1], v[:, 2])
        ok[c] = (v[:, 2] > 0) & (u >= 0) & (u < iw) & (w >= 0) & (w < ih) & (th[c] < fov)
        mx[c] = np.where(ok[c], u, -1).reshape(out_h, out_w).astype(np.float32)
        my[c] = np.where(ok[c], w, -1).reshape(out_h, out_w).astype(np.float32)
    # Ranked against the best VALID camera, so a pixel only one camera sees keeps weight 1
    # and no normalisation dip appears at the edge of coverage.
    best = np.stack([np.where(ok[c], th[c], np.inf) for c in CAMS]).min(0)
    wt = {}
    for c in CAMS:
        w = np.clip(1.0 - (th[c] - best) / seam, 0, 1) * ok[c]
        w *= np.clip((fov - th[c]) / feath, 0, 1)          # taper at the extreme periphery
        wt[c] = w.reshape(out_h, out_w).astype(np.float32)
    return mx, my, wt


def render_pano(tables, frames):
    mx, my, wt = tables
    h, w = wt[CAMS[0]].shape
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for c in CAMS:
        if frames.get(c) is None:
            continue
        acc += wt[c] * cv2.remap(frames[c], mx[c], my[c], cv2.INTER_LINEAR)
        wsum += wt[c]
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 0).astype(np.uint8)


def plane_near_pose(lm, p, R_wr, R_rig_cam1, radius=5.0, band=0.12, min_nz=0.97):
    """Ground plane from the landmarks around ONE pose, in that pose's own rig frame.
    Returns (height below the camera, normal in rig FLU) or None.

    Everything is relative to the current pose, so this never assumes the map is globally
    consistent - which it is not. A single plane fitted once in the odom frame drifts with
    the map and was reporting height swings of a metre on a rig carried at a fixed height.

    The floor is the lowest strong mode of the height histogram rather than a RANSAC
    winner: walls put points at every height, so the largest plane is not always the
    ground, but the lowest dense one is."""
    V = ((lm - p) @ R_wr) @ R_rig_cam1.T           # landmarks in rig FLU, camera at origin
    z = V[:, 2]
    sel = V[(np.hypot(V[:, 0], V[:, 1]) < radius) & (z > -3.0) & (z < -0.2)]
    if len(sel) < 200:
        return None
    hist, edges = np.histogram(sel[:, 2], bins=56, range=(-3.0, -0.2))
    strong = np.where(hist > 0.25 * hist.max())[0]
    if not len(strong):
        return None
    floor = 0.5 * (edges[strong[0]] + edges[strong[0] + 1])
    pts = sel[np.abs(sel[:, 2] - floor) < band]
    if len(pts) < 50:
        return None
    c = pts.mean(0)
    nv = np.linalg.svd(pts - c)[2][2]
    nv = nv if nv[2] > 0 else -nv
    if nv[2] < min_nz:
        return None
    return float(-(c @ nv)), nv


def scene_radius_near_pose(lm, p, R_wr, R_rig_cam1, lo=0.4, hi=25.0, min_pts=150,
                           el_max_deg=50.0, pct=25.0):
    """Robust distance to the scene around ONE pose, for the panorama sphere. None if thin.

    Rotation-only stitching puts all four cameras at a single point, which is only true for
    scene at infinity. Indoors the walls are 1-3 m away and the ~0.155 m baselines then
    throw the same object to two different bearings in the overlaps - the ghost. Putting
    the sphere at the scene's actual radius cancels it there.

    A LOW PERCENTILE, not the median. Ghost displacement goes as baseline/depth, so the
    cost is wildly asymmetric: too far still ghosts the near scene badly, while slightly
    too near costs the far scene almost nothing. VO landmarks make that worse by sitting
    preferentially on distant textured surfaces and on whatever shows through doorways, so
    the median runs long. Measured on cloud_20260903_123848: p50 gives a 3.8 m median
    against p25's 2.4 m, and the same frame rendered at 4 m still visibly doubles near
    objects that 2 m resolves cleanly.

    Restricted to the elevation band the panorama actually shows, so floor and ceiling
    points outside the canvas cannot pull the radius."""
    V = ((lm - p) @ R_wr) @ R_rig_cam1.T           # landmarks in rig FLU, camera at origin
    r = np.linalg.norm(V, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        el = np.degrees(np.arcsin(np.clip(V[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    sel = r[(r > lo) & (r < hi) & (np.abs(el) < el_max_deg)]
    if len(sel) < min_pts:
        return None
    return float(np.percentile(sel, pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag with /cuvslam/odometry (+ /cuvslam/landmarks)")
    ap.add_argument("--images", required=True, help="source camera bag")
    ap.add_argument("--calib", default="config/calib/imx296_1456x1088")
    ap.add_argument("--vstereo", default="config/rig/virtual_stereo_imx296.yaml")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics_imx296.yaml")
    ap.add_argument("--frames", type=int, default=180, help="max frames to log")
    ap.add_argument("--t-range", default=None, metavar="START:END",
                    help="log only this window, in seconds from the first pose (e.g. 40:52). "
                         "--frames then subsamples the WINDOW, so a short window is logged at "
                         "full rate. Without it --frames subsamples the whole run, which on a "
                         "57 s log is one pose in five - enough to miss a 0.75 s tracking "
                         "freeze entirely (5.0g).")
    ap.add_argument("--map-radius", type=float, default=20.0,
                    help="display-only: drop landmarks further than this (m) from the "
                         "trajectory. Low-parallax features triangulate to hundreds of "
                         "metres and otherwise dictate the 3D view's auto-framing. 0 = keep all")
    ap.add_argument("--fisheye", action="store_true",
                    help="show the 4 raw fisheyes instead of the 8 virtual pinholes: flat "
                         "panes in 2D, and UV-mapped spherical caps on the rig in 3D")
    ap.add_argument("--dome-radius", type=float, default=1.5,
                    help="radius (m) of the fisheye caps in the 3D view")
    ap.add_argument("--tex-scale", type=float, default=0.25,
                    help="downscale factor for the fisheye textures, to bound the .rrd")
    ap.add_argument("--dome-stride", type=int, default=4,
                    help="refresh the 3D dome textures every Nth frame; they are raw RGB "
                         "(Rerun rejects grayscale) and dominate the file size")
    ap.add_argument("--bev-height", type=float, default=None,
                    help="metres from cam1's OPTICAL CENTRE down to the ground, to enable "
                         "the BEV row. config/rig/ground_plane.yaml has status: unmeasured "
                         "and height_m: null, so there is no value to default to - anything "
                         "passed here is PROVISIONAL and assumes the plane is level")
    ap.add_argument("--bev-fit-plane", action="store_true",
                    help="fit the ground plane per frame from the landmarks around the "
                         "current pose, giving its own height and tilt. Drift-immune, and "
                         "the only sane option when the rig's height is not constant or "
                         "the map is not globally consistent. Overrides --bev-height")
    ap.add_argument("--bev-plane-radius", type=float, default=5.0,
                    help="horizontal radius (m) of landmarks used for the per-frame fit")
    ap.add_argument("--bev-extent", type=float, default=4.0, help="BEV half-extent (m)")
    ap.add_argument("--bev-ppm", type=float, default=70.0, help="BEV pixels per metre")
    ap.add_argument("--bev-max-incidence", type=float, default=75.0,
                    help="drop BEV pixels seen at a grazing angle beyond this (deg from "
                         "the plane normal). Caps the usable radius at h*tan(this), which "
                         "is what keeps a low pass from smearing")
    ap.add_argument("--panorama", action="store_true",
                    help="add an equirectangular 360 stitch of the 4 fisheyes, feather "
                         "blended in the overlaps")
    ap.add_argument("--pano-width", type=int, default=1280,
                    help="panorama width in px; height follows from the elevation range")
    ap.add_argument("--pano-elevation", type=float, default=50.0,
                    help="panorama elevation half-range (deg)")
    ap.add_argument("--pano-fov-half", type=float, default=90.0,
                    help="per-camera half-FOV (deg) used for the feather falloff")
    ap.add_argument("--pano-feather", type=float, default=25.0,
                    help="width (deg) of the taper at the outer edge of each camera's field")
    ap.add_argument("--pano-seam", type=float, default=8.0,
                    help="width (deg) of the cross-fade where two cameras are equidistant "
                         "from a ray. Each pixel otherwise comes from the camera looking "
                         "most directly at it. Widening this back toward 90 restores the "
                         "old behaviour, in which most of the horizon was a 50/50 average "
                         "of two cameras and everything off the sphere doubled")
    ap.add_argument("--pano-depth", default="auto",
                    help="panorama sphere radius: 'auto' (default) tracks the landmark "
                         "cloud per frame, 'inf' for the old rotation-only stitch, or a "
                         "fixed radius in metres. Rotation-only puts all four cameras at "
                         "one point, so the ~0.155 m baselines ghost everything closer "
                         "than a few metres - which indoors is everything")
    ap.add_argument("--upright", action=argparse.BooleanOptionalAction, default=True,
                    help="display-only: undo the 180 mount roll so the scene reads upright")
    ap.add_argument("--save", nargs="?", const="", default=None)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()

    vs = yaml.safe_load(open(a.vstereo))["virtual_pinhole"]
    W, H, focal = int(vs["width"]), int(vs["height"]), float(vs["focal_px"])
    cx, cy = W / 2.0, H / 2.0
    omni = {c: load_omni(pathlib.Path(a.calib) / f"{c}.yaml") for c in CAMS}
    rig = yaml.safe_load(open(a.rig))["rig_in_cam1"]
    signs = {-1: np.radians(-45), +1: np.radians(45)}
    maps = {(c, s): build_map(omni[c], signs[s], focal, W, H) for c in CAMS for s in (-1, 1)}
    # cam1_from_vcam (virtual shares the fisheye optical centre): rotation R_cam @ Ry(yaw).
    T_cam1_v = {(c, s): (np.array(rig[c])[:3, :3] @ rot_y(signs[s]), np.array(rig[c])[:3, 3])
                for c, s in VCAMS}

    odom_bag = find_bag(pathlib.Path(a.bag))
    ts, P, Q, child, clouds = read_bag(odom_bag)
    if len(P) < 2:
        sys.exit("need at least 2 poses")
    lm = max((c[1] for c in clouds), key=len) if clouds else np.zeros((0, 3), np.float32)
    lm_col = np.array([color_from_id(i) for i in range(len(lm))], np.uint8)
    if a.map_radius > 0 and len(lm):
        keep = np.linalg.norm(lm - np.asarray(P).mean(0), axis=1) < a.map_radius
        print("landmarks: %d of %d within %.0f m (max %.0f m)"
              % (keep.sum(), len(lm), a.map_radius,
                 np.linalg.norm(lm - np.asarray(P).mean(0), axis=1).max()))
        lm, lm_col = lm[keep], lm_col[keep]
    slam_P, slam_t, slam_lc, slam_lc_t, slam_edges = read_slam(odom_bag)
    if len(slam_P) or len(slam_lc):
        print("SLAM: %d optimised poses, %d loop closures, %d loop edges"
              % (len(slam_P), len(slam_lc), len(slam_edges)))
    obs = read_observations(odom_bag)
    obs_ts = np.array(sorted(obs)) if obs else np.zeros(0)
    print("observations: %d frames, %.0f features/frame"
          % (len(obs), np.mean([len(v) for v in obs.values()]) if obs else 0))

    # The modules are mounted inverted, so cam1's raw optical frame (= cuVSLAM's rig and
    # odom frame) is rolled 180: +y points physically UP. Undo it for DISPLAY only, as a
    # proper rotation of the whole scene - images, 2D features and 3D geometry together -
    # so nothing is mirrored and no calibration is touched.
    Rz180 = np.diag([-1.0, -1.0, 1.0]) if a.upright else np.eye(3)

    src = read_images(find_bag(pathlib.Path(a.images)), stride=1)
    stamps = np.array([s for s, _ in src["/cam1/image_raw"]])
    # Each camera stamps its own exposure midpoint, so match on nearest stamp.
    cam_ts = {c: np.array([s for s, _ in src[f"/{c}/image_raw"]]) for c in CAMS}
    cam_im = {c: [im for _, im in src[f"/{c}/image_raw"]] for c in CAMS}

    def frame_at(c, t, tol=0.03):
        k = int(np.abs(cam_ts[c] - t).argmin())
        return cam_im[c][k] if abs(cam_ts[c][k] - t) <= tol else None

    # ---- Ground plane. Either fixed in the rig frame at a hand-supplied height, or fitted
    # per frame from the landmarks around the current pose.
    bev_on = a.bev_height is not None or a.bev_fit_plane
    bev_n = int(2 * a.bev_extent * a.bev_ppm)
    R_rig_cam1 = bev_static = None
    fit_plane = a.bev_fit_plane
    if bev_on:
        gp = yaml.safe_load(open("config/rig/ground_plane.yaml"))
        R_rig_cam1 = np.array(gp["rig_frame"]["R_rig_cam1"], float)
        if not fit_plane:
            if gp["plane"]["status"] != "measured":
                print("WARNING: ground_plane.yaml status is '%s' and height_m is %s. The "
                      "BEV below uses --bev-height %.3f m and assumes the plane is LEVEL "
                      "and the height CONSTANT. It is PROVISIONAL - scale and flatness "
                      "are not validated. --bev-fit-plane drops both assumptions."
                      % (gp["plane"]["status"], gp["plane"]["height_m"], a.bev_height))
            bev_static = bev_maps(omni, rig, R_rig_cam1, a.bev_height, a.bev_extent,
                                  a.bev_ppm, max_incidence=a.bev_max_incidence)
            print("BEV %dx%d px at %.0f px/m, %.0f%% of the square is covered"
                  % (bev_static[2].shape[0], bev_static[2].shape[1], a.bev_ppm,
                     100.0 * (bev_static[2] >= 0).mean()))

    # ---- Panorama.
    #
    # A fixed radius builds the tables once. `auto` follows the scene, so they are rebuilt
    # when the radius moves and cached on a quantised key - the same shape as the BEV
    # per-frame plane fit below, and for the same reason: one radius chosen once is wrong
    # as soon as the rig changes room.
    pano_mode = str(a.pano_depth).strip().lower()
    if pano_mode in ("inf", "none", "infinity"):
        pano_depth, pano_auto = None, False
    elif pano_mode == "auto":
        pano_depth, pano_auto = None, True
    else:
        pano_depth, pano_auto = float(a.pano_depth), False
    pano_cache, pano_radii = {}, []

    pano = None
    if a.panorama:
        gp = yaml.safe_load(open("config/rig/ground_plane.yaml"))
        R_pano = np.array(gp["rig_frame"]["R_rig_cam1"], float)
        pano = pano_maps(omni, rig, R_pano, out_w=a.pano_width,
                         el_max_deg=a.pano_elevation, fov_half_deg=a.pano_fov_half,
                         feather_deg=a.pano_feather, seam_deg=a.pano_seam, depth=pano_depth)
        print("panorama sphere: %s" % ("auto (per-frame, from the landmark cloud)"
              if pano_auto else "infinity - rotation only, close objects WILL ghost"
              if pano_depth is None else "%.2f m" % pano_depth))
        cov = np.stack([pano[2][c] for c in CAMS]).max(0)
        print("panorama %dx%d, %.0f%% of the sphere band covered, %.0f%% in 2+ cameras"
              % (cov.shape[1], cov.shape[0], 100.0 * (cov > 0).mean(),
                 100.0 * ((np.stack([pano[2][c] for c in CAMS]) > 0).sum(0) > 1).mean()))

    # ---- Rerun blueprint: 8 image panes above/below a central 3D view (as in the example).
    def v2d(idx, name):
        return rrb.Spatial2DView(origin=f"rig/cam{idx}", name=name)
    labels = [f"{c} {'+' if s > 0 else '-'}45" for c, s in VCAMS]
    # Panes are grouped by physical camera, and the two rows run in OPPOSITE carve order so
    # that reading row 1 left-to-right and then row 2 left-to-right walks the ring
    # continuously, instead of jumping back across the rig at the row break:
    #
    #   row 1   cam1 +45  cam1 -45  cam2 +45  cam2 -45
    #   row 2   cam3 -45  cam3 +45  cam4 -45  cam4 +45
    #
    # VCAMS order is cuVSLAM's own and is what each observation's vcam index refers to, so
    # only the DISPLAY order changes here - never the array.
    order = ([VCAMS.index((c, s)) for c in CAMS[:2] for s in (+1, -1)] +
             [VCAMS.index((c, s)) for c in CAMS[2:] for s in (-1, +1)])
    # rig/camN is the virtual pinholes' namespace (N = 0..7); the raw cameras must not
    # share it or cam1..cam4 collide with vpin 1..4.
    hide3d = [f"- /rig/cam{i}/**" for i in range(8)] if a.fisheye else \
             [f"- /rig/cam{i}/**" for i in (1, 3, 5, 7)]
    rows = [
        rrb.Horizontal(contents=[v2d(i, labels[i]) for i in order[:4]]),
        rrb.Spatial3DView(name="3D", origin="/",
                          contents=["+ /**", "- /bev/**", "- /pano/**"] + hide3d),
        rrb.Horizontal(contents=[v2d(i, labels[i]) for i in order[4:]]),
    ]
    extras = []
    if bev_on:
        extras.append(rrb.Spatial2DView(origin="/bev", name="BEV ground plane (%s)"
                      % ("fitted per frame" if fit_plane else "PROVISIONAL height")))
    if pano is not None:
        # Name the pane for the depth mode actually in use. It used to say "rotation only"
        # unconditionally, which is only true for --pano-depth inf and is the opposite of
        # the default: auto puts the sphere on the landmark cloud precisely so close scene
        # does NOT ghost.
        how = ("sphere on the landmark cloud" if pano_auto else
               "rotation only: close objects ghost" if pano_depth is None else
               "sphere at %.1f m" % pano_depth)
        extras.append(rrb.Spatial2DView(origin="/pano", name="equirectangular 360 (%s)" % how))
    # The panorama is 1280x356 - a wide, short pane. 0.18 of the column made it a sliver
    # that reads as "the panorama is missing"; give it a real share.
    ex_share = 0.26
    scale = 1.0 - ex_share * len(extras)
    shares = [0.25 * scale, 0.5 * scale, 0.25 * scale] + [ex_share] * len(extras)
    rows += [rrb.Horizontal(contents=[e]) for e in extras]
    blueprint = rrb.Blueprint(rrb.Vertical(row_shares=shares, contents=rows),
                              rrb.TimePanel(state="collapsed"))

    rr.init("cuvslam_multicam", spawn=a.spawn)
    if a.spawn or a.serve:
        rr.send_blueprint(blueprint)
    if a.serve:
        rr.serve_web()
    out = None
    if a.save is not None or (not a.spawn and not a.serve):
        out = pathlib.Path(a.save) if a.save else odom_bag / "multicam.rrd"
        rr.save(str(out), default_blueprint=blueprint)
        # default_blueprint is only a FALLBACK: the viewer keeps a blueprint per
        # application id, so a .rrd opened in a viewer that has already shown another
        # cuvslam_multicam recording gets the OLD layout - and a pane added since (the
        # panorama, the BEV) simply never appears. Sending it again over the file sink
        # stores it as the ACTIVE blueprint, which wins.
        rr.send_blueprint(blueprint)

    # cuVSLAM convention: right-handed, X-right, Y-down, Z-forward.
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    # Static camera models under the rig; the rig transform below makes them move.
    for i, (c, s) in enumerate(VCAMS):
        R, t = T_cam1_v[(c, s)]
        # The panes are rotated 180 when upright, so roll each camera about its own optical
        # axis by the same amount - otherwise the image plane and the frustum disagree in 3D.
        R = R @ Rz180
        rr.log(f"rig/cam{i}", rr.Transform3D(translation=t, quaternion=R_to_quat(R),
                                             relation=rr.TransformRelation.ParentFromChild),
               rr.Pinhole(image_plane_distance=0.4,
                          image_from_camera=[[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
                          width=W, height=H), static=True)

    dome = {}
    if a.fisheye:
        ih, iw = 1088, 1456
        for c in CAMS:
            # No extra roll here: rotating a pane 180 and rolling its camera 180 cancel in
            # 3D, so the physically-correct pose is the raw one with the raw texture.
            rr.log(f"rig/raw_{c}", rr.Transform3D(translation=np.array(rig[c])[:3, 3],
                                              quaternion=R_to_quat(np.array(rig[c])[:3, :3]),
                                              relation=rr.TransformRelation.ParentFromChild),
                   static=True)
            dome[c] = fisheye_dome(omni[c], iw, ih, a.dome_radius)
        print("fisheye domes: %d triangles each" % len(dome[CAMS[0]][1]))

    sel = np.arange(len(P))
    if a.t_range:
        t_lo, t_hi = (float(x) for x in a.t_range.split(":"))
        rel = np.asarray(ts) - ts[0]
        sel = np.where((rel >= t_lo) & (rel <= t_hi))[0]
        if not len(sel):
            sys.exit("--t-range %s selects no poses (run spans 0..%.1f s)" % (a.t_range, rel[-1]))
        print("t-range %.1f..%.1f s -> %d of %d poses" % (t_lo, t_hi, len(sel), len(P)))
    step = max(1, len(sel) // a.frames)
    idxs = list(sel[::step])
    print("logging %d frames from %d poses, %d landmarks" % (len(idxs), len(P), len(lm)))

    if len(lm):
        rr.log("map/landmarks", rr.Points3D(lm @ Rz180.T, colors=lm_col, radii=0.01),
               static=True)

    # SLAM output is STATIC, not per-frame: it is the final globally optimised state, so it
    # is drawn once rather than grown along the timeline. Growing it was the bug that put
    # steps in the magenta line.
    if len(slam_P):
        rr.log("map/trajectory_slam",
               rr.LineStrips3D([slam_P @ Rz180.T], colors=[0xFF44CCFF], radii=0.012),
               static=True)
    if len(slam_lc):
        # Put each marker on the optimised trajectory AT ITS OWN INSTANT. The pose stored
        # with a closure is the one current when it fired, and later optimisations move the
        # trajectory under it - which is why the markers sat ~0.2 m off the final path and
        # read as belonging to neither line. Matching on the closure's timestamp puts them
        # exactly on it.
        marks = slam_lc
        if len(slam_P) and len(slam_t) and len(slam_lc_t) == len(slam_lc):
            marks = np.array([slam_P[int(np.abs(slam_t - t).argmin())] for t in slam_lc_t],
                             np.float32)
        rr.log("map/loop_closures",
               rr.Points3D(marks @ Rz180.T, colors=[255, 0, 0], radii=0.06,
                           labels=["loop %d" % (i + 1) for i in range(len(marks))]),
               static=True)
    # The edges are the point of the whole display: each one joins a pose to the EARLIER
    # pose it was matched against, so a closure reads as "here is where the rig recognised
    # it had been before" rather than as an unexplained marker.
    if len(slam_edges):
        rr.log("map/loop_edges",
               rr.LineStrips3D([e @ Rz180.T for e in slam_edges],
                               colors=[0xFFDD00FF], radii=0.02), static=True)
    traj = []
    heights, bev_cache = [], {}
    last_plane = None
    for n, i in enumerate(idxs):
        rr.set_time("wall", timestamp=ts[i])
        R_wr, t_wr = quat_to_R(Q[i]), P[i]
        # The rig body, the camera frusta and the images hang off this transform, so it must
        # follow the trajectory the scene is ABOUT. On pure VO the rig flies away with every
        # tracking failure while the optimised line stays put, which reads as the two being
        # unrelated. Rotation still comes from the odometry: slam_path carries positions we
        # trust more, but its orientation is not separately validated here.
        if len(slam_P) and len(slam_t):
            k = int(np.abs(slam_t - ts[i]).argmin())
            if abs(slam_t[k] - ts[i]) < 0.05:
                t_wr = slam_P[k]
        t_d = Rz180 @ t_wr
        R_d = Rz180 @ R_wr
        traj.append(t_d)
        rr.log("rig", rr.Transform3D(translation=t_d, quaternion=R_to_quat(R_d)))
        rr.log("rig/body", rr.Boxes3D(centers=[[0, 0.05, 0]], sizes=[[0.30, 0.16, 0.30]],
                                      colors=[0x2288FFAA]))
        rr.log("map/trajectory", rr.LineStrips3D([traj], colors=[0x33AAFFFF], radii=0.01))
        rr.log("map/head", rr.Points3D([t_d], colors=[0xFF3030FF], radii=0.04))

        key = stamps[int(np.abs(stamps - ts[i]).argmin())]
        ob = obs.get(ts[i])
        if ob is None and len(obs_ts):
            near = obs_ts[int(np.abs(obs_ts - ts[i]).argmin())]
            ob = obs[near] if abs(near - ts[i]) < 1e-3 else None

        if pano is not None:
            if pano_auto:
                # R_pano, not R_rig_cam1: the latter is only populated on the --bev
                # path, and the panorama must work without it.
                rad = scene_radius_near_pose(lm, t_wr, R_wr, R_pano,
                                             el_max_deg=a.pano_elevation)
                if rad is not None:
                    pano_radii.append(rad)
                    # Quantise so a jittering estimate does not rebuild every frame; 10%
                    # steps are far finer than the ghost is sensitive to.
                    qk = int(round(float(np.log(rad) / np.log(1.1))))
                    if qk not in pano_cache:
                        pano_cache[qk] = pano_maps(
                            omni, rig, R_pano, out_w=a.pano_width,
                            el_max_deg=a.pano_elevation, fov_half_deg=a.pano_fov_half,
                            feather_deg=a.pano_feather, seam_deg=a.pano_seam,
                            depth=1.1 ** qk)
                    pano = pano_cache[qk]
                    rr.log("pano/sphere_radius_m", rr.Scalars(rad))
            rr.log("pano/image",
                   rr.Image(render_pano(pano, {c: frame_at(c, key) for c in CAMS}))
                   .compress(jpeg_quality=80))

        if bev_on:
            tables = bev_static
            if fit_plane:
                got = plane_near_pose(lm, t_wr, R_wr, R_rig_cam1,
                                      radius=a.bev_plane_radius)
                if got is None:
                    got = last_plane
                last_plane = got
                if got is None:
                    tables = None
                else:
                    h, n_rig = got
                    heights.append(h)
                    pkey = (round(h, 2), tuple(np.round(n_rig, 3)))
                    if pkey in bev_cache:
                        tables = bev_cache[pkey]
                    else:
                        tables = bev_maps(omni, rig, R_rig_cam1, h, a.bev_extent, a.bev_ppm,
                                          normal=n_rig, max_incidence=a.bev_max_incidence)
                        bev_cache[pkey] = tables
            plan = np.zeros((bev_n, bev_n), np.uint8)
            if tables is not None:
                mx, my, owner = tables
                for ci, c in enumerate(CAMS):
                    fish = frame_at(c, key)
                    if fish is None:
                        continue
                    m = owner == ci
                    if m.any():
                        plan[m] = cv2.remap(fish, mx[c], my[c], cv2.INTER_LINEAR)[m]
            # Logged unconditionally - skipping would leave the previous frame on screen,
            # which reads as a frozen BEV rather than an absent one.
            rr.log("bev/image", rr.Image(plan).compress(jpeg_quality=80))
            if fit_plane and last_plane is not None:
                h, n_rig = last_plane
                rr.log("bev/height", rr.TextLog(
                    "h = %.2f m, tilt %.1f deg, usable radius %.2f m"
                    % (h, np.degrees(np.arccos(min(1.0, abs(n_rig[2])))),
                       max(0.0, h) * np.tan(np.radians(a.bev_max_incidence)))))

        if a.fisheye:
            for c in CAMS:
                fish = frame_at(c, key)
                if fish is None:
                    continue
                tex = cv2.resize(fish, None, fx=a.tex_scale, fy=a.tex_scale,
                                 interpolation=cv2.INTER_AREA)
                vp, tri, uv = dome[c]
                if n % a.dome_stride == 0:
                    rr.log(f"rig/raw_{c}/dome",
                           rr.Mesh3D(vertex_positions=vp, triangle_indices=tri,
                                     vertex_texcoords=uv,
                                     albedo_texture=cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB)))
                pane = cv2.rotate(fish, cv2.ROTATE_180) if a.upright else fish
                rr.log(f"rig/raw_{c}/image", rr.Image(pane).compress(jpeg_quality=70))
                if ob is None:
                    continue
                # Same observations as the carves below, mapped back through the Mei model
                # so the two rows can be compared directly.
                pts, cols = [], []
                for idx, (cc, s) in enumerate(VCAMS):
                    if cc != c:
                        continue
                    p = ob[ob[:, 2].astype(int) == idx]
                    if not len(p):
                        continue
                    r = np.stack([p[:, 0] - cx, p[:, 1] - cy,
                                  np.full(len(p), focal)], -1) @ rot_y(signs[s]).T
                    fu, fv = project_omni(omni[c], r[:, 0], r[:, 1], r[:, 2])
                    good = (fu >= 0) & (fu < 1456) & (fv >= 0) & (fv < 1088)
                    fu, fv = fu[good], fv[good]
                    if a.upright:
                        fu, fv = 1455.0 - fu, 1087.0 - fv
                    pts.append(np.stack([fu, fv], -1))
                    cols.append(np.array([color_from_id(int(q)) for q in p[good, 3]], np.uint8))
                if pts:
                    rr.log(f"rig/raw_{c}/features",
                           rr.Points2D(np.vstack(pts), colors=np.vstack(cols), radii=3.0))

        for idx, (c, s) in enumerate(VCAMS):
            fish = frame_at(c, key)
            if fish is None:
                continue
            v = cv2.remap(fish, *maps[(c, s)], cv2.INTER_LINEAR)
            if a.upright:
                v = cv2.rotate(v, cv2.ROTATE_180)
            rr.log(f"rig/cam{idx}/image", rr.Image(v).compress(jpeg_quality=80))
            if ob is not None:
                p = ob[ob[:, 2].astype(int) == idx]
                uv = p[:, :2]
                if a.upright:            # rotate the features with their pixels
                    uv = np.column_stack([W - 1 - uv[:, 0], H - 1 - uv[:, 1]])
                cols = np.array([color_from_id(int(t)) for t in p[:, 3]],
                                np.uint8).reshape(-1, 3)
                rr.log(f"rig/cam{idx}/observations",
                       rr.Points2D(uv, colors=cols, radii=3))

    if heights:
        print("BEV plane: %d distinct (height, tilt) tables over %d frames, h %.2f .. %.2f m"
              % (len(bev_cache), len(heights), min(heights), max(heights)))
    if pano_radii:
        pr = np.array(pano_radii)
        print("panorama sphere radius: %.2f-%.2f m (median %.2f) over %d frames, "
              "%d distinct remap tables built"
              % (pr.min(), pr.max(), float(np.median(pr)), len(pr), len(pano_cache)))
    print("done." + (" wrote %s" % out if out else ""))
    if out:
        print("  open with:  rerun %s" % out)


if __name__ == "__main__":
    main()
