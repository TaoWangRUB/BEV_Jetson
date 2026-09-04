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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag with /cuvslam/odometry (+ /cuvslam/landmarks)")
    ap.add_argument("--images", required=True, help="source camera bag")
    ap.add_argument("--calib", default="config/calib/imx296_1456x1088")
    ap.add_argument("--vstereo", default="config/rig/virtual_stereo_imx296.yaml")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics_imx296.yaml")
    ap.add_argument("--frames", type=int, default=180, help="max frames to log")
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

    # ---- Rerun blueprint: 8 image panes above/below a central 3D view (as in the example).
    def v2d(idx, name):
        return rrb.Spatial2DView(origin=f"rig/cam{idx}", name=name)
    labels = [f"{c} {'+' if s > 0 else '-'}45" for c, s in VCAMS]
    # Panes are grouped by physical camera, +45 then -45. VCAMS order is cuVSLAM's and is
    # what the observations' vcam index refers to, so only the display order changes here.
    order = [VCAMS.index((c, s)) for c in CAMS for s in (+1, -1)]
    # rig/camN is the virtual pinholes' namespace (N = 0..7); the raw cameras must not
    # share it or cam1..cam4 collide with vpin 1..4.
    hide3d = [f"- /rig/cam{i}/**" for i in range(8)] if a.fisheye else \
             [f"- /rig/cam{i}/**" for i in (1, 3, 5, 7)]
    rows = [
        rrb.Horizontal(contents=[v2d(i, labels[i]) for i in order[:4]]),
        rrb.Spatial3DView(name="3D", origin="/", contents=["+ /**"] + hide3d),
        rrb.Horizontal(contents=[v2d(i, labels[i]) for i in order[4:]]),
    ]
    shares = [0.25, 0.5, 0.25]
    if bev_on:
        rows.append(rrb.Horizontal(contents=[rrb.Spatial2DView(
            origin="bev", name="BEV ground plane (%s)"
                                % ("fitted per frame" if fit_plane
                                   else "PROVISIONAL height"))]))
        shares = [0.21, 0.37, 0.21, 0.21]
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

    step = max(1, len(P) // a.frames)
    idxs = list(range(0, len(P), step))
    print("logging %d frames from %d poses, %d landmarks" % (len(idxs), len(P), len(lm)))

    if len(lm):
        rr.log("map/landmarks", rr.Points3D(lm @ Rz180.T, colors=lm_col, radii=0.01),
               static=True)

    traj = []
    heights, bev_cache = [], {}
    last_plane = None
    for n, i in enumerate(idxs):
        rr.set_time("wall", timestamp=ts[i])
        R_wr, t_wr = quat_to_R(Q[i]), P[i]
        R_d, t_d = Rz180 @ R_wr, Rz180 @ t_wr
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
    print("done." + (" wrote %s" % out if out else ""))
    if out:
        print("  open with:  rerun %s" % out)


if __name__ == "__main__":
    main()
