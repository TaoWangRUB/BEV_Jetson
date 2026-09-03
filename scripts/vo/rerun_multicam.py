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
from rerun_virtual_pinholes import load_omni, build_map, rot_y   # noqa: E402
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

    # ---- Rerun blueprint: 8 image panes above/below a central 3D view (as in the example).
    def v2d(idx, name):
        return rrb.Spatial2DView(origin=f"rig/cam{idx}", name=name)
    labels = [f"{c} {'+' if s > 0 else '-'}45" for c, s in VCAMS]
    hide3d = [f"- /rig/cam{i}/**" for i in (1, 3, 5, 7)]   # show only the -45 frusta in 3D
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            row_shares=[0.25, 0.5, 0.25],
            contents=[
                rrb.Horizontal(contents=[v2d(i, labels[i]) for i in range(4)]),
                rrb.Spatial3DView(name="3D", origin="/", contents=["+ /**"] + hide3d),
                rrb.Horizontal(contents=[v2d(i, labels[i]) for i in range(4, 8)]),
            ]),
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

    step = max(1, len(P) // a.frames)
    idxs = list(range(0, len(P), step))
    print("logging %d frames from %d poses, %d landmarks" % (len(idxs), len(P), len(lm)))

    if len(lm):
        rr.log("map/landmarks", rr.Points3D(lm @ Rz180.T, colors=lm_col, radii=0.01),
               static=True)

    traj = []
    for i in idxs:
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

    print("done." + (" wrote %s" % out if out else ""))
    if out:
        print("  open with:  rerun %s" % out)


if __name__ == "__main__":
    main()
