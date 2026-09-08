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
import sys, os, argparse, collections, pathlib, numpy as np, cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_virtual_pinholes import load_omni, build_map, rot_y  # noqa: E402
from rerun_odometry import read_bag, find_bag, read_images       # noqa: E402

CAMS = ["cam1", "cam2", "cam3", "cam4"]
# 8 virtual cams as the node orders them: camN at yaw -45 then +45.
VCAMS = [(c, s) for c in CAMS for s in (-1, +1)]
# DISPLAY order, which is NOT VCAMS order. Panes are grouped by physical camera and the two
# rows run in OPPOSITE carve order, so reading row 1 left-to-right and then row 2 walks the
# ring continuously instead of jumping back across the rig at the row break:
#
#   row 1   cam1 +45  cam1 -45  cam2 +45  cam2 -45
#   row 2   cam3 -45  cam3 +45  cam4 -45  cam4 +45
#
# Defined HERE, once, because rerun_multicam imports from this module: it used to compute
# the same expression locally while this renderer used raw VCAMS, so the .rrd and the mp4
# disagreed about which pane was which.
DISPLAY_ORDER = ([VCAMS.index((c, s)) for c in CAMS[:2] for s in (+1, -1)] +
                 [VCAMS.index((c, s)) for c in CAMS[2:] for s in (-1, +1)])
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
    ap.add_argument("--t-range", default=None, metavar="START:END",
                    help="seconds from the first pose. --frames then subsamples the WINDOW, "
                         "so a short window renders at full rate. Same semantics as "
                         "rerun_multicam.py, and the way to grab a single frame for a look.")
    ap.add_argument("--map-radius", type=float, default=20.0,
                    help="display-only: drop landmarks further than this (m) from the "
                         "trajectory. Low-parallax features triangulate to hundreds of "
                         "metres. 0 = keep all")
    ap.add_argument("--gif", action="store_true", help="also write a .gif")
    # Orbital view for the 3D panel. The world is X-right, Y-down, Z-forward with up = -Y,
    # so --azim rotates about the vertical axis: +90 turns the view a quarter turn.
    ap.add_argument("--azim", type=float, default=30.0, help="view azimuth (deg)")
    ap.add_argument("--elev", type=float, default=22.0, help="view elevation (deg)")
    # The rows below mirror scripts/vo/rerun_multicam.py so the mp4 shows the same scene as
    # the .rrd. Rerun cannot export video (only --screenshot-to, a single frame), so the
    # only way to get the scene as an mp4 is to composite it here.
    ap.add_argument("--slam", action="store_true",
                    help="draw loop closures and loop edges from the bag's /cuvslam/slam_path"
                         " and /cuvslam/loop_closure_edges, as the .rrd does")
    ap.add_argument("--vo-bag", default=None, metavar="DIR",
                    help="a second run with SLAM OFF, drawn as the pure-VO reference in "
                         "green beside this run's magenta trajectory - the same pair the "
                         ".rrd shows. Replay both at the same rate.")
    ap.add_argument("--panorama", action="store_true", help="equirectangular 360 row")
    ap.add_argument("--bev-fit-plane", action="store_true",
                    help="BEV row, ground plane fitted per frame from nearby landmarks")
    ap.add_argument("--bev-extent", type=float, default=4.0)
    ap.add_argument("--bev-ppm", type=float, default=70.0)
    ap.add_argument("--bev-max-incidence", type=float, default=75.0)
    ap.add_argument("--bev-plane-radius", type=float, default=5.0)
    ap.add_argument("--bev-cache-size", type=int, default=24)
    ap.add_argument("--pano-width", type=int, default=1280)
    ap.add_argument("--pano-elevation", type=float, default=50.0)
    ap.add_argument("--pano-fov-half", type=float, default=90.0)
    ap.add_argument("--pano-feather", type=float, default=25.0)
    ap.add_argument("--pano-seam", type=float, default=8.0)
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

    # Late imports for the same reason read_observations is late: rerun_multicam imports
    # CAMS/VCAMS from THIS module, so a top-level import would be circular.
    bottom_on = a.panorama or a.bev_fit_plane
    pano_tables = R_ground = None
    bev_cache = collections.OrderedDict()
    slam_edges2d = slam_marks = None
    if bottom_on or a.slam:
        from rerun_multicam import (pano_maps, render_pano, bev_maps, plane_near_pose,
                                    read_slam)
        gp = yaml.safe_load(open("config/rig/ground_plane.yaml"))
        R_ground = np.array(gp["rig_frame"]["R_rig_cam1"], float)
        if a.panorama:
            pano_tables = pano_maps(omni, rig, R_ground, out_w=a.pano_width,
                                    el_max_deg=a.pano_elevation, fov_half_deg=a.pano_fov_half,
                                    feather_deg=a.pano_feather, seam_deg=a.pano_seam)
        if a.slam:
            sp, spt, lc, lct, edges = read_slam(odom_bag)
            print("SLAM overlay: %d optimised poses, %d closures, %d edges"
                  % (len(sp), len(lc), len(edges)))

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
    # Pure-VO reference, projected into the same display frame as this run's trajectory.
    ref2d = ref_t = None
    if a.vo_bag:
        rts, rP, _, _, _ = read_bag(find_bag(pathlib.Path(a.vo_bag)))
        print("pure-VO reference (%s): %d poses" % (a.vo_bag, len(rP)))
        ref_t = np.asarray(rts)
        ref_P = np.asarray(rP)

    sel = list(range(len(P)))
    if a.t_range:
        lo, hi = (float(x) for x in a.t_range.split(":"))
        rel = np.asarray(ts) - ts[0]
        sel = [j for j in sel if lo <= rel[j] <= hi]
        if not sel:
            sys.exit("--t-range %s selects no poses (run spans 0..%.1f s)"
                     % (a.t_range, rel[-1]))
        print("t-range %.1f..%.1f s -> %d of %d poses" % (lo, hi, len(sel), len(P)))
    step = max(1, len(sel) // a.frames)
    idxs = sel[::step]
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
    sx, sy = view_basis(azim=a.azim, elev=a.elev)
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
    # Colour the cloud by landmark id, as the .rrd does. A uniform grey cloud and a coloured
    # one carry the same geometry but only the coloured one shows that neighbouring points
    # are different landmarks rather than one smeared blob.
    lm2d_col = (np.array([color_from_id(i * 7) for i in range(len(lm_subd))], np.uint8)
                if len(lm_subd) else np.zeros((0, 3), np.uint8))
    traj2d = to2d(Pd)
    if a.vo_bag:
        ref2d = to2d(ref_P * roll)
    # The magenta line must be the trajectory the CLOSURES are anchored to. They are snapped
    # onto slam_P (the optimised path), so drawing this run's raw /cuvslam/odometry in magenta
    # instead put every marker beside the line rather than on it. The .rrd draws slam_P for
    # exactly this reason - and with --vo-bag supplying the clean VO in green, this run's own
    # SLAM-degraded odometry would be a third line answering a question nobody asked.
    slam2d = slam_t2 = None
    if a.slam and len(sp):
        slam2d = to2d(np.asarray(sp) * roll)
        slam_t2 = np.asarray(spt) if len(spt) == len(sp) else None

    # Bottom row mirrors the .rrd blueprint: BEV on the left third, panorama on the right
    # two thirds. The BEV is square and the panorama is 3.6:1, so each is fitted into its
    # own cell rather than stretched.
    BOTTOM_H = 320 if bottom_on else 0
    bev_w = canvas_w // 3
    pano_w = canvas_w - bev_w
    canvas_h = PANE_H * 2 + MID_H + BOTTOM_H

    # Loop closures/edges are drawn in the SAME orthographic projection as the trajectory,
    # so they land on it rather than beside it.
    # CAUSAL. Everything below is drawn only once it has happened. Logging the whole set
    # from frame 0 - which is what static=True does in the .rrd, and what this did first -
    # shows closures at t=60 s while the rig is 3 s into the run, and the dense chain of
    # markers reads as the trajectory itself while the real one is still a short stub.
    slam_edge_t = np.zeros(0)
    if a.slam and len(edges):
        slam_edges2d = [to2d(e * roll) for e in edges]
        if len(sp) and len(spt):
            # A closure exists from the LATER of its two endpoints: that is when the rig
            # recognised the place, not when it first saw it.
            slam_edge_t = np.array([max(spt[int(np.linalg.norm(sp - e[0], axis=1).argmin())],
                                        spt[int(np.linalg.norm(sp - e[1], axis=1).argmin())])
                                    for e in edges])
        else:
            slam_edge_t = np.full(len(edges), -np.inf)
    slam_mark_t = np.zeros(0)
    if a.slam and len(lc):
        marks = lc
        if len(sp) and len(spt) and len(lct) == len(lc):
            marks = np.array([sp[int(np.abs(spt - t).argmin())] for t in lct], np.float32)
        slam_marks = to2d(marks * roll)
        slam_mark_t = np.asarray(lct) if len(lct) == len(marks) else np.full(len(marks), -np.inf)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                             (canvas_w, canvas_h))
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
        for (x, y), col in zip(lm2d, lm2d_col):
            if 0 <= x < canvas_w and 0 <= y < MID_H:
                mid[y, x] = (int(col[2]), int(col[1]), int(col[0]))
        # Green = pure VO reference, magenta = this run - the same pairing as the .rrd, drawn
        # causally so the two are seen to diverge rather than presented as a finished result.
        # The SLAM-tracked path is the SUBJECT and carries the closure markers, so it is the
        # wide line; pure VO is the thin reference drawn under it. (The reverse was tried and
        # read backwards: the eye follows the widest stroke, which put the attention on the
        # reference rather than on the trajectory the markers annotate.) They nearly coincide
        # - 35.87 m of pure VO against 36.00 m optimised - so the thin green shows through
        # wherever they agree and separates wherever loop closure actually acted.
        if ref2d is not None:
            nref = int(np.searchsorted(ref_t, ts[i]))
            if nref > 1:
                cv2.polylines(mid, [ref2d[:nref]], False, (90, 230, 90), 2, cv2.LINE_AA)
        if slam2d is not None and slam_t2 is not None:
            nsl = int(np.searchsorted(slam_t2, ts[i]))
            if nsl > 1:
                cv2.polylines(mid, [slam2d[:nsl]], False, (200, 80, 255), 5, cv2.LINE_AA)
        elif ref2d is None:
            cv2.polylines(mid, [traj2d[: i + 1]], False, (200, 80, 255), 5, cv2.LINE_AA)
        # Loop edges under the trajectory, 1 px, matching the 0.002 radius in the .rrd.
        now = ts[i]
        if slam_edges2d is not None:
            for e, et in zip(slam_edges2d, slam_edge_t):
                if et <= now:
                    cv2.line(mid, tuple(e[0]), tuple(e[1]), (0, 221, 255), 1, cv2.LINE_AA)
        # Red, matching the .rrd. Filled markers merged into a solid line when they were the
        # only thing on a grey panel; against two coloured trajectories and a coloured cloud
        # they read as annotations again, so they follow the .rrd rather than diverging.
        if slam_marks is not None:
            for m, mt in zip(slam_marks, slam_mark_t):
                if mt <= now:
                    cv2.circle(mid, tuple(m), 4, (40, 40, 255), -1, cv2.LINE_AA)
        cv2.circle(mid, tuple(traj2d[0]), 6, (0, 220, 0), -1)
        cv2.circle(mid, tuple(traj2d[i]), 7, (0, 0, 255), -1)
        path_m = np.linalg.norm(np.diff(P[: i + 1], axis=0), axis=1).sum() if i else 0.0
        cv2.putText(mid, "cuVSLAM multicam VO   frame %d/%d   path %.1f m" % (i, len(P), path_m),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

        ordered = [panes[j] for j in DISPLAY_ORDER]
        top = np.hstack(ordered[:4])
        bot = np.hstack(ordered[4:])
        rows = [top, mid, bot]
        if bottom_on:
            strip = np.full((BOTTOM_H, canvas_w, 3), 18, np.uint8)
            fish = {c: frame_at(c, ts[i]) for c in CAMS}
            if a.bev_fit_plane and all(f is not None for f in fish.values()):
                got = plane_near_pose(lm, P[i], quat_to_R(Q[i]), R_ground,
                                      radius=a.bev_plane_radius)
                if got is not None:
                    h, n_rig = got
                    pkey = (round(h / 0.05), tuple(np.round(n_rig / 0.02).astype(int)))
                    if pkey in bev_cache:
                        tab = bev_cache[pkey]; bev_cache.move_to_end(pkey)
                    else:
                        tab = bev_maps(omni, rig, R_ground, h, a.bev_extent, a.bev_ppm,
                                       normal=n_rig, max_incidence=a.bev_max_incidence)
                        bev_cache[pkey] = tab
                        while len(bev_cache) > a.bev_cache_size:
                            bev_cache.popitem(last=False)
                    mx, my, owner = tab
                    plan = np.zeros(owner.shape, np.uint8)
                    for ci, c in enumerate(CAMS):
                        msk = owner == ci
                        if msk.any():
                            plan[msk] = cv2.remap(fish[c], mx[c], my[c], cv2.INTER_LINEAR)[msk]
                    side = min(BOTTOM_H, bev_w)
                    tile = cv2.cvtColor(cv2.resize(plan, (side, side)), cv2.COLOR_GRAY2BGR)
                    y0 = (BOTTOM_H - side) // 2; x0 = (bev_w - side) // 2
                    strip[y0:y0 + side, x0:x0 + side] = tile
                    cv2.putText(strip, "BEV h=%.2f m" % h, (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if pano_tables is not None and all(f is not None for f in fish.values()):
                pim = render_pano(pano_tables, fish)
                ph = int(pano_w * pim.shape[0] / pim.shape[1])
                ph = min(ph, BOTTOM_H)
                pw = int(ph * pim.shape[1] / pim.shape[0])
                tile = cv2.cvtColor(cv2.resize(pim, (pw, ph)), cv2.COLOR_GRAY2BGR)
                y0 = (BOTTOM_H - ph) // 2; x0 = bev_w + (pano_w - pw) // 2
                strip[y0:y0 + ph, x0:x0 + pw] = tile
                cv2.putText(strip, "equirectangular 360", (bev_w + 8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            rows.append(strip)
        frame = np.vstack(rows)
        writer.write(frame)
        if a.gif:
            gif_frames.append(cv2.cvtColor(cv2.resize(frame, (canvas_w // 2, canvas_h // 2)),
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
