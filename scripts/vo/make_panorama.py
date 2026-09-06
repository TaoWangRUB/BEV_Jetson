#!/usr/bin/env python3
"""Render the 360 equirectangular panorama to an IMAGE, from a raw log or a camera bag.

  make_panorama.py <raw_log_dir|camera_bag> --at 30 47 -o /tmp/pano
  make_panorama.py <raw_log_dir> --at 30 --depth inf        # rotation-only, for comparison
  make_panorama.py <raw_log_dir> --every 2 --video /tmp/pano.mp4

The stitch itself is NOT reimplemented here: `pano_maps()` and `render_pano()` come from
rerun_multicam.py, which is where the projection, the nearest-axis seam and the feather
live. This only exists because those were reachable only by building a whole Rerun scene -
a 1.2 GB .rrd to look at one panorama is the wrong trade, and the live path
(ros2/bev_cuvslam/src/bev_panorama_node.cpp) needs a TX2 holding Argus.

Defaults match config/bev_cuvslam/panorama_params.yaml so what you see here is what the
node would publish: 1280 wide (node: 1920), elevation +-50 deg, fisheye half-FOV 80 deg,
feather 20 deg.

DEPTH matters. The four cameras sit on ~0.155 m baselines, so 'inf' (rotation-only) ghosts
everything closer than a few metres - which indoors is everything. The default 4 m finite
sphere cancels parallax at roughly room scale.
"""
import sys, os, argparse, pathlib, numpy as np, cv2, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_multicam import pano_maps, render_pano          # noqa: E402
from rerun_virtual_pinholes import load_omni               # noqa: E402
from render_multicam_video import CAMS                     # noqa: E402
from rerun_odometry import find_bag, read_images           # noqa: E402


def load_raw(d, w, h):
    """Raw image log: camN.raw + camN_index.csv (see scripts/port/check_log_sets.py)."""
    mm, ts = {}, {}
    for c in CAMS:
        mm[c] = np.memmap(d / f"{c}.raw", dtype="u1", mode="r").reshape(-1, h, w)
        rows = [r for r in (d / f"{c}_index.csv").read_text().splitlines()
                if r and not r.startswith("#") and not r.startswith("stamp_ns")]
        ts[c] = np.array([float(r.split(",")[0]) for r in rows]) * 1e-9
    return mm, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="raw image log directory, or a camera rosbag")
    ap.add_argument("--at", type=float, nargs="*", default=[],
                    help="seconds from the first frame; one PNG each")
    ap.add_argument("--every", type=float, default=None,
                    help="render every N seconds instead of --at")
    ap.add_argument("-o", "--out", default="pano", help="output prefix for PNGs")
    ap.add_argument("--video", default=None, help="also write an mp4 (needs --every)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--calib", default="config/calib/imx296_1456x1088")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics_imx296.yaml")
    ap.add_argument("--ground", default="config/rig/ground_plane.yaml")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--elevation", type=float, default=50.0)
    ap.add_argument("--fov-half", type=float, default=80.0)
    ap.add_argument("--feather", type=float, default=20.0)
    ap.add_argument("--seam", type=float, default=8.0)
    ap.add_argument("--depth", default="4.0",
                    help="sphere radius in metres, or 'inf' for rotation-only")
    a = ap.parse_args()

    omni = {c: load_omni(pathlib.Path(a.calib) / f"{c}.yaml") for c in CAMS}
    rig = yaml.safe_load(open(a.rig))["rig_in_cam1"]
    R_rig_cam1 = np.array(yaml.safe_load(open(a.ground))["rig_frame"]["R_rig_cam1"], float)
    depth = None if str(a.depth).lower() in ("inf", "none") else float(a.depth)

    src = pathlib.Path(a.source)
    if (src / "geometry.txt").exists():
        g = dict(l.split() for l in (src / "geometry.txt").read_text().splitlines()
                 if len(l.split()) == 2)
        w, h = int(g["width"]), int(g["height"])
        mm, ts = load_raw(src, w, h)
        t0 = ts[CAMS[0]][0]
        span = ts[CAMS[0]][-1] - t0

        def frames_at(t):                       # nearest frame per camera, own stamps
            return {c: np.asarray(mm[c][int(np.abs(ts[c] - (t0 + t)).argmin())]) for c in CAMS}
    else:
        bag = read_images(find_bag(src), stride=1)
        cts = {c: np.array([s for s, _ in bag[f"/{c}/image_raw"]]) for c in CAMS}
        cim = {c: [im for _, im in bag[f"/{c}/image_raw"]] for c in CAMS}
        h, w = cim[CAMS[0]][0].shape[:2]
        t0 = cts[CAMS[0]][0]
        span = cts[CAMS[0]][-1] - t0

        def frames_at(t):
            return {c: cim[c][int(np.abs(cts[c] - (t0 + t)).argmin())] for c in CAMS}

    print("source %s: %dx%d, %.1f s" % (src, w, h, span))
    tab = pano_maps(omni, rig, R_rig_cam1, out_w=a.width, el_max_deg=a.elevation,
                    fov_half_deg=a.fov_half, feather_deg=a.feather, seam_deg=a.seam,
                    depth=depth, iw=w, ih=h)
    times = (list(np.arange(0, span, a.every)) if a.every else a.at) or [span / 2]

    vw = None
    for t in times:
        pano = render_pano(tab, frames_at(t))
        # A panorama that is 90% one value is a saturated capture, not a stitch problem.
        # Say so here rather than let it be read as a registration failure (5.0g).
        mode = int(np.bincount(pano.ravel()).argmax())
        flat = 100.0 * (pano == mode).mean()
        note = "  <-- SATURATED, not a stitch fault" if flat > 50 else ""
        print("  t=%6.2f s  mean %6.1f  std %5.1f  %5.1f%% at %d%s"
              % (t, pano.mean(), pano.std(), flat, mode, note))
        if a.video:
            if vw is None:
                vw = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                     (pano.shape[1], pano.shape[0]), False)
            vw.write(pano)
        else:
            p = "%s_t%05.1f.png" % (a.out, t)
            cv2.imwrite(p, pano)
            print("    wrote %s" % p)
    if vw is not None:
        vw.release()
        print("  wrote %s" % a.video)


if __name__ == "__main__":
    main()
