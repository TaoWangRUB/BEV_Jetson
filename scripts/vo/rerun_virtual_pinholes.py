#!/usr/bin/env python3
"""Visualise the 8 virtual pinholes cuVSLAM actually consumes, in Rerun.

  rerun_virtual_pinholes.py <camera_bag> [--calib DIR] [--vstereo YAML] [--rig YAML]
                            [--stride N] [--save [out.rrd]] [--spawn] [--serve]

Each fisheye is carved into two pinholes at yaw +-45 deg, exactly as
ros2/bev_cuvslam/include/bev_cuvslam/virtual_pinhole.hpp does it - the Mei/omni-radtan
projection is ported here (cv2.omnidir is opencv-contrib and absent from the host, which
is why the node hand-writes it too). The four facing pairs are chosen the same way the
node's rig build does: smallest angle between the two carves' optical axes.
"""
import sys, argparse, pathlib, numpy as np, cv2, yaml
import rerun as rr
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
CAMS = ["cam1", "cam2", "cam3", "cam4"]
# Neighbour pairs the node forms, from rig_extrinsics_imx296.yaml's frustum check.
PAIR_NAMES = {("cam1", "cam2"): "front", ("cam1", "cam3"): "left",
              ("cam2", "cam4"): "right", ("cam3", "cam4"): "rear"}


def find_bag(d: pathlib.Path) -> pathlib.Path:
    if list(d.glob("*.db3")):
        return d
    return next(p for p in d.rglob("*") if p.is_dir() and list(p.glob("*.db3")))


def load_omni(path: pathlib.Path):
    d = yaml.safe_load(open(path))
    d = d.get("cam0", d)
    xi, fx, fy, cx, cy = [float(v) for v in d["intrinsics"][:5]]
    k1, k2, p1, p2 = [float(v) for v in d["distortion_coeffs"]]
    return dict(xi=xi, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project_omni(o, X, Y, Z):
    n = np.sqrt(X * X + Y * Y + Z * Z)
    den = Z / n + o["xi"]
    xu, yu = (X / n) / den, (Y / n) / den
    r2 = xu * xu + yu * yu
    rad = 1.0 + o["k1"] * r2 + o["k2"] * r2 * r2
    xd = xu * rad + 2.0 * o["p1"] * xu * yu + o["p2"] * (r2 + 2.0 * xu * xu)
    yd = yu * rad + o["p1"] * (r2 + 2.0 * yu * yu) + 2.0 * o["p2"] * xu * yu
    return o["fx"] * xd + o["cx"], o["fy"] * yd + o["cy"]


def build_map(o, yaw, focal, w, h):
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    x, y, z = j - w / 2.0, i - h / 2.0, float(focal)
    c, s = np.cos(yaw), np.sin(yaw)
    X, Y, Z = c * x + s * z, y, -s * x + c * z          # ray in the fisheye frame
    u, v = project_omni(o, X, Y, Z)
    return u.astype(np.float32), v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--calib", default="config/calib/imx296_1456x1088")
    ap.add_argument("--vstereo", default="config/rig/virtual_stereo_imx296.yaml")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics_imx296.yaml")
    ap.add_argument("--stride", type=int, default=5, help="log every Nth frame set")
    ap.add_argument("--save", nargs="?", const="", default=None)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()

    vs = yaml.safe_load(open(a.vstereo))["virtual_pinhole"]
    W, H, focal = int(vs["width"]), int(vs["height"]), float(vs["focal_px"])
    omni = {c: load_omni(pathlib.Path(a.calib) / f"{c}.yaml") for c in CAMS}
    rig = yaml.safe_load(open(a.rig))["rig_in_cam1"]
    Rcam = {c: np.array(rig[c])[:3, :3] for c in CAMS}

    # Per camera, the two carves at yaw -45 / +45.
    signs = {-1: np.radians(-45), +1: np.radians(45)}
    maps = {c: {s: build_map(omni[c], y, focal, W, H) for s, y in signs.items()} for c in CAMS}

    # Facing carve for each neighbour pair: smallest angle between carve optical axes,
    # expressed in cam1's frame. Matches the node's rig-build selection.
    def axis(c, s):
        return Rcam[c] @ rot_y(signs[s]) @ np.array([0, 0, 1.0])
    pairs = []
    for (ca, cb), name in PAIR_NAMES.items():
        best = min(((sa, sb) for sa in (-1, 1) for sb in (-1, 1)),
                   key=lambda ss: np.arccos(np.clip(axis(ca, ss[0]) @ axis(cb, ss[1]), -1, 1)))
        ang = np.degrees(np.arccos(np.clip(axis(ca, best[0]) @ axis(cb, best[1]), -1, 1)))
        pairs.append((name, ca, best[0], cb, best[1], ang))
        print("pair %-5s: %s[%+d] <-> %s[%+d]  axes %.2f deg apart"
              % (name, ca, best[0] * 45, cb, best[1] * 45, ang))

    bag = find_bag(pathlib.Path(a.bag))
    frames = {c: [] for c in CAMS}
    with AnyReader([bag], default_typestore=TS) as r:
        topics = {f"/{c}/image_raw": c for c in CAMS}
        conns = [con for con in r.connections if con.topic in topics]
        counts = {c: 0 for c in CAMS}
        for con, t, raw in r.messages(connections=conns):
            c = topics[con.topic]
            n = counts[c]; counts[c] = n + 1
            if n % a.stride:
                continue
            m = r.deserialize(raw, con.msgtype)
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width)
            frames[c].append((stamp, img))

    rr.init("bev_virtual_pinholes", spawn=a.spawn)
    if a.serve:
        rr.serve_web()
    out = None
    if a.save is not None or (not a.spawn and not a.serve):
        out = pathlib.Path(a.save) if a.save else bag / "virtual_pinholes.rrd"
        rr.save(str(out))

    nset = min(len(frames[c]) for c in CAMS)
    for k in range(nset):
        stamp = frames["cam1"][k][0]
        rr.set_time("wall", timestamp=stamp)
        raw = {c: frames[c][k][1] for c in CAMS}
        for c in CAMS:
            rr.log(f"raw/{c}", rr.Image(raw[c]))
        for name, ca, sa, cb, sb, _ in pairs:
            va = cv2.remap(raw[ca], *maps[ca][sa], cv2.INTER_LINEAR)
            vb = cv2.remap(raw[cb], *maps[cb][sb], cv2.INTER_LINEAR)
            rr.log(f"pairs/{name}/{ca}_{'m' if sa < 0 else 'p'}45", rr.Image(va))
            rr.log(f"pairs/{name}/{cb}_{'m' if sb < 0 else 'p'}45", rr.Image(vb))

    print("logged %d frame sets, 8 virtual pinholes each (%dx%d, focal %.1f px)"
          % (nset, W, H, focal))
    if out:
        print("  wrote %s\n  open with:  rerun %s" % (out, out))


if __name__ == "__main__":
    main()
