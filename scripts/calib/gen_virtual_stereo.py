#!/usr/bin/env python3
"""Generate a virtual-pinhole stereo pair from two fisheyes, and write it as a bag.

A 190 deg fisheye cannot be flattened into one pinhole, but it can be carved into two,
each aimed +-45 deg off the optical axis and covering (fov-90) deg. Adjacent cameras
then each contribute the virtual pinhole that faces the other, and those two look the
same way across a real baseline: an ordinary stereo pair.

Which of the 2x2 index combinations is the facing one is NOT assumed - it is chosen by
the smallest angle between the two virtual optical axes, using the measured extrinsic.

The remap is built backwards, per virtual pixel: form its ray, project that ray through
the calibrated omni model, and record where it lands in the raw fisheye. The omni
calibration is consumed here and never appears downstream - what comes out is a plain
pinhole pair for a solver that only speaks pinhole.
"""
import argparse, glob, sqlite3, sys
import numpy as np, cv2, yaml, rosbag, rospy
from cv_bridge import CvBridge

def load_omni(path):
    d = yaml.safe_load(open(path))
    d = d.get("cam0", d)          # kalibr writes cam0:; our config/ is flat
    xi = float(d["intrinsics"][0])
    fx, fy, cx, cy = [float(v) for v in d["intrinsics"][1:5]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.array(d["distortion_coeffs"], np.float64).reshape(1, 4)
    return K, D, np.array([[xi]], np.float64), d["resolution"]

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)

def undist_map(K, D, xi, R, focal, w, h):
    """Backward map: for each virtual pixel, which raw fisheye pixel feeds it."""
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([j - w / 2.0, i - h / 2.0, np.full_like(j, focal, np.float64)], -1)
    rays = rays.reshape(-1, 3) @ R.T
    pts, _ = cv2.omnidir.projectPoints(rays.reshape(1, -1, 3).astype(np.float64),
                                       np.zeros(3), np.zeros(3), K, xi[0, 0], D)
    m = pts.reshape(h, w, 2).astype(np.float32)
    return cv2.convertMaps(m, None, cv2.CV_32FC1)

ap = argparse.ArgumentParser()
ap.add_argument("--bag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--topic-a", required=True); ap.add_argument("--topic-b", required=True)
ap.add_argument("--calib-a", required=True); ap.add_argument("--calib-b", required=True)
ap.add_argument("--extrinsic", required=True, help="yaml with T_to_from for this pair")
ap.add_argument("--pair", required=True)
ap.add_argument("--fov", type=float, default=190.0)
ap.add_argument("--width", type=int, default=480); ap.add_argument("--height", type=int, default=360)
a = ap.parse_args()

Ka, Da, xia, resa = load_omni(a.calib_a)
Kb, Db, xib, resb = load_omni(a.calib_b)
pin_fov = np.deg2rad(a.fov - 90.0)
focal = a.width / 2.0 / np.tan(pin_fov / 2.0)
print("virtual pinhole: %dx%d, fov %.0f deg, focal %.1f px" % (a.width, a.height, a.fov - 90, focal))

# measured extrinsic for this pair: T maps A into B's frame
ext = yaml.safe_load(open(a.extrinsic))[a.pair]
T = np.array(ext["T_to_from"], np.float64)
R_ba = T[:3, :3]

# pick the facing virtual pinholes: smallest angle between their optical axes
best = None
for ia, sa in ((0, -1), (1, +1)):
    for ib, sb in ((0, -1), (1, +1)):
        za = R_ba @ rot_y(sa * np.pi / 4) @ np.array([0, 0, 1.0])   # A's axis in B's frame
        zb = rot_y(sb * np.pi / 4) @ np.array([0, 0, 1.0])
        ang = np.degrees(np.arccos(np.clip(za @ zb, -1, 1)))
        print("   cand A[%d] vs B[%d]: axes %.1f deg apart" % (ia, ib, ang))
        if best is None or ang < best[0]:
            best = (ang, sa, sb, ia, ib)
ang, sa, sb, ia, ib = best
print("-> facing pair: A[%d] / B[%d], optical axes %.1f deg apart" % (ia, ib, ang))

# Write down what was chosen. The epipolar checker used to RE-DERIVE this from the same
# extrinsic and got the composition order wrong (transpose on the wrong side), which put the
# virtual pair 180 deg apart and made a correct bag look like bad data. One source of truth.
_Ra, _Rb = rot_y(sa * np.pi / 4), rot_y(sb * np.pi / 4)
yaml.safe_dump({"pair": a.pair, "width": a.width, "height": a.height,
                "fov_deg": a.fov - 90.0, "focal_px": float(focal),
                "carve": {"a_index": int(ia), "b_index": int(ib),
                          "a_sign": int(sa), "b_sign": int(sb),
                          "axes_apart_deg": float(ang)},
                # virtual-A -> virtual-B, i.e. what cv2.stereoRectify wants as (R, T)
                "R_vb_va": (_Rb.T @ R_ba @ _Ra).tolist(),
                "t_vb": (_Rb.T @ T[:3, 3]).tolist()},
               open(a.out + ".yaml", "w"), sort_keys=False, default_flow_style=None)
print("   wrote pair geometry to %s.yaml" % a.out)

mapax, mapay = undist_map(Ka, Da, xia, rot_y(sa * np.pi / 4), focal, a.width, a.height)
mapbx, mapby = undist_map(Kb, Db, xib, rot_y(sb * np.pi / 4), focal, a.width, a.height)

bridge, out = CvBridge(), rosbag.Bag(a.out, "w")
frames = {a.topic_a: {}, a.topic_b: {}}
for topic, msg, _ in rosbag.Bag(a.bag).read_messages(topics=[a.topic_a, a.topic_b]):
    frames[topic][msg.header.stamp.to_nsec()] = bridge.imgmsg_to_cv2(msg, "mono8")

kb = np.array(sorted(frames[a.topic_b]))
n = 0
for ts in sorted(frames[a.topic_a]):
    if len(kb) == 0: break
    j = int(np.argmin(np.abs(kb - ts)))
    if abs(kb[j] - ts) > 2_000_000:      # 2 ms: the rig is triggered to 1 us
        continue
    va = cv2.remap(frames[a.topic_a][ts], mapax, mapay, cv2.INTER_LINEAR)
    vb = cv2.remap(frames[a.topic_b][kb[j]], mapbx, mapby, cv2.INTER_LINEAR)
    stamp = rospy.Time(ts // 10**9, ts % 10**9)
    for topic, img in (("/vcam_a/image_raw", va), ("/vcam_b/image_raw", vb)):
        m = bridge.cv2_to_imgmsg(img, encoding="mono8")
        m.header.stamp = stamp
        out.write(topic, m, stamp)
    n += 1
out.close()
print("wrote %d virtual stereo frame pairs to %s" % (n, a.out))
