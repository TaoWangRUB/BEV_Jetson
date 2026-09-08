#!/usr/bin/env python
"""Keep only the frames a calibration can actually use, and spread them over the image.

Two reasons, one of which is not optional:

  * Kalibr's intrinsics initialiser runs a DLT pose fit per view and needs >=6 points.
    A view holding a single tag has 4, and the whole solve dies with a message about
    the DLT algorithm - which is what killed cam2, cam3 and cam4 here.
  * Hundreds of near-identical views cost hours and add nothing. What a fisheye needs
    is coverage at the PERIPHERY, so frames are chosen for the coverage they add
    against a per-cell quota rather than for being numerous.

Runs inside the tartancalib container (ROS1 python + cv2).
  python filter_bag.py in.bag out.bag /cam1/image_raw [more topics...] --min-tags 3 --max 220
"""
import sys, argparse
import numpy as np, cv2, rosbag
from cv_bridge import CvBridge

ap = argparse.ArgumentParser()
ap.add_argument("inbag"); ap.add_argument("outbag"); ap.add_argument("topics", nargs="+")
ap.add_argument("--min-tags", type=int, default=3)   # >=3 tags = 12 points, safely over the DLT floor
ap.add_argument("--max", type=int, default=220)
ap.add_argument("--quota", type=int, default=8)
ap.add_argument("--grid", type=int, default=8)
ap.add_argument("--report-only", action="store_true",
                help="report coverage and where it is short; write no bag")
a = ap.parse_args()

params = cv2.aruco.DetectorParameters_create()
params.markerBorderBits = 2
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeStep = 1
adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
bridge = CvBridge()
G = a.grid

def score(msg):
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    corners, ids, _ = cv2.aruco.detectMarkers(img, adict, parameters=params)
    if ids is None or len(ids) < a.min_tags:
        return None
    cells = set()
    for c in corners:
        cx, cy = c[0][:, 0].mean(), c[0][:, 1].mean()
        cells.add((min(G-1, int(cy/img.shape[0]*G)), min(G-1, int(cx/img.shape[1]*G))))
    sharp = cv2.Laplacian(img, cv2.CV_64F).var()
    return len(ids), cells, sharp

inbag = rosbag.Bag(a.inbag)
cand = {t: [] for t in a.topics}
other = []
for topic, msg, t in inbag.read_messages():
    if topic in a.topics:
        s = score(msg)
        if s:
            cand[topic].append((t, msg, s))
    else:
        other.append((topic, msg, t))

# For a PAIR, a frame only counts if both cameras saw the target at that instant: the
# extrinsic comes from the two poses of the same board pose, so a one-sided view is
# worthless here even though it would be fine for intrinsics.
if len(a.topics) == 2:
    ta, tb = a.topics
    stamps_b = {m.header.stamp.to_nsec(): i for i, (_, m, _) in enumerate(cand[tb])}
    keep_a, keep_b = [], []
    for tt, m, s in cand[ta]:
        k = m.header.stamp.to_nsec()
        near = min(stamps_b, key=lambda x: abs(x - k)) if stamps_b else None
        if near is not None and abs(near - k) < 2_000_000:
            keep_a.append((tt, m, s)); keep_b.append(cand[tb][stamps_b[near]])
    cand[ta], cand[tb] = keep_a, keep_b
    print("simultaneous usable pairs: %d" % len(keep_a))

# Greedy against the per-cell deficit, on the first topic's coverage.
prim = a.topics[0]
need = np.full((G, G), a.quota, int)
picked_idx, pool = [], list(range(len(cand[prim])))
while pool and len(picked_idx) < a.max:
    def val(i):
        _, _, (_, cells, sharp) = cand[prim][i]
        return (sum(min(1, need[r, c]) for (r, c) in cells), sharp)
    best = max(pool, key=val)
    if val(best)[0] == 0:
        break
    for (r, c) in cand[prim][best][2][1]:
        need[r, c] = max(0, need[r, c] - 1)
    picked_idx.append(best); pool.remove(best)

# ---- WHERE is the coverage missing --------------------------------------------------
# "6 cells still short" is not actionable standing at the rig. Name the regions instead,
# and name them in the orientation the OPERATOR sees: the modules are mounted inverted
# and the capture node publishes raw sensor orientation, so image row 0 is the BOTTOM of
# the preview. Reporting raw-image cells here would send them to the opposite corner.
short = [(r, c) for r in range(G) for c in range(G) if need[r, c] > 0]
print("\ncoverage deficit, raw image orientation (0 = quota of %d met):" % a.quota)
for r in range(G):
    print("   " + " ".join("%2d" % need[r, c] for c in range(G)))
if short:
    def region(r, c):
        rr, cc = G - 1 - r, G - 1 - c          # 180 deg -> what the preview shows
        v = "top" if rr < G / 3.0 else ("bottom" if rr >= 2 * G / 3.0 else "middle")
        h = "left" if cc < G / 3.0 else ("right" if cc >= 2 * G / 3.0 else "centre")
        return "%s-%s" % (v, h)
    tally = {}
    for (r, c) in short:
        k = region(r, c)
        tally[k] = tally.get(k, 0) + 1
    print("\n%d cells short. AS SEEN IN THE PREVIEW, show the board more at:" % len(short))
    for k in sorted(tally, key=lambda x: -tally[x]):
        print("   %-16s %d cell(s)" % (k, tally[k]))
else:
    print("\nevery cell reached quota - coverage is complete")

if a.report_only:
    raise SystemExit(0)

out = rosbag.Bag(a.outbag, "w")
n = 0
for i in sorted(picked_idx):
    for t in a.topics:
        tt, m, _ = cand[t][i]
        out.write(t, m, tt)
        n += 1
for topic, msg, t in other:
    out.write(topic, msg, t)
out.close()
print("wrote %d image messages from %d candidates (%d cells still short)"
      % (n, len(cand[prim]), int((need > 0).sum())))
