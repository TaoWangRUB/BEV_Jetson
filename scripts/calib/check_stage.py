#!/usr/bin/env python3
"""How usable is a recorded calibration stage? Detections and where they landed.

Frame count says nothing: what matters is how many frames the solver can use, and
whether the target visited the PERIPHERY, which is where fisheye distortion lives and
where a calibration that only saw the centre goes wrong.
"""
import argparse, sys
import numpy as np, cv2
import glob, os, sqlite3
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

ap = argparse.ArgumentParser()
ap.add_argument("bag"); ap.add_argument("topic")
ap.add_argument("--min-tags", type=int, default=4)
a = ap.parse_args()

# rosbag2_py is not in this image (ros-foxy-ros-base). A rosbag2 sqlite3 bag is just a
# db with topics(id,name) and messages(topic_id,timestamp,data), so read it directly.
db = sorted(glob.glob(os.path.join(a.bag, "*.db3")))
if not db:
    sys.exit("no .db3 in %s" % a.bag)
con = sqlite3.connect(db[0])
tid = con.execute("SELECT id FROM topics WHERE name=?", (a.topic,)).fetchone()
if not tid:
    sys.exit("topic %s not in the bag" % a.topic)
params = cv2.aruco.DetectorParameters_create()
params.markerBorderBits = 2
params.adaptiveThreshWinSizeStep = 1
params.adaptiveThreshWinSizeMin = 3
dic = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

GRID = 6
occ = np.zeros((GRID, GRID), dtype=int)
n = hits = total_tags = 0
for (data,) in con.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
                           (tid[0],)):
    msg = deserialize_message(bytes(data), Image)
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
    n += 1
    corners, ids, _ = cv2.aruco.detectMarkers(img, dic, parameters=params)
    if ids is None or len(ids) < a.min_tags:
        continue
    hits += 1
    total_tags += len(ids)
    for c in corners:
        cx, cy = c[0][:, 0].mean(), c[0][:, 1].mean()
        occ[min(GRID - 1, int(cy / msg.height * GRID)),
            min(GRID - 1, int(cx / msg.width * GRID))] += 1

print(f"{n} frames, {hits} usable ({100*hits/max(n,1):.0f}%), "
      f"{total_tags/max(hits,1):.1f} tags per usable frame")
print("\ncoverage (detections per cell, image split 6x6):")
for r in occ:
    print("  " + " ".join(f"{v:5d}" for v in r))
empty = int((occ == 0).sum())
print(f"\n{empty}/{GRID*GRID} cells never saw a tag"
      + ("  <-- move the target there and re-record" if empty > 8 else "  (good coverage)"))
