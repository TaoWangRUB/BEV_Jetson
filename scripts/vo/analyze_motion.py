#!/usr/bin/env python3
"""Section 5 verdict, host side: does the VO recover true scale, and how far does it drift.

  analyze_motion.py <motion_dir>

Reads /cuvslam/odometry out of the bag and answers 5.1 and 5.2 directly. Deliberately
reports PATH LENGTH alongside straight-line displacement: a trajectory that wanders and
comes back can hit the right endpoint with the wrong scale, and only the ratio of the two
shows it.
"""
import sys, pathlib, numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

d = pathlib.Path(sys.argv[1])
tape = float((d / "tape_metres.txt").read_text().strip())
bag = next(p for p in d.rglob("*") if p.is_dir() and list(p.glob("*.db3")))

ts, xyz = [], []
# Foxy bags carry no embedded type definitions, and rosbags >=0.10 refuses them without a
# typestore ("Bag contains no type definitions"). Name the distro the board actually runs.
with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_FOXY)) as r:
    conns = [c for c in r.connections if c.topic == "/cuvslam/odometry"]
    if not conns: sys.exit("no /cuvslam/odometry in the bag - did tracking ever start?")
    for con, t, raw in r.messages(connections=conns):
        m = r.deserialize(raw, con.msgtype)
        p = m.pose.pose.position
        ts.append(t * 1e-9); xyz.append([p.x, p.y, p.z])
ts = np.array(ts); P = np.array(xyz)
if len(P) < 10: sys.exit("only %d poses - tracking did not hold" % len(P))

span = ts[-1] - ts[0]
steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
path = steps.sum()
# One measurement, two readings: on a straight-line run this should equal the tape; on a
# return-to-origin run the tape is 0 and the same number IS the drift. Reporting it twice
# under two names would imply two independent checks.
disp = np.linalg.norm(P[-1] - P[0])

print("poses %d over %.1f s -> %.2f Hz odometry" % (len(P), span, len(P)/span))
print("  tape measure          %.3f m" % tape)
print("  straight-line displacement %.3f m%s"
      % (disp, "   (%+.1f%% vs tape)" % (100*(disp-tape)/tape) if tape > 0 else "   (tape 0 = return-to-origin run)"))
print("  path length           %.3f m   (wander ratio %.2f)" % (path, path/max(disp,1e-6)))
if tape > 0:
    print("  5.1 TRUE SCALE (within 5%%): %s" % ("PASS" if abs(disp-tape)/tape <= 0.05 else "FAIL"))
else:
    print("  5.2 return-to-origin drift %.3f m over a %.2f m path (%.1f%%)"
          % (disp, path, 100*disp/max(path, 1e-6)))
print("\n  5.4 vs the old bundled rig: it managed ~8.5 Hz; this run %.2f Hz" % (len(P)/span))
print("  NOTE: 5.2 is only meaningful if the rig was physically returned to its start pose.")
