#!/usr/bin/env python3
"""Prove every frame in a camera bag was processed, or say exactly which were not.

WHY THIS EXISTS. "No warning fired" is not evidence - it only says a counter stayed at
zero, and a counter can be wrong. This closes a LEDGER instead, computed from the bag
rather than from anything the VO node reports about itself:

  1. Upper bound, from the camera bag alone. For each cam1 frame, find the nearest frame
     from every other camera. The matcher picks the nearest and the skew gate rejects
     anything past --gate-us, so a cam1 frame whose partners are all inside the gate MUST
     pair with its own trigger edge - a neighbouring edge is a whole frame period away and
     could never win. That count is the most poses the bag can possibly yield.
  2. Actual, from the VO output bag. Every /cuvslam/odometry stamp is the cam1 stamp of its
     set (see publish()), so the pose stamps should be a SUBSET of the cam1 stamps with no
     duplicates - a duplicate would mean one frame was tracked twice.
  3. The difference, named per frame, and classified by WHERE it sits. Losses at the two
     boundaries are startup and shutdown, not the matcher: a leading run needs the four
     subscriptions connected before any set can form, and a trailing one is the recorder
     being stopped a couple of seconds after playback ends. Only a MID-RUN loss means a
     frame was actually thrown away, and that is the only thing this reports as a failure.

  audit_frames.py <camera_bag> <replay_out_dir> [--gate-us 1000]
"""
import argparse
import glob
import pathlib
import sqlite3
import sys

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

CAMS = ["cam1", "cam2", "cam3", "cam4"]


def bag_stamps(bag):
    """{cam: sorted stamps} straight out of the sqlite, no ROS in the path."""
    db = glob.glob(str(pathlib.Path(bag) / "*.db3"))
    if not db:
        sys.exit("no .db3 under %s" % bag)
    con = sqlite3.connect(db[0])
    tid = {n: i for i, n in con.execute("SELECT id,name FROM topics")}
    out = {}
    for c in CAMS:
        key = "/%s/image_raw" % c
        if key not in tid:
            sys.exit("%s has no %s" % (bag, key))
        out[c] = np.array([r[0] for r in con.execute(
            "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
            (tid[key],))], dtype=np.int64)
    return out


def pose_stamps(out_dir):
    ts = get_typestore(Stores.ROS2_FOXY)
    with AnyReader([pathlib.Path(out_dir)], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic == "/cuvslam/odometry"]
        if not conns:
            sys.exit("no /cuvslam/odometry in %s" % out_dir)
        st = []
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            st.append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)
    return np.array(st, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out_dir")
    ap.add_argument("--gate-us", type=int, default=1000,
                    help="must match the node's max_skew_us (default 1000)")
    a = ap.parse_args()
    gate = a.gate_us * 1000

    cam = bag_stamps(a.bag)
    c1 = cam["cam1"]
    print("camera bag %s" % a.bag)
    for c in CAMS:
        print("   %s: %d frames" % (c, len(cam[c])))

    feasible = np.ones(len(c1), dtype=bool)
    print("\nnearest partner per cam1 frame (the matcher takes the nearest):")
    for c in CAMS[1:]:
        o = cam[c]
        j = np.clip(np.searchsorted(o, c1), 1, len(o) - 1)
        d = np.minimum(np.abs(c1 - o[j - 1]), np.abs(c1 - o[j]))
        feasible &= d <= gate
        print("   %s: median %7.2f us  max %7.2f ms  inside the %d us gate: %d/%d"
              % (c, np.median(d) / 1e3, d.max() / 1e6, a.gate_us, int((d <= gate).sum()), len(d)))
    bound = int(feasible.sum())
    print("\nUPPER BOUND (sets this bag can form): %d of %d cam1 frames" % (bound, len(c1)))

    st = pose_stamps(a.out_dir)
    s_c1, s_od = set(c1.tolist()), set(st.tolist())
    dupes = len(st) - len(s_od)
    print("\nVO output %s" % a.out_dir)
    print("   poses: %d   unique: %d   duplicates: %d" % (len(st), len(s_od), dupes))
    print("   every pose stamp is a real cam1 stamp: %s" % (s_od <= s_c1))

    missing = sorted(s_c1 - s_od)
    idx = sorted(int(np.where(c1 == m)[0][0]) for m in missing)
    # A leading run is startup, a trailing run is shutdown. Peel both off; whatever is left
    # sits in the middle of a running pipeline and is the only real loss.
    lead, last = [], len(c1) - 1
    for k, i in enumerate(idx):
        if i != k:
            break
        lead.append(i)
    trail = []
    for k, i in enumerate(reversed(idx)):
        if i != last - k:
            break
        trail.append(i)
    midrun = [i for i in idx if i not in set(lead) | set(trail)]
    print("\nLEDGER")
    print("   bound %d - poses %d = %d unaccounted" % (bound, len(st), bound - len(st)))
    print("   cam1 frames with no pose: %d" % len(idx))
    print("   ...startup  (leading run from index 0): %d %s" % (len(lead), lead[:10]))
    print("   ...shutdown (trailing run to index %d): %d %s" % (last, len(trail), trail[:10]))
    print("   ...MID-RUN  (frames actually thrown away): %d %s" % (len(midrun), midrun[:20]))
    ok = (dupes == 0) and (s_od <= s_c1) and not midrun
    print("\n%s" % ("PASS - every frame between startup and shutdown was tracked exactly once."
                    if ok else "FAIL - %d frames were lost mid-run." % len(midrun)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
