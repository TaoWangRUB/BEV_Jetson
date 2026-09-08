#!/usr/bin/env python3
"""Section 5 verdict, host side: does the VO recover true scale, and how far does it drift.

  analyze_motion.py <motion_dir_or_bag> [--max-speed M_PER_S]

Reads /cuvslam/odometry out of the bag and answers 5.1 and 5.2 directly. Deliberately
reports PATH LENGTH alongside straight-line displacement: a trajectory that wanders and
comes back can hit the right endpoint with the wrong scale, and only the ratio of the two
shows it.

CONTINUITY IS CHECKED FIRST, and a broken trajectory suppresses the verdict rather than
being averaged into it. The 2026-09-06 host replay (5.0g) put a 50 m step inside one 50 ms
interval and still reported an 89 m "path length" and a 45 m "displacement" — both numbers
were arithmetic on a teleport. A step implying more than --max-speed (default 5 m/s, well
above anything a carried or driven rig does) is not motion, so it is counted, named, and
excluded, and 5.1/5.2 refuse to print a PASS while any remain.

`tape_metres.txt` in the directory is OPTIONAL: with it the run is a 5.1 straight-line scale
check, without it only continuity, rate and drift are reported.
"""
import sys, pathlib, numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MAX_SPEED = float(opts.get("--max-speed", 5.0))
# cuVSLAM's first poses are an initialisation transient: on the run1 replays every jump
# inside t < 1 s sat between 0.50 and 0.95 s, and the three replays disagreed there while
# agreeing to 4% everywhere else. Counting it as a tracking failure suppresses the verdict
# on runs that are fine, so it is excluded and REPORTED, never silently dropped.
SKIP_S = float(opts.get("--skip-s", 1.0))
d = pathlib.Path(args[0])
tape_f = d / "tape_metres.txt"
tape = float(tape_f.read_text().strip()) if tape_f.exists() else -1.0
bag = d if list(d.glob("*.db3")) else next(
    p for p in d.rglob("*") if p.is_dir() and list(p.glob("*.db3")))

ts, rx, xyz = [], [], []
# Foxy bags carry no embedded type definitions, and rosbags >=0.10 refuses them without a
# typestore ("Bag contains no type definitions"). Name the distro the board actually runs.
with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_FOXY)) as r:
    conns = [c for c in r.connections if c.topic == "/cuvslam/odometry"]
    if not conns: sys.exit("no /cuvslam/odometry in the bag - did tracking ever start?")
    for con, t, raw in r.messages(connections=conns):
        m = r.deserialize(raw, con.msgtype)
        p = m.pose.pose.position
        # HEADER stamp, not the bag receive time. The receive time is the REPLAY clock:
        # on a 0.5x replay it doubles every interval, so every speed comes out half its
        # true value and the continuity gate is twice as lenient as it reads. Receive
        # time is kept separately, because wall-clock throughput is a different question
        # from odometry rate and only one of them is a property of the VO.
        ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        rx.append(t * 1e-9); xyz.append([p.x, p.y, p.z])
ts = np.array(ts); rx = np.array(rx); P = np.array(xyz)
if len(P) < 10: sys.exit("only %d poses - tracking did not hold" % len(P))

span = ts[-1] - ts[0]
steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
dts = np.diff(ts)
# Continuity BEFORE any summary number, because every summary number is a sum over these.
speed = steps / np.maximum(dts, 1e-6)
rel = ts - ts[0]
init = rel[:-1] < SKIP_S
jump = (speed > MAX_SPEED) & ~init
wall = rx[-1] - rx[0]
print("poses %d over %.1f s of SENSOR time -> %.2f Hz odometry" % (len(P), span, len(P)/span))
print("  processed in %.1f s wall -> %.2f Hz throughput (replay rate ~%.2fx)"
      % (wall, len(P)/wall, span/max(wall, 1e-6)))
print("\n  CONTINUITY (a step over %.1f m/s is a tracking failure, not motion)" % MAX_SPEED)
n_init = ((speed > MAX_SPEED) & init).sum()
if n_init:
    print("  %d jump(s) inside the first %.1f s ignored as the initialisation transient"
          % (n_init, SKIP_S))
if jump.any():
    print("  %d of %d intervals JUMP, carrying %.1f m of the %.1f m raw path"
          % (jump.sum(), len(steps), steps[jump].sum(), steps.sum()))
    for i in np.argsort(steps)[::-1][:5]:
        if not jump[i]: break
        print("    t=%6.2f s  %8.3f m in %5.1f ms  (%7.1f m/s)"
              % (ts[i]-ts[0], steps[i], dts[i]*1e3, speed[i]))
    gap = dts > 3*np.median(dts)
    print("  odometry gaps > 3x median dt: %d (largest %.0f ms) - dropped sets show up here"
          % (gap.sum(), dts.max()*1e3 if len(dts) else 0))
else:
    print("  clean: no interval exceeds %.1f m/s (fastest %.2f m/s)" % (MAX_SPEED, speed.max()))
path = steps[~jump & ~init].sum()
# One measurement, two readings: on a straight-line run this should equal the tape; on a
# return-to-origin run the tape is 0 and the same number IS the drift. Reporting it twice
# under two names would imply two independent checks.
disp = np.linalg.norm(P[-1] - P[0])

print("\n  path length           %.3f m   (jump intervals excluded)" % path)
ok = ~jump & ~init
print("  median speed          %.2f m/s   p95 %.2f m/s"
      % (np.median(speed[ok]), np.percentile(speed[ok], 95)))
if jump.any():
    print("\n  VERDICT SUPPRESSED: the trajectory is discontinuous, so neither the endpoint")
    print("  nor the path length means anything. Explain the jumps, or cut the run to a")
    print("  continuous window, before quoting a scale or a drift figure.")
    sys.exit(2)
if tape < 0:
    print("\n  no tape_metres.txt: continuity and rate only, no 5.1 scale verdict.")
    print("  5.2 end-to-start distance %.3f m over a %.2f m path (%.1f%%) - only a drift"
          % (disp, path, 100*disp/max(path, 1e-6)))
    print("      figure if the rig was physically returned to its start pose.")
    sys.exit(0)
print("  tape measure          %.3f m" % tape)
print("  straight-line displacement %.3f m%s"
      % (disp, "   (%+.1f%% vs tape)" % (100*(disp-tape)/tape) if tape > 0 else "   (tape 0 = return-to-origin run)"))
print("  wander ratio          %.2f" % (path/max(disp, 1e-6)))
if tape > 0:
    print("  5.1 TRUE SCALE (within 5%%): %s" % ("PASS" if abs(disp-tape)/tape <= 0.05 else "FAIL"))
else:
    print("  5.2 return-to-origin drift %.3f m over a %.2f m path (%.1f%%)"
          % (disp, path, 100*disp/max(path, 1e-6)))
print("\n  5.4 vs the old bundled rig: it managed ~8.5 Hz; this run %.2f Hz" % (len(P)/span))
print("  NOTE: 5.2 is only meaningful if the rig was physically returned to its start pose.")
