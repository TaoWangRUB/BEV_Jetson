#!/usr/bin/env python3
"""Re-run cuVSLAM's frustum-intersection test on the poses the C++ node actually emits.

A sign error in the rig_from_fisheye * Ry composition still yields a perfectly valid rig -
unit quaternions, sensible translations - and cuVSLAM would accept it and simply fail to
find any stereo pairs. This is the check that turns that into a loud failure: the four
facing pairs must land near 0.94, and a wrong composition drops them to ~0.03.

Field order in the dump is: name focal width qx qy qz qw tx ty tz.

SECOND CHECK, added 2026-09-01 (3R.17). The frustum test above is BLIND to a left/right
swap: two extrinsic solves that put cam2 and cam3 on opposite sides of cam1 both score
~0.94 and both PASS, because each fisheye contributes two carves and one of them always
ends up facing a neighbour. That is not hypothetical - the round-1 and round-2 solves of
this rig differ by exactly that, and nothing in the pipeline caught it: not the ring
closure, not the epipolar residuals, not this frustum graph.

So check the layout too, if an extrinsics file is given:

    check_rig_poses.py <poses.txt> [rig_extrinsics.yaml] [rig_layout.yaml]

The rule comes from the mount. Cameras are inverted (rig_layout camera_roll_deg: 180) and
the VO path consumes RAW frames - the capture node memcpys, it does not rotate - so in
cam1's raw optical frame the 180 deg roll makes +x physically LEFT and +y physically UP.
Walking the ring one step from cam1 therefore lands at x < 0, two steps (the diagonal)
behind at z < 0, and three steps at x > 0. Signs only: this catches a swap, not a
calibration error, and it is deliberately loose about magnitudes.
"""
import sys, numpy as np

rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
names = [r[0] for r in rows]
f, W, H = float(rows[0][1]), int(float(rows[0][2])), 576
cx, cy = W/2.0, H/2.0

def T_of(r):
    x, y, z, w = [float(v) for v in r[3:7]]
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                  [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = [float(v) for v in r[7:10]]
    return T

T = [T_of(r) for r in rows]
side = int(np.sqrt(1000)); sx, sy = W/(side+2), H/(side+2)
gi, gj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
xy = np.stack([((gi.ravel()+1)*sx-cx)/f, ((gj.ravel()+1)*sy-cy)/f], -1)

def ov(i, j):
    M = np.linalg.inv(T[j]) @ T[i]; ok = np.ones(len(xy), bool)
    for d in (2.0, 4.0):                              # cuVSLAM's d_min / d_max
        p = np.concatenate([xy*d, np.full((len(xy),1), d)], -1)
        q = p @ M[:3,:3].T + M[:3,3]
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = f*q[:,0]/q[:,2] + cx; vv = f*q[:,1]/q[:,2] + cy
        ok &= (q[:,2] > 0) & (uu > 0) & (uu < W) & (vv > 0) & (vv < H)
    return ok.sum()/1000.0

deg = {n: 0 for n in names}
print("frustum graph from the poses the node builds:")
for i in range(len(T)):
    for j in range(i+1, len(T)):
        r = max(ov(i,j), ov(j,i))
        if r > 0.01:
            print("   %-18s %.3f   %s" % ("%s - %s" % (names[i], names[j]), r, "YES" if r > 0.5 else "no"))
            if r > 0.5: deg[names[i]] += 1; deg[names[j]] += 1
orphans = [n for n, d in deg.items() if d == 0]
if orphans:
    print("\nFAIL - no stereo partner for: %s" % ", ".join(orphans)); sys.exit(1)
print("\nPASS - all %d virtual cameras have a stereo partner" % len(names))


# ---------------------------------------------------------------- layout sign check
if len(sys.argv) > 2:
    import yaml
    ext = yaml.safe_load(open(sys.argv[2]))["rig_in_cam1"]
    layout = yaml.safe_load(open(sys.argv[3] if len(sys.argv) > 3
                                 else "config/rig/rig_layout.yaml"))
    ring = layout["ring_order"]                       # e.g. [cam1, cam2, cam4, cam3]
    if ring[0] not in ext:
        print("\nlayout check SKIPPED: %s not in extrinsics" % ring[0]); sys.exit(0)
    # step around the ring from the reference camera; expected sign of the dominant axis
    expect = {1: ("x", -1, "one step around the ring: physically right -> x < 0"),
              2: ("z", -1, "diagonal: behind cam1 -> z < 0"),
              3: ("x", +1, "three steps: physically left -> x > 0")}
    axis_i = {"x": 0, "y": 1, "z": 2}
    print("\nlayout sign check (raw inverted cam1 frame: +x = physically LEFT):")
    bad = []
    for step in (1, 2, 3):
        cam = ring[step]
        t = np.array(ext[cam], float)[:3, 3]
        ax, want, why = expect[step]
        got = t[axis_i[ax]]
        ok = (got * want) > 0
        print("   %-5s %-6s %+.4f m   %s   (%s)"
              % (cam, ax + " =", got, "OK" if ok else "WRONG SIGN", why))
        if not ok:
            bad.append("%s: %s = %+.4f, expected %s0" % (cam, ax, got, "<" if want < 0 else ">"))
    if bad:
        print("\nFAIL - the rig is mirrored or two cameras are swapped:")
        for b in bad:
            print("   " + b)
        print("   The frustum graph above cannot see this. Check which physical port fed")
        print("   which topic in the pair recordings, and whether the solve ran on raw or")
        print("   ISP-rotated frames (rig_layout.yaml camera_roll_deg).")
        sys.exit(1)
    print("\nPASS - camera positions match the physical layout in rig_layout.yaml")
