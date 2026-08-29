#!/usr/bin/env python3
"""Re-run cuVSLAM's frustum-intersection test on the poses the C++ node actually emits.

A sign error in the rig_from_fisheye * Ry composition still yields a perfectly valid rig -
unit quaternions, sensible translations - and cuVSLAM would accept it and simply fail to
find any stereo pairs. This is the check that turns that into a loud failure: the four
facing pairs must land near 0.94, and a wrong composition drops them to ~0.03.

Field order in the dump is: name focal width qx qy qz qw tx ty tz.
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
