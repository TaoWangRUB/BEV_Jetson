#!/usr/bin/env python3
"""Predict cuVSLAM's frustum-intersection graph for our 8 virtual pinholes, offline.

cuVSLAM does not take declared stereo pairs. It samples a grid over camera i's image,
back-projects each point to 2 m AND 4 m, projects both into camera j, and connects the
pair when the fraction landing inside j exceeds a threshold (default 0.5). Re-implemented
here from libs/camera/frustum_intersection_graph.cpp:33 so the pairing can be checked
before any board time - including the denominator, which is the nominal 1000 rather than
the 31x31=961 points actually sampled, so the ratio saturates at 0.961.
"""
import numpy as np, yaml, sys

ext = yaml.safe_load(open('config/rig/rig_extrinsics_imx296.yaml'))
vp  = yaml.safe_load(open('config/rig/virtual_stereo_imx296.yaml'))["virtual_pinhole"]
W, H, f = vp["width"], vp["height"], vp["focal_px"]
cx, cy  = vp["principal_point"]
THRESH  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

def rot_y(a):
    c,s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])

# the 8 virtual cameras: each fisheye contributes a -45 deg and a +45 deg pinhole
rig_from_cam, names = [], []
for c in ("cam1","cam2","cam4","cam3"):
    P = np.array(ext["rig_in_cam1"][c], float)
    for tag, sgn in (("L",-1), ("R",+1)):
        T = P.copy(); T[:3,:3] = P[:3,:3] @ rot_y(sgn*np.pi/4)
        rig_from_cam.append(T); names.append("%s_%s" % (c, tag))
N = len(names)

side = int(np.sqrt(1000))                       # 31
step_x, step_y = W/(side+2), H/(side+2)
gi, gj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
u = (gi.ravel()+1)*step_x; v = (gj.ravel()+1)*step_y
xy = np.stack([(u-cx)/f, (v-cy)/f], -1)          # normalised, zero distortion

def ratio(i, j):
    T = np.linalg.inv(rig_from_cam[j]) @ rig_from_cam[i]     # cam_j_from_cam_i
    ok = np.ones(len(xy), bool)
    for d in (2.0, 4.0):                                      # cuVSLAM's d_min, d_max
        p = np.concatenate([xy*d, np.full((len(xy),1), d)], -1)
        q = p @ T[:3,:3].T + T[:3,3]
        good = q[:,2] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = f*q[:,0]/q[:,2] + cx; vv = f*q[:,1]/q[:,2] + cy
        ok &= good & (uu>0) & (uu<W) & (vv>0) & (vv<H)
    return ok.sum()/1000.0

print("cuVSLAM frustum gate: threshold %.2f, sampled at 2 m and 4 m, max possible %.3f\n" % (THRESH, side*side/1000))
deg = {n:0 for n in names}
print("%-18s %8s   %s" % ("pair","ratio","connected"))
for i in range(N):
    for j in range(i+1, N):
        r = max(ratio(i,j), ratio(j,i))
        if r > 0.01:
            conn = r > THRESH
            if conn: deg[names[i]] += 1; deg[names[j]] += 1
            print("%-18s %8.3f   %s" % ("%s - %s" % (names[i], names[j]), r, "YES" if conn else "no"))
print("\ndegree per virtual camera (0 = dropped from primaries by cuVSLAM):")
for n in names: print("   %-8s %d" % (n, deg[n]))
orphans = [n for n,d in deg.items() if d == 0]
print("\n%s" % ("ALL 8 CONNECTED - multicam mode viable" if not orphans
                else "ORPHANS: %s" % ", ".join(orphans)))
