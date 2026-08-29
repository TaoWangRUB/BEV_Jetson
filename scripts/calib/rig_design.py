#!/usr/bin/env python3
"""Ring geometry for an N-camera surround rig, and the cuVSLAM graph it produces.

THE DESIGN RULE. Carve each fisheye into two virtual pinholes at +-S/2 with fov exactly S
(the camera separation). They then span [-S,0] and [0,S]: they meet on the optical axis,
so there is no blind cone and no self-overlap, and N of them tile exactly 360 deg. Any
other fov either leaves a gap (fov < S) or makes a camera's two pinholes overlap
(fov > S), and the second case is only harmless because the overlap ratio stays under
cuVSLAM's 0.5 gate.

WHY 4 CAMERAS CANNOT DO THIS. At S=90 the rule needs a pinhole reaching 90 deg off-axis;
the lens is D190/H160, so it delivers 80. Five at 72 deg is the first count that fits.

Depth precision scales as f*B, and with f = W/2/tan(S/2) and B = 2R sin(S/2) that is
proportional to R*cos(S/2) - so for a fixed ring radius, MORE cameras is monotonically
better, bounded by CSI ports, trigger channels and compute rather than by geometry.
"""
import sys, numpy as np

W, H_IMG, LENS_H = 768, 576, 160.0

def rot_y(a):
    c,s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def overlap(Ti, Tj, fov):
    """cuVSLAM's own gate, from libs/camera/frustum_intersection_graph.cpp:33."""
    f = W/2/np.tan(np.radians(fov)/2); cx, cy = W/2, H_IMG/2
    side = int(np.sqrt(1000)); sx, sy = W/(side+2), H_IMG/(side+2)
    gi, gj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    xy = np.stack([((gi.ravel()+1)*sx-cx)/f, ((gj.ravel()+1)*sy-cy)/f], -1)
    T = np.linalg.inv(Tj) @ Ti; ok = np.ones(len(xy), bool)
    for d in (2.0, 4.0):
        p = np.concatenate([xy*d, np.full((len(xy),1), d)], -1)
        q = p @ T[:3,:3].T + T[:3,3]
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = f*q[:,0]/q[:,2] + cx; vv = f*q[:,1]/q[:,2] + cy
        ok &= (q[:,2] > 0) & (uu > 0) & (uu < W) & (vv > 0) & (vv < H_IMG)
    return ok.sum()/1000.0

def design(N, R):
    S = 360.0/N; f = W/2/np.tan(np.radians(S)/2); B = 2*R*np.sin(np.radians(S)/2)
    return dict(N=N, S=S, fov=S, focal=f, reach=S, baseline=B, fB=f*B,
                fits=S <= LENS_H/2, vcams=2*N)

if __name__ == "__main__":
    R = float(sys.argv[1])/1000.0 if len(sys.argv) > 1 else 0.1041
    print("ring radius %.1f mm, lens H%.0f (pinhole may reach %.0f deg off-axis)\n"
          % (1000*R, LENS_H, LENS_H/2))
    print("%5s %7s %8s %9s %10s %9s %7s" % ("cams","sep","fov_pin","focal px","baseline","f*B","fits?"))
    for N in (4,5,6,7,8):
        d = design(N, R)
        print("%5d %6.1f %8.1f %9.1f %8.1f mm %9.1f %7s"
              % (d["N"], d["S"], d["fov"], d["focal"], 1000*d["baseline"], d["fB"],
                 "yes" if d["fits"] else "NO"))
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    d = design(N, R); S, FOV = d["S"], d["fov"]
    print("\ncuVSLAM frustum graph for N=%d (%d virtual cameras):" % (N, 2*N))
    rig, names = [], []
    for k in range(N):
        yaw = np.radians(k*S); P = np.eye(4)
        P[:3,:3] = rot_y(yaw); P[:3,3] = R*np.array([np.sin(yaw), 0, np.cos(yaw)])
        for tag, sgn in (("L",-1), ("R",+1)):
            T = P.copy(); T[:3,:3] = P[:3,:3] @ rot_y(sgn*np.radians(S/2))
            rig.append(T); names.append("c%d_%s" % (k+1, tag))
    deg = {n:0 for n in names}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            r = max(overlap(rig[i],rig[j],FOV), overlap(rig[j],rig[i],FOV))
            if r > 0.01:
                print("   %-12s %.3f  %s" % ("%s-%s"%(names[i],names[j]), r, "YES" if r>0.5 else "no"))
                if r > 0.5: deg[names[i]] += 1; deg[names[j]] += 1
    print("   %s" % ("all %d virtual cameras connected - multicam viable" % len(names)
                     if all(v>0 for v in deg.values()) else "ORPHANS: %s"
                     % [n for n,v in deg.items() if v==0]))
