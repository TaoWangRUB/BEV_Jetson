#!/usr/bin/env python3
"""Epipolar residual measured on TAG IDENTITY, not on descriptor matching.

Feature matchers guess correspondences and a repetitive indoor scene defeats them - the
first attempt reported a 4 px median with a 74 px p90, which says more about ORB than
about the rig. An AprilTag corner carries its own identity: tag 17's top-left corner in
one view IS tag 17's top-left corner in the other. Any residual vertical offset after
rectification is then geometry, not mismatching.
"""
import os, sys, numpy as np, cv2, yaml, rosbag
from cv_bridge import CvBridge

bag, ext_yaml, pair = sys.argv[1], sys.argv[2], sys.argv[3]
W   = int(os.environ.get("VS_W", 1280))
H   = int(os.environ.get("VS_H", 960))
FOV = float(os.environ.get("VS_FOV", 190.0))
focal = W / 2.0 / np.tan(np.radians(FOV - 90) / 2.0)
K = np.array([[focal,0,W/2.0],[0,focal,H/2.0],[0,0,1]]); D = np.zeros(5)
def rot_y(a):
    c,s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
# Prefer the geometry gen_virtual_stereo.py wrote alongside the bag. Re-deriving it here is
# how this tool broke twice: the carve combination is CHOSEN from the extrinsic, not fixed,
# and the composition order is easy to get backwards - both failures produced "0 corner
# correspondences", which reads as bad data rather than as a bug in the measurement.
side = bag + ".yaml"
if os.path.exists(side):
    g = yaml.safe_load(open(side))
    R = np.array(g["R_vb_va"]); t = np.array(g["t_vb"])
    print("pair geometry from %s: carve A[%+d45]/B[%+d45], axes %.2f deg apart"
          % (side, g["carve"]["a_sign"], g["carve"]["b_sign"], g["carve"]["axes_apart_deg"]))
else:
    T = np.array(yaml.safe_load(open(ext_yaml))[pair]["T_to_from"])
    R_ba = T[:3, :3]
    best = None
    for sa in (-1, +1):
        for sb in (-1, +1):
            za = R_ba @ rot_y(sa * np.pi / 4) @ np.array([0, 0, 1.0])
            zb = rot_y(sb * np.pi / 4) @ np.array([0, 0, 1.0])
            ang = np.degrees(np.arccos(np.clip(za @ zb, -1, 1)))
            if best is None or ang < best[0]:
                best = (ang, sa, sb)
    ang, sa, sb = best
    Ra, Rb = rot_y(sa * np.pi / 4), rot_y(sb * np.pi / 4)
    R = Rb.T @ R_ba @ Ra          # virtual-A -> virtual-B, matching the generator
    t = Rb.T @ T[:3, 3]
    print("derived: carve A[%+d45]/B[%+d45], axes %.2f deg apart" % (sa, sb, ang))

# WHICH virtual camera is the LEFT one is geometry, not argument order. cv2.stereoRectify
# expects camera 2 to sit to the right of camera 1 (T_x < 0); fed the other way it still
# rectifies, but every disparity comes out negative and the implied depth with it - which is
# how this reported a physically impossible -0.35 m while the MAGNITUDE was right. Order the
# pair by the sign of the baseline and swap the images to match.
swap_ab = t[0] > 0
if swap_ab:
    R, t = R.T, -R.T @ t
    print("pair ordered by geometry: virtual-B is the LEFT camera (images swapped)")

R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K,D,K,D,(W,H),R,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
m1 = cv2.initUndistortRectifyMap(K,D,R1,P1,(W,H),cv2.CV_32FC1)
m2 = cv2.initUndistortRectifyMap(K,D,R2,P2,(W,H),cv2.CV_32FC1)
base = abs(P2[0,3]/P2[0,0])

p = cv2.aruco.DetectorParameters_create(); p.markerBorderBits = 2
p.adaptiveThreshWinSizeMin = 3; p.adaptiveThreshWinSizeStep = 1
p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dic = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

br, frames = CvBridge(), {}
for topic, msg, _ in rosbag.Bag(bag).read_messages():
    frames.setdefault(topic, []).append(br.imgmsg_to_cv2(msg, "mono8"))
A, B = frames["/vcam_a/image_raw"], frames["/vcam_b/image_raw"]
if swap_ab:
    A, B = B, A

dys, dxs, nframes = [], [], 0
for ia, ib in zip(A, B):
    ra = cv2.remap(ia, m1[0], m1[1], cv2.INTER_LINEAR)
    rb = cv2.remap(ib, m2[0], m2[1], cv2.INTER_LINEAR)
    ca, ida, _ = cv2.aruco.detectMarkers(ra, dic, parameters=p)
    cb, idb, _ = cv2.aruco.detectMarkers(rb, dic, parameters=p)
    if ida is None or idb is None: continue
    da = {int(i): c[0] for i, c in zip(ida.ravel(), ca)}
    db = {int(i): c[0] for i, c in zip(idb.ravel(), cb)}
    shared = set(da) & set(db)
    if not shared: continue
    nframes += 1
    for tid in shared:
        for k in range(4):                       # all four corners of each shared tag
            dys.append(db[tid][k][1] - da[tid][k][1])
            dxs.append(da[tid][k][0] - db[tid][k][0])
dys, dxs = np.array(dys), np.array(dxs)
print("%d frame pairs with tags seen in BOTH rectified views, %d corner correspondences"
      % (nframes, len(dys)))
if len(dys):
    print("\nepipolar residual dy (should be 0 after rectification):")
    print("  median |dy| %.2f px | p90 %.2f px | rms %.2f px | signed median %+.2f px"
          % (np.median(np.abs(dys)), np.percentile(np.abs(dys),90),
             np.sqrt((dys**2).mean()), np.median(dys)))
    print("\ndisparity dx: median %.1f px, %.0f%% positive" % (np.median(dxs), 100*(dxs>0).mean()))
    print("implied depth at median disparity: %.2f m  (baseline %.4f m, focal %.0f px)"
          % (focal*base/np.median(dxs), base, focal))
