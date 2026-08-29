"""Disparity for the four virtual stereo pairs, on the fov=160 rectification that ships.

Epipolar residual only says rows line up. Disparity says whether the pair resolves depth.
Search range comes from the geometry, not a guess: d = f*B/Z, so f=548, B~0.149 m and a
nearest range of 0.35 m needs ~230 px.
"""
import os, sys, numpy as np, cv2, yaml, rosbag
from cv_bridge import CvBridge

W, H, FOV = 768, 576, 160.0
focal = W/2/np.tan(np.radians(FOV-90)/2)
K = np.array([[focal,0,W/2.],[0,focal,H/2.],[0,0,1]]); D = np.zeros(5)
def rot_y(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])

ext = yaml.safe_load(open(sys.argv[1]))
sgbm = cv2.StereoSGBM_create(
    minDisparity=16, numDisparities=256, blockSize=5,
    P1=8*5*5, P2=32*5*5, disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

tiles, br, rows = [], CvBridge(), []
for name in ("left","front","right","rear"):
    T = np.array(ext[name]["T_to_from"])
    R = rot_y(-np.pi/4).T @ T[:3,:3] @ rot_y(+np.pi/4)
    t = rot_y(-np.pi/4).T @ T[:3,3]
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K,D,K,D,(W,H),R,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    m1 = cv2.initUndistortRectifyMap(K,D,R1,P1,(W,H),cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K,D,R2,P2,(W,H),cv2.CV_32FC1)
    base = abs(P2[0,3]/P2[0,0])
    frames = {}
    for topic, msg, _ in rosbag.Bag(os.environ.get("VS_BAGS","/data/ros1/vclosed_%s.bag") % name).read_messages():
        frames.setdefault(topic, []).append(br.imgmsg_to_cv2(msg, "mono8"))
    A, B = frames["/vcam_a/image_raw"], frames["/vcam_b/image_raw"]
    i = len(A)//2
    ra = cv2.remap(A[i], m1[0], m1[1], cv2.INTER_LINEAR)
    rb = cv2.remap(B[i], m2[0], m2[1], cv2.INTER_LINEAR)
    nonblack = 100.0*(ra > 0).mean()
    disp = sgbm.compute(ra, rb).astype(np.float32)/16.0
    valid = disp > 16
    depth = np.where(valid, focal*base/np.maximum(disp,1e-6), 0)
    inr = valid & (depth > 0.2) & (depth < 4.0)
    rows.append((name, 100*inr.mean(), nonblack,
                 np.percentile(depth[inr],10), np.median(depth[inr]), np.percentile(depth[inr],90)))
    print("%-6s frames %3d  non-black %.0f%%  valid %.0f%%  depth p10 %.2f  med %.2f  p90 %.2f m"
          % (name, len(A), nonblack, 100*inr.mean(),
             np.percentile(depth[inr],10), np.median(depth[inr]), np.percentile(depth[inr],90)))
    dn = np.clip((disp-16)/240.0, 0, 1)
    vis = cv2.applyColorMap((dn*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = (28,28,28)
    tile = cv2.hconcat([cv2.cvtColor(ra, cv2.COLOR_GRAY2BGR), vis])
    cv2.rectangle(tile,(0,0),(tile.shape[1]-1,tile.shape[0]-1),(90,90,90),1)
    cv2.rectangle(tile,(0,0),(tile.shape[1],26),(18,18,18),-1)
    cv2.putText(tile, "%s  %s -> %s  baseline %.3f m  valid %.0f%%" %
                (name.upper(), ext[name]["from"], ext[name]["to"], base, 100*inr.mean()),
                (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235,235,235), 1, cv2.LINE_AA)
    tiles.append(tile)

grid = cv2.vconcat([cv2.hconcat([tiles[0],tiles[1]]), cv2.hconcat([tiles[2],tiles[3]])])
bar = np.zeros((34, grid.shape[1], 3), np.uint8); bar[:] = (18,18,18)
cv2.putText(bar, "fov 160 / 768x576 / f=548 px   left: rectified   right: disparity (turbo, near=red -> far=blue, grey=no match)",
            (12,22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1, cv2.LINE_AA)
cv2.imwrite(os.environ.get("VS_OUT","/data/disparity_4pairs_fov160.png"), cv2.vconcat([bar, grid]))
print("wrote disparity_4pairs_fov160.png")
