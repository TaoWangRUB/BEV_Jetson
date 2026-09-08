"""Reference map values via cv2.omnidir - the exact path that produced the validated
epipolar residuals - so the C++ port can be diffed against it rather than trusted."""
import numpy as np, cv2, yaml, sys
d = yaml.safe_load(open(sys.argv[1])); d = d.get("cam0", d)
xi = d["intrinsics"][0]; fx,fy,cx,cy = d["intrinsics"][1:5]
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],float); D = np.array([d["distortion_coeffs"]],float)
yaw = np.radians(float(sys.argv[2])); W,H,FOV = 768,576,70.0
focal = W/2/np.tan(np.radians(FOV)/2)
c,s = np.cos(yaw), np.sin(yaw)
R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
for i in (0,143,288,432,575):
    for j in (0,191,384,576,767):
        ray = R @ np.array([j-W/2.0, i-H/2.0, focal])
        p,_ = cv2.omnidir.projectPoints(ray.reshape(1,1,3), np.zeros(3), np.zeros(3), K, xi, D)
        print("%d %d %.6f %.6f" % (i, j, p[0,0,0], p[0,0,1]))
