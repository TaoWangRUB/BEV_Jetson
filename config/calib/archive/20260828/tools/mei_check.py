"""Verify a hand-written Mei (omni-radtan) projection against cv2.omnidir.

The node cannot call cv2.omnidir: it lives in opencv_contrib, which is absent from the
host OpenCV and cannot be assumed on the TX2 either. So the projection gets reimplemented
in C++ - and the reimplementation has to be checked against the reference that produced
the calibration, not merely against the formula in a paper.
"""
import numpy as np, cv2, yaml

d = yaml.safe_load(open('/repo/config/calib/imx296_1456x1088/cam1.yaml'))
xi, fx, fy, cx, cy = d["intrinsics"]
k1, k2, p1, p2 = d["distortion_coeffs"]
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
D = np.array([[k1,k2,p1,p2]], float)

rng = np.random.default_rng(0)
rays = rng.normal(size=(4000,3)); rays[:,2] = np.abs(rays[:,2])*0.3 + 0.05   # forward-ish
rays /= np.linalg.norm(rays, axis=1, keepdims=True)

ref, _ = cv2.omnidir.projectPoints(rays.reshape(1,-1,3), np.zeros(3), np.zeros(3), K, xi, D)
ref = ref.reshape(-1,2)

def mei(X):
    Xs = X/np.linalg.norm(X, axis=1, keepdims=True)
    den = Xs[:,2] + xi
    xu, yu = Xs[:,0]/den, Xs[:,1]/den
    r2 = xu*xu + yu*yu
    rad = 1 + k1*r2 + k2*r2*r2
    xd = xu*rad + 2*p1*xu*yu + p2*(r2 + 2*xu*xu)
    yd = yu*rad + p1*(r2 + 2*yu*yu) + 2*p2*xu*yu
    return np.stack([fx*xd + cx, fy*yd + cy], -1)

mine = mei(rays)
err = np.linalg.norm(mine - ref, axis=1)
ok = np.isfinite(err)
print("compared %d rays against cv2.omnidir" % ok.sum())
print("  max error %.3e px, mean %.3e px" % (err[ok].max(), err[ok].mean()))
print("  VERDICT: %s" % ("identical" if err[ok].max() < 1e-6 else "MISMATCH - do not port"))
