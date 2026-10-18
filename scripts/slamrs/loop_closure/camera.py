"""KB4 fisheye model, the rig, and SE(3) helpers.

The KB4 form is Basalt's: theta = atan2(r, z), so rays past 90 deg incidence are representable,
which the ~192 deg lenses need (README 4.8). Everything takes and returns numpy arrays; poses are
4x4 float64 matrices, quaternions are xyzw as slam-rs reports them.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def quat_xyzw_to_R(q):
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_xyzw(R):
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def pose7_to_T(p7):
    """[tx, ty, tz, qx, qy, qz, qw] -> 4x4."""
    p7 = np.asarray(p7, dtype=np.float64)
    return se3(quat_xyzw_to_R(p7[3:7]), p7[:3])


def T_to_pose7(T):
    return np.concatenate([T[:3, 3], R_to_quat_xyzw(T[:3, :3])])


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def rotvec_to_R(v):
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * th / (2 * np.sin(th))


def rot_angle_deg(R):
    return np.degrees(np.linalg.norm(R_to_rotvec(R)))


def yaw_of(R):
    """Heading about world +Z (the VIO's gravity axis)."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class KB4:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float
    width: int
    height: int
    theta_max: float = np.radians(98.0)  # beyond this the polynomial is unconstrained by the fit

    def _d(self, theta):
        t2 = theta * theta
        return theta * (1 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def project(self, P):
        """Camera-frame points (N,3) -> pixels (N,2) and a validity mask (inside the modelled FOV
        and the image)."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        d = self._d(theta)
        safe = r > 1e-12
        u = np.where(safe, self.fx * d * x / np.where(safe, r, 1.0) + self.cx, self.cx)
        v = np.where(safe, self.fy * d * y / np.where(safe, r, 1.0) + self.cy, self.cy)
        uv = np.stack([u, v], -1)
        ok = (theta < self.theta_max) & (u >= 0) & (u < self.width - 1) & (v >= 0) & (v < self.height - 1)
        return uv, ok

    def unproject(self, uv):
        """Pixels (N,2) -> unit bearings (N,3) in the camera frame. Newton on d(theta) = r_u."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        mx = (uv[:, 0] - self.cx) / self.fx
        my = (uv[:, 1] - self.cy) / self.fy
        ru = np.hypot(mx, my)
        theta = ru.copy()
        for _ in range(12):
            t2 = theta * theta
            f = self._d(theta) - ru
            fp = 1 + t2 * (3 * self.k1 + t2 * (5 * self.k2 + t2 * (7 * self.k3 + t2 * 9 * self.k4)))
            theta = theta - f / np.where(np.abs(fp) < 1e-9, 1e-9, fp)
        safe = ru > 1e-12
        s = np.sin(theta)
        bx = np.where(safe, s * mx / np.where(safe, ru, 1.0), 0.0)
        by = np.where(safe, s * my / np.where(safe, ru, 1.0), 0.0)
        bz = np.cos(theta)
        b = np.stack([bx, by, bz], -1)
        return b / np.linalg.norm(b, axis=1, keepdims=True)


@dataclass
class Rig:
    cams: list            # KB4 per camera
    T_imu_cam: list       # 4x4 per camera: camera pose in the IMU (rig) frame
    T_cam_imu: list       # inverses

    @classmethod
    def from_json(cls, path, theta_max_deg=98.0):
        c = json.load(open(Path(path)))["value0"]
        cams, Tic, Tci = [], [], []
        for intr, res, ext in zip(c["intrinsics"], c["resolution"], c["T_imu_cam"]):
            if intr["camera_type"] != "kb4":
                raise ValueError(f"expected kb4 intrinsics, got {intr['camera_type']}")
            i = intr["intrinsics"]
            cams.append(KB4(i["fx"], i["fy"], i["cx"], i["cy"], i["k1"], i["k2"], i["k3"], i["k4"],
                            int(res[0]), int(res[1]), np.radians(theta_max_deg)))
            T = se3(quat_xyzw_to_R([ext["qx"], ext["qy"], ext["qz"], ext["qw"]]), [ext["px"], ext["py"], ext["pz"]])
            Tic.append(T)
            Tci.append(inv(T))
        return cls(cams, Tic, Tci)

    @property
    def n(self):
        return len(self.cams)
