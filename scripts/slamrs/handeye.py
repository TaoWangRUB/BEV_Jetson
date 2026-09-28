"""Estimate R_imu_cam1 and the camera-IMU time offset from cuVSLAM odometry vs the gyro.

Angular velocity of cam1 from cuVSLAM (in cam1 frame) must equal R_cam1_imu @ gyro.
Scan the time offset, Kabsch-solve the rotation at each, keep the best.
"""
import sqlite3
import struct
import sys

import numpy as np


def read_odom(db, topic="/cuvslam/odometry"):
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    tid = con.execute("select id from topics where name=?", (topic,)).fetchone()[0]
    out = []
    for (data,) in con.execute("select data from messages where topic_id=? order by timestamp", (tid,)):
        b = bytes(data)
        o = 4  # CDR encapsulation header

        def al(n):
            nonlocal o
            o += (-(o - 4)) % n

        sec, nsec = struct.unpack_from("<iI", b, o); o += 8
        for _ in range(2):  # header.frame_id, child_frame_id
            al(4); n = struct.unpack_from("<I", b, o)[0]; o += 4 + n
        al(8)
        px, py, pz, qx, qy, qz, qw = struct.unpack_from("<7d", b, o)
        out.append((sec * 10**9 + nsec, px, py, pz, qx, qy, qz, qw))
    return np.array(out)


def quat_to_R(q):
    x, y, z, w = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def log_so3(R):
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s = np.where(th > 1e-9, th / (2 * np.sin(np.maximum(th, 1e-9))), 0.5)
    return v * s[:, None]


def kabsch(A, B):  # R minimising |R A - B|, rows are vectors
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def main(bag, imu_csv):
    od = read_odom(bag)
    t = od[:, 0]
    R = quat_to_R(od[:, 4:8])
    dt = np.diff(t) * 1e-9
    w_cam = log_so3(np.einsum("nji,njk->nik", R[:-1], R[1:])) / dt[:, None]  # body-frame rate
    tm = 0.5 * (t[:-1] + t[1:])
    # drop discontinuities (tracking re-inits) and near-zero-gap artefacts
    step = np.linalg.norm(np.diff(od[:, 1:4], axis=0), axis=1) / dt
    good = (step < 5.0) & (dt > 0.02) & (dt < 0.12) & (np.linalg.norm(w_cam, axis=1) < 6)

    imu = np.loadtxt(imu_csv, delimiter=",", comments="#")
    ti, gyro = imu[:, 0], imu[:, 4:7]
    rest = ti < ti[0] + 2e9
    gyro = gyro - gyro[rest].mean(0)  # rig is still in the first seconds of every log

    best = None
    for off_ms in np.arange(-40, 40.5, 0.5):
        tq = tm + off_ms * 1e6
        # average gyro over each odometry interval, like the finite difference does
        gi = np.stack([np.interp(tq, ti, gyro[:, k]) for k in range(3)], -1)
        ok = good & (tq > ti[0]) & (tq < ti[-1]) & (np.linalg.norm(gi, axis=1) > 0.05)
        Rci = kabsch(gi[ok], w_cam[ok])
        res = np.linalg.norm(gi[ok] @ Rci.T - w_cam[ok], axis=1)
        rms = np.sqrt((res ** 2).mean())
        if best is None or rms < best[0]:
            best = (rms, off_ms, Rci, ok.sum(), np.linalg.norm(w_cam[ok], axis=1).mean())
    rms, off_ms, Rci, n, wmean = best
    print(f"samples {n}, best offset t_imu = t_cam + {off_ms:.1f} ms, rms {rms:.4f} rad/s "
          f"(mean |w| {wmean:.3f} rad/s)")
    print("R_cam1_imu =\n", np.round(Rci, 4))
    # compare with the layout: angle between cam1's optical axis and body axes via R_imu_from_body
    R_imu_body = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
    R_body_cam1 = R_imu_body.T @ Rci.T
    z = R_body_cam1[:, 2]
    yaw = np.degrees(np.arctan2(z[1], z[0])); pitch = np.degrees(np.arcsin(-z[2]))
    print(f"cam1 optical axis in body FLU: {np.round(z, 3)} -> yaw {yaw:.1f} deg (expect ~+45, front-left), "
          f"pitch-down {pitch:.1f} deg")
    np.save(sys.argv[3] if len(sys.argv) > 3 else "R_cam1_imu.npy", np.concatenate([Rci.ravel(), [off_ms]]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
