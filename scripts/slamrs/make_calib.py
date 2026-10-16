"""Write the Basalt-format calibration JSON slam-rs reads, from the BEV rig files.

  python3 scripts/slamrs/make_calib.py [--scale 2] [-o config/slamrs/bev_calib_s2.json]

intrinsics : KB4 fitted to the round-2 Mei solves (fit_kb4.py), at 1/scale resolution
extrinsics : T_imu_cami = T_imu_cam1 @ T_cam1_cami
  T_cam1_cami from config/rig/rig_extrinsics_imx296.yaml (rig_in_cam1, round 2, 2026-09-01)
  T_cam1_imu  from the round-2 Kalibr cam-IMU solve, camimu_5ms - the solve whose Delta is
              config/calib/imu_mpu9250.yaml's timeshift_cam_imu. That file is not in git
              (datasets/ is ignored), so the committed JSON is the only tracked copy of it.
              handeye.py's log-only estimate agrees to 0.33 deg (run6) / 0.96 deg (run5).
noise      : imu_mpu9250.yaml's datasheet densities x5 (a VIO tuning choice, not a measurement)
Delta is NOT in this file: run_slamrs.py shifts the IMU stamps by it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from fit_kb4 import fit

REPO = Path(__file__).resolve().parents[2]
RIG = REPO / "config/rig/rig_extrinsics_imx296.yaml"
IMU = REPO / "config/calib/imu_mpu9250.yaml"
KALIBR = REPO / "datasets/calib_20260901/ros1/camimu_5ms-camchain-imucam.yaml"
NOISE_INFLATION = 5.0


def R_to_quat(R):  # xyzw
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    out = a.out or REPO / f"config/slamrs/bev_calib_s{a.scale}.json"

    rig = yaml.safe_load(open(RIG))["rig_in_cam1"]
    T_c1_ci = [np.array(rig[f"cam{i}"], float) for i in (1, 2, 3, 4)]
    T_c1_imu = np.array(yaml.safe_load(open(KALIBR))["cam0"]["T_cam_imu"], float)
    T_imu_c1 = np.linalg.inv(T_c1_imu)
    imu = yaml.safe_load(open(IMU))

    T_imu_cam, intr, res = [], [], []
    for i in range(4):
        T = T_imu_c1 @ T_c1_ci[i]
        q = R_to_quat(T[:3, :3])
        T_imu_cam.append(dict(px=T[0, 3], py=T[1, 3], pz=T[2, 3], qx=q[0], qy=q[1], qz=q[2], qw=q[3]))
        p, stats, wh = fit(i + 1, a.scale)
        print(json.dumps(stats))
        intr.append(dict(camera_type="kb4", intrinsics=dict(zip(["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"], map(float, p)))))
        res.append(list(wh))

    k = NOISE_INFLATION
    calib = dict(value0=dict(
        T_imu_cam=T_imu_cam, intrinsics=intr, resolution=res, vignette=[],
        calib_accel_bias=[0.0] * 9, calib_gyro_bias=[0.0] * 12,
        imu_update_rate=float(imu["update_rate"]),
        accel_noise_std=[k * imu["accelerometer_noise_density"]] * 3,
        gyro_noise_std=[k * imu["gyroscope_noise_density"]] * 3,
        accel_bias_std=[k * imu["accelerometer_random_walk"]] * 3,
        gyro_bias_std=[k * imu["gyroscope_random_walk"]] * 3,
        cam_time_offset_ns=0))
    json.dump(calib, open(out, "w"), indent=1, default=float)
    print(f"wrote {out}; IMU origin in cam1 frame (Kalibr) {np.round(T_c1_imu[:3, 3], 4)}")


if __name__ == "__main__":
    main()
