#!/usr/bin/env python3
"""Generate the cuVSLAM rig extrinsics (rig_from_camera, rig_from_imu) from the
physical layout of the BEV surround rig.

Rig frame: origin at the TX2 board center; X = right (lateral), Y = forward
(longitudinal, toward the top edge), Z = up. (Matches the MPU-9250 axes, and the
accel reads +Z at rest -> Z up.)

Cameras: 4 on the side faces of a 30 mm cube whose center is 40 mm above the
board center, faces 15 mm from the cube center:
    cam1 (port c) -> +Y    cam2 (port d) -> +X
    cam3 (port e) -> -Y    cam4 (port f) -> -X
Camera optical frame: Z = optical axis (outward), X = right-in-image, Y = down.
ASSUMPTION: cameras mounted upright, image-up = rig +Z. (Verify against the
flip-method used at capture; if the cameras are physically rolled 180, set
--cam-up 0 0 -1.)

IMU: on the board plane (Z~0), 2.65 mm off the X centerline, 27.5 mm forward
(+Y) of center; its axes already align with the rig (x-lateral, y-longitudinal),
so rig_from_imu rotation = identity.

    ./gen_rig_extrinsics.py [--cam-up 0 0 1] > config/rig/rig_extrinsics.yaml
"""
import argparse
import numpy as np

CUBE_UP = 0.040      # cube center height above board center (m)
FACE = 0.015         # face offset from cube center (m)
IMU_T = (0.00265, 0.0275, 0.0)   # IMU position in rig frame (m)

# cam -> (port, outward facing unit vector, translation). Rig is yawed 180 deg
# about Z, so facings/positions are rotated accordingly:
#   port c -> -Y, port d -> -X, port e -> +Y, port f -> +X
CAMS = {
    "cam1": ("c", (0, -1, 0), (0.0,  -FACE, CUBE_UP)),
    "cam2": ("d", (-1, 0, 0), (-FACE, 0.0,  CUBE_UP)),
    "cam3": ("e", (0, 1, 0), (0.0,  FACE, CUBE_UP)),
    "cam4": ("f", (1, 0, 0), (FACE, 0.0,  CUBE_UP)),
}


def rot_from_forward_up(forward, up):
    """Camera-optical R (columns = X_cam,Y_cam,Z_cam in rig frame). Z=forward,
    Y=down=-up, X=Y x Z."""
    z = np.array(forward, float); z /= np.linalg.norm(z)
    ydown = -np.array(up, float)
    ydown = ydown - z * (ydown @ z)      # orthogonalize against optical axis
    ydown /= np.linalg.norm(ydown)
    x = np.cross(ydown, z)
    return np.column_stack([x, ydown, z])


def quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return [w, x, y, z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-up", type=float, nargs=3, default=[0, 0, 1],
                    help="camera image-up direction in rig frame (default rig +Z)")
    args = ap.parse_args()
    up = args.cam_up

    print("# cuVSLAM rig extrinsics — rig_from_<frame>, generated from physical layout.")
    print("# rig frame: TX2 board center; X=right(lateral) Y=forward(longitudinal) Z=up. units: metres.")
    print(f"# camera image-up = {up} (rig frame). translations in m, quaternions wxyz.")
    print("rig_frame: board_center")
    print("cameras:")
    for cam, (port, fwd, t) in CAMS.items():
        R = rot_from_forward_up(fwd, up)
        q = quat_wxyz(R)
        print(f"  {cam}:")
        print(f"    port: {port}")
        print(f"    forward: [{fwd[0]}, {fwd[1]}, {fwd[2]}]")
        print(f"    t_xyz_m: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}]")
        print(f"    q_wxyz: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print("imu:")
    print(f"    t_xyz_m: [{IMU_T[0]:.5f}, {IMU_T[1]:.5f}, {IMU_T[2]:.5f}]")
    print("    q_wxyz: [1.0, 0.0, 0.0, 0.0]")


if __name__ == "__main__":
    main()
