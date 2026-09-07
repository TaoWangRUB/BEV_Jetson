#!/usr/bin/env python3
"""Triangulate scale-anchor points selected from a Rerun virtual-camera pane.

Rerun shows a selected observation as `/rig/camN/observations[id]`, plus its
2D position. This tool finds that observation, follows its cuVSLAM ID through
the replay, and intersects its calibrated rays to recover a world-space point.

Example (one click per endpoint):
  python3 scripts/vo/measure_replay_anchor.py datasets/replay_out/obs_20260903_140714 \
    --click 621.319583,1,261.491,138.396 \
    --click 621.319583,1,400.000,138.396
"""
import argparse
import pathlib
import sys

import numpy as np
import yaml
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from render_multicam_video import VCAMS, quat_to_R  # noqa: E402
from rerun_odometry import TS, find_bag, read_bag  # noqa: E402
from rerun_virtual_pinholes import rot_y  # noqa: E402


def read_observations(bag):
    """Return {timestamp: Nx4 float32 [u, v, virtual_camera, landmark_id]}."""
    observations = {}
    with AnyReader([bag], default_typestore=TS) as reader:
        connections = [connection for connection in reader.connections
                       if connection.topic == "/cuvslam/observations"]
        if not connections:
            raise RuntimeError("no /cuvslam/observations in this bag")
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            rows = (np.frombuffer(bytes(message.data), np.uint8)
                    .reshape(message.width, message.point_step)[:, :16].copy()
                    .view(np.float32).reshape(-1, 4))
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            observations[stamp] = rows
    return observations


def parse_click(value):
    try:
        timestamp, camera, u, v = (float(field) for field in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("click must be TIME,VCAM,U,V") from error
    if camera < 0 or camera >= len(VCAMS) or camera != int(camera):
        raise argparse.ArgumentTypeError("VCAM must be an integer from 0 to 7")
    return timestamp, int(camera), u, v


def nearest_observation(observations, click, max_time_error, max_pixel_error):
    timestamp, camera, u, v = click
    stamps = np.asarray(sorted(observations))
    index = int(np.abs(stamps - timestamp).argmin())
    stamp = stamps[index]
    if abs(stamp - timestamp) > max_time_error:
        raise RuntimeError("no observation frame within %.3f s of %.6f" %
                           (max_time_error, timestamp))
    rows = observations[stamp]
    rows = rows[rows[:, 2].astype(int) == camera]
    if not len(rows):
        raise RuntimeError("virtual camera %d has no observations at %.6f" % (camera, stamp))
    distances = np.hypot(rows[:, 0] - u, rows[:, 1] - v)
    row = rows[int(distances.argmin())]
    if distances.min() > max_pixel_error:
        raise RuntimeError("nearest observation is %.1f px away (limit %.1f px)" %
                           (distances.min(), max_pixel_error))
    return stamp, row, distances.min()


def ray_for_observation(row, stamp, pose_stamps, positions, quaternions, transforms, focal, cx, cy):
    pose_index = int(np.abs(pose_stamps - stamp).argmin())
    if abs(pose_stamps[pose_index] - stamp) > 0.03:
        raise RuntimeError("no odometry pose within 30 ms of observation %.6f" % stamp)
    camera = int(row[2])
    rotation_cam1_from_virtual, translation_cam1 = transforms[camera]
    ray_virtual = np.array([row[0] - cx, row[1] - cy, focal], float)
    ray_virtual /= np.linalg.norm(ray_virtual)
    ray_cam1 = ray_virtual @ rotation_cam1_from_virtual.T
    rotation_world_from_rig = quat_to_R(quaternions[pose_index])
    origin = positions[pose_index] + translation_cam1 @ rotation_world_from_rig.T
    direction = ray_cam1 @ rotation_world_from_rig.T
    return origin, direction / np.linalg.norm(direction)


def triangulate(rays):
    """Least-squares point closest to all world-space rays, plus residuals."""
    matrix = np.zeros((3, 3))
    vector = np.zeros(3)
    identity = np.eye(3)
    for origin, direction in rays:
        projector = identity - np.outer(direction, direction)
        matrix += projector
        vector += projector @ origin
    point = np.linalg.solve(matrix, vector)
    residuals = np.array([np.linalg.norm(np.cross(point - origin, direction))
                          for origin, direction in rays])
    return point, residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="replay bag directory containing observations and odometry")
    parser.add_argument("--click", action="append", required=True, type=parse_click,
                        help="Rerun hover values: sensor seconds, virtual camera, u, v")
    parser.add_argument("--vstereo", default="config/rig/virtual_stereo_imx296.yaml")
    parser.add_argument("--rig", default="config/rig/rig_extrinsics_imx296.yaml")
    parser.add_argument("--max-time-error", type=float, default=0.03)
    parser.add_argument("--max-pixel-error", type=float, default=8.0)
    parser.add_argument("--min-views", type=int, default=4)
    arguments = parser.parse_args()

    bag = find_bag(pathlib.Path(arguments.bag))
    pose_stamps, positions, quaternions, _, _ = read_bag(bag)
    observations = read_observations(bag)
    virtual = yaml.safe_load(open(arguments.vstereo))["virtual_pinhole"]
    rig = yaml.safe_load(open(arguments.rig))["rig_in_cam1"]
    focal = float(virtual["focal_px"])
    cx, cy = int(virtual["width"]) / 2.0, int(virtual["height"]) / 2.0
    signs = {-1: np.radians(-45), 1: np.radians(45)}
    transforms = [(np.asarray(rig[c])[:3, :3] @ rot_y(signs[sign]),
                   np.asarray(rig[c])[:3, 3]) for c, sign in VCAMS]

    points = []
    for click in arguments.click:
        stamp, selected, pixel_error = nearest_observation(
            observations, click, arguments.max_time_error, arguments.max_pixel_error)
        landmark_id = int(selected[3])
        matching = [(time, row) for time, rows in observations.items()
                    for row in rows if int(row[3]) == landmark_id]
        rays = [ray_for_observation(row, time, pose_stamps, positions, quaternions,
                                    transforms, focal, cx, cy)
                for time, row in matching]
        if len(rays) < arguments.min_views:
            raise RuntimeError("landmark %d has only %d views (need %d)" %
                               (landmark_id, len(rays), arguments.min_views))
        point, residuals = triangulate(rays)
        points.append(point)
        print("click t=%.6f vcam=%d uv=(%.2f, %.2f) -> id=%d (%.2f px)" %
              (stamp, int(selected[2]), selected[0], selected[1], landmark_id, pixel_error))
        print("  point [%.4f, %.4f, %.4f] m; %d views; ray residual median %.3f m, p95 %.3f m" %
              (*point, len(rays), np.median(residuals), np.percentile(residuals, 95)))

    for index in range(0, len(points) - 1, 2):
        print("anchor %d-%d distance %.4f m" %
              (index + 1, index + 2, np.linalg.norm(points[index + 1] - points[index])))


if __name__ == "__main__":
    main()