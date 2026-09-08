#!/usr/bin/env python3
"""Visualise a cuVSLAM replay in Rerun: trajectory, landmark point cloud, camera images.

  rerun_odometry.py <odom_or_cloud_dir> [--images <camera_bag>] [--save [out.rrd]]
                    [--spawn] [--serve] [--img-stride N]

<odom_or_cloud_dir> is the rosbag2 directory (or a parent of it) holding /cuvslam/odometry,
and optionally /cuvslam/landmarks. Pass --images to overlay the source camera frames
(the odometry bag carries no images), matched to poses by header stamp.
"""
import sys, argparse, pathlib, tempfile, hashlib, numpy as np
import rerun as rr
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
CAM_TOPICS = [f"/cam{i}/image_raw" for i in range(1, 5)]


def find_bag(d: pathlib.Path) -> pathlib.Path:
    if list(d.glob("*.db3")):
        return _ros2_safe_name(d)
    return _ros2_safe_name(
        next(p for p in d.rglob("*") if p.is_dir() and list(p.glob("*.db3"))))


def _ros2_safe_name(bag: pathlib.Path) -> pathlib.Path:
    """Work around rosbags dispatching on the SUFFIX, not the content.

    `AnyReader` decides ROS 1 vs ROS 2 with `any(x.suffix != '.bag' for x in paths)`
    (rosbags/highlevel/anyreader.py). Our own README tells people to build ROS 2 bags as
    `raw_log_to_bag.py -o /tmp/run1.bag`, which makes a *directory* called `run1.bag` — so
    every viewer that opens it gets the rosbag1 reader and dies with the very confusing
    "Could not open file ...: Is a directory". Hand back a suffix-free symlink instead;
    the caller's data is not touched or renamed.
    """
    if bag.suffix != ".bag":
        return bag
    link = pathlib.Path(tempfile.gettempdir()) / ("rosbag2-" + bag.name[:-4] + "-" +
                                                  hashlib.sha1(str(bag.resolve()).encode())
                                                  .hexdigest()[:8])
    if not link.is_symlink():
        link.symlink_to(bag.resolve(), target_is_directory=True)
    return link


def hstamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def read_bag(bag: pathlib.Path):
    ots, pos, quat, child = [], [], [], "cam1_optical_frame"
    clouds = []  # (stamp, Nx3)
    with AnyReader([bag], default_typestore=TS) as r:
        want = {"/cuvslam/odometry", "/cuvslam/landmarks"}
        conns = [c for c in r.connections if c.topic in want]
        if not any(c.topic == "/cuvslam/odometry" for c in conns):
            sys.exit("no /cuvslam/odometry in the bag - did tracking ever start?")
        for con, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, con.msgtype)
            if con.topic == "/cuvslam/odometry":
                child = m.child_frame_id
                p, o = m.pose.pose.position, m.pose.pose.orientation
                ots.append(hstamp(m)); pos.append([p.x, p.y, p.z]); quat.append([o.x, o.y, o.z, o.w])
            else:
                buf = np.frombuffer(bytes(m.data), np.uint8).reshape(m.width, m.point_step)
                xyz = buf[:, :12].copy().view(np.float32).reshape(-1, 3)
                clouds.append((hstamp(m), xyz))
    return np.array(ots), np.array(pos), np.array(quat), child, clouds


def read_images(bag: pathlib.Path, stride: int):
    """Return {topic: [(stamp, HxW uint8), ...]}, decimated by stride."""
    out = {t: [] for t in CAM_TOPICS}
    with AnyReader([bag], default_typestore=TS) as r:
        conns = [c for c in r.connections if c.topic in CAM_TOPICS]
        counts = {t: 0 for t in CAM_TOPICS}
        for con, t, raw in r.messages(connections=conns):
            n = counts[con.topic]; counts[con.topic] = n + 1
            if n % stride:
                continue
            m = r.deserialize(raw, con.msgtype)
            img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width)
            out[con.topic].append((hstamp(m), img))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--images", help="source camera bag to overlay frames from")
    ap.add_argument("--img-stride", type=int, default=3, help="log every Nth camera frame")
    ap.add_argument("--save", nargs="?", const="", default=None,
                    help="write an .rrd (default path: <bag>/replay.rrd)")
    ap.add_argument("--spawn", action="store_true", help="open the native viewer")
    ap.add_argument("--serve", action="store_true", help="start a web viewer and print the URL")
    a = ap.parse_args()

    bag = find_bag(pathlib.Path(a.path))
    ts, P, Q, child, clouds = read_bag(bag)
    if len(P) < 2:
        sys.exit("fewer than 2 poses - tracking did not hold")
    t0 = ts[0]
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    path_len = np.concatenate([[0.0], np.cumsum(steps)])
    dt = np.diff(ts)
    # A duplicate/near-zero dt would otherwise blow speed up to tens of m/s.
    safe = np.where(dt > 1e-3, dt, np.nan)
    speed = np.concatenate([[0.0], np.nan_to_num(steps / safe)])
    disp = np.linalg.norm(P - P[0], axis=1)
    images = read_images(find_bag(pathlib.Path(a.images)), a.img_stride) if a.images else {}

    rr.init("bev_cuvslam_replay", spawn=a.spawn)
    if a.serve:
        rr.serve_web()
    out = None
    if a.save is not None or (not a.spawn and not a.serve):
        out = pathlib.Path(a.save) if a.save else bag / "replay.rrd"
        rr.save(str(out))

    rr.log("odom", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("odom/trajectory", rr.LineStrips3D([P], colors=[0x33AAFFFF], radii=0.008), static=True)
    rr.log("odom/start", rr.Points3D([P[0]], colors=[0x00FF00FF], radii=0.05), static=True)
    rr.log("odom/end", rr.Points3D([P[-1]], colors=[0xFF0000FF], radii=0.05), static=True)

    for i in range(len(ts)):
        rr.set_time("wall", timestamp=ts[i])
        rr.log(f"odom/{child}", rr.Transform3D(translation=P[i], quaternion=Q[i], axis_length=0.15))
        rr.log("odom/head", rr.Points3D([P[i]], colors=[0xFFFF00FF], radii=0.03))
        rr.log("plots/speed_mps", rr.Scalars(float(speed[i])))
        rr.log("plots/path_length_m", rr.Scalars(float(path_len[i])))
        rr.log("plots/displacement_m", rr.Scalars(float(disp[i])))

    for stamp, xyz in clouds:
        rr.set_time("wall", timestamp=stamp)
        rr.log("odom/landmarks", rr.Points3D(xyz, colors=[0xBBBBBBFF], radii=0.01))

    for topic, frames in images.items():
        ent = "cameras/" + topic.strip("/").split("/")[0]
        for stamp, img in frames:
            rr.set_time("wall", timestamp=stamp)
            rr.log(ent, rr.Image(img))

    span = ts[-1] - t0
    ncloud = clouds[-1][1].shape[0] if clouds else 0
    print("logged %d poses over %.1fs (%.2f Hz)" % (len(ts), span, len(ts) / span))
    print("  path %.2f m | end displacement %.2f m | peak speed %.2f m/s"
          % (path_len[-1], disp[-1], np.nanmax(speed)))
    print("  landmark clouds: %d msgs, final map %d points" % (len(clouds), ncloud))
    if images:
        print("  camera frames overlaid: " + ", ".join(
            "%s=%d" % (k.split("/")[1], len(v)) for k, v in images.items()))
    if out:
        print("  wrote %s" % out)
        print("  open it with:  rerun %s" % out)


if __name__ == "__main__":
    main()
