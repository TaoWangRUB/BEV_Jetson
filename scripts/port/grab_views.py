#!/usr/bin/env python3
"""Grab frames from ROS 2 image topics, and/or montage the 4 cameras + panorama.

Run inside the Foxy container (rclpy + OpenCV). Two modes:

  # grab the first frame of each live topic -> <out-dir>/<sanitized-topic>.png
  python3 scripts/port/grab_views.py grab --out-dir /views --rotate180 \
      --topics /cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw

  # build a labeled montage (4 cams on top, panorama full-width below)
  python3 scripts/port/grab_views.py montage --dir /views --out /views/montage.png

Used by scripts/capture_montage_tx2.sh (which orchestrates the capture + panorama runs).
"""
import argparse
import os
import re
import time

CAM_LABELS = {
    "cam1": "/cam1  port f  +X right",
    "cam2": "/cam2  port d  -X left",
    "cam3": "/cam3  port e  +Y front",
    "cam4": "/cam4  port c  -Y back",
}


def sanitize(t):
    return re.sub(r"\W+", "_", t.strip("/"))


def cmd_grab(a):
    import cv2
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    os.makedirs(a.out_dir, exist_ok=True)
    rclpy.init()
    n = Node("grab_views")
    got = {}

    def make(t):
        def cb(m):
            if t in got:
                return
            img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width)
            if a.rotate180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(a.out_dir, sanitize(t) + ".png"), img)
            got[t] = True
        return cb

    for t in a.topics:
        n.create_subscription(Image, t, make(t), qos_profile_sensor_data)
    end = time.monotonic() + a.secs
    while rclpy.ok() and len(got) < len(a.topics) and time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    print("grabbed:", sorted(got))
    n.destroy_node()
    rclpy.shutdown()


def cmd_montage(a):
    import cv2
    import numpy as np

    tiles = []
    for c in ["cam1", "cam2", "cam3", "cam4"]:
        p = os.path.join(a.dir, "%s_image_raw.png" % c)
        im = cv2.imread(p, 0)
        im = np.zeros((270, 480), np.uint8) if im is None else cv2.resize(im, (480, 270))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, CAM_LABELS[c], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(im)
    row = np.hstack(tiles)  # 1920 wide

    pano = cv2.imread(os.path.join(a.dir, "bev_panorama.png"), 0)
    if pano is None:
        pano = np.zeros((540, 1920), np.uint8)
    h = int(1920 * pano.shape[0] / pano.shape[1])
    pano = cv2.cvtColor(cv2.resize(pano, (1920, h)), cv2.COLOR_GRAY2BGR)
    # az gridlines at each camera's optical axis (panorama center = az 0 = +Y front)
    for frac, txt in [(0.0, "c -Y back"), (0.25, "d -X left"), (0.5, "e +Y FRONT"),
                      (0.75, "f +X right"), (0.999, "c -Y")]:
        gx = int(frac * (pano.shape[1] - 1))
        cv2.line(pano, (gx, 0), (gx, pano.shape[0]), (0, 200, 0), 1)
        cv2.putText(pano, txt, (min(gx + 4, pano.shape[1] - 90), 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(a.out, np.vstack([row, pano]))
    print("montage ->", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grab")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--topics", nargs="+", required=True)
    g.add_argument("--rotate180", action="store_true")
    g.add_argument("--secs", type=float, default=12.0)
    g.set_defaults(func=cmd_grab)
    m = sub.add_parser("montage")
    m.add_argument("--dir", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_montage)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
