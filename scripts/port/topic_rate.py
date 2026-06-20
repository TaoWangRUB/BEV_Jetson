#!/usr/bin/env python3
"""Count messages on one or more topics over a fixed window and print the rate.

A buffering-proof alternative to `ros2 topic hz` for bring-up checks: it prints
once at the end (no streaming/SIGTERM-buffer-loss games). Matches the publisher's
default QoS (reliable / keep_last 10).

Usage (inside the Foxy container, after sourcing the overlay):
    python3 scripts/port/topic_rate.py --secs 10 /cam1/image_raw /cam2/image_raw ...
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topics", nargs="+")
    ap.add_argument("--secs", type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("topic_rate")
    counts = {t: 0 for t in args.topics}
    first = {t: None for t in args.topics}
    last = {t: None for t in args.topics}

    def make_cb(t):
        def cb(_msg):
            now = time.monotonic()
            counts[t] += 1
            if first[t] is None:
                first[t] = now
            last[t] = now
        return cb

    for t in args.topics:
        node.create_subscription(Image, t, make_cb(t), qos_profile_sensor_data)

    end = time.monotonic() + args.secs
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"--- rate over {args.secs:.0f}s ---")
    for t in args.topics:
        n = counts[t]
        span = (last[t] - first[t]) if (first[t] and last[t] and last[t] > first[t]) else 0.0
        hz = (n - 1) / span if span > 0 else 0.0
        print(f"{t}: {n} msgs, {hz:.2f} Hz")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
