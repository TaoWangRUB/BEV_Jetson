#!/usr/bin/env python3
"""Measure the timestamp spread across N camera topics.

For the no-hardware-sync IMX219 rig: tells you whether the 4 free-running streams
can be ApproximateTime-synced, and what `sync_slop_ms` would be needed. Reports the
spread (max-min header.stamp of the most-recent frame on each topic) sampled over
time, plus the best (smallest) spread seen.

Usage (inside the Foxy container, after sourcing):
    python3 scripts/port/sync_check.py --secs 12 /cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topics", nargs="+")
    ap.add_argument("--secs", type=float, default=12.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("sync_check")
    latest = {t: None for t in args.topics}
    counts = {t: 0 for t in args.topics}

    def make_cb(t):
        def cb(msg):
            latest[t] = stamp_ns(msg)
            counts[t] += 1
        return cb

    for t in args.topics:
        node.create_subscription(Image, t, make_cb(t), 10)

    best = None
    samples = []
    end = time.monotonic() + args.secs
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if all(latest[t] is not None for t in args.topics):
            vals = [latest[t] for t in args.topics]
            spread_ms = (max(vals) - min(vals)) / 1e6
            samples.append(spread_ms)
            if best is None or spread_ms < best:
                best = spread_ms

    print(f"--- sync check over {args.secs:.0f}s ---")
    for t in args.topics:
        print(f"  {t}: {counts[t]} msgs")
    if samples:
        avg = sum(samples) / len(samples)
        print(f"latest-frame spread across {len(args.topics)} cams: "
              f"best={best:.1f} ms  avg={avg:.1f} ms  (n={len(samples)} samples)")
        print("note: ApproximateTime needs sync_slop_ms >= the achievable 4-way spread; "
              "if best >> frame period, timestamps use different clocks/epochs.")
    else:
        print("never had a frame from all topics simultaneously — check topic names/QoS.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
