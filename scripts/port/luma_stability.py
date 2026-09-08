#!/usr/bin/env python3
"""Measure brightness stability across the camera streams.

This catches the trigger-mode trap: with the IMX296 hardware trigger the exposure IS
the trigger pulse width, so Argus AE cannot move its main actuator and hunts on gain
instead — a 3.5 Hz limit cycle swinging ~171% of the mean. Any single frame looks
fine; only the sequence shows it. Under a locked AE the peak-to-peak sits at a few
luma levels (mains flicker beating with the trigger rate is the usual residue).

Reports per camera: mean luma, peak-to-peak, sd — plus frame count, rate and gap
count, which come free from the same subscription. For rate or timestamp-spread
questions on their own, use topic_rate.py and sync_check.py next door.

Usage (inside the Foxy container, after sourcing the overlay):
    python3 scripts/port/luma_stability.py --seconds 30
"""
import argparse
import statistics

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Health(Node):
    def __init__(self, topics, seconds):
        super().__init__("capture_health")
        self.seconds = seconds
        self.stamps = {t: [] for t in topics}
        self.luma = {t: [] for t in topics}
        for t in topics:
            self.create_subscription(Image, t, lambda m, t=t: self.on_image(t, m),
                                     qos_profile_sensor_data)
        self.t0 = self.get_clock().now()

    def on_image(self, topic, msg):
        # The frame's own sensor timestamp, not arrival time - arrival ordering says
        # more about the executor than about the cameras.
        self.stamps[topic].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        self.luma[topic].append(float(buf.mean()))
        if (self.get_clock().now() - self.t0).nanoseconds * 1e-9 > self.seconds:
            raise SystemExit(0)


def report(topics, stamps, luma):
    print(f"\n{'topic':<20}{'frames':>7}{'rate':>9}{'gaps':>6}"
          f"{'luma mean':>11}{'p2p':>8}{'sd':>7}")
    for t in topics:
        ts, lu = stamps[t], luma[t]
        if len(ts) < 3:
            print(f"{t:<20}{len(ts):>7}   (too few frames)")
            continue
        dt = [b - a for a, b in zip(ts, ts[1:])]
        med = statistics.median(dt)
        gaps = sum(1 for d in dt if d > 1.5 * med)
        p2p = max(lu) - min(lu)
        print(f"{t:<20}{len(ts):>7}{1/med:>8.2f}/s{gaps:>6}"
              f"{statistics.mean(lu):>11.1f}{p2p:>8.1f}{statistics.pstdev(lu):>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--topics", nargs="+",
                    default=[f"/cam{i}/image_raw" for i in range(1, 5)])
    a = ap.parse_args()

    rclpy.init()
    node = Health(a.topics, a.seconds)
    print(f"listening for {a.seconds:.0f} s on {' '.join(a.topics)} ...")
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        report(a.topics, node.stamps, node.luma)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
