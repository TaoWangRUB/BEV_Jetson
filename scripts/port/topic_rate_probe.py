#!/usr/bin/env python3
"""Count what a SUBSCRIBER actually receives, to separate DDS from whatever consumes it.

WHY. On 2026-09-02 the capture node published a clean 30 Hz - its own set counter showed
150 sets per 5 s window at 66 us skew - while `ros2 bag record` wrote 6.1 Hz per camera,
with the disk only a quarter busy. That proves frames were lost between publisher and
recorder. It does NOT say where: DDS transport, CDR serialisation, and rosbag2's sqlite
writer are three different suspects and the bag rate cannot tell them apart.

This subscribes with the same SensorDataQoS the publisher uses and only counts. If it sees
~30 Hz, the transport is fine and the recorder is the bottleneck. If it sees ~6 Hz, the
loss is in the transport and a faster writer will not help.

  topic_rate_probe.py --seconds 20 /cam1/image_raw /cam2/image_raw ...
"""
import argparse, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Probe(Node):
    def __init__(self, topics):
        super().__init__("topic_rate_probe")
        self.n = {t: 0 for t in topics}
        self.bytes = {t: 0 for t in topics}
        self.t0 = time.monotonic()
        for t in topics:
            # same QoS as the publisher: best_effort. A reliable subscriber would simply not
            # match, receive nothing, and look exactly like a dead link.
            self.create_subscription(Image, t,
                                     lambda m, tt=t: self._got(tt, m), qos_profile_sensor_data)

    def _got(self, topic, msg):
        self.n[topic] += 1
        self.bytes[topic] += len(msg.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("topics", nargs="+")
    a = ap.parse_args()

    rclpy.init()
    p = Probe(a.topics)
    end = time.monotonic() + a.seconds
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(p, timeout_sec=0.1)
    dt = time.monotonic() - p.t0
    total_mb = 0.0
    print("\nreceived over %.1f s:" % dt)
    for t in a.topics:
        mb = p.bytes[t] / 1e6
        total_mb += mb
        print("  %-22s %5d msgs  %6.2f Hz  %6.1f MB/s" % (t, p.n[t], p.n[t] / dt, mb / dt))
    print("  %-22s %27.1f MB/s aggregate" % ("", total_mb / dt))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
