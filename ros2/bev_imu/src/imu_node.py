#!/usr/bin/env python3
"""MPU-9250 IMU node — samples stamped at the DATA-READY EDGE, on CLOCK_MONOTONIC.

Where the timestamp comes from is the whole point. A reader that polls SPI and stamps
each sample when the read returns dates it to "whenever userspace got round to it" —
an error that is invisible in the data and fatal to visual-inertial fusion. Here the
IMU's own INT line drives the timing: the node blocks on that edge, stamps it, and only
then fetches numbers whose time is already known.

The edge handling is delegated to the J106 project's `j106-imu-read.py` rather than
reimplemented, because three board-specific facts are easy to get silently wrong:

  * the MPU's INT is on gpio-298 and the J106 INVERTS it with no pull-up, so the sensor
    is configured push-pull and its assertion arrives as a FALLING edge;
  * the GPIO chardev stamps its own events with CLOCK_REALTIME while the cameras are on
    CLOCK_MONOTONIC — mixing them misdates everything, so the chardev is used to WAIT
    and the timestamp is taken here;
  * MPU-9250 FSYNC is not brought out on this carrier and Tegra GTE timestamping is
    Xavier-only, so waking on the edge is genuinely the best this board allows (~50 us
    median wake latency, MAD 2.8 us under SCHED_FIFO — the median is bias, absorbed by
    the camera-IMU offset; the MAD is the real limit).

TIMESTAMPS (the contract — README 4.7): header.stamp is CLOCK_MONOTONIC, matching
`argus_capture_node`. It is NOT ROS system time; never compare it against now().

The DLPF group delay is REPORTED, not applied: the edge marks when the FILTERED sample
was ready, and the gyro path lags the accel path by ~1.0 ms at every matched bandwidth,
so a single correction cannot serve both. A front end that treats one timestamp as
covering both inherits that error knowingly rather than silently.

  ros2 run bev_imu imu_node --ros-args -p reader:=/opt/j106-tools/j106-imu-read.py
"""
import argparse
import gc
import importlib.util
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def load_reader(path):
    """Import j106-imu-read.py as a module (its filename is not a valid identifier)."""
    if not os.path.exists(path):
        raise RuntimeError(
            "IMU reader not found at %s — it lives in the J106 project "
            "(auvidea-j106-tx2/tools/j106-imu-read.py). Mount it into the container or "
            "set the 'reader' parameter." % path)
    spec = importlib.util.spec_from_file_location("j106_imu_read", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ImuNode(Node):
    def __init__(self):
        super().__init__("bev_imu")
        self.reader_path = self.declare_parameter("reader", "/opt/j106-tools/j106-imu-read.py").value
        self.frame_id = self.declare_parameter("frame_id", "imu0").value
        topic = self.declare_parameter("topic", "/imu0").value
        self.csv_path = self.declare_parameter("csv", "").value
        # SCHED_FIFO halves the wake-latency tail (p99 122.7 us -> 75.4 us) and barely
        # moves the median. 0 disables.
        self.rt_priority = self.declare_parameter("rt_priority", 80).value

        self.args = argparse.Namespace(
            spidev=self.declare_parameter("spidev", "/dev/spidev1.0").value,
            spi_speed=self.declare_parameter("spi_speed", 1000000).value,
            gpio=self.declare_parameter("gpio", 298).value,
            edge=None,                       # derived from active_low by the reader
            rate=self.declare_parameter("rate", 200).value,
            gyro_fs=self.declare_parameter("gyro_fs", 500).value,
            accel_fs=self.declare_parameter("accel_fs", 4).value,
            gyro_dlpf=self.declare_parameter("gyro_dlpf", 1).value,
            accel_dlpf=self.declare_parameter("accel_dlpf", 1).value,
            active_low=self.declare_parameter("active_low", False).value,
        )

        self.pub = self.create_publisher(Imu, topic, qos_profile_sensor_data)
        self.imu = load_reader(self.reader_path)

    def run(self):
        if self.rt_priority:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO,
                                      os.sched_param(int(self.rt_priority)))
                self.get_logger().info("running SCHED_FIFO at priority %d" % self.rt_priority)
            except PermissionError:
                self.get_logger().warn(
                    "cannot set SCHED_FIFO (needs root/CAP_SYS_NICE) — the wake-latency "
                    "tail will be ~2x worse; the median, which Delta absorbs, is unaffected")

        spi, mpu, src, info = self.imu.open_imu(self.args)
        self.log_provenance(info)
        csv = self.open_csv(info)

        # A cyclic-GC pause stalls the sample loop, and a stalled loop misses data-ready
        # edges outright — measured as one 78 ms gap and 29 lost samples in 18 s, against
        # 0 lost and sd 15.7 us for the same reader with no ROS layer on top. Nothing here
        # builds reference cycles, so freeze what startup allocated and stop collecting;
        # refcounting still frees the per-sample garbage.
        gc.collect()
        gc.freeze()
        gc.disable()

        # Warm the publish path BEFORE the sample loop. The first few publishes allocate
        # rmw/serialisation internals, and that cost lands on whichever data-ready edges
        # they coincide with — measured as 6 samples lost in the first seconds and none
        # afterwards. Warm it on a throwaway topic so /imu0 never carries a fake sample.
        warm = self.create_publisher(Imu, "~/warmup", qos_profile_sensor_data)
        for _ in range(20):
            warm.publish(Imu())
        self.destroy_publisher(warm)

        stats = self.imu.Stats()
        msg = Imu()
        msg.orientation_covariance[0] = -1.0   # this IMU reports no orientation
        nominal_ns = int(1e9 / info["rate_hz"])
        published = 0
        try:
            for t_ns, vals, seq in self.imu.iter_samples(
                    mpu, src, nominal_ns, self.args.spi_speed, 1000, stats,
                    stop=lambda: not rclpy.ok()):
                ax, ay, az, gx, gy, gz, temp_c, clipped = vals
                m = msg                       # reused: publish() serialises, so this is safe
                m.header.stamp = rclpy.time.Time(nanoseconds=t_ns).to_msg()
                m.header.frame_id = self.frame_id
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = ax, ay, az
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gx, gy, gz
                self.pub.publish(m)
                if csv:
                    csv.write("%d,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.2f,%d\n"
                              % (t_ns, ax, ay, az, gx, gy, gz, temp_c, seq))
                published += 1
                if published % (info["rate_hz"] * 10) == 0:
                    self.get_logger().info(
                        "%d samples, %d dropped (edge seen late), %d late reads"
                        % (published, stats.drops, stats.late))
        finally:
            if csv:
                csv.close()
            mpu.standby()
            src.close()
            spi.close()
            self.get_logger().info("stopped: %d samples, %d dropped" % (published, stats.drops))

    def log_provenance(self, info):
        """Say what was configured. An IMU stream whose filter settings are unknown
        cannot be calibrated against later."""
        self.get_logger().info(
            "MPU-9250 WHO_AM_I=0x%02x on %s, INT gpio-%d (%s, %s offset %d), %s"
            % (info["who_am_i"], info["spidev"], info["gpio"], info["edge"],
               info["chip"], info["offset"], info["clock"]))
        self.get_logger().info(
            "rate %.2f Hz, gyro +/-%d dps DLPF %.0f Hz (group delay %.2f ms), "
            "accel +/-%d g DLPF %.0f Hz (group delay %.2f ms)"
            % (info["rate_hz"], info["gyro_fs_dps"], info["gyro_dlpf_bw_hz"],
               info["gyro_group_delay_ms"], info["accel_fs_g"],
               info["accel_dlpf_bw_hz"], info["accel_group_delay_ms"]))
        lag = info["gyro_group_delay_ms"] - info["accel_group_delay_ms"]
        self.get_logger().warn(
            "group delays are REPORTED, not applied: the gyro lags the accel by %.2f ms. "
            "camera<->IMU offset Delta is UNMEASURED (see README 4.7)" % lag)

    def open_csv(self, info):
        if not self.csv_path:
            return None
        f = open(self.csv_path, "w")
        f.write("# MPU-9250, timestamped at the data-ready edge on %s\n" % info["clock"])
        f.write("# spidev=%s int_gpio=%d edge=%s rate_hz=%.2f\n"
                % (info["spidev"], info["gpio"], info["edge"], info["rate_hz"]))
        f.write("# gyro_fs_dps=%d gyro_dlpf_bw_hz=%.0f gyro_group_delay_ms=%.2f\n"
                % (info["gyro_fs_dps"], info["gyro_dlpf_bw_hz"], info["gyro_group_delay_ms"]))
        f.write("# accel_fs_g=%d accel_dlpf_bw_hz=%.0f accel_group_delay_ms=%.2f\n"
                % (info["accel_fs_g"], info["accel_dlpf_bw_hz"], info["accel_group_delay_ms"]))
        f.write("# group delays are NOT applied; delta_camera_imu = UNMEASURED\n")
        f.write("#timestamp [ns],a_x [m s^-2],a_y,a_z,w_x [rad s^-1],w_y,w_z,temp [C],seq\n")
        return f


def main():
    rclpy.init()
    node = ImuNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:                      # bring-up errors must be legible
        node.get_logger().error(str(e))
        sys.exit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
