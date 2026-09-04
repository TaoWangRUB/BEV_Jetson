// Garmin LIDAR-Lite range node — readings arrive already stamped with the TRIGGER's pulse
// counter, which is the whole reason the sensor hangs off the camtrig MCU instead of the
// Tegra's i2c.
//
// TIMESTAMPS. Two columns, and only one of them is exact:
//
//   pulses     the trigger edge counter, produced on the MCU. The capture side's `seq`
//              advances exactly one per trigger edge, so this is the column to JOIN a
//              range to a camera frame on: an integer identity, no clock comparison and
//              no offset to estimate. Unlike delta_camera_imu (docs/timestamps.md), there is
//              nothing here left unmeasured.
//
//   t_mono_ns  CLOCK_MONOTONIC when this process READ the line — the same clock imu0.csv
//              and the camera CSVs use, so range is roughly comparable to those. It is
//              NOT the instant of the measurement: it carries the acquisition (5-20 ms on
//              a v3), UART transit (~24 bytes at 115200 ≈ 2 ms) and the scheduler wake.
//              Ordering information. Do not calibrate against it.
//
// THE PORT IS SHARED WITH TRIGGER CONTROL, and that shapes the design. /dev/ttyTHS1 is
// full duplex and carries both the range stream and the camtrig command console, so the
// constraint is not the wire but the fd: two readers steal each other's bytes. This node
// is therefore the SOLE owner of the port for the life of a recording. The firmware marks
// every unsolicited line with '!' so the two are separable — anything without the marker
// is a reply to somebody else's command and is ignored here.
//
// A MISSING SENSOR IS NOT AN ERROR. This node never throws and never exits because the
// rangefinder is absent or silent: it publishes nothing, says so once, and keeps running.
// The reason is concrete — the port it holds is the trigger console, so a node that died
// on a missing optional sensor would take camera triggering down with it. Note this is the
// OPPOSITE of bev_imu, which throws on spidev setup failure; the IMU is not optional.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/range.hpp>

namespace {

int64_t mono_ns()
{
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

// LIDAR-Lite v3 spec. Used for the sensor_msgs/Range envelope and to reject readings the
// part cannot actually produce — a 0 from a v3 means "closer than it can measure", not
// "the target is at the lens", and publishing it as 0 m would be a lie a consumer cannot
// detect.
constexpr float kMinRangeM = 0.05f;
constexpr float kMaxRangeM = 40.0f;
constexpr float kFovRad = 0.008f;   // ~0.5 deg beam divergence

}  // namespace

class RangeNode : public rclcpp::Node {
 public:
  RangeNode() : Node("range_node")
  {
    port_     = declare_parameter<std::string>("port", "/dev/ttyTHS1");
    divisor_  = declare_parameter<int>("divisor", 15);
    csv_path_ = declare_parameter<std::string>("csv", "");
    frame_id_ = declare_parameter<std::string>("frame_id", "range0");
    const auto topic = declare_parameter<std::string>("topic", "/range0");

    if (divisor_ < 1) {
      RCLCPP_WARN(get_logger(), "divisor %d < 1; using 1", divisor_);
      divisor_ = 1;
    }

    pub_ = create_publisher<sensor_msgs::msg::Range>(topic, rclcpp::SensorDataQoS());

    // Opening the port is the one thing that genuinely must work: it is also the trigger
    // console. Even so this only warns — a recording without range is still a recording,
    // and the failure is visible in the log and in the empty CSV.
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      RCLCPP_ERROR(get_logger(), "cannot open %s (%s) — running without range",
                   port_.c_str(), std::strerror(errno));
    } else if (!configure_port()) {
      RCLCPP_ERROR(get_logger(), "cannot configure %s (%s) — running without range",
                   port_.c_str(), std::strerror(errno));
      ::close(fd_);
      fd_ = -1;
    }

    open_csv();

    if (fd_ >= 0) {
      drain(300);                 // discard a previous session's stream or a stale reply
      write_cmd("range auto " + std::to_string(divisor_) + "\n");
      RCLCPP_INFO(get_logger(),
                  "range on %s, one reading every %d trigger pulses -> %s%s",
                  port_.c_str(), divisor_, topic.c_str(),
                  csv_path_.empty() ? "" : (" + " + csv_path_).c_str());
    }

    // 5 ms: well inside the ~500 ms between readings at the default divisor, so a line is
    // never left sitting in the kernel buffer long enough to bias t_mono_ns.
    timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] { poll(); });
  }

  ~RangeNode() override
  {
    if (fd_ >= 0) {
      // Leave the MCU quiet for whoever opens the port next. Best effort: a wedged board
      // must not stop the CSV being closed properly.
      write_cmd("range auto 0\n");
      ::usleep(200000);
      ::close(fd_);
    }
    if (csv_) {
      *csv_ << "# stopped: " << n_rows_ << " readings, " << n_absent_ << " absent events\n";
      csv_->close();
    }
    RCLCPP_INFO(get_logger(), "range: %ld readings, %ld absent events", n_rows_, n_absent_);
  }

 private:
  bool configure_port()
  {
    termios tio{};
    if (::tcgetattr(fd_, &tio) != 0) return false;
    ::cfmakeraw(&tio);
    ::cfsetispeed(&tio, B115200);
    ::cfsetospeed(&tio, B115200);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;
    return ::tcsetattr(fd_, TCSANOW, &tio) == 0;
  }

  void write_cmd(const std::string& s)
  {
    if (fd_ < 0) return;
    ssize_t n = ::write(fd_, s.data(), s.size());
    (void)n;
  }

  void drain(int ms)
  {
    char buf[512];
    for (int i = 0; i < ms / 10; i++) {
      while (::read(fd_, buf, sizeof(buf)) > 0) {}
      ::usleep(10000);
    }
  }

  void open_csv()
  {
    if (csv_path_.empty()) return;
    csv_ = std::make_unique<std::ofstream>(csv_path_);
    if (!csv_->is_open()) {
      RCLCPP_ERROR(get_logger(), "cannot open %s — not logging", csv_path_.c_str());
      csv_.reset();
      return;
    }
    *csv_ << "# Garmin LIDAR-Lite v3 on the camtrig MCU (I2C1, PB6/PB7)\n"
          << "# port=" << port_ << " divisor=" << divisor_
          << " (one reading every " << divisor_ << " trigger pulses)\n"
          << "# pulses = the trigger edge counter; JOIN ON THIS - the capture side's seq\n"
          << "#   advances one per edge, so range->frame is an exact integer match\n"
          << "# t_mono_ns = CLOCK_MONOTONIC when this node READ the line. Same clock as\n"
          << "#   imu0.csv and the camera CSVs, but NOT the measurement instant: it\n"
          << "#   includes acquisition (5-20 ms) + UART transit (~2 ms) + wake-up.\n"
          << "#   Ordering information only. delta_range_camera = UNMEASURED\n"
          << "#t_mono_ns,range_cm,pulses\n";
    csv_->flush();
  }

  void poll()
  {
    if (fd_ < 0) return;

    char chunk[512];
    ssize_t n;
    while ((n = ::read(fd_, chunk, sizeof(chunk))) > 0) {
      const int64_t t = mono_ns();
      buf_.append(chunk, static_cast<size_t>(n));

      size_t nl;
      while ((nl = buf_.find('\n')) != std::string::npos) {
        std::string line = buf_.substr(0, nl);
        buf_.erase(0, nl + 1);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        handle_line(line, t);
      }
      // A peer that stops reading must not let this grow without bound.
      if (buf_.size() > 4096) buf_.clear();
    }
  }

  void handle_line(const std::string& line, int64_t t)
  {
    // Only unsolicited lines are ours. Anything else is a reply to a command this node did
    // not send, and must not be parsed as data.
    if (line.empty() || line[0] != '!') return;
    const std::string body = line.substr(1);

    if (body.rfind("range_cm=", 0) != 0) {
      if (body.find("absent") != std::string::npos) {
        n_absent_++;
        RCLCPP_WARN(get_logger(), "rangefinder absent — continuing without it");
      } else if (body.find("present again") != std::string::npos) {
        RCLCPP_INFO(get_logger(), "rangefinder present again");
      }
      if (csv_) { *csv_ << "# " << body << " t_mono_ns=" << t << "\n"; csv_->flush(); }
      return;
    }

    long cm = 0, pulses = 0;
    if (std::sscanf(body.c_str(), "range_cm=%ld pulses=%ld", &cm, &pulses) != 2) {
      if (csv_) *csv_ << "# unparsed: " << body << "\n";
      return;
    }

    if (csv_) *csv_ << t << ',' << cm << ',' << pulses << '\n';
    n_rows_++;

    sensor_msgs::msg::Range msg;
    msg.header.stamp = rclcpp::Time(t);       // CLOCK_MONOTONIC, as bev_imu and the cameras
    msg.header.frame_id = frame_id_;
    msg.radiation_type = sensor_msgs::msg::Range::INFRARED;
    msg.field_of_view = kFovRad;
    msg.min_range = kMinRangeM;
    msg.max_range = kMaxRangeM;
    msg.range = static_cast<float>(cm) / 100.0f;

    // A v3 reports 0 when the target is closer than it can resolve. Publishing that as
    // 0 m would be indistinguishable from a real measurement, so use the REP-117
    // convention instead: -Inf means "closer than min_range".
    if (msg.range < kMinRangeM) msg.range = -std::numeric_limits<float>::infinity();
    else if (msg.range > kMaxRangeM) msg.range = std::numeric_limits<float>::infinity();

    pub_->publish(msg);
  }

  std::string port_, csv_path_, frame_id_, buf_;
  int divisor_ = 15;
  int fd_ = -1;
  long n_rows_ = 0, n_absent_ = 0;
  std::unique_ptr<std::ofstream> csv_;
  rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RangeNode>());
  rclcpp::shutdown();
  return 0;
}
