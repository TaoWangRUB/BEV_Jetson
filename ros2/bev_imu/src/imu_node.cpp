// MPU-9250 IMU node — samples stamped at the DATA-READY EDGE, on CLOCK_MONOTONIC.
//
// Where the timestamp comes from is the whole point. A reader that polls SPI in a loop
// and stamps each sample when the read returns dates it to "whenever userspace got round
// to it" — scheduler latency and all. That error is invisible in the data and fatal to
// visual-inertial fusion. Here the IMU's own data-ready INT line drives the timing: the
// node blocks on that edge, stamps it, and only then fetches numbers whose time is
// already known.
//
// TIMESTAMPS (the contract — README 4.7): header.stamp is CLOCK_MONOTONIC, matching
// argus_capture_node. It is NOT ROS system time; never compare it against now().
//
// Three facts about this board shape the code:
//   * The MPU's INT is on gpio-298 (GPIO9_MOTION_INT) and the J106 INVERTS it, with no
//     pull-up. So the sensor is configured push-pull ("totem pole") rather than the
//     default open-drain, and the Tegra sees the sensor's assertion as a FALLING edge.
//   * The GPIO chardev stamps its own events with CLOCK_REALTIME (gpiolib.c:
//     ktime_get_real_ns) while the cameras are on CLOCK_MONOTONIC. Mixing them misdates
//     everything by the REALTIME-MONOTONIC offset, and NTP can slew one under the other.
//     So the chardev is used only to WAIT; the timestamp is taken here.
//   * MPU-9250 FSYNC is not brought out by the J106 and Tegra GTE hardware timestamping
//     is Xavier-only, so waking userspace on the edge is genuinely the best this board
//     allows: ~50 us median wake latency, MAD 2.8 us under SCHED_FIFO. The median is
//     bias, absorbed by the camera<->IMU offset; the MAD is the real limit.
//
// The DLPF group delay is REPORTED, not applied: the edge marks when the FILTERED sample
// was ready, and the gyro path lags the accel path by ~1.0 ms at every matched bandwidth,
// so one correction cannot serve both.
//
// Needs root: /dev/spidev1.0 and /dev/gpiochip* are root-only, the device cgroup must
// allow them (compose runs this service privileged), and SCHED_FIFO needs CAP_SYS_NICE.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <linux/gpio.h>
#include <linux/spi/spidev.h>
#include <poll.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace {

// ---- MPU-9250 registers ---------------------------------------------------
constexpr uint8_t SMPLRT_DIV = 0x19, CONFIG = 0x1A, GYRO_CONFIG = 0x1B;
constexpr uint8_t ACCEL_CONFIG = 0x1C, ACCEL_CONFIG2 = 0x1D;
constexpr uint8_t INT_PIN_CFG = 0x37, INT_ENABLE = 0x38, ACCEL_XOUT_H = 0x3B;
constexpr uint8_t USER_CTRL = 0x6A, PWR_MGMT_1 = 0x6B, PWR_MGMT_2 = 0x6C, WHO_AM_I = 0x75;
constexpr uint8_t WHO_AM_I_EXPECTED = 0x71;

constexpr double G_MS2 = 9.80665;
constexpr double DEG2RAD = M_PI / 180.0;

// Datasheet group delays (ms) for DLPF_CFG / A_DLPF_CFG 0..4, with their bandwidths.
// The gyro path lags the accel path by ~1.0 ms at every matched setting.
struct Dlpf { double bw_hz, delay_ms; };
constexpr Dlpf GYRO_DLPF[5]  = {{250.0, 0.97}, {184.0, 2.9}, {92.0, 3.9}, {41.0, 5.9}, {20.0, 9.9}};
constexpr Dlpf ACCEL_DLPF[5] = {{218.1, 1.88}, {218.1, 1.88}, {99.0, 2.88}, {44.8, 4.88}, {21.2, 6.88}};

int gyro_fs_sel(int dps) {
  switch (dps) { case 250: return 0; case 500: return 1; case 1000: return 2; case 2000: return 3; }
  throw std::runtime_error("gyro_fs must be 250, 500, 1000 or 2000 dps");
}
double gyro_lsb(int dps) {
  switch (dps) { case 250: return 131.0; case 500: return 65.5; case 1000: return 32.8; default: return 16.4; }
}
int accel_fs_sel(int g) {
  switch (g) { case 2: return 0; case 4: return 1; case 8: return 2; case 16: return 3; }
  throw std::runtime_error("accel_fs must be 2, 4, 8 or 16 g");
}
double accel_lsb(int g) {
  switch (g) { case 2: return 16384.0; case 4: return 8192.0; case 8: return 4096.0; default: return 2048.0; }
}

int64_t mono_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

// ---- spidev ---------------------------------------------------------------
class Spi {
 public:
  Spi(const std::string& dev, uint32_t speed_hz) : speed_(speed_hz) {
    fd_ = ::open(dev.c_str(), O_RDWR);
    if (fd_ < 0) throw std::runtime_error("open " + dev + ": " + std::strerror(errno));
    uint8_t mode = 0, bits = 8;
    if (ioctl(fd_, SPI_IOC_WR_MODE, &mode) < 0 ||
        ioctl(fd_, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ioctl(fd_, SPI_IOC_WR_MAX_SPEED_HZ, &speed_) < 0)
      throw std::runtime_error(std::string("spidev setup: ") + std::strerror(errno));
  }
  ~Spi() { if (fd_ >= 0) ::close(fd_); }

  void xfer(const uint8_t* tx, uint8_t* rx, size_t n, uint32_t speed_hz) {
    spi_ioc_transfer tr{};
    tr.tx_buf = reinterpret_cast<uint64_t>(tx);
    tr.rx_buf = reinterpret_cast<uint64_t>(rx);
    tr.len = static_cast<uint32_t>(n);
    tr.speed_hz = speed_hz ? speed_hz : speed_;
    tr.bits_per_word = 8;
    if (ioctl(fd_, SPI_IOC_MESSAGE(1), &tr) < 0)
      throw std::runtime_error(std::string("SPI transfer: ") + std::strerror(errno));
  }

 private:
  int fd_ = -1;
  uint32_t speed_;
};

// ---- GPIO character device (v1 ABI — the only one on this 4.9 kernel) ------
struct Line { std::string chip; uint32_t offset; std::string label; };

// Map a sysfs GPIO number (e.g. 298) to (/dev/gpiochipN, offset). Deliberately not
// hard-coded: the chardev index and the sysfs base are assigned in probe order and are
// not guaranteed across kernels. Match on the chip's label and line count instead, which
// are properties of the hardware.
Line resolve_line(int global_gpio) {
  DIR* d = opendir("/sys/class/gpio");
  if (!d) throw std::runtime_error("cannot list /sys/class/gpio");
  std::string want_label;
  uint32_t want_lines = 0;
  int base = 0;
  while (dirent* e = readdir(d)) {
    if (std::strncmp(e->d_name, "gpiochip", 8) != 0) continue;
    const std::string dir = std::string("/sys/class/gpio/") + e->d_name;
    int b = 0, n = 0;
    std::string lbl;
    { std::ifstream f(dir + "/base"); if (!(f >> b)) continue; }
    { std::ifstream f(dir + "/ngpio"); if (!(f >> n)) continue; }
    { std::ifstream f(dir + "/label"); if (!(f >> lbl)) continue; }
    if (global_gpio >= b && global_gpio < b + n) { want_label = lbl; want_lines = n; base = b; break; }
  }
  closedir(d);
  if (want_label.empty()) throw std::runtime_error("no gpiochip owns gpio-" + std::to_string(global_gpio));

  for (int i = 0; i < 16; ++i) {
    const std::string path = "/dev/gpiochip" + std::to_string(i);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) continue;
    gpiochip_info info{};
    const bool ok = ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, &info) >= 0;
    ::close(fd);
    if (ok && want_label == info.label && want_lines == info.lines)
      return {path, static_cast<uint32_t>(global_gpio - base), want_label};
  }
  throw std::runtime_error("gpio-" + std::to_string(global_gpio) + " is on sysfs chip '" +
                           want_label + "' but no /dev/gpiochip* reports that label "
                           "(permission denied? this needs root)");
}

class EdgeSource {
 public:
  EdgeSource(int global_gpio, bool falling) {
    line_ = resolve_line(global_gpio);
    int chip_fd = ::open(line_.chip.c_str(), O_RDONLY);
    if (chip_fd < 0) throw std::runtime_error("open " + line_.chip + ": " + std::strerror(errno));
    gpioevent_request req{};
    req.lineoffset = line_.offset;
    req.handleflags = GPIOHANDLE_REQUEST_INPUT;
    req.eventflags = falling ? GPIOEVENT_REQUEST_FALLING_EDGE : GPIOEVENT_REQUEST_RISING_EDGE;
    std::snprintf(req.consumer_label, sizeof(req.consumer_label), "bev_imu");
    const int rc = ioctl(chip_fd, GPIO_GET_LINEEVENT_IOCTL, &req);
    ::close(chip_fd);
    if (rc < 0) {
      if (errno == EBUSY)
        throw std::runtime_error("gpio-" + std::to_string(global_gpio) + " is already claimed "
                                 "by another consumer — check /sys/kernel/debug/gpio");
      throw std::runtime_error(std::string("GPIO_GET_LINEEVENT: ") + std::strerror(errno));
    }
    fd_ = req.fd;
  }
  ~EdgeSource() { if (fd_ >= 0) ::close(fd_); }

  // Block for an edge. Returns how many were queued: >1 means the kernel had edges
  // waiting before we ran, and since the data registers only ever hold the newest
  // sample, every event beyond the last one is a sample lost.
  int wait(int timeout_ms, int64_t* stamp_ns) {
    pollfd p{fd_, POLLIN, 0};
    const int rc = ::poll(&p, 1, timeout_ms);
    if (rc <= 0) return 0;
    // Stamp FIRST — before draining the queue, before the SPI burst. Everything after
    // this point only fetches numbers whose time is already fixed.
    *stamp_ns = mono_ns();
    gpioevent_data ev[16];
    const ssize_t n = ::read(fd_, ev, sizeof(ev));
    if (n <= 0) return 0;
    return static_cast<int>(n / sizeof(gpioevent_data));
  }

  const Line& line() const { return line_; }

 private:
  int fd_ = -1;
  Line line_;
};

// ---- MPU-9250 -------------------------------------------------------------
class Mpu9250 {
 public:
  explicit Mpu9250(Spi& spi) : spi_(spi) {}

  uint8_t rd(uint8_t reg) {
    uint8_t tx[2] = {static_cast<uint8_t>(reg | 0x80), 0}, rx[2] = {0, 0};
    spi_.xfer(tx, rx, 2, kCfgSpeed);
    return rx[1];
  }

  void wr(uint8_t reg, uint8_t val, bool verify = true) {
    uint8_t tx[2] = {static_cast<uint8_t>(reg & 0x7F), val}, rx[2] = {0, 0};
    spi_.xfer(tx, rx, 2, kCfgSpeed);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (verify) {
      const uint8_t got = rd(reg);
      if (got != val) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "MPU reg 0x%02x: wrote 0x%02x, read back 0x%02x",
                      reg, val, got);
        throw std::runtime_error(buf);
      }
    }
  }

  uint8_t probe() {
    const uint8_t who = rd(WHO_AM_I);
    if (who != WHO_AM_I_EXPECTED) {
      char buf[192];
      std::snprintf(buf, sizeof(buf),
                    "WHO_AM_I = 0x%02x, expected 0x71 (MPU-9250). The J106 IMU is "
                    "Tegra186 SPI3 = /dev/spidev1.0; spidev0.0/2.0/3.0 are other controllers",
                    who);
      throw std::runtime_error(buf);
    }
    return who;
  }

  double configure(double rate_hz, int gyro_fs, int accel_fs, int gyro_dlpf, int accel_dlpf,
                   bool active_low) {
    wr(PWR_MGMT_1, 0x80, false);                       // device reset
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    wr(PWR_MGMT_1, 0x01);                              // auto-select best clock
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    wr(PWR_MGMT_2, 0x00);                              // all axes enabled
    wr(USER_CTRL, 0x10);                               // I2C_IF_DIS: SPI only
    wr(CONFIG, static_cast<uint8_t>(gyro_dlpf & 0x07));
    wr(GYRO_CONFIG, static_cast<uint8_t>(gyro_fs_sel(gyro_fs) << 3));
    wr(ACCEL_CONFIG, static_cast<uint8_t>(accel_fs_sel(accel_fs) << 3));
    wr(ACCEL_CONFIG2, static_cast<uint8_t>(accel_dlpf & 0x0F));   // accel_fchoice_b = 0
    const int div = static_cast<int>(std::lround(1000.0 / rate_hz)) - 1;
    if (div < 0 || div > 255)
      throw std::runtime_error("rate gives SMPLRT_DIV out of range; the DLPF base rate is "
                               "1 kHz, so pick 3.9..1000 Hz");
    wr(SMPLRT_DIV, static_cast<uint8_t>(div));
    // Push-pull, not the default open-drain: this line has no pull-up on the J106, so
    // open-drain would never present a usable high. Not latched (a ~50 us pulse per
    // sample), cleared by any read.
    wr(INT_PIN_CFG, static_cast<uint8_t>((active_low ? 0x80 : 0x00) | 0x10));
    wr(INT_ENABLE, 0x01);                              // RAW_RDY_EN
    accel_scale_ = G_MS2 / accel_lsb(accel_fs);
    gyro_scale_ = DEG2RAD / gyro_lsb(gyro_fs);
    return 1000.0 / (div + 1);
  }

  struct Sample { double ax, ay, az, gx, gy, gz, temp_c; bool clipped; };

  // accel(6) temp(2) gyro(6) in ONE transfer, so the six axes are one consistent sample.
  Sample read_burst(uint32_t speed_hz) {
    uint8_t tx[15] = {static_cast<uint8_t>(ACCEL_XOUT_H | 0x80)}, rx[15] = {0};
    spi_.xfer(tx, rx, sizeof(tx), speed_hz);
    auto be16 = [&](int i) { return static_cast<int16_t>((rx[1 + i * 2] << 8) | rx[2 + i * 2]); };
    const int16_t ax = be16(0), ay = be16(1), az = be16(2), t = be16(3),
                  gx = be16(4), gy = be16(5), gz = be16(6);
    const bool clipped = std::abs(ax) >= 32760 || std::abs(ay) >= 32760 || std::abs(az) >= 32760 ||
                         std::abs(gx) >= 32760 || std::abs(gy) >= 32760 || std::abs(gz) >= 32760;
    return {ax * accel_scale_, ay * accel_scale_, az * accel_scale_,
            gx * gyro_scale_, gy * gyro_scale_, gz * gyro_scale_,
            t / 333.87 + 21.0, clipped};
  }

  void standby() {
    try { wr(INT_ENABLE, 0x00, false); wr(PWR_MGMT_1, 0x40, false); } catch (...) {}
  }

 private:
  static constexpr uint32_t kCfgSpeed = 1000000;   // register access: MPU-9250 SPI max
  Spi& spi_;
  double accel_scale_ = G_MS2 / 8192.0, gyro_scale_ = DEG2RAD / 65.5;
};

}  // namespace

class ImuNode : public rclcpp::Node {
 public:
  ImuNode() : Node("bev_imu") {
    spidev_ = declare_parameter<std::string>("spidev", "/dev/spidev1.0");
    burst_speed_ = declare_parameter<int>("spi_speed", 1000000);
    gpio_ = declare_parameter<int>("gpio", 298);            // GPIO9_MOTION_INT
    rate_hz_ = declare_parameter<double>("rate", 200.0);
    gyro_fs_ = declare_parameter<int>("gyro_fs", 500);
    accel_fs_ = declare_parameter<int>("accel_fs", 4);
    gyro_dlpf_ = declare_parameter<int>("gyro_dlpf", 1);
    accel_dlpf_ = declare_parameter<int>("accel_dlpf", 1);
    // The sensor asserts INT high by default and the J106 inverts the line, so the
    // assertion reaches the Tegra as a falling edge. active_low flips both ends.
    active_low_ = declare_parameter<bool>("active_low", false);
    frame_id_ = declare_parameter<std::string>("frame_id", "imu0");
    csv_path_ = declare_parameter<std::string>("csv", "");
    // SCHED_FIFO halves the wake-latency tail (p99 122.7 -> 75.4 us) and barely moves the
    // median. 0 disables.
    rt_priority_ = declare_parameter<int>("rt_priority", 80);

    // RELIABLE, deep queue — not the usual SensorDataQoS. IMU samples are 60 bytes at
    // 200 Hz, so reliability is nearly free, and a recorder that misses samples is worse
    // than one that lags: measured best-effort into `ros2 bag record`, 927 samples
    // arrived out of 2140 published (87 Hz of 200). Kalibr solves the camera-IMU offset
    // from IMU motion, so dropped samples are dropped observability — and unevenly
    // dropped ones bias it rather than just weakening it.
    pub_ = create_publisher<sensor_msgs::msg::Imu>(
        declare_parameter<std::string>("topic", "/imu0"),
        rclcpp::QoS(rclcpp::KeepLast(2000)).reliable());
  }

  void run() {
    Spi spi(spidev_, static_cast<uint32_t>(burst_speed_));
    Mpu9250 mpu(spi);
    const uint8_t who = mpu.probe();
    const double actual_rate = mpu.configure(rate_hz_, gyro_fs_, accel_fs_, gyro_dlpf_,
                                             accel_dlpf_, active_low_);
    EdgeSource src(gpio_, /*falling=*/!active_low_);
    log_provenance(who, actual_rate, src);
    std::unique_ptr<std::ofstream> csv = open_csv(actual_rate, src);
    set_realtime();

    sensor_msgs::msg::Imu msg;
    msg.header.frame_id = frame_id_;
    msg.orientation_covariance[0] = -1.0;      // this IMU reports no orientation
    int64_t published = 0, dropped = 0, late = 0, clipped = 0;

    while (rclcpp::ok()) {
      int64_t t_ns = 0;
      const int n_ev = src.wait(1000, &t_ns);
      if (n_ev == 0) {
        RCLCPP_WARN(get_logger(), "no data-ready edge for 1 s — is the IMU still configured?");
        continue;
      }
      if (n_ev > 1) { ++late; dropped += n_ev - 1; }

      const auto s = mpu.read_burst(static_cast<uint32_t>(burst_speed_));
      if (s.clipped) ++clipped;

      msg.header.stamp = rclcpp::Time(t_ns);
      msg.linear_acceleration.x = s.ax;
      msg.linear_acceleration.y = s.ay;
      msg.linear_acceleration.z = s.az;
      msg.angular_velocity.x = s.gx;
      msg.angular_velocity.y = s.gy;
      msg.angular_velocity.z = s.gz;
      pub_->publish(msg);

      if (csv)
        *csv << t_ns << ',' << s.ax << ',' << s.ay << ',' << s.az << ','
             << s.gx << ',' << s.gy << ',' << s.gz << ',' << s.temp_c << ','
             << published << '\n';

      if (++published % static_cast<int64_t>(actual_rate * 10) == 0)
        RCLCPP_INFO(get_logger(), "%ld samples, %ld dropped (edge seen late), %ld late reads, "
                    "%ld clipped", published, dropped, late, clipped);
    }

    if (csv) csv->close();
    mpu.standby();
    RCLCPP_INFO(get_logger(), "stopped: %ld samples, %ld dropped", published, dropped);
  }

 private:
  void set_realtime() {
    if (rt_priority_ <= 0) return;
    sched_param sp{};
    sp.sched_priority = rt_priority_;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0)
      RCLCPP_INFO(get_logger(), "sample loop running SCHED_FIFO at priority %d", rt_priority_);
    else
      RCLCPP_WARN(get_logger(), "cannot set SCHED_FIFO (needs CAP_SYS_NICE) — the wake-latency "
                  "tail will be ~2x worse; the median, which Delta absorbs, is unaffected");
  }

  // Say what was configured: an IMU stream whose filter settings are unknown cannot be
  // calibrated against later.
  void log_provenance(uint8_t who, double rate, const EdgeSource& src) {
    RCLCPP_INFO(get_logger(), "MPU-9250 WHO_AM_I=0x%02x on %s, INT gpio-%d (%s edge, %s offset %u), "
                "CLOCK_MONOTONIC", who, spidev_.c_str(), gpio_, active_low_ ? "rising" : "falling",
                src.line().chip.c_str(), src.line().offset);
    RCLCPP_INFO(get_logger(), "rate %.2f Hz, gyro +/-%d dps DLPF %.0f Hz (group delay %.2f ms), "
                "accel +/-%d g DLPF %.0f Hz (group delay %.2f ms)",
                rate, gyro_fs_, GYRO_DLPF[gyro_dlpf_].bw_hz, GYRO_DLPF[gyro_dlpf_].delay_ms,
                accel_fs_, ACCEL_DLPF[accel_dlpf_].bw_hz, ACCEL_DLPF[accel_dlpf_].delay_ms);
    RCLCPP_WARN(get_logger(), "group delays are REPORTED, not applied: the gyro lags the accel by "
                "%.2f ms. camera<->IMU offset Delta is UNMEASURED (see README 4.7)",
                GYRO_DLPF[gyro_dlpf_].delay_ms - ACCEL_DLPF[accel_dlpf_].delay_ms);
  }

  std::unique_ptr<std::ofstream> open_csv(double rate, const EdgeSource& src) {
    if (csv_path_.empty()) return nullptr;
    auto f = std::make_unique<std::ofstream>(csv_path_);
    if (!f->is_open()) {
      RCLCPP_ERROR(get_logger(), "cannot open %s — not logging", csv_path_.c_str());
      return nullptr;
    }
    *f << "# MPU-9250, timestamped at the data-ready edge on CLOCK_MONOTONIC\n"
       << "# spidev=" << spidev_ << " int_gpio=" << gpio_ << " chip=" << src.line().chip
       << " offset=" << src.line().offset << " edge=" << (active_low_ ? "rising" : "falling")
       << " rate_hz=" << rate << "\n"
       << "# gyro_fs_dps=" << gyro_fs_ << " gyro_dlpf_bw_hz=" << GYRO_DLPF[gyro_dlpf_].bw_hz
       << " gyro_group_delay_ms=" << GYRO_DLPF[gyro_dlpf_].delay_ms << "\n"
       << "# accel_fs_g=" << accel_fs_ << " accel_dlpf_bw_hz=" << ACCEL_DLPF[accel_dlpf_].bw_hz
       << " accel_group_delay_ms=" << ACCEL_DLPF[accel_dlpf_].delay_ms << "\n"
       << "# group delays are NOT applied; delta_camera_imu = UNMEASURED\n"
       << "#timestamp [ns],a_x [m s^-2],a_y,a_z,w_x [rad s^-1],w_y,w_z,temp [C],seq\n";
    f->precision(9);
    *f << std::fixed;
    return f;
  }

  std::string spidev_, frame_id_, csv_path_;
  int burst_speed_, gpio_, gyro_fs_, accel_fs_, gyro_dlpf_, accel_dlpf_, rt_priority_;
  double rate_hz_;
  bool active_low_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImuNode>();
  try {
    node->run();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
