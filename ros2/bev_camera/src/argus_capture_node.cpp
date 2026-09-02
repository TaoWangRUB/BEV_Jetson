// 4-camera Argus capture node for ROS 2 Foxy (TX2/J106 BEV rig).
//
// Uses the libargus C++ API directly (the nvidia runtime mounts libnvargus into
// the container), so it needs no tegra gstreamer plugin. Opens the cameras on the
// requested carrier ports at a chosen sensor mode, acquires frames, extracts each
// frame's luma (Y) plane — which is exactly the grayscale image cuVSLAM wants —
// and publishes it as sensor_msgs/Image (mono8) on /camN/image_raw.
//
// TIMESTAMPS (the contract — see README 4.7): header.stamp is that frame's own
// EXPOSURE MIDPOINT on CLOCK_MONOTONIC, derived from the Argus sensor (SOF) timestamp
// minus half the exposure. It is NOT ROS system time, so it must never be compared
// against now(); the IMU has to be stamped on the same clock for the two to be fused.
//
// The rig is 4x IMX296 global-shutter on ports C-F, driven by an external hardware
// trigger, so every camera exposes on the same edge (measured skew 1 us). Two things
// follow, and both are load-bearing:
//   - cameras are addressed by PORT, resolved to an Argus sensor-id at runtime
//     (Argus numbers in /dev/video bind order, which is not port order);
//   - in trigger mode the exposure IS the trigger pulse width, so AE is locked.
// IMX219 modules on the same ports still work — the port table matches either family.
//
// Build/run inside cuvslam-foxy:tx2 with the Argus socket + /dev mounted, and the
// jetson_multimedia_api headers bind-mounted for the include path.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <yaml-cpp/yaml.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <bev_camera/msg/frame_meta.hpp>

#include <Argus/Argus.h>
#include <EGLStream/ArgusCaptureMetadata.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/FrameConsumer.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <nvbuf_utils.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

using namespace Argus;

namespace {

// Physical carrier port -> the i2c name the sensor on it reports. IMX296 sits at
// 0x1a/0x18 where IMX219 sits at 0x10/0x12, so the same port reports a different
// name depending on which module is fitted; match either family.
struct PortEntry { const char* port; const char* imx296; const char* imx219; };
constexpr PortEntry kPortTable[] = {
    {"a", nullptr,  "1-0010"},
    {"b", nullptr,  "1-0012"},
    {"c", "2-001a", "2-0010"},
    {"d", "2-0018", "2-0012"},
    {"e", "7-001a", "7-0010"},
    {"f", "7-0018", "7-0012"},
};

struct CameraNode { int sensor_id; std::string family; std::string i2c; };

// Resolve port -> Argus sensor-id from sysfs. Argus assigns sensor-ids in /dev/video
// bind order, which is NOT port order and is not stable across boots: binding port F
// before E was observed live to give video4=7-0012 (port f) and video5=7-0010 (port e),
// which shifts every hard-coded index by one and silently mislabels the whole rig —
// extrinsics then belong to the wrong images. So never hard-code it; read it.
std::map<std::string, CameraNode> scan_video_nodes() {
  std::vector<int> nodes;
  if (DIR* d = opendir("/sys/class/video4linux")) {
    while (dirent* e = readdir(d)) {
      int n = -1;
      if (std::sscanf(e->d_name, "video%d", &n) == 1) nodes.push_back(n);
    }
    closedir(d);
  }
  std::sort(nodes.begin(), nodes.end());  // numeric order == Argus sensor-id order

  std::map<std::string, CameraNode> out;
  for (size_t sid = 0; sid < nodes.size(); ++sid) {
    std::ifstream f("/sys/class/video4linux/video" + std::to_string(nodes[sid]) + "/name");
    std::string name;  // e.g. "vi-output, imx296 2-001a"
    if (!std::getline(f, name)) continue;
    const auto sp = name.rfind(' ');
    if (sp == std::string::npos) continue;
    const std::string i2c = name.substr(sp + 1);
    for (const auto& p : kPortTable) {
      if (p.imx296 && i2c == p.imx296) out[p.port] = {static_cast<int>(sid), "imx296", i2c};
      else if (p.imx219 && i2c == p.imx219) out[p.port] = {static_cast<int>(sid), "imx219", i2c};
    }
  }
  return out;
}

// The IMX296 driver exposes Fast Trigger mode as a module parameter, not a control.
bool external_trigger_active() {
  std::ifstream f("/sys/module/imx296/parameters/trigger_mode");
  int v = 0;
  return static_cast<bool>(f >> v) && v == 1;
}

// OpenCV writes "%YAML:1.0" (no space), which is not a valid YAML directive — strip it
// and the document marker, same as the VO nodes do.
YAML::Node load_opencv_yaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("%YAML", 0) == 0 || line == "---") continue;
    ss << line << "\n";
  }
  return YAML::Load(ss.str());
}

}  // namespace

class ArgusCaptureNode : public rclcpp::Node {
 public:
  ArgusCaptureNode() : Node("argus_capture") {
    ports_ = declare_parameter<std::vector<std::string>>("ports", {"c", "d", "e", "f"});
    topics_ = declare_parameter<std::vector<std::string>>(
        "topics", {"/cam1/image_raw", "/cam2/image_raw", "/cam3/image_raw", "/cam4/image_raw"});
    frame_ids_ = declare_parameter<std::vector<std::string>>(
        "frame_ids", {"cam1", "cam2", "cam3", "cam4"});
    width_ = declare_parameter<int>("width", 1456);    // IMX296 native
    height_ = declare_parameter<int>("height", 1088);
    fps_ = declare_parameter<int>("fps", 30);          // set by the trigger when triggered
    // A set whose frames span more than this is not a set. cuVSLAM's Multicamera gate
    // is 1 ms; the triggered rig measures 1 us, so anything near the limit is a fault.
    max_skew_us_ = declare_parameter<int>("max_skew_us", 1000);
    // Empty (default) = resolve from sysfs. Set it only to override a resolution you
    // have already checked by hand — it bypasses the port mapping entirely.
    auto forced = declare_parameter<std::vector<int64_t>>("sensor_ids", std::vector<int64_t>{});
    // "auto" locks AE only when the driver is actually in external-trigger mode.
    ae_lock_mode_ = declare_parameter<std::string>("ae_lock", "auto");
    ae_gain_ = declare_parameter<std::vector<double>>("ae_gain", {16.0, 16.0});
    ae_dgain_ = declare_parameter<std::vector<double>>("ae_dgain", {4.0, 4.0});
    // Publish only every Nth frame. For a CALIBRATION recording: the solvers want ~4 Hz
    // images, while four cameras at 30 Hz of 1456x1088 mono is ~190 MB/s, which the SD
    // cannot absorb — and a bag that drops frames on its own drops them unevenly and
    // without saying so. Decimating here is explicit and keeps whole synchronised sets:
    // every camera skips the same trigger edges, so a published set is still a set.
    // Timing metadata is published for EVERY frame regardless, so the frame-time fit and
    // the drop accounting still see the full sequence.
    publish_every_n_ = declare_parameter<int>("publish_every_n", 1);
    if (publish_every_n_ < 1) throw std::runtime_error("publish_every_n must be >= 1");

    // Timestamp convention: exposure midpoint (what a VIO wants) vs raw SOF.
    stamp_midpoint_ = declare_parameter<bool>("stamp_exposure_midpoint", true);
    // 0 = use the exposure Argus reports. Set it to the trigger pulse width when the
    // driver is in trigger mode and the commanded exposure is being ignored.
    exposure_us_ = declare_parameter<int>("exposure_us", 0);
    // Directory for the per-camera frame-time CSVs. Empty = do not write them.
    frame_log_dir_ = declare_parameter<std::string>("frame_log_dir", "");
    // WRITE FRAMES STRAIGHT TO DISK, BYPASSING ROS ENTIRELY.
    //
    // For offline debugging the frames never need to leave this machine, and sending them
    // through DDS to reach a recorder on the SAME board costs most of them: measured
    // 2026-09-02, the capture side held a clean 30 Hz (150 sets per 5 s window, 66 us skew)
    // while rosbag2 wrote 6.1 Hz per camera - four fifths lost in serialisation and loopback
    // UDP, with the disk only a quarter busy. Tuning DDS buffers is treating the symptom;
    // the frames should not be going through a network stack at all.
    //
    // When image_log_dir is set the node appends raw mono8 frames to one file per camera and
    // an index CSV beside it, and does NOT publish. Sequential writes, no serialisation.
    //
    // COMMA-SEPARATED = ONE TARGET PER CAMERA, and that is what makes a full-rate minute
    // possible. 4 cameras at 30 fps is ~190 MB/s and NO single target here takes it: measured
    // sustained write is 136 MB/s on eMMC, 62.6 on the SD, and RAM holds only ~21 s. Split
    // them and each target stays inside its own limit - 2 cameras to eMMC (95 MB/s), 1 to the
    // SD (48), 1 to RAM (48) - which fits 60 s with room on all three.
    //   image_log_dir:="/logs,/logs,/media/...,/ramlog"   (per camera, in `cameras` order)
    //   image_log_dir:="/logs"                            (one path, all cameras)
    image_log_dir_ = declare_parameter<std::string>("image_log_dir", "");
    queue_depth_ = static_cast<size_t>(declare_parameter<int>("write_queue_depth", 64));
    // Optional: refuse to publish under a calibration measured on another rig.
    calib_dir_ = declare_parameter<std::string>("calib_dir", "");

    n_ = ports_.size();
    if (n_ == 0) throw std::runtime_error("no ports requested");
    if (topics_.size() < n_ || frame_ids_.size() < n_)
      throw std::runtime_error("topics/frame_ids shorter than ports");

    resolve_sensor_ids(forced);
    ts_history_.resize(n_);
    off_us_.assign(n_, 0);

    trigger_active_ = external_trigger_active();
    ae_lock_ = (ae_lock_mode_ == "1" || ae_lock_mode_ == "true" || ae_lock_mode_ == "on") ||
               (ae_lock_mode_ == "auto" && trigger_active_);

    if (ae_lock_)
      RCLCPP_INFO(get_logger(), "%s -> locking AE (gain %.1f-%.1f, dgain %.1f-%.1f)",
                  trigger_active_ ? "external trigger active" : "ae_lock requested",
                  ae_gain_[0], ae_gain_[1], ae_dgain_[0], ae_dgain_[1]);

    if (!calib_dir_.empty()) check_calibration();
    if (!frame_log_dir_.empty()) open_frame_logs();
    if (!image_log_dir_.empty()) open_image_logs();

    // Best-effort sensor-data QoS: high-rate camera streams must never let a slow
    // reliable subscriber back-pressure (and block) the Argus capture thread.
    for (size_t i = 0; i < n_; ++i) {
      pubs_.push_back(create_publisher<sensor_msgs::msg::Image>(topics_[i], rclcpp::SensorDataQoS()));
      // /camN/image_raw -> /camN/frame_meta. Timing metadata travels as its own message
      // rather than inside the image, so a bag keeps it even when images are throttled.
      std::string base = topics_[i];
      const auto slash = base.rfind('/');
      base = (slash == std::string::npos) ? base : base.substr(0, slash);
      meta_pubs_.push_back(create_publisher<bev_camera::msg::FrameMeta>(
          base + "/frame_meta", rclcpp::SensorDataQoS()));
    }

    if (!setup_argus())
      throw std::runtime_error("Argus setup failed");
    running_ = true;
    worker_ = std::thread([this] { capture_loop(); });
    std::string decim;
    if (publish_every_n_ > 1)
      decim = ", images decimated 1/" + std::to_string(publish_every_n_) + " (~" +
              std::to_string(fps_ / publish_every_n_) + " Hz); metadata stays full rate";
    RCLCPP_INFO(get_logger(), "Argus capture up: %zu cameras @ %dx%d, trigger %s, AE %s%s",
                n_, width_, height_, trigger_active_ ? "external" : "free-running",
                ae_lock_ ? "locked" : "auto", decim.c_str());
  }

  ~ArgusCaptureNode() override {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    // Drain the writers AFTER the capture loop has stopped, so whatever is queued reaches
    // disk. Dropping it here would silently shorten the log by up to 8 frames per camera and
    // leave the index disagreeing with the .raw - the exact failure this logger already had
    // once, from an unflushed index.
    if (!writers_.empty()) {
      { for (size_t i = 0; i < n_; ++i) { std::lock_guard<std::mutex> lk(wq_mtx_[i]); }
        wq_stop_ = true; }
      for (auto& cv : wq_cv_) cv.notify_all();
      for (auto& t : writers_) if (t.joinable()) t.join();
      uint64_t dropped = 0;
      for (auto d : wq_dropped_) dropped += d;
      if (dropped)
        RCLCPP_WARN(get_logger(), "%lu frames dropped at the writer queue - the target could "
                    "not keep up. The index records only what was written.",
                    static_cast<unsigned long>(dropped));
      for (size_t i = 0; i < n_; ++i)
        RCLCPP_INFO(get_logger(), "  %s: %lu frames written", frame_ids_[i].c_str(),
                    static_cast<unsigned long>(image_frames_[i]));
      for (auto& f : image_logs_) if (f) f->close();
      for (auto& f : image_index_) if (f) f->close();
    }
    for (auto& f : frame_logs_) if (f) f->close();
    for (auto fd : dmabufs_) if (fd != -1) NvBufferDestroy(fd);
    if (egl_display_ != EGL_NO_DISPLAY) eglTerminate(egl_display_);
  }

 private:
  // Map each requested port to an Argus sensor-id, and say out loud what was found —
  // a mislabelled rig is invisible in the images and fatal in the extrinsics.
  void resolve_sensor_ids(const std::vector<int64_t>& forced) {
    if (!forced.empty()) {
      if (forced.size() != n_) throw std::runtime_error("sensor_ids override length != ports length");
      sensor_ids_ = forced;
      families_.assign(n_, "override");
      RCLCPP_WARN(get_logger(), "sensor_ids overridden by parameter — port mapping NOT verified");
      return;
    }
    const auto found = scan_video_nodes();
    std::vector<std::string> missing;
    sensor_ids_.resize(n_);
    families_.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
      const auto it = found.find(ports_[i]);
      if (it == found.end()) { missing.push_back(ports_[i]); continue; }
      sensor_ids_[i] = it->second.sensor_id;
      families_[i] = it->second.family;
      RCLCPP_INFO(get_logger(), "  port %s (%s %s) -> sensor-id %d -> %s",
                  ports_[i].c_str(), it->second.family.c_str(), it->second.i2c.c_str(),
                  it->second.sensor_id, frame_ids_[i].c_str());
    }
    if (!missing.empty()) {
      std::string list;
      for (const auto& m : missing) list += (list.empty() ? "" : ", ") + m;
      throw std::runtime_error("no camera on port(s): " + list +
                               " — refusing to start a partially populated rig");
    }
  }

  // Applying a calibration measured on another sensor or another resolution is a slow,
  // silent drift rather than a crash, so make it a startup error instead.
  void check_calibration() {
    for (size_t i = 0; i < n_; ++i) {
      const std::string path = calib_dir_ + "/" + frame_ids_[i] + ".yaml";
      YAML::Node y = load_opencv_yaml(path);
      const int w = y["image_width"].as<int>(), h = y["image_height"].as<int>();
      if (w != width_ || h != height_)
        throw std::runtime_error(path + ": calibrated at " + std::to_string(w) + "x" +
                                 std::to_string(h) + " but capturing at " +
                                 std::to_string(width_) + "x" + std::to_string(height_));
      if (y["sensor"] && families_[i] != "override") {
        const auto s = y["sensor"].as<std::string>();
        if (s != families_[i])
          throw std::runtime_error(path + ": calibrated on " + s + " but port " + ports_[i] +
                                   " carries " + families_[i]);
      } else if (!y["sensor"]) {
        RCLCPP_WARN(get_logger(), "%s does not state which sensor it was measured on", path.c_str());
      }
    }
    RCLCPP_INFO(get_logger(), "calibration in %s matches the live rig", calib_dir_.c_str());
  }

  // Headless EGLDisplay: in a container there is no window system, so Argus's
  // fallback eglGetDisplay(EGL_DEFAULT_DISPLAY) fails. Get a display straight
  // from the GPU device via EGL_EXT_platform_device (no X needed) and hand it to
  // each Argus output stream so the EGLStream consumer uses it.
  bool init_egl() {
    auto query_devices =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (query_devices && get_platform_display) {
      EGLDeviceEXT devs[8];
      EGLint n = 0;
      if (query_devices(8, devs, &n) && n > 0) {
        for (EGLint d = 0; d < n; ++d) {
          EGLDisplay dpy = get_platform_display(EGL_PLATFORM_DEVICE_EXT, devs[d], nullptr);
          if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) {
            egl_display_ = dpy;
            RCLCPP_INFO(get_logger(), "EGL headless display via device %d of %d", d, n);
            return true;
          }
        }
      }
    }
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) {
      egl_display_ = dpy;
      RCLCPP_WARN(get_logger(), "EGL using default display (needs window system)");
      return true;
    }
    RCLCPP_ERROR(get_logger(), "no usable EGLDisplay");
    return false;
  }

  bool setup_argus() {
    if (!init_egl()) return false;
    provider_ = UniqueObj<CameraProvider>(CameraProvider::create());
    auto* ip = interface_cast<ICameraProvider>(provider_.get());
    if (!ip) { RCLCPP_ERROR(get_logger(), "no ICameraProvider"); return false; }
    std::vector<CameraDevice*> devs;
    ip->getCameraDevices(&devs);
    RCLCPP_INFO(get_logger(), "Argus %s, %zu cameras present",
                ip->getVersion().c_str(), devs.size());

    sessions_.resize(n_);
    streams_.resize(n_);
    requests_.resize(n_);
    consumers_.resize(n_);
    dmabufs_.assign(n_, -1);

    for (size_t i = 0; i < n_; ++i) {
      int id = sensor_ids_[i];
      if (id < 0 || id >= (int)devs.size()) { RCLCPP_ERROR(get_logger(), "sensor-id %d absent", id); return false; }
      sessions_[i].reset(ip->createCaptureSession(devs[id]));
      auto* isession = interface_cast<ICaptureSession>(sessions_[i].get());
      if (!isession) { RCLCPP_ERROR(get_logger(), "no session for %d", id); return false; }

      UniqueObj<OutputStreamSettings> ss(isession->createOutputStreamSettings(STREAM_TYPE_EGL));
      auto* iss = interface_cast<IEGLOutputStreamSettings>(ss.get());
      iss->setEGLDisplay(egl_display_);
      iss->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
      iss->setResolution(Size2D<uint32_t>(width_, height_));
      iss->setMetadataEnable(true);
      streams_[i].reset(isession->createOutputStream(ss.get()));
      consumers_[i].reset(EGLStream::FrameConsumer::create(streams_[i].get()));

      requests_[i].reset(isession->createRequest());
      auto* ireq = interface_cast<IRequest>(requests_[i].get());
      ireq->enableOutputStream(streams_[i].get());
      // pick the sensor mode matching width/height
      auto* iprops = interface_cast<ICameraProperties>(devs[id]);
      std::vector<SensorMode*> modes; iprops->getAllSensorModes(&modes);
      for (auto* m : modes) {
        auto* im = interface_cast<ISensorMode>(m);
        if ((int)im->getResolution().width() == width_ && (int)im->getResolution().height() == height_) {
          interface_cast<ISourceSettings>(requests_[i].get())->setSensorMode(m); break;
        }
      }
      auto* isrc = interface_cast<ISourceSettings>(requests_[i].get());
      isrc->setFrameDurationRange(Range<uint64_t>(1e9 / fps_));

      // Under external trigger the exposure IS the trigger pulse width, so AE cannot
      // move its main actuator (the driver logs "ignoring <n>") and hunts on gain
      // instead — a measured 3.5 Hz limit cycle swinging 150 luma levels peak-to-peak,
      // 171% of the mean. Clamping gain and locking AE removes it (p2p 150.5 -> 0.8)
      // at the same mean brightness. Free-running capture is left untouched.
      if (ae_lock_) {
        auto* ireq_ac = interface_cast<IRequest>(requests_[i].get());
        auto* iac = interface_cast<IAutoControlSettings>(ireq_ac->getAutoControlSettings());
        if (iac) {
          isrc->setGainRange(Range<float>(ae_gain_[0], ae_gain_[1]));
          iac->setIspDigitalGainRange(Range<float>(ae_dgain_[0], ae_dgain_[1]));
          iac->setAeLock(true);
        } else {
          RCLCPP_WARN(get_logger(), "cam idx %zu: no IAutoControlSettings — AE left free", i);
        }
      }
      isession->repeat(requests_[i].get());
    }
    return true;
  }

  // Per-camera frame-time CSV, in the shape j106-record-sync.py writes (and
  // j106-frametime.py fits): one row per frame, plus a header block stating what the
  // numbers mean. A recording without that provenance cannot be re-interpreted later —
  // which clock, which trigger rate, which exposure, and whether Delta was ever measured.
  // One .raw per camera (concatenated mono8 frames, no header) plus an index CSV giving the
  // exposure-midpoint stamp and byte offset of each. A sidecar records the geometry so an
  // offline reader needs nothing from this repo:  numpy.memmap(path, uint8).reshape(-1, h, w)
  // ONE WRITER THREAD PER CAMERA.
  //
  // The writes used to happen inline in the capture loop, which serialised them: an iteration
  // paid the SUM of all four write latencies, so every camera ran at the pace of the slowest
  // device. Measured 2026-09-02: 29.7 fps with all four on tmpfs, 20.95 fps on eMMC, and only
  // 22.2 fps when SPLIT across eMMC/SD/RAM - the split fixed the bandwidth problem and not the
  // latency one, because the loop still waited on each write in turn.
  //
  // Both are needed for 30 fps: the split because 190 MB/s only fits across eMMC+SD+RAM
  // combined, and these threads because otherwise the loop blocks.
  //
  // The queue is BOUNDED and drops the OLDEST when full rather than blocking. Back-pressure
  // into the Argus loop is what desynchronises the cameras - a stalled consumer previously
  // showed up as one camera a whole trigger period behind - and for a debug log a counted
  // drop is far better than a set that is silently no longer a set.
  struct WriteJob { std::vector<uint8_t> data; uint64_t stamp; };

  void writer_thread(size_t i) {
    for (;;) {
      WriteJob job;
      {
        std::unique_lock<std::mutex> lk(wq_mtx_[i]);
        wq_cv_[i].wait(lk, [&] { return !wq_[i].empty() || wq_stop_; });
        if (wq_[i].empty()) return;                 // stopping and drained
        job = std::move(wq_[i].front());
        wq_[i].pop_front();
      }
      auto& f = *image_logs_[i];
      auto& idx = *image_index_[i];
      const long long off = static_cast<long long>(f.tellp());
      f.write(reinterpret_cast<const char*>(job.data.data()),
              static_cast<std::streamsize>(job.data.size()));
      idx << job.stamp << "," << off << "\n" << std::flush;
      if (!f || !idx) {
        if (!image_log_failed_.exchange(true))
          RCLCPP_ERROR(get_logger(), "image log write failed for %s after %lu frames - target "
                       "full or too slow. What is already written is intact and indexed.",
                       frame_ids_[i].c_str(), static_cast<unsigned long>(image_frames_[i]));
        return;
      }
      ++image_frames_[i];
    }
  }

  void open_image_logs() {
    image_logs_.resize(n_); image_index_.resize(n_); image_frames_.assign(n_, 0);
    image_dirs_.clear();
    for (size_t start = 0; start <= image_log_dir_.size();) {
      const size_t comma = image_log_dir_.find(',', start);
      const size_t end = (comma == std::string::npos) ? image_log_dir_.size() : comma;
      image_dirs_.push_back(image_log_dir_.substr(start, end - start));
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    if (image_dirs_.size() == 1) image_dirs_.assign(n_, image_dirs_[0]);
    if (image_dirs_.size() != n_)
      throw std::runtime_error("image_log_dir: give one path, or exactly one per camera (" +
                               std::to_string(n_) + "); got " + std::to_string(image_dirs_.size()));
    for (size_t i = 0; i < n_; ++i) {
      const std::string base = image_dirs_[i] + "/" + frame_ids_[i];
      image_logs_[i] = std::make_unique<std::ofstream>(base + ".raw", std::ios::binary);
      image_index_[i] = std::make_unique<std::ofstream>(base + "_index.csv");
      if (!image_logs_[i]->good() || !image_index_[i]->good())
        throw std::runtime_error("cannot open image log for " + frame_ids_[i] + " in " + image_log_dir_);
      *image_index_[i] << "# stamp_ns is the exposure midpoint; offset is the byte offset of\n"
                       << "# this frame in " << frame_ids_[i] << ".raw\n"
                       << "stamp_ns,offset\n";
    }
    // geometry.txt into EVERY directory used, so each one is self-describing. The parts get
    // gathered from three different mounts before conversion, and a directory that cannot say
    // its own frame size is useless on its own.
    for (const auto& dir : std::set<std::string>(image_dirs_.begin(), image_dirs_.end())) {
      std::ofstream meta(dir + "/geometry.txt");
      meta << "width " << width_ << "\nheight " << height_ << "\nencoding mono8\n"
           << "bytes_per_frame " << static_cast<size_t>(width_) * height_ << "\n"
           << "cameras " << n_ << "\n";
    }
    for (size_t i = 0; i < n_; ++i)
      RCLCPP_INFO(get_logger(), "  %s -> %s (raw, no ROS publish)",
                  frame_ids_[i].c_str(), image_dirs_[i].c_str());
    wq_ = std::vector<std::deque<WriteJob>>(n_);
    wq_mtx_ = std::vector<std::mutex>(n_);
    wq_cv_ = std::vector<std::condition_variable>(n_);
    wq_dropped_.assign(n_, 0);
    for (size_t i = 0; i < n_; ++i) writers_.emplace_back([this, i] { writer_thread(i); });
  }

  void open_frame_logs() {
    frame_logs_.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
      const std::string path = frame_log_dir_ + "/" + frame_ids_[i] + ".csv";
      frame_logs_[i] = std::make_unique<std::ofstream>(path);
      if (!frame_logs_[i]->is_open()) {
        RCLCPP_ERROR(get_logger(), "cannot open %s — frame logging disabled", path.c_str());
        frame_logs_.clear();
        return;
      }
      *frame_logs_[i]
          << "# " << frame_ids_[i] << " frame times — CLOCK_MONOTONIC, timestamp = "
          << (stamp_midpoint_ ? "exposure midpoint (SOF - exposure/2)" : "SOF (start of readout)")
          << "\n"
          << "# port=" << ports_[i] << " sensor=" << families_[i]
          << " resolution=" << width_ << "x" << height_
          << " trigger=" << (trigger_active_ ? "external" : "FREE-RUNNING — UNSYNCHRONISED")
          << " rate_hz=" << fps_ << "\n"
          << "# exposure_source=" << (exposure_us_ > 0 ? "trigger pulse width (exposure_us param)"
                                                       : "as reported by Argus — NOT the pulse width")
          << "\n"
          << "# delta_camera_imu = UNMEASURED (see README 4.7)\n"
          << "#timestamp [ns],seq,capture_id,t_sof [ns],exposure [ns],image\n";
    }
    RCLCPP_INFO(get_logger(), "writing frame-time CSVs to %s", frame_log_dir_.c_str());
  }

  // The instant this frame corresponds to, on CLOCK_MONOTONIC.
  //
  // Two corrections, and both are needed:
  //
  // 1. Take the SENSOR timestamp, not IFrame::getTime(). getTime() is the EGLStream
  //    frame time and therefore consumer-side: measured live it put the four cameras
  //    ~7 ms apart in the order this loop happens to visit them (cam4 -7.0, cam1 0,
  //    cam2 +6.8, cam3 +13.8 ms) — it was reporting the loop's own phase. Note the
  //    "30-86 ms spread" this project previously recorded for the free-running IMX219
  //    rig was measured through those same timestamps and is therefore partly the same
  //    artifact; the V4L2-measured free-running skew was 2.43 ms with 8.33 us/s drift.
  //    Sensor timestamps require setMetadataEnable(true) on the stream.
  //
  // 2. Walk SOF back to the EXPOSURE MIDPOINT. SOF is "the time the first data from
  //    this capture arrives from the sensor" (Argus/CaptureMetadata.h) — i.e. the start
  //    of READOUT, which on a global shutter is when the exposure has already finished.
  //    The exposure therefore spans [SOF - exposure, SOF] and the instant the frame
  //    actually depicts is SOF - exposure/2. That is what a VIO wants and what the J106
  //    timing model states; leaving it out biases every camera by half an exposure
  //    against the IMU. It cancels between cameras (one shared trigger edge) but not
  //    against anything else, so it would end up hidden inside Delta and move whenever
  //    the exposure changed.
  //
  // Under the hardware trigger the true exposure IS the trigger pulse width, and Argus
  // reports the value it commanded, which the driver may be ignoring — so exposure_us
  // overrides the reported value when the pulse width is known.
  struct FrameTiming {
    uint64_t stamp_ns = 0;      // what the image and the metadata are stamped with
    uint64_t sof_ns = 0;        // raw kernel SOF, kept so the correction stays undoable
    uint64_t exposure_ns = 0;
    uint32_t capture_id = 0;    // session-side: what Argus produced
    uint64_t number = 0;        // consumer-side: what reached us
  };

  FrameTiming frame_timing(EGLStream::Frame* frame, EGLStream::IFrame* iframe) {
    FrameTiming ft;
    ft.number = iframe->getNumber();
    uint64_t sof = 0, exposure_ns = 0;
    if (auto* iacm = interface_cast<EGLStream::IArgusCaptureMetadata>(frame)) {
      if (auto* imeta = interface_cast<ICaptureMetadata>(iacm->getMetadata())) {
        sof = imeta->getSensorTimestamp();
        exposure_ns = imeta->getSensorExposureTime();
        ft.capture_id = imeta->getCaptureId();
      }
    }
    if (sof == 0) {
      if (!warned_no_metadata_) {
        RCLCPP_WARN(get_logger(), "no capture metadata — falling back to EGLStream frame time, "
                    "which is consumer-side and NOT comparable across cameras or with the IMU");
        warned_no_metadata_ = true;
      }
      ft.stamp_ns = iframe->getTime();
      return ft;
    }
    // ONE exposure for the whole rig, not one per camera. All four expose on the same
    // trigger edge for the same pulse width, so the exposure is identical by
    // construction, while Argus reports each camera's own AE state — subtracting a
    // per-camera half-exposure would put differences into the timestamps that the
    // hardware does not have. (Measured set spread wanders 1-17 us between runs either
    // way, so this is correctness by construction, not a measured improvement.)
    // Latch the first value seen (or the parameter) and use it for every camera.
    if (rig_exposure_ns_ == 0)
      rig_exposure_ns_ = exposure_us_ > 0 ? static_cast<uint64_t>(exposure_us_) * 1000 : exposure_ns;
    exposure_ns = rig_exposure_ns_;

    if (!logged_exposure_) {
      RCLCPP_INFO(get_logger(), "stamping at %s (rig exposure %.3f ms, %s)",
                  stamp_midpoint_ ? "exposure midpoint" : "SOF (start of readout)",
                  exposure_ns / 1e6,
                  exposure_us_ > 0 ? "from the exposure_us parameter" : "as reported by Argus");
      if (trigger_active_ && exposure_us_ == 0)
        RCLCPP_WARN(get_logger(), "under external trigger the true exposure is the trigger PULSE "
                    "WIDTH, which the driver may be ignoring the commanded value for — set "
                    "exposure_us to the pulse width once it is known");
      logged_exposure_ = true;
    }
    ft.sof_ns = sof;
    ft.exposure_ns = exposure_ns;
    ft.stamp_ns = stamp_midpoint_ ? sof - exposure_ns / 2 : sof;
    return ft;
  }

  // The image and its timing metadata carry the SAME stamp, so a consumer can join them
  // without guessing, and the CSV row is the same record in the form a frame-time fit
  // wants. Both sequence counters are recorded: capture_id is what the Argus session
  // produced, frame_number is what was delivered here — when they diverge, the gap says
  // where the frame was lost.
  void publish_meta(size_t i, const FrameTiming& ft, bool image_published) {
    bev_camera::msg::FrameMeta m;
    m.header.stamp = rclcpp::Time(static_cast<int64_t>(ft.stamp_ns));
    m.header.frame_id = frame_ids_[i];
    m.frame_number = ft.number;
    m.capture_id = ft.capture_id;
    m.sof_ns = ft.sof_ns;
    m.exposure_ns = ft.exposure_ns;
    m.image_published = image_published;
    meta_pubs_[i]->publish(m);

    // Flush per row: the node is normally stopped with a signal, and an unflushed tail
    // means a truncated last line — which a parser reports as corruption rather than as
    // the missing frames it looks like. 120 small writes/s costs nothing.
    if (i < frame_logs_.size() && frame_logs_[i])
      *frame_logs_[i] << ft.stamp_ns << ',' << ft.number << ',' << ft.capture_id << ','
                      << ft.sof_ns << ',' << ft.exposure_ns << ',' << (image_published ? 1 : 0)
                      << std::endl;
  }

  // Inter-camera skew, measured the only way that means anything: by matching frames
  // that came from the SAME trigger edge.
  //
  // One pass of the capture loop takes one frame from each camera, but each camera's
  // EGLStream queue advances on its own, so a sweep can hand back frame k from one
  // camera and frame k+1 from the next. Comparing them by loop position measures the
  // loop's phase, not the rig's sync — it reported ~35 ms (one frame period at 30 Hz)
  // on a rig whose V4L2-measured skew is 1.0 us. So keep a short history per camera and
  // match each of camera 0's frames to the NEAREST frame from every other camera. The
  // spread of a matched set is the real skew; a match is always found, so what says the
  // rig is broken is that spread exceeding max_skew_us, not a failure to match.
  void measure_set_skew(const std::vector<uint64_t>& latest) {
    for (size_t i = 0; i < n_; ++i) {
      auto& h = ts_history_[i];
      if (h.empty() || h.back() != latest[i]) h.push_back(latest[i]);
      while (h.size() > kHistory) h.pop_front();
    }
    const uint64_t t0 = latest[0];
    uint64_t lo = t0, hi = t0;
    for (size_t i = 1; i < n_; ++i) {
      uint64_t best = 0; int64_t best_d = INT64_MAX;
      for (uint64_t t : ts_history_[i]) {
        const int64_t d = std::llabs(static_cast<int64_t>(t) - static_cast<int64_t>(t0));
        if (d < best_d) { best_d = d; best = t; }
      }
      if (best_d == INT64_MAX) return;             // nothing to match against yet
      lo = std::min(lo, best);
      hi = std::max(hi, best);
      // Signed offset per camera: a constant one is a pipeline/phase offset, a wandering
      // one is a sync problem. The two need different fixes, so report them apart.
      off_us_[i] = (static_cast<int64_t>(best) - static_cast<int64_t>(t0)) / 1000;
    }
    const int64_t spread_us = static_cast<int64_t>(hi - lo) / 1000;
    ++sets_;
    if (spread_us > max_skew_us_) ++bad_sets_;
    if (spread_us > worst_skew_us_) worst_skew_us_ = spread_us;

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report_ >= std::chrono::seconds(5)) {
      std::string offs;
      for (size_t i = 1; i < n_; ++i)
        offs += " " + frame_ids_[i] + "=" + std::to_string(off_us_[i]) + "us";
      // Writer-queue state goes in the SAME line as the capture stats, because the two
      // losses look identical from outside and are not: frames the writer dropped never
      // reached disk, frames Argus never delivered never existed. Reporting only at
      // shutdown was useless - the run timer SIGKILLs before the destructor - so this was
      // being inferred from frame counts, which cannot tell the two apart.
      std::string wq;
      if (!wq_.empty()) {
        uint64_t dropped = 0; size_t deepest = 0;
        for (size_t i = 0; i < n_; ++i) {
          std::lock_guard<std::mutex> lk(wq_mtx_[i]);
          dropped += wq_dropped_[i];
          deepest = std::max(deepest, wq_[i].size());
        }
        wq = "; writer q max " + std::to_string(deepest) + "/" +
             std::to_string(queue_depth_) + ", dropped " + std::to_string(dropped);
      }
      RCLCPP_INFO(get_logger(), "sets %ld, worst skew %ld us in the last window "
                  "(limit %d us), over-limit sets %ld total; offsets vs %s:%s%s",
                  sets_, worst_skew_us_, max_skew_us_, bad_sets_,
                  frame_ids_[0].c_str(), offs.c_str(), wq.c_str());
      worst_skew_us_ = 0;  // worst per window, not since boot
      last_report_ = now;
    }
  }

  void capture_loop() {
    std::vector<EGLStream::IFrameConsumer*> ifc(n_);
    for (size_t i = 0; i < n_; ++i)
      ifc[i] = interface_cast<EGLStream::IFrameConsumer>(consumers_[i].get());

    std::vector<bool> first(n_, true);
    std::vector<int> timeouts(n_, 0);
    std::vector<uint64_t> set_ts(n_, 0);
    int empty_sweeps = 0;
    try {
      while (running_ && rclcpp::ok()) {
        size_t got = 0;
        for (size_t i = 0; i < n_; ++i) {
          UniqueObj<EGLStream::Frame> frame(ifc[i]->acquireFrame(1000000000));  // 1s timeout
          auto* iframe = interface_cast<EGLStream::IFrame>(frame.get());
          if (!iframe) {
            if (++timeouts[i] % 5 == 1)
              RCLCPP_WARN(get_logger(), "cam idx %zu: acquireFrame timeout (#%d)", i, timeouts[i]);
            continue;
          }
          auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(iframe->getImage());
          if (!inb) { RCLCPP_WARN(get_logger(), "cam idx %zu: no IImageNativeBuffer", i); continue; }
          if (dmabufs_[i] == -1) {
            dmabufs_[i] = inb->createNvBuffer(Size2D<uint32_t>(width_, height_),
                                              NvBufferColorFormat_YUV420, NvBufferLayout_Pitch);
          } else {
            int rc = inb->copyToNvBuffer(dmabufs_[i]);
            if (rc != 0 && timeouts[i]++ % 30 == 0)
              RCLCPP_WARN(get_logger(), "cam idx %zu: copyToNvBuffer rc=%d", i, rc);
          }
          const FrameTiming ft = frame_timing(frame.get(), iframe);
          set_ts[i] = ft.stamp_ns;
          ++got;
          // Decimate on the TRIGGER EDGE INDEX, derived from the frame's own SOF time -
          // never on the per-camera frame number. Each camera's Argus counter starts
          // when ITS session starts, so counters are offset between cameras: with
          // every_n=3, cam1 published edges 0,3,6... while cam3 published 2,5,8...,
          // and the two never published the same instant. Measured as a constant 66.7 ms
          // (two edges) offset, which silently destroyed a whole pairwise-extrinsic
          // recording - every frame simultaneous at the sensor, none simultaneous in the
          // bag. All cameras share one trigger edge, so the edge index derived from SOF
          // is identical across them to within the 1 us skew.
          bool send = true;
          if (publish_every_n_ > 1) {
            const int64_t period_ns = 1000000000LL / std::max(1, fps_);
            const int64_t edge = (static_cast<int64_t>(ft.sof_ns) + period_ns / 2) / period_ns;
            send = (edge % publish_every_n_) == 0;
          }
          publish_meta(i, ft, send && publish_y(i, ft.stamp_ns));
          if (first[i]) { RCLCPP_INFO(get_logger(), "cam idx %zu: first frame published", i); first[i] = false; }
        }

        if (got == n_) {
          empty_sweeps = 0;
          measure_set_skew(set_ts);
        } else if (got == 0) {
          // Distinguish "the trigger stopped" from "a camera died": in trigger mode a
          // sensor with no pulse produces no frames at all, and every camera goes quiet
          // together because they share one edge.
          if (++empty_sweeps % 5 == 1) {
            if (trigger_active_)
              RCLCPP_ERROR(get_logger(), "no frames from ANY camera while the driver is in "
                           "external-trigger mode — is the pulse generator running?");
            else
              RCLCPP_ERROR(get_logger(), "no frames from any camera");
          }
        }
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "capture_loop threw: %s", e.what());
    } catch (...) {
      RCLCPP_ERROR(get_logger(), "capture_loop threw unknown exception");
    }
  }

  // Returns whether the pixels actually went out: the caller records that in the frame's
  // timing message, so a consumer joining on the stamp knows an image is missing rather
  // than finding an unmatched timing record and having to guess.
  bool publish_y(size_t i, uint64_t t_ns) {
    NvBufferParams params;
    if (NvBufferGetParams(dmabufs_[i], &params) != 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cam idx %zu: NvBufferGetParams failed", i);
      return false;
    }
    void* mapped = nullptr;
    if (NvBufferMemMap(dmabufs_[i], 0, NvBufferMem_Read, &mapped) != 0 || !mapped) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cam idx %zu: NvBufferMemMap failed", i);
      return false;
    }
    NvBufferMemSyncForCpu(dmabufs_[i], 0, &mapped);

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp = rclcpp::Time(static_cast<int64_t>(t_ns));
    msg->header.frame_id = frame_ids_[i];
    msg->width = width_;
    msg->height = height_;
    msg->encoding = "mono8";
    msg->is_bigendian = 0;
    msg->step = width_;
    msg->data.resize(static_cast<size_t>(width_) * height_);
    const uint32_t pitch = params.pitch[0];
    const uint8_t* src = static_cast<const uint8_t*>(mapped);
    for (int r = 0; r < height_; ++r)
      memcpy(msg->data.data() + r * width_, src + r * pitch, width_);

    NvBufferMemUnMap(dmabufs_[i], 0, &mapped);
    if (!image_log_dir_.empty()) {
      // Hand the buffer to this camera's writer and return: a move, not a disk wait.
      //
      // DEPTH IS ABOUT SETS, NOT THROUGHPUT. Depth 8 (12.7 MB) gave 93.1% complete 4-camera
      // sets: the cameras drop INDEPENDENTLY, so a 1.5-4.5% per-camera loss compounded into
      // 7% of trigger edges missing somebody, and a set that is not a set is worth very little
      // from a synchronised rig. 64 frames is ~101 MB per camera - the board has ~6 GB spare -
      // and buys enough slack to ride out an eMMC or SD flush stall instead of dropping.
      {
        std::lock_guard<std::mutex> lk(wq_mtx_[i]);
        if (wq_[i].size() >= queue_depth_) { wq_[i].pop_front(); ++wq_dropped_[i]; }
        wq_[i].push_back(WriteJob{std::move(msg->data), t_ns});
      }
      wq_cv_[i].notify_one();
      return true;                 // deliberately NOT published: see image_log_dir_
    }
    pubs_[i]->publish(std::move(msg));
    return true;
  }

  std::vector<int64_t> sensor_ids_;
  std::vector<std::string> ports_, families_, topics_, frame_ids_;
  std::string ae_lock_mode_, calib_dir_;
  std::vector<double> ae_gain_, ae_dgain_;
  bool trigger_active_ = false, ae_lock_ = false, warned_no_metadata_ = false;
  bool stamp_midpoint_ = true, logged_exposure_ = false;
  int exposure_us_ = 0;
  uint64_t rig_exposure_ns_ = 0;
  int width_, height_, fps_, max_skew_us_, publish_every_n_ = 1;
  int64_t sets_ = 0, bad_sets_ = 0, worst_skew_us_ = 0;
  static constexpr size_t kHistory = 8;                 // ~0.27 s at 30 Hz
  std::vector<std::deque<uint64_t>> ts_history_;
  std::vector<int64_t> off_us_;
  std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
  size_t n_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> pubs_;
  std::vector<rclcpp::Publisher<bev_camera::msg::FrameMeta>::SharedPtr> meta_pubs_;
  std::string frame_log_dir_, image_log_dir_;
  std::vector<std::unique_ptr<std::ofstream>> frame_logs_;
  std::vector<std::unique_ptr<std::ofstream>> image_logs_, image_index_;
  std::vector<uint64_t> image_frames_;
  std::vector<std::string> image_dirs_;
  size_t queue_depth_ = 64;
  std::vector<std::deque<WriteJob>> wq_;
  std::vector<std::mutex> wq_mtx_;
  std::vector<std::condition_variable> wq_cv_;
  std::vector<uint64_t> wq_dropped_;
  std::vector<std::thread> writers_;
  bool wq_stop_ = false;
  std::atomic<bool> image_log_failed_{false};
  UniqueObj<CameraProvider> provider_;
  std::vector<UniqueObj<CaptureSession>> sessions_;
  std::vector<UniqueObj<OutputStream>> streams_;
  std::vector<UniqueObj<Request>> requests_;
  std::vector<UniqueObj<EGLStream::FrameConsumer>> consumers_;
  std::vector<int> dmabufs_;
  EGLDisplay egl_display_ = EGL_NO_DISPLAY;
  std::atomic<bool> running_{false};
  std::thread worker_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArgusCaptureNode>());
  rclcpp::shutdown();
  return 0;
}
