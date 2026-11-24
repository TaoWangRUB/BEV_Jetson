// cuVSLAM on a raw rig log, in-process: no ROS, no bag, no DDS, no wall clock.
//
// retarget-vo-to-imx296-rig task 5.10c. A bag replay through cuvslam_multicam_node plays in
// real time, so whenever Track() or Slam::Track() overruns the set budget the transport drops
// sets - even at 0.4x the SLAM replays lose sets at keyframes (README section 6) - and two
// replays of one bag drop different sets and differ by a median 1.13 m. It also needs the log
// converted to a bag first, and the bags in datasets/replay_out were cut to the IMU motion
// window (raw_log_to_bag.py --motion), so cuVSLAM never saw the frames slam-rs closes its long
// loop on. This reads the log the way slam-rs's driver does and calls the tracker once per set,
// every set, in order.
//
// The per-set path is the node's, unchanged: four frames matched on cam1's stamp within the
// 1 ms skew gate, each fisheye carved into two virtual pinholes by the SAME maps
// (bev_cuvslam/rig_build.hpp - the node, the fused node and this share one implementation),
// Odometry::Track on the 8 images stamped with their own exposure midpoints, then
// Slam::Track on the tracker's state. Only the transport differs.
//
// Deterministic by default, which is the other half of 5.10c: async_sba off (bundle adjustment
// inline, one pass per keyframe instead of whatever the wall clock allowed) and SLAM in sync
// mode (loop closure and pose-graph optimisation inline). --async-sba 1 --slam-sync 0 restores
// the node's threading, e.g. to see what it costs.
//
// Outputs in <out dir>, one row per set where there is one:
//   odometry.csv       t_ns, status (1 tracked, 0 lost), pose of cam1's optical frame in
//                      cuVSLAM's world - the same pose /cuvslam/odometry carries - keyframe,
//                      track_us, remap_us, slam_us (+ imu_us with --imu: RegisterImuMeasurement
//                      for the samples up to this set, NOT inside track_us)
//   slam_odometry.csv  t_ns, the SLAM-corrected pose of that set (/cuvslam/slam_odometry)
//   slam_path.csv      the FINAL optimised trajectory, GetAllSlamPoses after the last set
//   loop_closures.csv  event, t_ns of the set it fired on, landmark counts, and every closure
//                      pose cuVSLAM reported, de-duplicated on its stamp (lc_pose rows)
//   imu_state.csv      --imu only: per Track() call, gravity in the rig frame once cuVSLAM has
//                      initialised it (empty before), world velocity, gyro and accel biases
//   summary.json       counts and the configuration used
//
// --multisensor switches OdometryMode::Multicamera to the experimental cuNLS
// OdometryMode::Multisensor, which only exists in a USE_CUNLS=ON library (replay_raw.sh with
// CUVSLAM_LIB_DIR). No camera is declared a depth camera: the 8 virtual pinholes and the four
// virtual stereo pairs the frustum graph finds are the same as in Multicamera mode (precision
// FIG: depth_ids is {0} there and empty here, which changes nothing - camera 0 already has a
// partner). --imu adds the MPU-9250 to the rig (Rig::imus) and feeds every sample up to each
// set's stamp through RegisterImuMeasurement before Track(); cuVSLAM then switches the bundle
// adjustment to its inertial one and initialises gravity itself. The samples are prepared the
// way scripts/slamrs/run_slamrs.py load_imu() prepares them for slam-rs (see load_imu below).
//
// Built and run by scripts/vo/replay_raw.sh.

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "cuvslam/cuvslam2.h"
#include "bev_cuvslam/rig_build.hpp"
#include "bev_cuvslam/virtual_pinhole.hpp"

namespace {

struct Frame {
  int64_t t;
  int64_t off;
};

struct Mapped {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

Mapped map_file(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st{};
  fstat(fd, &st);
  void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  return {static_cast<const uint8_t*>(p), static_cast<size_t>(st.st_size)};
}

// camN_index.csv: two comment lines, a "stamp_ns,offset" header, then one row per frame.
// A row whose frame runs past the end of the .raw is dropped, as raw_log_to_bag.py does.
std::vector<Frame> read_index(const std::string& path, size_t raw_size, int64_t bpf) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<Frame> v;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#' || line.rfind("stamp", 0) == 0) continue;
    const auto c = line.find(',');
    const Frame fr{std::stoll(line.substr(0, c)), std::stoll(line.substr(c + 1))};
    if (fr.off >= 0 && static_cast<size_t>(fr.off + bpf) <= raw_size) v.push_back(fr);
  }
  return v;
}

std::vector<int64_t> parse_list(const std::string& s) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoll(tok));
  return v;
}

void write_pose(std::ofstream& f, const cuvslam::Pose& p) {
  for (int i = 0; i < 3; ++i) f << ',' << p.translation[i];
  for (int i = 0; i < 4; ++i) f << ',' << p.rotation[i];
}

int64_t us_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - t).count();
}

// ---- the IMU (--imu) ------------------------------------------------------------------------
struct ImuSample {
  int64_t t;
  double a[3], w[3];
};

struct ImuLog {
  std::vector<ImuSample> s;
  size_t rows = 0, rest_rows = 0, not_increasing = 0;
  double a0[3] = {0, 0, 0}, a0_norm = 0, gyro_rest[3] = {0, 0, 0};
};

// imu0.csv: '#' lines, then t_ns,a_x,a_y,a_z,w_x,w_y,w_z,temp,seq in the MPU-9250's own axes -
// the frame Kalibr's T_cam_imu was solved in. Prepared exactly as scripts/slamrs/run_slamrs.py
// load_imu() prepares it for slam-rs, so both estimators see the same samples:
//   * stamps moved onto the camera clock: t - Delta (Kalibr: t_imu = t_cam + timeshift_cam_imu);
//   * "slamrs" prep: the gyro mean over the first 2 s of the log (the rig is still) subtracted,
//     and of the accelerometer's rest mean a0 only the part ALONG a0 beyond 9.80665 m/s^2 (|a0|
//     reads ~10.3; one pose observes only that component). "raw" prep leaves both to cuVSLAM;
//   * a sample whose stamp does not exceed the previous ROW's is dropped (np.diff(t) > 0).
ImuLog load_imu(const std::string& path, int64_t delta_ns, bool slamrs_prep) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<ImuSample> all;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    ImuSample m{};
    const char* p = line.c_str();
    char* e = nullptr;
    m.t = std::strtoll(p, &e, 10) - delta_ns;
    double v[6];
    for (double& x : v) {
      if (*e != ',') throw std::runtime_error("bad IMU row in " + path + ": " + line);
      x = std::strtod(e + 1, &e);
    }
    for (int k = 0; k < 3; ++k) { m.a[k] = v[k]; m.w[k] = v[3 + k]; }
    all.push_back(m);
  }
  if (all.empty()) throw std::runtime_error("no IMU samples in " + path);
  ImuLog out;
  out.rows = all.size();
  const int64_t rest_end = all.front().t + 2'000'000'000;
  for (const auto& m : all) {
    if (m.t >= rest_end) continue;
    ++out.rest_rows;
    for (int k = 0; k < 3; ++k) { out.a0[k] += m.a[k]; out.gyro_rest[k] += m.w[k]; }
  }
  for (int k = 0; k < 3; ++k) { out.a0[k] /= out.rest_rows; out.gyro_rest[k] /= out.rest_rows; }
  out.a0_norm = std::sqrt(out.a0[0] * out.a0[0] + out.a0[1] * out.a0[1] + out.a0[2] * out.a0[2]);
  if (slamrs_prep) {
    double acc_off[3];
    for (int k = 0; k < 3; ++k) acc_off[k] = out.a0[k] - 9.80665 * out.a0[k] / out.a0_norm;
    for (auto& m : all)
      for (int k = 0; k < 3; ++k) { m.w[k] -= out.gyro_rest[k]; m.a[k] -= acc_off[k]; }
  }
  out.s.reserve(all.size());
  for (size_t i = 0; i < all.size(); ++i) {
    if (i > 0 && all[i].t <= all[i - 1].t) { ++out.not_increasing; continue; }
    out.s.push_back(all[i]);
  }
  return out;
}

// rig_from_imu. The rig frame IS cam1's raw optical frame (rig_build.hpp). The only tracked copy
// of the round-2 Kalibr cam1-IMU solve is config/slamrs/bev_calib_s2.json: Basalt's T_imu_cam[0],
// the pose of cam1 in the IMU frame (p_imu = T_imu_cam * p_cam1; make_calib.py writes it as
// inv(Kalibr T_cam_imu) @ rig_in_cam1[cam1] = identity). cuVSLAM wants v_rig = rig_from_imu * v_imu,
// i.e. its inverse - Kalibr's T_cam_imu itself. JSON is YAML, so yaml-cpp reads it.
cv::Matx44d load_rig_from_imu(const std::string& calib_json) {
  const YAML::Node c = YAML::LoadFile(calib_json)["value0"]["T_imu_cam"][0];
  double x = c["qx"].as<double>(), y = c["qy"].as<double>(), z = c["qz"].as<double>(),
         w = c["qw"].as<double>();
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  x /= n; y /= n; z /= n; w /= n;
  const cv::Matx33d R(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
  const cv::Vec3d p(c["px"].as<double>(), c["py"].as<double>(), c["pz"].as<double>());
  const cv::Matx33d Rt = R.t();
  const cv::Vec3d ti = -(Rt * p);
  return cv::Matx44d(Rt(0, 0), Rt(0, 1), Rt(0, 2), ti[0],
                     Rt(1, 0), Rt(1, 1), Rt(1, 2), ti[1],
                     Rt(2, 0), Rt(2, 1), Rt(2, 2), ti[2],
                     0, 0, 0, 1);
}

// Which file the loader actually mapped for a library - the Multisensor run must not silently
// pick up the cuNLS-less build_host libcuvslam.so from an LD_LIBRARY_PATH, nor CUDA 13.2's
// libcusparse.so.12 (replay_raw.sh puts CUDA 12 first).
std::string loaded_library(const std::string& name) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    const auto k = line.find('/');
    if (k != std::string::npos && line.find(name, k) != std::string::npos) return line.substr(k);
  }
  return "?";
}

void usage() {
  std::fprintf(stderr,
      "usage: cuvslam_raw_replay <imglog dir> <out dir> [--multicam-mode precision|performance|moderate]\n"
      "         [--slam] [--slam-primaries 0,1,2,5] [--slam-max-map-size 0] [--slam-throttling-ms 0]\n"
      "         [--async-sba 0|1] [--slam-sync 0|1] [--max-sets N] [--verbosity N]\n"
      "         [--calib-dir DIR] [--rig YAML] [--vstereo YAML] [--max-skew-us 1000]\n"
      "         [--multisensor [--imu [--imu-yaml YAML] [--imu-calib JSON] [--imu-noise-scale 1]\n"
      "                                [--imu-prep slamrs|raw]]]\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { usage(); return 2; }
  const std::string log_dir = argv[1], out_dir = argv[2];
  // The node's defaults, from bev_cuvslam.launch.py.
  std::string calib_dir = "config/calib/imx296_1456x1088";
  std::string rig_path = "config/rig/rig_extrinsics_imx296.yaml";
  std::string vstereo_path = "config/rig/virtual_stereo_imx296.yaml";
  std::string mode = "precision", primaries_s = "-1";
  bool slam_on = false, async_sba = false, slam_sync = true;
  int max_map_size = 0, throttling_ms = 0, verbosity = 0;
  int64_t max_sets = 0, max_skew_ns = 1000000;
  bool multisensor = false, use_imu = false;
  std::string imu_yaml = "config/calib/imu_mpu9250.yaml", imu_calib = "config/slamrs/bev_calib_s2.json";
  std::string imu_prep = "slamrs";
  double imu_noise_scale = 1.0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--multicam-mode") mode = val();
    else if (a == "--slam") slam_on = true;
    else if (a == "--slam-primaries") primaries_s = val();
    else if (a == "--slam-max-map-size") max_map_size = std::stoi(val());
    else if (a == "--slam-throttling-ms") throttling_ms = std::stoi(val());
    else if (a == "--async-sba") async_sba = std::stoi(val()) != 0;
    else if (a == "--slam-sync") slam_sync = std::stoi(val()) != 0;
    else if (a == "--max-sets") max_sets = std::stoll(val());
    else if (a == "--verbosity") verbosity = std::stoi(val());
    else if (a == "--calib-dir") calib_dir = val();
    else if (a == "--rig") rig_path = val();
    else if (a == "--vstereo") vstereo_path = val();
    else if (a == "--max-skew-us") max_skew_ns = std::stoll(val()) * 1000;
    else if (a == "--multisensor") multisensor = true;
    else if (a == "--imu") use_imu = true;
    else if (a == "--imu-yaml") imu_yaml = val();
    else if (a == "--imu-calib") imu_calib = val();
    else if (a == "--imu-noise-scale") imu_noise_scale = std::stod(val());
    else if (a == "--imu-prep") imu_prep = val();
    else { usage(); return 2; }
  }
  if (use_imu && !multisensor) {
    std::fprintf(stderr, "--imu needs --multisensor (Multicamera mode takes no IMU)\n");
    return 2;
  }
  if (imu_prep != "slamrs" && imu_prep != "raw") { usage(); return 2; }

  // ---- the log ------------------------------------------------------------------------
  int width = 0, height = 0;
  int64_t bpf = 0;
  {
    std::ifstream g(log_dir + "/geometry.txt");
    if (!g) throw std::runtime_error("no geometry.txt in " + log_dir);
    std::string k;
    std::string v;
    while (g >> k >> v) {
      if (k == "width") width = std::stoi(v);
      else if (k == "height") height = std::stoi(v);
      else if (k == "bytes_per_frame") bpf = std::stoll(v);
      else if (k == "encoding" && v != "mono8") throw std::runtime_error("encoding " + v + " is not mono8");
    }
  }
  if (bpf != static_cast<int64_t>(width) * height)
    throw std::runtime_error("bytes_per_frame is not width*height - not a mono8 log");
  const std::array<std::string, 4> cams = {"cam1", "cam2", "cam3", "cam4"};
  std::array<Mapped, 4> raw;
  std::array<std::vector<Frame>, 4> idx;
  for (int c = 0; c < 4; ++c) {
    raw[c] = map_file(log_dir + "/" + cams[c] + ".raw");
    idx[c] = read_index(log_dir + "/" + cams[c] + "_index.csv", raw[c].size, bpf);
  }
  // cams 2-4 searched by stamp, so sorted copies; cam1 is the anchor, in capture order.
  std::array<std::vector<Frame>, 4> by_t = idx;
  for (int c = 1; c < 4; ++c)
    std::sort(by_t[c].begin(), by_t[c].end(), [](const Frame& a, const Frame& b) { return a.t < b.t; });

  // ---- the rig and the tracker, as the node builds them ---------------------------------
  int major = 0, minor = 0, patch = 0;
  cuvslam::GetVersion(&major, &minor, &patch);
  if (verbosity > 0) cuvslam::SetVerbosity(verbosity);
  cuvslam::WarmUpGPU();
  auto vrig = bev_cuvslam::BuildVirtualRig(calib_dir, rig_path, vstereo_path,
                                           {cams[0], cams[1], cams[2], cams[3]});
  for (const auto& w : vrig.warnings) std::fprintf(stderr, "WARNING: %s\n", w.c_str());
  const auto& vpin = vrig.vpin;
  const auto& vsrc = vrig.vsrc;

  // ---- the IMU, when asked for: noise, rate and Delta from the IMU yaml, extrinsic from the
  // slam-rs calibration (the same numbers slam-rs runs on, see load_rig_from_imu) ----------------
  ImuLog imu;
  cv::Matx44d rig_from_imu_m = cv::Matx44d::eye();
  int64_t imu_delta_ns = 0;
  cuvslam::ImuCalibration imu_cal{};
  if (use_imu) {
    const YAML::Node iy = YAML::LoadFile(imu_yaml);
    imu_delta_ns = std::llround(iy["timeshift_cam_imu"].as<double>() * 1e9);
    rig_from_imu_m = load_rig_from_imu(imu_calib);
    imu_cal.rig_from_imu = bev_cuvslam::pose_from_matrix(rig_from_imu_m);
    imu_cal.gyroscope_noise_density = static_cast<float>(imu_noise_scale * iy["gyroscope_noise_density"].as<double>());
    imu_cal.gyroscope_random_walk = static_cast<float>(imu_noise_scale * iy["gyroscope_random_walk"].as<double>());
    imu_cal.accelerometer_noise_density =
        static_cast<float>(imu_noise_scale * iy["accelerometer_noise_density"].as<double>());
    imu_cal.accelerometer_random_walk =
        static_cast<float>(imu_noise_scale * iy["accelerometer_random_walk"].as<double>());
    imu_cal.frequency = static_cast<float>(iy["update_rate"].as<double>());
    vrig.rig.imus.push_back(imu_cal);
    imu = load_imu(log_dir + "/imu0.csv", imu_delta_ns, imu_prep == "slamrs");
  }

  cuvslam::Odometry::Config cfg = cuvslam::Odometry::GetDefaultConfig();
  cfg.odometry_mode = multisensor ? cuvslam::Odometry::OdometryMode::Multisensor
                                  : cuvslam::Odometry::OdometryMode::Multicamera;
  cfg.multicam_mode = bev_cuvslam::ParseMulticamMode(mode);
  cfg.use_gpu = true;
  cfg.async_sba = async_sba;
  cfg.rectified_stereo_camera = false;           // see the node: our facing pinholes are 1-1.4 deg apart
  // GetState() throws unless the exports are on; it carries the keyframe flag and is what
  // Slam::Track consumes. The node turns them on for SLAM and for OBS=1 replays alike.
  cfg.enable_observations_export = true;
  cfg.enable_landmarks_export = true;
  cuvslam::Odometry tracker(vrig.rig, cfg);

  std::unique_ptr<cuvslam::Slam> slam;
  std::vector<uint8_t> primary;
  if (slam_on) {
    const auto prim = parse_list(primaries_s);
    if (prim.empty() || (prim.size() == 1 && prim[0] == -1)) {
      for (size_t k = 0; k < vpin.size(); ++k) primary.push_back(static_cast<uint8_t>(k));
    } else {
      for (int64_t k : prim) {
        if (k < 0 || static_cast<size_t>(k) >= vpin.size())
          throw std::runtime_error("--slam-primaries index out of range");
        primary.push_back(static_cast<uint8_t>(k));
      }
    }
    cuvslam::Slam::Config sc = cuvslam::Slam::GetDefaultConfig();
    sc.use_gpu = true;
    sc.sync_mode = slam_sync;
    sc.enable_reading_internals = true;
    sc.throttling_time_ms = static_cast<uint32_t>(throttling_ms);
    sc.max_map_size = static_cast<uint32_t>(max_map_size);
    slam = std::make_unique<cuvslam::Slam>(vrig.rig, primary, sc);
    slam->EnableReadingData(cuvslam::Slam::DataLayer::LoopClosure, 4096);
    slam->EnableReadingData(cuvslam::Slam::DataLayer::PoseGraph, 4096);
  }
  std::printf("cuVSLAM %d.%d.%d, %zu virtual cameras, multicam_mode %s, async_sba %d, SLAM %s\n",
              major, minor, patch, vpin.size(), mode.c_str(), async_sba ? 1 : 0,
              slam_on ? ("on, " + std::to_string(primary.size()) + " primaries, sync " +
                         std::to_string(slam_sync ? 1 : 0)).c_str() : "off");
  std::string ms_primaries;
  if (multisensor) {
    // A rig whose pairs the FIG missed would track monocular without a word (memory:
    // bev-debug-read-source) - so say what it built.
    for (uint8_t c : tracker.GetPrimaryCameras()) ms_primaries += (ms_primaries.empty() ? "" : ",") + std::to_string(c);
    std::printf("Multisensor (cuNLS), no depth cameras, primaries [%s], IMU %s\n  library %s\n  cuSPARSE %s\n",
                ms_primaries.c_str(), use_imu ? "on" : "off", loaded_library("libcuvslam.so").c_str(),
                loaded_library("libcusparse.so").c_str());
    if (use_imu) {
      std::printf("  IMU %zu rows -> %zu samples (%zu not increasing), Delta %.4f ms, prep %s: |a|rest %.3f, "
                  "gyro rest mean %.5f %.5f %.5f rad/s, noise x%.2f (g %.3g/%.3g, a %.3g/%.3g), %.0f Hz\n",
                  imu.rows, imu.s.size(), imu.not_increasing, imu_delta_ns * 1e-6, imu_prep.c_str(), imu.a0_norm,
                  imu.gyro_rest[0], imu.gyro_rest[1], imu.gyro_rest[2], imu_noise_scale,
                  imu_cal.gyroscope_noise_density, imu_cal.gyroscope_random_walk,
                  imu_cal.accelerometer_noise_density, imu_cal.accelerometer_random_walk, imu_cal.frequency);
      std::printf("  rig_from_imu t %.4f %.4f %.4f; accelerometer at rest in the rig frame %.3f %.3f %.3f "
                  "(raw cam1: +y is physically up)\n",
                  rig_from_imu_m(0, 3), rig_from_imu_m(1, 3), rig_from_imu_m(2, 3),
                  rig_from_imu_m(0, 0) * imu.a0[0] + rig_from_imu_m(0, 1) * imu.a0[1] + rig_from_imu_m(0, 2) * imu.a0[2],
                  rig_from_imu_m(1, 0) * imu.a0[0] + rig_from_imu_m(1, 1) * imu.a0[1] + rig_from_imu_m(1, 2) * imu.a0[2],
                  rig_from_imu_m(2, 0) * imu.a0[0] + rig_from_imu_m(2, 1) * imu.a0[1] + rig_from_imu_m(2, 2) * imu.a0[2]);
    }
  }

  // ---- the loop -------------------------------------------------------------------------
  mkdir(out_dir.c_str(), 0755);
  std::ofstream odo(out_dir + "/odometry.csv"), so, lcf, imuf;
  odo.precision(9);
  odo << "t_ns,status,tx,ty,tz,qx,qy,qz,qw,keyframe,track_us,remap_us,slam_us" << (use_imu ? ",imu_us\n" : "\n");
  // The extra column goes on the end of every row, lost ones included; IMU-less rows are unchanged.
  const char* odo_eol = use_imu ? nullptr : "\n";
  int64_t imu_us = 0;
  auto end_row = [&]() {
    if (odo_eol) odo << odo_eol;
    else odo << ',' << imu_us << '\n';
  };
  size_t imu_next = 0, imu_registered = 0, imu_rejected = 0;
  int64_t imu_last_t = std::numeric_limits<int64_t>::min(), gravity_t = -1, n_gravity = 0;
  if (use_imu) {
    imuf.open(out_dir + "/imu_state.csv");
    imuf.precision(9);
    imuf << "t_ns,status,gx,gy,gz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n";
  }
  if (slam_on) {
    so.open(out_dir + "/slam_odometry.csv");
    so.precision(9);
    so << "t_ns,tx,ty,tz,qx,qy,qz,qw\n";
    lcf.open(out_dir + "/loop_closures.csv");
    lcf.precision(9);
    lcf << "kind,event,t_ns,tracked,pnp,good,tx,ty,tz,qx,qy,qz,qw\n";
  }
  std::array<cv::Mat, 8> vimg;
  std::set<int64_t> lc_seen;
  bool lc_prev = false;
  int64_t n_sets = 0, n_skew = 0, n_tracked = 0, n_lost = 0, n_kf = 0, lc_events = 0, pgo_events = 0;
  int64_t worst_skew = 0;
  const auto t_start = std::chrono::steady_clock::now();
  for (const Frame& anchor : idx[0]) {
    if (max_sets && n_sets >= max_sets) break;
    std::array<Frame, 4> set;
    set[0] = anchor;
    int64_t lo = anchor.t, hi = anchor.t;
    for (int c = 1; c < 4; ++c) {
      const auto& v = by_t[c];
      auto it = std::lower_bound(v.begin(), v.end(), anchor.t,
                                 [](const Frame& f, int64_t t) { return f.t < t; });
      const Frame* best = nullptr;
      if (it != v.end()) best = &*it;
      if (it != v.begin()) {
        const Frame* prev = &*(it - 1);
        if (!best || std::llabs(prev->t - anchor.t) < std::llabs(best->t - anchor.t)) best = prev;
      }
      if (!best) throw std::runtime_error(cams[c] + " has no frames");
      set[c] = *best;
      lo = std::min(lo, best->t);
      hi = std::max(hi, best->t);
    }
    ++n_sets;
    if (hi - lo > max_skew_ns) { ++n_skew; continue; }   // dropped and counted, never re-stamped
    worst_skew = std::max(worst_skew, hi - lo);

    const auto t_remap = std::chrono::steady_clock::now();
    cuvslam::Odometry::ImageSet images;
    images.reserve(vpin.size());
    for (size_t k = 0; k < vpin.size(); ++k) {
      const int c = vsrc[k];
      const cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(raw[c].data + set[c].off));
      cv::remap(src, vimg[k], vpin[k].map1, vpin[k].map2, cv::INTER_LINEAR);
      cuvslam::Image im{};
      im.pixels = vimg[k].data;
      im.width = vimg[k].cols;
      im.height = vimg[k].rows;
      im.pitch = static_cast<int32_t>(vimg[k].step);
      im.encoding = cuvslam::ImageData::Encoding::MONO;
      im.data_type = cuvslam::ImageData::DataType::UINT8;
      im.is_gpu_mem = false;
      im.timestamp_ns = set[c].t;                // each virtual camera keeps its fisheye's stamp
      im.camera_index = static_cast<uint32_t>(k);
      images.push_back(im);
    }
    const int64_t remap_us = us_since(t_remap);

    if (use_imu) {
      // Every sample up to this set's stamp (cuVSLAM's frame time is images[0]'s = cam1's), in
      // order, before Track() - cuvslam2.h's call pattern. A skew-dropped set leaves its samples
      // to the next one, so nothing is lost.
      const auto t_imu = std::chrono::steady_clock::now();
      for (; imu_next < imu.s.size() && imu.s[imu_next].t <= anchor.t; ++imu_next) {
        const ImuSample& m = imu.s[imu_next];
        if (m.t < imu_last_t) { ++imu_rejected; continue; }
        cuvslam::ImuMeasurement im{};
        im.timestamp_ns = m.t;
        im.linear_accelerations = {static_cast<float>(m.a[0]), static_cast<float>(m.a[1]), static_cast<float>(m.a[2])};
        im.angular_velocities = {static_cast<float>(m.w[0]), static_cast<float>(m.w[1]), static_cast<float>(m.w[2])};
        try {
          tracker.RegisterImuMeasurement(0, im);
          imu_last_t = m.t;
          ++imu_registered;
        } catch (const std::exception& e) {
          if (++imu_rejected <= 5)
            std::fprintf(stderr, "RegisterImuMeasurement(%ld) failed: %s\n", static_cast<long>(m.t), e.what());
        }
      }
      imu_us = us_since(t_imu);
    }

    const auto t_track = std::chrono::steady_clock::now();
    cuvslam::PoseEstimate est;
    try {
      est = tracker.Track(images);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Track() failed at %ld: %s\n", static_cast<long>(anchor.t), e.what());
      ++n_lost;
      odo << anchor.t << ",0,,,,,,,,0," << us_since(t_track) << ',' << remap_us << ",0";
      end_row();
      continue;
    }
    const int64_t track_us = us_since(t_track);
    if (use_imu) imu_last_t = std::max(imu_last_t, anchor.t);
    cuvslam::Odometry::State st;
    tracker.GetState(st);
    if (st.keyframe) ++n_kf;
    if (use_imu) {
      const auto g = tracker.GetLastGravity();
      const auto s = tracker.GetImuState();
      if (g) {
        ++n_gravity;
        if (gravity_t < 0) {
          gravity_t = anchor.t;
          std::printf("  gravity initialised at set %ld (%.2f s): %.3f %.3f %.3f in the rig frame\n",
                      static_cast<long>(n_sets), (anchor.t - idx[0].front().t) * 1e-9, (*g)[0], (*g)[1], (*g)[2]);
        }
      }
      imuf << anchor.t << ',' << (est.world_from_rig ? 1 : 0);
      if (g) imuf << ',' << (*g)[0] << ',' << (*g)[1] << ',' << (*g)[2];
      else imuf << ",,,";
      if (s) {
        for (const auto* v : {&s->velocity, &s->gyro_bias, &s->acc_bias})
          for (int k = 0; k < 3; ++k) imuf << ',' << (*v)[k];
      } else {
        imuf << ",,,,,,,,,";
      }
      imuf << '\n';
    }
    if (!est.world_from_rig) {                   // the node publishes nothing and skips SLAM
      ++n_lost;
      odo << anchor.t << ",0,,,,,,,," << (st.keyframe ? 1 : 0) << ',' << track_us << ',' << remap_us << ",0";
      end_row();
      continue;
    }
    ++n_tracked;
    int64_t slam_us = 0;
    if (slam) {
      const auto t_s = std::chrono::steady_clock::now();
      slam->Track(st);
      slam_us = us_since(t_s);
      so << anchor.t;
      write_pose(so, slam->GetPose());
      so << '\n';
      cuvslam::Slam::Metrics m{};
      slam->GetSlamMetrics(m);
      if (m.lc_status && !lc_prev) {             // a level, so count the rising edge
        ++lc_events;
        lcf << "event," << lc_events << ',' << anchor.t << ',' << m.lc_tracked_landmarks_count << ','
            << m.lc_pnp_landmarks_count << ',' << m.lc_good_landmarks_count << ",,,,,,,\n";
        std::vector<cuvslam::PoseStamped> poses;
        slam->GetLoopClosurePoses(poses);        // a rolling window: accumulate, de-duplicate
        for (const auto& ps : poses) {
          if (!lc_seen.insert(ps.timestamp_ns).second) continue;
          lcf << "lc_pose," << lc_events << ',' << ps.timestamp_ns << ",,,";
          write_pose(lcf, ps.pose);
          lcf << '\n';
        }
      }
      lc_prev = m.lc_status;
      if (m.pgo_status) ++pgo_events;
    }
    odo << anchor.t << ",1";
    write_pose(odo, est.world_from_rig->pose);
    odo << ',' << (st.keyframe ? 1 : 0) << ',' << track_us << ',' << remap_us << ',' << slam_us;
    end_row();
    if (n_sets % 200 == 0)
      std::printf("  %ld sets, %ld tracked, %ld loop closures, %.0f s\n", static_cast<long>(n_sets),
                  static_cast<long>(n_tracked), static_cast<long>(lc_events), us_since(t_start) * 1e-6);
  }
  const double wall_s = us_since(t_start) * 1e-6;

  size_t n_path = 0;
  if (slam) {
    std::vector<cuvslam::PoseStamped> path;
    slam->GetAllSlamPoses(path);
    n_path = path.size();
    std::ofstream pf(out_dir + "/slam_path.csv");
    pf.precision(9);
    pf << "t_ns,tx,ty,tz,qx,qy,qz,qw\n";
    for (const auto& ps : path) {
      pf << ps.timestamp_ns;
      write_pose(pf, ps.pose);
      pf << '\n';
    }
  }

  std::ofstream js(out_dir + "/summary.json");
  js << "{\n  \"log\": \"" << log_dir << "\",\n  \"cuvslam\": \"" << major << '.' << minor << '.' << patch
     << "\",\n  \"multicam_mode\": \"" << mode << "\",\n  \"async_sba\": " << (async_sba ? "true" : "false")
     << ",\n  \"slam\": " << (slam_on ? "true" : "false") << ",\n  \"slam_sync\": " << (slam_sync ? "true" : "false")
     << ",\n  \"slam_primaries\": [";
  for (size_t i = 0; i < primary.size(); ++i) js << (i ? ", " : "") << int(primary[i]);
  js << "],\n  \"slam_max_map_size\": " << max_map_size << ",\n  \"frames\": [" << idx[0].size() << ", "
     << idx[1].size() << ", " << idx[2].size() << ", " << idx[3].size() << "],\n  \"sets\": " << n_sets
     << ",\n  \"skew_dropped\": " << n_skew << ",\n  \"worst_skew_us\": " << worst_skew / 1000
     << ",\n  \"tracked\": " << n_tracked << ",\n  \"lost\": " << n_lost << ",\n  \"keyframes\": " << n_kf
     << ",\n  \"loop_closure_events\": " << lc_events << ",\n  \"pgo_events\": " << pgo_events
     << ",\n  \"slam_path_poses\": " << n_path << ",\n  \"wall_s\": " << wall_s;
  if (multisensor) {                            // Multicamera summaries keep their exact shape
    js << ",\n  \"odometry_mode\": \"multisensor\",\n  \"libcuvslam\": \"" << loaded_library("libcuvslam.so")
       << "\",\n  \"libcusparse\": \"" << loaded_library("libcusparse.so") << "\",\n  \"primary_cameras\": ["
       << ms_primaries << "],\n  \"imu\": ";
    if (!use_imu) {
      js << "null";
    } else {
      js.precision(9);
      js << "{\n    \"csv\": \"" << log_dir << "/imu0.csv\",\n    \"yaml\": \"" << imu_yaml << "\",\n    \"calib\": \""
         << imu_calib << "\",\n    \"prep\": \"" << imu_prep << "\",\n    \"delta_ns\": " << imu_delta_ns
         << ",\n    \"noise_scale\": " << imu_noise_scale
         << ",\n    \"gyroscope_noise_density\": " << imu_cal.gyroscope_noise_density
         << ",\n    \"gyroscope_random_walk\": " << imu_cal.gyroscope_random_walk
         << ",\n    \"accelerometer_noise_density\": " << imu_cal.accelerometer_noise_density
         << ",\n    \"accelerometer_random_walk\": " << imu_cal.accelerometer_random_walk
         << ",\n    \"frequency\": " << imu_cal.frequency << ",\n    \"rig_from_imu\": [";
      for (int r = 0; r < 4; ++r) {
        js << (r ? ", [" : "[");
        for (int c = 0; c < 4; ++c) js << (c ? ", " : "") << rig_from_imu_m(r, c);
        js << ']';
      }
      js << "],\n    \"rows\": " << imu.rows << ",\n    \"samples\": " << imu.s.size() << ",\n    \"not_increasing\": "
         << imu.not_increasing << ",\n    \"registered\": " << imu_registered << ",\n    \"rejected\": " << imu_rejected
         << ",\n    \"rest_rows\": " << imu.rest_rows << ",\n    \"acc_rest_norm\": " << imu.a0_norm
         << ",\n    \"gyro_rest_mean\": [" << imu.gyro_rest[0] << ", " << imu.gyro_rest[1] << ", " << imu.gyro_rest[2]
         << "],\n    \"gravity_init_t_ns\": " << gravity_t << ",\n    \"sets_with_gravity\": " << n_gravity
         << "\n  }";
    }
  }
  js << "\n}\n";
  std::printf("%ld sets (%ld over the %ld us skew gate), %ld tracked, %ld lost, %ld keyframes, "
              "%ld loop closures, %zu optimised poses, %.0f s -> %s\n",
              static_cast<long>(n_sets), static_cast<long>(n_skew), static_cast<long>(max_skew_ns / 1000),
              static_cast<long>(n_tracked), static_cast<long>(n_lost), static_cast<long>(n_kf),
              static_cast<long>(lc_events), n_path, wall_s, out_dir.c_str());
  if (use_imu)
    std::printf("IMU: %zu samples registered, %zu rejected; gravity from %s, on %ld sets\n", imu_registered,
                imu_rejected, gravity_t < 0 ? "never" : std::to_string(gravity_t).c_str(),
                static_cast<long>(n_gravity));
  return 0;
}
