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
//                      track_us, remap_us, slam_us
//   slam_odometry.csv  t_ns, the SLAM-corrected pose of that set (/cuvslam/slam_odometry)
//   slam_path.csv      the FINAL optimised trajectory, GetAllSlamPoses after the last set
//   loop_closures.csv  event, t_ns of the set it fired on, landmark counts, and every closure
//                      pose cuVSLAM reported, de-duplicated on its stamp (lc_pose rows)
//   summary.json       counts and the configuration used
//
// Built and run by scripts/vo/replay_raw.sh.

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

void usage() {
  std::fprintf(stderr,
      "usage: cuvslam_raw_replay <imglog dir> <out dir> [--multicam-mode precision|performance|moderate]\n"
      "         [--slam] [--slam-primaries 0,1,2,5] [--slam-max-map-size 0] [--slam-throttling-ms 0]\n"
      "         [--async-sba 0|1] [--slam-sync 0|1] [--max-sets N] [--verbosity N]\n"
      "         [--calib-dir DIR] [--rig YAML] [--vstereo YAML] [--max-skew-us 1000]\n");
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
    else { usage(); return 2; }
  }

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

  cuvslam::Odometry::Config cfg = cuvslam::Odometry::GetDefaultConfig();
  cfg.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
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

  // ---- the loop -------------------------------------------------------------------------
  mkdir(out_dir.c_str(), 0755);
  std::ofstream odo(out_dir + "/odometry.csv"), so, lcf;
  odo.precision(9);
  odo << "t_ns,status,tx,ty,tz,qx,qy,qz,qw,keyframe,track_us,remap_us,slam_us\n";
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

    const auto t_track = std::chrono::steady_clock::now();
    cuvslam::PoseEstimate est;
    try {
      est = tracker.Track(images);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "Track() failed at %ld: %s\n", static_cast<long>(anchor.t), e.what());
      ++n_lost;
      odo << anchor.t << ",0,,,,,,,,0," << us_since(t_track) << ',' << remap_us << ",0\n";
      continue;
    }
    const int64_t track_us = us_since(t_track);
    cuvslam::Odometry::State st;
    tracker.GetState(st);
    if (st.keyframe) ++n_kf;
    if (!est.world_from_rig) {                   // the node publishes nothing and skips SLAM
      ++n_lost;
      odo << anchor.t << ",0,,,,,,,," << (st.keyframe ? 1 : 0) << ',' << track_us << ',' << remap_us << ",0\n";
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
    odo << ',' << (st.keyframe ? 1 : 0) << ',' << track_us << ',' << remap_us << ',' << slam_us << '\n';
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
     << ",\n  \"slam_path_poses\": " << n_path << ",\n  \"wall_s\": " << wall_s << "\n}\n";
  std::printf("%ld sets (%ld over the %ld us skew gate), %ld tracked, %ld lost, %ld keyframes, "
              "%ld loop closures, %zu optimised poses, %.0f s -> %s\n",
              static_cast<long>(n_sets), static_cast<long>(n_skew), static_cast<long>(max_skew_ns / 1000),
              static_cast<long>(n_tracked), static_cast<long>(n_lost), static_cast<long>(n_kf),
              static_cast<long>(lc_events), n_path, wall_s, out_dir.c_str());
  return 0;
}
