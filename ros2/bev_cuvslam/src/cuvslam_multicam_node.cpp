// cuVSLAM 4-camera (surround fisheye) multicamera visual-odometry node, ROS 2 Foxy.
//
// Runs the CUDA-10.2-ported libcuvslam in OdometryMode::Multicamera (visual only;
// cuVSLAM v15 does NOT fuse an IMU in multicam mode — the IMU is fused externally
// by an EKF). Loads per-camera MEI / omni-radtan intrinsics (camN.yaml, via LoadOmni)
// + the ring-closed rig extrinsics (rig_extrinsics_imx296.yaml), builds the cuVSLAM
// rig, synchronizes 4 image topics, calls Track(), and publishes nav_msgs/Odometry
// + a TF.
//
// NOT Kannala-Brandt, which this comment claimed until 2026-09-01. These lenses are
// ~192 deg and pinhole-equi diverged on every camera; the calibration is Mei. cuVSLAM
// never sees either model - each fisheye is carved into two virtual PINHOLES with no
// distortion, because the remap has already removed it (see the rig loop below).
//
// SYNC: the rig is hardware-triggered (4x IMX296 on one STM32 edge, measured skew 1 us),
// so the cameras really do capture the same instant and each frame carries its own
// exposure-midpoint timestamp (docs/timestamps.md). A set is four frames whose stamps span less
// than max_skew_us — cuVSLAM's own Multicamera gate is 1 ms — and a set that fails is
// DROPPED AND COUNTED, never re-stamped.
//
// The previous version had to bundle the latest frame per camera and hand cuVSLAM one
// synthesised timestamp, because the free-running IMX219 rig could not produce a
// coherent set at all. That workaround is gone: it threw away the per-frame time on the
// one rig that has it, and it made an unsynchronised set look acceptable instead of
// making it visible.

#include <array>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <set>
#include <unordered_map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include "cuvslam/cuvslam2.h"
#include "bev_cuvslam/rig_build.hpp"
#include "bev_cuvslam/virtual_pinhole.hpp"

namespace {

// Load a YAML file, tolerating the OpenCV "%YAML:1.0 / ---" preamble.
YAML::Node load_yaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("%YAML", 0) == 0) continue;   // skip directive
    if (line == "---") continue;
    ss << line << "\n";
  }
  return YAML::Load(ss.str());
}

// 4x4 row-major matrix from yaml (rig_in_cam1 blocks).
// cuVSLAM Pose (quaternion xyzw + translation) from a 4x4. cuVSLAM uses the OpenCV
// convention - x right, y down, z forward - which is what our extrinsics are already in.
// rig_from_<frame> pose from a node with t_xyz_m + q_wxyz (yaml is wxyz; cuVSLAM wants xyzw).
cuvslam::Pose load_pose(const YAML::Node& n) {
  cuvslam::Pose p;
  auto t = n["t_xyz_m"];
  p.translation = {t[0].as<float>(), t[1].as<float>(), t[2].as<float>()};
  auto q = n["q_wxyz"];  // [w,x,y,z]
  p.rotation = {q[1].as<float>(), q[2].as<float>(), q[3].as<float>(), q[0].as<float>()};  // -> x,y,z,w
  return p;
}

}  // namespace

class CuvslamMulticamNode : public rclcpp::Node {
 public:
  CuvslamMulticamNode() : Node("cuvslam_multicam") {
    calib_dir_ = declare_parameter<std::string>("calib_dir", "config/calib/imx296_1456x1088");
    rig_path_ = declare_parameter<std::string>("rig_extrinsics", "config/rig/rig_extrinsics_imx296.yaml");
    vstereo_path_ = declare_parameter<std::string>("virtual_stereo", "config/rig/virtual_stereo_imx296.yaml");
    cams_ = declare_parameter<std::vector<std::string>>("cameras", {"cam1", "cam2", "cam3", "cam4"});
    topics_ = declare_parameter<std::vector<std::string>>(
        "image_topics", {"/cam1/image_raw", "/cam2/image_raw", "/cam3/image_raw", "/cam4/image_raw"});
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    // NOT "base_link", and the difference matters. cuVSLAM reports world_from_rig, and
    // this node's rig frame IS cam1's optical frame (z forward, x right, y down),
    // additionally rolled 180 deg by the inverted mount. Publishing that as base_link
    // would tell every tf consumer it is FLU on the vehicle, which it is not - and a
    // 180 deg roll produces trajectories that look entirely plausible. Publishing a
    // true base_link needs R_body_from_cam1, which is not measured; see
    // config/rig/rig_layout.yaml and 3R.16b. Override the parameter only once it is.
    base_frame_ = declare_parameter<std::string>("base_frame", "cam1_optical_frame");
    // A set whose frames span more than this is not a set. cuVSLAM's Multicamera gate is
    // 1 ms; the triggered rig measures 1 us, so anything near the limit is a fault, not
    // something to widen the window for.
    max_skew_ns_ = static_cast<int64_t>(declare_parameter<int>("max_skew_us", 1000)) * 1000;
    // How far back to look for a matching frame from the other cameras. A few frame
    // periods is plenty when they share a trigger edge; more just delays noticing a fault.
    history_ = static_cast<size_t>(declare_parameter<int>("match_history", 8));
    // Off by default: the landmark export slows an already Track()-bound node, so only
    // enable it for visualisation/debug runs, not the §5 rate measurement.
    publish_landmarks_ = declare_parameter<bool>("publish_landmarks", false);
    landmark_stride_ = declare_parameter<int>("landmark_stride", 3);
    publish_observations_ = declare_parameter<bool>("publish_observations", false);
    // cuVSLAM's own logging is OFF unless we ask: SetVerbosity(0) is the library default and
    // nothing in this node used to call it, so the library was silent by construction. Worth
    // knowing what it will and will not say (libs/odometry/multi_visual_odometry_base.cpp):
    // its three tracking messages — "images are not available", "Failed to track on the 2D
    // tracking stage", "Failed to track on the PnP stage" — all sit on paths that return
    // false, which reaches us as an EMPTY world_from_rig and the "tracking lost" warning
    // below. So they cover the case we already see, and say nothing about the one that
    // actually bit us in 5.0g: a solve that SUCCEEDS on a featureless view and returns a
    // zero delta. 1=Error 2=Warning 3=Message (Release caps at 3).
    // SLAM / loop closure. Odometry alone has no pose graph and nothing pins the
    // trajectory, which is the structural source of the drift in section 5. cuvslam::Slam
    // adds the graph and the loop closures on top of the SAME Odometry - it consumes
    // Odometry::State, it does not replace the tracker.
    enable_slam_ = declare_parameter<bool>("enable_slam", false);
    slam_map_path_ = declare_parameter<std::string>("slam_map_path", "");
    // 0 = no throttle. The header suggests 1000 ms for real-time mapping; leave it open
    // offline, where wall-clock is not the constraint.
    slam_throttling_ms_ = declare_parameter<int>("slam_throttling_ms", 0);
    // 300 poses is the header's real-time figure; 0 is an unlimited graph.
    slam_max_map_size_ = declare_parameter<int>("slam_max_map_size", 300);
    debug_dump_dir_ = declare_parameter<std::string>("cuvslam_debug_dump_dir", "");
    // Saturation gate. See check_exposure(): cuVSLAM has no image-quality input at all, so
    // a blown frame reaches the solver looking like a valid one.
    sat_level_ = declare_parameter<int>("saturation_level", 227);
    sat_warn_frac_ = declare_parameter<double>("saturation_warn_fraction", 0.5);
    const int verbosity = declare_parameter<int>("cuvslam_verbosity", 0);
    if (verbosity > 0) {
      cuvslam::SetVerbosity(verbosity);
      RCLCPP_INFO(get_logger(), "cuVSLAM library verbosity %d (its own messages go to stdout)",
                  verbosity);
    }

    if (cams_.size() != 4 || topics_.size() != 4)
      throw std::runtime_error("this node is wired for exactly 4 cameras");

    build_tracker();

    // IMAGE QoS: best-effort is right on the rig and wrong on a replay.
    //
    // Live, a slow VO must never back-pressure the camera, so SensorDataQoS (BEST_EFFORT,
    // KeepLast) is correct. On bag replay it silently costs about a sixth of the run:
    // `ros2 bag play` publishes RELIABLE (the bag's offered_qos_profiles is empty, so
    // rosbag2 uses its default), a BEST_EFFORT reader matches it but tells DDS not to
    // retransmit, and a 1456x1088 mono image is ~1.5 MB - far over the UDP datagram limit,
    // so every sample is fragmented and losing one fragment loses the whole frame. The
    // orphaned partners then pair across trigger edges and die on the 1 ms skew gate.
    // Measured before this parameter existed: 81-84 % of sets reached the matcher at 0.25x,
    // 0.5x and 1.0x alike - rate was never the variable (5.10).
    const std::string qos_mode = declare_parameter<std::string>("image_qos", "sensor_data");
    const int qos_depth = declare_parameter<int>("image_qos_depth", 10);
    rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(qos_depth)));
    if (qos_mode == "reliable") {
      qos.reliable();
      RCLCPP_INFO(get_logger(), "image QoS RELIABLE depth %d — for bag replay, where losing "
                  "a fragment loses a whole 1.5 MB frame. Do NOT use on the live rig: it lets "
                  "a slow tracker back-pressure the camera.", qos_depth);
    } else {
      qos.best_effort();
      RCLCPP_INFO(get_logger(), "image QoS BEST_EFFORT depth %d (live-rig default)", qos_depth);
    }
    for (size_t i = 0; i < 4; ++i)
      subs_[i] = create_subscription<Img>(topics_[i], qos,
          [this, i](const Img::ConstSharedPtr msg) { on_frame(i, msg); });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("cuvslam/odometry", 10);
    if (enable_slam_) {
      slam_pub_ = create_publisher<nav_msgs::msg::Odometry>("cuvslam/slam_odometry", 10);
      // Latched: a loop closure is a rare event, and a viewer or recorder attaching later
      // must still see the last one rather than wait for the next.
      lc_pub_ = create_publisher<nav_msgs::msg::Path>(
          "cuvslam/loop_closures", rclcpp::QoS(10).transient_local());
      slam_path_pub_ = create_publisher<nav_msgs::msg::Path>(
          "cuvslam/slam_path", rclcpp::QoS(2).transient_local());
      lc_edge_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
          "cuvslam/loop_closure_edges", rclcpp::QoS(10).transient_local());
    }
    if (publish_landmarks_)
      cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("cuvslam/landmarks", 10);
    if (publish_observations_)
      obs_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("cuvslam/observations", 10);
    tf_bc_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    RCLCPP_INFO(get_logger(), "cuVSLAM multicam VO up: 4 fisheyes -> %zu virtual pinholes, "
                "mode=Multicamera (visual only), sets gated at %.1f ms skew on real per-frame "
                "timestamps.", vpin_.size(), max_skew_ns_ / 1e6);
    // Say the frame out loud. Anyone reading this trajectory in rviz or fusing it with
    // the IMU needs to know it is not a vehicle frame, and the pose alone will not tell
    // them - a 180 deg roll still traces a plausible path.
    RCLCPP_INFO(get_logger(),
                "odometry frame: %s -> %s. The child frame is CAM1'S OPTICAL FRAME "
                "(z forward, x right, y down), rolled 180 deg by the inverted mount. "
                "It is NOT base_link and NOT FLU. Translation magnitudes and "
                "return-to-origin drift are frame-independent and unaffected; anything "
                "wanting a vehicle frame must compose R_body_from_cam1, which is not yet "
                "measured (config/rig/rig_layout.yaml, 3R.16b).",
                odom_frame_.c_str(), base_frame_.c_str());
  }

 private:
  using Img = sensor_msgs::msg::Image;

  void build_tracker() {
    int major, minor, patch;
    cuvslam::GetVersion(&major, &minor, &patch);
    RCLCPP_INFO(get_logger(), "cuVSLAM %d.%d.%d — warming up GPU...", major, minor, patch);
    cuvslam::WarmUpGPU();

    // Carve each fisheye into two virtual pinholes. cuVSLAM cannot consume the raw
    // cameras at all: its only fisheye model is equidistant, capped below 180 deg, and
    // these lenses fit ~192 deg. See virtual_pinhole.hpp.
    //
    // Built by bev_cuvslam/rig_build.hpp, SHARED with the fused node - see the note there
    // on why this is one implementation and not two.
    auto vrig = bev_cuvslam::BuildVirtualRig(calib_dir_, rig_path_, vstereo_path_,
                                             {cams_[0], cams_[1], cams_[2], cams_[3]});
    for (const auto& w : vrig.warnings) RCLCPP_WARN(get_logger(), "%s", w.c_str());
    vpin_ = std::move(vrig.vpin);
    vsrc_ = std::move(vrig.vsrc);
    cuvslam::Rig rig = std::move(vrig.rig);
    for (size_t k = 0; k < vpin_.size(); ++k)
      RCLCPP_INFO(get_logger(), "  vcam %zu = %s %+.0f deg: %dx%d f=%.1f", k,
                  cams_[vsrc_[k]].c_str(), vpin_[k].yaw_rad * 180.0 / CV_PI,
                  vpin_[k].width, vpin_[k].height, vpin_[k].focal);

    // No IMU in multicam mode (cuVSLAM v15 limitation).

    cuvslam::Odometry::Config cfg = cuvslam::Odometry::GetDefaultConfig();
    cfg.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
    cfg.multicam_mode = cuvslam::Odometry::MulticameraMode::Precision;
    cfg.use_gpu = true;
    // REPRODUCIBILITY, and why this flag is NOT the fix.
    //
    // async_sba runs bundle adjustment on a background thread, so how many iterations land
    // between two frames depends on wall-clock arrival. Replaying run1_motion at 0.4x and
    // 0.2x - both essentially lossless, 1153 and 1155 of 1155 sets - gave trajectories a
    // median 1.13 m apart. That is LARGER than the 0.70 m difference between pure VO and
    // the loop-closed trajectory, so no offline A/B of the two means anything.
    //
    // Turning it off does not fix it and makes it worse: synchronous BA is slower, so the
    // node drops MORE sets (1022 vs 1153 at 0.4x) and drops different ones, and the two
    // rates then diverged by a median 3.28 m. The frame-drop difference is the dominant
    // term, not the BA threading.
    //
    // The real fix is to stop dropping frames at all, which means taking the wall clock out
    // of the loop: read the bag in-process and call Track() per set, with no DDS and no
    // real-time coupling (retarget-vo-to-imx296-rig 5.10c). Until that exists, treat any
    // offline difference smaller than ~1 m as noise. Default stays ON, matching the library
    // and the live rig.
    cfg.async_sba = declare_parameter<bool>("async_sba", true);
    // Leave rectified_stereo_camera FALSE. Setting it swaps in the horizontal-only
    // tracker, which cannot move vertically, and demands that paired cameras have
    // identical rotation matrices to 1e-6 - our facing pinholes sit 1.0-1.4 deg apart.
    // The default 2D LK tracker absorbs that residual instead.
    cfg.rectified_stereo_camera = false;
    // Export the accumulated 3D landmark map so the node can publish it as a point cloud
    // (GetFinalLandmarks, odometry start frame). The header warns export costs time and
    // memory, so it is off by the default config and only worth it for visualisation runs.
    cfg.enable_final_landmarks_export = publish_landmarks_;
    // Per the API header, the final-landmarks flag already implies observations export.
    cfg.enable_observations_export = publish_observations_;
    // Slam::Track takes Odometry::State, and GetState() THROWS unless export is on. So
    // enabling SLAM forces the export we otherwise keep off for rate measurements - that
    // cost is the reason enable_slam defaults to false, not an oversight.
    state_readable_ = enable_slam_ || publish_observations_ || publish_landmarks_;
    if (enable_slam_) {
      cfg.enable_observations_export = true;
      cfg.enable_landmarks_export = true;
    }
    // cuVSLAM's own debug facility: every Track() call's images plus the rig config are
    // written here in edex format, which its offline tools read. Off unless asked — it
    // writes a PNG per virtual camera per frame, so it fills a disk at 8 cameras x 20 Hz.
    if (!debug_dump_dir_.empty()) {
      cfg.debug_dump_directory = debug_dump_dir_;
      RCLCPP_WARN(get_logger(), "cuVSLAM edex debug dump ENABLED -> %s (8 images per set, "
                  "this will fill the disk)", debug_dump_dir_.c_str());
    }
    tracker_ = std::make_unique<cuvslam::Odometry>(rig, cfg);

    if (enable_slam_) {
      cuvslam::Slam::Config sc = cuvslam::Slam::GetDefaultConfig();
      sc.use_gpu = true;
      // Left at the library default for the same reason as async_sba: running SLAM inline
      // slows Track() and costs frames, which moves the result more than the threading does.
      sc.sync_mode = declare_parameter<bool>("slam_sync_mode", false);
      sc.enable_reading_internals = true;   // pose graph + loop-closure layers
      sc.map_cache_path = slam_map_path_;   // empty = in memory only
      sc.throttling_time_ms = static_cast<uint32_t>(slam_throttling_ms_);
      sc.max_map_size = static_cast<uint32_t>(slam_max_map_size_);
      // Every virtual pinhole is primary, matching MulticameraMode::Precision above
      // ("all cameras are primary"). Anything narrower would quietly change which views
      // can close a loop.
      std::vector<uint8_t> primary(vpin_.size());
      for (size_t i = 0; i < vpin_.size(); ++i) primary[i] = static_cast<uint8_t>(i);
      slam_ = std::make_unique<cuvslam::Slam>(rig, primary, sc);
      slam_->EnableReadingData(cuvslam::Slam::DataLayer::LoopClosure, 4096);
      slam_->EnableReadingData(cuvslam::Slam::DataLayer::PoseGraph, 4096);
      RCLCPP_INFO(get_logger(), "SLAM ON: pose graph + loop closure over %zu primary cameras "
                  "(max_map_size %d, throttle %d ms, map %s). /cuvslam/slam_odometry carries "
                  "the corrected pose; /cuvslam/odometry stays PURE VO so the section-5 drift "
                  "numbers remain comparable.",
                  primary.size(), slam_max_map_size_, slam_throttling_ms_,
                  slam_map_path_.empty() ? "in memory" : slam_map_path_.c_str());
    }
  }

  // Keep a short history per camera; camera 0 arriving tries to form a set from it.
  // Matching is by TIMESTAMP, never by arrival order — the streams are separate DDS
  // subscriptions and their delivery order says nothing about which trigger edge a frame
  // came from.
  void on_frame(size_t idx, const Img::ConstSharedPtr& m) {
    std::array<Img::ConstSharedPtr, 4> msgs;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      auto& h = hist_[idx];
      h.push_back(m);
      while (h.size() > history_) h.pop_front();

      // ANCHOR ON CAM1'S OLDEST BUFFERED FRAME, NOT ON CAM1'S ARRIVAL.
      //
      // This matched on arrival until 2026-09-01 (`if (idx != 0) return;` and t0 from the
      // just-received message), which races DDS delivery order: when cam1's frame for a
      // trigger edge lands first, the SAME-EDGE frames from cam2..4 have not been
      // delivered yet, so the nearest candidate in their history is the PREVIOUS edge.
      // Every set was then rejected at exactly one frame period - 33.3 ms at 30 Hz -
      // while the capture node reported 8 us of real skew on the very same frames. The
      // gate was right and the hardware was right; the matching was wrong, and it looked
      // exactly like a dead trigger.
      //
      // So: take the OLDEST cam1 frame as the anchor, and only match once every other
      // camera has delivered a frame at or after it. That is the proof that no better
      // candidate can still arrive. The anchor is consumed either way, matched or
      // dropped, so a genuinely unpaired frame cannot wedge the queue.
      if (hist_[0].empty()) return;
      const int64_t t0 = rclcpp::Time(hist_[0].front()->header.stamp).nanoseconds();
      for (size_t i = 1; i < 4; ++i) {
        if (hist_[i].empty()) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no frame yet from %s",
                               cams_[i].c_str());
          return;
        }
        if (rclcpp::Time(hist_[i].back()->header.stamp).nanoseconds() < t0) return;
      }

      msgs[0] = hist_[0].front();
      hist_[0].pop_front();
      int64_t lo = t0, hi = t0;
      for (size_t i = 1; i < 4; ++i) {
        int64_t best_d = INT64_MAX;
        for (const auto& cand : hist_[i]) {
          const int64_t t = rclcpp::Time(cand->header.stamp).nanoseconds();
          if (std::llabs(t - t0) < best_d) { best_d = std::llabs(t - t0); msgs[i] = cand; }
        }
        const int64_t t = rclcpp::Time(msgs[i]->header.stamp).nanoseconds();
        lo = std::min(lo, t);
        hi = std::max(hi, t);
      }
      const int64_t skew = hi - lo;
      ++sets_;
      if (skew > max_skew_ns_) {
        ++dropped_sets_;
        // Do not widen the window and do not re-stamp: on a triggered rig this means the
        // trigger, a camera, or the capture node is at fault, and the VO cannot fix it.
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "set skew %.1f ms > %.1f ms — dropped (%ld of %ld). Is the trigger running?",
            skew / 1e6, max_skew_ns_ / 1e6, dropped_sets_, sets_);
        return;
      }
      if (skew > worst_skew_ns_) worst_skew_ns_ = skew;
    }
    report();
    track_and_publish(msgs);
  }

  void report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report_ < std::chrono::seconds(5)) return;
    RCLCPP_INFO(get_logger(), "sets %ld, worst skew %.0f us in the last window, %ld dropped total, "
                "remap %ld us for %zu virtual cameras, Track() mean %ld us / max %ld us, "
                "worst saturation %.0f%%",
                sets_, worst_skew_ns_ / 1e3, dropped_sets_, remap_us_, vpin_.size(),
                track_n_ ? track_us_sum_ / track_n_ : 0, track_us_max_, 100.0 * sat_worst_);
    if (kf_n_ || nkf_n_) {
      const double kf_frac = 100.0 * kf_n_ / std::max<int64_t>(1, kf_n_ + nkf_n_);
      RCLCPP_INFO(get_logger(), "  Track(): keyframe %ld us over %ld frames (%.0f%%), "
                  "non-keyframe %ld us over %ld",
                  kf_n_ ? kf_us_sum_ / kf_n_ : 0, kf_n_, kf_frac,
                  nkf_n_ ? nkf_us_sum_ / nkf_n_ : 0, nkf_n_);
      // A KEYFRAME FRACTION NEAR 100% IS TRACKING DISTRESS, not a busy map. The tracker
      // declares a keyframe when it cannot carry on against the existing landmarks, so
      // "every frame is a keyframe" means it is re-anchoring constantly. Measured on run1:
      // 24-54% for the healthy stretch, then 100% at t=45 s - exactly the saturated window -
      // and still 94-97% after. It also triples the cost, since a keyframe triangulates,
      // adds to the map and triggers SBA while a normal frame only solves PnP.
      if (kf_frac > 90.0)
        RCLCPP_WARN(get_logger(), "  %.0f%% of frames are KEYFRAMES — the tracker is "
                    "re-anchoring on nearly every frame, which is what tracking distress "
                    "looks like from the inside. Check the saturation line above.", kf_frac);
    }
    track_us_sum_ = 0; track_us_max_ = 0; track_n_ = 0;
    kf_us_sum_ = 0; kf_n_ = 0; nkf_us_sum_ = 0; nkf_n_ = 0;
    if (slam_)
      RCLCPP_INFO(get_logger(), "  SLAM: %ld loop closures, %ld pose-graph optimisations",
                  lc_events_, pgo_events_);
    worst_skew_ns_ = 0;
    sat_worst_ = 0.0;
    last_report_ = now;
  }

  void track_and_publish(const std::array<Img::ConstSharedPtr, 4>& msgs) {
    std::vector<cv_bridge::CvImageConstPtr> holds(4);  // keep source buffers alive
    for (uint32_t i = 0; i < 4; ++i) holds[i] = cv_bridge::toCvShare(msgs[i], "mono8");

    // Remap each fisheye into its two virtual pinholes. The maps are built once at
    // startup; this is a fixed-point bilinear gather, the cheapest form of remap.
    const auto t_remap = std::chrono::steady_clock::now();
    cuvslam::Odometry::ImageSet images;
    images.reserve(vpin_.size());
    for (size_t k = 0; k < vpin_.size(); ++k) {
      cv::remap(holds[vsrc_[k]]->image, vimg_[k], vpin_[k].map1, vpin_[k].map2, cv::INTER_LINEAR);
      cuvslam::Image im{};
      im.pixels = vimg_[k].data;
      im.width = vimg_[k].cols;
      im.height = vimg_[k].rows;
      im.pitch = static_cast<int32_t>(vimg_[k].step);
      im.encoding = cuvslam::ImageData::Encoding::MONO;
      im.data_type = cuvslam::ImageData::DataType::UINT8;
      im.is_gpu_mem = false;
      // Each virtual camera inherits the exposure-midpoint stamp of the fisheye it was
      // carved from. The set already passed the skew gate, so cuVSLAM's own 1 ms check
      // passes on the real timestamps rather than on a synthesised one.
      im.timestamp_ns = rclcpp::Time(msgs[vsrc_[k]]->header.stamp).nanoseconds();
      im.camera_index = static_cast<uint32_t>(k);
      images.push_back(im);
    }
    remap_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t_remap).count();

    check_exposure(holds);

    cuvslam::PoseEstimate est;
    const auto t_track = std::chrono::steady_clock::now();
    try {
      est = tracker_->Track(images);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Track() failed: %s", e.what());
      return;
    }
    // Track() is the cycle-time budget: the remap is a few ms, this is the rest. Reported so
    // "can this board keep up with 20 Hz" is a measurement rather than an inference from the
    // output rate, which is also capped by the replay rate and by sets lost in transport.
    // Measured 2026-09-06: host 6.5-10 ms mean; TX2 50-90 ms (5.4). Only the TX2 is
    // compute-bound.
    track_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t_track).count();
    track_us_sum_ += track_us_;
    track_us_max_ = std::max(track_us_max_, track_us_);
    ++track_n_;
    // Split the timing by KEYFRAME. Track() does not do the same work every frame: a
    // non-key frame is a PnP solve against the recent landmarks, while a keyframe also
    // triangulates, calls map_.add_keyframe(), and triggers SBA
    // (pipelines/track_online_multi.cpp, the `if (frameState == FrameState::Key)` branch).
    // Reporting one mean over both hides a bimodal distribution and makes the keyframe cost
    // look like jitter.
    if (state_readable_) {
      try {
        cuvslam::Odometry::State st;
        tracker_->GetState(st);
        if (st.keyframe) { kf_us_sum_ += track_us_; ++kf_n_; }
        else             { nkf_us_sum_ += track_us_; ++nkf_n_; }
      } catch (const std::exception&) { state_readable_ = false; }
    }
    if (!est.world_from_rig) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "tracking lost (no pose)");
      return;
    }
    check_pose_health(*est.world_from_rig, msgs[0]->header.stamp);
    publish(*est.world_from_rig, msgs[0]->header.stamp);
    if (slam_) {
      slam_track(msgs[0]->header.stamp);
      // The optimised trajectory has to go out on a CADENCE, not only when a loop closes.
      // It was published from publish_loop_closures() alone, so /cuvslam/slam_path always
      // ended at the LAST CLOSURE rather than at the end of the run - 41.6 s of a 54 s run
      // on one replay, 37.4 s on another, each exactly its final closure. It read as SLAM
      // giving up mid-run, and it made an optimised-vs-VO path comparison meaningless
      // because the two covered different intervals.
      if (++slam_path_countdown_ >= slam_path_every_) {
        slam_path_countdown_ = 0;
        // Path AND edges together, from the same graph state. Publishing the edges only on
        // closures left them describing a graph up to 18 s older than the path they were
        // drawn against, so they no longer lay on it.
        publish_slam_path(msgs[0]->header.stamp);
        // ReadPoseGraph() plus the edge scan is far more expensive than GetAllSlamPoses(),
        // and doing both every 20 sets cost 124 frames (1031 against 1153 at the same rate).
        // Dropped frames put GAPS in the optimised path, and a gap drawn as a chord looks
        // exactly like a jump - which is what sent us looking for a bug that was not there.
        // The graph changes only on a closure, so a fifth of the rate loses nothing.
        if (++edge_countdown_ >= 5) { edge_countdown_ = 0; publish_loop_edges(msgs[0]->header.stamp); }
      }
    }
    if (publish_landmarks_ && landmark_stride_ > 0 && (sets_ % landmark_stride_) == 0)
      publish_landmarks(msgs[0]->header.stamp);
    if (publish_observations_)
      publish_observations(msgs[0]->header.stamp);
  }

  void publish_observations(const builtin_interfaces::msg::Time& stamp) {
    // The 2D features cuVSLAM actually tracked this frame, per virtual camera. Packed as
    // one cloud: x=u, y=v, z=virtual camera index, plus the landmark id for colouring.
    sensor_msgs::msg::PointCloud2 pc;
    pc.header.stamp = stamp;
    pc.header.frame_id = odom_frame_;
    pc.height = 1;
    pc.is_dense = true;
    sensor_msgs::PointCloud2Modifier mod(pc);
    mod.setPointCloud2Fields(4,
                             "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                             "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                             "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                             "id", 1, sensor_msgs::msg::PointField::FLOAT32);
    std::vector<cuvslam::Observation> all;
    for (uint32_t ci = 0; ci < vpin_.size(); ++ci) {
      const auto obs = tracker_->GetLastObservations(ci);
      all.insert(all.end(), obs.begin(), obs.end());
    }
    mod.resize(all.size());
    sensor_msgs::PointCloud2Iterator<float> ix(pc, "x"), iy(pc, "y"), iz(pc, "z"),
        iid(pc, "id");
    for (const auto& o : all) {
      ix[0] = o.u; iy[0] = o.v; iz[0] = static_cast<float>(o.camera_index);
      iid[0] = static_cast<float>(o.id & 0xFFFFFFu);   // low bits are enough to colour by
      ++ix; ++iy; ++iz; ++iid;
    }
    obs_pub_->publish(pc);
  }

  void publish_landmarks(const builtin_interfaces::msg::Time& stamp) {
    // GetFinalLandmarks: the whole map, id -> xyz, already in the odometry start frame,
    // so it drops straight into a cloud on odom_frame_ with no extra transform.
    const auto lms = tracker_->GetFinalLandmarks();
    sensor_msgs::msg::PointCloud2 pc;
    pc.header.stamp = stamp;
    pc.header.frame_id = odom_frame_;
    pc.height = 1;
    pc.is_dense = true;
    sensor_msgs::PointCloud2Modifier mod(pc);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(lms.size());
    sensor_msgs::PointCloud2Iterator<float> ix(pc, "x"), iy(pc, "y"), iz(pc, "z");
    for (const auto& kv : lms) {
      ix[0] = kv.second[0]; iy[0] = kv.second[1]; iz[0] = kv.second[2];
      ++ix; ++iy; ++iz;
    }
    cloud_pub_->publish(pc);
  }

  // THE FAULT cuVSLAM DOES NOT HAVE A NAME FOR: a correctly exposed frame and a blown one
  // are the same thing to it.
  //
  // Its Track() contract says "if after several calls visual odometry is not able to recover,
  // then invalid pose will be returned" — but that path is only reached when the solve FAILS
  // (multi_visual_odometry_base.cpp returns false and cuvslam2.cpp hands back an empty
  // world_from_rig). On the 2026-09-06 run1 the solve kept succeeding on a saturated view and
  // returned a zero delta, so nothing in the library ever considered it a fault. There is no
  // image-quality field in Config, no quality output in PoseEstimate, and the only quality
  // signal at all is the covariance — which by then is already garbage.
  //
  // So the rig has to notice for itself. Exposure here is the STM32 trigger pulse width, not
  // Argus AE (AE is locked on purpose: under external trigger it cannot reach its actuator and
  // hunts on gain, 4.7). That makes this a fault the OPERATOR can act on and the software
  // cannot: the fix is the pulse width, or the route.
  //
  // Cost: one sample every 8th pixel each way, so 1/64 of the frame, ~25k reads per camera.
  void check_exposure(const std::vector<cv_bridge::CvImageConstPtr>& holds) {
    double worst = 0.0;
    size_t worst_cam = 0;
    for (size_t i = 0; i < holds.size(); ++i) {
      const cv::Mat& im = holds[i]->image;
      size_t hot = 0, n = 0;
      for (int y = 0; y < im.rows; y += 8) {
        const uint8_t* row = im.ptr<uint8_t>(y);
        for (int x = 0; x < im.cols; x += 8, ++n)
          if (row[x] >= sat_level_) ++hot;
      }
      const double frac = n ? static_cast<double>(hot) / n : 0.0;
      if (frac > worst) { worst = frac; worst_cam = i; }
    }
    if (worst >= sat_warn_frac_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "%s is %.0f%% saturated at/above %d — the scene is brighter than the trigger pulse "
          "width can hold. Features die here and the pose will freeze, then jump. Shorten the "
          "pulse (j106-trigctl.py), not the AE: AE is locked under external trigger and "
          "cannot fix this.",
          cams_[worst_cam].c_str(), 100.0 * worst, sat_level_);
    }
    sat_worst_ = std::max(sat_worst_, worst);
  }

  // A pose cuVSLAM returns is not the same thing as a pose it MEASURED.
  //
  // On the 2026-09-06 run1 replay (5.0g) the rig walked into a blank white wall: fixed
  // 4.986 ms exposure, frame mean luma 102 -> 224, and 88 % of the image left with zero
  // local 16x16 contrast. cuVSLAM then returned the SAME pose twelve sets running — the
  // last one it was sure of — with the covariance climbing, and then one pose 50 m away
  // (with a NEGATIVE variance on one run) from which it carried on as if nothing had
  // happened. Every one of those passed the `world_from_rig` check above, so the node
  // published a frozen pose and then a teleport as measurements, and nothing said a word.
  //
  // This does not drop or repair anything — a pose the tracker stands behind is still the
  // best estimate available, and silently withholding it would be the same class of bug.
  // It makes the failure audible on the live rig, where nobody is running the offline
  // continuity check in scripts/vo/analyze_motion.py.
  void check_pose_health(const cuvslam::PoseWithCovariance& pwc,
                         const builtin_interfaces::msg::Time& stamp) {
    const auto& t = pwc.pose.translation;
    const int64_t now_ns = rclcpp::Time(stamp).nanoseconds();
    // A negative variance on the diagonal is not a large uncertainty, it is a broken
    // solve: report it whatever the pose looks like.
    for (int i = 0; i < 6; ++i) {
      if (pwc.covariance_xyz_rpy[i * 6 + i] < 0.0) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
            "pose covariance diagonal %d is NEGATIVE (%.3g) — the solve broke, this pose "
            "and the ones after it are not trustworthy", i, pwc.covariance_xyz_rpy[i * 6 + i]);
        break;
      }
    }
    if (have_last_pose_) {
      const double d = std::sqrt(std::pow(t[0] - last_t_[0], 2) +
                                 std::pow(t[1] - last_t_[1], 2) +
                                 std::pow(t[2] - last_t_[2], 2));
      const double dt = (now_ns - last_pose_ns_) * 1e-9;
      if (d == 0.0) {
        // Bit-identical translation is the tracker repeating itself, not a rig at rest:
        // a stationary rig still jitters in the last decimal.
        ++frozen_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "pose has not moved at all for %ld sets — cuVSLAM is repeating its last "
            "estimate, not measuring. Featureless view (blank wall, blown highlights)?",
            frozen_);
      } else {
        if (frozen_ >= frozen_warn_sets_)
          RCLCPP_WARN(get_logger(), "pose moving again after %ld frozen sets", frozen_);
        frozen_ = 0;
      }
      if (dt > 0.0 && d / dt > max_speed_mps_) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
            "pose JUMPED %.2f m in %.0f ms (%.0f m/s, limit %.1f) — tracking was lost and "
            "re-initialised somewhere else. Everything downstream of here is in a new frame.",
            d, dt * 1e3, d / dt, max_speed_mps_);
      }
    }
    last_t_ = {t[0], t[1], t[2]};
    last_pose_ns_ = now_ns;
    have_last_pose_ = true;
  }

  // Hand the tracker's state to SLAM and publish the corrected pose beside the raw VO one.
  // Never in place of it: section 5's drift and scale figures are measured on pure VO, and
  // silently swapping the topic's meaning would invalidate every one of them.
  void slam_track(const builtin_interfaces::msg::Time& stamp) {
    try {
      cuvslam::Odometry::State st;
      tracker_->GetState(st);
      slam_->Track(st);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "SLAM Track() failed: %s", e.what());
      return;
    }
    const cuvslam::Pose p = slam_->GetPose();
    nav_msgs::msg::Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = odom_frame_;
    od.child_frame_id = base_frame_;
    od.pose.pose.position.x = p.translation[0];
    od.pose.pose.position.y = p.translation[1];
    od.pose.pose.position.z = p.translation[2];
    od.pose.pose.orientation.x = p.rotation[0];
    od.pose.pose.orientation.y = p.rotation[1];
    od.pose.pose.orientation.z = p.rotation[2];
    od.pose.pose.orientation.w = p.rotation[3];
    slam_pub_->publish(od);

    cuvslam::Slam::Metrics m{};
    try {
      slam_->GetSlamMetrics(m);
    } catch (const std::exception&) {
      return;
    }
    // lc_status is a level, not an edge: count the RISING edge so "12 loop closures" means
    // twelve events rather than however many frames the flag happened to stay up for.
    if (m.lc_status && !lc_prev_) {
      ++lc_events_;
      RCLCPP_INFO(get_logger(), "LOOP CLOSURE %ld: %u landmarks tracked, %u in PnP, %u good",
                  lc_events_, m.lc_tracked_landmarks_count, m.lc_pnp_landmarks_count,
                  m.lc_good_landmarks_count);
      publish_loop_closures(stamp);
    }
    lc_prev_ = m.lc_status;
    if (m.pgo_status) ++pgo_events_;
  }

  // Where the loop closed. GetLoopClosurePoses returns a rolling last-10 window, so the
  // events have to be ACCUMULATED and de-duplicated on timestamp - taking the latest
  // message loses older closures and double-counts the ones still inside the window.
  // Same approach as cuVSLAM's own euroc example (reported_loop_closures).
  void publish_loop_closures(const builtin_interfaces::msg::Time& stamp) {
    std::vector<cuvslam::PoseStamped> poses;
    try {
      slam_->GetLoopClosurePoses(poses);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "GetLoopClosurePoses failed: %s", e.what());
      return;
    }
    for (const auto& ps : poses) {
      if (!lc_seen_.insert(ps.timestamp_ns).second) continue;   // already reported
      // A Path, not a PoseArray: PoseArray carries ONE header stamp for the whole array, so
      // a consumer cannot tell WHEN each closure happened and can only place the marker by
      // nearest-point search. Path stamps every pose, so the viewer can put each marker on
      // the optimised trajectory at its own instant.
      geometry_msgs::msg::PoseStamped q;
      q.header.frame_id = odom_frame_;
      q.header.stamp = rclcpp::Time(ps.timestamp_ns);
      q.pose.position.x = ps.pose.translation[0];
      q.pose.position.y = ps.pose.translation[1];
      q.pose.position.z = ps.pose.translation[2];
      q.pose.orientation.x = ps.pose.rotation[0];
      q.pose.orientation.y = ps.pose.rotation[1];
      q.pose.orientation.z = ps.pose.rotation[2];
      q.pose.orientation.w = ps.pose.rotation[3];
      lc_accum_.push_back(q);
    }
    nav_msgs::msg::Path pa;
    pa.header.stamp = stamp;
    pa.header.frame_id = odom_frame_;
    pa.poses = lc_accum_;
    lc_pub_->publish(pa);
  }

  // THE OPTIMISED TRAJECTORY, not the stream of GetPose() values.
  //
  // /cuvslam/slam_odometry is the current corrected pose, and accumulating it into a line
  // is wrong: a loop closure re-optimises the WHOLE graph, so every earlier point in such a
  // line is stale and the line steps at each closure. cuVSLAM's own app says so outright -
  // "if slam is enabled, overwrite all slam poses in the end after LCs and PGOs" - and
  // re-reads get_all_slam_poses() rather than keeping what it accumulated
  // (tools/cuvslam_app/cuvslam_app.py). GetAllSlamPoses() returns the whole trajectory as
  // currently optimised, which is globally consistent and has no steps in it.
  void publish_slam_path(const builtin_interfaces::msg::Time& stamp) {
    std::vector<cuvslam::PoseStamped> poses;
    try {
      slam_->GetAllSlamPoses(poses);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "GetAllSlamPoses failed: %s", e.what());
      return;
    }
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = odom_frame_;
    path.poses.reserve(poses.size());
    for (const auto& ps : poses) {
      geometry_msgs::msg::PoseStamped p;
      p.header.frame_id = odom_frame_;
      p.header.stamp = rclcpp::Time(ps.timestamp_ns);
      p.pose.position.x = ps.pose.translation[0];
      p.pose.position.y = ps.pose.translation[1];
      p.pose.position.z = ps.pose.translation[2];
      p.pose.orientation.x = ps.pose.rotation[0];
      p.pose.orientation.y = ps.pose.rotation[1];
      p.pose.orientation.z = ps.pose.rotation[2];
      p.pose.orientation.w = ps.pose.rotation[3];
      path.poses.push_back(p);
    }
    slam_path_pub_->publish(path);
  }

  // The loop-closure EDGES: which pose was matched to which earlier one. This is the thing
  // that makes a closure legible - a marker on its own says a loop closed, an edge says
  // where it closed BACK TO. cuVSLAM leaves this as a "future extension" in the euroc
  // example (ReadPoseGraph is commented out there), so the reading of it is ours: a graph
  // edge whose two node ids are not adjacent is not a sequential odometry link.
  void publish_loop_edges(const builtin_interfaces::msg::Time& stamp) {
    std::shared_ptr<const cuvslam::Slam::PoseGraph> g;
    try {
      g = slam_->ReadPoseGraph();
    } catch (const std::exception&) {
      return;
    }
    if (!g || g->nodes.empty()) return;
    std::unordered_map<uint64_t, const cuvslam::Pose*> by_id;
    for (const auto& n : g->nodes) by_id[n.id] = &n.node_pose;
    geometry_msgs::msg::PoseArray pa;   // consecutive PAIRS: [from, to, from, to, ...]
    pa.header.stamp = stamp;
    pa.header.frame_id = odom_frame_;
    size_t n_loop = 0;
    for (const auto& e : g->edges) {
      const uint64_t lo = std::min(e.node_from, e.node_to), hi = std::max(e.node_from, e.node_to);
      if (hi - lo <= 1) continue;                       // sequential odometry link
      auto a = by_id.find(e.node_from), b = by_id.find(e.node_to);
      if (a == by_id.end() || b == by_id.end()) continue;
      for (const cuvslam::Pose* q : {a->second, b->second}) {
        geometry_msgs::msg::Pose m;
        m.position.x = q->translation[0];
        m.position.y = q->translation[1];
        m.position.z = q->translation[2];
        m.orientation.w = 1.0;
        pa.poses.push_back(m);
      }
      ++n_loop;
    }
    lc_edge_pub_->publish(pa);
    RCLCPP_INFO(get_logger(), "  pose graph: %zu nodes, %zu edges, %zu of them non-sequential "
                "(loop links)", g->nodes.size(), g->edges.size(), n_loop);
  }

  void publish(const cuvslam::PoseWithCovariance& pwc, const builtin_interfaces::msg::Time& stamp) {
    const cuvslam::Pose& p = pwc.pose;
    nav_msgs::msg::Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = odom_frame_;
    od.child_frame_id = base_frame_;
    od.pose.pose.position.x = p.translation[0];
    od.pose.pose.position.y = p.translation[1];
    od.pose.pose.position.z = p.translation[2];
    od.pose.pose.orientation.x = p.rotation[0];
    od.pose.pose.orientation.y = p.rotation[1];
    od.pose.pose.orientation.z = p.rotation[2];
    od.pose.pose.orientation.w = p.rotation[3];
    // Since cuVSLAM v17 the 6x6 covariance is already row-major [x,y,z,Rx,Ry,Rz] (field
    // renamed to covariance_xyz_rpy) — the same order ROS Odometry wants, so copy directly.
    // (Up to v15 it was [Rx,Ry,Rz,x,y,z] and needed a {3,4,5,0,1,2} permutation.)
    for (int i = 0; i < 36; ++i) od.pose.covariance[i] = pwc.covariance_xyz_rpy[i];
    odom_pub_->publish(od);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = odom_frame_;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = p.translation[0];
    tf.transform.translation.y = p.translation[1];
    tf.transform.translation.z = p.translation[2];
    tf.transform.rotation = od.pose.pose.orientation;
    tf_bc_->sendTransform(tf);
  }

  std::string calib_dir_, rig_path_, odom_frame_, base_frame_;
  std::vector<std::string> cams_, topics_;
  std::unique_ptr<cuvslam::Odometry> tracker_;
  std::array<rclcpp::Subscription<Img>::SharedPtr, 4> subs_;
  std::array<std::deque<Img::ConstSharedPtr>, 4> hist_;
  std::vector<bev_cuvslam::VirtualPinhole> vpin_;   // 8: two per fisheye
  std::vector<int> vsrc_;                           // which fisheye feeds each virtual cam
  std::array<cv::Mat, 8> vimg_;                     // remap destinations, reused each set
  int64_t remap_us_ = 0;
  std::string vstereo_path_;
  size_t history_ = 8;
  int64_t max_skew_ns_ = 1000000, worst_skew_ns_ = 0, sets_ = 0, dropped_sets_ = 0;
  // Pose-health state (check_pose_health). max_speed_mps_ is deliberately far above
  // anything the rig does — 5 m/s is a sprint — so it fires on failures, not on fast motion.
  std::array<double, 3> last_t_{};
  int64_t last_pose_ns_ = 0, frozen_ = 0, frozen_warn_sets_ = 3;
  bool have_last_pose_ = false;
  double max_speed_mps_ = 5.0;
  std::unique_ptr<cuvslam::Slam> slam_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr slam_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr lc_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr lc_edge_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr slam_path_pub_;
  std::set<int64_t> lc_seen_;
  std::vector<geometry_msgs::msg::PoseStamped> lc_accum_;
  bool enable_slam_ = false, lc_prev_ = false;
  std::string slam_map_path_, debug_dump_dir_;
  int slam_throttling_ms_ = 0, slam_max_map_size_ = 300;
  int64_t lc_events_ = 0, pgo_events_ = 0;
  // 20 sets = 1 s at the rig's 20 Hz, so the last published path is at most a second short
  // of the end even when the node is SIGKILLed (which the replay wrapper does).
  int slam_path_countdown_ = 0, slam_path_every_ = 20, edge_countdown_ = 0;
  int64_t track_us_ = 0, track_us_sum_ = 0, track_us_max_ = 0, track_n_ = 0;
  int64_t kf_us_sum_ = 0, kf_n_ = 0, nkf_us_sum_ = 0, nkf_n_ = 0;
  bool state_readable_ = false;
  int sat_level_ = 227;
  double sat_warn_frac_ = 0.5, sat_worst_ = 0.0;
  std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
  std::mutex mtx_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obs_pub_;
  bool publish_landmarks_ = false;
  int landmark_stride_ = 3;
  bool publish_observations_ = false;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_bc_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CuvslamMulticamNode>());
  rclcpp::shutdown();
  return 0;
}
