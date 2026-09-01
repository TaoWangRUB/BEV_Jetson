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
// exposure-midpoint timestamp (README 4.7). A set is four frames whose stamps span less
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
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include "cuvslam/cuvslam2.h"
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
cv::Matx44d load_matrix4(const YAML::Node& n) {
  cv::Matx44d M;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) M(i, j) = n[i][j].as<double>();
  return M;
}

// cuVSLAM Pose (quaternion xyzw + translation) from a 4x4. cuVSLAM uses the OpenCV
// convention - x right, y down, z forward - which is what our extrinsics are already in.
cuvslam::Pose pose_from_matrix(const cv::Matx44d& M) {
  cuvslam::Pose p;
  p.translation = {static_cast<float>(M(0,3)), static_cast<float>(M(1,3)), static_cast<float>(M(2,3))};
  const double tr = M(0,0) + M(1,1) + M(2,2);
  double q[4];  // w, x, y, z
  if (tr > 0) {
    const double t = std::sqrt(tr + 1.0) * 2.0;
    q[0] = 0.25*t; q[1] = (M(2,1)-M(1,2))/t; q[2] = (M(0,2)-M(2,0))/t; q[3] = (M(1,0)-M(0,1))/t;
  } else if (M(0,0) > M(1,1) && M(0,0) > M(2,2)) {
    const double t = std::sqrt(1.0 + M(0,0) - M(1,1) - M(2,2)) * 2.0;
    q[0] = (M(2,1)-M(1,2))/t; q[1] = 0.25*t; q[2] = (M(0,1)+M(1,0))/t; q[3] = (M(0,2)+M(2,0))/t;
  } else if (M(1,1) > M(2,2)) {
    const double t = std::sqrt(1.0 + M(1,1) - M(0,0) - M(2,2)) * 2.0;
    q[0] = (M(0,2)-M(2,0))/t; q[1] = (M(0,1)+M(1,0))/t; q[2] = 0.25*t; q[3] = (M(1,2)+M(2,1))/t;
  } else {
    const double t = std::sqrt(1.0 + M(2,2) - M(0,0) - M(1,1)) * 2.0;
    q[0] = (M(1,0)-M(0,1))/t; q[1] = (M(0,2)+M(2,0))/t; q[2] = (M(1,2)+M(2,1))/t; q[3] = 0.25*t;
  }
  p.rotation = {static_cast<float>(q[1]), static_cast<float>(q[2]),
                static_cast<float>(q[3]), static_cast<float>(q[0])};  // x, y, z, w
  return p;
}

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

    if (cams_.size() != 4 || topics_.size() != 4)
      throw std::runtime_error("this node is wired for exactly 4 cameras");

    build_tracker();

    auto qos = rclcpp::SensorDataQoS();
    for (size_t i = 0; i < 4; ++i)
      subs_[i] = create_subscription<Img>(topics_[i], qos,
          [this, i](const Img::ConstSharedPtr msg) { on_frame(i, msg); });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("cuvslam/odometry", 10);
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
    const YAML::Node rig_y = load_yaml(rig_path_);
    const YAML::Node vs_y = load_yaml(vstereo_path_);
    const YAML::Node vp_y = vs_y["virtual_pinhole"];
    const int vw = vp_y["width"].as<int>(), vh = vp_y["height"].as<int>();
    const double vfov = vp_y["fov_deg"].as<double>();

    cuvslam::Rig rig;
    for (size_t i = 0; i < 4; ++i) {
      const auto omni = bev_cuvslam::LoadOmni(calib_dir_ + "/" + cams_[i] + ".yaml");
      if (omni.width != vp_y["source_width"].as<int>(omni.width))
        RCLCPP_WARN(get_logger(), "%s calibrated at %dx%d - check it matches the live rig",
                    cams_[i].c_str(), omni.width, omni.height);
      // rig frame IS cam1's optical frame, which is how rig_in_cam1 is expressed.
      // NOT the FLU body frame of rig_layout.yaml - see the frame note there (3R.16b).
      const cv::Matx44d rig_from_fisheye = load_matrix4(rig_y["rig_in_cam1"][cams_[i]]);
      for (int k = 0; k < 2; ++k) {
        const double yaw = (k == 0 ? -1.0 : +1.0) * CV_PI / 4.0;
        vpin_.push_back(bev_cuvslam::BuildVirtualPinhole(omni, yaw, vw, vh, vfov));
        const double c = std::cos(yaw), sn = std::sin(yaw);
        const cv::Matx44d Ry(c,0,sn,0,  0,1,0,0,  -sn,0,c,0,  0,0,0,1);
        cuvslam::Camera cam;
        cam.size = {vw, vh};
        cam.focal = {static_cast<float>(vpin_.back().focal), static_cast<float>(vpin_.back().focal)};
        cam.principal = {static_cast<float>(vpin_.back().cx), static_cast<float>(vpin_.back().cy)};
        // Pinhole with NO distortion: the remap already removed it. Anything else here
        // would be applying the correction twice.
        cam.distortion.model = cuvslam::Distortion::Model::Pinhole;
        cam.distortion.parameters = {};
        cam.rig_from_camera = pose_from_matrix(rig_from_fisheye * Ry);
        rig.cameras.push_back(cam);
        vsrc_.push_back(i);
        RCLCPP_INFO(get_logger(), "  vcam %zu = %s %+.0f deg: %dx%d f=%.1f",
                    vpin_.size() - 1, cams_[i].c_str(), yaw * 180.0 / CV_PI, vw, vh,
                    vpin_.back().focal);
      }
    }
    // No IMU in multicam mode (cuVSLAM v15 limitation).

    cuvslam::Odometry::Config cfg = cuvslam::Odometry::GetDefaultConfig();
    cfg.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
    cfg.multicam_mode = cuvslam::Odometry::MulticameraMode::Precision;
    cfg.use_gpu = true;
    // Leave rectified_stereo_camera FALSE. Setting it swaps in the horizontal-only
    // tracker, which cannot move vertically, and demands that paired cameras have
    // identical rotation matrices to 1e-6 - our facing pinholes sit 1.0-1.4 deg apart.
    // The default 2D LK tracker absorbs that residual instead.
    cfg.rectified_stereo_camera = false;
    tracker_ = std::make_unique<cuvslam::Odometry>(rig, cfg);
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
      if (idx != 0) return;

      const int64_t t0 = rclcpp::Time(m->header.stamp).nanoseconds();
      msgs[0] = m;
      int64_t lo = t0, hi = t0;
      for (size_t i = 1; i < 4; ++i) {
        int64_t best_d = INT64_MAX;
        for (const auto& cand : hist_[i]) {
          const int64_t t = rclcpp::Time(cand->header.stamp).nanoseconds();
          if (std::llabs(t - t0) < best_d) { best_d = std::llabs(t - t0); msgs[i] = cand; }
        }
        if (!msgs[i]) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no frame yet from %s",
                               cams_[i].c_str());
          return;
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
                "remap %ld us for %zu virtual cameras",
                sets_, worst_skew_ns_ / 1e3, dropped_sets_, remap_us_, vpin_.size());
    worst_skew_ns_ = 0;
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

    cuvslam::PoseEstimate est;
    try {
      est = tracker_->Track(images);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Track() failed: %s", e.what());
      return;
    }
    if (!est.world_from_rig) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "tracking lost (no pose)");
      return;
    }
    publish(*est.world_from_rig, msgs[0]->header.stamp);
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
    // cuVSLAM 6x6 covariance is row-major in order [Rx,Ry,Rz,x,y,z]; ROS Odometry wants
    // [x,y,z,Rx,Ry,Rz]. Remap with perm[ros]=cuvslam index = {3,4,5,0,1,2}.
    static constexpr int perm[6] = {3, 4, 5, 0, 1, 2};
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        od.pose.covariance[r * 6 + c] = pwc.covariance[perm[r] * 6 + perm[c]];
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
  std::vector<size_t> vsrc_;                        // which fisheye feeds each virtual cam
  std::array<cv::Mat, 8> vimg_;                     // remap destinations, reused each set
  int64_t remap_us_ = 0;
  std::string vstereo_path_;
  size_t history_ = 8;
  int64_t max_skew_ns_ = 1000000, worst_skew_ns_ = 0, sets_ = 0, dropped_sets_ = 0;
  std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
  std::mutex mtx_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_bc_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CuvslamMulticamNode>());
  rclcpp::shutdown();
  return 0;
}
