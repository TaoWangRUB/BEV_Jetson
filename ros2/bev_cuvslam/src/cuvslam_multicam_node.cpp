// cuVSLAM 4-camera (surround fisheye) multicamera visual-odometry node, ROS 2 Foxy.
//
// Runs the CUDA-10.2-ported libcuvslam in OdometryMode::Multicamera (visual only;
// cuVSLAM v15 does NOT fuse an IMU in multicam mode — the IMU is fused externally
// by an EKF). Loads per-camera KANNALA_BRANDT intrinsics (camN.yaml) + the rig
// extrinsics (rig_extrinsics.yaml), builds the cuVSLAM rig, synchronizes 4 image
// topics, calls Track(), and publishes nav_msgs/Odometry + a TF.
//
// NOTE on sync: the IMX219 cameras have no hardware trigger and free-run at slightly
// different rates (measured 4-cam spread ~30–86 ms). cuVSLAM wants <1 ms. We bundle the
// latest frame per camera (driver = index 0 triggers Track) and feed the set one unified
// timestamp so cuVSLAM accepts it. The cameras did NOT capture the same instant, so this
// is the rig's main accuracy limiter under motion — real fix is hardware frame sync.

#include <array>
#include <cstdlib>
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

// Build a cuVSLAM Camera (intrinsics + fisheye distortion) from a KANNALA_BRANDT yaml.
cuvslam::Camera load_intrinsics(const std::string& path) {
  YAML::Node y = load_yaml(path);
  cuvslam::Camera c;
  c.size = {y["image_width"].as<int>(), y["image_height"].as<int>()};
  YAML::Node pp = y["projection_parameters"];
  c.focal = {pp["mu"].as<float>(), pp["mv"].as<float>()};
  c.principal = {pp["u0"].as<float>(), pp["v0"].as<float>()};
  YAML::Node dp = y["distortion_parameters"];
  c.distortion.model = cuvslam::Distortion::Model::Fisheye;  // 4-coeff equidistant = OpenCV fisheye
  c.distortion.parameters = {dp["k2"].as<float>(), dp["k3"].as<float>(),
                             dp["k4"].as<float>(), dp["k5"].as<float>()};
  return c;
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
    calib_dir_ = declare_parameter<std::string>("calib_dir", "scripts/config/calib");
    rig_path_ = declare_parameter<std::string>("rig_extrinsics", "config/rig/rig_extrinsics.yaml");
    cams_ = declare_parameter<std::vector<std::string>>("cameras", {"cam1", "cam2", "cam3", "cam4"});
    topics_ = declare_parameter<std::vector<std::string>>(
        "image_topics", {"/cam1/image_raw", "/cam2/image_raw", "/cam3/image_raw", "/cam4/image_raw"});
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    // Staleness bound for the latest-frame bundler. The IMX219 rig free-runs with no HW
    // trigger; measured 4-cam spread is ~30–86 ms (slowest cam ~14 Hz). 120 ms accepts
    // most sets (~8 Hz odom; 80 ms gave ~5 Hz). Higher = more rate but more inter-camera
    // skew (accuracy cost under motion). Lower it once frames are hardware-synced.
    int slop_ms = declare_parameter<int>("sync_slop_ms", 120);

    if (cams_.size() != 4 || topics_.size() != 4)
      throw std::runtime_error("this node is wired for exactly 4 cameras");

    build_tracker();

    // Sync the 4 image streams (ApproximateTime: IMX219 has no hardware trigger).
    // Latest-frame bundler (the IMX219 rig has no hardware trigger, and ApproximateTime
    // can't reliably match 4 drifting, best-effort, different-rate streams). Cache the
    // newest frame per camera; the driver camera (index 0) triggers a Track() using the
    // most-recent frame from the others, as long as they're within max_stale_ns_.
    max_stale_ns_ = static_cast<int64_t>(slop_ms) * 1000000;
    auto qos = rclcpp::SensorDataQoS();
    for (size_t i = 0; i < 4; ++i)
      subs_[i] = create_subscription<Img>(topics_[i], qos,
          [this, i](const Img::ConstSharedPtr msg) { on_frame(i, msg); });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("cuvslam/odometry", 10);
    tf_bc_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    RCLCPP_INFO(get_logger(), "cuVSLAM multicam VO up: 4 cameras, mode=Multicamera (visual only).");
  }

 private:
  using Img = sensor_msgs::msg::Image;

  void build_tracker() {
    int major, minor, patch;
    cuvslam::GetVersion(&major, &minor, &patch);
    RCLCPP_INFO(get_logger(), "cuVSLAM %d.%d.%d — warming up GPU...", major, minor, patch);
    cuvslam::WarmUpGPU();

    YAML::Node rig_y = load_yaml(rig_path_);
    cuvslam::Rig rig;
    for (size_t i = 0; i < 4; ++i) {
      cuvslam::Camera cam = load_intrinsics(calib_dir_ + "/" + cams_[i] + ".yaml");
      cam.rig_from_camera = load_pose(rig_y["cameras"][cams_[i]]);
      rig.cameras.push_back(cam);
      RCLCPP_INFO(get_logger(), "  %s: %dx%d f=(%.1f,%.1f) c=(%.1f,%.1f)", cams_[i].c_str(),
                  cam.size[0], cam.size[1], cam.focal[0], cam.focal[1], cam.principal[0], cam.principal[1]);
    }
    // No IMU in multicam mode (cuVSLAM v15 limitation).

    cuvslam::Odometry::Config cfg = cuvslam::Odometry::GetDefaultConfig();
    cfg.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
    cfg.multicam_mode = cuvslam::Odometry::MulticameraMode::Precision;
    cfg.use_gpu = true;
    tracker_ = std::make_unique<cuvslam::Odometry>(rig, cfg);
  }

  // Cache every frame; only the driver camera (index 0) triggers a Track().
  void on_frame(size_t idx, const Img::ConstSharedPtr& m) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      latest_[idx] = m;
    }
    if (idx != 0) return;

    std::array<Img::ConstSharedPtr, 4> msgs;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      msgs = latest_;
    }
    const int64_t t0 = rclcpp::Time(m->header.stamp).nanoseconds();
    for (size_t i = 1; i < 4; ++i) {
      if (!msgs[i]) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no frame yet from %s", cams_[i].c_str());
        return;
      }
      const int64_t dt = std::llabs(t0 - rclcpp::Time(msgs[i]->header.stamp).nanoseconds());
      if (dt > max_stale_ns_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "%s stale by %.0f ms (> %ld ms) — skipping set", cams_[i].c_str(), dt / 1e6, max_stale_ns_ / 1000000);
        return;
      }
    }
    track_and_publish(msgs);
  }

  void track_and_publish(const std::array<Img::ConstSharedPtr, 4>& msgs) {
    // The IMX219 rig has no hardware trigger, so ApproximateTime-matched frames are
    // tens of ms apart — but cuVSLAM's Multicamera mode hard-rejects sets whose
    // timestamps differ by >1 ms. Present the matched set under one unified timestamp
    // (cam0's) so cuVSLAM accepts it. Caveat: the cameras did NOT capture the same
    // instant, so cross-camera geometry is skewed under motion (degrades tracking).
    // The real fix is hardware frame sync; this unblocks bring-up. See sync_slop_ms.
    const int64_t base_ts = rclcpp::Time(msgs[0]->header.stamp).nanoseconds();
    std::vector<cv_bridge::CvImageConstPtr> holds(4);  // keep pixel buffers alive during Track()
    cuvslam::Odometry::ImageSet images;
    images.reserve(4);
    for (uint32_t i = 0; i < 4; ++i) {
      holds[i] = cv_bridge::toCvShare(msgs[i], "mono8");
      const cv::Mat& g = holds[i]->image;
      cuvslam::Image im{};
      im.pixels = g.data;
      im.width = g.cols;
      im.height = g.rows;
      im.pitch = static_cast<int32_t>(g.step);
      im.encoding = cuvslam::ImageData::Encoding::MONO;
      im.data_type = cuvslam::ImageData::DataType::UINT8;
      im.is_gpu_mem = false;
      im.timestamp_ns = base_ts;  // unified per-set timestamp (see note above)
      im.camera_index = i;
      images.push_back(im);
    }

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
  std::array<Img::ConstSharedPtr, 4> latest_;
  std::mutex mtx_;
  int64_t max_stale_ns_{120000000};
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_bc_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CuvslamMulticamNode>());
  rclcpp::shutdown();
  return 0;
}
