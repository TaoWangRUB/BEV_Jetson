// bev_ground_stitch — a metric top-down mosaic of the ground around the 4x IMX296 rig.
//
// WHAT MAKES THIS DIFFERENT FROM bev_panorama_node. That node maps each camera onto a
// sphere at infinity using rotation only: it reads q_wxyz and never reads a translation.
// Camera POSITION therefore cancels out of its mapping, so two cameras 150 mm apart
// looking at an object 0.5 m away cannot agree, and the seam parallax is a property of
// the method rather than of the calibration. No recalibration removes it.
//
// Projecting onto a PLANE instead restores the translation to the model. Every output
// cell is a specific point on the ground in the rig frame; it is transformed into each
// camera by the full T_cam_rig and projected through that camera's own model. A point
// that really lies on that plane therefore maps to one output cell no matter which
// camera saw it. That is a construction, not a tuning result.
//
// WHAT IT DOES NOT DO, stated once and not softened anywhere below: a single-surface
// projection is parallax-free ON THAT SURFACE ONLY. A pole standing on the ground has
// its top off the plane, so it smears radially outward from whichever camera saw it, by
// roughly (height_above_plane / rig_height) x (distance from the camera). A 0.3 m box at
// 1.5 m under a 0.25 m rig smears on the order of a metre. That is geometry; it is not a
// seam-blending problem and no weighting fixes it.
//
// FRAMES. The extrinsics in rig_extrinsics_imx296.yaml are solved directly from the
// images argus_capture_node publishes — raw sensor orientation, which on this rig is
// INVERTED. They are therefore already self-consistent with those images and NO 180 deg
// roll is applied here. (bev_panorama_node does apply one, because it consumes a
// different, "nominal image-up" extrinsic lineage. Mixing the two puts every camera
// 180 deg out and still looks plausible.) The rig FLU frame the output is expressed in
// comes from config/rig/ground_plane.yaml, which is also where openspec task 3R.16b's
// missing optical->FLU rotation is written down.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include "bev_ground/omni_model.hpp"

namespace {

constexpr size_t kNCam = 4;

cv::Matx44d LoadMatrix4(const YAML::Node& n) {
  if (!n || n.size() != 4) throw std::runtime_error("expected a 4x4 matrix");
  cv::Matx44d M;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) M(i, j) = n[i][j].as<double>();
  return M;
}

// Inverse of a rigid transform, done as [R^T | -R^T t] rather than a general inverse:
// a general 4x4 solve on an almost-orthonormal block quietly introduces scale.
cv::Matx44d RigidInverse(const cv::Matx44d& T) {
  cv::Matx33d R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R(i, j) = T(i, j);
  const cv::Vec3d t(T(0, 3), T(1, 3), T(2, 3));
  const cv::Matx33d Rt = R.t();
  const cv::Vec3d ti = -(Rt * t);
  cv::Matx44d out = cv::Matx44d::eye();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out(i, j) = Rt(i, j);
    out(i, 3) = ti[i];
  }
  return out;
}

double WrapPi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

}  // namespace

class BevGroundStitchNode : public rclcpp::Node {
 public:
  BevGroundStitchNode() : Node("bev_ground_stitch") {
    calib_dir_ = declare_parameter<std::string>("calib_dir", "config/calib/imx296_1456x1088");
    rig_path_ = declare_parameter<std::string>("rig_extrinsics", "config/rig/rig_extrinsics_imx296.yaml");
    plane_path_ = declare_parameter<std::string>("ground_plane", "config/rig/ground_plane.yaml");
    cams_ = declare_parameter<std::vector<std::string>>("cameras", {"cam1", "cam2", "cam3", "cam4"});
    topics_ = declare_parameter<std::vector<std::string>>(
        "image_topics", {"/cam1/image_raw", "/cam2/image_raw", "/cam3/image_raw", "/cam4/image_raw"});
    // Walking the rig, not walking the names: cam1(FL) -> cam2(FR) -> cam4(BR) -> cam3(BL).
    // cam2 and cam3 are DIAGONAL and share no overlap; treating them as neighbours would
    // put a blend band where there is nothing to blend.
    ring_ = declare_parameter<std::vector<std::string>>("ring_order", {"cam1", "cam2", "cam4", "cam3"});

    // --- the stale-calibration guard (spec: "a stale calibration is refused") ----------
    live_w_ = declare_parameter<int>("image_width", 1456);
    live_h_ = declare_parameter<int>("image_height", 1088);
    live_sensor_ = declare_parameter<std::string>("sensor", "imx296");

    // --- output raster ----------------------------------------------------------------
    res_ = declare_parameter<double>("resolution_m_per_px", 0.01);
    range_f_ = declare_parameter<double>("range_forward_m", 2.0);
    range_b_ = declare_parameter<double>("range_back_m", 2.0);
    range_l_ = declare_parameter<double>("range_left_m", 2.0);
    range_r_ = declare_parameter<double>("range_right_m", 2.0);

    // --- optics gating and blending ---------------------------------------------------
    // 95 deg is the vendor D190 half-angle. The fitted Mei model stays monotonic out to
    // ~119 deg, but the AprilGrid never reached there, so trusting it that far is
    // extrapolation. The node takes the min of this and the model's own limit.
    lens_fov_half_deg_ = declare_parameter<double>("lens_fov_half_deg", 95.0);
    fov_feather_deg_ = declare_parameter<double>("fov_feather_deg", 8.0);
    border_feather_px_ = declare_parameter<double>("border_feather_px", 24.0);
    sector_power_ = declare_parameter<double>("sector_power", 4.0);

    // --- synchronised sets ------------------------------------------------------------
    max_skew_ns_ = static_cast<int64_t>(declare_parameter<int>("max_skew_us", 1000)) * 1000;
    history_ = static_cast<size_t>(declare_parameter<int>("match_history", 8));

    // --- the plane, and the refusal to guess it ---------------------------------------
    allow_unmeasured_ = declare_parameter<bool>("allow_unmeasured_plane", false);
    provisional_h_ = declare_parameter<double>("provisional_height_m", 0.0);

    // --- photometric ------------------------------------------------------------------
    equalize_ = declare_parameter<bool>("equalize_exposure", true);
    gain_smooth_ = declare_parameter<double>("gain_smoothing", 0.9);
    gain_limit_ = declare_parameter<double>("gain_limit", 2.0);

    rig_frame_ = declare_parameter<std::string>("rig_frame", "rig");
    out_frame_ = declare_parameter<std::string>("output_frame", "bev_ground");
    publish_source_ = declare_parameter<bool>("publish_source_mask", true);
    // Set true only if something upstream already rotated the images 180 (csi_sender
    // FLIP=2 does). The capture node does NOT, and the extrinsics assume it does not.
    input_rot180_ = declare_parameter<bool>("input_rotated_180", false);

    if (cams_.size() != kNCam || topics_.size() != kNCam)
      throw std::runtime_error("this node is wired for exactly 4 cameras");

    LoadPlaneAndRigFrame();
    LoadCalibration();
    BuildMaps();

    auto qos = rclcpp::SensorDataQoS();
    for (size_t i = 0; i < kNCam; ++i)
      subs_[i] = create_subscription<Img>(topics_[i], qos,
          [this, i](const Img::ConstSharedPtr msg) { OnFrame(i, msg); });

    pub_ = create_publisher<sensor_msgs::msg::Image>("bev/ground", rclcpp::SensorDataQoS());
    if (publish_source_)
      src_pub_ = create_publisher<sensor_msgs::msg::Image>("bev/ground/source", rclcpp::SensorDataQoS());
    // Latched, so a consumer that starts later still learns the scale and the origin.
    info_pub_ = create_publisher<std_msgs::msg::String>(
        "bev/ground/info", rclcpp::QoS(1).transient_local().reliable());
    PublishInfo();

    static_tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now();
    tf.header.frame_id = rig_frame_;
    tf.child_frame_id = out_frame_;
    // bev_ground keeps the rig's FLU axes and sits at the foot of the perpendicular from
    // the rig origin to the plane, so output z is 0 by construction.
    tf.transform.translation.x = -height_ * n_[0];
    tf.transform.translation.y = -height_ * n_[1];
    tf.transform.translation.z = -height_ * n_[2];
    tf.transform.rotation.w = 1.0;
    static_tf_->sendTransform(tf);

    RCLCPP_INFO(get_logger(),
        "bev_ground_stitch up: %dx%d cells at %.1f mm/px, %.2f m fwd / %.2f m back / "
        "%.2f m left / %.2f m right, plane %s at h=%.3f m.",
        out_w_, out_h_, res_ * 1000.0, range_f_, range_b_, range_l_, range_r_,
        provisional_ ? "PROVISIONAL" : "measured", height_);
    RCLCPP_INFO(get_logger(),
        "pixel -> metres in frame '%s':  x_forward = %.4f - (row+0.5)*%.5f ,  "
        "y_left = %.4f - (col+0.5)*%.5f ,  z = 0",
        out_frame_.c_str(), range_f_, res_, range_l_, res_);
  }

 private:
  using Img = sensor_msgs::msg::Image;

  // ------------------------------------------------------------------------------------
  // Startup: the plane, and the rig frame it is expressed in
  // ------------------------------------------------------------------------------------
  void LoadPlaneAndRigFrame() {
    const YAML::Node y = bev_ground::LoadYaml(plane_path_);
    const YAML::Node rf = y["rig_frame"];
    if (!rf || !rf["R_rig_cam1"])
      throw std::runtime_error(plane_path_ + ": no rig_frame.R_rig_cam1 — the output frame "
                               "is undefined and a BEV in an undefined frame is not a "
                               "measurement");
    const YAML::Node R = rf["R_rig_cam1"];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) R_rig_c1_(i, j) = R[i][j].as<double>();
    // Cheap trap for a hand-edited matrix that is no longer a rotation.
    const cv::Matx33d should_be_I = R_rig_c1_ * R_rig_c1_.t();
    double worst = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        worst = std::max(worst, std::abs(should_be_I(i, j) - (i == j ? 1.0 : 0.0)));
    if (worst > 1e-6 || cv::determinant(R_rig_c1_) < 0.0)
      throw std::runtime_error(plane_path_ + ": R_rig_cam1 is not a rotation (orthonormality "
                               "off by " + std::to_string(worst) + ")");
    if (rf["check"] && rf["check"]["expected_azimuth_deg"]) {
      for (const auto& kv : rf["check"]["expected_azimuth_deg"])
        expect_az_deg_[kv.first.as<std::string>()] = kv.second.as<double>();
      az_tol_deg_ = rf["check"]["azimuth_tolerance_deg"].as<double>(15.0);
    }

    const YAML::Node p = y["plane"];
    const std::string status = p && p["status"] ? p["status"].as<std::string>() : "unmeasured";
    const bool has_height = p && p["height_m"] && !p["height_m"].IsNull();

    if (status == "measured" && has_height) {
      height_ = p["height_m"].as<double>();
      if (p["normal"] && !p["normal"].IsNull())
        for (int i = 0; i < 3; ++i) n_[i] = p["normal"][i].as<double>();
      const double nn = std::sqrt(n_[0]*n_[0] + n_[1]*n_[1] + n_[2]*n_[2]);
      if (nn < 1e-9) throw std::runtime_error(plane_path_ + ": plane.normal is degenerate");
      for (int i = 0; i < 3; ++i) n_[i] /= nn;
      if (n_[2] <= 0.0)
        throw std::runtime_error(plane_path_ + ": plane.normal must point UP (+z in FLU); "
                                 "a downward normal flips the whole projection");
      provisional_ = false;
      RCLCPP_INFO(get_logger(), "ground plane MEASURED: h=%.4f m, normal [%.4f %.4f %.4f], "
                  "method '%s' (%s)", height_, n_[0], n_[1], n_[2],
                  p["method"] ? p["method"].as<std::string>("?").c_str() : "?",
                  p["date"] ? p["date"].as<std::string>("?").c_str() : "?");
      const double tilt = std::acos(std::min(1.0, n_[2])) * 180.0 / M_PI;
      if (p["uncertainty"] && p["uncertainty"]["tilt_deg"] && !p["uncertainty"]["tilt_deg"].IsNull()) {
        const double dt = p["uncertainty"]["tilt_deg"].as<double>();
        const double reach = std::max({range_f_, range_b_, range_l_, range_r_});
        // A tilt error is a height error that grows linearly with distance, so quote it
        // where it is worst rather than at the rig, where it is invisible.
        RCLCPP_INFO(get_logger(), "plane tilt %.2f deg; stated tilt uncertainty %.2f deg "
                    "=> up to %.0f mm of ground-position error at %.1f m from the rig",
                    tilt, dt, 1000.0 * reach * std::tan(dt * M_PI / 180.0), reach);
      }
      return;
    }

    // Not measured. The spec forbids defaulting one, so the only way forward is an
    // explicit provisional height from the operator, and the output says so.
    std::string why = "plane.status is '" + status + "'";
    if (!has_height) why += " and plane.height_m is null";
    if (!allow_unmeasured_)
      throw std::runtime_error(
          plane_path_ + ": " + why + ". Every ground-plane projection depends on it and an "
          "error here is indistinguishable from a camera calibration error, so this node "
          "will not invent one. Measure it (see the method in that file), or — for a bench "
          "look only — set allow_unmeasured_plane:=true AND provisional_height_m:=<metres "
          "from cam1's optical centre to the floor>. Output is then stamped PROVISIONAL "
          "and its scale is not a measurement.");
    if (provisional_h_ <= 0.0)
      throw std::runtime_error(
          plane_path_ + ": allow_unmeasured_plane is set but provisional_height_m is not. "
          "A height still has to come from somewhere; the node will not guess it.");
    height_ = provisional_h_;
    n_ = {0.0, 0.0, 1.0};
    provisional_ = true;
    RCLCPP_WARN(get_logger(), "**********************************************************");
    RCLCPP_WARN(get_logger(), "GROUND PLANE IS PROVISIONAL. %s.", why.c_str());
    RCLCPP_WARN(get_logger(), "Using operator-supplied h=%.3f m and ASSUMING the rig is "
                "level. Scale in this output is not a measurement and the parallax "
                "residual is not attributable to the calibration.", height_);
    RCLCPP_WARN(get_logger(), "**********************************************************");
  }

  // ------------------------------------------------------------------------------------
  // Startup: intrinsics, extrinsics, and the checks that make them refusable
  // ------------------------------------------------------------------------------------
  void LoadCalibration() {
    const YAML::Node rig_y = bev_ground::LoadYaml(rig_path_);
    if (!rig_y["rig_in_cam1"])
      throw std::runtime_error(rig_path_ + ": no rig_in_cam1 block. The pairwise blocks are "
                               "four independent solves; a stitch needs one rigid rig or the "
                               "seams land in four inconsistent places.");

    for (size_t i = 0; i < kNCam; ++i) {
      const std::string path = calib_dir_ + "/" + cams_[i] + ".yaml";
      omni_[i] = bev_ground::LoadOmni(path);

      // Applying a calibration measured at another resolution, on another sensor, or with
      // a model that cannot represent the lens is a slow silent drift, not a crash. Make
      // it a startup error instead.
      if (omni_[i].width != live_w_ || omni_[i].height != live_h_)
        throw std::runtime_error(path + ": calibrated at " + std::to_string(omni_[i].width) +
                                 "x" + std::to_string(omni_[i].height) + " but the live "
                                 "cameras are " + std::to_string(live_w_) + "x" +
                                 std::to_string(live_h_));
      if (!omni_[i].sensor.empty() && !live_sensor_.empty() && omni_[i].sensor != live_sensor_)
        throw std::runtime_error(path + ": calibrated on '" + omni_[i].sensor +
                                 "' but the rig carries '" + live_sensor_ + "'");
      if (omni_[i].sensor.empty())
        RCLCPP_WARN(get_logger(), "%s does not state which sensor it was measured on",
                    path.c_str());
      if (!omni_[i].model.empty() && omni_[i].model != "omni")
        throw std::runtime_error(path + ": camera_model is '" + omni_[i].model + "'. This "
                                 "node projects through the Mei (omni-radtan) model because "
                                 "the lens is ~165 deg horizontal and no narrower model can "
                                 "represent it. Refusing to refit.");
      if (!omni_[i].distortion.empty() && omni_[i].distortion != "radtan")
        throw std::runtime_error(path + ": distortion_model is '" + omni_[i].distortion +
                                 "', expected radtan");

      T_cam1_cam_[i] = LoadMatrix4(rig_y["rig_in_cam1"][cams_[i]]);
    }

    // Verify the rig frame against the extrinsics rather than trusting the yaml. A
    // dropped 180 deg roll or a swapped ribbon produces numbers that still look plausible
    // and a BEV that is merely wrong, so check the one thing that pins it: each camera
    // must look out of the quadrant its label claims.
    cv::Matx44d T_rig_c1 = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) T_rig_c1(i, j) = R_rig_c1_(i, j);
    for (size_t i = 0; i < kNCam; ++i) {
      const cv::Matx44d T_rig_cam = T_rig_c1 * T_cam1_cam_[i];
      const cv::Vec3d axis(T_rig_cam(0, 2), T_rig_cam(1, 2), T_rig_cam(2, 2));
      const cv::Vec3d pos(T_rig_cam(0, 3), T_rig_cam(1, 3), T_rig_cam(2, 3));
      const double az = std::atan2(axis[1], axis[0]) * 180.0 / M_PI;
      const double el = std::asin(std::max(-1.0, std::min(1.0, axis[2] / cv::norm(axis)))) * 180.0 / M_PI;
      cam_az_rad_[i] = az * M_PI / 180.0;
      RCLCPP_INFO(get_logger(),
          "  %s: at (F %+.3f, L %+.3f, U %+.3f) m, looking az %+.2f deg, el %+.2f deg",
          cams_[i].c_str(), pos[0], pos[1], pos[2], az, el);
      auto it = expect_az_deg_.find(cams_[i]);
      if (it != expect_az_deg_.end()) {
        const double err = std::abs(WrapPi((az - it->second) * M_PI / 180.0)) * 180.0 / M_PI;
        if (err > az_tol_deg_)
          throw std::runtime_error(
              cams_[i] + " looks at azimuth " + std::to_string(az) + " deg but the rig layout "
              "says " + std::to_string(it->second) + " deg (" + std::to_string(err) +
              " deg off, tolerance " + std::to_string(az_tol_deg_) + "). Either the rig frame "
              "in " + plane_path_ + " is wrong, a camera ribbon is swapped, or these "
              "extrinsics belong to a different lineage than the images. Do not stitch.");
      }
      if (el > 2.0)
        RCLCPP_WARN(get_logger(), "%s tilts UP by %.1f deg — it will see little ground",
                    cams_[i].c_str(), el);
    }
  }

  // ------------------------------------------------------------------------------------
  // Startup: the remap tables. This is where the whole projection lives; the per-frame
  // path below is a gather and a weighted sum.
  // ------------------------------------------------------------------------------------
  void BuildMaps() {
    out_w_ = static_cast<int>(std::lround((range_l_ + range_r_) / res_));
    out_h_ = static_cast<int>(std::lround((range_f_ + range_b_) / res_));
    if (out_w_ < 8 || out_h_ < 8 || out_w_ > 8192 || out_h_ > 8192)
      throw std::runtime_error("output raster " + std::to_string(out_w_) + "x" +
                               std::to_string(out_h_) + " is implausible — check "
                               "resolution_m_per_px and the range_* parameters");
    const size_t N = static_cast<size_t>(out_w_) * out_h_;

    cv::Matx44d T_rig_c1 = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) T_rig_c1(i, j) = R_rig_c1_(i, j);

    const double feather_rad = fov_feather_deg_ * M_PI / 180.0;
    std::vector<float> wsum(N, 0.0f);
    for (size_t ci = 0; ci < kNCam; ++ci) {
      // T_cam_rig = (rig_in_cam1[cam])^-1 . (cam1 <- rig)
      const cv::Matx44d T_cam_rig = RigidInverse(T_cam1_cam_[ci]) * RigidInverse(T_rig_c1);
      const double model_limit = bev_ground::ModelValidHalfAngleRad(omni_[ci]);
      const double theta_max = std::min(model_limit, lens_fov_half_deg_ * M_PI / 180.0);
      RCLCPP_INFO(get_logger(), "  %s: Mei fit stays single-valued to %.1f deg, lens stated "
                  "%.1f deg -> gating rays at %.1f deg incidence",
                  cams_[ci].c_str(), model_limit * 180.0 / M_PI, lens_fov_half_deg_,
                  theta_max * 180.0 / M_PI);

      cv::Mat fmap(out_h_, out_w_, CV_32FC2, cv::Scalar(-1, -1));
      w_[ci] = cv::Mat::zeros(out_h_, out_w_, CV_32F);
      for (int row = 0; row < out_h_; ++row) {
        auto* mrow = fmap.ptr<cv::Vec2f>(row);
        auto* wrow = w_[ci].ptr<float>(row);
        const double X = range_f_ - (row + 0.5) * res_;              // rig forward, metres
        for (int col = 0; col < out_w_; ++col) {
          const double Y = range_l_ - (col + 0.5) * res_;            // rig left, metres
          // The plane is n.P = -h, so the cell's height follows from its (X, Y).
          const double Z = -(height_ + n_[0] * X + n_[1] * Y) / n_[2];

          const double Xc = T_cam_rig(0,0)*X + T_cam_rig(0,1)*Y + T_cam_rig(0,2)*Z + T_cam_rig(0,3);
          const double Yc = T_cam_rig(1,0)*X + T_cam_rig(1,1)*Y + T_cam_rig(1,2)*Z + T_cam_rig(1,3);
          const double Zc = T_cam_rig(2,0)*X + T_cam_rig(2,1)*Y + T_cam_rig(2,2)*Z + T_cam_rig(2,3);

          const double rho = std::sqrt(Xc*Xc + Yc*Yc + Zc*Zc);
          if (rho < 1e-6) continue;
          const double theta = std::acos(std::max(-1.0, std::min(1.0, Zc / rho)));
          if (theta >= theta_max) continue;

          double u, v;
          bev_ground::ProjectOmni(omni_[ci], Xc, Yc, Zc, &u, &v);
          if (input_rot180_) { u = omni_[ci].width - 1 - u; v = omni_[ci].height - 1 - v; }
          if (u < 0.0 || v < 0.0 || u > omni_[ci].width - 1.0 || v > omni_[ci].height - 1.0)
            continue;

          // Feather toward the edge of the trusted field, and toward the image border, so
          // a camera's contribution dies away instead of ending in a line.
          const double w_fov = std::clamp((theta_max - theta) / feather_rad, 0.0, 1.0);
          const double edge = std::min(std::min(u, v),
                                       std::min(omni_[ci].width - 1.0 - u,
                                                omni_[ci].height - 1.0 - v));
          const double w_edge = std::clamp(edge / border_feather_px_, 0.0, 1.0);

          // Sector ownership: each camera dominates around its own bearing and hands over
          // to its neighbour at the 45 deg bisector, where both weights are equal. This is
          // PHOTOMETRIC ONLY — it decides whose pixels are seen, never where they land.
          // The geometry above is already identical for both cameras.
          double w_sector = 1.0;
          const double radius = std::hypot(X, Y);
          if (radius > 0.05) {
            const double dpsi = WrapPi(std::atan2(Y, X) - cam_az_rad_[ci]);
            const double c = std::cos(dpsi);
            w_sector = c > 0.0 ? std::pow(c, sector_power_) : 0.0;
          }

          const float wgt = static_cast<float>(w_fov * w_edge * w_sector);
          if (wgt <= 0.0f) continue;
          mrow[col] = cv::Vec2f(static_cast<float>(u), static_cast<float>(v));
          wrow[col] = wgt;
          wsum[static_cast<size_t>(row) * out_w_ + col] += wgt;
        }
      }
      cv::convertMaps(fmap, cv::Mat(), map1_[ci], map2_[ci], CV_16SC2);
    }

    // Normalise once, at build time, so the per-frame path is a plain weighted sum.
    src_ = cv::Mat::zeros(out_h_, out_w_, CV_8U);
    std::array<size_t, kNCam + 1> covered_by{};
    for (int row = 0; row < out_h_; ++row) {
      auto* srow = src_.ptr<uint8_t>(row);
      for (int col = 0; col < out_w_; ++col) {
        const size_t idx = static_cast<size_t>(row) * out_w_ + col;
        size_t n_contrib = 0;
        float best = 0.0f;
        int best_ci = -1;
        for (size_t ci = 0; ci < kNCam; ++ci) {
          float& wv = w_[ci].at<float>(row, col);
          if (wv > 0.0f) {
            ++n_contrib;
            if (wv > best) { best = wv; best_ci = static_cast<int>(ci); }
            wv /= wsum[idx];
          }
        }
        covered_by[std::min(n_contrib, kNCam)]++;
        srow[col] = best_ci < 0 ? 0 : static_cast<uint8_t>(best_ci + 1);
      }
    }
    const double n_tot = static_cast<double>(N);
    RCLCPP_INFO(get_logger(), "coverage: %.1f%% of cells see no camera, %.1f%% see one, "
                "%.1f%% see two or more (the overlap the seams live in)",
                100.0 * covered_by[0] / n_tot, 100.0 * covered_by[1] / n_tot,
                100.0 * (n_tot - covered_by[0] - covered_by[1]) / n_tot);
    if (covered_by[0] > 0) {
      // The blind patch is directly under the rig: the cameras look outward, so the ground
      // beneath them falls below even a 95 deg half-field. Quote its reach so it is not
      // mistaken for a fault.
      double worst = 0.0;
      for (int row = 0; row < out_h_; ++row)
        for (int col = 0; col < out_w_; ++col)
          if (src_.at<uint8_t>(row, col) == 0) {
            const double X = range_f_ - (row + 0.5) * res_, Y = range_l_ - (col + 0.5) * res_;
            worst = std::max(worst, std::hypot(X, Y));
          }
      RCLCPP_INFO(get_logger(), "uncovered cells reach %.2f m from the rig origin — the "
                  "blind patch beneath the rig, which is geometry, not a fault", worst);
    }

    // Overlap bands, per ADJACENT pair, used both for exposure matching and for the
    // per-seam mismatch number. Built from ring_order because cam2/cam3 are diagonal and
    // share nothing.
    for (size_t k = 0; k < kNCam; ++k) {
      const auto ia = CamIndex(ring_[k]), ib = CamIndex(ring_[(k + 1) % kNCam]);
      Pair p;
      p.a = ia; p.b = ib; p.name = cams_[ia] + "-" + cams_[ib];
      for (int row = 0; row < out_h_; ++row)
        for (int col = 0; col < out_w_; ++col)
          if (w_[ia].at<float>(row, col) > 0.15f && w_[ib].at<float>(row, col) > 0.15f)
            p.idx.push_back(static_cast<int>(row) * out_w_ + col);
      RCLCPP_INFO(get_logger(), "  seam %s: %zu overlap cells", p.name.c_str(), p.idx.size());
      if (p.idx.size() < 50)
        RCLCPP_WARN(get_logger(), "seam %s has almost no overlap on the plane — exposure "
                    "matching and the seam metric will be meaningless for it",
                    p.name.c_str());
      pairs_.push_back(std::move(p));
    }

    for (size_t ci = 0; ci < kNCam; ++ci) warp_[ci].create(out_h_, out_w_, CV_8U);
    acc_.create(out_h_, out_w_, CV_32F);
    out_.create(out_h_, out_w_, CV_8U);
  }

  size_t CamIndex(const std::string& name) const {
    for (size_t i = 0; i < kNCam; ++i)
      if (cams_[i] == name) return i;
    throw std::runtime_error("ring_order names '" + name + "', which is not in cameras");
  }

  // ------------------------------------------------------------------------------------
  // Per frame
  // ------------------------------------------------------------------------------------
  // Match by TIMESTAMP, and — the part that is easy to get wrong — do not depend on
  // ARRIVAL ORDER either.
  //
  // argus_capture_node publishes a sweep in index order, cam1 first, every frame. A
  // matcher that only tries to build a set when cam1 arrives therefore races: if cam1's
  // callback runs before cam2/3/4 have been delivered, the nearest partner still in
  // history is the PREVIOUS trigger edge, one frame period away, and the set is thrown
  // out as unsynchronised. That failure has nothing to do with the trigger and it comes
  // and goes with scheduling. (cuvslam_multicam_node's matcher is anchored on cam1 in
  // exactly this way — worth checking there when it first runs on the board.)
  //
  // So: every arrival tries to complete a set, and whichever camera happens to be last
  // is the one that succeeds. A set already emitted for this edge is skipped rather than
  // re-stitched.
  //
  // A set that fails the skew gate is NOT counted as dropped, because a better partner
  // may still be in flight. What is counted is a frame EVICTED from history having never
  // belonged to an emitted set — the honest measure of "the rig is producing frames that
  // do not form sets", and the one that stays quiet when only the order was awkward.
  void OnFrame(size_t idx, const Img::ConstSharedPtr& m) {
    std::array<Img::ConstSharedPtr, kNCam> msgs;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      auto& h = hist_[idx];
      h.push_back(Slot{m, false});
      while (h.size() > history_) {
        if (!h.front().used) ++unmatched_[idx];
        h.pop_front();
      }

      const int64_t t0 = rclcpp::Time(m->header.stamp).nanoseconds();
      if (std::llabs(t0 - last_emit_ns_) <= max_skew_ns_) return;   // this edge is done

      std::array<Slot*, kNCam> slots{};
      slots[idx] = &h.back();
      int64_t lo = t0, hi = t0;
      for (size_t i = 0; i < kNCam; ++i) {
        if (i == idx) continue;
        int64_t best_d = INT64_MAX;
        for (auto& cand : hist_[i]) {
          const int64_t d = std::llabs(rclcpp::Time(cand.msg->header.stamp).nanoseconds() - t0);
          if (d < best_d) { best_d = d; slots[i] = &cand; }
        }
        if (!slots[i]) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no frame yet from %s",
                               cams_[i].c_str());
          return;
        }
        const int64_t t = rclcpp::Time(slots[i]->msg->header.stamp).nanoseconds();
        lo = std::min(lo, t);
        hi = std::max(hi, t);
      }
      const int64_t skew = hi - lo;
      if (skew > max_skew_ns_) {
        closest_attempt_ns_ = std::min(closest_attempt_ns_, skew);
        return;
      }
      for (size_t i = 0; i < kNCam; ++i) {
        slots[i]->used = true;
        msgs[i] = slots[i]->msg;
      }
      last_emit_ns_ = t0;
      ++sets_;
      worst_skew_ns_ = std::max(worst_skew_ns_, skew);
    }
    Stitch(msgs);
  }

  void Stitch(const std::array<Img::ConstSharedPtr, kNCam>& msgs) {
    const auto t_start = std::chrono::steady_clock::now();
    std::array<cv_bridge::CvImageConstPtr, kNCam> holds;
    for (size_t i = 0; i < kNCam; ++i) {
      try {
        holds[i] = cv_bridge::toCvShare(msgs[i], "mono8");
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "cv_bridge on %s: %s",
                             cams_[i].c_str(), e.what());
        return;
      }
      if (holds[i]->image.cols != live_w_ || holds[i]->image.rows != live_h_) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
            "%s arrived %dx%d but the maps were built for %dx%d — the calibration guard "
            "was passed a resolution the cameras are not using",
            cams_[i].c_str(), holds[i]->image.cols, holds[i]->image.rows, live_w_, live_h_);
        return;
      }
      cv::remap(holds[i]->image, warp_[i], map1_[i], map2_[i], cv::INTER_LINEAR,
                cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    if (equalize_) UpdateGains();

    acc_.setTo(0.0f);
    for (size_t ci = 0; ci < kNCam; ++ci) {
      const float g = static_cast<float>(gain_[ci]);
      for (int row = 0; row < out_h_; ++row) {
        const float* wrow = w_[ci].ptr<float>(row);
        const uint8_t* srow = warp_[ci].ptr<uint8_t>(row);
        float* arow = acc_.ptr<float>(row);
        for (int col = 0; col < out_w_; ++col)
          if (wrow[col] > 0.0f) arow[col] += wrow[col] * g * srow[col];
      }
    }
    acc_.convertTo(out_, CV_8U);
    // Measured up to here: the projection work proper. Serialising and publishing 160 kB
    // costs several times this again, and lumping the two together hides which one is the
    // problem when the rate disappoints. Host reference for a 400x400 raster: ~2.0 ms in
    // the four remaps, ~0.6 ms in the weighted accumulate, 13 us in the 8-bit convert.
    stitch_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t_start).count();

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp = msgs[0]->header.stamp;   // the set's own trigger edge, not now()
    msg->header.frame_id = out_frame_;
    msg->width = out_w_; msg->height = out_h_;
    msg->encoding = "mono8"; msg->step = out_w_;
    msg->data.assign(out_.data, out_.data + static_cast<size_t>(out_w_) * out_h_);
    pub_->publish(std::move(msg));

    if (src_pub_ && src_pub_->get_subscription_count() > 0) {
      auto s = std::make_unique<sensor_msgs::msg::Image>();
      s->header.stamp = msgs[0]->header.stamp;
      s->header.frame_id = out_frame_;
      s->width = out_w_; s->height = out_h_;
      s->encoding = "mono8"; s->step = out_w_;
      s->data.assign(src_.data, src_.data + static_cast<size_t>(out_w_) * out_h_);
      src_pub_->publish(std::move(s));
    }

    ++published_;
    Report();
  }

  // Per-camera photometric gain from the overlaps. cam2 and cam4 face windows and can sit
  // 30 deg of exposure apart from cam1/cam3, which shows as a brightness step exactly on
  // the seam and is easily mistaken for a geometric error.
  //
  // For each adjacent pair we want g_a.m_a = g_b.m_b, i.e. l_a - l_b = log(m_b/m_a) with
  // l = log g. That is a least-squares problem on the 4-cycle; its Laplacian is singular
  // along the all-ones direction (only ratios are observable), so the gauge is fixed by
  // adding J/4, which selects the mean-zero solution.
  void UpdateGains() {
    cv::Matx44d L = cv::Matx44d::zeros();
    cv::Vec4d r(0, 0, 0, 0);
    size_t usable = 0;
    for (auto& p : pairs_) {
      if (p.idx.size() < 50) continue;
      double sa = 0.0, sb = 0.0;
      for (int i : p.idx) {
        sa += warp_[p.a].data[i];
        sb += warp_[p.b].data[i];
      }
      const double ma = sa / p.idx.size(), mb = sb / p.idx.size();
      p.mean_a = ma; p.mean_b = mb;
      if (ma < 4.0 || mb < 4.0) continue;   // too dark to divide by; leave this seam out
      const double d = std::log(mb / ma);
      p.log_ratio = d;
      L(p.a, p.a) += 1.0; L(p.b, p.b) += 1.0;
      L(p.a, p.b) -= 1.0; L(p.b, p.a) -= 1.0;
      r[p.a] += d; r[p.b] -= d;
      ++usable;
    }
    if (usable < 2) return;                 // not enough constraints; keep the last gains
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) L(i, j) += 0.25;
    cv::Vec4d l;
    if (!cv::solve(cv::Mat(L), cv::Mat(r), l, cv::DECOMP_LU)) return;
    for (size_t i = 0; i < kNCam; ++i) {
      const double g = std::clamp(std::exp(l[i]), 1.0 / gain_limit_, gain_limit_);
      gain_[i] = gains_valid_ ? gain_smooth_ * gain_[i] + (1.0 - gain_smooth_) * g : g;
    }
    gains_valid_ = true;
  }

  void Report() {
    const auto t = std::chrono::steady_clock::now();
    if (t - last_report_ < std::chrono::seconds(5)) return;
    const double secs = std::chrono::duration<double>(t - last_report_).count();
    const int64_t unmatched = std::accumulate(unmatched_.begin(), unmatched_.end(), int64_t{0});
    RCLCPP_INFO(get_logger(), "%.1f Hz out, %ld sets, worst skew %.0f us, project %ld us (publish extra), "
                "gains [%.2f %.2f %.2f %.2f]%s",
                (published_ - last_published_) / secs, sets_,
                worst_skew_ns_ / 1e3, stitch_us_,
                gain_[0], gain_[1], gain_[2], gain_[3],
                provisional_ ? "  [PROVISIONAL PLANE]" : "");
    if (unmatched > last_unmatched_) {
      // Frames that aged out of history without ever belonging to a set. On a rig whose
      // measured skew is 1 us, a nonzero and growing count means the trigger, a camera,
      // or the capture node is at fault — not something to widen the gate for.
      RCLCPP_WARN(get_logger(), "%ld frames (+%ld) never formed a set [%ld %ld %ld %ld]; "
                  "closest attempt was %.2f ms against a %.2f ms gate. Is the trigger running?",
                  unmatched, unmatched - last_unmatched_, unmatched_[0], unmatched_[1],
                  unmatched_[2], unmatched_[3],
                  closest_attempt_ns_ == INT64_MAX ? 0.0 : closest_attempt_ns_ / 1e6,
                  max_skew_ns_ / 1e6);
      last_unmatched_ = unmatched;
      closest_attempt_ns_ = INT64_MAX;
    }
    // Per-seam mismatch AFTER gain matching. This is a PHOTOMETRIC number in grey levels:
    // it rises when the two cameras disagree about the overlap, which happens both for a
    // misregistered plane and for anything standing above it. It is NOT the millimetres-
    // on-the-ground parallax residual the spec asks to be published — that one needs a
    // target of known geometry laid across the seam and is measured offline.
    std::string seams;
    for (const auto& p : pairs_) {
      if (p.idx.size() < 50) continue;
      double s = 0.0;
      for (int i : p.idx)
        s += std::abs(gain_[p.a] * warp_[p.a].data[i] - gain_[p.b] * warp_[p.b].data[i]);
      seams += "  " + p.name + " " + std::to_string(static_cast<int>(s / p.idx.size())) +
               " lv (" + std::to_string(static_cast<int>(20.0 * p.log_ratio / std::log(10.0))) + " dB)";
    }
    RCLCPP_INFO(get_logger(), "seam mismatch after gain match (grey levels, exposure step in dB):%s",
                seams.c_str());
    worst_skew_ns_ = 0;
    last_published_ = published_;
    last_report_ = t;
  }

  // Latched, so a consumer that starts later still learns how to read a pixel.
  void PublishInfo() {
    std::ostringstream ss;
    ss.precision(6); ss << std::fixed;
    ss << "frame: " << out_frame_ << "\n"
       << "parent_frame: " << rig_frame_ << "   # FLU: x forward, y left, z up\n"
       << "encoding: mono8\n"
       << "width_px: " << out_w_ << "\nheight_px: " << out_h_ << "\n"
       << "resolution_m_per_px: " << res_ << "\n"
       << "range_forward_m: " << range_f_ << "\nrange_back_m: " << range_b_ << "\n"
       << "range_left_m: " << range_l_ << "\nrange_right_m: " << range_r_ << "\n"
       << "plane_status: " << (provisional_ ? "PROVISIONAL" : "measured") << "\n"
       << "rig_height_m: " << height_ << "\n"
       << "plane_normal_rig: [" << n_[0] << ", " << n_[1] << ", " << n_[2] << "]\n"
       << "# row 0 is the most-forward row; col 0 is the leftmost column.\n"
       << "pixel_to_metres: |\n"
       << "  x_forward_m = " << range_f_ << " - (row + 0.5) * " << res_ << "\n"
       << "  y_left_m    = " << range_l_ << " - (col + 0.5) * " << res_ << "\n"
       << "  z_m         = 0            # by construction: " << out_frame_
       << " sits on the plane\n"
       << "# The stated resolution is measured in the rig's HORIZONTAL projection. On a\n"
       << "# plane tilted by t, true distance along the ground is 1/cos(t) times this;\n"
       << "# below 3 deg that is under 0.15% and is ignored.\n"
       << "source_mask_topic: bev/ground/source   # 0 = no coverage, 1..4 = dominant "
       << "camera in cameras[] order\n";
    std_msgs::msg::String msg;
    msg.data = ss.str();
    info_pub_->publish(msg);
  }

  struct Pair {
    size_t a{}, b{};
    std::string name;
    std::vector<int> idx;
    double mean_a{}, mean_b{}, log_ratio{};
  };

  std::string calib_dir_, rig_path_, plane_path_, rig_frame_, out_frame_, live_sensor_;
  std::vector<std::string> cams_, topics_, ring_;
  int live_w_{}, live_h_{}, out_w_{}, out_h_{};
  double res_{}, range_f_{}, range_b_{}, range_l_{}, range_r_{};
  double lens_fov_half_deg_{}, fov_feather_deg_{}, border_feather_px_{}, sector_power_{};
  double height_ = 0.0, provisional_h_ = 0.0, gain_smooth_{}, gain_limit_{}, az_tol_deg_ = 15.0;
  std::array<double, 3> n_{0.0, 0.0, 1.0};
  bool provisional_ = true, allow_unmeasured_ = false, equalize_ = true;
  bool publish_source_ = true, input_rot180_ = false, gains_valid_ = false;

  cv::Matx33d R_rig_c1_ = cv::Matx33d::eye();
  std::map<std::string, double> expect_az_deg_;
  std::array<bev_ground::OmniIntrinsics, kNCam> omni_;
  std::array<cv::Matx44d, kNCam> T_cam1_cam_;
  std::array<double, kNCam> cam_az_rad_{};
  std::array<cv::Mat, kNCam> map1_, map2_, w_, warp_;
  std::vector<Pair> pairs_;
  cv::Mat acc_, out_, src_;
  std::array<double, kNCam> gain_{1.0, 1.0, 1.0, 1.0};

  // A frame plus whether it ever belonged to an emitted set, so eviction can count the
  // ones that never did.
  struct Slot { Img::ConstSharedPtr msg; bool used; };
  std::array<rclcpp::Subscription<Img>::SharedPtr, kNCam> subs_;
  std::array<std::deque<Slot>, kNCam> hist_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_, src_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr info_pub_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;

  std::mutex mtx_;
  size_t history_ = 8;
  int64_t max_skew_ns_ = 1000000, worst_skew_ns_ = 0, sets_ = 0;
  int64_t last_emit_ns_ = INT64_MIN / 4, closest_attempt_ns_ = INT64_MAX, last_unmatched_ = 0;
  std::array<int64_t, kNCam> unmatched_{};
  int64_t stitch_us_ = 0, published_ = 0, last_published_ = 0;
  std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<BevGroundStitchNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("bev_ground_stitch"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
