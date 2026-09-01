// Building the cuVSLAM rig from this project's calibration files - ONE implementation,
// used by both the modular node and the fused node.
//
// WHY SHARED RATHER THAN COPIED. On 2026-09-01 two extrinsic solves of this rig turned out
// to be MIRRORED (cam2 and cam3 on opposite sides of cam1) and every check in the project
// passed both of them - the ring closure, the epipolar residuals, and cuVSLAM's own frustum
// graph, which scores ~0.94 either way. A second copy of this code drifting from the first
// is the same class of silent, plausible-looking error. Both nodes now carve identically by
// construction. scripts/vo/verify_rig_build.sh compiles the helpers below directly out of
// this header, so the offline gate tests what the nodes actually run.
//
// Frame convention is stated once, in config/rig/rig_extrinsics_imx296.yaml: `rig_in_cam1`
// is each camera's pose in cam1's RAW optical frame, +x physically LEFT because the modules
// are inverted and the capture path does not rotate.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "bev_cuvslam/virtual_pinhole.hpp"
#include "cuvslam/cuvslam2.h"

namespace bev_cuvslam {

cv::Matx44d load_matrix4(const YAML::Node& n) {
  cv::Matx44d M;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) M(i, j) = n[i][j].as<double>();
  return M;
}

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

// The 8 virtual pinholes and the cuVSLAM rig they form. `vsrc[k]` is the fisheye that
// virtual camera k is carved from.
struct VirtualRig {
  cuvslam::Rig rig;
  std::vector<VirtualPinhole> vpin;
  std::vector<int> vsrc;
  // Task 1.5's stale-calibration guard. Returned rather than logged so this header does not
  // depend on rclcpp; the caller logs them. A recalibration is the exact event it exists for.
  std::vector<std::string> warnings;
};

// Carve each of the 4 fisheyes into two virtual pinholes at -+45 deg and build the rig.
// The lenses are ~192 deg and cuVSLAM's only fisheye model is the equidistant one, capped
// below 180 deg, so this carve is REQUIRED, not an optimisation (README 4.8).
inline VirtualRig BuildVirtualRig(const std::string& calib_dir, const std::string& rig_path,
                                  const std::string& vstereo_path,
                                  const std::vector<std::string>& cams) {
  VirtualRig out;
  const YAML::Node rig_y = YAML::LoadFile(rig_path);
  const YAML::Node vp_y = YAML::LoadFile(vstereo_path)["virtual_pinhole"];
  const int vw = vp_y["width"].as<int>(), vh = vp_y["height"].as<int>();
  const double vfov = vp_y["fov_deg"].as<double>();

  for (size_t i = 0; i < cams.size(); ++i) {
    const auto omni = LoadOmni(calib_dir + "/" + cams[i] + ".yaml");
    if (omni.width != vp_y["source_width"].as<int>(omni.width))
      out.warnings.push_back(cams[i] + " calibrated at " + std::to_string(omni.width) + "x" +
                             std::to_string(omni.height) + " - check it matches the live rig");
    // rig frame IS cam1's optical frame, which is how rig_in_cam1 is expressed.
    // NOT the FLU body frame of rig_layout.yaml - see the frame note there (3R.16b).
    const cv::Matx44d rig_from_fisheye = load_matrix4(rig_y["rig_in_cam1"][cams[i]]);
    for (int k = 0; k < 2; ++k) {
      const double yaw = (k == 0 ? -1.0 : +1.0) * CV_PI / 4.0;
      out.vpin.push_back(BuildVirtualPinhole(omni, yaw, vw, vh, vfov));
      out.vsrc.push_back(static_cast<int>(i));
      const double c = std::cos(yaw), sn = std::sin(yaw);
      const cv::Matx44d Ry(c,0,sn,0,  0,1,0,0,  -sn,0,c,0,  0,0,0,1);
      cuvslam::Camera cam;
      cam.size = {vw, vh};
      cam.focal = {static_cast<float>(out.vpin.back().focal), static_cast<float>(out.vpin.back().focal)};
      cam.principal = {static_cast<float>(out.vpin.back().cx), static_cast<float>(out.vpin.back().cy)};
      // Pinhole with NO distortion: the remap already removed it. Anything else here
      // would be applying the correction twice.
      cam.distortion.model = cuvslam::Distortion::Model::Pinhole;
      cam.distortion.parameters = {};
      cam.rig_from_camera = pose_from_matrix(rig_from_fisheye * Ry);
      out.rig.cameras.push_back(cam);
    }
  }
  return out;
}

}  // namespace bev_cuvslam
