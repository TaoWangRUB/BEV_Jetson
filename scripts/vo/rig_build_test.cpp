#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include "cuvslam/cuvslam2.h"
#include "bev_cuvslam/virtual_pinhole.hpp"
#include "helpers.inc"
int main(int argc, char** argv) {
  const YAML::Node rig_y = YAML::LoadFile(argv[1]);
  const YAML::Node vp_y  = YAML::LoadFile(argv[2])["virtual_pinhole"];
  const int vw = vp_y["width"].as<int>(), vh = vp_y["height"].as<int>();
  const double vfov = vp_y["fov_deg"].as<double>();
  const char* cams[4] = {"cam1","cam2","cam3","cam4"};
  for (int i = 0; i < 4; ++i) {
    const auto omni = bev_cuvslam::LoadOmni(std::string(argv[3]) + "/" + cams[i] + ".yaml");
    const cv::Matx44d rig_from_fisheye = load_matrix4(rig_y["rig_in_cam1"][cams[i]]);
    for (int k = 0; k < 2; ++k) {
      const double yaw = (k == 0 ? -1.0 : +1.0) * CV_PI / 4.0;
      const auto vp = bev_cuvslam::BuildVirtualPinhole(omni, yaw, vw, vh, vfov);
      const double c = std::cos(yaw), sn = std::sin(yaw);
      const cv::Matx44d Ry(c,0,sn,0, 0,1,0,0, -sn,0,c,0, 0,0,0,1);
      const cuvslam::Pose p = pose_from_matrix(rig_from_fisheye * Ry);
      printf("%s_%c %.6f %.1f", cams[i], k ? 'R' : 'L', vp.focal, (double)vw);
      for (int a = 0; a < 4; ++a) printf(" %.9f", p.rotation[a]);
      for (int a = 0; a < 3; ++a) printf(" %.9f", p.translation[a]);
      printf("\n");
    }
  }
  return 0;
}
