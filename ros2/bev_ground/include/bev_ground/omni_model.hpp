// Mei (omni-radtan) camera model, plus the two derived quantities the ground stitch needs:
// how far off-axis the fitted model may be trusted, and where a ray lands on the sensor.
//
// WHY THE MODEL IS MEI AND NOT KANNALA-BRANDT. These lenses are 1.78 mm, vendor D190/H160
// on 1/3". Our own fits measure H-FOV 164.8-165.6 deg, i.e. rays past 82 deg incidence in
// the horizontal alone. The equidistant model (pinhole-equi / KB) parameterises through
// theta = arctan(r) and cannot represent a ray at or past 90 deg; it diverged on every
// camera. The panorama node implements KB only, which is why it cannot consume this
// rig's calibration at all and why this is a new node rather than a patch to that one.
//
// PROVENANCE OF ProjectOmni. This is the same function as
// ros2/bev_cuvslam/include/bev_cuvslam/virtual_pinhole.hpp, which was checked against
// cv2.omnidir.projectPoints over 4000 rays and agrees to 5e-13 px. It is duplicated here
// rather than shared because bev_cuvslam's CMake hard-REQUIREs libcuvslam, and making a
// stitcher unbuildable without a VO library would be the worse coupling. The two copies
// must stay identical; CheckOmniAgainstVirtualPinhole in the test notes says how.

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace bev_ground {

// Load a YAML file, tolerating the OpenCV "%YAML:1.0 / ---" preamble that some of our
// calibration files carry and yaml-cpp will not parse.
inline YAML::Node LoadYaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("%YAML", 0) == 0) continue;
    if (line == "---") continue;
    ss << line << "\n";
  }
  return YAML::Load(ss.str());
}

struct OmniIntrinsics {
  double xi{}, fx{}, fy{}, cx{}, cy{};
  double k1{}, k2{}, p1{}, p2{};
  int width{}, height{};
  std::string sensor, model, distortion;
};

// Accepts both layouts we have: a kalibr/tartancalib camchain (cam0: ...) and our flat
// config/calib/<res>/camN.yaml.
inline OmniIntrinsics LoadOmni(const std::string& path) {
  YAML::Node y = LoadYaml(path);
  if (y["cam0"]) y = y["cam0"];
  const auto in = y["intrinsics"], di = y["distortion_coeffs"], re = y["resolution"];
  if (!in || in.size() != 5)
    throw std::runtime_error(path + ": expected 5 omni intrinsics [xi,fx,fy,cx,cy], got " +
                             std::to_string(in ? in.size() : 0) +
                             " — is this a Kannala-Brandt calibration?");
  if (!di || di.size() != 4) throw std::runtime_error(path + ": expected 4 radtan coefficients");
  if (!re || re.size() != 2) throw std::runtime_error(path + ": expected resolution [w,h]");
  OmniIntrinsics o;
  o.xi = in[0].as<double>();
  o.fx = in[1].as<double>();  o.fy = in[2].as<double>();
  o.cx = in[3].as<double>();  o.cy = in[4].as<double>();
  o.k1 = di[0].as<double>();  o.k2 = di[1].as<double>();
  o.p1 = di[2].as<double>();  o.p2 = di[3].as<double>();
  o.width = re[0].as<int>();  o.height = re[1].as<int>();
  o.sensor = y["sensor"] ? y["sensor"].as<std::string>() : std::string();
  o.model = y["camera_model"] ? y["camera_model"].as<std::string>() : std::string();
  o.distortion = y["distortion_model"] ? y["distortion_model"].as<std::string>() : std::string();
  return o;
}

// Project a ray (need not be normalised) onto the fisheye image.
inline void ProjectOmni(const OmniIntrinsics& o, double X, double Y, double Z,
                        double* u, double* v) {
  const double n = std::sqrt(X*X + Y*Y + Z*Z);
  const double den = Z/n + o.xi;
  const double xu = (X/n)/den, yu = (Y/n)/den;
  const double r2 = xu*xu + yu*yu;
  const double rad = 1.0 + o.k1*r2 + o.k2*r2*r2;
  const double xd = xu*rad + 2.0*o.p1*xu*yu + o.p2*(r2 + 2.0*xu*xu);
  const double yd = yu*rad + o.p1*(r2 + 2.0*yu*yu) + 2.0*o.p2*xu*yu;
  *u = o.fx*xd + o.cx;
  *v = o.fy*yd + o.cy;
}

// The largest incidence angle at which this fit is still a FUNCTION of direction.
//
// With xi ~ 2 the Mei denominator (cos(theta) + xi) stays positive for every theta, so
// ProjectOmni will cheerfully return a pixel for a ray coming from BEHIND the camera —
// folded back on top of a legitimate forward ray. Nothing in the projection itself
// rejects it. The radial polynomial 1 + k1.r^2 + k2.r^4 also turns over eventually, and
// past that turnover two different directions share one pixel.
//
// So the gate is the angle at which the projected radius stops increasing. On our four
// cameras that lands at 119-120 deg, which is BEYOND the vendor's D190 (95 deg half) and
// therefore extrapolation into a region the AprilGrid never reached. The caller is
// expected to take the min of this and a stated lens half-angle; this function only
// reports where the arithmetic gives up.
inline double ModelValidHalfAngleRad(const OmniIntrinsics& o, int steps = 4000) {
  double prev_r = -1.0, best_theta = 0.0;
  for (int i = 0; i <= steps; ++i) {
    const double th = (M_PI * 2.0 / 3.0) * i / steps;   // scan to 120 deg
    double u, v;
    ProjectOmni(o, std::sin(th), 0.0, std::cos(th), &u, &v);
    const double r = std::abs(u - o.cx);
    if (r < prev_r) break;
    prev_r = r;
    best_theta = th;
  }
  return best_theta;
}

}  // namespace bev_ground
