// Carve each fisheye into two virtual pinhole cameras, so cuVSLAM can consume this rig.
//
// WHY THIS EXISTS. Our lenses fit ~192 deg diagonal (vendor spec D190/H160, and the
// omni-radtan fits agree). cuVSLAM's only fisheye model is equidistant, which
// parameterises through x/z with theta = arctan(r) and therefore cannot represent a ray
// at or past 90 deg incidence - see cuvslam2.h, "works only for FOV < 180". So handing it
// the raw fisheyes is not an option we rejected on quality grounds; it is not expressible.
// Each camera is instead split into two pinholes at +-45 deg, and the pinhole facing the
// neighbour forms an ordinary stereo pair with the neighbour's facing pinhole.
//
// The Mei (omni-radtan) projection below is written out by hand rather than calling
// cv::omnidir, which lives in opencv_contrib: it is absent from our host OpenCV and
// cannot be assumed present on the TX2. It was checked against cv2.omnidir.projectPoints
// over 4000 rays and agrees to 5e-13 px, so this is the same model the calibration was
// solved with, not a lookalike.
//
// FOV COMES FROM THE HORIZONTAL FIELD, NOT THE DIAGONAL. The split is by yaw, so a
// pinhole of (H - 90) deg is what the lens can actually feed. Carving (D - 90) = 100 deg
// asks each pinhole to reach 95 deg off-axis where the lens delivers 80, and every
// rectified view comes back with a black wedge - measured at 90% non-black against 100%
// once corrected to the horizontal 160.

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace bev_cuvslam {

// Mei / omni-radtan intrinsics, as solved by tartancalib.
struct OmniIntrinsics {
  double xi{}, fx{}, fy{}, cx{}, cy{};
  double k1{}, k2{}, p1{}, p2{};
  int width{}, height{};
};

inline OmniIntrinsics LoadOmni(const std::string& path) {
  YAML::Node y = YAML::LoadFile(path);
  if (y["cam0"]) y = y["cam0"];              // kalibr writes cam0:; our config/ is flat
  const auto in = y["intrinsics"], di = y["distortion_coeffs"], re = y["resolution"];
  if (!in || in.size() != 5) throw std::runtime_error("expected 5 omni intrinsics in " + path);
  OmniIntrinsics o;
  o.xi = in[0].as<double>();
  o.fx = in[1].as<double>(); o.fy = in[2].as<double>();
  o.cx = in[3].as<double>(); o.cy = in[4].as<double>();
  o.k1 = di[0].as<double>(); o.k2 = di[1].as<double>();
  o.p1 = di[2].as<double>(); o.p2 = di[3].as<double>();
  o.width = re[0].as<int>(); o.height = re[1].as<int>();
  return o;
}

// Project a ray onto the fisheye image through the Mei model. Verified against
// cv2.omnidir.projectPoints to 5e-13 px.
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

// One virtual pinhole: a plain pinhole camera rotated yaw_rad off the fisheye's axis.
struct VirtualPinhole {
  int width{}, height{};
  double focal{}, cx{}, cy{};
  double yaw_rad{};          // -pi/4 or +pi/4
  cv::Mat map1, map2;        // remap tables, fixed-point (CV_16SC2 + CV_16UC1)
};

// Build the BACKWARD map: for each virtual pixel, which raw fisheye pixel feeds it.
// Backward is the only direction that fills the output without holes.
inline VirtualPinhole BuildVirtualPinhole(const OmniIntrinsics& o, double yaw_rad,
                                          int width, int height, double fov_deg) {
  VirtualPinhole vp;
  vp.width = width; vp.height = height; vp.yaw_rad = yaw_rad;
  vp.focal = width/2.0 / std::tan(fov_deg*CV_PI/180.0/2.0);
  vp.cx = width/2.0; vp.cy = height/2.0;

  const double c = std::cos(yaw_rad), s = std::sin(yaw_rad);   // R = Ry(yaw)
  cv::Mat fmap(height, width, CV_32FC2);
  for (int i = 0; i < height; ++i) {
    auto* row = fmap.ptr<cv::Vec2f>(i);
    for (int j = 0; j < width; ++j) {
      const double x = j - vp.cx, y = i - vp.cy, z = vp.focal;
      // ray in the FISHEYE's frame: R * (x, y, f)
      const double X =  c*x + s*z, Y = y, Z = -s*x + c*z;
      double u, v;
      ProjectOmni(o, X, Y, Z, &u, &v);
      row[j] = cv::Vec2f(static_cast<float>(u), static_cast<float>(v));
    }
  }
  cv::convertMaps(fmap, cv::Mat(), vp.map1, vp.map2, CV_16SC2);
  return vp;
}

}  // namespace bev_cuvslam
