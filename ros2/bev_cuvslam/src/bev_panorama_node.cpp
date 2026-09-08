// Surround panorama node: capture the 4 fisheye to GPU (Argus -> NVMM -> CUDA, the same
// zero-copy bridge as the fused VO node) and stitch them into one equirectangular panorama on
// the GPU (custom kernel in stitch_kernel.cu), using a remap table precomputed from the KB
// intrinsics + rig extrinsics. Publishes /bev/panorama (mono8) for rviz; optional mp4.
//
// Build/run in cuvslam-foxy:tx2 (Argus + EGL + CUDA). See openspec surround-panorama-stitch.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/FrameConsumer.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <nvbuf_utils.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#ifdef Success
#undef Success
#endif
#ifdef Status
#undef Status
#endif
#ifdef None
#undef None
#endif
#ifdef Bool
#undef Bool
#endif

using namespace Argus;

// Kernel launcher (stitch_kernel.cu) — extern "C" so it crosses the nvcc/g++-8 boundary.
extern "C" void launch_equirect_stitch(uint8_t* out, int W, int H,
    const void* c0, const void* c1, const void* c2, const void* c3,
    int cpitch, int cw, int ch,
    const void* uv0, const void* uv1, const void* uv2, const void* uv3,
    const void* w0, const void* w1, const void* w2, const void* w3);

namespace {
YAML::Node load_yaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss; std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("%YAML", 0) == 0) continue;
    if (line == "---") continue;
    ss << line << "\n";
  }
  return YAML::Load(ss.str());
}

// Mei / omni-radtan intrinsics — the model the IMX296 rig is calibrated in.
//
// WHY NOT KANNALA-BRANDT. These lenses are 1.78 mm D190/H160 on 1/3", a genuine >180 deg
// fisheye (~192 deg diagonal by our own fits). The equidistant model parameterises through
// th = atan(r) with r = x/z, which CANNOT represent a ray at or past 90 deg incidence: it
// needs z > 0. It diverged on every camera during calibration and it silently drops the
// outer part of every image here. See config/calib/imx296_1456x1088/cam1.yaml.
//
// The unified (Mei) model projects through a sphere offset by xi, so z may be negative and
// rays past 90 deg are representable. That is why the calibration is in this model and why
// this node only accepts it.
struct Mei {
  double xi, fx, fy, cx, cy;
  double k1, k2, p1, p2;      // radtan
  int w, h;
};

Mei load_mei(const std::string& path) {
  YAML::Node y = load_yaml(path);

  // Refuse the old model loudly rather than mis-projecting.  This is the exact bug this
  // node shipped with: it defaulted to the previous IMX219 rig's KANNALA_BRANDT set
  // (1640x1232, mu ~788 px), which on IMX296 pixels implies a 2.72 mm lens against a
  // 1.78 mm part - a ~48% focal error, applied silently.
  if (y["model_type"]) {
    throw std::runtime_error(
        path + ": this is a KANNALA_BRANDT (equidistant) calibration, which cannot "
               "represent the >180 deg IMX296 lenses and is almost certainly the retired "
               "IMX219 set. Use config/calib/imx296_1456x1088 (camera_model: omni).");
  }
  if (y["camera_model"] && y["camera_model"].as<std::string>() != "omni")
    throw std::runtime_error(path + ": expected camera_model: omni, got " +
                             y["camera_model"].as<std::string>());

  Mei c;
  auto res = y["resolution"];
  c.w = res[0].as<int>();
  c.h = res[1].as<int>();
  auto in = y["intrinsics"];              // [xi, fx, fy, cx, cy]
  c.xi = in[0].as<double>();
  c.fx = in[1].as<double>(); c.fy = in[2].as<double>();
  c.cx = in[3].as<double>(); c.cy = in[4].as<double>();
  auto d = y["distortion_coeffs"];        // [k1, k2, p1, p2]
  c.k1 = d[0].as<double>(); c.k2 = d[1].as<double>();
  c.p1 = d[2].as<double>(); c.p2 = d[3].as<double>();
  return c;
}

// Project a ray (need not be normalised) to pixel coordinates.  Returns false when the ray
// falls outside the model's representable cone.  `theta_out` is the true incidence angle
// from the optical axis, valid past 90 deg — which is what makes the FOV gate and the
// feather weight meaningful on a >180 deg lens.
bool mei_project(const Mei& c, double X, double Y, double Z, double* u, double* v,
                 double* theta_out) {
  const double n = std::sqrt(X*X + Y*Y + Z*Z);
  if (n < 1e-12) return false;
  const double xs = X / n, ys = Y / n, zs = Z / n;

  *theta_out = std::acos(std::max(-1.0, std::min(1.0, zs)));

  // Unified-model validity. For xi > 1 the forward map is injective only while
  // zs > -1/xi; past that two rays share a pixel and the sample would be meaningless.
  const double denom = zs + c.xi;
  if (denom < 1e-9) return false;
  if (c.xi > 1.0 && zs <= -1.0 / c.xi) return false;

  const double mx = xs / denom, my = ys / denom;
  const double r2 = mx*mx + my*my;
  const double rad = 1.0 + c.k1*r2 + c.k2*r2*r2;
  const double dx = 2.0*c.p1*mx*my + c.p2*(r2 + 2.0*mx*mx);
  const double dy = c.p1*(r2 + 2.0*my*my) + 2.0*c.p2*mx*my;

  *u = c.fx * (rad*mx + dx) + c.cx;
  *v = c.fy * (rad*my + dy) + c.cy;
  return true;
}
// Rotation matrix (camera->rig) from q_wxyz, row-major R[9].
void quat_to_R(const YAML::Node& n, double R[9]) {
  auto q = n["q_wxyz"];
  double w = q[0].as<double>(), x = q[1].as<double>(), y = q[2].as<double>(), z = q[3].as<double>();
  R[0] = 1 - 2*(y*y+z*z); R[1] = 2*(x*y-w*z);   R[2] = 2*(x*z+w*y);
  R[3] = 2*(x*y+w*z);     R[4] = 1 - 2*(x*x+z*z); R[5] = 2*(y*z-w*x);
  R[6] = 2*(x*z-w*y);     R[7] = 2*(y*z+w*x);   R[8] = 1 - 2*(x*x+y*y);
}

// camera->rig rotation for one camera, accepting either rig-extrinsics layout:
//
//   board_center / q_wxyz   the panorama lineage (IMX219 era): cameras.<cam>.q_wxyz, in a
//                           nominal level body frame.
//   rig_in_cam1 / 4x4       the IMX296 lineage: each camera's pose as a 4x4 in CAM1'S RAW
//                           OPTICAL FRAME. The rotation block is already camera->rig with
//                           rig == cam1, so it is used directly.
//
// ⚠ The two are NOT interchangeable as a reference frame. `rig_in_cam1` makes cam1's
// optical axis the rig axis, so the panorama comes out oriented to cam1 rather than
// levelled; a cam1->body transform would be needed for a level horizon, and none is
// defined in this repo yet.
bool rig_R_for(const YAML::Node& rig, const std::string& cam, double R[9]) {
  if (rig["cameras"] && rig["cameras"][cam] && rig["cameras"][cam]["q_wxyz"]) {
    quat_to_R(rig["cameras"][cam], R);
    return true;
  }
  if (rig["rig_in_cam1"] && rig["rig_in_cam1"][cam]) {
    const YAML::Node m = rig["rig_in_cam1"][cam];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R[3*r + c] = m[r][c].as<double>();
    return true;
  }
  return false;
}
}  // namespace

class PanoramaNode : public rclcpp::Node {
 public:
  PanoramaNode() : Node("bev_panorama") {
    // Defaults follow the IMX296 rig: 1456x1088 native, Mei intrinsics from
    // config/calib/imx296_1456x1088 (calib_session 20260901) and the ring-closed
    // rig_extrinsics_imx296.yaml. The previous defaults pointed at the RETIRED IMX219 set
    // (scripts/config/1640x1232, KANNALA_BRANDT) - wrong sensor, wrong resolution, wrong
    // model - which is why the stitch never registered. That set has been deleted.
    //
    // flip_180 is false here on purpose: the IMX296 intrinsics and extrinsics were solved
    // on the RAW, already-inverted frames, so the roll must not be applied a second time.
    // The IMX219 lineage (rig_extrinsics_calibrated.yaml) needed flip_180: true.
    //
    // cam1..4 = ports c,d,e,f; Argus indices compact over present sensors (port b absent):
    // a=0,c=1,d=2,e=3,f=4, so ports c,d,e,f = sensor_ids [1,2,3,4].
    calib_dir_ = declare_parameter<std::string>("calib_dir", "config/calib/imx296_1456x1088");
    rig_path_ = declare_parameter<std::string>("rig_extrinsics", "config/rig/rig_extrinsics_imx296.yaml");
    cams_ = declare_parameter<std::vector<std::string>>("cameras", {"cam1", "cam2", "cam3", "cam4"});
    sensor_ids_ = declare_parameter<std::vector<int64_t>>("sensor_ids", {1, 2, 3, 4});
    cam_w_ = declare_parameter<int>("width", 1456);    // capture/output res (matches calib)
    cam_h_ = declare_parameter<int>("height", 1088);
    sensor_w_ = declare_parameter<int>("sensor_width", 1456);
    sensor_h_ = declare_parameter<int>("sensor_height", 1088);
    fps_ = declare_parameter<int>("fps", 60);
    out_w_ = declare_parameter<int>("pano_width", 1920);
    out_h_ = declare_parameter<int>("pano_height", 540);
    el_max_deg_ = declare_parameter<double>("elevation_max_deg", 50.0);
    fov_max_deg_ = declare_parameter<double>("fisheye_fov_half_deg", 80.0);
    feather_deg_ = declare_parameter<double>("feather_deg", 15.0);
    roll180_ = declare_parameter<bool>("flip_180", false);  // IMX296 calib is already in the raw (inverted) frame
    save_video_ = declare_parameter<std::string>("save_video", "");
    if (cams_.size() != 4 || sensor_ids_.size() != 4)
      throw std::runtime_error("panorama node is wired for exactly 4 cameras");

    pub_ = create_publisher<sensor_msgs::msg::Image>("bev/panorama", rclcpp::SensorDataQoS());
    if (cudaFree(0) != cudaSuccess) throw std::runtime_error("cudaFree(0) failed");

    build_maps();
    if (!setup_argus()) throw std::runtime_error("Argus setup failed");
    if (!save_video_.empty()) {
      vw_.open(save_video_, cv::VideoWriter::fourcc('m','p','4','v'), 15.0, cv::Size(out_w_, out_h_), false);
      if (!vw_.isOpened()) RCLCPP_WARN(get_logger(), "could not open video %s", save_video_.c_str());
      else RCLCPP_INFO(get_logger(), "recording panorama -> %s", save_video_.c_str());
    }
    running_ = true;
    worker_ = std::thread([this] { loop(); });
    RCLCPP_INFO(get_logger(), "panorama up: 4x%dx%d fisheye -> %dx%d equirect (GPU stitch)",
                cam_w_, cam_h_, out_w_, out_h_);
  }

  ~PanoramaNode() override {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    if (vw_.isOpened()) vw_.release();
    for (auto& s : sessions_)
      if (auto* is = interface_cast<ICaptureSession>(s.get())) is->stopRepeat();
    for (size_t i = 0; i < 4; ++i) {
      if (cu_res_[i]) cuGraphicsUnregisterResource(cu_res_[i]);
      if (egl_img_[i] != EGL_NO_IMAGE_KHR) NvDestroyEGLImage(egl_display_, egl_img_[i]);
      if (dmabuf_[i] != -1) NvBufferDestroy(dmabuf_[i]);
      if (d_uv_[i]) cudaFree(d_uv_[i]);
      if (d_w_[i]) cudaFree(d_w_[i]);
    }
    if (d_out_) cudaFree(d_out_);
    if (egl_display_ != EGL_NO_DISPLAY) eglTerminate(egl_display_);
  }

 private:
  // Precompute, per camera, the equirect remap: per output pixel -> fisheye (u,v) + feather weight.
  void build_maps() {
    YAML::Node rig = load_yaml(rig_path_);
    const double el_max = el_max_deg_ * M_PI / 180.0;
    const double fov_max = fov_max_deg_ * M_PI / 180.0;
    const double feather = feather_deg_ * M_PI / 180.0;
    const size_t N = (size_t)out_w_ * out_h_;
    std::vector<float> huv(2 * N), hw(N);

    for (size_t ci = 0; ci < 4; ++ci) {
      Mei kb = load_mei(calib_dir_ + "/" + cams_[ci] + ".yaml");
      if (kb.w != cam_w_ || kb.h != cam_h_)
        RCLCPP_WARN(get_logger(), "%s calib is %dx%d but capture is %dx%d", cams_[ci].c_str(), kb.w, kb.h, cam_w_, cam_h_);
      double R[9];
      if (!rig_R_for(rig, cams_[ci], R))
        throw std::runtime_error(rig_path_ + ": no rotation for " + cams_[ci] +
                                 " (expected cameras.<cam>.q_wxyz or rig_in_cam1.<cam>)");
      int covered = 0;
      for (int y = 0; y < out_h_; ++y) {
        double el = el_max - (2.0 * el_max) * (y + 0.5) / out_h_;        // top=+el_max
        for (int x = 0; x < out_w_; ++x) {
          double az = (2.0 * M_PI) * (x + 0.5) / out_w_ - M_PI;          // center=0=forward(+Y)
          // ray in rig frame: X=right, Y=forward, Z=up
          double dr[3] = {std::sin(az) * std::cos(el), std::cos(az) * std::cos(el), std::sin(el)};
          // d_cam = R^T * d_rig
          double X = R[0]*dr[0] + R[3]*dr[1] + R[6]*dr[2];
          double Y = R[1]*dr[0] + R[4]*dr[1] + R[7]*dr[2];
          double Z = R[2]*dr[0] + R[5]*dr[1] + R[8]*dr[2];
          // Cameras are mounted upside-down (the extrinsics assume image-up=+Z, but the modules
          // are rolled 180 deg about the optical axis). Apply that roll in-camera-frame (X,Y -> -X,-Y)
          // so we sample the correct pixel -> upright, correctly-placed panorama (no output rotation).
          if (roll180_) { X = -X; Y = -Y; }
          size_t idx = (size_t)y * out_w_ + x;
          float wgt = 0.0f, uu = -1.0f, vv = -1.0f;
          // No `Z > 0` gate any more: on a >180 deg lens the useful image extends past
          // 90 deg incidence, and the unified model represents it. mei_project() reports
          // the true incidence angle, so the FOV cut and the feather are taken against a
          // quantity that stays meaningful out there.
          double u, v, th;
          if (mei_project(kb, X, Y, Z, &u, &v, &th) && th < fov_max) {
            if (u >= 0 && v >= 0 && u <= kb.w - 1 && v <= kb.h - 1) {
              uu = (float)u; vv = (float)v;
              double fw = (fov_max - th) / feather;                    // feather toward FOV edge
              wgt = (float)std::max(0.0, std::min(1.0, fw));
              if (wgt > 0) ++covered;
            }
          }
          huv[2*idx] = uu; huv[2*idx+1] = vv; hw[idx] = wgt;
        }
      }
      RCLCPP_INFO(get_logger(), "%s: %d/%zu output px covered", cams_[ci].c_str(), covered, N);
      cudaMalloc(&d_uv_[ci], 2 * N * sizeof(float));
      cudaMalloc(&d_w_[ci], N * sizeof(float));
      cudaMemcpy(d_uv_[ci], huv.data(), 2 * N * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_w_[ci], hw.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_out_, N);
    pano_.create(out_h_, out_w_, CV_8UC1);
  }

  bool init_egl() {
    auto qd = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto gpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (qd && gpd) {
      EGLDeviceEXT devs[8]; EGLint n = 0;
      if (qd(8, devs, &n) && n > 0)
        for (EGLint d = 0; d < n; ++d) {
          EGLDisplay dpy = gpd(EGL_PLATFORM_DEVICE_EXT, devs[d], nullptr);
          if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) { egl_display_ = dpy; return true; }
        }
    }
    RCLCPP_ERROR(get_logger(), "no usable EGLDisplay"); return false;
  }

  bool setup_argus() {
    if (!init_egl()) return false;
    provider_ = UniqueObj<CameraProvider>(CameraProvider::create());
    auto* ip = interface_cast<ICameraProvider>(provider_.get());
    if (!ip) return false;
    std::vector<CameraDevice*> devs; ip->getCameraDevices(&devs);
    RCLCPP_INFO(get_logger(), "Argus %s, %zu cameras", ip->getVersion().c_str(), devs.size());
    sessions_.resize(4); streams_.resize(4); requests_.resize(4); consumers_.resize(4);
    for (size_t i = 0; i < 4; ++i) {
      int id = sensor_ids_[i];
      if (id < 0 || id >= (int)devs.size()) { RCLCPP_ERROR(get_logger(), "sensor %d absent", id); return false; }
      sessions_[i].reset(ip->createCaptureSession(devs[id]));
      auto* is = interface_cast<ICaptureSession>(sessions_[i].get());
      if (!is) return false;
      UniqueObj<OutputStreamSettings> ss(is->createOutputStreamSettings(STREAM_TYPE_EGL));
      auto* iss = interface_cast<IEGLOutputStreamSettings>(ss.get());
      iss->setEGLDisplay(egl_display_);
      iss->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
      iss->setResolution(Size2D<uint32_t>(cam_w_, cam_h_));
      iss->setMetadataEnable(true);
      streams_[i].reset(is->createOutputStream(ss.get()));
      consumers_[i].reset(EGLStream::FrameConsumer::create(streams_[i].get()));
      requests_[i].reset(is->createRequest());
      interface_cast<IRequest>(requests_[i].get())->enableOutputStream(streams_[i].get());
      auto* iprops = interface_cast<ICameraProperties>(devs[id]);
      std::vector<SensorMode*> modes; iprops->getAllSensorModes(&modes);
      for (auto* m : modes) {
        auto* im = interface_cast<ISensorMode>(m);
        if ((int)im->getResolution().width() == sensor_w_ && (int)im->getResolution().height() == sensor_h_) {
          interface_cast<ISourceSettings>(requests_[i].get())->setSensorMode(m); break;
        }
      }
      interface_cast<ISourceSettings>(requests_[i].get())->setFrameDurationRange(Range<uint64_t>(1e9 / fps_));
      is->repeat(requests_[i].get());
    }
    return true;
  }

  bool ensure_gpu_buffer(size_t i, EGLStream::NV::IImageNativeBuffer* inb) {
    if (dmabuf_[i] != -1) { inb->copyToNvBuffer(dmabuf_[i]); return true; }
    dmabuf_[i] = inb->createNvBuffer(Size2D<uint32_t>(cam_w_, cam_h_), NvBufferColorFormat_YUV420, NvBufferLayout_Pitch);
    if (dmabuf_[i] < 0) return false;
    egl_img_[i] = NvEGLImageFromFd(egl_display_, dmabuf_[i]);
    if (egl_img_[i] == EGL_NO_IMAGE_KHR) return false;
    if (cuGraphicsEGLRegisterImage(&cu_res_[i], egl_img_[i], CU_GRAPHICS_REGISTER_FLAGS_NONE) != CUDA_SUCCESS) return false;
    CUeglFrame ef;
    if (cuGraphicsResourceGetMappedEglFrame(&ef, cu_res_[i], 0, 0) != CUDA_SUCCESS) return false;
    dev_y_[i] = ef.frame.pPitch[0]; dev_pitch_[i] = ef.pitch;
    return true;
  }

  void loop() {
    cudaSetDevice(0); cudaFree(0);  // bind primary context on this thread (for the EGL/CUDA bridge)
    std::array<EGLStream::IFrameConsumer*, 4> ifc;
    for (size_t i = 0; i < 4; ++i) ifc[i] = interface_cast<EGLStream::IFrameConsumer>(consumers_[i].get());
    const size_t N = (size_t)out_w_ * out_h_;
    while (running_ && rclcpp::ok()) {
      bool ok = true;
      for (size_t i = 0; i < 4 && ok; ++i) {
        UniqueObj<EGLStream::Frame> frame(ifc[i]->acquireFrame(1000000000));
        auto* iframe = interface_cast<EGLStream::IFrame>(frame.get());
        if (!iframe) { ok = false; break; }
        auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(iframe->getImage());
        if (!inb || !ensure_gpu_buffer(i, inb)) { ok = false; break; }
      }
      if (!ok) continue;
      launch_equirect_stitch(d_out_, out_w_, out_h_,
          dev_y_[0], dev_y_[1], dev_y_[2], dev_y_[3], (int)dev_pitch_[0], cam_w_, cam_h_,
          d_uv_[0], d_uv_[1], d_uv_[2], d_uv_[3], d_w_[0], d_w_[1], d_w_[2], d_w_[3]);
      if (cudaMemcpy(pano_.data, d_out_, N, cudaMemcpyDeviceToHost) != cudaSuccess) continue;

      auto msg = std::make_unique<sensor_msgs::msg::Image>();
      msg->header.stamp = now(); msg->header.frame_id = "rig";
      msg->width = out_w_; msg->height = out_h_; msg->encoding = "mono8";
      msg->step = out_w_; msg->data.assign(pano_.data, pano_.data + N);
      pub_->publish(std::move(msg));
      if (vw_.isOpened()) vw_.write(pano_);
    }
  }

  std::string calib_dir_, rig_path_, save_video_;
  std::vector<std::string> cams_;
  std::vector<int64_t> sensor_ids_;
  int cam_w_, cam_h_, sensor_w_, sensor_h_, fps_, out_w_, out_h_;
  double el_max_deg_, fov_max_deg_, feather_deg_;
  bool roll180_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  cv::VideoWriter vw_; cv::Mat pano_;

  UniqueObj<CameraProvider> provider_;
  std::vector<UniqueObj<CaptureSession>> sessions_;
  std::vector<UniqueObj<OutputStream>> streams_;
  std::vector<UniqueObj<Request>> requests_;
  std::vector<UniqueObj<EGLStream::FrameConsumer>> consumers_;
  EGLDisplay egl_display_ = EGL_NO_DISPLAY;
  std::array<int, 4> dmabuf_{-1, -1, -1, -1};
  std::array<EGLImageKHR, 4> egl_img_{EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR};
  std::array<CUgraphicsResource, 4> cu_res_{nullptr, nullptr, nullptr, nullptr};
  std::array<void*, 4> dev_y_{nullptr, nullptr, nullptr, nullptr};
  std::array<uint32_t, 4> dev_pitch_{0, 0, 0, 0};
  std::array<void*, 4> d_uv_{nullptr, nullptr, nullptr, nullptr};
  std::array<void*, 4> d_w_{nullptr, nullptr, nullptr, nullptr};
  uint8_t* d_out_ = nullptr;
  std::atomic<bool> running_{false};
  std::thread worker_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  std::signal(SIGTERM, [](int) { rclcpp::shutdown(); });
  auto node = std::make_shared<PanoramaNode>();
  while (rclcpp::ok()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  node.reset();
  rclcpp::shutdown();
  return 0;
}
