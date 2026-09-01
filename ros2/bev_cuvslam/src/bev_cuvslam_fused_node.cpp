// Fused zero-copy Argus -> cuVSLAM VO node (ROS 2 Foxy, TX2/J106).
//
// One process captures N IMX219 via libargus and runs cuVSLAM Track() on the frames
// AS GPU MEMORY — Argus NVMM Y(luma) plane is bridged to a CUDA device pointer
// (NvEGLImageFromFd -> cuGraphicsEGLRegisterImage), so there is NO host copy and NO
// DDS image transport on the tracking path (cf. the modular capture+VO pipeline's
// Argus->NVMM->CPU->DDS->GPU round-trip). Publishes only /cuvslam/odometry + TF.
//
// Bridge validated by scripts/port/egl_cuda_spike.cpp (device Y plane == CPU path).
// Build/run inside cuvslam-foxy:tx2 (Argus socket + /dev + jetson_multimedia_api + CUDA).
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

// cuVSLAM header FIRST: the EGL/X11 headers below #define Success/None/Status/Bool as
// macros, which would clobber cuVSLAM's Result<T>::Success() (cuvslam2.h:627).
#include "cuvslam/cuvslam2.h"

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
// Undo X11 macro pollution so it can't bite later TUs/identifiers.
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
cuvslam::Camera load_intrinsics(const std::string& path) {
  YAML::Node y = load_yaml(path);
  cuvslam::Camera c;
  c.size = {y["image_width"].as<int>(), y["image_height"].as<int>()};
  YAML::Node pp = y["projection_parameters"];
  c.focal = {pp["mu"].as<float>(), pp["mv"].as<float>()};
  c.principal = {pp["u0"].as<float>(), pp["v0"].as<float>()};
  YAML::Node dp = y["distortion_parameters"];
  c.distortion.model = cuvslam::Distortion::Model::Fisheye;
  c.distortion.parameters = {dp["k2"].as<float>(), dp["k3"].as<float>(),
                             dp["k4"].as<float>(), dp["k5"].as<float>()};
  return c;
}
cuvslam::Pose load_pose(const YAML::Node& n) {
  cuvslam::Pose p;
  auto t = n["t_xyz_m"]; p.translation = {t[0].as<float>(), t[1].as<float>(), t[2].as<float>()};
  auto q = n["q_wxyz"];  p.rotation = {q[1].as<float>(), q[2].as<float>(), q[3].as<float>(), q[0].as<float>()};
  return p;
}
}  // namespace

class FusedNode : public rclcpp::Node {
 public:
  FusedNode() : Node("bev_cuvslam_fused") {
    // Default = the measured best full-FOV config (see openspec fused-zerocopy §6): capture the
    // full-FOV 1640x1232 mode and ISP-downscale to 832x624 output → ~22 Hz (the sensor's fps
    // ceiling), full surround FOV, lowest CPU. Override params for other configs.
    calib_dir_ = declare_parameter<std::string>("calib_dir", "scripts/config/832x624");
    rig_path_ = declare_parameter<std::string>("rig_extrinsics", "config/rig/rig_extrinsics_vo.yaml");
    cams_ = declare_parameter<std::vector<std::string>>("cameras", {"cam1", "cam2", "cam3", "cam4"});
    sensor_ids_ = declare_parameter<std::vector<int64_t>>("sensor_ids", {1, 2, 3, 4});
    width_ = declare_parameter<int>("width", 832);     // OUTPUT resolution (-> cuVSLAM + calib)
    height_ = declare_parameter<int>("height", 624);
    // Sensor mode to capture; larger than output → Argus ISP downscales in NVMM (stays zero-copy).
    sensor_width_ = declare_parameter<int>("sensor_width", 1640);
    sensor_height_ = declare_parameter<int>("sensor_height", 1232);
    fps_ = declare_parameter<int>("fps", 60);          // request high; the sensor mode caps it (1640x1232 → 22 fps)
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    // NOT "base_link", and the difference matters. cuVSLAM reports world_from_rig, and
    // this node's rig frame IS cam1's optical frame (z forward, x right, y down),
    // additionally rolled 180 deg by the inverted mount. Publishing that as base_link
    // would tell every tf consumer it is FLU on the vehicle, which it is not - and a
    // 180 deg roll produces trajectories that look entirely plausible. Publishing a
    // true base_link needs R_body_from_cam1, which is not measured; see
    // config/rig/rig_layout.yaml and 3R.16b. Override the parameter only once it is.
    base_frame_ = declare_parameter<std::string>("base_frame", "cam1_optical_frame");
    if (cams_.size() != 4 || sensor_ids_.size() != 4)
      throw std::runtime_error("fused node is wired for exactly 4 cameras");

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("cuvslam/odometry", 10);
    tf_bc_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // Primary CUDA context current BEFORE any driver-API EGL/CUDA interop, shared with
    // cuVSLAM's runtime context (validated in the bridge spike).
    if (cudaFree(0) != cudaSuccess) throw std::runtime_error("cudaFree(0) failed");
    cuInit(0);

    build_tracker();
    if (!setup_argus()) throw std::runtime_error("Argus setup failed");
    running_ = true;
    worker_ = std::thread([this] { capture_loop(); });
    RCLCPP_INFO(get_logger(), "fused Argus->cuVSLAM VO up: 4 cameras, GPU zero-copy, mode=Multicamera.");
  }

  ~FusedNode() override {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    // Release Argus so the session doesn't leak in nvargus-daemon (a leaked session wedges
    // the daemon -> "no session" next run). stopRepeat() only — waitForIdle() can block and
    // hang shutdown; the UniqueObj session/stream dtors below finish teardown.
    for (auto& s : sessions_)
      if (auto* is = interface_cast<ICaptureSession>(s.get())) is->stopRepeat();
    for (size_t i = 0; i < 4; ++i) {
      if (cu_res_[i]) cuGraphicsUnregisterResource(cu_res_[i]);
      if (egl_img_[i] != EGL_NO_IMAGE_KHR) NvDestroyEGLImage(egl_display_, egl_img_[i]);
      if (dmabuf_[i] != -1) NvBufferDestroy(dmabuf_[i]);
    }
    if (egl_display_ != EGL_NO_DISPLAY) eglTerminate(egl_display_);
  }

 private:
  void build_tracker() {
    int mj, mn, pt; cuvslam::GetVersion(&mj, &mn, &pt);
    RCLCPP_INFO(get_logger(), "cuVSLAM %d.%d.%d — warming up GPU...", mj, mn, pt);
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
    cuvslam::Odometry::Config cfg = cuvslam::Odometry::GetDefaultConfig();
    cfg.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
    cfg.multicam_mode = cuvslam::Odometry::MulticameraMode::Precision;
    cfg.use_gpu = true;
    tracker_ = std::make_unique<cuvslam::Odometry>(rig, cfg);
  }

  bool init_egl() {
    auto qd = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto gpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (qd && gpd) {
      EGLDeviceEXT devs[8]; EGLint n = 0;
      if (qd(8, devs, &n) && n > 0)
        for (EGLint d = 0; d < n; ++d) {
          EGLDisplay dpy = gpd(EGL_PLATFORM_DEVICE_EXT, devs[d], nullptr);
          if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) {
            egl_display_ = dpy;
            RCLCPP_INFO(get_logger(), "EGL headless display via device %d/%d", d, n);
            return true;
          }
        }
    }
    RCLCPP_ERROR(get_logger(), "no usable EGLDisplay");
    return false;
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
      iss->setResolution(Size2D<uint32_t>(width_, height_));
      iss->setMetadataEnable(true);
      streams_[i].reset(is->createOutputStream(ss.get()));
      consumers_[i].reset(EGLStream::FrameConsumer::create(streams_[i].get()));
      requests_[i].reset(is->createRequest());
      interface_cast<IRequest>(requests_[i].get())->enableOutputStream(streams_[i].get());
      auto* iprops = interface_cast<ICameraProperties>(devs[id]);
      std::vector<SensorMode*> modes; iprops->getAllSensorModes(&modes);
      bool matched = false;
      for (auto* m : modes) {
        auto* im = interface_cast<ISensorMode>(m);
        if ((int)im->getResolution().width() == sensor_width_ && (int)im->getResolution().height() == sensor_height_) {
          interface_cast<ISourceSettings>(requests_[i].get())->setSensorMode(m); matched = true; break;
        }
      }
      if (!matched)
        RCLCPP_WARN(get_logger(), "no sensor mode %dx%d for cam %zu; using default (ISP scales to %dx%d output)",
                    sensor_width_, sensor_height_, i, width_, height_);
      else if (i == 0)
        RCLCPP_INFO(get_logger(), "sensor mode %dx%d -> output %dx%d", sensor_width_, sensor_height_, width_, height_);
      interface_cast<ISourceSettings>(requests_[i].get())->setFrameDurationRange(Range<uint64_t>(1e9 / fps_));
      is->repeat(requests_[i].get());
    }
    return true;
  }

  // First frame for camera i: create the persistent NVMM buffer and register it ONCE as a
  // CUDA device pointer (copyToNvBuffer later updates it in place; the cached ptr stays valid).
  bool ensure_gpu_buffer(size_t i, EGLStream::NV::IImageNativeBuffer* inb) {
    if (dmabuf_[i] != -1) { inb->copyToNvBuffer(dmabuf_[i]); return true; }
    dmabuf_[i] = inb->createNvBuffer(Size2D<uint32_t>(width_, height_),
                                     NvBufferColorFormat_YUV420, NvBufferLayout_Pitch);
    if (dmabuf_[i] < 0) { RCLCPP_ERROR(get_logger(), "createNvBuffer cam %zu failed", i); return false; }
    egl_img_[i] = NvEGLImageFromFd(egl_display_, dmabuf_[i]);
    if (egl_img_[i] == EGL_NO_IMAGE_KHR) { RCLCPP_ERROR(get_logger(), "NvEGLImageFromFd cam %zu failed", i); return false; }
    CUresult rr = cuGraphicsEGLRegisterImage(&cu_res_[i], egl_img_[i], CU_GRAPHICS_REGISTER_FLAGS_NONE);
    if (rr != CUDA_SUCCESS) {
      const char* s = nullptr; cuGetErrorString(rr, &s);
      RCLCPP_ERROR(get_logger(), "cuGraphicsEGLRegisterImage cam %zu failed: %s", i, s ? s : "?"); return false;
    }
    CUeglFrame ef;
    if (cuGraphicsResourceGetMappedEglFrame(&ef, cu_res_[i], 0, 0) != CUDA_SUCCESS) return false;
    dev_y_[i] = ef.frame.pPitch[0];
    dev_pitch_[i] = ef.pitch;
    RCLCPP_INFO(get_logger(), "%s GPU buffer registered: dev_ptr=%p pitch=%u", cams_[i].c_str(), dev_y_[i], dev_pitch_[i]);
    return true;
  }

  void capture_loop() {
    // CUDA context currency is per-thread: bind the runtime primary context to THIS worker
    // thread before any driver-API EGL/CUDA interop (the constructor's cudaFree(0) only bound
    // the main thread). Shares cuVSLAM's runtime context, so our device ptrs are valid in Track.
    cudaSetDevice(0);
    cudaFree(0);
    std::array<EGLStream::IFrameConsumer*, 4> ifc;
    for (size_t i = 0; i < 4; ++i) ifc[i] = interface_cast<EGLStream::IFrameConsumer>(consumers_[i].get());
    std::array<bool, 4> first; first.fill(true);
    using ms = std::chrono::duration<double, std::milli>;
    double acc_acq = 0, acc_trk = 0; int acc_n = 0;          // windowed timing accumulators
    auto win0 = std::chrono::steady_clock::now();            // -> avg over the window, not a noisy single sample

    while (running_ && rclcpp::ok()) {
      const auto t_loop = std::chrono::steady_clock::now();
      int64_t ts0 = 0; bool ok = true;
      for (size_t i = 0; i < 4 && ok; ++i) {
        UniqueObj<EGLStream::Frame> frame(ifc[i]->acquireFrame(1000000000));
        auto* iframe = interface_cast<EGLStream::IFrame>(frame.get());
        if (!iframe) { ok = false; break; }
        auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(iframe->getImage());
        if (!inb || !ensure_gpu_buffer(i, inb)) { ok = false; break; }
        if (i == 0) ts0 = (int64_t)iframe->getTime();
        if (first[i]) { RCLCPP_INFO(get_logger(), "%s first GPU frame", cams_[i].c_str()); first[i] = false; }
      }
      if (!ok) continue;
      const auto t_acq = std::chrono::steady_clock::now();  // after 4-cam acquire (+GPU copy)

      // Build the cuVSLAM ImageSet from the 4 GPU Y planes (unified timestamp = cam0's).
      cuvslam::Odometry::ImageSet images; images.reserve(4);
      for (uint32_t i = 0; i < 4; ++i) {
        cuvslam::Image im{};
        im.pixels = dev_y_[i];
        im.width = width_;
        im.height = height_;
        im.pitch = (int32_t)dev_pitch_[i];
        im.encoding = cuvslam::ImageData::Encoding::MONO;
        im.data_type = cuvslam::ImageData::DataType::UINT8;
        im.is_gpu_mem = true;
        im.timestamp_ns = ts0;
        im.camera_index = i;
        images.push_back(im);
      }
      cuvslam::PoseEstimate est;
      try {
        est = tracker_->Track(images);
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Track() failed: %s", e.what());
        continue;
      }
      const auto t_trk = std::chrono::steady_clock::now();  // after Track()
      // Accumulate and report WINDOWED AVERAGES (publishing one pose per loop, so this rate
      // == sustained odom rate; it cannot exceed the camera fps). Single-iteration prints
      // were misleading (a queue-drain burst can momentarily exceed input fps).
      acc_acq += ms(t_acq - t_loop).count();
      acc_trk += ms(t_trk - t_acq).count();
      ++acc_n;
      const double win = ms(t_trk - win0).count();
      if (win >= 2000.0) {
        RCLCPP_INFO(get_logger(), "avg/%.1fs: %.1f Hz (n=%d)  acquire+gpucopy=%.1f ms  Track=%.1f ms",
                    win / 1000.0, acc_n * 1000.0 / win, acc_n, acc_acq / acc_n, acc_trk / acc_n);
        acc_acq = acc_trk = 0; acc_n = 0; win0 = t_trk;
      }
      if (!est.world_from_rig) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "tracking lost (no pose)");
        continue;
      }
      publish(*est.world_from_rig, ts0);
    }
  }

  void publish(const cuvslam::PoseWithCovariance& pwc, int64_t ts_ns) {
    const cuvslam::Pose& p = pwc.pose;
    nav_msgs::msg::Odometry od;
    od.header.stamp = rclcpp::Time(ts_ns);
    od.header.frame_id = odom_frame_;
    od.child_frame_id = base_frame_;
    od.pose.pose.position.x = p.translation[0];
    od.pose.pose.position.y = p.translation[1];
    od.pose.pose.position.z = p.translation[2];
    od.pose.pose.orientation.x = p.rotation[0];
    od.pose.pose.orientation.y = p.rotation[1];
    od.pose.pose.orientation.z = p.rotation[2];
    od.pose.pose.orientation.w = p.rotation[3];
    static constexpr int perm[6] = {3, 4, 5, 0, 1, 2};  // cuVSLAM [Rx,Ry,Rz,x,y,z] -> ROS [x,y,z,Rx,Ry,Rz]
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        od.pose.covariance[r * 6 + c] = pwc.covariance[perm[r] * 6 + perm[c]];
    odom_pub_->publish(od);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = od.header.stamp;
    tf.header.frame_id = odom_frame_;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = p.translation[0];
    tf.transform.translation.y = p.translation[1];
    tf.transform.translation.z = p.translation[2];
    tf.transform.rotation = od.pose.pose.orientation;
    tf_bc_->sendTransform(tf);
  }

  std::string calib_dir_, rig_path_, odom_frame_, base_frame_;
  std::vector<std::string> cams_;
  std::vector<int64_t> sensor_ids_;
  int width_, height_, sensor_width_, sensor_height_, fps_;
  std::unique_ptr<cuvslam::Odometry> tracker_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_bc_;

  // Argus
  UniqueObj<CameraProvider> provider_;
  std::vector<UniqueObj<CaptureSession>> sessions_;
  std::vector<UniqueObj<OutputStream>> streams_;
  std::vector<UniqueObj<Request>> requests_;
  std::vector<UniqueObj<EGLStream::FrameConsumer>> consumers_;
  EGLDisplay egl_display_ = EGL_NO_DISPLAY;

  // Per-camera GPU bridge (registered once, updated in place by copyToNvBuffer)
  std::array<int, 4> dmabuf_{-1, -1, -1, -1};
  std::array<EGLImageKHR, 4> egl_img_{EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR, EGL_NO_IMAGE_KHR};
  std::array<CUgraphicsResource, 4> cu_res_{nullptr, nullptr, nullptr, nullptr};
  std::array<void*, 4> dev_y_{nullptr, nullptr, nullptr, nullptr};
  std::array<uint32_t, 4> dev_pitch_{0, 0, 0, 0};

  std::atomic<bool> running_{false};
  std::thread worker_;
};

namespace { volatile std::sig_atomic_t g_stop = 0; }

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  // This node has no ROS callbacks (the worker thread captures, tracks, and publishes), so
  // there's nothing to spin — just wait until shutdown. rclcpp handles SIGINT (-> !ok());
  // catch SIGTERM (docker stop) with a flag (calling rclcpp::shutdown() from a signal handler
  // can deadlock). Either way we destruct the node and release Argus/EGL/CUDA promptly instead
  // of being SIGKILL'd after the grace period (a leaked Argus session wedges nvargus-daemon).
  std::signal(SIGTERM, [](int) { g_stop = 1; });
  auto node = std::make_shared<FusedNode>();
  while (rclcpp::ok() && !g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  node.reset();  // run the destructor (clean Argus/EGL/CUDA release) before context shutdown
  rclcpp::shutdown();
  return 0;
}
