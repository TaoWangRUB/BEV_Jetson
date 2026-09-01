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
#include <algorithm>
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
#include "bev_cuvslam/rig_build.hpp"
#include "bev_cuvslam/virtual_pinhole_gpu.h"

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
}  // namespace

class FusedNode : public rclcpp::Node {
 public:
  FusedNode() : Node("bev_cuvslam_fused") {
    // Default = the measured best full-FOV config (see openspec fused-zerocopy §6): capture the
    // full-FOV 1640x1232 mode and ISP-downscale to 832x624 output → ~22 Hz (the sensor's fps
    // ceiling), full surround FOV, lowest CPU. Override params for other configs.
    calib_dir_ = declare_parameter<std::string>("calib_dir", "config/calib/imx296_1456x1088");
    // rig_extrinsics_imx296.yaml, NOT rig_extrinsics_vo.yaml: the latter belongs to the
    // panorama lineage and has the 180 deg mount roll folded in. These extrinsics are solved
    // directly from the frames the sensor delivers, so folding it in again puts every camera
    // 180 deg out (3R.16).
    rig_path_ = declare_parameter<std::string>("rig_extrinsics", "config/rig/rig_extrinsics_imx296.yaml");
    vstereo_path_ = declare_parameter<std::string>("virtual_stereo", "config/rig/virtual_stereo_imx296.yaml");
    max_skew_ns_ = (int64_t)declare_parameter<int>("max_skew_us", 1000) * 1000;
    cams_ = declare_parameter<std::vector<std::string>>("cameras", {"cam1", "cam2", "cam3", "cam4"});
    sensor_ids_ = declare_parameter<std::vector<int64_t>>("sensor_ids", {1, 2, 3, 4});
    // NATIVE, and not negotiable: this is the resolution the intrinsics were solved at, and
    // the source of the carve. Downscaling here would silently invalidate camN.yaml. The old
    // 832x624/1640x1232 defaults were the IMX219 rig's.
    width_ = declare_parameter<int>("width", 1456);
    height_ = declare_parameter<int>("height", 1088);
    sensor_width_ = declare_parameter<int>("sensor_width", 1456);
    sensor_height_ = declare_parameter<int>("sensor_height", 1088);
    fps_ = declare_parameter<int>("fps", 30);          // the trigger sets the real rate
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
    // Same carve, same rig, same file parsing as the modular node - bev_cuvslam/rig_build.hpp
    // is shared deliberately (see the note there). The ONLY difference between the two nodes
    // is where the gather runs: cv::remap on the CPU there, launch_vpin_remap on the GPU here.
    auto vrig = bev_cuvslam::BuildVirtualRig(calib_dir_, rig_path_, vstereo_path_, cams_);
    for (const auto& w : vrig.warnings) RCLCPP_WARN(get_logger(), "%s", w.c_str());
    vpin_ = std::move(vrig.vpin);
    vsrc_ = std::move(vrig.vsrc);
    cuvslam::Rig rig = std::move(vrig.rig);
    vw_ = vpin_[0].width; vh_ = vpin_[0].height;
    for (size_t k = 0; k < vpin_.size(); ++k)
      RCLCPP_INFO(get_logger(), "  vcam %zu = %s %+.0f deg: %dx%d f=%.1f", k,
                  cams_[vsrc_[k]].c_str(), vpin_[k].yaw_rad * 180.0 / CV_PI,
                  vpin_[k].width, vpin_[k].height, vpin_[k].focal);

    // Upload the carve tables once. These are the SAME maps the CPU path uses (the float
    // ones BuildVirtualPinhole keeps), not a second computation that could disagree.
    desc_.resize(vpin_.size());
    for (size_t k = 0; k < vpin_.size(); ++k) {
      const size_t map_bytes = (size_t)vw_ * vh_ * sizeof(float) * 2;
      void *dmap = nullptr, *ddst = nullptr;
      if (cudaMalloc(&dmap, map_bytes) != cudaSuccess ||
          cudaMalloc(&ddst, (size_t)vw_ * vh_) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed for the virtual-pinhole tables");
      if (cudaMemcpy(dmap, vpin_[k].mapf.ptr<float>(), map_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy failed uploading a carve map");
      desc_[k].map = (const float2*)dmap;
      desc_[k].dst = (uint8_t*)ddst;
      desc_[k].sw = width_; desc_[k].sh = height_;
    }
    if (cudaMalloc(&dev_desc_, desc_.size() * sizeof(VPinDesc)) != cudaSuccess)
      throw std::runtime_error("cudaMalloc failed for the descriptor array");
    RCLCPP_INFO(get_logger(), "carve tables on the GPU: %zu virtual pinholes at %dx%d",
                vpin_.size(), vw_, vh_);

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
      std::array<int64_t, 4> ts{};
      // Acquiring one frame per camera in a loop does NOT give you a set. Each consumer
      // has its own queue, and the four frames that come back can sit on different trigger
      // edges - which is exactly what happened: 2673 of 2673 sets rejected at 33.3 ms, one
      // frame period at 30 Hz, on a rig whose measured hardware skew is 8 us. The modular
      // node hit the same wall for a different reason (DDS delivery order) and the answer
      // is the same: ALIGN, then gate. Gating alone just rejects everything.
      auto acquire_one = [&](size_t i) -> bool {
        // Short timeout so shutdown is prompt: this loop blocks, and launch escalates
        // SIGINT -> SIGTERM -> SIGKILL after 5 s. A SIGKILLed run leaks an Argus session
        // and the next start fails with "Argus setup failed", which has cost time already.
        UniqueObj<EGLStream::Frame> frame(ifc[i]->acquireFrame(200000000));
        auto* iframe = interface_cast<EGLStream::IFrame>(frame.get());
        if (!iframe) return false;
        auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(iframe->getImage());
        if (!inb || !ensure_gpu_buffer(i, inb)) return false;
        // The SENSOR timestamp, not IFrame::getTime(). getTime() is consumer-side and
        // measured the capture loop's own phase - it put the four cameras ~7 ms apart in
        // visit order (README 4.7, and 4.2 flagged this node as carrying the same bug).
        uint64_t sof = 0, expo = 0;
        if (auto* iacm = interface_cast<EGLStream::IArgusCaptureMetadata>(frame.get()))
          if (auto* imeta = interface_cast<ICaptureMetadata>(iacm->getMetadata())) {
            sof = imeta->getSensorTimestamp();
            expo = imeta->getSensorExposureTime();
          }
        if (sof == 0) {
          if (!warned_no_meta_) {
            RCLCPP_WARN(get_logger(), "no capture metadata - falling back to the EGLStream "
                        "frame time, which is consumer-side and NOT synchronised");
            warned_no_meta_ = true;
          }
          sof = (uint64_t)iframe->getTime();
        }
        ts[i] = (int64_t)sof - (int64_t)expo / 2;  // exposure midpoint: what the image represents
        if (first[i]) { RCLCPP_INFO(get_logger(), "%s first GPU frame", cams_[i].c_str()); first[i] = false; }
        return true;
      };

      for (size_t i = 0; i < 4 && ok; ++i) ok = running_ && acquire_one(i);
      if (!ok) continue;

      // Advance whichever camera is behind until the four sit on one edge. Bounded, so a
      // genuinely broken camera cannot spin here - it falls through to the gate below and
      // is reported as a wide set, which is the honest outcome.
      for (int guard = 0; guard < 8 && ok && running_; ++guard) {
        const int64_t a = *std::min_element(ts.begin(), ts.end());
        const int64_t b = *std::max_element(ts.begin(), ts.end());
        if (b - a <= max_skew_ns_) break;
        for (size_t i = 0; i < 4 && ok; ++i)
          if (b - ts[i] > max_skew_ns_) ok = acquire_one(i);
      }
      ts0 = ts[0];
      if (!ok) continue;

      // Gate the set on REAL per-frame stamps, exactly as the modular node does. A set that
      // fails is dropped and counted, never re-stamped: on a triggered rig a wide set means
      // the trigger, a camera or the capture path is at fault, and the VO cannot fix it.
      const int64_t lo = *std::min_element(ts.begin(), ts.end());
      const int64_t hi = *std::max_element(ts.begin(), ts.end());
      ++sets_;
      if (hi - lo > max_skew_ns_) {
        ++dropped_sets_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "set skew %.1f ms > %.1f ms - dropped (%ld of %ld). Is the trigger running?",
            (hi - lo) / 1e6, max_skew_ns_ / 1e6, dropped_sets_, sets_);
        continue;
      }
      if (hi - lo > worst_skew_ns_) worst_skew_ns_ = hi - lo;
      const auto t_acq = std::chrono::steady_clock::now();  // after 4-cam acquire (+GPU copy)

      // Carve on the GPU. One launch for all 8 virtual cameras; the pixels never leave the
      // device the NVMM bridge handed us.
      for (size_t k = 0; k < desc_.size(); ++k) {
        desc_[k].src = (const uint8_t*)dev_y_[vsrc_[k]];
        desc_[k].spitch = (int)dev_pitch_[vsrc_[k]];
      }
      if (cudaMemcpy(dev_desc_, desc_.data(), desc_.size() * sizeof(VPinDesc),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "descriptor upload failed");
        continue;
      }
      if (launch_vpin_remap(dev_desc_, (int)desc_.size(), vw_, vh_, 0) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "virtual-pinhole carve failed");
        continue;
      }

      // The ImageSet is the 8 CARVED views, not the 4 fisheyes: cuVSLAM's only fisheye model
      // is equidistant and cannot represent these ~192 deg lenses at all (README 4.8).
      cuvslam::Odometry::ImageSet images; images.reserve(desc_.size());
      for (uint32_t k = 0; k < desc_.size(); ++k) {
        cuvslam::Image im{};
        im.pixels = desc_[k].dst;
        im.width = vw_;
        im.height = vh_;
        im.pitch = vw_;
        im.encoding = cuvslam::ImageData::Encoding::MONO;
        im.data_type = cuvslam::ImageData::DataType::UINT8;
        im.is_gpu_mem = true;
        im.timestamp_ns = ts[vsrc_[k]];
        im.camera_index = k;
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
    // Since cuVSLAM v17 the covariance is already row-major [x,y,z,Rx,Ry,Rz] (field
    // renamed to covariance_xyz_rpy) — the same order ROS Odometry wants, so copy directly.
    // (Up to v15 it was [Rx,Ry,Rz,x,y,z] and needed a {3,4,5,0,1,2} permutation.)
    for (int i = 0; i < 36; ++i) od.pose.covariance[i] = pwc.covariance_xyz_rpy[i];
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
  int vw_ = 0, vh_ = 0;
  std::string vstereo_path_;
  int64_t max_skew_ns_ = 1000000, worst_skew_ns_ = 0, sets_ = 0, dropped_sets_ = 0;
  bool warned_no_meta_ = false;
  std::vector<bev_cuvslam::VirtualPinhole> vpin_;
  std::vector<int> vsrc_;
  std::vector<VPinDesc> desc_;
  VPinDesc* dev_desc_ = nullptr;
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
