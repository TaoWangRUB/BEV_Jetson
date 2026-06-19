// 4-camera Argus capture node for ROS 2 Foxy (TX2/J106 BEV rig).
//
// Uses the libargus C++ API directly (the nvidia runtime mounts libnvargus into
// the container), so it needs no tegra gstreamer plugin. Opens N IMX219 cameras
// at a chosen sensor mode, acquires frames, extracts each frame's luma (Y) plane
// — which is exactly the grayscale image cuVSLAM wants — and publishes it as
// sensor_msgs/Image (mono8) on /camN/image_raw with the capture timestamp.
//
// Build/run inside cuvslam-foxy:tx2 with the Argus socket + /dev mounted, and the
// jetson_multimedia_api headers bind-mounted for the include path.

#include <atomic>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/FrameConsumer.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <nvbuf_utils.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

using namespace Argus;

class ArgusCaptureNode : public rclcpp::Node {
 public:
  ArgusCaptureNode() : Node("argus_capture") {
    sensor_ids_ = declare_parameter<std::vector<int64_t>>("sensor_ids", {0, 1, 2, 3});
    topics_ = declare_parameter<std::vector<std::string>>(
        "topics", {"/cam1/image_raw", "/cam2/image_raw", "/cam3/image_raw", "/cam4/image_raw"});
    frame_ids_ = declare_parameter<std::vector<std::string>>(
        "frame_ids", {"cam1", "cam2", "cam3", "cam4"});
    width_ = declare_parameter<int>("width", 1640);
    height_ = declare_parameter<int>("height", 1232);
    fps_ = declare_parameter<int>("fps", 20);
    n_ = sensor_ids_.size();

    for (size_t i = 0; i < n_; ++i)
      pubs_.push_back(create_publisher<sensor_msgs::msg::Image>(topics_[i], 10));

    if (!setup_argus())
      throw std::runtime_error("Argus setup failed");
    running_ = true;
    worker_ = std::thread([this] { capture_loop(); });
    RCLCPP_INFO(get_logger(), "Argus capture up: %zu cameras @ %dx%d", n_, width_, height_);
  }

  ~ArgusCaptureNode() override {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    for (auto fd : dmabufs_) if (fd != -1) NvBufferDestroy(fd);
    if (egl_display_ != EGL_NO_DISPLAY) eglTerminate(egl_display_);
  }

 private:
  // Headless EGLDisplay: in a container there is no window system, so Argus's
  // fallback eglGetDisplay(EGL_DEFAULT_DISPLAY) fails. Get a display straight
  // from the GPU device via EGL_EXT_platform_device (no X needed) and hand it to
  // each Argus output stream so the EGLStream consumer uses it.
  bool init_egl() {
    auto query_devices =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (query_devices && get_platform_display) {
      EGLDeviceEXT devs[8];
      EGLint n = 0;
      if (query_devices(8, devs, &n) && n > 0) {
        for (EGLint d = 0; d < n; ++d) {
          EGLDisplay dpy = get_platform_display(EGL_PLATFORM_DEVICE_EXT, devs[d], nullptr);
          if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) {
            egl_display_ = dpy;
            RCLCPP_INFO(get_logger(), "EGL headless display via device %d of %d", d, n);
            return true;
          }
        }
      }
    }
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) {
      egl_display_ = dpy;
      RCLCPP_WARN(get_logger(), "EGL using default display (needs window system)");
      return true;
    }
    RCLCPP_ERROR(get_logger(), "no usable EGLDisplay");
    return false;
  }

  bool setup_argus() {
    if (!init_egl()) return false;
    provider_ = UniqueObj<CameraProvider>(CameraProvider::create());
    auto* ip = interface_cast<ICameraProvider>(provider_.get());
    if (!ip) { RCLCPP_ERROR(get_logger(), "no ICameraProvider"); return false; }
    std::vector<CameraDevice*> devs;
    ip->getCameraDevices(&devs);
    RCLCPP_INFO(get_logger(), "Argus %s, %zu cameras present",
                ip->getVersion().c_str(), devs.size());

    sessions_.resize(n_);
    streams_.resize(n_);
    requests_.resize(n_);
    consumers_.resize(n_);
    dmabufs_.assign(n_, -1);

    for (size_t i = 0; i < n_; ++i) {
      int id = sensor_ids_[i];
      if (id < 0 || id >= (int)devs.size()) { RCLCPP_ERROR(get_logger(), "sensor-id %d absent", id); return false; }
      sessions_[i].reset(ip->createCaptureSession(devs[id]));
      auto* isession = interface_cast<ICaptureSession>(sessions_[i].get());
      if (!isession) { RCLCPP_ERROR(get_logger(), "no session for %d", id); return false; }

      UniqueObj<OutputStreamSettings> ss(isession->createOutputStreamSettings(STREAM_TYPE_EGL));
      auto* iss = interface_cast<IEGLOutputStreamSettings>(ss.get());
      iss->setEGLDisplay(egl_display_);
      iss->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
      iss->setResolution(Size2D<uint32_t>(width_, height_));
      iss->setMetadataEnable(true);
      streams_[i].reset(isession->createOutputStream(ss.get()));
      consumers_[i].reset(EGLStream::FrameConsumer::create(streams_[i].get()));

      requests_[i].reset(isession->createRequest());
      auto* ireq = interface_cast<IRequest>(requests_[i].get());
      ireq->enableOutputStream(streams_[i].get());
      // pick the sensor mode matching width/height
      auto* iprops = interface_cast<ICameraProperties>(devs[id]);
      std::vector<SensorMode*> modes; iprops->getAllSensorModes(&modes);
      for (auto* m : modes) {
        auto* im = interface_cast<ISensorMode>(m);
        if ((int)im->getResolution().width() == width_ && (int)im->getResolution().height() == height_) {
          interface_cast<ISourceSettings>(requests_[i].get())->setSensorMode(m); break;
        }
      }
      interface_cast<ISourceSettings>(requests_[i].get())->setFrameDurationRange(Range<uint64_t>(1e9 / fps_));
      isession->repeat(requests_[i].get());
    }
    return true;
  }

  void capture_loop() {
    std::vector<EGLStream::IFrameConsumer*> ifc(n_);
    for (size_t i = 0; i < n_; ++i)
      ifc[i] = interface_cast<EGLStream::IFrameConsumer>(consumers_[i].get());

    while (running_ && rclcpp::ok()) {
      for (size_t i = 0; i < n_; ++i) {
        UniqueObj<EGLStream::Frame> frame(ifc[i]->acquireFrame(1000000000));  // 1s timeout
        auto* iframe = interface_cast<EGLStream::IFrame>(frame.get());
        if (!iframe) continue;
        auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(iframe->getImage());
        if (!inb) continue;
        if (dmabufs_[i] == -1)
          dmabufs_[i] = inb->createNvBuffer(Size2D<uint32_t>(width_, height_),
                                            NvBufferColorFormat_YUV420, NvBufferLayout_Pitch);
        else
          inb->copyToNvBuffer(dmabufs_[i]);
        publish_y(i, iframe->getTime());
      }
    }
  }

  void publish_y(size_t i, uint64_t t_ns) {
    NvBufferParams params;
    if (NvBufferGetParams(dmabufs_[i], &params) != 0) return;
    void* mapped = nullptr;
    if (NvBufferMemMap(dmabufs_[i], 0, NvBufferMem_Read, &mapped) != 0 || !mapped) return;
    NvBufferMemSyncForCpu(dmabufs_[i], 0, &mapped);

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp = rclcpp::Time(static_cast<int64_t>(t_ns));
    msg->header.frame_id = frame_ids_[i];
    msg->width = width_;
    msg->height = height_;
    msg->encoding = "mono8";
    msg->is_bigendian = 0;
    msg->step = width_;
    msg->data.resize(static_cast<size_t>(width_) * height_);
    const uint32_t pitch = params.pitch[0];
    const uint8_t* src = static_cast<const uint8_t*>(mapped);
    for (int r = 0; r < height_; ++r)
      memcpy(msg->data.data() + r * width_, src + r * pitch, width_);

    NvBufferMemUnMap(dmabufs_[i], 0, &mapped);
    pubs_[i]->publish(std::move(msg));
  }

  std::vector<int64_t> sensor_ids_;
  std::vector<std::string> topics_, frame_ids_;
  int width_, height_, fps_;
  size_t n_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> pubs_;
  UniqueObj<CameraProvider> provider_;
  std::vector<UniqueObj<CaptureSession>> sessions_;
  std::vector<UniqueObj<OutputStream>> streams_;
  std::vector<UniqueObj<Request>> requests_;
  std::vector<UniqueObj<EGLStream::FrameConsumer>> consumers_;
  std::vector<int> dmabufs_;
  EGLDisplay egl_display_ = EGL_NO_DISPLAY;
  std::atomic<bool> running_{false};
  std::thread worker_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArgusCaptureNode>());
  rclcpp::shutdown();
  return 0;
}
