// EGL->CUDA bridge spike (fused-zerocopy-argus-cuvslam, task 1).
//
// One IMX219 via Argus -> NVMM NvBuffer (dmabuf) -> NvEGLImageFromFd ->
// cuGraphicsEGLRegisterImage -> CUeglFrame Y-plane DEVICE pointer. Validates the
// zero-copy bridge by cudaMemcpy2D'ing the mapped Y plane back to host and comparing
// it against the legacy NvBufferMemMap (CPU) path for the same frame.
//
// Build/run inside cuvslam-foxy:tx2 with the Argus socket + /dev + jetson_multimedia_api
// mounted. Standalone (no ROS). Prints OK / MISMATCH and the device ptr + pitch.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

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

using namespace Argus;

static const int W = 1640, H = 1232, SENSOR = 0;

#define CK(expr, what) do { if (!(expr)) { fprintf(stderr, "[FAIL] %s\n", what); return 2; } } while (0)
#define CU(expr) do { CUresult r = (expr); if (r != CUDA_SUCCESS) { const char* s=nullptr; cuGetErrorString(r,&s); fprintf(stderr,"[CU FAIL] %s -> %s\n", #expr, s?s:"?"); return 3; } } while (0)

static EGLDisplay init_egl() {
  auto qd = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
  auto gpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
  if (qd && gpd) {
    EGLDeviceEXT devs[8]; EGLint n = 0;
    if (qd(8, devs, &n) && n > 0) {
      for (EGLint d = 0; d < n; ++d) {
        EGLDisplay dpy = gpd(EGL_PLATFORM_DEVICE_EXT, devs[d], nullptr);
        if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr)) {
          printf("[egl] headless display via device %d/%d\n", d, n);
          return dpy;
        }
      }
    }
  }
  return EGL_NO_DISPLAY;
}

int main() {
  // --- CUDA primary context current BEFORE any driver-API interop (design decision) ---
  CK(cudaFree(0) == cudaSuccess, "cudaFree(0) init primary ctx");
  CU(cuInit(0));
  CUcontext ctx; CU(cuCtxGetCurrent(&ctx));
  printf("[cuda] primary context current=%p\n", (void*)ctx);

  EGLDisplay egl = init_egl();
  CK(egl != EGL_NO_DISPLAY, "init_egl");

  // --- Argus: one camera, YUV420 EGL stream, FrameConsumer ---
  UniqueObj<CameraProvider> provider(CameraProvider::create());
  auto* ip = interface_cast<ICameraProvider>(provider.get());
  CK(ip, "ICameraProvider");
  std::vector<CameraDevice*> devs; ip->getCameraDevices(&devs);
  printf("[argus] %s, %zu cameras\n", ip->getVersion().c_str(), devs.size());
  CK(SENSOR < (int)devs.size(), "sensor present");

  UniqueObj<CaptureSession> session(ip->createCaptureSession(devs[SENSOR]));
  auto* isession = interface_cast<ICaptureSession>(session.get());
  CK(isession, "session");

  UniqueObj<OutputStreamSettings> ss(isession->createOutputStreamSettings(STREAM_TYPE_EGL));
  auto* iss = interface_cast<IEGLOutputStreamSettings>(ss.get());
  iss->setEGLDisplay(egl);
  iss->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
  iss->setResolution(Size2D<uint32_t>(W, H));
  iss->setMetadataEnable(true);
  UniqueObj<OutputStream> stream(isession->createOutputStream(ss.get()));
  UniqueObj<EGLStream::FrameConsumer> consumer(EGLStream::FrameConsumer::create(stream.get()));
  auto* ifc = interface_cast<EGLStream::IFrameConsumer>(consumer.get());

  UniqueObj<Request> request(isession->createRequest());
  auto* ireq = interface_cast<IRequest>(request.get());
  ireq->enableOutputStream(stream.get());
  auto* iprops = interface_cast<ICameraProperties>(devs[SENSOR]);
  std::vector<SensorMode*> modes; iprops->getAllSensorModes(&modes);
  for (auto* m : modes) {
    auto* im = interface_cast<ISensorMode>(m);
    if ((int)im->getResolution().width() == W && (int)im->getResolution().height() == H) {
      interface_cast<ISourceSettings>(request.get())->setSensorMode(m); break;
    }
  }
  isession->repeat(request.get());

  // --- acquire one frame -> persistent NvBuffer (dmabuf) ---
  UniqueObj<EGLStream::Frame> frame(ifc->acquireFrame(2000000000));
  auto* iframe = interface_cast<EGLStream::IFrame>(frame.get());
  CK(iframe, "acquireFrame");
  auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(iframe->getImage());
  CK(inb, "IImageNativeBuffer");
  int fd = inb->createNvBuffer(Size2D<uint32_t>(W, H), NvBufferColorFormat_YUV420, NvBufferLayout_Pitch);
  CK(fd >= 0, "createNvBuffer");

  NvBufferParams params;
  CK(NvBufferGetParams(fd, &params) == 0, "NvBufferGetParams");
  const uint32_t cpu_pitch = params.pitch[0];
  printf("[nvbuf] fd=%d  Y: pitch=%u  w=%u h=%u\n", fd, cpu_pitch, params.width[0], params.height[0]);

  // --- bridge: dmabuf -> EGLImage -> CUDA mapped frame ---
  EGLImageKHR eglimg = NvEGLImageFromFd(egl, fd);
  CK(eglimg != EGL_NO_IMAGE_KHR, "NvEGLImageFromFd");

  CUgraphicsResource res = nullptr;
  CU(cuGraphicsEGLRegisterImage(&res, eglimg, CU_GRAPHICS_REGISTER_FLAGS_NONE));
  CUeglFrame ef;
  CU(cuGraphicsResourceGetMappedEglFrame(&ef, res, 0, 0));
  printf("[cuda-egl] planeCount=%d frameType=%d colorFmt=%d  Y: ptr=%p pitch=%u w=%u h=%u\n",
         ef.planeCount, (int)ef.frameType, (int)ef.eglColorFormat,
         ef.frame.pPitch[0], ef.pitch, ef.width, ef.height);

  // --- correctness: cudaMemcpy2D Y plane (device) -> host ---
  std::vector<uint8_t> from_cuda((size_t)W * H);
  CK(cudaMemcpy2D(from_cuda.data(), W, ef.frame.pPitch[0], ef.pitch, W, H,
                  cudaMemcpyDeviceToHost) == cudaSuccess, "cudaMemcpy2D");

  // unmap/unregister before touching the same fd via CPU map
  CU(cuGraphicsUnregisterResource(res));
  NvDestroyEGLImage(egl, eglimg);

  // --- reference: NvBufferMemMap (CPU) Y plane ---
  void* mapped = nullptr;
  CK(NvBufferMemMap(fd, 0, NvBufferMem_Read, &mapped) == 0 && mapped, "NvBufferMemMap");
  NvBufferMemSyncForCpu(fd, 0, &mapped);
  std::vector<uint8_t> from_cpu((size_t)W * H);
  for (int r = 0; r < H; ++r)
    memcpy(from_cpu.data() + (size_t)r * W, (uint8_t*)mapped + (size_t)r * cpu_pitch, W);
  NvBufferMemUnMap(fd, 0, &mapped);

  // --- compare ---
  size_t diff = 0; long sum = 0;
  for (size_t i = 0; i < from_cpu.size(); ++i) { int d = (int)from_cuda[i] - from_cpu[i]; if (d) ++diff; sum += from_cpu[i]; }
  double meanlum = (double)sum / from_cpu.size();
  printf("[compare] %zu / %zu px differ; mean luma=%.1f (non-black => real image)\n", diff, from_cpu.size(), meanlum);

  NvBufferDestroy(fd);
  isession->stopRepeat();
  isession->waitForIdle();
  eglTerminate(egl);

  if (diff == 0) { printf("[RESULT] OK: zero-copy device Y plane matches CPU path exactly.\n"); return 0; }
  printf("[RESULT] MISMATCH: %zu px differ (pitch/format bug).\n", diff);
  return 1;
}
