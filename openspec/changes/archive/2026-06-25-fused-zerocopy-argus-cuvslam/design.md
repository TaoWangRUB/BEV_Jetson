## Context

The modular pipeline ([bring-up-end-to-end-vo](../bring-up-end-to-end-vo/design.md)) tracks
at ~8.5 Hz but pays Argus→NVMM→**CPU**→DDS→**GPU** per frame. Both halves already exist in one
container: `argus_capture_node` ([ros2/bev_camera](../../../ros2/bev_camera/src/argus_capture_node.cpp))
sets up a headless EGL display (`EGL_PLATFORM_DEVICE_EXT`), an Argus `FrameConsumer` per camera,
and turns each frame into an NVMM `NvBuffer` (dmabuf) via `IImageNativeBuffer::createNvBuffer`;
`cuvslam_multicam_node` ([ros2/bev_cuvslam](../../../ros2/bev_cuvslam/src/cuvslam_multicam_node.cpp))
runs the latest-frame bundler + unified-timestamp + cuVSLAM `Track()`. cuVSLAM accepts a device
pointer when `cuvslam::Image.is_gpu_mem = true` ([cuvslam2.h](../../../third_party/cuVSLAM/libs/cuvslam/cuvslam2.h)).
Board: L4T **R32.7.6 / CUDA 10.2** — the **legacy `nvbuf_utils` NvBuffer API** (not JetPack-5
`NvBufSurface`), with `NvEGLImageFromFd` for the EGL bridge.

## Goals / Non-Goals

**Goals:**
- One process: Argus capture + cuVSLAM, no host copy / no DDS image transport on the track path.
- Each camera's NVMM **Y(luma) plane → CUDA device ptr** fed to cuVSLAM (`is_gpu_mem=true`).
- Publish only `/cuvslam/odometry` + TF; parity with the modular node at ≥ its rate, less CPU.
- Reuse the proven bundler/staleness/unified-timestamp logic and calibration/rig loading.

**Non-Goals:**
- Changing cuVSLAM, the calibration, the bundler algorithm, or hardware sync (still no HW trigger).
- Removing the modular two-node path (kept for bag inspection / bring-up).
- 6th camera, IMU/EKF — separate changes.

## Decisions

- **Bridge = persistent per-camera NvBuffer + one-time EGL/CUDA registration.** Keep the node's
  existing pattern (create one `NvBuffer` per camera, `copyToNvBuffer` each frame — an on-GPU
  NVMM→NVMM copy, no host touch). Register that stable buffer **once** per camera:
  `NvEGLImageFromFd(egl_display, fd)` → `cuGraphicsEGLRegisterImage` →
  `cuGraphicsResourceGetMappedEglFrame` → `CUeglFrame`; feed `frame.frame.pPitch[0]` (Y plane
  device ptr) + `frame.pitch` to cuVSLAM. Stable device ptr = simplest lifetime, eliminates the
  CPU round-trip (the dominant cost). *(Future micro-opt: register the Argus frame image
  directly to drop even the GPU→GPU copy — deferred; lifetime is trickier.)*
- **CUDA context:** force the runtime **primary context** current on the capture/track thread
  (`cudaFree(0)` / `cudaSetDevice(0)` before any driver-API `cuGraphics*`), so the driver-API
  interop and cuVSLAM's runtime context are the same — no cross-context pointer use.
- **One node, one thread** drives Argus acquire → copyToNvBuffer → bundle → `Track()`; reuse the
  index-0 driver-camera trigger + `sync_slop_ms` staleness bound.
- **Packaging:** a new executable in `ros2/bev_cuvslam` (links Argus/EGL like `bev_camera` +
  `libcuvslam.so`); a `scripts/run_vo_fused_tx2.sh` launcher (single container). The modular
  nodes stay.

## Risks / Trade-offs

- **EGL↔CUDA interop on R32.7/CUDA 10.2** is the main unknown — `cuGraphicsEGLRegisterImage` +
  `NvEGLImageFromFd` are Tegra-specific and version-sensitive; validate a single camera end to
  end before wiring all 4. Fallback: the modular pipeline already works.
- **Pitch/format**: must pass the device **pitch** (not width) and the **Y plane only** (mono8);
  a wrong pitch silently corrupts tracking. Cross-check against a one-off CPU readback of the
  mapped buffer during bring-up.
- **cuVSLAM `is_gpu_mem=true`** path must actually consume device memory in the CUDA-10.2 port —
  verify it doesn't assume host pointers; a one-camera smoke test gates the rest.
- **Resource lifetime**: register-once avoids per-frame churn, but EGL images / CUDA resources
  must be unregistered + `NvDestroyEGLImage`'d on shutdown; watch GPU mem over a long run.
- **Expected win is CPU/latency, not tracking quality** — the inter-camera skew limit is
  unchanged (still no HW sync). If the win is marginal, the modular path remains the default.
