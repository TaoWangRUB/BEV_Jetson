## Why

The modular bring-up pipeline copies every frame **Argus → NVMM(GPU) → CPU (NvBufferMemMap)
→ DDS (CPU→CPU) → cuVSLAM upload (CPU→GPU)** — 3 copies + a CPU round-trip, even though the
frame *starts* on the GPU and cuVSLAM *wants* it on the GPU. On the TX2 (8 GB shared, ~6 weak
cores) those copies cost CPU cycles and latency that matter when 4–6 cameras feed VO. cuVSLAM
already accepts GPU buffers (`cuvslam::Image.is_gpu_mem = true`), and the single
`cuvslam-foxy:tx2` image already carries Argus + NVIDIA-EGL + CUDA + `libcuvslam.so`, so the
fused node builds with **no new image**. This is staged-plan **step 2** from
[bring-up-end-to-end-vo](../bring-up-end-to-end-vo/design.md) — gated on step 1, which now
tracks (~8.5 Hz odom).

## What Changes

- Add a **single-process fused node** that does Argus capture **and** cuVSLAM `Track()`,
  reusing the bring-up node's latest-frame bundler and unified-timestamp logic.
- Bridge each Argus frame's **NVMM Y(luma) plane → CUDA device pointer** via the existing
  NVIDIA-EGL path (`NvBufSurface`/EGLImage → `cuGraphicsEGLRegisterImage` →
  `cuGraphicsResourceGetMappedEglFrame`), and pass it to cuVSLAM with `is_gpu_mem = true`
  (correct device `pitch`), **eliminating the CPU round-trip**.
- Publish **only** `/cuvslam/odometry` + `odom→base_link` TF (no image topics over DDS).
- Keep the modular two-node path as-is (bring-up / bag inspection); the fused node is the
  low-latency runtime.

## Capabilities

### New Capabilities
- `fused-vo`: a single-process node that feeds Argus NVMM frames to cuVSLAM as GPU memory
  (zero CPU round-trip), publishing odometry + TF on the TX2.

### Modified Capabilities
<!-- None: the `visual-odometry` capability from the bring-up change is not yet archived to
     openspec/specs/, so this is added as a separate capability rather than a delta. -->

## Impact

- New node in `ros2/bev_cuvslam` (fused executable) reusing the Argus setup from
  `ros2/bev_camera/src/argus_capture_node.cpp` and the tracker/bundler from
  `cuvslam_multicam_node.cpp`.
- NVIDIA EGL↔CUDA interop (`cuGraphicsEGLRegisterImage`, `NvBufSurface`) on L4T R32.7.6 /
  CUDA 10.2 — links the CUDA driver API + Jetson Multimedia API (already mounted/baked).
- Same `cuvslam-foxy:tx2` container; `libcuvslam.so` unchanged. A new `run_vo_fused_tx2.sh`
  launcher (one container, no inter-node DDS).
- Risk surface: EGL→CUDA mapping correctness, NVMM pitch/format, CUDA-context lifetime.
