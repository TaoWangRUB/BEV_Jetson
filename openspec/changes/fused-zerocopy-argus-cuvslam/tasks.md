## 1. EGL→CUDA bridge spike (one camera) ✅ VALIDATED (`scripts/port/egl_cuda_spike.cpp`)

- [x] 1.1 `NvEGLImageFromFd` → `cuGraphicsEGLRegisterImage` → `cuGraphicsResourceGetMappedEglFrame` gives the Y-plane **device ptr** + **pitch=1792** (planeCount=3, frameType=PITCH) — works on L4T R32.7/CUDA 10.2
- [x] 1.2 `cudaFree(0)` makes the primary context current; driver-API interop runs with no `INVALID_CONTEXT` (shared with cuVSLAM's runtime context)
- [x] 1.3 `cudaMemcpy2D` readback of the device Y plane **matches the CPU `NvBufferMemMap` path exactly** (0 / 2,020,480 px differ) — pitch=1792 (device) must be passed to cuVSLAM, not width=1640

## 2. Fused node skeleton — built, runs to setup ✅ (`bev_cuvslam_fused_node.cpp`)

- [x] 2.1 New executable in `ros2/bev_cuvslam` (CMake links Argus/EGL/`nvbuf_utils`/CUDA driver+runtime/`libcuvslam.so`); reuses Argus setup + tracker. Build gotcha fixed: **include `cuvslam2.h` before the EGL/X11 headers** (X11 `#define Success` clobbers `Result::Success`)
- [x] 2.2 Per camera: persistent `NvBuffer`, `copyToNvBuffer` each frame, register EGL/CUDA **once**, cache device ptr+pitch. Gotcha fixed: **CUDA context is per-thread** — bind the primary context (`cudaSetDevice(0)`/`cudaFree(0)`) on the *worker* thread or `cuGraphicsEGLRegisterImage` fails
- [x] 2.3 Round-robin acquire builds the 4-cam GPU `ImageSet` (`is_gpu_mem=true`, device ptr, device pitch, unified cam0 timestamp)

## 3. Tracking + output ✅

- [x] 3.1 `Track()` on the GPU `ImageSet`; publish `/cuvslam/odometry` (pose+covariance) + `odom→base_link` TF (shared publish path)
- [x] 3.2 No image publishers — frames never leave the process
- [x] 3.3 Verified (after `nvargus-daemon` restart): 4 GPU buffers register (`dev_ptr`, pitch=1792), `/cuvslam/odometry` publishes **~8.6 Hz**, no "tracking lost", pose at origin stationary. Two bugs fixed en route: X11 `Success` macro, per-thread CUDA context

## 4. Validation

- [x] 4.1 Tracking parity: fused ~8.6 Hz vs modular ~7.5 Hz, no "tracking lost", pose holds at origin stationary (same calib/rig). Motion check still pending a physical move (shared with bring-up 3.4)
- [x] 4.2 **CPU comparison (the real win): fused 23.5% vs modular 74.3%** (single container, capture+VO, same cameras) — ~3× less CPU. Timing breakdown shows why the *rate* is unchanged: it's **`Track()`-bound (~90 ms)**, not data-path-bound (acquire+GPU-copy ≈24 ms). Zero-copy cuts CPU/latency, not GPU compute. Data path verified GPU-only (no `NvBufferMemMap`/`cv_bridge`/`memcpy`/`cudaMemcpy` of pixels; `is_gpu_mem=true`)
- [~] 4.3 Clean shutdown: dtor releases Argus (`stopRepeat`; dropped `waitForIdle` — it hangs the dtor) + EGL/CUDA; SIGTERM flag handler. `docker stop` still hits the 10 s grace then SIGKILL (dtor/rclcpp teardown not prompt) — practical guidance: stop the interactive launcher with **Ctrl-C** (SIGINT, rclcpp-handled) or restart `nvargus-daemon` between detached runs (the documented TX2 norm). Multi-minute GPU-mem soak still TODO
- [x] 4.4 `scripts/run_vo_fused_tx2.sh` launcher added (docs note TODO)

## 5. Wrap-up

- [ ] 5.1 Record results (rate, CPU delta, latency if measured) in the change; update README/roadmap
