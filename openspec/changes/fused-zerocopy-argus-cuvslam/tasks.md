## 1. EGL→CUDA bridge spike (one camera) ✅ VALIDATED (`scripts/port/egl_cuda_spike.cpp`)

- [x] 1.1 `NvEGLImageFromFd` → `cuGraphicsEGLRegisterImage` → `cuGraphicsResourceGetMappedEglFrame` gives the Y-plane **device ptr** + **pitch=1792** (planeCount=3, frameType=PITCH) — works on L4T R32.7/CUDA 10.2
- [x] 1.2 `cudaFree(0)` makes the primary context current; driver-API interop runs with no `INVALID_CONTEXT` (shared with cuVSLAM's runtime context)
- [x] 1.3 `cudaMemcpy2D` readback of the device Y plane **matches the CPU `NvBufferMemMap` path exactly** (0 / 2,020,480 px differ) — pitch=1792 (device) must be passed to cuVSLAM, not width=1640

## 2. Fused node skeleton — built, runs to setup ✅ (`bev_cuvslam_fused_node.cpp`)

- [x] 2.1 New executable in `ros2/bev_cuvslam` (CMake links Argus/EGL/`nvbuf_utils`/CUDA driver+runtime/`libcuvslam.so`); reuses Argus setup + tracker. Build gotcha fixed: **include `cuvslam2.h` before the EGL/X11 headers** (X11 `#define Success` clobbers `Result::Success`)
- [x] 2.2 Per camera: persistent `NvBuffer`, `copyToNvBuffer` each frame, register EGL/CUDA **once**, cache device ptr+pitch. Gotcha fixed: **CUDA context is per-thread** — bind the primary context (`cudaSetDevice(0)`/`cudaFree(0)`) on the *worker* thread or `cuGraphicsEGLRegisterImage` fails
- [x] 2.3 Round-robin acquire builds the 4-cam GPU `ImageSet` (`is_gpu_mem=true`, device ptr, device pitch, unified cam0 timestamp)

## 3. Tracking + output — code complete; runtime verify blocked on daemon

- [x] 3.1 `Track()` on the GPU `ImageSet`; publish `/cuvslam/odometry` (pose+covariance) + `odom→base_link` TF (shared publish path)
- [x] 3.2 No image publishers — frames never leave the process
- [ ] 3.3 Confirm odometry publishes with live tracking — **BLOCKED**: the Argus daemon is wedged (leaked sessions from SIGKILL'd test runs; modular node also fails `no session`). Needs `sudo systemctl restart nvargus-daemon` on the host (no passwordless sudo from `tx2-eth`), then rerun `bev_cuvslam_fused_node`

## 4. Validation

- [ ] 4.1 Tracking parity vs the modular node (same calib/rig): rate ≥ ~8.5 Hz, pose holds at origin stationary, tracks under motion
- [ ] 4.2 CPU comparison: `docker stats` / per-process CPU for fused vs modular two-node at the same camera rate — confirm the fused node is lower
- [ ] 4.3 Soak test (several minutes): GPU memory stable (no leak), no CUDA faults; clean shutdown releases Argus/EGL/CUDA
- [ ] 4.4 `scripts/run_vo_fused_tx2.sh` launcher + docs note in `docs/build_and_run.md` §5

## 5. Wrap-up

- [ ] 5.1 Record results (rate, CPU delta, latency if measured) in the change; update README/roadmap
