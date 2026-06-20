## 1. EGL→CUDA bridge spike (one camera)

- [ ] 1.1 In a throwaway test, take one Argus camera's persistent `NvBuffer` (dmabuf fd) and `NvEGLImageFromFd(egl_display, fd)` → `cuGraphicsEGLRegisterImage` → `cuGraphicsResourceGetMappedEglFrame`; print the `CUeglFrame` Y-plane device ptr + pitch
- [ ] 1.2 Make the runtime primary CUDA context current (`cudaFree(0)`) before the driver-API calls; confirm no `CUDA_ERROR_INVALID_CONTEXT`
- [ ] 1.3 Validate correctness: `cudaMemcpy2D` the mapped Y plane back to host and compare against the existing `NvBufferMemMap` path for the same frame (pixels match)

## 2. Fused node skeleton

- [ ] 2.1 New executable in `ros2/bev_cuvslam` (CMake: link Argus, EGL, `nvbuf_utils`, CUDA driver API, `libcuvslam.so`); reuse the Argus setup from `argus_capture_node.cpp` and tracker/bundler from `cuvslam_multicam_node.cpp`
- [ ] 2.2 Per camera: create one persistent `NvBuffer`, `copyToNvBuffer` each frame, register EGL/CUDA **once**, cache the device ptr + pitch
- [ ] 2.3 Wire the index-0 driver-camera trigger + `sync_slop_ms` staleness bundler; build the cuVSLAM `ImageSet` with `is_gpu_mem=true`, device ptr, device pitch, unified per-set timestamp

## 3. Tracking + output

- [ ] 3.1 Call `Track()` on the GPU `ImageSet`; publish `/cuvslam/odometry` (pose+covariance) + `odom→base_link` TF (reuse the modular node's publish path)
- [ ] 3.2 Do NOT create image publishers (frames never leave the process)
- [ ] 3.3 Bring up one container; confirm odometry publishes with live tracking, no "tracking lost"

## 4. Validation

- [ ] 4.1 Tracking parity vs the modular node (same calib/rig): rate ≥ ~8.5 Hz, pose holds at origin stationary, tracks under motion
- [ ] 4.2 CPU comparison: `docker stats` / per-process CPU for fused vs modular two-node at the same camera rate — confirm the fused node is lower
- [ ] 4.3 Soak test (several minutes): GPU memory stable (no leak), no CUDA faults; clean shutdown releases Argus/EGL/CUDA
- [ ] 4.4 `scripts/run_vo_fused_tx2.sh` launcher + docs note in `docs/build_and_run.md` §5

## 5. Wrap-up

- [ ] 5.1 Record results (rate, CPU delta, latency if measured) in the change; update README/roadmap
