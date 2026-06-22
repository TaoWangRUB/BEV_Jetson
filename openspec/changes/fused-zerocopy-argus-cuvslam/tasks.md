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

## 7. Fused (zero-copy) vs modular (ROS2 GPU→CPU→GPU) head-to-head

- [x] 7.1/7.2/7.3 Head-to-head at **1640×1232, native fps, same scene**:

  | Pipeline | data path | sustained odom | CPU |
  |---|---|---|---|
  | **Modular** (ROS2 GPU→CPU→GPU) | Argus→NVMM→**CPU memmap**→DDS→cv_bridge→GPU re-upload | **9.2 Hz** | **78.5%** |
  | **Fused** (zero-copy) | Argus→NVMM→CUDA (EGL register) →cuVSLAM | **19.8 Hz** | **23.1%** |

  - **Zero-copy = ~3.4× less CPU AND ~2.1× higher rate** at equal resolution. The per-frame 4× 2 MP `NvBufferMemMap` memcpy + DDS image (de)serialize + cv_bridge + CPU→GPU re-upload both burns CPU *and* throttles throughput (caps modular at 9.2 Hz).
  - **Rate gap is scene-dependent**: when Track is small (~29 ms here) the CPU-copy overhead dominates the modular loop → fused wins ~2× on rate; when Track is heavy (~90 ms scene) both are Track-bound and closer (earlier: modular 7.5 vs fused 8.6 Hz). CPU win (~3×) holds regardless.

## 6. Resolution / fps sweep (find the rate sweet spot; Track is the bottleneck)

- [ ] 6.1 Decouple **sensor mode** from **output resolution** in the fused node (params `sensor_width/sensor_height` vs `width/height`); Argus ISP downscales in NVMM (stays zero-copy)
- [ ] 6.2 Calib per output res: `1640x1232/` and `1280x720/` (done, on board); generate ½-scaled `820x616` (= 1640÷2) and `640x360` (= 1280÷2) — KB intrinsics scale linearly (mu,mv,u0,v0,w,h ×0.5; k2..k5 unchanged)
- [x] 6.3 Measured — **windowed-average + sustained-odom, native fps** (`fps:=60`, sensor caps). Earlier tables were invalid: single-sample throttle prints AND a hidden 20 fps cap (`fps_` default 20 throttled every config). Corrected:

  | Cfg | sensor (max fps) | →output | FOV | Track | sustained odom | CPU |
  |---|---|---|---|---|---|---|
  | A | 1640×1232 (22) | 1640×1232 | full | 29 ms | 17.9 Hz | 22% |
  | **B** | 1640×1232 (22) | **832×624** | **full** | **12 ms** | **22.3 Hz** | **15%** |
  | C | 1280×720 (44) | 1280×720 | crop | 23 ms | 26.7 Hz | 29% |
  | D | 1280×720 (44) | 640×360 | crop | 20 ms | **34.8 Hz** | 31% |

- [x] 6.4 Findings + recommendation (physically consistent now — no rate exceeds its sensor fps):
  - **Within a mode, downscaling raises rate up to the input-fps cap + cuts Track & CPU, same FOV.** A→B: 17.9→**22.3 Hz** (B hits the 22 fps ceiling; A's 29 ms Track is too slow to), CPU 22→15%. Best **full-FOV** config = **B (1640→832×624): 22.3 Hz, 15% CPU**.
  - **>22 Hz requires the 720p mode (44 fps), which CROPS the FOV** (loses surround overlap) and costs more CPU (more frames/s → more Track/s): C 26.7 Hz/29%, D 34.8 Hz/31%.
  - **Per-call Track rises with the processing rate, not just pixels.** D(640, 230k px)=20 ms > B(832, 519k px)=12 ms even though 640 is smaller — because D runs at **34.8 Hz vs B's 22.3 Hz**: more Track calls/s keeps the GPU busier and overlaps cuVSLAM's **async SBA** across frames → contention → longer measured per-call Track. (At the *same* rate/FOV, Track does scale with pixels: A 1640→29 ms vs B 832→12 ms.)
  - **acquire time = mostly *waiting* for frames** when input-capped (B 32 ms idle wait at 22 fps; D 9 ms at 44 fps) — not real work.
  - **Recommendation: default to 1640×1232 → 832×624 (full FOV, 22 Hz @ sensor ceiling, lowest CPU).** Use 1280×720→640×360 only if ~35 Hz is needed AND the cropped FOV (reduced surround overlap) is acceptable.
