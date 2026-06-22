## 1. CUDA-in-colcon build spike

- [x] 1.1 Add a trivial `.cu` (e.g. a fill/copy kernel with an `extern "C"` launcher) to the `bev_cuvslam` package; `enable_language(CUDA)`, `CMAKE_CUDA_HOST_COMPILER=g++-8`, `CMAKE_CUDA_ARCHITECTURES=62`
- [x] 1.2 Confirm it compiles (nvcc+g++-8) and links into a gcc-9 node + runs on the GPU. Fallback if it fights colcon: standalone nvcc lib + `extern "C"` launcher (libcuvslam pattern)

## 2. Equirect remap precompute (CPU, startup)

- [x] 2.1 Load KB intrinsics (`scripts/config/<WxH>/camN.yaml`) + extrinsics (`rig_from_camera`); build R_rig_from_cam
- [x] 2.2 For each output pixel (az,el) → rig ray → per-camera transform + OpenCV-fisheye project → `float2 uv` + feather `weight`; upload the 4 maps to device

## 3. CUDA stitch kernel

- [x] 3.1 Kernel: per output pixel, `Σ w·bilinear(cam,uv)/Σ w` over the 4 cameras (mono8), guarded for out-of-FOV (w=0) and border
- [x] 3.2 Wire the cameras' device pointers (from the NVMM→CUDA bridge) + maps → kernel → device panorama

## 4. Node + output

- [x] 4.1 `bev_panorama_node`: reuse Argus capture + EGL/CUDA bridge; round-robin acquire → stitch kernel → panorama
- [x] 4.2 Download + publish `/bev/panorama` (`sensor_msgs/Image` mono8); params: width/height/elev/calib_dir/sensor mode
- [x] 4.3 Optional `save_video` → `cv::VideoWriter`; clean shutdown (release Argus/EGL/CUDA + finalize video)

## 5. Run + validate

- [x] 5.1 Build via colcon in the container; run; confirm `/bev/panorama` displays in rviz (covers ~360°, overlaps blended)
- [x] 5.2 Record rate + CPU/GPU; note seam quality
- [x] 5.3 compose `panorama` service + launch/params yaml + docs note in build_and_run.md

## Results (validated 2026-06-20)

- CUDA-in-colcon build works (nvcc + g++-8 + sm_62 inside the gcc-9 workspace; CMake pins them).
- `/bev/panorama` publishes **mono8 1920×540 @ ~29 Hz** (GPU stitch); each camera maps ~100k px,
  blended overlaps. Verified visually — a coherent 360° equirect of the lab scene.
- Caveats (expected): wavy horizon + faint seams from physical-layout (non-BA) extrinsics + no
  exposure compensation; poles black (outside the ±70° / camera FOV).
- Run: `docker compose run --rm panorama` (params in bev_cuvslam/config/panorama_params.yaml),
  then add an Image display on `/bev/panorama` in rviz. `save_video:=bags/pano.mp4` to record.
