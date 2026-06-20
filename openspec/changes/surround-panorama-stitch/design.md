## Context

The fused VO node ([bev_cuvslam_fused_node.cpp](../../../ros2/bev_cuvslam/src/bev_cuvslam_fused_node.cpp))
already captures the 4 IMX219 via Argus and bridges each NVMM Y-plane to a **CUDA device pointer**
(`NvEGLImageFromFd`→`cuGraphicsEGLRegisterImage`). The rig is calibrated: KB (equidistant) fisheye
intrinsics per output resolution in `scripts/config/<WxH>/camN.yaml`, extrinsics
(`rig_from_camera`, rig frame X=right/Y=forward/Z=up) in `config/rig/rig_extrinsics.yaml`. No CUDA
stitch lib is available (VPI not installed; OpenCV 4.2 no CUDA), so we write the kernel.

## Goals / Non-Goals

**Goals:** GPU equirectangular stitch of the 4 fisheye; precomputed remap from calib; weight-blended
overlaps; publish `/bev/panorama` for rviz + optional mp4; runs in the existing image.

**Non-Goals:** feature-based/seamless stitching or exposure compensation; bundle-adjusting the
extrinsics; ground-plane BEV/IPM (separate change); colorization (cameras are mono Y).

## Decisions

- **Reuse the fused node's capture+bridge** (single process). The stitch path replaces Track():
  device pointers → CUDA kernel → panorama. No host pixel copy until the final publish download.
- **Equirect mapping (precomputed CPU, once):** output W×H over azimuth φ∈[−π,π], elevation
  θ∈[−θmax,θmax]. Per output pixel: ray in rig frame `d_rig=[sinφ cosθ, cosφ cosθ, sinθ]` (φ=0→+Y
  forward). For each camera: `d_cam = R_rig_from_cam^T · d_rig`; if `d_cam.z>0` and incidence angle
  `α=atan2(‖d_cam.xy‖,d_cam.z)` within FOV, project with the **OpenCV fisheye** model
  (`θd=α(1+k2α²+k3α⁴+k4α⁶+k5α⁸)`, `u=mu·θd·x̂+u0`, `v=mv·θd·ŷ+v0`, using yaml k2..k5). Store per
  camera: `float2 uv` + `float weight` (feather = smoothstep falloff near the FOV edge; 0 if out).
- **Kernel:** per output pixel accumulate `Σ w_c·bilinear(cam_c,uv_c) / Σ w_c` over the 4 cameras
  (w=0 skips). mono8 in/out. Maps live in device memory (4×(float2+float) ≈ 60 MB at 1920×640 — fine).
- **Build:** add the `.cu` to the `bev_cuvslam` colcon target via `enable_language(CUDA)` with
  `CMAKE_CUDA_HOST_COMPILER=g++-8` + `CMAKE_CUDA_ARCHITECTURES=62` (nvcc 10.2 needs gcc≤8; the .cpp
  still builds gcc-9, links via libstdc++ forward-compat — same as linking libcuvslam). **Spike this
  first** (task 1); fallback = a standalone nvcc-built kernel lib with an `extern "C"` launcher
  linked into the node (the libcuvslam pattern).
- **Output:** download panorama → `sensor_msgs/Image` mono8 on `/bev/panorama`; `save_video` param
  → `cv::VideoWriter` (OpenCV 4.2, CPU, fine for a viz stream).

## Risks / Trade-offs

- **CUDA-in-colcon toolchain mix** (gcc-9 host for .cpp, g++-8 for .cu) is the main build risk →
  spiked first; the extern-C standalone-lib fallback is proven (cuVSLAM).
- **Seams/ghosting**: extrinsics are physical-layout (not BA'd) + ~1.5 cm parallax → misalignment
  at the overlaps. Acceptable for a viz/monitoring panorama; feather-blending hides hard cuts.
- **Output download + publish** is a host copy (small, one image) — fine; only the stitch is GPU-bound.
- **Elevation coverage** limited by the cameras' vertical FOV; poles will be empty (black) — expected.
