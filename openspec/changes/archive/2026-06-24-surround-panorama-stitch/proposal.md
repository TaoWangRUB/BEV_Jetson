## Why

The 4 fisheye cameras give ~360° coverage but we can only inspect them as separate raw feeds.
A single **stitched surround panorama** makes the rig observable at a glance (framing, exposure,
overlap, dropouts) in rviz, and is a stepping stone toward the project's BEV/occupancy goals.
The rig is **calibrated** (KB intrinsics per output resolution + rig extrinsics), so this is a
**precomputed remap**, not feature-based stitching. No turnkey CUDA stitch lib is available on
the board (VPI not installed; OpenCV 4.2 has no CUDA), but the fused node already lands the 4
frames as **CUDA device pointers**, so a small custom CUDA kernel does it on-GPU.

## What Changes

- Add a single-process node `bev_panorama_node` that captures the 4 cameras to GPU (reusing the
  Argus + NVMM→CUDA bridge from the fused VO node) and stitches them into one **equirectangular
  panorama** on the GPU.
- Precompute, at startup, a per-camera remap table (output equirect pixel → fisheye uv + feather
  weight) from the KB intrinsics (`scripts/config/<WxH>`) + `rig_extrinsics.yaml`.
- A custom CUDA kernel samples (bilinear) + weight-blends the in-FOV cameras per output pixel.
- Publish the panorama as `sensor_msgs/Image` (mono8) on `/bev/panorama` for rviz; optional mp4
  via `cv::VideoWriter` (param).

## Capabilities

### New Capabilities
- `surround-panorama`: GPU equirectangular stitch of the 4 fisheye into one image, published for
  rviz and optionally recorded to video.

### Modified Capabilities
<!-- none -->

## Impact

- New executable in `ros2/bev_cuvslam` (or a sibling pkg) reusing the fused node's Argus/EGL/CUDA
  bridge; a CUDA `.cu` kernel (nvcc, sm_62, host g++-8) linked into the gcc-9 node.
- Deps already present: Argus/EGL/`nvbuf_utils`, CUDA driver+runtime, OpenCV 4.2 (image msg/video).
- New compose service `panorama` + launch/params; docs note.
- Caveat: extrinsics are physical-layout (not bundle-adjusted) + parallax → visible seams (OK for viz).
