# BEV cuVSLAM — Docker setup, build & run (TX2 / JetPack 4.6)

End-to-end guide to building and running the 4-camera omnidirectional VO stack on
the Jetson TX2 (Auvidea J106). Covers the single Docker image, building the ported
cuVSLAM library, building the ROS 2 workspace, and running the capture + VO nodes.

Target board: TX2, JetPack 4.6.x (L4T R32.7.6), **CUDA 10.2 / r440**, nvidia
container runtime. Repo lives on the SD card at
`/media/nvidia/workspace/BEV_Jetson` (referred to below as `$BEV`).

```bash
export BEV=/media/nvidia/workspace/BEV_Jetson
```

---

## 0. One-time board prep

```bash
cd $BEV
sudo ./scripts/setup_tx2_docker.sh    # docker + nvidia runtime on the SD card
# log out/in afterwards so your user joins the docker group
```

This registers the `nvidia` Docker runtime, which mounts the host driver libs
(`tegra/`, `tegra-egl/`, `libcuda`, `libEGL_nvidia`, GLVND config) into containers
when run with `--runtime nvidia -e NVIDIA_DRIVER_CAPABILITIES=all`.

---

## 1. The one Docker image

A **single** image builds and runs everything. The r440 driver only initializes up
to **glibc 2.31 (Ubuntu 20.04)** (22.04/24.04 fail driver init — glibc-gated), and
`nvcc 10.2` needs host **gcc ≤ 8**. Ubuntu 20.04 carries ROS 2 Foxy *and* an
installable `gcc-8`, so there's no need for a separate 18.04 build container:

| Image | Base | Purpose |
|-------|------|---------|
| `cuvslam-foxy:tx2` | Ubuntu 20.04 + ROS 2 Foxy + gcc-8 + cmake 3.27 | Build `libcuvslam.so` **and** the ROS 2 nodes, and run them |

Build it (from the repo root, on the TX2):

```bash
cd $BEV
docker build -t cuvslam-foxy:tx2 -f docker/Dockerfile.cuvslam-foxy .
```

The image bakes everything the build + GPU + camera need:
- `gcc-8`/`g++-8` (nvcc 10.2 host compiler) + cmake 3.27 (focal's 3.16 is too old);
- `libcuda.so.1` SONAME symlink + `tegra` on the ld path (so CUDA's driver resolves);
- `tegra-egl` on the ld path + `__EGL_VENDOR_LIBRARY_FILENAMES=.../10_nvidia.json`
  (so GLVND loads the **NVIDIA** EGL — with EGLStream — not the Mesa fallback);
- GLVND dev/runtime libs (`libegl1`, `libgles2`, `libegl-dev`, …) + `libx11-dev`
  for the `bev_camera` Argus/EGL build.

The entrypoint runs `ldconfig` once at start (the tegra libs only exist after the
runtime mounts them).

---

## 2. Build the cuVSLAM library (CUDA-10.2 port)

Run the image with the host CUDA mounted and apply the port. nvcc compiles the
`.cu` files with `g++-8` (pinned by the script) regardless of the 20.04 default gcc-9:

```bash
cd $BEV
docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v $BEV:/workspace -w /workspace \
  cuvslam-foxy:tx2 bash scripts/build_cuvslam_tx2gpu.sh
# -> third_party/cuVSLAM/build_tx2gpu/bin/libcuvslam.so
```

The script (`scripts/build_cuvslam_tx2gpu.sh`) applies the CUDA-10.2 fixes
(C++17→14, `sm_62`, cuSOLVER-11 guards, `cudaMallocAsync`→`cudaMalloc`, …) — see
[docs/cuvslam_tx2.md](cuvslam_tx2.md) for the rationale. It is idempotent; re-run
after a submodule update. (`./scripts/port/build_and_validate.sh` does the image
build + this + a WarmUpGPU smoke test in one shot.)

---

## 3. Build the ROS 2 workspace (Foxy)

Same image; `bev_cuvslam` links the `.so` from step 2, `bev_camera` needs the
Jetson Multimedia API headers (Argus/EGLStream) bind-mounted:

```bash
cd $BEV
docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v $BEV:/workspace -w /workspace \
  cuvslam-foxy:tx2 bash -lc '
    source /opt/ros/foxy/setup.bash &&
    colcon build --packages-select bev_camera bev_cuvslam \
                 --cmake-args -DCMAKE_BUILD_TYPE=Release'
# -> install/{bev_camera,bev_cuvslam}
```

> If you change `docker/Dockerfile.cuvslam-foxy`, rebuild the image (step 1)
> before rebuilding the workspace.

---

## 4. Run

A helper to avoid repeating the long `docker run` line. The camera node needs the
**Argus socket** and **`/dev`**; both nodes need `--network host` so ROS 2 DDS
discovery works across containers (or run both in one container):

```bash
runfoxy() {        # usage: runfoxy <ros2-command...>
  docker run --rm -it --runtime nvidia --network host \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
    -v /tmp/argus_socket:/tmp/argus_socket -v /dev:/dev \
    -v $BEV:/workspace -w /workspace \
    cuvslam-foxy:tx2 "$@"
}
```

### 4a. Camera capture node

```bash
sudo systemctl restart nvargus-daemon        # on the HOST, first

runfoxy ros2 run bev_camera argus_capture_node \
  --ros-args -p sensor_ids:='[0,1,2,3]' -p width:=1640 -p height:=1232 -p fps:=20
```

Expected:
```
[argus_capture]: EGL headless display via device 0 of 1
[argus_capture]: Argus 0.98.3 (multi-process), 5 cameras present
[argus_capture]: Argus capture up: 4 cameras @ 1640x1232
```
Publishes mono8 `/cam1/image_raw` … `/cam4/image_raw` (the luma plane — exactly
what cuVSLAM wants). The node gets a headless `EGLDisplay` via
`EGL_PLATFORM_DEVICE_EXT` — **no X server needed**.

Verify from another shell:
```bash
runfoxy ros2 topic hz /cam1/image_raw
```

### 4b. cuVSLAM multicam VO node

Intrinsics (`camN.yaml`, KANNALA_BRANDT, 1640×1232) live under `scripts/config/calib/`;
extrinsics under `config/rig/rig_extrinsics.yaml`. That folder is also the node's
default `calib_dir`, so the override below is only needed if you move the files:

```bash
runfoxy ros2 run bev_cuvslam cuvslam_multicam_node --ros-args \
  -p calib_dir:=scripts/config/calib \
  -p rig_extrinsics:=config/rig/rig_extrinsics.yaml \
  -p cameras:='[cam1,cam2,cam3,cam4]'
```

Builds a 4-camera cuVSLAM `Rig` (overlapping fisheye views auto-form stereo pairs),
runs `Odometry::Track()` on each synchronized 4-image set, and publishes
`nav_msgs/Odometry` on `/odom` + the `odom→base_link` TF.

Verify:
```bash
runfoxy ros2 topic echo /odom --no-arr
```

> **Two containers / one container** — DDS discovery needs `--network host` (set
> in `runfoxy`). To avoid cross-container DDS entirely, run both nodes in a single
> container: start `argus_capture_node` in the background, then run
> `cuvslam_multicam_node` in the foreground.

---

## 5. Data path & a performance note

With the modular two-node design the image path is:
```
Argus ISP → NVMM (GPU) → CPU (NvBufferMemMap) → DDS (CPU→CPU) → cuVSLAM uploads (CPU→GPU)
```
i.e. **3 copies + a CPU round-trip**, even though the frame starts on the GPU and
cuVSLAM wants it back there. Bandwidth is small (~0.5 GB/s of the TX2's ~60 GB/s),
but it costs CPU cycles + latency. ROS 2 Foxy has no GPU-buffer transport
(NITROS/type-adaptation is Isaac-ROS/Humble-only), so this is inherent to the
split.

For the lowest-latency build, **fuse capture + cuVSLAM into one process** and feed
Argus NVMM straight to cuVSLAM as GPU memory (`cuvslam::Image.is_gpu_mem = true`,
via `NvBufSurface`→CUDA / `cuGraphicsEGLRegisterImage`), publishing only the
odometry over ROS 2. The modular split here is kept for bring-up (easy to record
bags / inspect topics).

This is a planned migration, and **the single `cuvslam-foxy:tx2` image already has
everything it needs** — Argus, NVIDIA EGL, the CUDA toolchain (nvcc/g++-8) and
`libcuvslam.so` are all in one place — so the fused node builds in the same
container with no new image. The Argus NVMM→CUDA bridge reuses the same NVIDIA-EGL
path the capture node already relies on.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `libEGL.so.1: cannot open shared object file` | EGL libs missing — rebuild the Foxy image (step 1); they're baked in. |
| `libEGL warning: DRI2 … eglCreateStreamKHR not found` | GLVND loaded **Mesa** EGL. Ensure `tegra-egl` is on the ld path + `__EGL_VENDOR_LIBRARY_FILENAMES=.../10_nvidia.json` (baked in the image). |
| `Failed to initialize EGLDisplay (getDefaultDisplay)` | Headless display not used. The node already uses `EGL_PLATFORM_DEVICE_EXT`; check the NVIDIA EGL vendor is selected (above). |
| `cusolver … NOT_INITIALIZED` / driver “insufficient” | `libcuda.so.1` SONAME not resolved — entrypoint `ldconfig` + the baked symlink fix it on 18.04/20.04. **24.04 cannot** init r440 (glibc-gated). |
| Argus sees 5 cameras but a session fails | Argus is reliable with 4 concurrent sessions, races at 5+. Use `sensor_ids:='[0,1,2,3]'`. Restart `nvargus-daemon` on the host between runs. |
| `cannot find -lEGL / -lGLESv2` at build | Need GLVND dev symlinks — `libegl-dev libgles-dev` (baked in the Foxy image). |
