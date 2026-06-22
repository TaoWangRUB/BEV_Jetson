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

All container parameters (nvidia runtime, host-CUDA + Jetson-MMAPI mounts, Argus socket,
`/dev`, host networking, env) live in [`docker-compose.yml`](../docker-compose.yml) at the
repo root — so every command below is a short `docker compose` invocation instead of a long
`docker run`. Build the image once (from the repo root, on the TX2):

```bash
cd $BEV
docker compose build
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

Apply the port + build (nvcc compiles the `.cu` files with `g++-8`, pinned by the script):

```bash
docker compose run --rm build-cuvslam
# -> third_party/cuVSLAM/build_tx2gpu/bin/libcuvslam.so
```

The script (`scripts/build_cuvslam_tx2gpu.sh`) applies the CUDA-10.2 fixes
(C++17→14, `sm_62`, cuSOLVER-11 guards, `cudaMallocAsync`→`cudaMalloc`, …) — see
[docs/cuvslam_tx2.md](cuvslam_tx2.md) for the rationale. It is idempotent; re-run
after a submodule update. (`./scripts/port/build_and_validate.sh` does the image
build + this + a WarmUpGPU smoke test in one shot.)

---

## 3. Build the ROS 2 workspace (Foxy)

`bev_cuvslam` links the `.so` from step 2; `bev_camera` needs the Jetson Multimedia API
headers (mounted by the compose file):

```bash
docker compose run --rm build-ws
# -> install/{bev_camera,bev_cuvslam}
```

> If you change `docker/Dockerfile.cuvslam-foxy`, rerun `docker compose build` before
> rebuilding the workspace.

---

## 4. Run

> The **fused zero-copy node (§5) is the recommended runtime** (~3× less CPU, ~2× rate).
> The modular two-node pipeline here is kept for bring-up / bag inspection.

Restart the Argus daemon on the host first (clears any leaked session), then run capture + VO
together in **one** container (cross-container DDS discovery fails on this setup, so both
nodes share a container via the `modular` service):

```bash
sudo systemctl restart nvargus-daemon        # on the HOST
docker compose run --rm modular
```

The `modular` service backgrounds `argus_capture_node` — publishing mono8
`/cam1/image_raw` … `/cam4/image_raw` (the luma plane cuVSLAM wants) via a headless
`EGLDisplay` (`EGL_PLATFORM_DEVICE_EXT`, **no X server**) — then launches
`cuvslam_multicam_node` → `/cuvslam/odometry` + `odom→base_link` TF. Expected:
```
[argus_capture]: Argus capture up: 4 cameras @ 1640x1232
[cuvslam_multicam]: cuVSLAM multicam VO up: 4 cameras, mode=Multicamera
```

Inspect from another shell with `docker compose run --rm shell` (then `ros2 topic hz
/cam1/image_raw`, `ros2 topic echo /cuvslam/odometry`, …), or run only the camera node with
`docker compose run --rm capture`. Intrinsics (`camN.yaml`, KANNALA_BRANDT) live under
`scripts/config/<WxH>/`; the VO node defaults to `scripts/config/calib` + extrinsics in
`config/rig/rig_extrinsics.yaml`.

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

## 5. Fused zero-copy node (recommended runtime)

The modular two-node pipeline (§4) moves every frame
**Argus ISP → NVMM(GPU) → CPU (`NvBufferMemMap`) → DDS (CPU→CPU) → cuVSLAM upload (CPU→GPU)**
— 3 copies + a CPU round-trip, even though the frame starts on the GPU and cuVSLAM wants it
there. ROS 2 Foxy has no GPU-buffer transport (NITROS is Isaac-ROS/Humble-only), so the
split is inherently lossy.

**`bev_cuvslam_fused_node`** fuses capture + cuVSLAM into one process and feeds the Argus
NVMM **Y(luma) plane straight to cuVSLAM as GPU memory** (`cuvslam::Image.is_gpu_mem=true`)
via the EGL→CUDA bridge (`NvEGLImageFromFd` → `cuGraphicsEGLRegisterImage`), publishing only
`/cuvslam/odometry` + the `odom→base_link` TF — no host copy, no DDS image transport. Same
`cuvslam-foxy:tx2` image (Argus + NVIDIA-EGL + CUDA + `libcuvslam.so` are all already there).

### Run it (params from a yaml, via launch)

```bash
sudo systemctl restart nvargus-daemon        # on the HOST, first
docker compose run --rm fused                # RECORD=1 docker compose run --rm fused  → also bags odom+tf
```

Node params live in [`ros2/bev_cuvslam/config/fused_vo_params.yaml`](../ros2/bev_cuvslam/config/fused_vo_params.yaml)
(default = the best full-FOV config below); the `fused` service in
[`docker-compose.yml`](../docker-compose.yml) holds the container params. Override the param
file with `... bev_cuvslam_fused.launch.py params:=/abs/path.yaml`. The node decouples
**sensor mode** (`sensor_width/height`) from
**output** (`width/height`): set the sensor to a full-FOV mode and the ISP downscales the output
in NVMM (still zero-copy).

### Measured comparison (4 cams, TX2, stationary bench, native fps)

**Fused vs the modular ROS2 pipeline @ 1640×1232 (same scene/methodology):**

| Pipeline | data path | sustained odom | CPU |
|----------|-----------|---------------:|----:|
| Modular (ROS2 GPU→CPU→GPU) | NVMM→CPU memmap→DDS→cv_bridge→GPU re-upload | **9.2 Hz** | **78%** |
| **Fused** (zero-copy NVMM→CUDA) | NVMM→CUDA→cuVSLAM | **19.8 Hz** | **23%** |

Zero-copy is **~3.4× less CPU and ~2× the rate** at equal resolution — the CPU image
round-trip both burns cycles and throttles throughput. (The rate gap shrinks when the scene
makes `Track()` heavy, i.e. both become Track-bound; the CPU win holds regardless.)

**Resolution / fps sweep (fused; output res = what cuVSLAM sees):**

| sensor (max fps) | →output | FOV | Track | odom | CPU |
|------------------|---------|-----|------:|-----:|----:|
| 1640×1232 (22) | 1640×1232 | full | 29 ms | 17.9 Hz | 22% |
| **1640×1232 (22)** | **832×624** | **full** | 12 ms | **22.3 Hz** | **15%** ← default |
| 1280×720 (44) | 1280×720 | crop | 23 ms | 26.7 Hz | 29% |
| 1280×720 (44) | 640×360 | crop | 20 ms | 34.8 Hz | 31% |

Takeaways: **downscaling a full-FOV mode raises the rate up to the sensor's fps ceiling and
cuts CPU at full FOV** → the **1640→832×624** default (~22 Hz, full surround FOV, lowest CPU).
Beating ~22 Hz needs the 720p mode (44 fps) which **crops the fisheye FOV** (hurts the
surround overlap) and costs more CPU. Per-call `Track` rises with the *processing rate*
(more frames/s → cuVSLAM async-SBA overlaps across frames → GPU contention), not just pixels.
See the [fused-zerocopy OpenSpec change](../openspec/changes/fused-zerocopy-argus-cuvslam/tasks.md)
for the full data. Calibration per output resolution lives in `scripts/config/<WxH>/`
(scale with [`scripts/calib/scale_calib.py`](../scripts/calib/scale_calib.py)).

---

## 6. Surround panorama (rviz)

`bev_panorama_node` stitches the 4 fisheye into one **equirectangular panorama on the GPU** —
same Argus→NVMM→CUDA bridge as the fused node, plus a custom CUDA kernel that bilinear-samples +
feather-blends each camera using a remap table precomputed from the KB intrinsics + rig
extrinsics. No CUDA stitch lib needed (VPI absent, OpenCV-4.2 has no CUDA). Publishes
`/bev/panorama` (mono8) — measured **1920×540 @ ~29 Hz**.

```bash
docker compose run --rm panorama        # params: ros2/bev_cuvslam/config/panorama_params.yaml
# then in rviz add an Image display on /bev/panorama
# record: ... bev_panorama.launch.py params:=... with save_video set, or edit the yaml
```

Caveats: the extrinsics are physical-layout (not bundle-adjusted) and the cameras have parallax,
so expect a slightly wavy horizon + faint seams (no exposure compensation). Poles are black
(outside the ±`elevation_max_deg` / camera FOV). Tune `pano_width/height`, `elevation_max_deg`,
`feather_deg` in the params yaml.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `libEGL.so.1: cannot open shared object file` | EGL libs missing — rebuild the Foxy image (step 1); they're baked in. |
| `libEGL warning: DRI2 … eglCreateStreamKHR not found` | GLVND loaded **Mesa** EGL. Ensure `tegra-egl` is on the ld path + `__EGL_VENDOR_LIBRARY_FILENAMES=.../10_nvidia.json` (baked in the image). |
| `Failed to initialize EGLDisplay (getDefaultDisplay)` | Headless display not used. The node already uses `EGL_PLATFORM_DEVICE_EXT`; check the NVIDIA EGL vendor is selected (above). |
| `cusolver … NOT_INITIALIZED` / driver “insufficient” | `libcuda.so.1` SONAME not resolved — entrypoint `ldconfig` + the baked symlink fix it on 18.04/20.04. **24.04 cannot** init r440 (glibc-gated). |
| Argus sees 5 cameras but a session fails | Argus is reliable with 4 concurrent sessions, races at 5+. Use `sensor_ids:='[0,1,2,3]'`. Restart `nvargus-daemon` on the host between runs. |
| `cannot find -lEGL / -lGLESv2` at build | Need GLVND dev symlinks — `libegl-dev libgles-dev` (baked in the Foxy image). |
