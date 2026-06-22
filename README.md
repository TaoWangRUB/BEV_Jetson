# BEV — Omnidirectional Visual Perception on Jetson TX2

OmniNxt-style omnidirectional visual perception on a **Jetson TX2 / Auvidea J106**
carrier with **4–6 IMX219 fisheye cameras**, for:

- **360° visual-inertial odometry / SLAM** (primary),
- **obstacle depth / occupancy** (planned), and
- **surround bird's-eye-view** (planned).

The centerpiece is **NVIDIA cuVSLAM ported to CUDA 10.2** — the TX2's permanent
toolkit ceiling — so the board runs an efficient, GPU-accelerated, multi-camera
feature-based VIO that upstream only supports on Orin/CUDA 12+. See
[docs/cuvslam_tx2.md](docs/cuvslam_tx2.md) for the port details.

> **Status (2026-06-17):** cuVSLAM is **built and runtime-validated** on the TX2
> (`libcuvslam.so`, sm_62, CUDA 10.2; `WarmUpGPU()` runs on the r440 driver).
> Next: the ROS 2 multi-camera wrapper. See [Roadmap](#roadmap).

---

## 1. Hardware setup & configuration

| Item | Detail |
|------|--------|
| Compute | NVIDIA **Jetson TX2** (Tegra186, Pascal **sm_62**) |
| Carrier | Auvidea **J106 + M110** |
| Firmware | L4T **R32.7.6** / JetPack **4.6.x**, kernel `4.9.337-tegra` (patched 6-CSI IMX219 kernel) |
| CUDA / driver | **CUDA 10.2.300** / driver **r440** (the TX2's hard ceiling — no JetPack 5) |
| Host OS | Ubuntu **18.04**, aarch64 |
| Cameras | 6× CSI **IMX219**, **160° fisheye** lenses. Currently **5/6** probe → `/dev/video0–4` |
| IMU | none fitted yet (cuVSLAM inertial mode needs one; vision-only multicam works first) |
| Storage | 28 GB eMMC (root) + **SD card 117 GB** mounted at `/media/nvidia/workspace` |
| Power mode | `nvpmodel` MAXN |
| Access | `ssh tx2-eth` (`10.42.0.157`, M110 eth) or `tx2-wlan` (`192.168.0.168`) |

### Storage policy
The eMMC is small, so all heavy state lives on the SD card: the BEV repo at
`/media/nvidia/workspace/BEV`, and **Docker's `data-root` is relocated to
`/media/nvidia/workspace/docker`**. The SD is pinned in `/etc/fstab` (`nofail`)
with a systemd drop-in so Docker starts only after it mounts.

### Camera bring-up
The 6-CSI IMX219 device-tree + driver work lives in a separate repo
(`auvidea-j106-tx2`): a carrier DTB + a shared-reset driver patch on L4T R32.7.6.
One sensor (i2c bus 1, addr `0x12`) is not yet probing — hence 5/6.

**Port ↔ camera ↔ sensor mapping** (Argus `sensor_id` == `/dev/video` index). Only the 4
calibrated ports c–f are used (port a = video0 is the unused 5th; port b is empty). The
modules are mounted **upside-down** (180° roll). Capture with **`sensor_ids=[1,2,3,4]`**,
`cameras=[cam1,cam2,cam3,cam4]` (`sensor_ids=[0,1,2,3]` is wrong — it grabs port a).

| cam | port | /dev/video | sensor_id | rig dir | HFOV/VFOV @720p | @1640×1232 |
|-----|------|-----------|-----------|---------|-----------------|------------|
| cam1 | c | video1 | 1 | −Y (back)  | 94.9° / 52.9° | 125.2° / 91.5° |
| cam2 | d | video2 | 2 | −X (left)  | 97.7° / 54.1° | 126.8° / 92.7° |
| cam3 | e | video3 | 3 | +Y (front) | 96.8° / 53.4° | 128.2° / 91.8° |
| cam4 | f | video4 | 4 | +X (right) | 98.7° / 54.4° | 129.1° / 94.4° |

(FOV from the KB calib inverted at the image edges; 720p is a 16:9 crop, 1640×1232 the full
2×2-binned sensor. rig frame: X=right, Y=forward, Z=up.)

### One-time board prep
[scripts/setup_tx2_docker.sh](scripts/setup_tx2_docker.sh) (run with `sudo`):
installs the NVIDIA container runtime, registers the `nvidia` Docker runtime,
moves `data-root` to the SD, installs the `docker compose` v2 plugin, and adds the
user to the `docker` group.

---

## 2. Software design

### The governing constraint
The TX2 is frozen at **CUDA 10.2 / driver r440 / sm_62** and cannot be upgraded
(NVIDIA never shipped JetPack 5 for it). That rules out modern AI-perception stacks
and the *stock* cuVSLAM (which targets CUDA 12/13). The strategy: run **classical,
GPU-accelerated, feature-based** perception — and **port** cuVSLAM down to CUDA 10.2
rather than settle for a CPU estimator (too slow for 4–6 cameras).

### Pipeline (target)
```
4–6× IMX219 (160° fisheye)
  → GStreamer / nvarguscamerasrc capture → ROS 2 image topics
  → cuVSLAM multicam VIO  (Odometry::Track(vector<Image>...))   → odometry / pose
  → [planned] stereo/MVS depth → occupancy grid (Octomap / Nav2 costmap)
  → [planned] IPM surround BEV
```

### cuVSLAM CUDA-10.2 port (core of this repo)
`third_party/cuVSLAM` is the upstream library (submodule, pinned **v15.0.0**),
adapted to CUDA 10.2 entirely through **idempotent, build-time fixes** — the
submodule itself is never modified in git. The fixes (see
[docs/cuvslam_tx2.md](docs/cuvslam_tx2.md)): C++ std 17→14, an automated
C++17→C++14 source converter (nested namespaces / inline vars), arch pin to sm_62,
CUDA-11 cuSOLVER enum guards, `cudaMallocAsync`→`cudaMalloc`, an `std::pmr` guard,
and `-lstdc++fs`.

### Why Docker at all (the board is already 18.04)?
The TX2 rootfs *is* Ubuntu 18.04 + CUDA 10.2, so cuVSLAM **could** be built
natively. Docker is used for **the ROS 2 runtime**: native 18.04 only reaches ROS 2
Eloquent/Dashing (both EOL), and modern ROS 2 (Foxy) needs Ubuntu 20.04. The
container provides 20.04 + Foxy on top of the 18.04 kernel/r440 driver, plus
isolation/reproducibility matching the rover workflow — without touching the board
rootfs.

### One container does everything
The **r440 driver only initializes up to glibc 2.31 (Ubuntu 20.04)** — 22.04/24.04
fail driver init (glibc-gated). And nvcc 10.2 needs host **gcc ≤ 8**. 20.04 carries
both ROS 2 Foxy *and* an installable `gcc-8`, so a **single image** builds and runs
the whole stack:

| Image | Base | Purpose |
|-------|------|---------|
| `cuvslam-foxy:tx2` | Ubuntu 20.04 + ROS 2 Foxy + gcc-8 | Build `libcuvslam.so` (nvcc→g++-8), build **and** run the ROS 2 nodes + cuVSLAM GPU ([Dockerfile.cuvslam-foxy](docker/Dockerfile.cuvslam-foxy)) |

It **bind-mounts the host CUDA 10.2** at `/usr/local/cuda` and uses the **nvidia
container runtime** (rover-style). nvcc compiles the `.cu` files with `g++-8`; the
ROS 2 nodes build with gcc-9 and link the `.so` via libstdc++ forward-compatibility.
GPU + EGL access in a non-`l4t-base` container needs the `tegra`/`tegra-egl` ld
paths + `ldconfig` so `libcuda.so.1` / `libEGL_nvidia` resolve (baked into the image).
See [docs/build_and_run.md](docs/build_and_run.md) for the full build/run guide.

> **Future: fused capture + VO.** This same container already carries Argus, EGL,
> CUDA and cuVSLAM, so the planned single-process node that feeds Argus NVMM
> straight to cuVSLAM as GPU memory (`is_gpu_mem=true`, avoiding the GPU→CPU→DDS→GPU
> copies) builds here too — no new image needed. See [§ data path](docs/build_and_run.md#5-data-path--a-performance-note).

---

## 3. Repo layout

| Path | What |
|------|------|
| [docker-compose.yml](docker-compose.yml) | All container params + per-purpose services (`build-cuvslam`, `build-ws`, `fused`, `modular`, `capture`, `shell`) |
| [docker/](docker/) | Single 20.04/Foxy build+run container ([Dockerfile.cuvslam-foxy](docker/Dockerfile.cuvslam-foxy)) + entrypoint |
| [docs/build_and_run.md](docs/build_and_run.md) | Docker setup, build & run guide |
| [ros2/bev_camera/](ros2/bev_camera/) | 4-camera libargus capture node → `/camN/image_raw` |
| [ros2/bev_cuvslam/](ros2/bev_cuvslam/) | 4-camera cuVSLAM multicam VO node → `/odom` |
| [scripts/](scripts/) | Build/port/setup/calib shell + python helpers (traced below) |
| [intrinsic_calib.py](intrinsic_calib.py) | Fisheye intrinsic calibration (KANNALA_BRANDT) |
| [docs/cuvslam_tx2.md](docs/cuvslam_tx2.md) | cuVSLAM port: rationale, fixes, reproduce |
| [third_party/cuVSLAM/](third_party/cuVSLAM/) | Upstream cuVSLAM (submodule, v15.0.0) |
| `OmniNxt.pdf` | Reference paper (omnidirectional aerial perception) |

### Scripts (what runs what)

Full index with one-line descriptions: **[scripts/README.md](scripts/README.md)**. The key ones on
the build/port path:

| Script | Run where | Purpose / called by |
|--------|-----------|---------------------|
| [scripts/setup_tx2_docker.sh](scripts/setup_tx2_docker.sh) | TX2 host (sudo) | One-time: Docker + nvidia runtime on the SD card |
| [scripts/port/build_and_validate.sh](scripts/port/build_and_validate.sh) | TX2 host | One command: builds the Foxy image → `libcuvslam.so` → WarmUpGPU smoke test |
| [scripts/build_cuvslam_tx2gpu.sh](scripts/build_cuvslam_tx2gpu.sh) | **inside** Foxy container | The CUDA-10.2 port build (applies all fixes); called by `build_and_validate.sh` |
| [scripts/port/downgrade_cuvslam_cpp17.py](scripts/port/downgrade_cuvslam_cpp17.py) | inside container | Idempotent C++17→C++14 source converter; called by `build_cuvslam_tx2gpu.sh` |
| [scripts/port/smoke_test.cpp](scripts/port/smoke_test.cpp) | inside container | `WarmUpGPU()` runtime validation; compiled by `build_and_validate.sh` |
| [scripts/calib/grid_view_tx2.sh](scripts/calib/grid_view_tx2.sh) | TX2 host | Live 4-camera grid preview (calibration framing aid) |
| [docker/entrypoint_foxy.sh](docker/entrypoint_foxy.sh) | container entry | `ldconfig` (tegra libcuda/EGL) + source ROS 2 / workspace |

For the run/capture/calibration helpers (VO wrappers, panorama montage, intrinsic + extrinsic
calibration, measurement) see the [scripts index](scripts/README.md).

---

## 4. Build & run

All container parameters live in [`docker-compose.yml`](docker-compose.yml), so most steps are short
`docker compose` commands run from the repo root **on the TX2** (`/media/nvidia/workspace/BEV_Jetson`).

### Accessing the board (eth or wifi)

The board is reachable two ways — pick whichever link is up and set these once in your host shell:

```bash
export TX2=tx2-eth        # ethernet  10.42.0.157   (or:  export TX2=tx2-wlan   # wifi  192.168.0.168)
export BEVDIR=/media/nvidia/workspace/BEV_Jetson
```

Every board command then has two forms — **on TX2** (after `ssh $TX2`) or **from host** (wrap it):

```bash
ssh    $TX2 "cd $BEVDIR && <command>"      # non-interactive (capture, build, calibration capture)
ssh -t $TX2 "cd $BEVDIR && <command>"      # interactive / needs a TTY (compose run you watch live)
```

Sync code host→board: edit on host → `git push` → `ssh $TX2 "cd $BEVDIR && git pull --no-recurse-submodules"`.
Sudo password on the board is `nvidia`.

### 4.1 Build

```bash
# on TX2 (repo root)
sudo ./scripts/setup_tx2_docker.sh        # 0. one-time board prep (log out/in for the docker group)
docker compose build                      # 1. build the cuvslam-foxy:tx2 image
docker compose run --rm build-cuvslam     # 2. build libcuvslam.so (CUDA-10.2 port)
docker compose run --rm build-ws          # 3. colcon build the ROS 2 workspace (bev_camera + bev_cuvslam)
```
From host (example): `ssh $TX2 "cd $BEVDIR && docker compose run --rm build-ws"`.
`./scripts/port/build_and_validate.sh` does image → `libcuvslam.so` → WarmUpGPU smoke test in one shot.

### 4.2 Run VO

```bash
# on TX2
docker compose run --rm fused             # fused zero-copy Argus->cuVSLAM VO (recommended)
docker compose run --rm modular           # modular capture + VO (ROS2 GPU->CPU->GPU), for comparison
RECORD=1 docker compose run --rm fused    # also bag /cuvslam/odometry + /tf into bags/
```
From host (watch it live): `ssh -t $TX2 "cd $BEVDIR && docker compose run --rm fused"`.
Params: [`ros2/bev_cuvslam/config/fused_vo_params.yaml`](ros2/bev_cuvslam/config/fused_vo_params.yaml).
See **[docs/build_and_run.md](docs/build_and_run.md)** for the full guide + **fused vs modular** numbers.

### 4.3 Capture & view the surround panorama

Montage (4 fisheye views + stitched 360° panorama) into one image:
```bash
# on TX2
./scripts/capture_montage_tx2.sh /tmp/bev.png
# from host (capture, then pull the image off the board):
ssh $TX2 "cd $BEVDIR && ./scripts/capture_montage_tx2.sh /tmp/bev.png" && scp $TX2:/tmp/bev.png .
```
Live panorama for rviz (publishes `/bev/panorama`, mono8): `docker compose run --rm panorama`
(then add an Image display on `/bev/panorama`). Just the raw camera topics: `docker compose run --rm capture`.

If Argus wedges (after a SIGKILL'd run), reset the daemon:
`ssh $TX2 "echo nvidia | sudo -S systemctl restart nvargus-daemon"`. A stuck **port D** camera needs
`ssh $TX2 "echo 1 | sudo tee /sys/bus/i2c/devices/2-0010/j106_reset_recover"` (see
[auvidea-j106-tx2/README.md](../auvidea-j106-tx2/README.md#L299)).

### 4.4 Intrinsic calibration (per camera)

Identify which Argus sensor-id is which physical camera (live grid on the TX2 HDMI):
```bash
./scripts/calib/grid_view_tx2.sh "0 1 2 3 4"            # on TX2 (HDMI display)
```
Calibrate one camera against a checkerboard (Kannala-Brandt fisheye), live on the TX2 display:
```bash
# on TX2 — sensor-id 1 = port c = cam1; 11x9 inner-corner board, 30 mm squares; modules are upside-down (--flip 180)
python3 scripts/calib/online_calib.py --id 1 --board 11x9 --square 30 --flip 180 --width 1640 --height 1232
```
Or calibrate offline from saved raw frames, and scale a calibrated set to another output resolution:
```bash
python3 scripts/calib/offline_calib.py --id 1 --board 11x9 --square 30 --out config/calib
python3 scripts/calib/scale_calib.py --in scripts/config/1640x1232 --out scripts/config/832x624 --to-width 832 --to-height 624
```
Intrinsics live in `scripts/config/<WxH>/camN.yaml`.

### 4.5 Extrinsic calibration (rig rotations → panorama seams)

Full procedure + how to read the output: **[docs/extrinsic_calibration.md](docs/extrinsic_calibration.md)**.
Short version (do it in good light, rig aimed at a distant textured scene):
```bash
# 1. capture N sets on the TX2 while you slowly pan the rig
ssh $TX2 "cd $BEVDIR && ./scripts/calib/capture_calib_sets.sh 10 3"
# 2. pull the sets to the host
cd scripts/calib/capture && rm -rf set* && scp -qr $TX2:$BEVDIR/scripts/calib/capture/'set*' . && cd -
# 3. solve on the host (writes config/rig/rig_extrinsics_calibrated.yaml + a before/after render)
python3 scripts/calib/extrinsic_calib.py --images scripts/calib/capture/set*
# 4. deploy: panorama_params already points at the calibrated yaml -> commit/push, then on the board:
git add config/rig/rig_extrinsics_calibrated.yaml && git commit -m "recalibrate extrinsics" && git push
ssh $TX2 "cd $BEVDIR && git pull --no-recurse-submodules && ./scripts/capture_montage_tx2.sh /tmp/bev.png"
```

### 4.6 Tune the stitch (panorama params)

Edit [`ros2/bev_cuvslam/config/panorama_params.yaml`](ros2/bev_cuvslam/config/panorama_params.yaml),
then re-run `docker compose run --rm panorama` (or the montage). The tunable knobs:

| param | default | effect |
|---|---|---|
| `pano_width` × `pano_height` | 1920 × 540 | output equirect canvas size |
| `elevation_max_deg` | 50 | vertical coverage (±); beyond it the poles are black |
| `fisheye_fov_half_deg` | 65 | per-camera half-HFOV used in the remap (~127/2 at full sensor) |
| `feather_deg` | 20 | width of the overlap blend band (bigger = softer seams) |
| `flip_180` | true | apply the 180° roll for the upside-down mounting |
| `rig_extrinsics` | …`_calibrated.yaml` | which extrinsics file to stitch with |
| `save_video` | "" | set a path (e.g. `bags/pano.mp4`) to also record |

For hands-on tuning, the interactive tuner runs on the **host** against captured frames (sliders for
each camera's yaw/pitch/roll + translation + scene depth; live re-render; **Save** → `rig_extrinsics_tuned.yaml`):
```bash
python3 scripts/calib/pano_tuner.py        # open http://localhost:8000  (scroll=zoom, drag=pan)
```

---

## 5. Roadmap

- [x] TX2 board prep, Docker on SD, GPU passthrough into containers
- [x] **Port cuVSLAM to CUDA 10.2** — build + runtime-validated on the r440 driver
- [x] gcc-8 `.so` loads + links in the ROS 2 **Foxy (20.04)** container
- [x] ROS 2 multicam VO node (`bev_cuvslam`) — N cameras via `Odometry::Track(vector<Image>)`
- [x] 4× IMX219 (160° fisheye) capture (`bev_camera`, libargus) → ROS 2 topics + fisheye/extrinsic calibration
- [ ] Run capture → VO end-to-end; verify tracking
- [ ] Fix the 6th camera (i2c bus 1 @ `0x12`); add IMU node + EKF fusion (cuVSLAM multicam can't fuse IMU directly)
- [ ] Depth / occupancy grid; IPM surround BEV
