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

### Two-container design (why)
nvcc 10.2 requires host **gcc ≤ 8** (so cuVSLAM is built on Ubuntu 18.04), but
ROS 2 needs a modern Ubuntu. So:

| Image | Base | Purpose |
|-------|------|---------|
| `cuvslam-build:tx2` | Ubuntu 18.04 + gcc-8 | Build `libcuvslam.so` for CUDA 10.2 ([Dockerfile.cuvslam-build](docker/Dockerfile.cuvslam-build)) |
| `bev_cuvslam_*_jazzy` | Ubuntu 24.04 + ROS 2 Jazzy | Runtime / ROS 2 dev ([Dockerfile](docker/Dockerfile), [docker-compose.yml](docker/docker-compose.yml)) |

Both **bind-mount the host CUDA 10.2** at `/usr/local/cuda` and use the **nvidia
container runtime** (rover-style). The gcc-8-built `.so` links into the gcc-13
Jazzy container via libstdc++ forward-compatibility. GPU access in a non-`l4t-base`
container needs `/etc/ld.so.conf.d/nvidia-tegra.conf` + `ldconfig` so `libcuda.so.1`
resolves (baked into the build image).

---

## 3. Repo layout

| Path | What |
|------|------|
| [docker/](docker/) | Build + runtime container definitions and compose |
| [scripts/setup_tx2_docker.sh](scripts/setup_tx2_docker.sh) | One-time board Docker/runtime prep |
| [scripts/build_cuvslam_tx2gpu.sh](scripts/build_cuvslam_tx2gpu.sh) | cuVSLAM CUDA-10.2 port build (applies all fixes) |
| [scripts/port/downgrade_cuvslam_cpp17.py](scripts/port/downgrade_cuvslam_cpp17.py) | Idempotent C++17→C++14 source converter |
| [scripts/port/build_and_validate.sh](scripts/port/build_and_validate.sh) | One command: build image + lib + runtime smoke test |
| [scripts/port/smoke_test.cpp](scripts/port/smoke_test.cpp) | `WarmUpGPU()` runtime validation |
| [intrinsic_calib.py](intrinsic_calib.py) | Fisheye intrinsic calibration (KANNALA_BRANDT) |
| [docs/cuvslam_tx2.md](docs/cuvslam_tx2.md) | cuVSLAM port: rationale, fixes, reproduce |
| [third_party/cuVSLAM/](third_party/cuVSLAM/) | Upstream cuVSLAM (submodule, v15.0.0) |
| `OmniNxt.pdf` | Reference paper (omnidirectional aerial perception) |

---

## 4. Build & run

```bash
# On the TX2, repo at /media/nvidia/workspace/BEV:

# 1. One-time board prep (root)
sudo ./scripts/setup_tx2_docker.sh      # log out/in afterwards for the docker group

# 2. Build + runtime-validate cuVSLAM for CUDA 10.2 (one command)
./scripts/port/build_and_validate.sh
#    -> third_party/cuVSLAM/build_tx2gpu/bin/libcuvslam.so
#    -> "WarmUpGPU() completed ... on the r440 driver"

# 3. ROS 2 Jazzy dev container (host CUDA + GPU)
export ARCH=$(uname -m)
docker compose -f docker/docker-compose.yml up -d bev_cuvslam
docker compose -f docker/docker-compose.yml exec bev_cuvslam bash
```

---

## 5. Roadmap

- [x] TX2 board prep, Docker on SD, GPU passthrough into containers
- [x] **Port cuVSLAM to CUDA 10.2** — build + runtime-validated on the r440 driver
- [ ] Confirm the gcc-8 `.so` loads + runs in the ROS 2 Jazzy (24.04) container
- [ ] ROS 2 multicam wrapper — extend a single-pair node to N cameras via `Odometry::Track(vector<Image>...)`
- [ ] 4× IMX219 (160° fisheye) capture → ROS 2 topics + fisheye/extrinsic calibration
- [ ] Fix the 6th camera (i2c bus 1 @ `0x12`); add an IMU for inertial mode
- [ ] Depth / occupancy grid; IPM surround BEV
