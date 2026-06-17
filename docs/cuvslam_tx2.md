# cuVSLAM on Jetson TX2 (J106) — port notes

## Goal
Port NVIDIA cuVSLAM (`third_party/cuVSLAM`, submodule pinned to **v15.0.0**) to run
4× IMX219 multi-camera VIO on the TX2/J106, in a Docker env mirroring the
`ackermann_rover_humble` setup (Ubuntu 24.04 + ROS 2 Jazzy, host CUDA bind-mounted).

## Board (probed 2026-06-17, `ssh tx2-eth`)
- L4T **R32.7.6** / JetPack 4.6.x, kernel 4.9.337-tegra (patched j106 camera kernel)
- **CUDA 10.2.300**, driver **r440**, Ubuntu 18.04 host, Tegra186 (**sm_62**)
- 5/6 IMX219 up (`/dev/video0–4`); 160° lenses; no IMU yet
- SD card 117G at `/media/nvidia/workspace` (data-root + workspace live here)

## Expected outcome on TX2: build FAILS (three hard CUDA-10.2 walls)
1. cuVSLAM CUDA TUs are **C++17**; `nvcc 10.2` only accepts c++03/11/14
   (confirmed: `nvcc fatal: Value 'c++17' is not defined`). C++17 device code
   needs CUDA 11.0+.
2. `nvcc 10.2` requires host **gcc ≤ 8**; Ubuntu 24.04 ships gcc-13 (no gcc-8).
3. ~52 uses of `cudaMallocAsync` / CUDA-Graphs need CUDA 11.2+ **and** driver
   r470+. TX2 is frozen at CUDA 10.2 / r440 (no JetPack 5 for TX2, ever).

The TX2 cannot be upgraded past JetPack 4.6, so these are permanent. cuVSLAM runs
only on Orin/Thor (CUDA 12+). On the TX2, the efficient VIO path is
**VINS-Fisheye / D2SLAM** (build against CUDA 10.2) — the dev container here is
the correct base for that pivot too.

## Workflow
```bash
# On the TX2 (in /media/nvidia/workspace/BEV):
sudo ./scripts/setup_tx2_docker.sh        # one-time: runtime + data-root on SD + docker group
export ARCH=$(uname -m)                    # aarch64
docker compose -f docker/docker-compose.yml build bev_cuvslam
docker compose -f docker/docker-compose.yml up -d bev_cuvslam
docker compose -f docker/docker-compose.yml exec bev_cuvslam bash
#   inside the container:
./scripts/build_cuvslam.sh                 # captures the wall in build/cuvslam_build.log
```
