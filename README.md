# BEV_Jetson

OmniNxt-style omnidirectional visual perception on a **Jetson TX2 / Auvidea J106+M110**: four
hardware-triggered fisheye cameras, an IMU and a rangefinder feeding **NVIDIA cuVSLAM**, plus a
stitched 360° panorama and a ground-plane BEV.

The board is the whole story. The TX2 is stuck on **JetPack 4.4 / CUDA 10.2 / driver r440**, which
only initializes up to glibc 2.31, which means **Ubuntu 20.04 and ROS 2 Foxy**. cuVSLAM ships for
CUDA 11+ and C++17. Getting it to run here — the port, the toolchain, and the camera models it will
accept — is most of what this repo contains.

**Status.** VO runs end to end: four cameras → eight virtual pinholes → `cuvslam::Odometry` in
`Multicamera` mode, ~11 Hz on 4-camera replay. Intrinsics and rig extrinsics are calibrated. Current
work is diagnostic visualization (Rerun) and absolute scale.

---

## 1. Hardware

| part | detail |
|---|---|
| Compute | Jetson TX2 on Auvidea J106 + M110 camera carrier |
| Cameras | **4× IMX296** global shutter, 1456×1088 mono, **~192° fisheye** |
| Sync | STM32 on the M110 drives one trigger edge to all four; `t_frame = SOF − exposure/2` |
| IMU | MPU-9250 over SPI, GPIO-edge timestamped (`bev_imu`) |
| Range | single-point rangefinder (`bev_range`), for scale and floor height |
| Software | JetPack 4.4, CUDA 10.2, sm_62, Ubuntu 20.04, ROS 2 Foxy — all inside one Docker image |

Two frames, and they are not the same one:

- **body** — FLU: x forward, y left, z up. The vehicle frame; ROS `base_link`.
- **rig** — cuVSLAM's word for its own reference frame.

The cameras sit at ±45° and ±135° in body yaw, 90° apart, and the modules are mounted **inverted**
(180° roll). The authoritative layout, including the measured IMU axes, is
[config/rig/rig_layout.yaml](config/rig/rig_layout.yaml) — calibration results are *checked against*
it, because a dropped 180° roll or a swapped axis still produces plausible-looking numbers.

## 2. Why virtual pinholes

cuVSLAM accepts four distortion models, and **omni/Mei is not one of them**. Its `Fisheye` model is
also mathematically capped below 180° FOV. Our lenses are ~192°, so neither direct route is legal.

Instead each fisheye is remapped on the GPU into **two virtual pinholes**, giving cuVSLAM eight plain
`Pinhole` cameras that tile the ring. This is the OmniNxt architecture, and it is why the pipeline
calibrates omni in the first place. The full argument — including the arctan proof of the 180°
ceiling and the frustum-overlap gate that otherwise rejects a surround rig — is in
[docs/cuvslam_tx2.md](docs/cuvslam_tx2.md).

## 3. Quick start

Everything runs through [docker-compose.yml](docker-compose.yml) from the repo root **on the TX2**
(`/media/nvidia/workspace/BEV_Jetson`).

```bash
export TX2=tx2-eth                              # or tx2-wlan
export BEVDIR=/media/nvidia/workspace/BEV_Jetson
ssh    $TX2 "cd $BEVDIR && <command>"           # non-interactive
ssh -t $TX2 "cd $BEVDIR && <command>"           # interactive (a run you watch live)
```

Sync code host→board: edit on host → `git push` → `ssh $TX2 "cd $BEVDIR && git pull --no-recurse-submodules"`.

**Build**

```bash
sudo ./scripts/setup_tx2_docker.sh        # one-time board prep (log out/in for the docker group)
docker compose build                      # the cuvslam-foxy:tx2 image
docker compose run --rm build-cuvslam     # libcuvslam.so, CUDA-10.2 port
docker compose run --rm build-ws          # colcon build the ROS 2 workspace
```

`./scripts/port/build_and_validate.sh` does image → `libcuvslam.so` → `WarmUpGPU` smoke test in one shot.

**Run**

```bash
docker compose run --rm fused             # fused zero-copy Argus -> cuVSLAM VO (recommended)
docker compose run --rm modular           # modular capture + VO, for comparison
docker compose run --rm logonly           # raw frames straight to disk: no DDS, no VO, no rosbag2
docker compose run --rm panorama          # publish /bev/panorama (mono8) for rviz
```

`docker compose run --rm shell` drops you into the container. See
[docs/build_and_run.md](docs/build_and_run.md) for the full guide, the recording modes and their
measured throughput, and the fused-vs-modular numbers.

**Look at a run**

```bash
python3 scripts/vo/rerun_multicam.py <obs_dir> --panorama --bev-fit-plane --serve
```

A Rerun viewer with the eight pinholes, the trajectory and landmarks, a ground-plane BEV and a 360°
panorama. `./scripts/capture_montage_tx2.sh /tmp/bev.png` gets a single still montage off the board.

**When a camera goes black** (a montage tile is dark, or the node logs `acquireFrame timeout`), reset
from least to most invasive — each step fixes a different failure:

```bash
# 1. Argus daemon wedged (most common, after a SIGKILL'd run leaked a session) -> all cams may go black
ssh $TX2 "docker ps -aq --filter ancestor=cuvslam-foxy:tx2 | xargs -r docker rm -f; echo nvidia | sudo -S systemctl restart nvargus-daemon"

# 2. One camera binds but won't stream: pulse the shared reset from a BOUND sibling on the same bus
ssh $TX2 "echo 1 | sudo tee /sys/bus/i2c/devices/2-0010/j106_reset_recover; echo nvidia | sudo -S systemctl restart nvargus-daemon"

# 3. Still dead -> true cold power-cycle (pull the DC barrel jack ~10 s; a soft reboot leaves the rails up)
```

If it binds at i2c (`i2cdetect -y -r 2` shows `UU`) but never delivers frames after all three steps,
it is a physical CSI fault: re-seat the ribbon at both ends, or swap that port's module. Always stop
runs with `docker stop`, not `docker rm -f` — a SIGKILL leaks the Argus session and wedges the daemon.

## 4. Calibration

| stage | output | how |
|---|---|---|
| Intrinsics | `config/calib/imx296_1456x1088/camN.yaml` (omni/Mei + radtan) | tartancalib / quarterKalibr on recorded bags |
| Virtual stereo | `config/rig/virtual_stereo_imx296.yaml` | `scripts/calib/gen_virtual_stereo.py` |
| Extrinsics | `config/rig/rig_extrinsics_imx296.yaml` | `scripts/calib/pair_extrinsics.py` + `close_rig_ring.py` |
| Ground plane | `config/rig/ground_plane.yaml` | fitted from landmarks near the pose |

Procedure and how to read the results: **[docs/extrinsic_calibration.md](docs/extrinsic_calibration.md)**.
The end-to-end capture→solve walkthrough is
[scripts/calib/calibration_pipeline.ipynb](scripts/calib/calibration_pipeline.ipynb).

`extrinsic_calib.py`, `pano_tuner.py`, `scale_calib.py` and `fold_roll_for_vo.py` are IMX219-lineage
(equidistant intrinsics, `board_center` rig format) and now exit on startup — their inputs no longer
exist. Porting them means teaching them the Mei projection; `mei_project()` in
[bev_panorama_node.cpp](ros2/bev_cuvslam/src/bev_panorama_node.cpp) is the reference.

## 5. Repo layout

| path | what |
|---|---|
| [ros2/bev_camera](ros2/bev_camera) | Argus capture node — trigger-synced frames plus `FrameMeta` |
| [ros2/bev_cuvslam](ros2/bev_cuvslam) | VO nodes (fused + modular), virtual-pinhole and panorama CUDA kernels |
| [ros2/bev_imu](ros2/bev_imu) | MPU-9250 SPI reader with GPIO-edge timestamps |
| [ros2/bev_range](ros2/bev_range) | rangefinder node |
| [config/calib](config/calib) | camera intrinsics, with dated archives |
| [config/rig](config/rig) | rig layout, extrinsics, virtual stereo, ground plane |
| [docker](docker) | the Foxy / CUDA-10.2 images and entrypoint |
| [patch](patch) | the cuVSLAM and cuNLS CUDA-10.2 port patches |
| [scripts](scripts/README.md) | build, capture, calibration, replay and analysis tools — see the index |
| [openspec](openspec) | change proposals, designs and task logs |
| [third_party](third_party) | cuVSLAM, OKVIS2, OpenMAVIS submodules |

## 6. Docs

| doc | covers |
|---|---|
| [docs/build_and_run.md](docs/build_and_run.md) | full build and run guide, recording modes, benchmarks |
| [docs/cuvslam_tx2.md](docs/cuvslam_tx2.md) | the CUDA-10.2 port, and which camera models cuVSLAM can consume |
| [docs/timestamps.md](docs/timestamps.md) | the camera/IMU timestamp contract — the four rules |
| [docs/extrinsic_calibration.md](docs/extrinsic_calibration.md) | rig extrinsics procedure |
| [scripts/README.md](scripts/README.md) | index of every script, and where it runs |

## 7. Roadmap

- [x] Port cuVSLAM to CUDA 10.2 / C++14, running on the TX2 GPU
- [x] Argus capture with a hardware trigger and one clock for camera + IMU
- [x] Calibrate the IMX296 rig: intrinsics, virtual stereo, extrinsics
- [x] VO end to end — 8 virtual pinholes, `Multicamera` mode
- [x] Replay diagnostics in Rerun: pinholes, trajectory, BEV, 360° panorama
- [ ] Resolve absolute scale (suspected ~15–20% underestimate; it enters only via extrinsic translations)
- [ ] Fuse the IMU into the VO
- [ ] Port the nearest-axis panorama seam into `bev_panorama_node`
- [ ] Ground-plane BEV stitch on the board
