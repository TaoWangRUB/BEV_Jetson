# BEV_Jetson

OmniNxt-style omnidirectional visual perception on a **Jetson TX2 / Auvidea J106+M110**: four
hardware-triggered fisheye cameras, an IMU and a rangefinder feeding **NVIDIA cuVSLAM**, plus a
stitched 360° panorama and a ground-plane BEV.

The TX2 is stuck on **JetPack 4.4 / CUDA 10.2 / driver r440**, which only initializes up to
glibc 2.31 — so **Ubuntu 20.04 and ROS 2 Foxy** on the board. cuVSLAM ships for CUDA 11+ and
C++17; the TX2 path is a source port. Capture, live VO, and raw logging still run **on the
board**. Offline bag → VO resim runs on the **host** (x86_64 / CUDA 12.x) so large bags do not
OOM the TX2.

**Status.** VO runs end to end: four cameras → eight virtual pinholes → `cuvslam::Odometry` in
`Multicamera` mode (~11–16 Hz on replay). Intrinsics and rig extrinsics are calibrated. Current
work is diagnostic visualization (Rerun) and absolute scale.

### Two Docker environments

| | **TX2** (`docker-compose.yml`) | **Host** (`docker-compose.host.yml`) |
|---|---|---|
| Image | `cuvslam-foxy:tx2` | `bev-host-cuvslam:latest` |
| Dockerfile | [docker/Dockerfile.cuvslam-foxy](docker/Dockerfile.cuvslam-foxy) | [docker/Dockerfile.host-cuvslam](docker/Dockerfile.host-cuvslam) |
| Arch / CUDA | aarch64, host-mounted **CUDA 10.2**, sm_62 | x86_64, host-mounted **CUDA 12.x**, sm_86 |
| `libcuvslam.so` | `third_party/cuVSLAM/build_tx2gpu/` (TX2 port patch) | `third_party/cuVSLAM/build_host/` (**no** TX2 port) |
| ROS install | `install/` | `install_host/` |
| What it runs | live capture, fused VO, `log_rig`, modular on-board | bag replay via modular VO only (`BEV_BUILD_ARGUS=OFF`) |
| Also | — | [docker/Dockerfile.host-foxy](docker/Dockerfile.host-foxy) = DDS viewer only (no CUDA / no VO) |

Do **not** mix the two trees: never apply the TX2 port patch on a host-build checkout of
`third_party/cuVSLAM`, and do not point the host `LD_LIBRARY_PATH` at `build_tx2gpu`.

---

## 1. Hardware

| part | detail |
|---|---|
| Compute | Jetson TX2 on Auvidea J106 + M110 camera carrier |
| Cameras | **4× IMX296** global shutter, 1456×1088 mono, **~192° fisheye** |
| Sync | STM32 on the M110 drives one trigger edge to all four; `t_frame = SOF − exposure/2` |
| IMU | MPU-9250 over SPI, GPIO-edge timestamped (`bev_imu`) |
| Range | single-point rangefinder (`bev_range`), for scale and floor height |
| Board software | JetPack 4.4, CUDA 10.2, sm_62, Ubuntu 20.04, ROS 2 Foxy (`cuvslam-foxy:tx2`) |
| Host software | x86_64, CUDA 12.x, Foxy in `bev-host-cuvslam` (offline resim) |

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

## 3. Procedures

**Live capture / logging / fused VO** run through [docker-compose.yml](docker-compose.yml) from
the repo root **on the TX2** (`/media/nvidia/workspace/BEV_Jetson`). **Offline bag → VO** uses
[docker-compose.host.yml](docker-compose.host.yml) on the laptop. Edit on the host, push, pull on
the board.

```bash
export TX2=tx2-eth                              # or tx2-wlan
export BEVDIR=/media/nvidia/workspace/BEV_Jetson
ssh    $TX2 "cd $BEVDIR && <command>"           # non-interactive
ssh -t $TX2 "cd $BEVDIR && <command>"           # interactive (a run you watch live)

# sync host → board
git push
ssh $TX2 "cd $BEVDIR && git pull --no-recurse-submodules"
```

Script index (what each file does, and where it runs): [scripts/README.md](scripts/README.md).
Deep build / fused-vs-modular numbers: [docs/build_and_run.md](docs/build_and_run.md).

### 3.0 One-time build

**On the TX2** (live VO + logging):

```bash
# from $BEVDIR
sudo ./scripts/setup_tx2_docker.sh        # once: docker + nvidia runtime (log out/in for the group)
docker compose build                      # cuvslam-foxy:tx2 image
docker compose run --rm build-cuvslam     # libcuvslam.so (CUDA-10.2 port → build_tx2gpu/)
docker compose run --rm build-ws          # ROS 2 workspace → install/
```

`./scripts/port/build_and_validate.sh` does image → `libcuvslam.so` → `WarmUpGPU` smoke test in one
shot. Re-run `build-ws` after any C++ change; re-run `build-cuvslam` only after a cuVSLAM / patch
change.

**On the host** (offline bag → VO; needs NVIDIA Container Toolkit + a CUDA 12.x toolkit at
`/usr/local/cuda`, or set `CUDA_HOST_MOUNT`):

```bash
# from the repo root on the laptop
docker compose -f docker-compose.host.yml build
docker compose -f docker-compose.host.yml run --rm build-cuvslam-host   # → build_host/
docker compose -f docker-compose.host.yml run --rm build-ws-host        # → install_host/
```

Re-run `build-ws-host` after `bev_cuvslam` / `bev_camera` changes; re-run `build-cuvslam-host`
only after a cuVSLAM submodule bump. The host build **refuses** if the TX2 port stamp is present
under `third_party/cuVSLAM/`.


### 3.1 Run the whole pipeline (live VO)

The deployed path is the **fused** node: Argus NVMM → CUDA → cuVSLAM in one process (no DDS
images). Prefer it for measurement. The **modular** path (capture node + VO over DDS) is for
bring-up and bag replay — the fused node cannot replay a bag, by construction.

**Preflight (do this after every reboot / power cycle).** `trigger_mode` and generator polarity
reset silently; a free-running or wrong-polarity rig still produces images and then wastes a run.

```bash
# on the TX2 host — after every reboot / power cycle
echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode
# Prefer USB CDC when the last run left the rangefinder streaming on ttyTHS1
# (a leftover `range auto` stream makes trigctl on THS1 hang → log_rig REFUSING):
python3 /home/nvidia/j106-trigctl.py --port /dev/ttyACM0 raw 'fps 20'
python3 /home/nvidia/j106-trigctl.py --port /dev/ttyACM0 raw 'pol 0'
python3 /home/nvidia/j106-trigctl.py --port /dev/ttyACM0 status
# expect: running=1, polarity=active_low, fps_milli=20000, lidar_stream_div=0
# if a previous log left the stream on:  ... raw 'range auto 0'
sudo systemctl restart nvargus-daemon; sleep 2
```

**Live VO**

```bash
docker compose run --rm fused             # recommended: zero-copy VO → /cuvslam/odometry + TF
# RECORD=1 docker compose run --rm fused  # also bags odometry + TF (no images)

docker compose run --rm modular           # capture + VO over DDS (bring-up / comparison)
docker compose run --rm panorama          # /bev/panorama for rviz
docker compose run --rm shell             # interactive Foxy shell
```

Params: [ros2/bev_cuvslam/config/fused_vo_params.yaml](ros2/bev_cuvslam/config/fused_vo_params.yaml)
(`ports: [c,d,e,f]`, `exposure_us`, `ae_lock: auto`). Override with
`... bev_cuvslam_fused.launch.py params:=/abs/path.yaml`.

**Watch the log for these three.** cuVSLAM reports none of them — it has no image-quality input,
no quality field in `PoseEstimate`, and its own tracking messages only fire when the solve *fails*,
which is not what any of these are. The node checks instead:

| line | what it means |
|---|---|
| `camN is NN% saturated at/above 227` | the scene is brighter than the trigger pulse width can hold. Features die and the pose will freeze, then jump. **Shorten the pulse** (`j106-trigctl.py`), not the AE — AE is locked because under external trigger it cannot reach its actuator and hunts on gain |
| `pose has not moved at all for N sets` | cuVSLAM is repeating its last estimate, not measuring |
| `pose JUMPED x m in y ms` | tracking was lost and re-initialised elsewhere; everything after is in a new frame |

A negative covariance diagonal is reported too — that is a rank-deficient solve, not a large
uncertainty, and it is the only quality signal cuVSLAM actually exposes. Tune the gate with
`saturation_level` / `saturation_warn_fraction`; `cuvslam_verbosity` and `cuvslam_debug_dump_dir`
turn on the library's own logging and its edex dump of every `Track()` call.

**Measured motion test (tape + odometry)** — gates on trigger_mode / polarity / pulse width,
restarts Argus, writes a run directory under `/media/nvidia/workspace/motion_<label>_…`:

```bash
# on the TX2, from $BEVDIR
./scripts/vo/run_motion_test.sh walk1 5.0                 # fused VO, bag odom+tf; tape = 5.0 m
MOTION_SECONDS=60 ./scripts/vo/run_motion_test.sh walk1 5.0   # auto-stop after 60 s
./scripts/vo/run_motion_test.sh walk1 5.0 --record-images     # modular + camera bag (replayable)
# then on the host:
python3 scripts/vo/analyze_motion.py /path/to/motion_walk1_…
```

Do **two passes** if you want both live numbers and a replayable image bag: `--record-images`
competes for CPU/I/O and is a different pipeline. Stop with `Ctrl-C` / `docker stop`, never
`docker rm -f` — a SIGKILL leaks the Argus session.

**Quick visual check that cameras are alive**

```bash
./scripts/capture_montage_tx2.sh /tmp/bev.png    # 4-up still off the board
# or live preview (host + board): scripts/stream/csi_sender.sh + csi_receiver.sh
```

**When a camera goes black** (`acquireFrame timeout`, dark montage tile) — least to most invasive:

```bash
# 1. Argus wedged (usual after SIGKILL)
ssh $TX2 "docker ps -aq --filter ancestor=cuvslam-foxy:tx2 | xargs -r docker rm -f; echo nvidia | sudo -S systemctl restart nvargus-daemon"

# 2. One cam binds but won't stream — shared reset from a bound sibling
ssh $TX2 "echo 1 | sudo tee /sys/bus/i2c/devices/2-0010/j106_reset_recover; echo nvidia | sudo -S systemctl restart nvargus-daemon"

# 3. Still dead → cold power-cycle (pull the DC barrel ~10 s; soft reboot leaves rails up)
```

If i2c shows `UU` but frames never arrive after all three, it is a CSI ribbon fault.

---

### 3.2 Log data (raw cameras + IMU + range)

**Do not bag images on the board.** `ros2 bag record` tops out at ~6–7 fps (writer-bound). Log
**raw** at the trigger rate, then convert offline.

The one command for a full rig recording (cameras + IMU + range into one directory) is
[scripts/log_rig.sh](scripts/log_rig.sh). It runs on the **TX2 host**, gates on trigger_mode /
polarity / generator rate, restarts `nvargus-daemon`, and brackets the cameras with the IMU.

```bash
# on the TX2, from $BEVDIR — prefer 20 fps (see rates below)
python3 /home/nvidia/j106-trigctl.py --port /dev/ttyTHS1 fps 20

MOTION_SECONDS=60 LOG_LABEL=run1 ./scripts/log_rig.sh
# → /home/nvidia/logs/imglog_run1_<stamp>/
#    cam{1..4}.raw          concatenated mono8 (1456×1088)
#    cam{1..4}_index.csv    stamp_ns, byte_offset
#    cam{1..4}.csv          every trigger edge (seq, capture_id, sof, exposure, image=0|1)
#    geometry.txt           width / height / bytes_per_frame
#    imu0.csv               CLOCK_MONOTONIC, brackets the cameras (~200 Hz)
#    range0.csv             one reading per trigger edge by default (RANGE_DIV=1 → 20 Hz
#                           at the preferred fps); join on pulses, not stamp (±1 frame)

# override range rate only if you must (higher than 1 is slower; 1 is the firmware max)
# RANGE_DIV=2 MOTION_SECONDS=60 LOG_LABEL=run1 ./scripts/log_rig.sh
```

**Rates (measured on this board)**

| config | bandwidth | result |
|---|---|---|
| **20 fps, all four on eMMC** (`LOG_DIR=/logs`) | ~127 MB/s | **lossless** interior sets; practical max ~90 s of free eMMC |
| 30 fps, split `LOG_DIRS=/logs,/logs,/sdlog,/ramlog` | ~190 MB/s split | ~97 % complete sets; degrades over the run |
| 30 fps, all four on `/logs` | 190 MB/s vs 136 | **refused** by the bandwidth gate (would drop frames) |

```bash
# 20 fps, single target (default / recommended)
MOTION_SECONDS=60 LOG_LABEL=run1 ./scripts/log_rig.sh

# 30 fps only if you need it — must split, and accept residual loss
python3 /home/nvidia/j106-trigctl.py --port /dev/ttyTHS1 fps 30
LOG_DIRS="/logs,/logs,/sdlog,/ramlog" MOTION_SECONDS=60 LOG_LABEL=run30 ./scripts/log_rig.sh
```

Images alone (no IMU): `EXPOSURE_US=4986 TRIGGER_FPS=20 LOG_DIR=/logs MOTION_SECONDS=60 docker compose run --rm logonly`
— but prefer `log_rig.sh`; it owns the preflight. Cameras-only detail and the bandwidth /
warmup rules live in the header of [scripts/log_raw.sh](scripts/log_raw.sh).

**Space.** eMMC is tight (~16 GB free typical). Delete or copy off finished logs before the next
long run. Quote **motion duration**, not wall-clock recording length — stop when the motion
stops (see §3.3).

**Pull a log to the host**

```bash
scp -r $TX2:/home/nvidia/logs/imglog_run1_<stamp> datasets/
```

---

### 3.3 Analyse logged data

All of these run on the **host** (or anywhere with the repo + python deps), against a raw log
directory. Completeness and usefulness are separate: a stationary rig produces a flawless log
with nothing in it for VO.

**1. Is the log intact? Was the rig moving?**

```bash
python3 scripts/port/check_log_sets.py datasets/imglog_run1_<stamp>
```

Reports complete 4-camera sets, **interior** completeness (excludes the ragged start/stop edge),
the **motion window** from `imu0.csv`, and — when present — **IMU rate / seq gaps / camera
bracket** plus **rangefinder rate / pulse-step misses / divisor**. Look for
`INTERIOR sets: … <- LOSSLESS`, `frames DURING motion`, and range `missed pulse steps: 0`.

**2. Where did a missing edge go?** (sensor missed it vs capture loop too slow)

```bash
python3 scripts/port/locate_frame_loss.py datasets/imglog_run1_<stamp>
# optional: --period-us 50000   # 20 fps; default = median inter-frame gap
```

**3. Convert raw → rosbag2 (Foxy v4), then replay VO on the host**

```bash
# on the host — convert only the moving part (same motion_window() as check_log_sets)
python3 scripts/port/raw_log_to_bag.py datasets/imglog_run1_<stamp> \
  -o /tmp/run1.bag --motion
# optional: --pad-s 1.0   # keep ±1 s of stillness so VO can initialise
# optional: --compress    # zstd the .db3
# avoid --max-frames N alone: it takes the FIRST N frames (often the stationary lead-in)

# preferred: modular VO on the laptop (avoids TX2 OOM on ~60 s / multi-GB bags)
./scripts/vo/replay_host.sh /tmp/run1.bag 0.5
# → datasets/replay_out/odom_<stamp>/   (/cuvslam/odometry + /tf)
# RATE=0.25 for slower play / fewer set drops; default rate is 0.5

# for the Rerun viewer below, which needs the tracked features:
OBS=1 ./scripts/vo/replay_host.sh /tmp/run1.bag 0.25
# → datasets/replay_out/obs_<stamp>/    (+ /cuvslam/landmarks + /cuvslam/observations)
# the landmark export runs on the Track() thread, so keep OBS off for any run whose
# RATE or trajectory is the measurement
```

Foxy’s `ros2 bag play` on this stack has **no `--clock`**; VO matches on image header stamps.
A few percent of sets can show 50 ms skew during playback (bag topics delivered out of lock-step) —
that is a **replay** artifact, not missing frames in the raw log. Prefer `check_log_sets` for
logging health.

The cause is **transport, not compute, and slowing the replay does not fix it.** Measured on
`run1_motion`, whose own header stamps are max 1 µs skew with zero sets over 1 ms — so the
conversion is not at fault:

| replay rate | sets reaching the matcher | skew-gate drops | poses |
|---|---|---|---|
| 0.25× | 961 / 1155 (83 %) | 31 | 933 |
| 0.5× | 937 / 1155 (81 %) | 48 | 907 |
| 1.0× | 967 / 1155 (84 %) | 36 | 961 |

About a sixth of the sets never arrive at **any** rate. The node is nowhere near saturated on the
host — `Track()` is 6.5–10 ms and the remap ~5 ms, so ~15 ms/set, about 65 Hz — and at 1.0× it
sustained 16.7 Hz, i.e. every set that reached it. The loss is `SensorDataQoS()` being
**best-effort** over four 1.5 MB image streams: DDS discards fragments silently, the orphaned
frames pair across trigger edges, and the set fails the 1 ms gate at exactly one frame period
(50 ms at 20 Hz). The fix is reliable QoS and larger DDS buffers, not a slower replay.

TX2-side bag replay still works for short clips (`docker compose run --rm shell` + modular launch +
`ros2 bag play -r 0.5`), but a full ~7 GB motion bag has been OOM-killed on the board.

**4. Visual diagnostics (Rerun)** — eight virtual pinholes with the tracked features, trajectory,
landmarks, optional panorama / BEV. Needs an **`obs_*`** bag (step 3 with `OBS=1`) **and** the camera
bag. `rerun-sdk` will not install into the system python here (PEP 668), so it lives in a venv:

```bash
python3 -m venv --system-site-packages .venv     # once; needs python3.12-venv
.venv/bin/pip install rerun-sdk

.venv/bin/python scripts/vo/rerun_multicam.py datasets/replay_out/obs_<stamp> \
  --images /tmp/run1.bag --t-range 40:52 --frames 300 \
  --panorama --bev-fit-plane --save /tmp/run1.rrd
```

`--t-range START:END` picks a window and `--frames` then subsamples **that**; without it `--frames`
subsamples the whole run, which on a 57 s log is one pose in five — enough to step straight over a
0.75 s tracking freeze. `--save` writes an `.rrd` (`.venv/bin/rerun file.rrd`), `--spawn` opens the
desktop viewer, `--serve` serves it. Budget ~0.3 MB/frame, or ~1.2 s and ~0.45 MB per frame with
`--panorama --bev-fit-plane --fisheye` all on: render a long run **in `--t-range` chunks**, because
the whole recording is buffered in memory before the file is written and a 965-frame full-featured
render has been OOM-killed on a 31 GB laptop.

For just the 360° stitch, skip Rerun entirely — [scripts/vo/make_panorama.py](scripts/vo/make_panorama.py)
writes PNGs or an mp4 straight from a raw log or a camera bag:

```bash
.venv/bin/python scripts/vo/make_panorama.py datasets/imglog_run1_<stamp> --at 30 47 -o /tmp/pano
```

**5. Scale / drift verdict** (from a `run_motion_test.sh` directory that has `tape_metres.txt`)

```bash
python3 scripts/vo/analyze_motion.py /path/to/motion_walk1_…
# 5.1: tape > 0 → true-scale pass/fail (±5 %)
# 5.2: tape = 0 → return-to-origin drift over path length
```

`tape_metres.txt` is optional — without it you still get rate and continuity. **Continuity is
checked first and a discontinuous run suppresses the verdict** rather than being averaged into it:
a step implying more than `--max-speed` (5 m/s) is a tracking failure, not motion, and summing over
one gives a number that means nothing. The first second is excluded and named as cuVSLAM's
initialisation transient.

**6. Live capture health** (while a node is publishing images — modular / capture, not fused)

```bash
# inside docker compose run --rm shell, with capture already up
python3 scripts/port/luma_stability.py --seconds 30   # AE hunting → large luma p2p
python3 scripts/port/sync_check.py                    # inter-camera timestamp spread
python3 scripts/port/topic_rate.py /cam1/image_raw
```

**Typical offline chain**

```text
log_rig.sh (TX2)
  → scp/rsync to datasets/
  → check_log_sets.py
  → raw_log_to_bag.py --motion
  → replay_host.sh          # host cuVSLAM
  → analyze_motion.py / rerun_multicam.py
```

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
| [docker](docker) | TX2 Foxy/CUDA-10.2 image, host cuVSLAM image, DDS-only `host-foxy` |
| [docker-compose.yml](docker-compose.yml) | TX2 services (fused, log_rig, build-…) |
| [docker-compose.host.yml](docker-compose.host.yml) | host build + bag-replay services |
| [patch](patch) | the cuVSLAM and cuNLS CUDA-10.2 port patches (TX2 only) |
| [scripts](scripts/README.md) | build, capture, calibration, replay and analysis tools — see the index |
| [openspec](openspec) | change proposals, designs and task logs |
| [third_party](third_party) | cuVSLAM, OKVIS2, OpenMAVIS submodules |

## 6. Docs

| doc | covers |
|---|---|
| [README §3](README.md#3-procedures) | **how to run, log, and analyse** — start here for commands |
| [docs/build_and_run.md](docs/build_and_run.md) | full build and run guide, recording modes, benchmarks |
| [docs/cuvslam_tx2.md](docs/cuvslam_tx2.md) | the CUDA-10.2 port, and which camera models cuVSLAM can consume |
| [docker-compose.host.yml](docker-compose.host.yml) | host offline VO build/replay (see §3.0 / §3.3) |
| [docs/timestamps.md](docs/timestamps.md) | the camera/IMU timestamp contract — the four rules |
| [docs/extrinsic_calibration.md](docs/extrinsic_calibration.md) | rig extrinsics procedure |
| [scripts/README.md](scripts/README.md) | index of every script, and where it runs |

## 7. Roadmap

- [x] Port cuVSLAM to CUDA 10.2 / C++14, running on the TX2 GPU
- [x] Host offline VO (`bev-host-cuvslam` + `replay_host.sh`) for bag resim without TX2 OOM
- [x] Argus capture with a hardware trigger and one clock for camera + IMU
- [x] Calibrate the IMX296 rig: intrinsics, virtual stereo, extrinsics
- [x] VO end to end — 8 virtual pinholes, `Multicamera` mode
- [x] Replay diagnostics in Rerun: pinholes, trajectory, BEV, 360° panorama
- [ ] Resolve absolute scale (suspected ~15–20% underestimate; it enters only via extrinsic translations)
- [ ] Fuse the IMU into the VO
- [ ] Port the nearest-axis panorama seam into `bev_panorama_node`
- [ ] Ground-plane BEV stitch on the board
