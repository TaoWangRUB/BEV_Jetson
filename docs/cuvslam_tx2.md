# cuVSLAM on Jetson TX2 (J106) — CUDA 10.2 port

NVIDIA cuVSLAM (`third_party/cuVSLAM`, submodule pinned to **v17.0.0**) ported to
build and run on the **TX2 / JetPack 4.6 / CUDA 10.2 / sm_62**, for 4× IMX219
multi-camera VIO. cuVSLAM upstream targets CUDA 12/13 (Orin); this port adapts it
down to CUDA 10.2 — the TX2's permanent ceiling.

## Status: ✅ built + runtime-validated (2026-06-17)
- `libcuvslam.so` (aarch64, embedded **`libcuvslam.1.sm_62.cubin`**, linked vs
  CUDA 10.2 cusolver/cublas `.so.10`). Build exit 0, all deps resolve.
- `WarmUpGPU()` smoke test (`scripts/port/smoke_test.cpp`) initializes the GPU
  context + kernels **on the r440 driver** — exit 0. It executes, not just compiles.

## Board (probed via `ssh tx2-eth`)
L4T R32.7.6 / JetPack 4.6.x, kernel 4.9.337-tegra, **CUDA 10.2.300**, driver
**r440**, Ubuntu 18.04 host, Tegra186 (**sm_62**). 5/6 IMX219 on `/dev/video0–4`
(160° lenses). SD at `/media/nvidia/workspace` (Docker data-root + workspace).

## Why the port was needed (the floor)
nvcc 10.2 / r440 can't do what stock cuVSLAM needs: C++17 *device* code (needs
CUDA 11.0+), `cudaMallocAsync` (CUDA 11.2+/r470+), `-arch=native` (CUDA 11.5+).
The TX2 can't go past JetPack 4.6, so these are permanent — hence a source port,
not a version bump. The *algorithm* is classic feature-based VIO and runs fine on
a TX2; only the codebase's toolchain assumptions needed adapting.

## The fixes (all idempotent; applied at build time — submodule stays at v17.0.0)
Driver: `scripts/build_cuvslam_tx2gpu.sh` + `scripts/port/downgrade_cuvslam_cpp17.py`.

1. `CMAKE_CUDA_STANDARD 17 → 14` (kernels use no C++17 device features).
2. Restore `libs/log` (was lost to an over-broad rsync exclude).
3. Pin arch: `-DCMAKE_CUDA_ARCHITECTURES=62`. Since v17 upstream only emits
   `-arch=all` when the caller leaves `CMAKE_CUDA_ARCHITECTURES` unset, so the
   old `-arch=all` → `-arch=sm_62` source rewrite is no longer needed.
4. **C++17 → C++14 converter** (`downgrade_cuvslam_cpp17.py`): nested namespaces
   `namespace a::b {` → nested blocks; `inline constexpr` → `constexpr`. ~429 files.
5. Guard CUDA-11 cuSOLVER IRS enum cases (`IRS_PARAMS_INVALID_{PREC,REFINE,
   MAXITER}`, `IRS_INFOS_NOT_DESTROYED`, `IRS_MATRIX_SINGULAR`, `INVALID_WORKSPACE`).
6. `cudaMallocAsync(...)` → `cudaMalloc(...)` (1 call; also removes the r470 need).
7. Guard the fetched `dense_hash_map` `std::pmr` alias (gcc-8 libstdc++ has no pmr).
8. Link `-lstdc++fs` (gcc-8 keeps `std::filesystem` in a separate library).
9. Floating-point `std::from_chars` → `strtod` (new in v17: `libs/common/parse_utils.cpp`
   instantiates `Parse<float>`, and libstdc++ covers integral types only before gcc-11).
- Runtime: add `/etc/ld.so.conf.d/nvidia-tegra.conf` + `ldconfig` so `libcuda.so.1`
  (under `tegra/`) resolves in-container (baked into `Dockerfile.cuvslam-foxy`).

## cuNLS (v17, `USE_CUNLS`)

v17 flipped `USE_CUNLS` ON by default. cuNLS is the CUDA nonlinear least-squares
backend behind `OdometryMode::Multisensor`, and it targets CUDA 12 / sm_75+. It is
ported separately — see [patch/cunls/README.md](../patch/cunls/README.md) — and the
build script fetches, patches and injects it via `FETCHCONTENT_SOURCE_DIR_CUNLS`.
`USE_CUNLS=OFF ./scripts/build_cuvslam_tx2gpu.sh` drops back to a cuNLS-free build.

**cuDSS is the one piece that cannot be ported.** It ships as a prebuilt binary only,
published for CUDA 12/13 exclusively — no CUDA 10.2 build, no Tegra build, no source.
cuVSLAM never selects it (`multisensor_pose_estimator` asks for `DenseQR`; cuNLS's own
default is `BlockSparsePCG`), so the backend is removed outright. The cuSPARSE
`A^T*A` multiplier goes the same way: it needs the SpGEMM-reuse API (CUDA 11.3+) and
is likewise unselected, the default being the custom `Fast` kernel.

The rest is the same shape as the cuVSLAM port — C++14 device syntax, missing CUDA
10.2 APIs (`cublasSgemvStridedBatched`, `cusparseCsrSetPointers`,
`cusparseSpMV_preprocess`, `thrust::cuda::par_nosync`), and a second copy of the same
cuSOLVER IRS enum table guarded in fix 5 above.

**Finding these:** grepping for constructs proved unreliable — it missed a structured
binding spelled `const auto &[a, b]`, uppercase enum constants (the sweep matched only
lowercase `cusolver*`), and Thrust entirely. What works is enumerating every
`cuda|cublas|cusolver|cusparse|thrust` identifier *and* every uppercase
`CUDA_|CUBLAS_|CUSOLVER_|CUSPARSE_` constant, then testing each against the board's own
headers, plus compiling with the host nvcc at `-std=c++14 -arch=sm_62` **treating
warning #3356 as an error** (nvcc 12 only warns on C++17 where nvcc 10.2 rejects).

## Reproduce
```bash
# On the TX2, in /media/nvidia/workspace/BEV_Jetson — one image builds everything:
docker build -f docker/Dockerfile.cuvslam-foxy -t cuvslam-foxy:tx2 .
docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/local/cuda:/usr/local/cuda:ro -v "$PWD":/workspace -w /workspace \
  cuvslam-foxy:tx2 bash -lc './scripts/build_cuvslam_tx2gpu.sh'
# -> third_party/cuVSLAM/build_tx2gpu/bin/libcuvslam.so
```

## Next
Done: `libcuvslam.so` is wired into a ROS 2 **Foxy (20.04)** wrapper (gcc-8 `.so`
→ gcc-9 via libstdc++ forward-compat; same tegra-ld fix in the Foxy image) —
`bev_cuvslam` runs N cameras via `cuvslam::Odometry::Track(vector<Image>)`, fed by
4× IMX296 (~192° fisheye) capture (`bev_camera`) through virtual pinholes. See
[build_and_run.md](build_and_run.md). (Foxy, not Jazzy: the r440 driver only
initializes up to glibc 2.31 / Ubuntu 20.04.)

---

## Camera models — what cuVSLAM can actually consume

Calibration is only useful if the VO can load it, and cuVSLAM accepts **four** distortion models
([`libs/cuvslam/cuvslam2.h`](../third_party/cuVSLAM/libs/cuvslam/cuvslam2.h)):

| model | parameters | notes |
|---|---|---|
| `Pinhole` | 0 | no distortion — what rectified/virtual-stereo images are |
| `Fisheye` | 4 | equidistant. **Coefficients are compatible with Kalibr `pinhole-equi` and `cv::fisheye`** |
| `Brown` | 5 | 3 radial + 2 tangential |
| `Polynomial` | 8 | first 8 OpenCV coefficients |

Two things are **not** on that list, and between them they decide the whole frontend.

### There is no omni / EUCM / double-sphere model

tartancalib (and Kalibr) can solve `omni-none`, `omni-radtan`, `eucm-none`, `ds-none`; **cuVSLAM can
consume none of them.** Our IMX296 lenses are calibrated `omni-radtan` (Mei,
`config/calib/imx296_1456x1088/camN.yaml`), so those intrinsics are not loadable by the VO as they
stand. That leaves one route, and combined with the 180° limit below it is forced:

- **Rectified** — keep `omni-radtan` and remap each fisheye into two virtual pinholes
  (`config/rig/virtual_stereo_imx296.yaml`), so cuVSLAM only ever sees plain `Pinhole`.
  This is the OmniNxt architecture, and it is why their pipeline calibrates omni in the first place.

The alternative — solving `pinhole-equi` on the same bags and feeding cuVSLAM `Fisheye` directly —
is only open to lenses under 180°. Ours are ~192°, so it is not.

### The `Fisheye` path stops at 180° FOV — and it is the parameterization, not the algorithm

cuVSLAM states the limit outright: *"this (pinhole + undistort) approach works only for FOV < 180°.
TUMVI has ~190°."* The reason is visible in its own projection formula:

```
x_n = x/z,  y_n = y/z,  r = sqrt(x_n^2 + y_n^2)
radial(r) = arctan(r) * (1 + k1*arctan^2(r) + k2*arctan^4(r) + ...)
```

The incidence angle is θ = arctan(r). As θ → 90°, z → 0 and r → ∞; past 90°, z < 0 and `x/z` flips
sign, folding the ray into the wrong half-plane. arctan(r) ∈ [0°, 90°) for any finite r, so the
representable half-FOV is strictly under 90°. That is arithmetic, not an implementation shortcut —
and `FisheyeCameraModel`'s `max_normalized_uv_radius` / `max_xy_radius` guards exist because of it.

**Kannala-Brandt itself is not limited.** Compute θ directly from the ray
(`θ = atan2(sqrt(x²+y²), z)`) and ≥180° is fine — which is what ORB-SLAM3 and OpenMAVIS do, and why
cuVSLAM notes their coefficients are incompatible with its own. The ceiling is inherited from being
coefficient-compatible with `pinhole-equi` / `cv::fisheye`; the `pinhole-` prefix in Kalibr's model
name is announcing exactly this.

Practically: it binds only if a lens exceeds 180°, and ours do. The IMX296 modules calibrate to
~192°, so the direct `Fisheye` route is illegal for this rig and the virtual-pinhole remap above is
not a preference but a requirement.

### The multi-camera gate: frustum overlap

`OdometryMode::Multicamera` requires **every camera to share frustum overlap with at least one
other**, tested in [`frustum_intersection_graph.cpp`](../third_party/cuVSLAM/libs/camera/frustum_intersection_graph.cpp):
1000 points back-projected over a depth range (`d_min = -2`, `d_max = -4`) must intersect the other
camera's frustum in at least `intersected_num_points_ratio_threshold` — **0.5** upstream.

A surround rig with fisheyes ~90° apart overlaps far less than 50%, so this gate is what rejects the
rig. Our tree patches it to read `CUVSLAM_FRUSTUM_THRESHOLD` from the environment (e.g. `0.05`) so
the pairing can be tuned without a rebuild. The hard-coded depth range is the second lever if pairs
still refuse to connect.

---

