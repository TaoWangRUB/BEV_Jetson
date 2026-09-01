# cuVSLAM on Jetson TX2 (J106) — CUDA 10.2 port

NVIDIA cuVSLAM (`third_party/cuVSLAM`, submodule pinned to **v15.0.0**) ported to
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
4× IMX219 (160° fisheye) capture (`bev_camera`) + fisheye calibration. See
[build_and_run.md](build_and_run.md). (Foxy, not Jazzy: the r440 driver only
initializes up to glibc 2.31 / Ubuntu 20.04.)
