#!/usr/bin/env bash
# Empirical port: build cuVSLAM's GPU path on CUDA 10.2 (TX2) with gcc-8 + C++14.
# Run inside the gcc-8 build container (docker/Dockerfile.cuvslam-build) with the
# host CUDA mounted at /usr/local/cuda. Iterative: applies the known fixes, then
# surfaces whatever CUDA-12 API still remains. Log: build/cuvslam_tx2gpu.log
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/third_party/cuVSLAM"
BUILD="${SRC}/build_tx2gpu"
LOG="${REPO_ROOT}/build/cuvslam_tx2gpu.log"
mkdir -p "${REPO_ROOT}/build"

# --- known fix #1: nvcc 10.2 has no -std=c++17; the kernels use no C++17 -------
sed -i 's/set(CMAKE_CUDA_STANDARD 17)/set(CMAKE_CUDA_STANDARD 14)/' \
    "${SRC}/cmake/cuVSLAMUtils.cmake"

# --- known fix #2: TX2 SoC = sm_62 (replace -arch=all if present) --------------
KCM="${SRC}/libs/cuda_modules/cuda_kernels/CMakeLists.txt"
[[ -f "$KCM" ]] && grep -q '\-arch=all' "$KCM" && sed -i 's|-arch=all|-arch=sm_62|g' "$KCM"

echo "Configuring cuVSLAM GPU for CUDA 10.2 / sm_62 / gcc-8 / C++14 ..."
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-8 \
    -DUSE_RERUN=OFF -DUSE_CERES=OFF -DUSE_NVTX=OFF 2>&1 | tee "${LOG}"
cmake --build "${BUILD}" -j2 --target cuvslam 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
echo "cuVSLAM GPU (CUDA 10.2) build exit: ${RC}   (log: ${LOG})"
exit "${RC}"
