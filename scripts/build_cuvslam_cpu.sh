#!/usr/bin/env bash
# Build cuVSLAM in CPU-only mode (-DUSE_CUDA=OFF). No CUDA toolchain needed:
# the full feature-based VIO algorithm compiles as pure C++17 host code with the
# default gcc, sidestepping every CUDA-10.2 wall (nvcc c++17, gcc-8, async-alloc).
# Runs on CPU (no GPU acceleration). Log: build/cuvslam_cpu_build.log
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/third_party/cuVSLAM"
BUILD="${SRC}/build_cpu"
LOG="${REPO_ROOT}/build/cuvslam_cpu_build.log"
mkdir -p "${REPO_ROOT}/build"
echo "Configuring cuVSLAM CPU-only (USE_CUDA=OFF)..."
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF -DUSE_RERUN=OFF -DUSE_CERES=OFF -DUSE_NVTX=OFF 2>&1 | tee "${LOG}"
cmake --build "${BUILD}" -j"$(nproc)" --target cuvslam 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
echo "cuVSLAM CPU-only build exit: ${RC}   (log: ${LOG})"
exit "${RC}"
