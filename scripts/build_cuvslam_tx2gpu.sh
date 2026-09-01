#!/usr/bin/env bash
# Empirical port: build cuVSLAM's GPU path on CUDA 10.2 (TX2) with gcc-8 + C++14.
# Run inside the Foxy container (docker/Dockerfile.cuvslam-foxy, which ships gcc-8
# for nvcc) with the host CUDA mounted at /usr/local/cuda. Pins g++-8 as the CUDA
# host compiler regardless of the container default. Idempotent: applies the known
# fixes, then surfaces whatever CUDA-12 API still remains. Log: build/cuvslam_tx2gpu.log
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/third_party/cuVSLAM"
BUILD="${SRC}/build_tx2gpu"
LOG="${REPO_ROOT}/build/cuvslam_tx2gpu.log"
mkdir -p "${REPO_ROOT}/build"

# --- CUDA-10.2 / sm_62 source port: applied as one patch ----------------------
# All in-place source changes cuVSLAM needs for nvcc 10.2 / gcc-8 / C++14 live in
# patch/cuvslam/0001-cuda102-tx2-port.patch (generated from the pinned submodule;
# see patch/cuvslam/README.md). Applied with `patch` rather than `git apply` so it
# works even when the submodule has no git metadata on the board — the submodule
# tree stays a plain checkout, so a top-level `git pull` on the TX2 never trips on
# a dirty/!git submodule. Idempotent (skips if already applied). Covered fixes:
#   1. CMAKE_CUDA_STANDARD 17 -> 14             (cmake/cuVSLAMUtils.cmake)
#   2. (obsolete since v17: upstream emits -arch=all only when the caller
#      leaves CMAKE_CUDA_ARCHITECTURES unset; we pass 62 below)
#   3. -march=native guarded to CXX only        (cmake/cuVSLAMUtils.cmake)
#   4. C++17 -> C++14 device-syntax downgrade    (~429 files)
#   5. cuSOLVER-11 IRS enum guards               (culib_helper.h)
#   6. cudaMallocAsync -> cudaMalloc             (selection_v2.cpp)
# Regenerate after a submodule bump: scripts/port/regen_cuvslam_patch.sh
# (see patch/cuvslam/README.md).
#
# USE_CUNLS (default ON here; override with USE_CUNLS=OFF): cuNLS is the CUDA nonlinear
# least-squares backend behind v17's OdometryMode::Multisensor. Upstream targets CUDA 12 /
# sm_75+ and its CMake downloads a prebuilt cuDSS archive, which NVIDIA publishes for
# CUDA 12/13 ONLY -- no CUDA 10.2 build, no Tegra build, and no source to compile. cuVSLAM
# never actually selects cuDSS (multisensor_pose_estimator asks for DenseQR; cuNLS's own
# default is BlockSparsePCG), so patch/cunls/0001-cuda102-tx2-port.patch removes that
# backend outright along with the C++17/sm_75 assumptions. The source is pre-populated and
# patched below, then handed to cuVSLAM's FetchContent via FETCHCONTENT_SOURCE_DIR_CUNLS
# so no download or upstream PATCH_COMMAND runs.
PATCH="${REPO_ROOT}/patch/cuvslam/0001-cuda102-tx2-port.patch"
if patch -p1 -d "${SRC}" --dry-run --reverse --force <"${PATCH}" >/dev/null 2>&1; then
    echo "cuVSLAM CUDA-10.2 port patch already applied."
elif patch -p1 -d "${SRC}" --dry-run --force <"${PATCH}" >/dev/null 2>&1; then
    patch -p1 -d "${SRC}" --force <"${PATCH}"
    echo "Applied cuVSLAM CUDA-10.2 port patch."
else
    echo "ERROR: cuVSLAM port patch neither applies cleanly nor is already applied" >&2
    echo "       — submodule may be out of sync with the pin it was generated from." >&2
    exit 1
fi

# --- cuNLS: pre-populate + patch, so FetchContent neither downloads nor fetches cuDSS ---
USE_CUNLS="${USE_CUNLS:-ON}"
CUNLS_ARGS=(-DUSE_CUNLS=OFF)
if [[ "${USE_CUNLS}" == "ON" ]]; then
    CUNLS_VER="${CUNLS_VERSION:-Release_07_13_2026}"
    CUNLS_TAR="${REPO_ROOT}/build/cunls-${CUNLS_VER}.tar.gz"
    CUNLS_SRC="${REPO_ROOT}/build/cuNLS-${CUNLS_VER}"
    CUNLS_PATCH="${REPO_ROOT}/patch/cunls/0001-cuda102-tx2-port.patch"
    if [[ ! -f "${CUNLS_TAR}" ]]; then
        echo "Downloading cuNLS ${CUNLS_VER} ..."
        curl -sSL -o "${CUNLS_TAR}" \
          "https://github.com/nvidia-isaac/cuNLS/archive/refs/tags/${CUNLS_VER}.tar.gz" || {
            echo "ERROR: cuNLS download failed; re-run with USE_CUNLS=OFF to skip it." >&2; exit 1; }
    fi
    if [[ ! -d "${CUNLS_SRC}" ]]; then
        tar xzf "${CUNLS_TAR}" -C "${REPO_ROOT}/build"
    fi
    if patch -p1 -d "${CUNLS_SRC}" --dry-run --reverse --force <"${CUNLS_PATCH}" >/dev/null 2>&1; then
        echo "cuNLS CUDA-10.2 port patch already applied."
    elif patch -p1 -d "${CUNLS_SRC}" --dry-run --force <"${CUNLS_PATCH}" >/dev/null 2>&1; then
        patch -p1 -d "${CUNLS_SRC}" --force <"${CUNLS_PATCH}"
        echo "Applied cuNLS CUDA-10.2 port patch."
    else
        echo "ERROR: cuNLS port patch neither applies cleanly nor is already applied" >&2
        echo "       — regenerate with scripts/port/regen_cunls_patch.sh" >&2
        exit 1
    fi
    CUNLS_ARGS=(-DUSE_CUNLS=ON "-DFETCHCONTENT_SOURCE_DIR_CUNLS=${CUNLS_SRC}")
fi

echo "Configuring cuVSLAM GPU for CUDA 10.2 / sm_62 / gcc-8 / C++14 (USE_CUNLS=${USE_CUNLS}) ..."
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=62 \
    -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-8 \
    -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs \
    -DUSE_RERUN=OFF -DUSE_CERES=OFF -DUSE_NVTX=OFF \
    "${CUNLS_ARGS[@]}" 2>&1 | tee "${LOG}"

# --- known fix #7: the fetched dense_hash_map dep defines an unconditional
#     std::pmr alias; gcc-8's libstdc++ has no std::pmr (added in gcc-9). cuVSLAM
#     doesn't use the pmr variant, so guard that block with a feature test. -------
DHM="${BUILD}/_deps/dense_hash_map-src/include/jg/dense_hash_map.hpp"
if [[ -f "$DHM" ]] && ! grep -q '__cpp_lib_memory_resource' "$DHM"; then
    sed -i '/^namespace pmr$/i #if defined(__cpp_lib_memory_resource)' "$DHM"
    sed -i '\|^} // namespace pmr$|a #endif' "$DHM"
    echo "patched dense_hash_map std::pmr guard"
fi

cmake --build "${BUILD}" -j2 --target cuvslam 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
echo "cuVSLAM GPU (CUDA 10.2) build exit: ${RC}   (log: ${LOG})"
exit "${RC}"
