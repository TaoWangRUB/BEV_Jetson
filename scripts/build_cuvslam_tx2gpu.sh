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

# --- known fix #2: TX2 SoC = sm_62. nvcc 10.2 rejects both '-arch=all' (manual,
#     CUDA 11.5+) and '-arch=native' (CMake 3.27 default when CUDA_ARCHITECTURES
#     is unset). Rewrite the manual flag AND pin CMAKE_CUDA_ARCHITECTURES=62. -----
KCM="${SRC}/libs/cuda_modules/cuda_kernels/CMakeLists.txt"
[[ -f "$KCM" ]] && grep -q '\-arch=all' "$KCM" && sed -i 's|-arch=all|-arch=sm_62|g' "$KCM"

# --- known fix #3: '-march=native' (aarch64 host flag) is added to ALL compile
#     languages and leaks into nvcc, which misparses it as 'arch=native'.
#     Guard it to CXX only so the CUDA kernels don't receive it. -----------------
UTILS="${SRC}/cmake/cuVSLAMUtils.cmake"
[[ -f "$UTILS" ]] && sed -i \
    's|INTERFACE -march=native)|INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)|' "$UTILS"

# --- known fix #4: downgrade C++17 device syntax (nested namespaces + inline
#     variables) to C++14 so nvcc 10.2 can parse the device-reachable headers.
#     Idempotent; cuVSLAM submodule stays at v15.0.0. ----------------------------
python3 "${REPO_ROOT}/scripts/port/downgrade_cuvslam_cpp17.py" "${SRC}"

# --- known fix #5: cuSOLVER IRS enum values _INVALID_{PREC,REFINE,MAXITER} were
#     added in CUDA 11; 10.2 only has the base _INVALID. Guard the 3 newer case
#     labels (error-string mapping only). Idempotent. -----------------------------
CULIB="${SRC}/libs/cuda_modules/culib_helper.h"
if [[ -f "$CULIB" ]] && ! grep -q 'CUDART_VERSION >= 11000' "$CULIB"; then
    # block 1: PARAMS_INVALID_{PREC,REFINE,MAXITER}
    sed -i '/case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:/i #if CUDART_VERSION >= 11000' "$CULIB"
    sed -i '/return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";/a #endif' "$CULIB"
    # block 2: IRS_INFOS_NOT_DESTROYED, IRS_MATRIX_SINGULAR, INVALID_WORKSPACE
    sed -i '/case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:/i #if CUDART_VERSION >= 11000' "$CULIB"
    sed -i '/return "CUSOLVER_STATUS_INVALID_WORKSPACE";/a #endif' "$CULIB"
fi

echo "Configuring cuVSLAM GPU for CUDA 10.2 / sm_62 / gcc-8 / C++14 ..."
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=62 \
    -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-8 \
    -DUSE_RERUN=OFF -DUSE_CERES=OFF -DUSE_NVTX=OFF 2>&1 | tee "${LOG}"
cmake --build "${BUILD}" -j2 --target cuvslam 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
echo "cuVSLAM GPU (CUDA 10.2) build exit: ${RC}   (log: ${LOG})"
exit "${RC}"
