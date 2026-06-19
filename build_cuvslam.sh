#!/usr/bin/env bash
# Build cuVSLAM (third_party/cuVSLAM) for the current platform.
# Run INSIDE the BEV dev container (docker/docker-compose.yml), where the host
# CUDA toolkit is mounted at /usr/local/cuda.
#
# On a Jetson TX2 (CUDA 10.2) this is EXPECTED TO FAIL: cuVSLAM v15 targets
# CUDA 12/13 (C++17 device code + cudaMallocAsync), neither available on TX2.
# The script runs anyway to capture the exact compiler wall in the log.
#   Log: build/cuvslam_build.log
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${CUVSLAM_SRC_DIR:-${REPO_ROOT}/third_party/cuVSLAM}"
BUILD="${SRC}/build"
LOG="${REPO_ROOT}/build/cuvslam_build.log"
mkdir -p "${REPO_ROOT}/build"
ARCH="$(uname -m)"

if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    echo "ERROR: /usr/local/cuda/bin/nvcc not found — is the host CUDA mounted into the container?" >&2
    exit 1
fi
CUDA_VER="$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "CUDA toolkit: ${CUDA_VER}   arch: ${ARCH}"

if [[ "${ARCH}" == "aarch64" ]]; then
    if   grep -q tegra234 /proc/device-tree/compatible 2>/dev/null; then SM=sm_87   # Orin
    elif grep -q tegra194 /proc/device-tree/compatible 2>/dev/null; then SM=sm_72   # Xavier
    elif grep -q tegra186 /proc/device-tree/compatible 2>/dev/null; then SM=sm_62   # TX2
    else SM=sm_72; fi
    echo "Jetson SoC SM target: ${SM}"
    KCM="${SRC}/libs/cuda_modules/cuda_kernels/CMakeLists.txt"
    if [[ -f "$KCM" ]] && grep -q '\-arch=all' "$KCM"; then
        sed -i "s|-arch=all|-arch=${SM}|g" "$KCM"
        echo "Patched -arch=all -> -arch=${SM}"
    fi
    # glibc 2.34+ leaves an empty 8-byte librt.a that nvlink cannot open.
    LIBRT=/usr/lib/aarch64-linux-gnu/librt.a
    if [[ -f "$LIBRT" ]] && (( $(stat -c%s "$LIBRT") < 64 )); then
        T=$(mktemp -d); echo 'void __bev_librt_stub(void){}' > "$T/s.c"
        gcc -c "$T/s.c" -o "$T/s.o"; sudo ar rcs "$LIBRT" "$T/s.o"; rm -rf "$T"
        echo "Replaced empty librt.a with a stub archive"
    fi
fi

# cuVSLAM wants gcc-11 (its CUDA-11.4 host compiler). CUDA 10.2 needs gcc<=8 —
# unavailable on Ubuntu 24.04 — which is one of the expected TX2 failures.
HOSTCXX=g++-11; command -v g++-11 >/dev/null 2>&1 || HOSTCXX=g++

set -x
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER="$(command -v ${HOSTCXX})" \
    -DUSE_RERUN=OFF -DUSE_CERES=OFF -DUSE_NVTX=OFF \
    2>&1 | tee "${LOG}"
cmake --build "${BUILD}" -j2 --target cuvslam 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
set +x
echo "cuVSLAM build exit code: ${RC}   (full log: ${LOG})"
exit "${RC}"
