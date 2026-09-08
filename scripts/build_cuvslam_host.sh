#!/usr/bin/env bash
# Build libcuvslam.so for the HOST (x86_64 / CUDA 12.x). No TX2 CUDA-10.2 port patch.
#
# Pattern from ackermann_rover_humble/scripts/build_cuvslam.sh: run inside the
# bev-host-cuvslam container with host CUDA mounted at /usr/local/cuda.
#
#   docker compose -f docker-compose.host.yml run --rm build-cuvslam-host
#   # or, already inside the container:
#   ./scripts/build_cuvslam_host.sh
#
# Output: third_party/cuVSLAM/build_host/bin/libcuvslam.so
# Refuses to run if the TX2 port patch is applied to the submodule tree.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/third_party/cuVSLAM"
BUILD="${SRC}/build_host"
LOG="${REPO_ROOT}/build/cuvslam_host.log"
mkdir -p "${REPO_ROOT}/build"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "REFUSING: build_cuvslam_host.sh is for x86_64 hosts (got $(uname -m))." >&2
  echo "  On the TX2 use scripts/build_cuvslam_tx2gpu.sh instead." >&2
  exit 1
fi

if [[ -f "${SRC}/.tx2-port-stamp" ]]; then
  echo "REFUSING: third_party/cuVSLAM has the TX2 CUDA-10.2 port applied." >&2
  echo "  Host builds need a pristine v17 tree. On a machine that also builds for" >&2
  echo "  the TX2, keep separate worktrees, or revert the port before building here:" >&2
  echo "    patch -p1 -d third_party/cuVSLAM -R < patch/cuvslam/0001-cuda102-tx2-port.patch" >&2
  echo "    rm -f third_party/cuVSLAM/.tx2-port-stamp third_party/cuVSLAM/.tx2-port-applied.patch" >&2
  exit 1
fi

if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  echo "REFUSING: /usr/local/cuda/bin/nvcc missing — mount the host CUDA toolkit." >&2
  exit 1
fi

# gcc-11 (ubuntu-toolchain-r/test on Focal) — floating-point std::from_chars.
if ! command -v g++-11 >/dev/null; then
  echo "REFUSING: g++-11 missing inside bev-host-cuvslam." >&2
  exit 1
fi

# Ampere laptop (RTX A2000) = sm_86. Override with CMAKE_CUDA_ARCHITECTURES=...
ARCHS="${CMAKE_CUDA_ARCHITECTURES:-86}"
# Multicamera (visual) VO does not need cuNLS; keep the first host bring-up lean.
USE_CUNLS="${USE_CUNLS:-OFF}"
HOST_CC="${HOST_CC:-gcc-11}"
HOST_CXX="${HOST_CXX:-g++-11}"

echo "Configuring cuVSLAM for host CUDA $($(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')) / sm_${ARCHS} (USE_CUNLS=${USE_CUNLS}, CXX=${HOST_CXX}) ..."
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCHS}" \
    -DCMAKE_C_COMPILER="${HOST_CC}" -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
    -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
    -DUSE_RERUN=OFF -DUSE_CERES=OFF -DUSE_NVTX=OFF \
    -DUSE_CUNLS="${USE_CUNLS}" \
    2>&1 | tee "${LOG}"

cmake --build "${BUILD}" -j"$(nproc)" --target cuvslam 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
echo "cuVSLAM host build exit: ${RC}   (log: ${LOG})"
echo "  lib: ${BUILD}/bin/libcuvslam.so"
exit "${RC}"
