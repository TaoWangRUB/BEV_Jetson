#!/usr/bin/env bash
# One command: build the Foxy image (gcc-8 + ROS 2 Foxy, if missing), build
# libcuvslam.so for CUDA 10.2 / sm_62, then run the WarmUpGPU runtime smoke test
# on the GPU. Run on the TX2 from the BEV repo root (needs the nvidia docker runtime).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
IMG="${IMG:-cuvslam-foxy:tx2}"

GPU_RUN=(docker run --rm --runtime nvidia
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
  -v /usr/local/cuda:/usr/local/cuda:ro -v "$REPO_ROOT":/workspace -w /workspace "$IMG")

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "== building image $IMG =="
  docker build -f docker/Dockerfile.cuvslam-foxy -t "$IMG" .
fi

echo "== building libcuvslam.so (CUDA 10.2 / sm_62) =="
"${GPU_RUN[@]}" bash -lc './scripts/build_cuvslam_tx2gpu.sh'

echo "== runtime smoke test (WarmUpGPU on r440 driver) =="
"${GPU_RUN[@]}" bash -lc '
  ldconfig   # tegra libcuda.so.1 (nvidia-tegra.conf baked in the image)
  g++-8 -std=c++17 -I third_party/cuVSLAM/libs/cuvslam scripts/port/smoke_test.cpp \
    -L third_party/cuVSLAM/build_tx2gpu/bin -lcuvslam -o /tmp/smoke
  LD_LIBRARY_PATH=/workspace/third_party/cuVSLAM/build_tx2gpu/bin:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib /tmp/smoke
'
echo "== DONE: libcuvslam.so built and validated =="
echo "   lib: third_party/cuVSLAM/build_tx2gpu/bin/libcuvslam.so"
