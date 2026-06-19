#!/usr/bin/env bash
# Runs as root at container start: register the tegra libcuda.so.1 the NVIDIA
# runtime just mounted (so CUDA's driver resolves), then source ROS 2 Foxy.
set -e
ldconfig
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi
exec "$@"
