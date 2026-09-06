#!/usr/bin/env bash
# Host cuVSLAM container entry: source Foxy (+ workspace if built). No tegra ldconfig.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f /workspace/install_host/setup.bash ]; then
    source /workspace/install_host/setup.bash
elif [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi
exec "$@"
