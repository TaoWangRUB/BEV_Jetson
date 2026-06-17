#!/usr/bin/env bash
# Source ROS 2 and the workspace overlay (if built), then exec the command.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi
exec "$@"
