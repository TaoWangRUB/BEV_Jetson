#!/usr/bin/env bash
# Thin wrapper around docker compose: run the fused zero-copy Argus -> cuVSLAM VO.
# All container parameters live in docker-compose.yml (service `fused`); node params in
# ros2/bev_cuvslam/config/fused_vo_params.yaml. Stop with Ctrl-C (clean Argus release).
#
#   ./scripts/run_vo_fused_tx2.sh            # run fused VO (default 1640->832x624, full FOV)
#   RECORD=1 ./scripts/run_vo_fused_tx2.sh   # also bag /cuvslam/odometry + /tf into bags/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."       # repo root = compose file location
exec docker compose run --rm fused
