#!/usr/bin/env bash
# Thin wrapper around docker compose: run the MODULAR capture + cuVSLAM VO (ROS2 GPU->CPU->GPU)
# in one container. All container params live in docker-compose.yml (service `modular`).
# Kept for bring-up / bag inspection; the fused zero-copy node (run_vo_fused_tx2.sh) is the
# recommended runtime (~3x less CPU). Stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."       # repo root = compose file location
exec docker compose run --rm modular
