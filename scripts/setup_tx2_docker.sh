#!/usr/bin/env bash
# One-time TX2 (JetPack 4.6 / L4T R32.7) Docker prep for the BEV/cuVSLAM stack:
#   1. Pin the SD card mount in /etc/fstab (nofail) so it is present before docker.
#   2. Install the NVIDIA container runtime (GPU passthrough into containers).
#   3. Configure /etc/docker/daemon.json: nvidia runtime + data-root on the SD card.
#   4. Add the invoking user to the 'docker' group.
#
# Run with:  sudo ./scripts/setup_tx2_docker.sh
set -euo pipefail

SD_MOUNT="${SD_MOUNT:-/media/nvidia/workspace}"
DATA_ROOT="${SD_MOUNT}/docker"
REAL_USER="${SUDO_USER:-$(id -un)}"

if [[ $EUID -ne 0 ]]; then echo "Run with sudo: sudo $0" >&2; exit 1; fi

echo "== 1. Pin SD mount in fstab (nofail) =="
if ! mountpoint -q "$SD_MOUNT"; then
    echo "ERROR: $SD_MOUNT is not mounted. Mount the SD card first." >&2; exit 1
fi
SD_DEV="$(findmnt -no SOURCE "$SD_MOUNT")"
SD_UUID="$(blkid -s UUID -o value "$SD_DEV")"
SD_FSTYPE="$(findmnt -no FSTYPE "$SD_MOUNT")"
if ! grep -q "$SD_UUID" /etc/fstab 2>/dev/null; then
    cp /etc/fstab "/etc/fstab.bak.$(date +%s)"
    echo "UUID=$SD_UUID  $SD_MOUNT  $SD_FSTYPE  defaults,nofail,x-systemd.device-timeout=10  0  2" >> /etc/fstab
    systemctl daemon-reload
    echo "  added fstab entry: $SD_UUID -> $SD_MOUNT ($SD_FSTYPE)"
else
    echo "  fstab already pins $SD_UUID"
fi

echo "== 2. Install NVIDIA container runtime =="
if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    apt-get update || true
    if ! apt-get install -y nvidia-container-toolkit nvidia-container-runtime nvidia-docker2; then
        echo "ERROR: apt could not install the runtime. Ensure the Jetson apt source is enabled:" >&2
        echo "  /etc/apt/sources.list.d/nvidia-l4t-apt-source.list  (deb .../jetson/common r32.7 main; .../jetson/t186 r32.7 main)" >&2
        exit 1
    fi
else
    echo "  nvidia-container-runtime already present"
fi

echo "== 3. Configure /etc/docker/daemon.json (nvidia runtime + data-root on SD) =="
mkdir -p "$DATA_ROOT"
systemctl stop docker || true
[[ -f /etc/docker/daemon.json ]] && cp /etc/docker/daemon.json "/etc/docker/daemon.json.bak.$(date +%s)"
cat > /etc/docker/daemon.json <<JSON
{
    "data-root": "${DATA_ROOT}",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
JSON
# Make docker wait for the SD mount at boot.
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/10-sd-dataroot.conf <<UNIT
[Unit]
RequiresMountsFor=${DATA_ROOT}
UNIT
systemctl daemon-reload
systemctl start docker

echo "== 4. Add ${REAL_USER} to docker group =="
usermod -aG docker "$REAL_USER"

echo
echo "DONE. Log out/in (or run 'newgrp docker') so the group change applies."
docker info 2>/dev/null | grep -E "Docker Root Dir|Runtimes|Default Runtime" || true
