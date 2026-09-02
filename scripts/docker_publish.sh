#!/usr/bin/env bash
# Publish the TX2 container image to Docker Hub.
#
# WHY BOTHER. On 2026-09-02 a careless `docker image prune -af` deleted every image on the
# board - nothing was running, so -a took all of them - and cost an hour of rebuilding on a
# TX2. The layers were genuinely gone: the docker root dropped to 340 KB and the space was
# reclaimed, so there was nothing to recover. With the image on Hub that is a `docker pull`
# instead.
#
# WHAT IS IN IT: Ubuntu 20.04 + ROS 2 Foxy + gcc-8/9 + an entrypoint script. cuVSLAM, the
# calibration and the repo are BIND-MOUNTED at runtime (docker-compose.yml), not baked in,
# so this image carries nothing project-specific and is safe to publish.
#
#   ./scripts/docker_publish.sh              # push tx2 + a dated tag
#   REPO=you/other ./scripts/docker_publish.sh
set -euo pipefail

REPO="${REPO:-wtlove876/cuvslam-foxy}"
LOCAL="${LOCAL:-cuvslam-foxy:tx2}"
DATE_TAG="${DATE_TAG:-tx2-$(date +%Y%m%d)}"

docker image inspect "$LOCAL" >/dev/null 2>&1 || {
  echo "no local image '$LOCAL'. Build it first:  docker compose build" >&2; exit 1; }

# `docker info` only reports a username for some credential stores, so check the config too.
if ! docker info 2>/dev/null | grep -qi "^ *Username:" \
   && ! grep -q '"auths"[[:space:]]*:[[:space:]]*{[[:space:]]*"' ~/.docker/config.json 2>/dev/null; then
  cat >&2 <<'MSG'
not logged in to Docker Hub. Run this yourself - a token is better than a password here,
because it is scoped, revocable, and it sits in ~/.docker/config.json on a dev board:

  docker login -u wtlove876        # paste an access token when asked for a password
                                   # Hub -> Account Settings -> Security -> New Access Token
MSG
  exit 1
fi

SIZE=$(docker image inspect "$LOCAL" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}')
echo "publishing $LOCAL ($SIZE) as:"
echo "    $REPO:tx2"
echo "    $REPO:$DATE_TAG        <- pin this in a compose file to make a build reproducible"
echo

for t in tx2 "$DATE_TAG"; do
  docker tag "$LOCAL" "$REPO:$t"
  echo "pushing $REPO:$t ..."
  docker push "$REPO:$t"
done

echo
echo "done. To restore this image on a fresh board:"
echo "    docker pull $REPO:tx2 && docker tag $REPO:tx2 $LOCAL"
echo "The compose file builds from docker/Dockerfile.cuvslam-foxy by default, so retag to"
echo "$LOCAL rather than editing compose - that keeps a local rebuild working too."
