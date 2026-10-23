#!/bin/sh
# Prepare the patched dependency sources that this workspace's [patch.crates-io] points at.
# Run once per checkout (and again after `rm -rf target`); idempotent.
set -eu
cd "$(dirname "$0")"
prep() { # <crate-version> <patch file under patches/>
    dst="target/patch/$1"
    [ -f "$dst/.patched" ] && { echo "$1: prepared"; return; }
    src=$(ls -d "$HOME"/.cargo/registry/src/*/"$1" 2>/dev/null | head -1)
    [ -n "$src" ] || { echo "$1 is not in the cargo registry; run a build first so cargo fetches it" >&2; exit 1; }
    rm -rf "$dst"; mkdir -p target/patch; cp -R "$src" "$dst"
    (cd "$dst" && patch -p1 --no-backup-if-mismatch < "../../../patches/$2")
    touch "$dst/.patched"; echo "$1: patched"
}
prep cubecl-spirv-0.11.0-pre.3 cubecl-spirv-0.11.0-pre.3-workgroup-variable.patch
