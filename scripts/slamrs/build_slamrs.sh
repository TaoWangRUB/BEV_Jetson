#!/usr/bin/env bash
# Build slam-rs's Python core (slam_rs/_core.so) in the third_party/rerun-examples submodule.
#
#   ./scripts/slamrs/build_slamrs.sh                 # host
#   PYTHON=~/micromamba/envs/slamrs/bin/python ./scripts/slamrs/build_slamrs.sh   # TX2
#
# Upstream builds through pixi, which also pulls dataforge and the rest of the monorepo's
# Python stack. None of that is needed to call slam_rs._core from run_slamrs.py: the core is
# one cargo crate plus one checksummed dependency patch. So this does exactly those two steps.
#
# PyO3 is built abi3-py312: the interpreter that imports the result must be Python >= 3.12.
# Ubuntu 18.04 on the TX2 has 3.6, hence PYTHON pointing at a conda-forge env there.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="${SLAM_RS_DIR:-${REPO}/third_party/rerun-examples/packages/slam-rs}"
PYTHON="${PYTHON:-python3}"
export PATH="${HOME}/.cargo/bin:${PATH}"

[[ -f "${PKG}/Cargo.toml" ]] || {
  echo "REFUSING: no slam-rs at ${PKG}. git submodule update --init third_party/rerun-examples" >&2; exit 1; }
command -v cargo >/dev/null || {
  echo "REFUSING: no cargo. curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal" >&2; exit 1; }
"${PYTHON}" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' || {
  echo "REFUSING: ${PYTHON} is older than 3.12, and the core is built abi3-py312" >&2; exit 1; }

# The [patch.crates-io] entry in Cargo.toml points into target/patch/, which upstream's
# slam-rs-patch-deps task creates. Its module needs only the stdlib; the tyro CLI around it
# does not, so load the module file directly.
"${PYTHON}" - "${PKG}" <<'PY'
import importlib.util, sys
from pathlib import Path
pkg = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("ppd", pkg / "slam_rs/apis/prepare_patched_deps.py")
m = importlib.util.module_from_spec(spec); sys.modules["ppd"] = m; spec.loader.exec_module(m)
m.main(m.Config(package_dir=pkg))
PY

cd "${PKG}"
PYO3_PYTHON="$(command -v "${PYTHON}")" cargo build --release -p slam-rs-py
cp target/release/lib_core.so slam_rs/_core.so
echo "built ${PKG}/slam_rs/_core.so ($(git -C "${PKG}" log -1 --format='%h %ad' --date=short))"
