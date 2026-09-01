#!/usr/bin/env bash
# Regenerate patch/cuvslam/0001-cuda102-tx2-port.patch from the pinned cuVSLAM
# submodule. Run after every submodule bump; see patch/cuvslam/README.md.
# Leaves the submodule tree pristine (the patch is applied at build time by
# scripts/build_cuvslam_tx2gpu.sh with the `patch` util, not git).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/third_party/cuVSLAM"
OUT="${REPO_ROOT}/patch/cuvslam/0001-cuda102-tx2-port.patch"

git -C "$SRC" checkout -- .          # start pristine

# 1. CMAKE_CUDA_STANDARD 17 -> 14 — nvcc 10.2 has no -std=c++17.
sed -i 's/set(CMAKE_CUDA_STANDARD 17)/set(CMAKE_CUDA_STANDARD 14)/' "$SRC/cmake/cuVSLAMUtils.cmake"

# 2. (obsolete since v17) -arch=all is now emitted only when the caller does NOT
#    pass -DCMAKE_CUDA_ARCHITECTURES; build_cuvslam_tx2gpu.sh passes 62, so nvcc
#    never sees `all`. No source edit needed.

# 3. -march=native guarded to C++ only, so it never leaks into nvcc.
sed -i 's|INTERFACE -march=native)|INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)|' \
    "$SRC/cmake/cuVSLAMUtils.cmake"

# 5. cuSOLVER IRS enum cases that only exist in CUDA >= 11.
python3 - "$SRC/libs/cuda_modules/culib_helper.h" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
for first, last in (("CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC", "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER"),
                    ("CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED", "CUSOLVER_STATUS_INVALID_WORKSPACE")):
    blk = re.search(r'( *case %s:\n.*?case %s:\n *return "[^"]*";\n)' % (first, last), s, re.S)
    assert blk, "cuSOLVER guard block not found: " + first
    s = s.replace(blk.group(1), "#if CUDART_VERSION >= 11000\n" + blk.group(1) + "#endif\n", 1)
open(p, "w").write(s)
PY

# 6. cudaMallocAsync (CUDA 11.2+/r470) -> cudaMalloc (works on r440).
sed -i 's|cudaMallocAsync(&sort_temp_buffer_, sort_temp_buffer_size_, stream)|cudaMalloc(\&sort_temp_buffer_, sort_temp_buffer_size_)|' \
    "$SRC/libs/cuda_modules/selection_v2.cpp"

# 4. the bulk: C++17 -> C++14 device-syntax downgrade (~400 files).
python3 "${REPO_ROOT}/scripts/port/downgrade_cuvslam_cpp17.py" "$SRC"

# capture, then reset the submodule back to pristine
git -C "$SRC" diff > "$OUT"
git -C "$SRC" checkout -- .
echo "Regenerated $OUT ($(grep -c '^diff --git' "$OUT") files, $(wc -l < "$OUT") lines)"
