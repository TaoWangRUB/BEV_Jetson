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

# 4b. std::from_chars for floating point is gcc-11+; the TX2 toolchain is gcc-8,
#     where <charconv> covers integral types only (__cpp_lib_to_chars undefined).
#     parse_utils.cpp is new in v17 and instantiates Parse<float>.
python3 - "$SRC/libs/common/parse_utils.cpp" <<'PYFC'
import sys
p = sys.argv[1]
s = open(p).read()
old_inc = "#include <charconv>\n#include <cstddef>"
new_inc = ("#include <cerrno>\n#include <charconv>\n#include <cstddef>\n"
           "#include <cstdlib>\n#include <type_traits>")
assert s.count(old_inc) == 1, "parse_utils.cpp includes not found"
s = s.replace(old_inc, new_inc, 1)

old = """  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end) {
    ThrowParseError(expected, v);
  }
  return parsed;"""
new = """#if defined(__cpp_lib_to_chars)
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end) {
    ThrowParseError(expected, v);
  }
#else
  // libstdc++ implements from_chars for integral types only until gcc-11, and the
  // TX2 builds with gcc-8 (where __cpp_lib_to_chars is undefined). Integrals keep
  // the fast path; floating point goes through strtod on a NUL-terminated copy.
  // strtod is locale-sensitive where from_chars is not, but this build never calls
  // setlocale, so it stays in the C locale and the decimal point matches.
  if constexpr (std::is_floating_point<T>::value) {
    const std::string buf(v);
    const char* cbegin = buf.c_str();
    char* conv_end = nullptr;
    errno = 0;
    const double value = std::strtod(cbegin, &conv_end);
    // Reject trailing garbage exactly as the ptr != end check above does.
    if (conv_end != cbegin + buf.size() || errno == ERANGE) {
      ThrowParseError(expected, v);
    }
    parsed = static_cast<T>(value);
  } else {
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
      ThrowParseError(expected, v);
    }
  }
#endif
  return parsed;"""
assert s.count(old) == 1, "parse_utils.cpp Parse body not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)
PYFC

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
