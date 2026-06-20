# cuVSLAM CUDA-10.2 / TX2 port patch

`0001-cuda102-tx2-port.patch` holds **all source changes** the cuVSLAM submodule
(`third_party/cuVSLAM`, pinned to v15.0.0) needs to build on the Jetson TX2 —
nvcc 10.2 / gcc-8 / C++14 / sm_62. See [docs/cuvslam_tx2.md](../../docs/cuvslam_tx2.md)
for the rationale behind each fix.

## Why a patch (not in-place edits)

The build previously `sed`/python-edited the submodule **in place**. That left the
submodule working tree dirty, which — combined with a board checkout that has no
`.git/modules/third_party/cuVSLAM` metadata — made `git` operations on the TX2
fail (`fatal: not a git repository …/modules/third_party/cuVSLAM`).

Keeping the changes as a patch lets the submodule stay a **pristine plain checkout**:
the build applies the patch with the `patch` utility (no git needed), so a
top-level `git pull` on the board never trips over a dirty/non-git submodule.

## What it changes

Generated from the pinned submodule; 396 files. The fixes:

1. `CMAKE_CUDA_STANDARD 17 → 14` — nvcc 10.2 has no `-std=c++17` (kernels use no C++17).
2. `-arch=all → -arch=sm_62` — nvcc 10.2 rejects `all`/`native`; TX2 SoC is sm_62.
3. `-march=native` guarded to `$<COMPILE_LANGUAGE:CXX>` so it doesn't leak into nvcc.
4. **C++17 → C++14 device-syntax downgrade** (~394 files): nested namespaces
   `namespace a::b {` → nested blocks; `inline constexpr` → `constexpr`.
5. cuSOLVER IRS enum cases guarded with `#if CUDART_VERSION >= 11000` (CUDA-11-only).
6. `cudaMallocAsync(...) → cudaMalloc(...)` (CUDA 11.2+/r470 → works on r440).

(Fix #7 — the fetched `dense_hash_map` `std::pmr` guard — patches a file under
`build_tx2gpu/_deps/`, which only exists after CMake fetches it, so it stays inline
in `scripts/build_cuvslam_tx2gpu.sh`, not here.)

## How it's applied

`scripts/build_cuvslam_tx2gpu.sh` applies it idempotently before configuring:

```bash
patch -p1 -d third_party/cuVSLAM --force < patch/cuvslam/0001-cuda102-tx2-port.patch
```

It skips if already applied (reverse dry-run) and errors if the tree is out of sync
with the pin the patch was generated from.

## Regenerating (after a submodule bump)

```bash
SRC=third_party/cuVSLAM
git -C "$SRC" checkout .                                   # pristine
# fixes 1–3,5,6 (sed) — see the history of scripts/build_cuvslam_tx2gpu.sh:
sed -i 's/set(CMAKE_CUDA_STANDARD 17)/set(CMAKE_CUDA_STANDARD 14)/' "$SRC/cmake/cuVSLAMUtils.cmake"
sed -i 's|-arch=all|-arch=sm_62|g' "$SRC/libs/cuda_modules/cuda_kernels/CMakeLists.txt"
sed -i 's|INTERFACE -march=native)|INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)|' "$SRC/cmake/cuVSLAMUtils.cmake"
# … cuSOLVER guards + cudaMallocAsync (see cuvslam_tx2.md) …
# fix 4 (the bulk): the C++17→14 rewriter
python3 scripts/port/downgrade_cuvslam_cpp17.py "$SRC"
# capture + reset
git -C "$SRC" diff > patch/cuvslam/0001-cuda102-tx2-port.patch
git -C "$SRC" checkout .
```
