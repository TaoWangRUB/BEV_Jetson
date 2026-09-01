# cuVSLAM CUDA-10.2 / TX2 port patch

`0001-cuda102-tx2-port.patch` holds **all source changes** the cuVSLAM submodule
(`third_party/cuVSLAM`, pinned to v17.0.0) needs to build on the Jetson TX2 —
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

Generated from the pinned submodule; 430 files. The fixes:

1. `CMAKE_CUDA_STANDARD 17 → 14` — nvcc 10.2 has no `-std=c++17` (kernels use no C++17).
2. *(obsolete since v17)* — upstream now emits `-arch=all` only when the caller leaves
   `CMAKE_CUDA_ARCHITECTURES` unset. `build_cuvslam_tx2gpu.sh` passes
   `-DCMAKE_CUDA_ARCHITECTURES=62`, so nvcc never sees `all`; no source edit needed.
3. `-march=native` guarded to `$<COMPILE_LANGUAGE:CXX>` so it doesn't leak into nvcc.
4. **C++17 → C++14 device-syntax downgrade** (~429 files): nested namespaces
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
./scripts/port/regen_cuvslam_patch.sh
```

It resets the submodule to pristine, replays fixes 1/3/5/6 plus the
`downgrade_cuvslam_cpp17.py` rewrite, captures the diff, and resets the tree again.
Verify afterwards that the CUDA-reachable surface has no leftover C++17 — the `.cu`/
`.cuh` files only include a handful of local headers, so:

```bash
patch -p1 -d third_party/cuVSLAM --force < patch/cuvslam/0001-cuda102-tx2-port.patch
grep -rEn 'namespace [a-z_]+::|inline constexpr' \
  $(find third_party/cuVSLAM/libs -name '*.cu' -o -name '*.cuh') \
  third_party/cuVSLAM/libs/cuda_modules/cuda_kernels/*.h
git -C third_party/cuVSLAM checkout -- .
```

Only closing-brace comments (`}}  // namespace a::b`) and `using namespace a::b;`
should remain — both are valid C++14.

## `0002-frustum-threshold-env.optional.patch` (optional, not applied by the build)

Makes cuVSLAM's multi-camera stereo-pair gate tunable at runtime via
`CUVSLAM_FRUSTUM_THRESHOLD` and traces each pair's measured overlap ratio. It was a
debugging aid for the divergent 4-fisheye BEV rig, where the hardcoded 0.5
overlap threshold rejected every pair.

**Possibly obsolete on v17** — v17 fixes the back-projection depth range
(`d_min`/`d_max` were `-2`/`-4` in v15, so every probe point landed behind the camera
and `point_d_*_j.z() <= 0` skipped it, driving the ratio to ~0 regardless of
geometry). That alone may let the BEV rig's adjacent pairs clear the 0.5 gate — it
has **not** been measured on the real rig yet.

v17 also adds `camera::MulticameraMode::Manual` (explicit primary/secondary lists, no
overlap check), but it is **not reachable from the public API**: `ToMulticamMode()` in
`libs/cuvslam/cuvslam2.cpp` maps only Performance/Precision/Moderate, and upstream's
own TODO there reads "What about Manual mode hidden from cuvslam API?". It is settable
only via the internal `sof` gflags path. So Manual is not an option for `bev_cuvslam`
without patching the API.

Keep this patch as the diagnostic fallback until the v17 ratios are measured on the
rig. Apply by hand if needed:

```bash
patch -p1 -d third_party/cuVSLAM < patch/cuvslam/0002-frustum-threshold-env.optional.patch
```
